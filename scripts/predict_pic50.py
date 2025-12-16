"""
TDP1 pIC50 Prediction Script
==============================
This script uses the trained ChemBERTa-77M-MTR model to predict pIC50 values
for a batch of compounds provided as SMILES strings in a TXT file.

Usage:
    python predict_pic50.py --input compounds.txt --output predictions.csv

Input TXT format:
    One SMILES string per line.

Output CSV format:
    SMILES,Predicted_pIC50
    Sorted by Predicted_pIC50 in descending order (most active first)

Author: NokkTsang
Date: December 2025
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

# Directory paths (relative to project root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INPUT_DIR = os.path.join(PROJECT_ROOT, "input")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Model paths
MODEL_PATH = os.path.join(PROJECT_ROOT, "Model", "ChemBERTa-77M-MTR-sample-weighting")
BASE_MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"

# Create directories if they don't exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model parameters
MAX_LENGTH = 512
ACTIVE_THRESHOLD = 6.0
FLUSH_INTERVAL = 100000  # Write to disk every 100K compounds

# ============================================================================
# Prediction Functions
# ============================================================================


def load_model_and_tokenizer(model_path, base_model_name, device):
    """
    Load the trained model and tokenizer.

    Args:
        model_path (str): Path to the trained model directory
        base_model_name (str): Name of the base model for tokenizer
        device (torch.device): Device to load model on

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading tokenizer from {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    print(f"Loading model from {model_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    print(f"Model loaded successfully on {device}")
    return model, tokenizer


def get_batch_filename(output_base, batch_num):
    """Generate batch filename like: predictions_123M_batch_001.csv"""
    base, ext = os.path.splitext(output_base)
    return f"{base}_batch_{batch_num:03d}{ext}"


def find_existing_batches(output_base):
    """Find already completed batch files and return the highest batch number."""
    base, ext = os.path.splitext(output_base)
    batch_dir = os.path.dirname(output_base) or "."
    
    existing_batches = []
    for f in os.listdir(batch_dir):
        if f.startswith(os.path.basename(base) + "_batch_") and f.endswith(ext):
            try:
                # Extract batch number from filename
                batch_num = int(f.split("_batch_")[1].split(".")[0])
                existing_batches.append(batch_num)
            except (ValueError, IndexError):
                continue
    
    return sorted(existing_batches)


def predict_pic50_streaming(
    input_file, output_file, model, tokenizer, device, batch_size=32, max_length=512, flush_interval=100000
):
    """
    Predict pIC50 values and write to separate batch files.
    Supports resuming from the last completed batch.

    Args:
        input_file (str): Path to input file with SMILES strings
        output_file (str): Path to output CSV file (used as base name for batch files)
        model: Trained ChemBERTa model
        tokenizer: ChemBERTa tokenizer
        device: torch device (cpu or cuda)
        batch_size (int): Number of compounds to process at once
        max_length (int): Maximum sequence length
        flush_interval (int): Write to disk every N compounds (each batch file)

    Returns:
        tuple: (total_count, valid_count, failed_count, all_predictions_for_stats)
    """
    model.eval()
    
    # Count total lines first
    print("Counting total compounds...")
    with open(input_file, "r") as f:
        total_lines = sum(1 for line in f if line.strip())
    print(f"Total compounds to process: {total_lines}")
    
    # Check for existing batches (resume capability)
    existing_batches = find_existing_batches(output_file)
    start_batch = 0
    skip_lines = 0
    
    if existing_batches:
        start_batch = max(existing_batches)
        skip_lines = start_batch * flush_interval
        print(f"\n[Resume] Found {len(existing_batches)} existing batch file(s)")
        print(f"[Resume] Skipping first {skip_lines:,} compounds (batches 1-{start_batch})")
        print(f"[Resume] Resuming from batch {start_batch + 1}")
    
    # Statistics
    all_predictions = []
    processed_count = 0
    valid_count = 0
    failed_count = 0
    current_batch_num = start_batch
    
    # Buffer for current batch
    buffer = []
    lines_skipped = 0
    
    with open(input_file, "r") as in_f:
        smiles_batch = []
        
        with tqdm(total=total_lines, desc="Predicting", unit="compounds", initial=skip_lines) as pbar:
            for line in in_f:
                smiles = line.strip()
                if not smiles:
                    continue
                
                # Skip already processed lines
                if lines_skipped < skip_lines:
                    lines_skipped += 1
                    continue
                    
                smiles_batch.append(smiles)
                
                # Process when we have enough for a batch
                if len(smiles_batch) >= batch_size:
                    predictions = _predict_batch(smiles_batch, model, tokenizer, device, max_length)
                    
                    for smi, pred in zip(smiles_batch, predictions):
                        buffer.append((smi, pred))
                        all_predictions.append(pred)
                        if np.isnan(pred):
                            failed_count += 1
                        else:
                            valid_count += 1
                    
                    processed_count += len(smiles_batch)
                    pbar.update(len(smiles_batch))
                    smiles_batch = []
                    
                    # Write batch file when buffer is full
                    if len(buffer) >= flush_interval:
                        current_batch_num += 1
                        batch_file = get_batch_filename(output_file, current_batch_num)
                        
                        with open(batch_file, "w") as out_f:
                            out_f.write("SMILES,Predicted_pIC50\n")
                            for smi, pred in buffer:
                                out_f.write(f"{smi},{pred:.4f}\n")
                        
                        total_so_far = skip_lines + processed_count
                        print(f"\n[Batch {current_batch_num}] Saved {batch_file} ({total_so_far:,} compounds total)")
                        buffer = []
            
            # Process remaining SMILES
            if smiles_batch:
                predictions = _predict_batch(smiles_batch, model, tokenizer, device, max_length)
                
                for smi, pred in zip(smiles_batch, predictions):
                    buffer.append((smi, pred))
                    all_predictions.append(pred)
                    if np.isnan(pred):
                        failed_count += 1
                    else:
                        valid_count += 1
                
                processed_count += len(smiles_batch)
                pbar.update(len(smiles_batch))
    
    # Write remaining buffer as final batch
    if buffer:
        current_batch_num += 1
        batch_file = get_batch_filename(output_file, current_batch_num)
        
        with open(batch_file, "w") as out_f:
            out_f.write("SMILES,Predicted_pIC50\n")
            for smi, pred in buffer:
                out_f.write(f"{smi},{pred:.4f}\n")
        
        total_so_far = skip_lines + processed_count
        print(f"\n[Batch {current_batch_num}] Saved {batch_file} ({total_so_far:,} compounds total)")
    
    total_processed = skip_lines + processed_count
    print(f"\n[Complete] All {total_processed:,} compounds processed")
    print(f"[Complete] Output saved in {current_batch_num} batch file(s)")
    print(f"[Complete] Files: {get_batch_filename(output_file, 1)} ... {get_batch_filename(output_file, current_batch_num)}")
    
    return total_processed, valid_count, failed_count, all_predictions


def _predict_batch(smiles_batch, model, tokenizer, device, max_length):
    """
    Predict pIC50 for a single batch of SMILES.
    """
    try:
        encodings = tokenizer(
            smiles_batch,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_predictions = outputs.logits.squeeze().cpu().numpy()
        
        if len(smiles_batch) == 1:
            return [float(batch_predictions)]
        else:
            return batch_predictions.tolist()
    except Exception as e:
        print(f"\nError processing batch: {e}")
        return [np.nan] * len(smiles_batch)


def classify_activity(pic50_value, threshold=6.0):
    """
    Classify compound activity based on pIC50 value.

    Args:
        pic50_value (float): Predicted pIC50 value
        threshold (float): Activity threshold

    Returns:
        str: Activity classification
    """
    if np.isnan(pic50_value):
        return "Error"
    elif pic50_value >= 7.0:
        return "Highly Active"
    elif pic50_value >= threshold:
        return "Active"
    else:
        return "Inactive"


# ============================================================================
# Main Prediction Pipeline
# ============================================================================


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Predict pIC50 values for compounds using trained ChemBERTa-77M-MTR model"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input TXT filename (will be read from 'input/' directory) or full path",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV filename (will be saved to 'output/' directory) or full path",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for prediction (default: 32)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=MODEL_PATH,
        help=f"Path to trained model (default: {MODEL_PATH})",
    )

    args = parser.parse_args()

    # Handle input file path
    if os.path.isabs(args.input) or os.path.exists(args.input):
        input_file = args.input
    else:
        input_file = os.path.join(INPUT_DIR, args.input)

    # Validate input file
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print(
            f"Please place your input file in the 'input/' directory or provide a full path."
        )
        sys.exit(1)

    # Handle output file path
    if os.path.isabs(args.output):
        output_file = args.output
    else:
        output_file = os.path.join(OUTPUT_DIR, args.output)

    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' not found!")
        print(f"Please ensure the model is located at: {MODEL_PATH}")
        sys.exit(1)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    try:
        model, tokenizer = load_model_and_tokenizer(
            args.model_path, BASE_MODEL_NAME, device
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Make predictions with streaming (writes to disk in batches)
    print(f"\nPredicting pIC50 values (streaming mode - writes every {FLUSH_INTERVAL:,} compounds)...")
    print(f"Output file: {output_file}")
    
    total_count, valid_count, failed_count, all_predictions = predict_pic50_streaming(
        input_file,
        output_file,
        model,
        tokenizer,
        device,
        batch_size=args.batch_size,
        max_length=MAX_LENGTH,
        flush_interval=FLUSH_INTERVAL,
    )

    # Print summary statistics
    print("\n" + "=" * 70)
    print("PREDICTION SUMMARY")
    print("=" * 70)
    valid_predictions = [p for p in all_predictions if not np.isnan(p)]
    print(f"Total compounds:        {total_count}")
    print(f"Valid predictions:      {valid_count}")
    print(f"Failed predictions:     {failed_count}")
    print(f"\nOutput files: {os.path.dirname(output_file) or 'output/'}")
    print(f"  Format: {os.path.basename(output_file).replace('.csv', '')}_batch_XXX.csv")
    print(f"\nTo merge all batch files, use:")
    print(f"  PowerShell: Get-Content *_batch_*.csv | Set-Content merged.csv")
    print(f"  Or use Python/pandas to concatenate the files.")

    if len(valid_predictions) > 0:
        print(f"\npIC50 Statistics:")
        print(f"  Mean:                 {np.mean(valid_predictions):.2f}")
        print(f"  Median:               {np.median(valid_predictions):.2f}")
        print(f"  Std Dev:              {np.std(valid_predictions):.2f}")
        print(f"  Min:                  {np.min(valid_predictions):.2f}")
        print(f"  Max:                  {np.max(valid_predictions):.2f}")

        # Show top predictions info
        sorted_preds = sorted(valid_predictions, reverse=True)
        top_10_preds = sorted_preds[:10]
        print(f"\nTop 10 Predicted pIC50 values:")
        print("-" * 70)
        for i, pred in enumerate(top_10_preds, 1):
            print(f"  {i}. pIC50 = {pred:.4f}")
        print(f"\nTo find top compounds, sort the output CSV by Predicted_pIC50 column.")

    print("=" * 70)
    print("\nNote: Model performance is optimized for Drug Discovery Metrics")
    print("      and Virtual Screening Metrics. Results should be validated")
    print("      experimentally for hit-to-lead optimization.")
    print("=" * 70)


if __name__ == "__main__":
    main()

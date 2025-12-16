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


def predict_pic50_batch(
    smiles_list, model, tokenizer, device, batch_size=32, max_length=512
):
    """
    Predict pIC50 values for a batch of SMILES strings.

    Args:
        smiles_list (list): List of SMILES strings
        model: Trained ChemBERTa model
        tokenizer: ChemBERTa tokenizer
        device: torch device (cpu or cuda)
        batch_size (int): Number of compounds to process at once
        max_length (int): Maximum sequence length

    Returns:
        list: List of predicted pIC50 values
    """
    model.eval()
    predictions = []

    # Process in batches with progress bar
    num_batches = (len(smiles_list) + batch_size - 1) // batch_size

    with tqdm(total=len(smiles_list), desc="Predicting", unit="compounds") as pbar:
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i : i + batch_size]

            try:
                # Tokenize batch
                encodings = tokenizer(
                    batch_smiles,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

                # Move to device
                input_ids = encodings["input_ids"].to(device)
                attention_mask = encodings["attention_mask"].to(device)

                # Make predictions
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    batch_predictions = outputs.logits.squeeze().cpu().numpy()

                # Handle single prediction case
                if len(batch_smiles) == 1:
                    predictions.append(float(batch_predictions))
                else:
                    predictions.extend(batch_predictions.tolist())

            except Exception as e:
                print(f"\nError processing batch {i//batch_size + 1}: {e}")
                # Add NaN for failed predictions
                predictions.extend([np.nan] * len(batch_smiles))

            pbar.update(len(batch_smiles))

    return predictions


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

    # Load input data
    print(f"\nLoading input data from {input_file}...")
    try:
        with open(input_file, "r") as f:
            smiles_list = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading input TXT: {e}")
        sys.exit(1)

    print(f"Loaded {len(smiles_list)} compounds")

    # Make predictions
    print(f"\nPredicting pIC50 values...")
    predictions = predict_pic50_batch(
        smiles_list,
        model,
        tokenizer,
        device,
        batch_size=args.batch_size,
        max_length=MAX_LENGTH,
    )

    # Create results list and sort by predicted pIC50 in descending order
    results = list(zip(smiles_list, predictions))
    results.sort(
        key=lambda x: x[1] if not np.isnan(x[1]) else float("-inf"), reverse=True
    )

    # Save results
    print(f"\nSaving predictions to {output_file}...")
    try:
        with open(output_file, "w") as f:
            f.write("SMILES,Predicted_pIC50\n")
            for smiles, pred in results:
                f.write(f"{smiles},{pred:.4f}\n")
        print(f"Predictions saved successfully!")
    except Exception as e:
        print(f"Error saving output CSV: {e}")
        sys.exit(1)

    # Print summary statistics
    print("\n" + "=" * 70)
    print("PREDICTION SUMMARY")
    print("=" * 70)
    valid_predictions = [p for p in predictions if not np.isnan(p)]
    print(f"Total compounds:        {len(smiles_list)}")
    print(f"Valid predictions:      {len(valid_predictions)}")
    print(f"Failed predictions:     {len(smiles_list) - len(valid_predictions)}")

    if len(valid_predictions) > 0:
        print(f"\npIC50 Statistics:")
        print(f"  Mean:                 {np.mean(valid_predictions):.2f}")
        print(f"  Median:               {np.median(valid_predictions):.2f}")
        print(f"  Std Dev:              {np.std(valid_predictions):.2f}")
        print(f"  Min:                  {np.min(valid_predictions):.2f}")
        print(f"  Max:                  {np.max(valid_predictions):.2f}")

        # Highlight top 10% most active compounds
        top_percent = max(1, int(len(results) * 0.1))  # At least 1 compound
        print(
            f"\nTop 10% Most Active Compounds ({top_percent} compounds, Predicted pIC50):"
        )
        print("-" * 70)
        for smiles, pred in results[:top_percent]:
            display_smiles = smiles[:47] + "..." if len(smiles) > 50 else smiles
            print(f"  {pred:.2f}  -  {display_smiles}")

    print("=" * 70)
    print("\nNote: Model performance is optimized for Drug Discovery Metrics")
    print("      and Virtual Screening Metrics. Results should be validated")
    print("      experimentally for hit-to-lead optimization.")
    print("=" * 70)


if __name__ == "__main__":
    main()

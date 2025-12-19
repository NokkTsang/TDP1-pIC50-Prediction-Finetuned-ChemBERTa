"""
TDP1 pIC50 Prediction Results Analyzer
=======================================
This script analyzes the prediction results from batch CSV files.

Features:
- Count total compounds across all batch files
- Calculate pIC50 statistics
- Classify compounds by activity level
- Extract top active compounds
- Generate summary report

Usage:
    python results_analyze.py
    python results_analyze.py --output_dir path/to/output
    python results_analyze.py --top_n 1000 --threshold 6.0

Author: Generated for NokkTsang
Date: December 2025
"""

import os
import sys
import argparse
import glob
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
SAVE_DIR = PROJECT_ROOT  # Save analysis results to project root

# Activity thresholds
HIGHLY_ACTIVE_THRESHOLD = 7.0
ACTIVE_THRESHOLD = 6.0
MODERATE_THRESHOLD = 5.0

# ============================================================================
# Analysis Functions
# ============================================================================


def find_batch_files(output_dir, exclude_patterns=None):
    """
    Find all batch CSV files in the output directory.
    
    Args:
        output_dir (str): Path to output directory
        exclude_patterns (list): Patterns to exclude (e.g., ['example_'])
    
    Returns:
        list: Sorted list of batch file paths
    """
    if exclude_patterns is None:
        exclude_patterns = ['example_']
    
    # Find all CSV files with batch pattern
    batch_pattern = os.path.join(output_dir, "*_batch_*.csv")
    batch_files = glob.glob(batch_pattern)
    
    # Filter out excluded patterns
    filtered_files = []
    for f in batch_files:
        filename = os.path.basename(f)
        if not any(pattern in filename for pattern in exclude_patterns):
            filtered_files.append(f)
    
    # Sort by batch number
    def get_batch_num(filepath):
        try:
            filename = os.path.basename(filepath)
            # Extract batch number from filename like predictions_123M_batch_001.csv
            parts = filename.split("_batch_")
            if len(parts) == 2:
                return int(parts[1].split(".")[0])
        except:
            pass
        return 0
    
    return sorted(filtered_files, key=get_batch_num)


def count_total_compounds(batch_files):
    """
    Count total compounds across all batch files.
    
    Args:
        batch_files (list): List of batch file paths
    
    Returns:
        tuple: (total_count, file_counts_dict)
    """
    total = 0
    file_counts = {}
    
    print("Counting compounds in batch files...")
    for filepath in tqdm(batch_files, desc="Scanning files"):
        # Count lines (subtract 1 for header)
        with open(filepath, 'r') as f:
            count = sum(1 for _ in f) - 1  # Subtract header
        file_counts[os.path.basename(filepath)] = count
        total += count
    
    return total, file_counts


def analyze_predictions(batch_files, sample_size=None):
    """
    Analyze pIC50 predictions from batch files.
    
    Args:
        batch_files (list): List of batch file paths
        sample_size (int): If provided, sample this many rows for faster analysis
    
    Returns:
        dict: Analysis results
    """
    all_predictions = []
    
    print("\nLoading predictions for analysis...")
    for filepath in tqdm(batch_files, desc="Loading data"):
        df = pd.read_csv(filepath)
        all_predictions.extend(df['Predicted_pIC50'].tolist())
    
    predictions = np.array(all_predictions)
    
    # Calculate statistics
    stats = {
        'total_compounds': len(predictions),
        'mean': np.mean(predictions),
        'median': np.median(predictions),
        'std': np.std(predictions),
        'min': np.min(predictions),
        'max': np.max(predictions),
        'q25': np.percentile(predictions, 25),
        'q75': np.percentile(predictions, 75),
        'q90': np.percentile(predictions, 90),
        'q95': np.percentile(predictions, 95),
        'q99': np.percentile(predictions, 99),
    }
    
    # Activity classification
    stats['highly_active_count'] = int(np.sum(predictions >= HIGHLY_ACTIVE_THRESHOLD))
    stats['active_count'] = int(np.sum((predictions >= ACTIVE_THRESHOLD) & (predictions < HIGHLY_ACTIVE_THRESHOLD)))
    stats['moderate_count'] = int(np.sum((predictions >= MODERATE_THRESHOLD) & (predictions < ACTIVE_THRESHOLD)))
    stats['inactive_count'] = int(np.sum(predictions < MODERATE_THRESHOLD))
    
    # Percentage calculations
    total = len(predictions)
    stats['highly_active_pct'] = 100 * stats['highly_active_count'] / total
    stats['active_pct'] = 100 * stats['active_count'] / total
    stats['moderate_pct'] = 100 * stats['moderate_count'] / total
    stats['inactive_pct'] = 100 * stats['inactive_count'] / total
    
    return stats, predictions


def extract_top_compounds(batch_files, top_n=1000, threshold=None):
    """
    Extract top compounds by pIC50 value.
    
    Args:
        batch_files (list): List of batch file paths
        top_n (int): Number of top compounds to extract
        threshold (float): If provided, extract all compounds above this threshold
    
    Returns:
        pd.DataFrame: Top compounds with SMILES and pIC50
    """
    all_data = []
    
    print(f"\nExtracting top compounds...")
    for filepath in tqdm(batch_files, desc="Processing"):
        df = pd.read_csv(filepath)
        
        if threshold is not None:
            # Filter by threshold
            df = df[df['Predicted_pIC50'] >= threshold]
        
        all_data.append(df)
    
    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    
    # Sort by pIC50 descending
    combined = combined.sort_values('Predicted_pIC50', ascending=False)
    
    if threshold is None:
        # Return top N
        return combined.head(top_n)
    else:
        # Return all above threshold
        return combined


def plot_distribution(predictions, output_dir, stats):
    """
    Plot pIC50 distribution histogram with activity thresholds.
    
    Args:
        predictions (np.array): Array of pIC50 predictions
        output_dir (str): Directory to save the plot
        stats (dict): Statistics dictionary
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Plot 1: Full Distribution ---
    ax1 = axes[0]
    
    # Create histogram
    n, bins, patches = ax1.hist(predictions, bins=100, color='steelblue', 
                                 edgecolor='black', alpha=0.7, density=False)
    
    # Add vertical lines for thresholds
    ax1.axvline(x=ACTIVE_THRESHOLD, color='green', linestyle='--', linewidth=2, 
                label=f'Active threshold (pIC50={ACTIVE_THRESHOLD})')
    ax1.axvline(x=HIGHLY_ACTIVE_THRESHOLD, color='red', linestyle='--', linewidth=2, 
                label=f'Highly Active threshold (pIC50={HIGHLY_ACTIVE_THRESHOLD})')
    ax1.axvline(x=MODERATE_THRESHOLD, color='orange', linestyle='--', linewidth=2, 
                label=f'Moderate threshold (pIC50={MODERATE_THRESHOLD})')
    
    # Add mean and median lines
    ax1.axvline(x=stats['mean'], color='purple', linestyle='-', linewidth=2, 
                label=f"Mean ({stats['mean']:.2f})")
    ax1.axvline(x=stats['median'], color='magenta', linestyle=':', linewidth=2, 
                label=f"Median ({stats['median']:.2f})")
    
    ax1.set_xlabel('Predicted pIC50', fontsize=12)
    ax1.set_ylabel('Number of Compounds', fontsize=12)
    ax1.set_title('pIC50 Distribution - Full Range', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Format y-axis with millions
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    # --- Plot 2: Zoomed to Active Region ---
    ax2 = axes[1]
    
    # Filter for active region
    active_region = predictions[predictions >= 5.0]
    
    n2, bins2, patches2 = ax2.hist(active_region, bins=50, color='forestgreen', 
                                    edgecolor='black', alpha=0.7)
    
    # Color bars by activity level
    for i, (patch, left_edge) in enumerate(zip(patches2, bins2[:-1])):
        if left_edge >= HIGHLY_ACTIVE_THRESHOLD:
            patch.set_facecolor('red')
        elif left_edge >= ACTIVE_THRESHOLD:
            patch.set_facecolor('green')
        else:
            patch.set_facecolor('orange')
    
    ax2.axvline(x=ACTIVE_THRESHOLD, color='green', linestyle='--', linewidth=2)
    ax2.axvline(x=HIGHLY_ACTIVE_THRESHOLD, color='red', linestyle='--', linewidth=2)
    
    ax2.set_xlabel('Predicted pIC50', fontsize=12)
    ax2.set_ylabel('Number of Compounds', fontsize=12)
    ax2.set_title('pIC50 Distribution - Active Region (pIC50 ≥ 5.0)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add text annotation for counts
    textstr = f"Active (≥6.0): {stats['active_count'] + stats['highly_active_count']:,}\n"
    textstr += f"({stats['active_pct'] + stats['highly_active_pct']:.2f}%)"
    ax2.text(0.95, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Format y-axis with thousands
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.0f}K' if x >= 1000 else f'{x:.0f}'))
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, 'pIC50_distribution.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Distribution plot saved to: {plot_file}")
    
    # Also save as PDF for publication quality
    plot_pdf = os.path.join(output_dir, 'pIC50_distribution.pdf')
    plt.savefig(plot_pdf, format='pdf', bbox_inches='tight')
    print(f"📊 Distribution plot saved to: {plot_pdf}")
    
    plt.show()


def print_summary_report(stats, num_files):
    """
    Print a formatted summary report.
    """
    print("\n" + "=" * 80)
    print("                     PREDICTION RESULTS ANALYSIS REPORT")
    print("=" * 80)
    
    print(f"\n📁 DATA OVERVIEW")
    print("-" * 40)
    print(f"  Batch files analyzed:     {num_files:,}")
    print(f"  Total compounds:          {stats['total_compounds']:,}")
    
    print(f"\n📊 pIC50 STATISTICS")
    print("-" * 40)
    print(f"  Mean:                     {stats['mean']:.4f}")
    print(f"  Median:                   {stats['median']:.4f}")
    print(f"  Std Dev:                  {stats['std']:.4f}")
    print(f"  Min:                      {stats['min']:.4f}")
    print(f"  Max:                      {stats['max']:.4f}")
    
    print(f"\n📈 PERCENTILE DISTRIBUTION")
    print("-" * 40)
    print(f"  25th percentile (Q1):     {stats['q25']:.4f}")
    print(f"  75th percentile (Q3):     {stats['q75']:.4f}")
    print(f"  90th percentile:          {stats['q90']:.4f}")
    print(f"  95th percentile:          {stats['q95']:.4f}")
    print(f"  99th percentile:          {stats['q99']:.4f}")
    
    print(f"\n🎯 ACTIVITY CLASSIFICATION")
    print("-" * 40)
    print(f"  Highly Active (pIC50 ≥ {HIGHLY_ACTIVE_THRESHOLD}):")
    print(f"    Count:                  {stats['highly_active_count']:,}")
    print(f"    Percentage:             {stats['highly_active_pct']:.4f}%")
    
    print(f"\n  Active ({ACTIVE_THRESHOLD} ≤ pIC50 < {HIGHLY_ACTIVE_THRESHOLD}):")
    print(f"    Count:                  {stats['active_count']:,}")
    print(f"    Percentage:             {stats['active_pct']:.4f}%")
    
    print(f"\n  Moderate ({MODERATE_THRESHOLD} ≤ pIC50 < {ACTIVE_THRESHOLD}):")
    print(f"    Count:                  {stats['moderate_count']:,}")
    print(f"    Percentage:             {stats['moderate_pct']:.4f}%")
    
    print(f"\n  Inactive (pIC50 < {MODERATE_THRESHOLD}):")
    print(f"    Count:                  {stats['inactive_count']:,}")
    print(f"    Percentage:             {stats['inactive_pct']:.4f}%")
    
    # Combined active summary
    total_active = stats['highly_active_count'] + stats['active_count']
    total_active_pct = stats['highly_active_pct'] + stats['active_pct']
    print(f"\n  📌 TOTAL ACTIVE (pIC50 ≥ {ACTIVE_THRESHOLD}):")
    print(f"    Count:                  {total_active:,}")
    print(f"    Percentage:             {total_active_pct:.4f}%")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze TDP1 pIC50 prediction results"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Directory containing batch CSV files (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=1000,
        help="Number of top compounds to extract (default: 1000)"
    )
    
    args = parser.parse_args()
    
    # Find batch files
    print(f"Scanning directory: {args.output_dir}")
    batch_files = find_batch_files(args.output_dir)
    
    if not batch_files:
        print("Error: No batch files found!")
        print(f"Looking for pattern: *_batch_*.csv in {args.output_dir}")
        sys.exit(1)
    
    print(f"Found {len(batch_files)} batch files")
    print(f"  First: {os.path.basename(batch_files[0])}")
    print(f"  Last:  {os.path.basename(batch_files[-1])}")
    
    # Count total compounds
    total_count, file_counts = count_total_compounds(batch_files)
    print(f"\nTotal compounds across all files: {total_count:,}")
    
    # Analyze predictions
    stats, predictions = analyze_predictions(batch_files)
    
    # Print summary report
    print_summary_report(stats, len(batch_files))
    
    # Plot distribution (save to project root)
    plot_distribution(predictions, SAVE_DIR, stats)
    
    # Extract and save top compounds (save to project root)
    print(f"\nExtracting top {args.top_n} compounds...")
    top_compounds = extract_top_compounds(batch_files, top_n=args.top_n)
    
    top_file = os.path.join(SAVE_DIR, f"top_{args.top_n}_compounds.csv")
    top_compounds.to_csv(top_file, index=False)
    print(f"Saved to: {top_file}")
    
    print(f"\nTop 10 compounds:")
    print("-" * 60)
    for i, row in top_compounds.head(10).iterrows():
        smiles = row['SMILES'][:50] + "..." if len(row['SMILES']) > 50 else row['SMILES']
        print(f"  {row['Predicted_pIC50']:.4f}  {smiles}")
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()

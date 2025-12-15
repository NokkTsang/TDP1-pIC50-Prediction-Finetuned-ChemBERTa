"""
TDP1 pIC50 Prediction - ChemBERTa-MTR Training with Sample Weighting
======================================================================
This script implements sample weighting strategy instead of oversampling
for training ChemBERTa-MTR on the TDP1 pIC50 dataset.

Strategy:
1. Load final_pIC50_dataset.csv (177,092 compounds)
2. Stratified split (70/15/15) maintaining 2.1% active ratio
3. NO oversampling - use original distribution
4. Apply sample weights: higher weight for rare active compounds
5. Train ChemBERTa-77M-MTR with weighted loss function

Author: NokkTsang
Date: November 2025
"""

import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from collections import Counter

# ============================================================================
# Configuration
# ============================================================================

# Paths
DATA_PATH = "Data/Large_Data_177024/original data/final_pIC50_dataset.csv"
SPLIT_DIR = "Data/Large_Data_177024/stratified_split"  # Shared split directory
MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"
OUTPUT_DIR = "Model/ChemBERTa-77M-MTR-sample-weighting"
RESULTS_DIR = "results_ChemBERTa-77M-MTR-sample-weighting"

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)

# Hyperparameters (optimized by Optuna - 10 trials, 1.45 hours, val_loss=0.1793)
# Previous values: LR=4.66e-05 → 4.61e-05, BATCH=64 → 32, WEIGHT_DECAY=0.004 → 0.0011
#                  EPOCHS=10 → 7, DROPOUT=0.073 → 0.100, WARMUP=0.232 → 0.076
LEARNING_RATE = 4.6070962557673397e-05
BATCH_SIZE = 32
WEIGHT_DECAY = 0.0011178018548585339
NUM_EPOCHS = 7
DROPOUT = 0.09954554774384414
WARMUP_RATIO = 0.07602832965396175
MAX_LENGTH = 512

# Activity threshold
ACTIVE_THRESHOLD = 6.0

# Random seed for reproducibility
SEED = 1155193167
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================================
# Custom Dataset Class with Sample Weights
# ============================================================================

class WeightedSMILESDataset(Dataset):
    """PyTorch Dataset for SMILES strings and pIC50 values with sample weights."""
    
    def __init__(self, smiles_list, pic50_list, weights, tokenizer, max_length=512):
        self.smiles = smiles_list
        self.pic50 = pic50_list
        self.weights = weights
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        smiles = str(self.smiles[idx])
        pic50 = float(self.pic50[idx])
        weight = float(self.weights[idx])
        
        encoding = self.tokenizer(
            smiles,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(pic50, dtype=torch.float),
            'weight': torch.tensor(weight, dtype=torch.float)
        }

# ============================================================================
# Custom Weighted Trainer
# ============================================================================

class WeightedDataCollator:
    """Custom data collator that preserves the 'weight' field if present."""
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer  # tokenizer not actually used here
    
    def __call__(self, features):
        if isinstance(features[0], dict):
            # Check if 'weight' field exists in the first feature
            has_weight = 'weight' in features[0]
            
            # Handle labels
            labels = torch.stack([
                f['labels'] if isinstance(f['labels'], torch.Tensor) 
                else torch.tensor(f['labels'], dtype=torch.float) 
                for f in features
            ])
            
            # Handle input_ids and attention_mask
            input_ids = torch.stack([f['input_ids'] for f in features])
            attention_mask = torch.stack([f['attention_mask'] for f in features])
            
            # Handle weights - create dummy weights if not present
            if has_weight:
                weights = torch.stack([
                    f['weight'] if isinstance(f['weight'], torch.Tensor) 
                    else torch.tensor(f['weight'], dtype=torch.float) 
                    for f in features
                ])
            else:
                # Create dummy weights for evaluation (not used in eval loss)
                weights = torch.ones(len(features), dtype=torch.float)
            
            batch = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'weight': weights
            }
            return batch
        else:
            raise ValueError(f"Unexpected feature type: {type(features[0])}")
        
        return batch

class WeightedMSETrainer(Trainer):
    """Custom Trainer with weighted MSE loss."""
    
    def _remove_unused_columns(self, dataset, description=None):
        """
        Override to keep the 'weight' column.
        Transformers' Trainer automatically removes columns not used by the model,
        but we need 'weight' for our custom loss function.
        """
        # Don't remove any columns - we need 'weight' for loss calculation
        return dataset
    
    def get_train_dataloader(self):
        """
        Override to ensure 'weight' field is not removed from features.
        """
        from torch.utils.data import DataLoader, RandomSampler
        
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        
        train_sampler = RandomSampler(train_dataset)
        
        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            num_workers=0,  # Single process to avoid serialization issues
            pin_memory=True
        )
    
    def get_eval_dataloader(self, eval_dataset=None):
        """
        Override to ensure 'weight' field is not removed from features.
        """
        from torch.utils.data import DataLoader, SequentialSampler
        
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator
        
        eval_sampler = SequentialSampler(eval_dataset)
        
        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            sampler=eval_sampler,
            collate_fn=data_collator,
            num_workers=0,
            pin_memory=True
        )
    
    def get_test_dataloader(self, test_dataset):
        """
        Override to ensure 'weight' field is not removed from features during prediction.
        """
        from torch.utils.data import DataLoader, SequentialSampler
        
        data_collator = self.data_collator
        test_sampler = SequentialSampler(test_dataset)
        
        return DataLoader(
            test_dataset,
            batch_size=self.args.eval_batch_size,
            sampler=test_sampler,
            collate_fn=data_collator,
            num_workers=0,
            pin_memory=True
        )
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute weighted MSE loss.
        
        Loss = mean(weight * (prediction - target)^2)
        
        Args:
            num_items_in_batch: Added for compatibility with Transformers 4.x
        """
        labels = inputs.pop("labels")
        weights = inputs.pop("weight")
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)  # Only squeeze last dimension
        
        # Weighted MSE loss
        loss = torch.mean(weights * (logits - labels) ** 2)
        
        return (loss, outputs) if return_outputs else loss

# ============================================================================
# Data Preparation Functions
# ============================================================================

def load_and_prepare_data(data_path, active_threshold=6.0):
    """Load dataset and prepare for stratified splitting."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"Total compounds: {len(df)}")
    print(f"pIC50 range: {df['pIC50'].min():.2f} - {df['pIC50'].max():.2f}")
    print(f"pIC50 mean: {df['pIC50'].mean():.2f} ± {df['pIC50'].std():.2f}")
    
    # Create binary activity label for stratification
    df['is_active'] = (df['pIC50'] >= active_threshold).astype(int)
    
    active_count = df['is_active'].sum()
    inactive_count = len(df) - active_count
    active_pct = (active_count / len(df)) * 100
    
    print(f"\nActivity distribution (threshold = {active_threshold}):")
    print(f"  Active: {active_count} ({active_pct:.1f}%)")
    print(f"  Inactive: {inactive_count} ({100-active_pct:.1f}%)")
    print(f"  Imbalance ratio: {inactive_count/active_count:.1f}:1")
    
    return df

def stratified_split_data(df, test_size=0.15, val_size=0.15, force_new_split=False):
    """
    Perform stratified train/val/test split with caching.
    
    If split files exist, load them. Otherwise, create new split and save.
    This ensures all models (MLM, MTR, oversampling, sample-weighting) use identical splits.
    
    Args:
        df: Full dataset
        test_size: Test set proportion
        val_size: Validation set proportion
        force_new_split: If True, ignore cached files and create new split
    
    Returns:
        train_df, val_df, test_df
    """
    # Define split file paths
    train_path = os.path.join(SPLIT_DIR, "train.csv")
    val_path = os.path.join(SPLIT_DIR, "validation.csv")
    test_path = os.path.join(SPLIT_DIR, "test.csv")
    
    # Check if split files exist
    splits_exist = all(os.path.exists(p) for p in [train_path, val_path, test_path])
    
    if splits_exist and not force_new_split:
        # Load existing splits
        print("\nLoading existing stratified split from cache...")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        print(f"Loaded cached splits:")
        for name, split_df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
            active = split_df['is_active'].sum()
            total = len(split_df)
            print(f"   {name}: {total} samples ({active} active, {active/total*100:.1f}%)")
        
        print(f"\nUsing SEED={SEED} for reproducibility")
        print(f"Split files: {SPLIT_DIR}")
        
    else:
        # Create new split
        if force_new_split:
            print("\nForce creating new stratified split...")
        else:
            print("\nCreating new stratified split (first time)...")
        
        # First split: train + val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=SEED,
            stratify=df['is_active']
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=SEED,
            stratify=train_val_df['is_active']
        )
        
        # Print split statistics
        print(f"Split completed:")
        for name, split_df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
            active = split_df['is_active'].sum()
            total = len(split_df)
            print(f"   {name}: {total} samples ({active} active, {active/total*100:.1f}%)")
        
        # Save splits to CSV
        print(f"\nSaving split to {SPLIT_DIR}...")
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        print(f"Splits saved for future use")
        print(f"SEED={SEED} ensures reproducibility")
    
    return train_df, val_df, test_df

def compute_sample_weights(df, active_threshold=6.0, weight_method='balanced'):
    """
    Compute sample weights based on class imbalance.
    
    Args:
        df: DataFrame with pIC50 and is_active columns
        active_threshold: Threshold for active compounds
        weight_method: 'balanced' (sklearn style) or 'inverse' (custom)
    
    Returns:
        Array of sample weights
    """
    print(f"\nComputing sample weights using '{weight_method}' method...")
    
    binary_labels = df['is_active'].values
    
    if weight_method == 'balanced':
        # sklearn-style balanced weights: n_samples / (n_classes * n_samples_per_class)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(binary_labels),
            y=binary_labels
        )
        
        # Map class weights to sample weights
        weight_map = {0: class_weights[0], 1: class_weights[1]}
        sample_weights = np.array([weight_map[label] for label in binary_labels])
        
    elif weight_method == 'inverse':
        # Custom inverse frequency weights
        n_total = len(binary_labels)
        n_active = binary_labels.sum()
        n_inactive = n_total - n_active
        
        # Weight inversely proportional to class frequency
        active_weight = n_total / (2 * n_active)
        inactive_weight = n_total / (2 * n_inactive)
        
        sample_weights = np.where(binary_labels == 1, active_weight, inactive_weight)
    
    else:
        raise ValueError(f"Unknown weight_method: {weight_method}")
    
    # Print weight statistics
    active_mask = binary_labels == 1
    print(f"  Active compounds weight:   {sample_weights[active_mask].mean():.4f}")
    print(f"  Inactive compounds weight: {sample_weights[~active_mask].mean():.4f}")
    print(f"  Weight ratio (active/inactive): {sample_weights[active_mask].mean() / sample_weights[~active_mask].mean():.2f}x")
    print(f"  Total weight sum: {sample_weights.sum():.2f}")
    
    return sample_weights

def save_split_summary(train_df, val_df, test_df, train_weights, output_path):
    """Save data split summary to JSON."""
    summary = {
        "model": "ChemBERTa-77M-MTR",
        "strategy": "Sample Weighting (No Oversampling)",
        "active_threshold": ACTIVE_THRESHOLD,
        "sample_weighting": {
            "method": "balanced",
            "active_weight": float(train_weights[train_df['is_active'] == 1].mean()),
            "inactive_weight": float(train_weights[train_df['is_active'] == 0].mean()),
            "weight_ratio": float(train_weights[train_df['is_active'] == 1].mean() / 
                                 train_weights[train_df['is_active'] == 0].mean())
        },
        "splits": {
            "train": {
                "total": len(train_df),
                "active": int(train_df['is_active'].sum()),
                "inactive": int((train_df['is_active'] == 0).sum()),
                "active_percentage": float(train_df['is_active'].mean() * 100)
            },
            "validation": {
                "total": len(val_df),
                "active": int(val_df['is_active'].sum()),
                "inactive": int((val_df['is_active'] == 0).sum()),
                "active_percentage": float(val_df['is_active'].mean() * 100)
            },
            "test": {
                "total": len(test_df),
                "active": int(test_df['is_active'].sum()),
                "inactive": int((test_df['is_active'] == 0).sum()),
                "active_percentage": float(test_df['is_active'].mean() * 100)
            }
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSplit summary saved to {output_path}")
    return summary

# ============================================================================
# Model Training Functions
# ============================================================================

def compute_metrics(eval_pred):
    """Compute regression and drug discovery metrics."""
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # Regression metrics
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, predictions)
    
    # Drug discovery metrics
    binary_labels = (labels >= ACTIVE_THRESHOLD).astype(int)
    try:
        auc_roc = roc_auc_score(binary_labels, predictions)
    except:
        auc_roc = 0.5
    
    # Ranking differential
    active_preds = predictions[binary_labels == 1]
    inactive_preds = predictions[binary_labels == 0]
    ranking_diff = active_preds.mean() - inactive_preds.mean() if len(active_preds) > 0 else 0
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'auc_roc': auc_roc,
        'ranking_differential': ranking_diff
    }

def train_model(train_dataset, val_dataset, output_dir):
    """Train ChemBERTa-MTR model with sample weighting."""
    print("\nInitializing ChemBERTa-MTR model...")
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,
        hidden_dropout_prob=DROPOUT
    )
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on: {device}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mae",
        greater_is_better=False,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        use_cpu=False,
        no_cuda=False,
        dataloader_num_workers=0,  # Disable multiprocessing for custom collator
        seed=SEED,
        report_to="none"
    )
    
    # Initialize weighted trainer
    trainer = WeightedMSETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=WeightedDataCollator(train_dataset.tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    print("\nStarting training with sample weighting...")
    train_result = trainer.train()
    
    # Save model
    print(f"\nSaving best model to {output_dir}...")
    trainer.save_model(output_dir)
    
    return trainer, train_result

# ============================================================================
# Evaluation Functions
# ============================================================================

def compute_enrichment_metrics(pred_values, true_labels, top_k_percentages=[1, 2, 5, 10]):
    """
    Compute Enrichment Factor (EF) and Precision@K for virtual screening.
    
    Args:
        pred_values: Predicted pIC50 values
        true_labels: Binary labels (1=active, 0=inactive)
        top_k_percentages: List of top K% to evaluate
    
    Returns:
        Dictionary with EF and Precision@K metrics
    """
    n_total = len(pred_values)
    n_actives = true_labels.sum()
    
    # Sort by predicted values (descending)
    sorted_indices = np.argsort(pred_values)[::-1]
    sorted_labels = true_labels[sorted_indices]
    
    enrichment_metrics = {}
    
    for k_pct in top_k_percentages:
        # Calculate number of compounds in top K%
        k = max(1, int(n_total * k_pct / 100))
        
        # Get actives in top K
        top_k_labels = sorted_labels[:k]
        n_actives_in_top_k = top_k_labels.sum()
        
        # Precision@K
        precision_k = n_actives_in_top_k / k
        
        # Enrichment Factor
        enrichment_factor = (n_actives_in_top_k / n_actives) / (k / n_total) if n_actives > 0 else 0
        
        enrichment_metrics[f'top_{k_pct}%'] = {
            'compounds_screened': k,
            'actives_found': int(n_actives_in_top_k),
            'precision': float(precision_k),
            'enrichment_factor': float(enrichment_factor)
        }
    
    return enrichment_metrics

def evaluate_model(trainer, test_dataset, test_df, output_dir):
    """Comprehensive model evaluation on test set."""
    print("\nEvaluating model on test set...")
    
    # Get predictions
    predictions = trainer.predict(test_dataset)
    pred_values = predictions.predictions.flatten()
    true_values = test_df['pIC50'].values
    
    # Overall metrics
    mse = mean_squared_error(true_values, pred_values)
    mae = mean_absolute_error(true_values, pred_values)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_values, pred_values)
    
    # Drug discovery metrics
    binary_labels = (true_values >= ACTIVE_THRESHOLD).astype(int)
    auc_roc = roc_auc_score(binary_labels, pred_values)
    
    active_mask = binary_labels == 1
    active_preds = pred_values[active_mask]
    inactive_preds = pred_values[~active_mask]
    ranking_diff = active_preds.mean() - inactive_preds.mean()
    
    # Conditional metrics
    active_mse = mean_squared_error(true_values[active_mask], pred_values[active_mask])
    active_mae = mean_absolute_error(true_values[active_mask], pred_values[active_mask])
    active_r2 = r2_score(true_values[active_mask], pred_values[active_mask])
    
    inactive_mse = mean_squared_error(true_values[~active_mask], pred_values[~active_mask])
    inactive_mae = mean_absolute_error(true_values[~active_mask], pred_values[~active_mask])
    inactive_r2 = r2_score(true_values[~active_mask], pred_values[~active_mask])
    
    # Enrichment metrics
    enrichment_metrics = compute_enrichment_metrics(pred_values, binary_labels)
    
    # Compile results
    results = {
        "model": "ChemBERTa-77M-MTR-sample-weighting",
        "overall_metrics": {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2)
        },
        "drug_discovery_metrics": {
            "auc_roc": float(auc_roc),
            "ranking_differential": float(ranking_diff),
            "active_mean_prediction": float(active_preds.mean()),
            "inactive_mean_prediction": float(inactive_preds.mean()),
            "prediction_range": {
                "min": float(pred_values.min()),
                "max": float(pred_values.max())
            }
        },
        "enrichment_metrics": enrichment_metrics,
        "conditional_metrics": {
            "active_compounds": {
                "count": int(active_mask.sum()),
                "mse": float(active_mse),
                "mae": float(active_mae),
                "r2": float(active_r2)
            },
            "inactive_compounds": {
                "count": int((~active_mask).sum()),
                "mse": float(inactive_mse),
                "mae": float(inactive_mae),
                "r2": float(inactive_r2)
            }
        }
    }
    
    # Save results
    results_path = os.path.join(RESULTS_DIR, "test_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTest results saved to {results_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SET PERFORMANCE SUMMARY - SAMPLE WEIGHTING")
    print("="*70)
    print(f"\nOverall Metrics:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"\nDrug Discovery Metrics:")
    print(f"  AUC-ROC:              {auc_roc:.4f}")
    print(f"  Ranking Differential: {ranking_diff:.4f} pIC50 units")
    print(f"  Active Predictions:   {active_preds.mean():.2f} ± {active_preds.std():.2f}")
    print(f"  Inactive Predictions: {inactive_preds.mean():.2f} ± {inactive_preds.std():.2f}")
    print(f"\nEnrichment Metrics (Virtual Screening):")
    for k_pct, metrics in enrichment_metrics.items():
        print(f"  {k_pct}:")
        print(f"    Compounds screened: {metrics['compounds_screened']}")
        print(f"    Actives found:      {metrics['actives_found']} / {active_mask.sum()}")
        print(f"    Precision@{k_pct}:       {metrics['precision']:.4f}")
        print(f"    Enrichment Factor:  {metrics['enrichment_factor']:.2f}x")
    print(f"\nConditional Performance:")
    print(f"  Active ({active_mask.sum()} compounds):   MAE = {active_mae:.4f}, R² = {active_r2:.4f}")
    print(f"  Inactive ({(~active_mask).sum()} compounds): MAE = {inactive_mae:.4f}, R² = {inactive_r2:.4f}")
    print("="*70)
    
    return results, pred_values, true_values

def plot_results(pred_values, true_values, output_dir):
    """Create comprehensive visualization of results."""
    print("\nGenerating visualizations...")
    
    binary_labels = (true_values >= ACTIVE_THRESHOLD).astype(int)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('ChemBERTa-MTR Performance - Sample Weighting Strategy', 
                 fontsize=16, fontweight='bold')
    
    # 1. Predictions vs True
    ax = axes[0, 0]
    active_mask = binary_labels == 1
    ax.scatter(true_values[~active_mask], pred_values[~active_mask], 
               alpha=0.3, s=10, label='Inactive', color='blue')
    ax.scatter(true_values[active_mask], pred_values[active_mask], 
               alpha=0.6, s=20, label='Active', color='red')
    ax.plot([true_values.min(), true_values.max()], 
            [true_values.min(), true_values.max()], 
            'k--', lw=2, label='Perfect')
    ax.axhline(y=ACTIVE_THRESHOLD, color='green', linestyle='--', alpha=0.5)
    ax.axvline(x=ACTIVE_THRESHOLD, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('True pIC50', fontsize=12)
    ax.set_ylabel('Predicted pIC50', fontsize=12)
    ax.set_title('Predictions vs True Values', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Residual plot
    ax = axes[0, 1]
    residuals = pred_values - true_values
    ax.scatter(true_values[~active_mask], residuals[~active_mask], 
               alpha=0.3, s=10, color='blue')
    ax.scatter(true_values[active_mask], residuals[active_mask], 
               alpha=0.6, s=20, color='red')
    ax.axhline(y=0, color='k', linestyle='--', lw=2)
    ax.axvline(x=ACTIVE_THRESHOLD, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('True pIC50', fontsize=12)
    ax.set_ylabel('Residual', fontsize=12)
    ax.set_title('Residual Plot', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 3. Distribution comparison
    ax = axes[0, 2]
    ax.hist(true_values, bins=50, alpha=0.5, label='True', color='blue', density=True)
    ax.hist(pred_values, bins=50, alpha=0.5, label='Predicted', color='red', density=True)
    ax.axvline(x=ACTIVE_THRESHOLD, color='green', linestyle='--', lw=2, label='Threshold')
    ax.set_xlabel('pIC50', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('pIC50 Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Box plot by activity
    ax = axes[1, 0]
    data_to_plot = [pred_values[~active_mask], pred_values[active_mask]]
    bp = ax.boxplot(data_to_plot, labels=['Inactive', 'Active'], patch_artist=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('red')
    ax.axhline(y=ACTIVE_THRESHOLD, color='green', linestyle='--', lw=2)
    ax.set_ylabel('Predicted pIC50', fontsize=12)
    ax.set_title('Prediction by Activity', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5. Error distribution
    ax = axes[1, 1]
    ax.hist(residuals, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', lw=2)
    ax.set_xlabel('Prediction Error', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Error Distribution (MAE={np.abs(residuals).mean():.3f})', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 6. Metrics summary
    ax = axes[1, 2]
    ax.axis('off')
    
    mse = mean_squared_error(true_values, pred_values)
    mae = mean_absolute_error(true_values, pred_values)
    r2 = r2_score(true_values, pred_values)
    auc_roc = roc_auc_score(binary_labels, pred_values)
    ranking_diff = pred_values[active_mask].mean() - pred_values[~active_mask].mean()
    
    metrics_text = f"""
    PERFORMANCE METRICS
    {'='*40}
    
    Model: ChemBERTa-MTR
    Strategy: Sample Weighting
    
    Regression Metrics:
    • MSE:  {mse:.4f}
    • MAE:  {mae:.4f}
    • RMSE: {np.sqrt(mse):.4f}
    • R²:   {r2:.4f}
    
    Drug Discovery Metrics:
    • AUC-ROC: {auc_roc:.4f}
    • Ranking Δ: {ranking_diff:+.3f} pIC50
    
    Dataset Statistics:
    • Total: {len(true_values)}
    • Active: {active_mask.sum()} ({active_mask.sum()/len(true_values)*100:.1f}%)
    • Inactive: {(~active_mask).sum()}
    """
    
    ax.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(RESULTS_DIR, "comprehensive_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {plot_path}")
    plt.close()

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main training pipeline."""
    print("\n" + "="*70)
    print("TDP1 pIC50 PREDICTION - ChemBERTa-MTR TRAINING")
    print("Sample Weighting Strategy (No Oversampling)")
    print("="*70)
    
    # Check CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! This script requires GPU.")
    
    device = torch.device("cuda")
    print(f"\nDevice: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    
    # 1. Load data
    df = load_and_prepare_data(DATA_PATH, ACTIVE_THRESHOLD)
    
    # 2. Stratified split
    train_df, val_df, test_df = stratified_split_data(df)
    
    # 3. Compute sample weights for training set
    train_weights = compute_sample_weights(train_df, ACTIVE_THRESHOLD, weight_method='balanced')
    
    # For validation/test, use uniform weights (no weighting during evaluation)
    val_weights = np.ones(len(val_df))
    test_weights = np.ones(len(test_df))
    
    # 4. Save split summary
    split_summary = save_split_summary(
        train_df, 
        val_df, 
        test_df,
        train_weights,
        os.path.join(RESULTS_DIR, "split_summary.json")
    )
    
    # 5. Initialize tokenizer
    print("\nLoading ChemBERTa-MTR tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 6. Create datasets with weights
    print("Creating weighted PyTorch datasets...")
    train_dataset = WeightedSMILESDataset(
        train_df['SMILES'].tolist(),
        train_df['pIC50'].tolist(),
        train_weights,
        tokenizer,
        MAX_LENGTH
    )
    
    val_dataset = WeightedSMILESDataset(
        val_df['SMILES'].tolist(),
        val_df['pIC50'].tolist(),
        val_weights,
        tokenizer,
        MAX_LENGTH
    )
    
    test_dataset = WeightedSMILESDataset(
        test_df['SMILES'].tolist(),
        test_df['pIC50'].tolist(),
        test_weights,
        tokenizer,
        MAX_LENGTH
    )
    
    print(f"Train dataset: {len(train_dataset)} samples (weighted)")
    print(f"Validation dataset: {len(val_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    
    # 7. Train model
    trainer, train_result = train_model(train_dataset, val_dataset, OUTPUT_DIR)
    
    # 8. Evaluate on test set
    results, pred_values, true_values = evaluate_model(
        trainer, 
        test_dataset, 
        test_df, 
        OUTPUT_DIR
    )
    
    # 9. Create visualizations
    plot_results(pred_values, true_values, RESULTS_DIR)
    
    # 10. Save training history
    history_path = os.path.join(RESULTS_DIR, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"Results saved to: {RESULTS_DIR}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

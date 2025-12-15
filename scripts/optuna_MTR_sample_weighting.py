"""
Optuna Hyperparameter Optimization for ChemBERTa-MTR with Sample Weighting
===========================================================================
Fast optimization (10 trials, 2-3 hours) to find best hyperparameters
for sample weighting strategy.

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
    TrainerCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import optuna
from datetime import datetime
import gc

# ============================================================================
# Configuration
# ============================================================================

DATA_PATH = "Data/Large_Data_177024/original data/final_pIC50_dataset.csv"
SPLIT_DIR = "Data/Large_Data_177024/stratified_split"
MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"
OPTUNA_DIR = "optuna_results_MTR_sample_weighting"
ACTIVE_THRESHOLD = 6.0
SEED = 1155193167

os.makedirs(OPTUNA_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================================
# Custom Dataset and Trainer
# ============================================================================

class WeightedSMILESDataset(Dataset):
    """PyTorch Dataset with sample weights."""
    
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

class WeightedDataCollator:
    """Custom data collator that preserves the 'weight' field."""
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        # Handle both dict and tensor inputs
        if isinstance(features[0], dict):
            # Extract weights and labels
            weights = torch.stack([f['weight'] if isinstance(f['weight'], torch.Tensor) else torch.tensor(f['weight'], dtype=torch.float) for f in features])
            labels = torch.stack([f['labels'] if isinstance(f['labels'], torch.Tensor) else torch.tensor(f['labels'], dtype=torch.float) for f in features])
            
            # Stack input_ids and attention_mask
            input_ids = torch.stack([f['input_ids'] for f in features])
            attention_mask = torch.stack([f['attention_mask'] for f in features])
            
            batch = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'weight': weights
            }
        else:
            raise ValueError(f"Unexpected feature type: {type(features[0])}")
        
        return batch

class WeightedMSETrainer(Trainer):
    """Custom Trainer with weighted MSE loss."""
    
    def _remove_unused_columns(self, dataset, description=None):
        """Override to keep the 'weight' column."""
        return dataset
    
    def get_train_dataloader(self):
        """Override to ensure 'weight' field is not removed from features."""
        from torch.utils.data import DataLoader, RandomSampler
        
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        train_sampler = RandomSampler(train_dataset)
        
        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            num_workers=0,
            pin_memory=True
        )
    
    def get_eval_dataloader(self, eval_dataset=None):
        """Override to ensure 'weight' field is not removed from features."""
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
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        weights = inputs.pop("weight")
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)
        loss = torch.mean(weights * (logits - labels) ** 2)
        return (loss, outputs) if return_outputs else loss

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_and_prepare_data():
    """Load dataset and prepare for stratified splitting."""
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df['is_active'] = (df['pIC50'] >= ACTIVE_THRESHOLD).astype(int)
    
    active_count = df['is_active'].sum()
    inactive_count = len(df) - active_count
    
    print(f"Loaded {len(df)} compounds")
    print(f"   Active: {active_count} ({active_count/len(df)*100:.1f}%)")
    print(f"   Inactive: {inactive_count} ({inactive_count/len(df)*100:.1f}%)")
    
    return df

def load_or_create_split(df):
    """Load existing split or create new one."""
    train_path = os.path.join(SPLIT_DIR, "train.csv")
    val_path = os.path.join(SPLIT_DIR, "validation.csv")
    test_path = os.path.join(SPLIT_DIR, "test.csv")
    
    if all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        print("\nLoading cached stratified split...")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        print("Loaded cached splits")
    else:
        print("\nCreating stratified split...")
        train_val_df, test_df = train_test_split(
            df, test_size=0.15, random_state=SEED, stratify=df['is_active']
        )
        train_df, val_df = train_test_split(
            train_val_df, test_size=0.15/(1-0.15), random_state=SEED, 
            stratify=train_val_df['is_active']
        )
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        print("Split created and saved")
    
    for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        active = split_df['is_active'].sum()
        print(f"   {name}: {len(split_df)} samples ({active} active, {active/len(split_df)*100:.1f}%)")
    
    return train_df, val_df, test_df

def compute_sample_weights(df):
    """Compute balanced sample weights."""
    binary_labels = df['is_active'].values
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(binary_labels),
        y=binary_labels
    )
    weight_map = {0: class_weights[0], 1: class_weights[1]}
    sample_weights = np.array([weight_map[label] for label in binary_labels])
    return sample_weights

# ============================================================================
# Optuna Objective
# ============================================================================

def objective(trial, train_dataset, val_dataset):
    """Optuna objective function for hyperparameter optimization."""
    
    print(f"\n{'='*70}")
    print(f"Trial {trial.number + 1}")
    print(f"{'='*70}")
    
    # Hyperparameter search space
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 96])
    weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True)
    num_epochs = trial.suggest_int("num_epochs", 3, 7)
    dropout = trial.suggest_float("dropout", 0.05, 0.2)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.2)
    
    print(f"Hyperparameters:")
    print(f"   LR: {learning_rate:.2e}, Batch: {batch_size}, Weight Decay: {weight_decay:.4f}")
    print(f"   Epochs: {num_epochs}, Dropout: {dropout:.3f}, Warmup: {warmup_ratio:.3f}")
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,
        hidden_dropout_prob=dropout
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(OPTUNA_DIR, f"trial_{trial.number}"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="no",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        seed=SEED,
        report_to="none",
        disable_tqdm=True
    )
    
    # Early stopping callback
    class EarlyStoppingCallback(TrainerCallback):
        def __init__(self, patience=1):
            self.patience = patience
            self.best_loss = float('inf')
            self.counter = 0
        
        def on_evaluate(self, args, state, control, metrics, **kwargs):
            current_loss = metrics.get('eval_loss', float('inf'))
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    control.should_training_stop = True
    
    # Train
    trainer = WeightedMSETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=WeightedDataCollator(),
        callbacks=[EarlyStoppingCallback(patience=1)]
    )
    
    trainer.train()
    
    # Evaluate
    eval_results = trainer.evaluate()
    val_loss = eval_results['eval_loss']
    
    print(f"Trial {trial.number + 1} completed - Val Loss: {val_loss:.4f}")
    
    # Clean up
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()
    
    return val_loss

# ============================================================================
# Main Optimization Pipeline
# ============================================================================

def main():
    print("\n" + "="*70)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION - ChemBERTa-MTR Sample Weighting")
    print("="*70)
    print(f"Fast optimization: 10 trials, ~2-3 hours")
    print(f"SEED: {SEED}")
    
    # Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available! GPU required.")
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    df = load_and_prepare_data()
    train_df, val_df, test_df = load_or_create_split(df)
    
    # Compute weights
    print("\nComputing sample weights...")
    train_weights = compute_sample_weights(train_df)
    val_weights = np.ones(len(val_df))
    
    active_weight = train_weights[train_df['is_active'] == 1].mean()
    inactive_weight = train_weights[train_df['is_active'] == 0].mean()
    print(f"   Active weight: {active_weight:.2f}x")
    print(f"   Inactive weight: {inactive_weight:.2f}x")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = WeightedSMILESDataset(
        train_df['SMILES'].tolist(),
        train_df['pIC50'].tolist(),
        train_weights,
        tokenizer
    )
    
    val_dataset = WeightedSMILESDataset(
        val_df['SMILES'].tolist(),
        val_df['pIC50'].tolist(),
        val_weights,
        tokenizer
    )
    
    print(f"   Train: {len(train_dataset)} samples (weighted)")
    print(f"   Val: {len(val_dataset)} samples")
    
    # Create Optuna study
    print("\nCreating Optuna study...")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        study_name=f"MTR_sample_weighting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Run optimization
    print("\n" + "="*70)
    print("STARTING OPTIMIZATION")
    print("="*70)
    start_time = datetime.now()
    
    study.optimize(
        lambda trial: objective(trial, train_dataset, val_dataset),
        n_trials=10,
        show_progress_bar=True
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 3600
    
    # Save results
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETED!")
    print("="*70)
    print(f"Duration: {duration:.2f} hours")
    print(f"Best trial: {study.best_trial.number + 1}")
    print(f"Best val loss: {study.best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
    
    # Save best params
    best_params_path = os.path.join(OPTUNA_DIR, "best_params.json")
    with open(best_params_path, 'w') as f:
        json.dump({
            'best_params': study.best_params,
            'best_value': study.best_value,
            'best_trial': study.best_trial.number,
            'n_trials': len(study.trials),
            'duration_hours': duration
        }, f, indent=2)
    
    print(f"\nBest parameters saved to: {best_params_path}")
    
    # Save all trials
    trials_data = []
    for trial in study.trials:
        trials_data.append({
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'state': trial.state.name
        })
    
    trials_path = os.path.join(OPTUNA_DIR, "all_trials.json")
    with open(trials_path, 'w') as f:
        json.dump(trials_data, f, indent=2)
    
    print(f"All trials saved to: {trials_path}")
    
    # Print top 5 trials
    print(f"\nTop 5 Trials:")
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))
    for i, trial in enumerate(sorted_trials[:5], 1):
        print(f"   {i}. Trial {trial.number + 1}: Loss={trial.value:.4f}")
        print(f"      LR={trial.params['learning_rate']:.2e}, Batch={trial.params['batch_size']}")
    
    print("\n" + "="*70)
    print("Next step: Use best hyperparameters in training script")
    print("="*70)
    print(f"\nUpdate ChemBERTa-77M-MTR-sample-weighting.py with:")
    print(f"LEARNING_RATE = {study.best_params['learning_rate']:.2e}")
    print(f"BATCH_SIZE = {study.best_params['batch_size']}")
    print(f"WEIGHT_DECAY = {study.best_params['weight_decay']:.4f}")
    print(f"NUM_EPOCHS = {study.best_params['num_epochs']}")
    print(f"DROPOUT = {study.best_params['dropout']:.3f}")
    print(f"WARMUP_RATIO = {study.best_params['warmup_ratio']:.3f}")
    print()

if __name__ == "__main__":
    main()

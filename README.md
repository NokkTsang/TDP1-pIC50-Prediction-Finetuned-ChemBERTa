# TDP1 pIC50 Prediction - Using ChemBERTa-77M-MTR

## Usage

This section provides instructions for using the trained ChemBERTa-77M-MTR model to predict pIC50 values for novel compounds.

### Quick Prediction with Script

The easiest way to make predictions is using the provided prediction script:

#### Prerequisites

```bash
pip install torch transformers pandas numpy tqdm
```

#### Basic Usage

```bash
# Input file should be placed in the 'input/' directory
python scripts/predict_pic50.py --input your_compounds.csv --output predictions.csv

# Or use full paths
python scripts/predict_pic50.py --input /path/to/your_compounds.csv --output /path/to/predictions.csv
```

#### Input Format

Place your input CSV file in the `input/` directory. It must contain a `SMILES` column with valid SMILES strings:

```csv
SMILES
CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O
CC(=O)Oc1ccccc1C(=O)O
CN1C=NC2=C1C(=O)N(C(=O)N2C)C
```

Additional columns (e.g., compound IDs, names) will be preserved in the output.

#### Output Format

The script generates a CSV file in the `output/` directory with:
- All original columns from input
- `Predicted_pIC50`: Predicted pIC50 value for each compound
- **Sorted by Predicted_pIC50 in descending order** (most potent compounds first)

Example output:
```csv
SMILES,Predicted_pIC50
CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O,7.45
CC(=O)Oc1ccccc1C(=O)O,6.23
CN1C=NC2=C1C(=O)N(C(=O)N2C)C,5.12
```

#### Advanced Options

```bash
# Specify custom SMILES column name
python scripts/predict_pic50.py --input data.csv --output results.csv --smiles_column "Canonical_SMILES"

# Adjust batch size for GPU memory (default: 32)
python scripts/predict_pic50.py --input data.csv --output results.csv --batch_size 64

# Use custom model path
python scripts/predict_pic50.py --input data.csv --output results.csv --model_path "path/to/model"
```

#### Example Workflow

```bash
# 1. Test with the provided example file (located in input/ directory)
python scripts/predict_pic50.py --input example_input.csv --output example_predictions.csv

# 2. Place your own compounds CSV in the input/ directory and run prediction
python scripts/predict_pic50.py --input your_compounds.csv --output predictions.csv

# 3. Find results in the output/ directory. The script will display:
#    - Progress bar during prediction
#    - Summary statistics (mean, median, min, max pIC50)
#    - Top 10% most active compounds
```

#### Important Notes

⚠️ **Model Performance Context**: This model is optimized for **Drug Discovery Metrics** and **Virtual Screening Metrics**. The predictions are designed to:
- Identify potential TDP1 inhibitors in virtual screening campaigns
- Prioritize compounds for experimental validation
- Support hit-to-lead optimization

**Recommendations**:
- Use predictions as a **screening tool**, not absolute activity values
- Always validate top predictions experimentally
- Consider the model's focus on drug-like chemical space
- Results are most reliable for compounds similar to the training data distribution

## Future Improvements

### ChemBERTa-77M-MTR Optimization

**Enhance Optuna Hyperparameter Search:**
The current optimization uses only **10 trials (~1.5 hours)**, which provides a baseline but may not fully explore the hyperparameter space. Increasing to **50-100 trials** would allow more thorough exploration of learning rates, batch sizes, weight decay, and dropout combinations, potentially improving validation loss by 10-20%. Additionally, implementing **multi-objective optimization** to simultaneously minimize overall loss and maximize performance on active compounds (pIC50 ≥ 6.0) could better balance the model's ability to identify high-potency inhibitors.

**Extended Search Space:**
Beyond the current 6 hyperparameters, incorporate additional parameters such as **layer-wise learning rates** (lower rates for pretrained layers, higher for the regression head), **gradient clipping thresholds**, and **learning rate schedulers** (cosine decay, polynomial warmup). These refinements can stabilize training and improve convergence, especially for the imbalanced dataset.

**Advanced Training Strategies:**
Implement **stratified sample weighting** where ultra-potent compounds (pIC50 > 8.0) receive 100-200x weight instead of uniform 23.3x for all actives. Alternatively, use **focal loss** with γ=2-5 to heavily penalize mispredictions on rare extreme values, forcing the model to learn the full pIC50 range. Additionally, **two-stage modeling** (first classify into activity bins, then regress within bins) could prevent range compression. **Ensemble methods** combining the top 3-5 Optuna trials could reduce prediction variance and improve robustness, especially for extreme values.

**Expected Impact:**
Addressing ultra-potent compound prediction through stratified weighting + focal loss could reduce errors on pIC50 > 8.0 from **3-4 units to <1 unit**. Combined with extended hyperparameter search, validation loss could decrease from ~0.179 to **~0.150-0.160**, translating to better identification across the full activity spectrum.

### Data-Driven Optimization Strategies

**SMILES Augmentation:**
Generate multiple equivalent SMILES representations using **SMILES enumeration** (randomizing atom ordering while preserving structure). For the **275 ultra-potent compounds (pIC50 > 8.0)**, create **20-50 variants each** to effectively increase their representation from ~192 to ~4,000-10,000 training examples. For moderate actives, 5-10 variants suffice. This targeted augmentation addresses the extreme rarity issue (0.16% of dataset) while acting as regularization, improving generalization by 8-15% overall and **potentially 50%+ for ultra-potent prediction**.

**Active Learning for Imbalanced Data:**
Instead of fixed sample weights (23.3x for actives), implement **uncertainty-based reweighting** where compounds with high prediction uncertainty receive dynamically increased weights during training. After each epoch, identify the top 10% most uncertain predictions and boost their weights by 1.5x, allowing the model to focus on hard-to-learn boundary cases between active and inactive compounds. This adaptive approach can improve AUC-ROC by 5-10% compared to static weighting. **Critically**, separately track ultra-potent compounds (pIC50 > 8.0) and maintain minimum 100x weight to prevent range compression.

**Scaffold-Based Data Splitting:**
Replace the current random stratified split with **scaffold splitting** using molecular scaffolds (core structures). This ensures that chemically similar compounds don't appear in both training and test sets, providing a more realistic evaluation of the model's ability to generalize to novel chemical series. While this typically increases validation loss initially, it produces models that perform 15-20% better on truly out-of-distribution compounds in real drug discovery scenarios.

**Physicochemical Property Filtering:**
Pre-filter the dataset to remove compounds violating **Lipinski's Rule of Five** or with poor ADME properties, as these are unlikely drug candidates regardless of pIC50. This focuses the model on drug-like chemical space, reducing noise from ~15% of compounds (26,500 entries) and improving prediction accuracy on viable drug candidates by 10-15%.

**Expected Combined Impact:**
Combining targeted SMILES augmentation (20-50x for ultra-potent) + stratified sample weighting (100-200x for pIC50 > 8.0) + external data enrichment (doubling ultra-potent examples) + scaffold splitting could:
- **Reduce ultra-potent prediction error from 3-4 units to <1 unit** (addressing the critical weakness)
- Improve validation loss from ~0.179 to **~0.140-0.155** (13-22% gain)
- Achieve consistent performance across the full activity spectrum, not just moderate range

These data-centric approaches directly address the **root cause** (extreme rarity of ultra-potent compounds) rather than just symptom management through hyperparameters.

### Alternative Architecture

- Explore **Molformer** ([IBM/molformer](https://github.com/IBM/molformer)) as an alternative transformer architecture specifically designed for molecular property prediction, which may offer complementary strengths to ChemBERTa-MTR's multi-task pretraining.

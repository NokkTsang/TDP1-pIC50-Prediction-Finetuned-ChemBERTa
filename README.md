# TDP1 pIC50 Prediction - Using ChemBERTa-77M-MTR

## Usage

This section provides instructions for using the trained ChemBERTa-77M-MTR model to predict pIC50 values for novel compounds.

### Prepare Prediction Data

For large-scale virtual screening, you can use the PubChem compound library (~123 million compounds).

**Download the pre-processed data file:**

- [Google Drive: PubChem_123M_prediction.gz](https://drive.google.com/file/d/1-tP6yzSQv0vXTiB2lBt0Xoo_eimn6M53/view?usp=sharing) (1.47 GB)

_Original source: [PubChem FTP server](ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/) - `CID-SMILES.gz`_

```bash
# 1. Download from Google Drive and place in data/ folder

# 2. Unzip (or use an app)
gunzip data/PubChem_123M_prediction.gz

# 3. Run script to extract SMILES and filter out training compounds (to avoid data leakage)
python scripts/prepare_prediction_data.py
```

Output: `input/final_smiles_prediction.txt` with ~123 million SMILES ready for prediction.

### Quick Prediction with Script

The easiest way to make predictions is using the provided prediction script:

#### Prerequisites

```bash
pip install torch transformers pandas numpy tqdm
```

#### Basic Usage

```bash
# Input file should be placed in the 'input/' directory
python scripts/predict_pic50.py --input your_compounds.txt --output predictions.csv

# Or use full paths
python scripts/predict_pic50.py --input /path/to/your_compounds.txt --output /path/to/predictions.csv
```

#### Input Format

Place your input TXT file in the `input/` directory. One SMILES string per line:

```
CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O
CC(=O)Oc1ccccc1C(=O)O
CN1C=NC2=C1C(=O)N(C(=O)N2C)C
```

#### Output Format

The script generates a CSV file in the `output/` directory with:

- Format: `SMILES,Predicted_pIC50`
- **Sorted by Predicted_pIC50 in descending order** (most potent compounds first)

Example output:

```csv
SMILES,Predicted_pIC50
CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O,7.4500
CC(=O)Oc1ccccc1C(=O)O,6.2300
CN1C=NC2=C1C(=O)N(C(=O)N2C)C,5.1200
```

#### Advanced Options

```bash
# Adjust batch size for GPU memory (default: 32)
python scripts/predict_pic50.py --input data.txt --output results.csv --batch_size 64

# Use custom model path
python scripts/predict_pic50.py --input data.txt --output results.csv --model_path "path/to/model"
```

#### Example Workflow

```bash
# 1. Test with the provided example file (located in input/ directory)
python scripts/predict_pic50.py --input example_input.txt --output example_predictions.csv

# 2. Place your own compounds TXT in the input/ directory and run prediction
python scripts/predict_pic50.py --input your_compounds.txt --output predictions.csv

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

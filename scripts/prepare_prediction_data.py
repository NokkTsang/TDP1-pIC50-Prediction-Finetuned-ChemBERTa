import os

# === Input files ===
prediction_path = "data/PubChem_123M_prediction.gz"
train_path = "data/177k_training.txt"

# === Output file ===
output_path = "input/final_smiles_prediction.txt"

# === 1. Load training SMILES into a set ===
train_smiles = set()

with open(train_path, "r") as f:
    next(f)  # skip header: SMILES,pIC50
    for line in f:
        line = line.strip()
        if not line:
            continue
        smiles = line.split(",")[0]  # SMILES is before the comma
        train_smiles.add(smiles)

print(f"Training SMILES loaded: {len(train_smiles)}")

# === 2. Extract SMILES from CID-SMILES and filter ===
count_extracted = 0
count_kept = 0

with open(prediction_path, "r") as infile, open(output_path, "w") as outfile:
    for line in infile:
        line = line.strip()
        if not line:
            continue

        # Each line format: <CID>\t<SMILES>
        parts = line.split("\t")
        if len(parts) != 2:
            continue  # skip malformed lines

        smiles = parts[1]
        count_extracted += 1

        # Keep only if not in training dataset
        if smiles not in train_smiles:
            outfile.write(smiles + "\n")
            count_kept += 1

print("Filtering + Extraction completed")
print(f"Total SMILES extracted from CID-SMILES: {count_extracted}")
print(f"SMILES kept after removing training SMILES: {count_kept}")
print(f"Output saved to: {output_path}")

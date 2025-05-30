from rdkit import Chem
from rdkit.Chem import AllChem
import os
import csv
import re

def safe_filename(smiles):
    # Replace any character that is not alphanumeric, dot, underscore or dash with underscore
    return re.sub(r'[^A-Za-z0-9._-]', '_', smiles)

def get3d(smiles):
    try:
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=10)
            AllChem.MMFFOptimizeMolecule(mol, mmffVariant='MMFF94s')
            return mol
        else:
            failed_smiles.append(smiles)
            return None
    except Exception as e:
        failed_smiles.append(smiles)
        print(f"[Failed] SMILES: {smiles} -> {e}")
        return None

# List to capture any SMILES strings that failed processing
failed_smiles = []
# List of all SMILES from the input
smiles_list = []

input_file = 'smile.csv'
smiles_field = 'Drug'

output_dir = 'drug_sdf'
os.makedirs(output_dir, exist_ok=True)

# Prepare index file to map SMILES to generated SDF filenames
index_path = os.path.join(output_dir, 'smiles_to_filename_train.csv')
index_list = []

# Read SMILES strings from CSV
with open(input_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        smiles = row[smiles_field].strip()
        smiles_list.append(smiles)

# Generate 3D conformers and write to SDF
for smiles in smiles_list:
    mol3d = get3d(smiles)
    if mol3d:
        filename = safe_filename(smiles)[:150]
        sdf_path = os.path.join(output_dir, f"{filename}.sdf")

        writer = Chem.SDWriter(sdf_path)
        writer.write(mol3d)
        writer.close()

        index_list.append({'smiles': smiles, 'filename': f"{filename}.sdf"})
        print(f"[Success] Wrote {sdf_path}")
    else:
        print(f"[Failed] SMILES: {smiles}")

# Write mapping of SMILES to filenames
with open(index_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['smiles', 'filename'])
    writer.writeheader()
    writer.writerows(index_list)

# Summary of failures
print(f"\nNumber of failed SMILES: {len(failed_smiles)}")
if failed_smiles:
    print("Failed SMILES examples:", failed_smiles[:5])

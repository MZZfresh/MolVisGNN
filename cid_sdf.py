from rdkit import Chem
from rdkit.Chem import AllChem
import pubchempy as pcp
import os
import csv

# List to track any CIDs that fail processing
failed_cids = []

def get3d(drug_id, smiles):
    try:
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            mol3d = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol3d, randomSeed=10)
            AllChem.MMFFGetMoleculeForceField(
                mol3d,
                AllChem.MMFFGetMoleculeProperties(mol3d, mmffVariant='MMFF94s')
            )
            return mol3d
        else:
            print(f"[Warning] No SMILES found for CID {drug_id}")
            failed_cids.append(drug_id)
            return None
    except Exception as e:
        print(f"[Error] Failed to process CID {drug_id}: {e}")
        failed_cids.append(drug_id)
        return None

# Read CID and SMILES pairs from CSV
drugid_list = []
smiles_list = []
with open('drug_smiles.csv', 'r') as infile:
    reader = csv.reader(infile)
    for row in reader:
        drugid_list.append(row[0].strip())
        smiles_list.append(row[1].strip())

# Generate and save 3D structures
output_folder = 'drug_sdf'
os.makedirs(output_folder, exist_ok=True)

for cid, smiles in zip(drugid_list, smiles_list):
    mol3d = get3d(cid, smiles)
    if mol3d:
        sdf_path = os.path.join(output_folder, f"{cid}_output.sdf")
        writer = Chem.SDWriter(sdf_path)
        writer.write(mol3d)
        writer.close()
        print(f"[Success] 3D structure for CID {cid} saved to {sdf_path}")
    else:
        print(f"[Failed] Could not generate 3D structure for CID {cid}")

# Summary of any failures
print("Failed CIDs:", failed_cids)

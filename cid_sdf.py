
from rdkit import Chem
from rdkit.Chem import AllChem
import pubchempy as pcp
import os
import csv

faile_cid = []
def get3d(drug_id,smiles):
    try:
        if smiles:
            m = Chem.MolFromSmiles(smiles)
            m3d = Chem.AddHs(m)
            AllChem.EmbedMolecule(m3d, randomSeed=10)
            ff = AllChem.MMFFGetMoleculeForceField(m3d, AllChem.MMFFGetMoleculeProperties(m3d, mmffVariant='MMFF94s'))
            ff.Minimize(maxIts=200)
            return m3d
            
        else:
            print(f"无法获取CID {drug_id} 的SMILES")
            faile_cid.append(drug_id)
            return None
    except Exception as e:
        print(f"无法处理CID {drug_id}: {e}")
        faile_cid.append(drug_id)
        return None

# 从CSV文件中读取CID列表
drugid_list = []
smiles_list = []
with open('drug_smiles.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        drugid_list.append(row[0].strip())
        smiles_list.append(row[1].strip())


# Generate 3D structure from CID
for drug_id, smiles in zip(drugid_list, smiles_list):  # Fixed to zip the drugid_list and smiles_list
    m3d = get3d(drug_id, smiles)  # Fixed argument to pass both drug_id and smiles

    if m3d:
        output_folder = 'drug_sdf'
        os.makedirs(output_folder, exist_ok=True)

        # Save the 3D structure to an SDF file
        output_filename = os.path.join(output_folder, f"{drug_id}_output.sdf")

        w = Chem.SDWriter(output_filename)
        w.write(m3d)
        w.close()
        print(f"CID {drug_id} 3D结构已保存到 {output_filename}")
    else:
        faile_cid.append(drug_id)
        print(f"无法为CID {drug_id} 生成3D结构")

print("失败的cid : ", faile_cid)

# MolVisGNN
Leveraging 3D molecular spatial visual information and multi-perspective representations for drug discovery
# Drug 3D Preparation & Feature Extraction

A simple pipeline to generate 3D SDF files from SMILES or PubChem CIDs, render six orthogonal views in PyMOL, and extract 3D spatial features as NumPy arrays.

---

## 🚀 Overview

1. **Generate SDF**  
   - `smiles_sdf.py` — convert SMILES strings into 3D SDF files  
   - `cid_sdf.py`    — fetch SMILES by PubChem CID and convert into 3D SDF  

2. **Render Six Views**  
   - Use PyMOL to load each SDF, orient to the six orthogonal views (front, right, back, left, top, bottom), and export each image plus corresponding `.npy` arrays.

3. **Extract 3D Features**  
   - `3dfeature.py` — load the six-view NumPy arrays and compute spatial visual descriptors for downstream modeling.

The `main.py` files in the three files can run different tasks.

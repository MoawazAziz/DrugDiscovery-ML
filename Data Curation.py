import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import SaltRemover
import warnings
warnings.filterwarnings("ignore")

print("🚀 Starting Data Curation Pipeline...")

# 1. Load Data
df = pd.read_csv("DS_BreastCancer.csv", low_memory=False)
print(f"Initial rows: {len(df)}")

# 2. Drop Critical Missing Values
# We need SMILES, IC50 value, Units, and Molecular Weight for accurate conversion
df = df.dropna(subset=["smiles", "ic50_value", "ic50_units", "molecular_weight"])
print(f"After dropping NaNs: {len(df)}")

# 3. Filter IC50 Units
# Keep only standard units we can reliably convert
valid_units = ["nM", "ug.mL-1"]
df = df[df["ic50_units"].isin(valid_units)].copy()
print(f"After filtering units ({valid_units}): {len(df)}")

# 4. Filter Standard Relation
# For regression, we need exact values, not '>' or '<'
# Keep only '=' or approximate '~'. Drop '>' or '<' as they are bounds.
if "standard_relation" in df.columns:
    df = df[df["standard_relation"].isin(["=", "~", "approx"])].copy()
    print(f"After filtering relations (=, ~): {len(df)}")

# 5. Convert IC50 to nM (SCIENTIFIC FIX)
# Formula: IC50(nM) = (IC50(ug/mL) * 1e6) / Molecular Weight
def convert_to_nM(row):
    if row["ic50_units"] == "nM":
        return row["ic50_value"]
    elif row["ic50_units"] == "ug.mL-1":
        if row["molecular_weight"] > 0:
            return (row["ic50_value"] * 1e6) / row["molecular_weight"]
        else:
            return np.nan
    return np.nan

df["ic50_nM"] = df.apply(convert_to_nM, axis=1)
df = df.dropna(subset=["ic50_nM"])

# 6. Filter Valid IC50 Range
# Remove zeros (log error) and physically impossible values (> 100 mM)
df = df[(df["ic50_nM"] > 0) & (df["ic50_nM"] < 1e8)]
print(f"After filtering IC50 range (0 - 100 mM): {len(df)}")

# 7. Calculate pIC50
df["pIC50"] = -np.log10(df["ic50_nM"] * 1e-9) # Standard pIC50 = -log10(M)
# Alternative consistent with your previous code: 9 - log10(nM)
df["pIC50"] = 9 - np.log10(df["ic50_nM"])

# 8. SMILES Standardization (Remove Salts)
# Keeps only the largest fragment (main molecule)
remover = SaltRemover.SaltRemover()
def clean_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            # Remove salts
            mol = remover.StripMol(mol, dontRemoveEverything=True)
            # Convert back to SMILES
            return Chem.MolToSmiles(mol)
        return None
    except:
        return None

print("Standardizing SMILES...")
df["smiles_clean"] = df["smiles"].apply(clean_smiles)
df = df.dropna(subset=["smiles_clean"])
print(f"After SMILES cleaning: {len(df)}")

# 9. Remove Exact Duplicates (based on clean SMILES)
df = df.drop_duplicates(subset=["smiles_clean"])
print(f"After removing duplicates: {len(df)}")

# 10. Create Activity Label (for Classification)
# Standard threshold: pIC50 >= 6.0 (1 µM)
df["active"] = (df["pIC50"] >= 6.0).astype(int)

# 11. Save Cleaned Dataset
output_file = "DS_BreastCancer_Cleaned.csv"
df.to_csv(output_file, index=False)
print(f"\n✅ Cleaning Complete!")
print(f"💾 Saved: {output_file}")

# 12. Final Statistics
print("\n" + "="*40)
print("FINAL DATASET STATISTICS")
print("="*40)
print(f"Total Compounds: {len(df)}")
print(f"Active (pIC50 ≥ 6): {df['active'].sum()} ({df['active'].mean():.1%})")
print(f"Inactive (pIC50 < 6): {len(df) - df['active'].sum()} ({1 - df['active'].mean():.1%})")
print(f"pIC50 Range: {df['pIC50'].min():.2f} - {df['pIC50'].max():.2f}")
print(f"IC50 nM Range: {df['ic50_nM'].min():.2e} - {df['ic50_nM'].max():.2e}")
print(f"Units Converted: {df['ic50_units'].value_counts().to_dict()}")
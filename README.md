# ScaffoldAware-ML-DualERa-PI3Ka

> Scaffold-aware machine learning (ML-QSAR) framework integrating molecular docking and molecular dynamics simulations to prioritize dual ERα/PI3Kα inhibitor candidates for ER-positive breast cancer.

---

## Overview

This repository contains the complete machine learning and structure-based drug discovery pipeline developed for the manuscript:

**Scaffold-Aware Machine Learning and Structure-Based Validation Prioritize Dual ERα/PI3Kα Inhibitor Candidates for ER-Positive Breast Cancer**

The workflow combines:

- Data curation from ChEMBL
- Molecular descriptor generation
- Morgan fingerprints
- Scaffold-aware data splitting
- Machine learning model benchmarking
- External scaffold validation
- Applicability domain analysis
- Compound prioritization
- Molecular docking
- Molecular dynamics simulations

---

# Workflow

```
Raw ChEMBL Dataset
        │
        ▼
Data Curation
        │
        ▼
SMILES Standardization
        │
        ▼
Duplicate Removal
        │
        ▼
pIC50 Calculation
        │
        ▼
Activity Labeling
        │
        ▼
Morgan Fingerprints + Molecular Descriptors
        │
        ▼
Scaffold-aware Train/Test Split
        │
        ▼
Model Training
(Random Forest
ExtraTrees
LightGBM
XGBoost
CatBoost)
        │
        ▼
5-Fold Cross Validation
        │
        ▼
Best Model Selection
        │
        ▼
External Scaffold Validation
        │
        ▼
Applicability Domain
        │
        ▼
Compound Prioritization
        │
        ▼
Molecular Docking
        │
        ▼
Molecular Dynamics Simulation
```

---

# Repository Structure

```
ScaffoldAware-ML-DualERa-PI3Ka/

│── data/
│     ├── DS_BreastCancer.csv
│     ├── DS_BreastCancer_Cleaned.csv
│
│── scripts/
│     ├── data_curation.py
│     ├── scaffold_ml_pipeline.py
│
│── outputs/
│     ├── figures/
│     ├── tables/
│     ├── metrics/
│     ├── predictions/
│
│── docking/
│
│── molecular_dynamics/
│
│── README.md
│── requirements.txt
│── LICENSE
│── CITATION.cff
```

---

# Dataset

The study utilizes publicly available ChEMBL bioactivity data for the MCF7 breast cancer cell line.

Original dataset

ChEMBL Release 36

Processed dataset

DS_BreastCancer.csv

---

# Data Curation

The dataset was curated using an automated RDKit-based pipeline.

Cleaning steps

- Removed missing values
- Filtered supported IC50 units (nM and ug/mL)
- Converted ug/mL into nM
- Removed non-exact activity values (> and <)
- Removed invalid IC50 values
- Standardized SMILES
- Removed salts
- Removed duplicate molecules
- Calculated pIC50
- Generated activity labels

---

## Data Curation Statistics

| Step | Compounds |
|-------|----------:|
| Initial Dataset | 46,291 |
| After Missing Value Removal | 44,927 |
| After Unit Filtering | 44,908 |
| After Relation Filtering | 34,792 |
| After IC50 Validation | 34,773 |
| After SMILES Standardization | 34,773 |
| After Duplicate Removal | **34,707** |

Final dataset

- Total compounds: **34,707**
- Active: **7,255 (20.9%)**
- Inactive: **27,452 (79.1%)**
- pIC50 range: **1.01–13.70**

---

# Molecular Features

The machine learning models were trained using:

## Molecular fingerprints

- Morgan Fingerprints
- Radius = 2
- 2048 bits

## Molecular descriptors

- Molecular Weight
- LogP
- HBA
- HBD
- PSA
- Rotatable Bonds
- Rule of 3
- Rule of 5 Violations
- CX LogP
- CX LogD
- Molecular Species
- QED
- Natural Product Likeness

---

# Machine Learning Models

Five supervised learning algorithms were evaluated.

- Random Forest
- Extra Trees
- LightGBM
- XGBoost
- CatBoost

Model selection was performed using

- ROC-AUC
- PR-AUC
- MCC
- Balanced Accuracy
- F1 Score

---

# Validation Strategy

The pipeline employs a rigorous scaffold-aware validation strategy.

- Murcko scaffold splitting
- 80% Train/Validation
- 20% External Test
- Five-fold cross-validation
- Applicability domain analysis
- Y-randomization

This prevents scaffold leakage and provides a realistic estimate of model generalization.

---

# Structure-Based Validation

The highest-ranked compounds were further validated using

- Molecular Docking
- Binding Pose Analysis
- Molecular Dynamics Simulation
- Stability Analysis
- Binding Free Energy

---

# Requirements

Python ≥ 3.10

Main packages

```
pandas
numpy
scikit-learn
rdkit
matplotlib
xgboost
lightgbm
catboost
```

Install

```bash
pip install -r requirements.txt
```

---

# Running the Pipeline

Data curation

```bash
python scripts/data_curation.py
```

Machine learning pipeline

```bash
python scripts/scaffold_ml_pipeline.py
```

---

# Output

The pipeline automatically generates

- Curated dataset
- Scaffold split
- Model metrics
- ROC curves
- Precision-Recall curves
- Calibration plots
- Descriptor importance
- Applicability domain
- Ranked compounds
- Prediction tables

---

# Data Availability

The datasets used in this study are publicly available.

**ChEMBL Release 36**

https://www.ebi.ac.uk/chembl/

FTP

https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/

Processed MCF7 dataset

https://www.kaggle.com/datasets/sabetm/ds-breastcancer-mcf7/data

No proprietary or private datasets were used.

---

# Citation

If you use this repository, please cite:

Aziz M., et al.

**Scaffold-Aware Machine Learning and Structure-Based Validation Prioritize Dual ERα/PI3Kα Inhibitor Candidates for ER-Positive Breast Cancer.**

(Under Review)

---

# License

This repository is released under the MIT License.

---

## Author

**Moawaz Aziz**

University of Electronic Science and Technology of China (UESTC)

Email: your_email@outlook.com

GitHub: https://github.com/YourUsername

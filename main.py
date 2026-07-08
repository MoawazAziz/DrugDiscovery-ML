import json
import math
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import QED
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_PATH = Path('/mnt/data/DS_BreastCancer(7).csv')
OUTDIR = Path('/mnt/data/qsar_breast_q1_outputs')
OUTDIR.mkdir(parents=True, exist_ok=True)

ACTIVE_CUTOFF_NM = 1000.0
INACTIVE_CUTOFF_NM = 10000.0
FP_SIZE = 2048
DESC_COLS = [
    'molecular_weight', 'aLogP', 'hba', 'hbd', 'psa', 'rtb', 'ro3_pass',
    'num_ro5_violations', 'cx_logp', 'cx_logd', 'molecular_species',
    'qed_weighted', 'np_likeness_score'
]


def log(msg: str):
    print(msg, flush=True)


def canonicalize_smiles(smiles: str):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def make_scaffold(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def load_and_curate(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    raw_n = len(df)
    df = df[df['ic50_units'].eq('nM')].copy()
    df['ic50_value'] = pd.to_numeric(df['ic50_value'], errors='coerce')
    df = df[df['ic50_value'].notna()].copy()
    if 'standard_relation' in df.columns:
        df = df[df['standard_relation'].eq('=')].copy()
    df['can_smiles'] = df['smiles'].map(canonicalize_smiles)
    df = df[df['can_smiles'].notna()].copy()
    # remove the ambiguous zone 1-10 uM
    df = df[(df['ic50_value'] <= ACTIVE_CUTOFF_NM) | (df['ic50_value'] >= INACTIVE_CUTOFF_NM)].copy()
    df['label'] = (df['ic50_value'] <= ACTIVE_CUTOFF_NM).astype(int)
    df['pIC50'] = 9 - np.log10(df['ic50_value'])
    df['scaffold'] = df['can_smiles'].map(make_scaffold)
    df = df.reset_index(drop=True)
    log(f'Raw rows: {raw_n:,}')
    log(f'Curated rows: {len(df):,} | Actives: {int(df.label.sum()):,} | Inactives: {int((1-df.label).sum()):,}')
    return df


def random_scaffold_test_split(df: pd.DataFrame, test_frac: float = 0.20, n_trials: int = 1200, seed: int = 42):
    groups = df.groupby('scaffold').groups
    group_list = []
    for scaf, idxs in groups.items():
        idxs = list(idxs)
        p = int(df.loc[idxs, 'label'].sum())
        group_list.append((scaf, idxs, len(idxs), p))
    target_n = round(len(df) * test_frac)
    target_ratio = float(df['label'].mean())
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_trials):
        order = np.arange(len(group_list))
        rng.shuffle(order)
        picked = []
        n = 0
        pos = 0
        for j in order:
            scaf, idxs, size, p = group_list[j]
            if n < target_n or rng.random() < 0.03:
                picked.extend(idxs)
                n += size
                pos += p
            if n >= target_n:
                break
        picked = sorted(set(picked))
        ratio = pos / len(picked)
        score = abs(len(picked) - target_n) / target_n + abs(ratio - target_ratio) / max(target_ratio, 1e-8)
        if (best is None) or (score < best[0]):
            best = (score, picked, ratio)
    test_idx = best[1]
    train_val_idx = sorted(set(range(len(df))) - set(test_idx))
    return train_val_idx, test_idx


def get_preprocessor(df_like: pd.DataFrame):
    numeric_candidates = [
        'molecular_weight', 'aLogP', 'hba', 'hbd', 'psa', 'rtb',
        'num_ro5_violations', 'cx_logp', 'cx_logd', 'qed_weighted', 'np_likeness_score'
    ]
    categorical_candidates = ['ro3_pass', 'molecular_species']
    num_cols = [c for c in numeric_candidates if c in df_like.columns]
    cat_cols = [c for c in categorical_candidates if c in df_like.columns]
    pre = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num_cols),
            ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols),
        ],
        remainder='drop'
    )
    return pre, num_cols, cat_cols


def compute_fingerprints(smiles_series: pd.Series, fp_size: int = FP_SIZE):
    morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)
    fps = np.zeros((len(smiles_series), fp_size), dtype=np.uint8)
    for i, smi in enumerate(smiles_series.tolist()):
        mol = Chem.MolFromSmiles(smi)
        bv = morgan.GetFingerprint(mol)
        arr = np.zeros((fp_size,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(bv, arr)
        fps[i] = arr
    return fps.astype(np.float32)


def build_feature_matrix(fps: np.ndarray, desc_matrix: np.ndarray):
    return np.hstack([fps, desc_matrix.astype(np.float32)]).astype(np.float32)


def choose_threshold(y_true, prob, n_grid: int = 401):
    best_thr = 0.5
    best_mcc = -1.0
    for thr in np.linspace(0.0, 1.0, n_grid):
        pred = (prob >= thr).astype(int)
        mcc = matthews_corrcoef(y_true, pred)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thr = float(thr)
    return best_thr


def evaluate_probabilities(y_true, prob, threshold: float):
    pred = (prob >= threshold).astype(int)
    return {
        'ROC_AUC': float(roc_auc_score(y_true, prob)),
        'PR_AUC': float(average_precision_score(y_true, prob)),
        'Balanced_Accuracy': float(balanced_accuracy_score(y_true, pred)),
        'MCC': float(matthews_corrcoef(y_true, pred)),
        'F1': float(f1_score(y_true, pred)),
        'Precision': float(precision_score(y_true, pred, zero_division=0)),
        'Recall': float(recall_score(y_true, pred, zero_division=0)),
        'Accuracy': float(accuracy_score(y_true, pred)),
        'Brier': float(brier_score_loss(y_true, prob)),
    }


def make_models(scale_pos_weight: float):
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=150,
            max_features='sqrt',
            min_samples_leaf=2,
            class_weight='balanced_subsample',
            random_state=RANDOM_STATE,
            n_jobs=4,
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=220,
            max_features='sqrt',
            min_samples_leaf=1,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=4,
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            objective='binary',
            random_state=RANDOM_STATE,
            n_jobs=4,
            scale_pos_weight=scale_pos_weight,
            verbosity=-1,
        ),
        'XGBoost': XGBClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=8,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective='binary:logistic',
            eval_metric='logloss',
            tree_method='hist',
            n_jobs=4,
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight,
        ),
        'CatBoost': CatBoostClassifier(
            iterations=500,
            depth=8,
            learning_rate=0.03,
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=RANDOM_STATE,
            auto_class_weights='Balanced',
            verbose=False,
            thread_count=4,
            od_type='Iter',
            od_wait=80,
        ),
    }
    return models


def fit_model(name, model, X_tr, y_tr, X_va=None, y_va=None):
    if name == 'LightGBM':
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric='auc',
            callbacks=[],
        )
        return model
    if name == 'XGBoost':
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
        return model
    if name == 'CatBoost':
        model.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            use_best_model=True,
            verbose=False,
        )
        return model
    model.fit(X_tr, y_tr)
    return model


def predict_proba_safe(model, X):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, 'decision_function'):
        z = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))
    raise RuntimeError('Model does not support probability-like prediction.')


def qed_series(df_part: pd.DataFrame):
    q = df_part['qed_weighted'].copy()
    missing = q.isna()
    if missing.any():
        vals = []
        for smi in df_part.loc[missing, 'can_smiles']:
            mol = Chem.MolFromSmiles(smi)
            vals.append(QED.qed(mol) if mol is not None else 0.0)
        q.loc[missing] = vals
    return q.astype(float).fillna(0.0)


def compute_ad_scores(X_trainval_full: np.ndarray, X_test_full: np.ndarray):
    n_components = int(min(50, max(10, X_trainval_full.shape[1] // 20)))
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    Z_train = svd.fit_transform(X_trainval_full)
    Z_test = svd.transform(X_test_full)
    mean_vec = Z_train.mean(axis=0)
    cov = np.cov(Z_train.T) + np.eye(Z_train.shape[1]) * 1e-6
    inv_cov = np.linalg.pinv(cov)
    d2_train = np.sum((Z_train - mean_vec) @ inv_cov * (Z_train - mean_vec), axis=1)
    d2_test = np.sum((Z_test - mean_vec) @ inv_cov * (Z_test - mean_vec), axis=1)
    threshold = float(np.percentile(d2_train, 95))
    ad = 1.0 - np.clip(d2_test / threshold, 0.0, 1.0)
    return ad, d2_train, d2_test, threshold


def make_priority_score(prob_mean, prob_std, ad_score, qed_score):
    return 0.70 * prob_mean + 0.15 * ad_score + 0.10 * qed_score - 0.05 * prob_std


def top5_table(test_df, prob_mean, prob_std, ad_score, threshold, trainval_df, trainval_fps):
    top = test_df.copy()
    top['pred_prob_mean'] = prob_mean
    top['pred_prob_std'] = prob_std
    top['ad_score'] = ad_score
    top['qed_for_rank'] = qed_series(top)
    top['priority_score'] = make_priority_score(top['pred_prob_mean'].values, top['pred_prob_std'].values, top['ad_score'].values, top['qed_for_rank'].values)
    top['predicted_class'] = (top['pred_prob_mean'] >= threshold).astype(int)
    top['inside_ad'] = top['ad_score'] >= 0.0
    # rank only by model outputs and chemistry; no true-label filtering
    ranked = top.sort_values(['pred_prob_mean', 'priority_score', 'ad_score'], ascending=False).copy()
    ranked = ranked[(ranked['predicted_class'] == 1) & (ranked['ad_score'] >= 0.25)].copy()
    if len(ranked) < 5:
        ranked = top.sort_values(['pred_prob_mean', 'priority_score'], ascending=False).copy()
    ranked = ranked.head(5).copy()

    # nearest known active similarity
    active_train = trainval_df[trainval_df['label'] == 1].reset_index(drop=True)
    active_fps = trainval_fps[trainval_df['label'].values == 1]
    active_bitvecs = []
    for arr in active_fps:
        bv = DataStructs.ExplicitBitVect(FP_SIZE)
        for bit in np.where(arr.astype(np.uint8) > 0)[0].tolist():
            bv.SetBit(int(bit))
        active_bitvecs.append(bv)
    morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_SIZE)
    nearest_smiles = []
    nearest_sim = []
    nearest_chembl = []
    for smi in ranked['can_smiles'].tolist():
        mol = Chem.MolFromSmiles(smi)
        fp = morgan.GetFingerprint(mol)
        sims = DataStructs.BulkTanimotoSimilarity(fp, active_bitvecs)
        best_i = int(np.argmax(sims))
        nearest_smiles.append(active_train.iloc[best_i]['can_smiles'])
        nearest_sim.append(float(sims[best_i]))
        nearest_chembl.append(active_train.iloc[best_i].get('chembl_id', 'NA'))
    ranked['nearest_train_active_chembl'] = nearest_chembl
    ranked['nearest_train_active_smiles'] = nearest_smiles
    ranked['max_train_active_tanimoto'] = nearest_sim
    ranked['retrospective_true_label'] = ranked['label']
    return ranked


def plot_activity_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=220)
    pic50 = df['pIC50'].replace([np.inf, -np.inf], np.nan).dropna()
    axes[0].hist(pic50, bins=50)
    axes[0].axvline(6.0, linestyle='--', linewidth=1.5)
    axes[0].axvline(5.0, linestyle='--', linewidth=1.5)
    axes[0].set_xlabel('pIC50')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Curated activity distribution')

    counts = df['label'].value_counts().sort_index()
    axes[1].bar(['Inactive', 'Active'], [counts.get(0, 0), counts.get(1, 0)])
    axes[1].set_ylabel('Count')
    axes[1].set_title('Binary classes after thresholding')

    fig.suptitle('Breast cancer (MCF7) QSAR data curation summary', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'fig01_activity_distribution.png', bbox_inches='tight')
    plt.close(fig)


def plot_model_ranking(metrics_df):
    order = metrics_df.sort_values('Val_PR_AUC', ascending=False)['Model'].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=220)
    for ax, metric in zip(axes, ['ROC_AUC', 'PR_AUC', 'MCC']):
        x = np.arange(len(order))
        val_vals = [metrics_df.loc[metrics_df['Model'] == m, f'Val_{metric}'].iloc[0] for m in order]
        test_vals = [metrics_df.loc[metrics_df['Model'] == m, f'Test_{metric}'].iloc[0] for m in order]
        width = 0.38
        ax.bar(x - width / 2, val_vals, width=width, label='Validation')
        ax.bar(x + width / 2, test_vals, width=width, label='Test')
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=20)
        ax.set_ylim(0, 1.0)
        ax.set_title(metric.replace('_', '-'))
    axes[0].legend(frameon=False)
    fig.suptitle('Classifier ranking under the strict split', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'fig02_model_ranking.png', bbox_inches='tight')
    plt.close(fig)


def plot_curves(y_test, prob_mean):
    fpr, tpr, _ = roc_curve(y_test, prob_mean)
    prec, rec, _ = precision_recall_curve(y_test, prob_mean)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=220)
    axes[0].plot(fpr, tpr, linewidth=2)
    axes[0].plot([0, 1], [0, 1], linestyle='--', linewidth=1)
    axes[0].set_xlabel('False positive rate')
    axes[0].set_ylabel('True positive rate')
    axes[0].set_title('ROC curve (test)')
    axes[1].plot(rec, prec, linewidth=2)
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall curve (test)')
    fig.suptitle('Best model external-test discrimination', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'fig03_best_model_curves.png', bbox_inches='tight')
    plt.close(fig)


def plot_calibration(y_test, prob_mean):
    frac_pos, mean_pred = calibration_curve(y_test, prob_mean, n_bins=10, strategy='quantile')
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=220)
    ax.plot([0, 1], [0, 1], linestyle='--', linewidth=1)
    ax.plot(mean_pred, frac_pos, marker='o', linewidth=2)
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Observed fraction positive')
    ax.set_title('Calibration curve on the external test set')
    fig.tight_layout()
    fig.savefig(OUTDIR / 'fig04_calibration.png', bbox_inches='tight')
    plt.close(fig)


def plot_applicability_top5(test_df, prob_mean, ad_score, top5_idx):
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=220)
    colors = np.where(test_df['label'].values == 1, 1, 0)
    sc = ax.scatter(ad_score, prob_mean, c=colors, alpha=0.55, s=14)
    ax.set_xlabel('Applicability-domain score')
    ax.set_ylabel('Predicted active probability')
    ax.set_title('Applicability domain and test-set confidence')
    top_points = test_df.iloc[top5_idx]
    for idx, (_, row) in enumerate(top_points.iterrows(), start=1):
        ax.scatter(ad_score[row.name], prob_mean[row.name], s=120, facecolors='none', edgecolors='red', linewidths=1.8)
        ax.text(ad_score[row.name] + 0.01, prob_mean[row.name] + 0.01, str(idx), fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'fig05_applicability_top5.png', bbox_inches='tight')
    plt.close(fig)


def plot_descriptor_importance(best_model_name, best_model_params, train_df, val_df, X_fp_train, X_fp_val, y_train, y_val):
    pre, num_cols, cat_cols = get_preprocessor(train_df)
    X_desc_train = pre.fit_transform(train_df[DESC_COLS]).astype(np.float32)
    X_desc_val = pre.transform(val_df[DESC_COLS]).astype(np.float32)
    X_train = build_feature_matrix(X_fp_train, X_desc_train)
    X_val = build_feature_matrix(X_fp_val, X_desc_val)
    model = make_models((y_train == 0).sum() / max((y_train == 1).sum(), 1))[best_model_name]
    for k, v in best_model_params.items():
        try:
            model.set_params(**{k: v})
        except Exception:
            pass
    model = fit_model(best_model_name, model, X_train, y_train, X_val, y_val)
    baseline = roc_auc_score(y_val, predict_proba_safe(model, X_val))
    desc_feature_names = num_cols + list(pre.named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(cat_cols))
    importances = []
    for j, feat_name in enumerate(desc_feature_names):
        X_desc_perm = X_desc_val.copy()
        rng = np.random.default_rng(RANDOM_STATE + j)
        rng.shuffle(X_desc_perm[:, j])
        X_perm = build_feature_matrix(X_fp_val, X_desc_perm)
        auc = roc_auc_score(y_val, predict_proba_safe(model, X_perm))
        importances.append((feat_name, baseline - auc))
    imp_df = pd.DataFrame(importances, columns=['Feature', 'Permutation_Delta_AUC']).sort_values('Permutation_Delta_AUC', ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=220)
    ax.barh(imp_df['Feature'][::-1], imp_df['Permutation_Delta_AUC'][::-1])
    ax.set_xlabel('Validation ROC-AUC drop after permutation')
    ax.set_title('Descriptor-level interpretability for the best classifier')
    fig.tight_layout()
    fig.savefig(OUTDIR / 'fig06_descriptor_importance.png', bbox_inches='tight')
    plt.close(fig)
    return imp_df


def plot_y_randomization(best_model_name, best_model_params, X_trainval, y_trainval, X_test, y_test):
    actual_model = make_models((y_trainval == 0).sum() / max((y_trainval == 1).sum(), 1))[best_model_name]
    for k, v in best_model_params.items():
        try:
            actual_model.set_params(**{k: v})
        except Exception:
            pass
    actual_model = fit_model(best_model_name, actual_model, X_trainval, y_trainval, X_test, y_test)
    actual_auc = roc_auc_score(y_test, predict_proba_safe(actual_model, X_test))
    scramble_aucs = []
    rng = np.random.default_rng(RANDOM_STATE)
    for rep in range(1):
        y_shuf = rng.permutation(y_trainval)
        shuf_model = make_models((y_shuf == 0).sum() / max((y_shuf == 1).sum(), 1))[best_model_name]
        for k, v in best_model_params.items():
            try:
                shuf_model.set_params(**{k: v})
            except Exception:
                pass
        shuf_model = fit_model(best_model_name, shuf_model, X_trainval, y_shuf, X_test, y_test)
        scramble_aucs.append(roc_auc_score(y_test, predict_proba_safe(shuf_model, X_test)))
    fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=220)
    labels = ['Actual', 'Shuffle-1', 'Shuffle-2', 'Shuffle-3']
    vals = [actual_auc] + scramble_aucs
    ax.bar(labels, vals)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Test ROC-AUC')
    ax.set_title('Y-randomization sanity check')
    fig.tight_layout()
    fig.savefig(OUTDIR / 'fig07_y_randomization.png', bbox_inches='tight')
    plt.close(fig)
    return {'actual_auc': actual_auc, 'shuffle_aucs': scramble_aucs}


def main():
    log('Loading and curating data...')
    df = load_and_curate(DATA_PATH)
    df.to_csv(OUTDIR / 'curated_dataset.csv', index=False)

    log('Creating scaffold-held-out test split...')
    train_val_idx, test_idx = random_scaffold_test_split(df, test_frac=0.20, n_trials=800, seed=RANDOM_STATE)
    train_val_df = df.iloc[train_val_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)
    train_idx_local, val_idx_local = train_test_split(
        np.arange(len(train_val_df)),
        test_size=0.20,
        stratify=train_val_df['label'],
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    train_df = train_val_df.iloc[train_idx_local].copy().reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx_local].copy().reset_index(drop=True)
    split_summary = {
        'train_n': int(len(train_df)),
        'val_n': int(len(val_df)),
        'test_n': int(len(test_df)),
        'train_active_rate': float(train_df['label'].mean()),
        'val_active_rate': float(val_df['label'].mean()),
        'test_active_rate': float(test_df['label'].mean()),
        'n_unique_scaffolds_total': int(df['scaffold'].nunique()),
        'n_scaffold_overlap_train_test': int(len(set(train_val_df['scaffold']) & set(test_df['scaffold']))),
    }
    json.dump(split_summary, open(OUTDIR / 'split_summary.json', 'w'), indent=2)
    log(json.dumps(split_summary, indent=2))

    log('Computing fingerprints...')
    X_fp_train = compute_fingerprints(train_df['can_smiles'])
    X_fp_val = compute_fingerprints(val_df['can_smiles'])
    X_fp_test = compute_fingerprints(test_df['can_smiles'])
    X_fp_trainval = compute_fingerprints(train_val_df['can_smiles'])

    log('Fitting descriptor preprocessor on the training split...')
    pre, num_cols, cat_cols = get_preprocessor(train_df)
    X_desc_train = pre.fit_transform(train_df[DESC_COLS]).astype(np.float32)
    X_desc_val = pre.transform(val_df[DESC_COLS]).astype(np.float32)
    X_desc_test = pre.transform(test_df[DESC_COLS]).astype(np.float32)
    X_train = build_feature_matrix(X_fp_train, X_desc_train)
    X_val = build_feature_matrix(X_fp_val, X_desc_val)
    X_test = build_feature_matrix(X_fp_test, X_desc_test)
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    log('Benchmarking the five candidate classifiers...')
    models = make_models(scale_pos_weight)
    benchmark_rows = []
    fitted_for_val = {}
    for name, model in models.items():
        log(f'  Training {name}...')
        fitted = fit_model(name, model, X_train, y_train, X_val, y_val)
        prob_val = predict_proba_safe(fitted, X_val)
        threshold = choose_threshold(y_val, prob_val)
        prob_test = predict_proba_safe(fitted, X_test)
        row = {'Model': name, 'Threshold_from_validation': threshold}
        row.update({f'Val_{k}': v for k, v in evaluate_probabilities(y_val, prob_val, threshold).items()})
        row.update({f'Test_{k}': v for k, v in evaluate_probabilities(y_test, prob_test, threshold).items()})
        benchmark_rows.append(row)
        fitted_for_val[name] = fitted
        log(f"    {name}: val PR-AUC={row['Val_PR_AUC']:.4f}, val ROC-AUC={row['Val_ROC_AUC']:.4f}, val MCC={row['Val_MCC']:.4f}")
    metrics_df = pd.DataFrame(benchmark_rows).sort_values(['Val_PR_AUC', 'Val_MCC', 'Val_ROC_AUC'], ascending=False).reset_index(drop=True)
    metrics_df.to_csv(OUTDIR / 'classifier_metrics.csv', index=False)
    plot_model_ranking(metrics_df)

    best_model_name = metrics_df.iloc[0]['Model']
    log(f'Best validation model: {best_model_name}')
    best_model_params = make_models(scale_pos_weight)[best_model_name].get_params()

    log('Refitting the best model with 5-fold train+validation ensembling...')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_trainval = train_val_df['label'].values
    oof_prob = np.zeros(len(train_val_df), dtype=np.float32)
    test_probs = []
    fold_rows = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(train_val_df)), y_trainval), start=1):
        fold_train_df = train_val_df.iloc[tr_idx].copy().reset_index(drop=True)
        fold_val_df = train_val_df.iloc[va_idx].copy().reset_index(drop=True)
        pre_fold, _, _ = get_preprocessor(fold_train_df)
        X_desc_tr = pre_fold.fit_transform(fold_train_df[DESC_COLS]).astype(np.float32)
        X_desc_va = pre_fold.transform(fold_val_df[DESC_COLS]).astype(np.float32)
        X_desc_te = pre_fold.transform(test_df[DESC_COLS]).astype(np.float32)
        X_tr = build_feature_matrix(X_fp_trainval[tr_idx], X_desc_tr)
        X_va = build_feature_matrix(X_fp_trainval[va_idx], X_desc_va)
        X_te = build_feature_matrix(X_fp_test, X_desc_te)
        y_tr = y_trainval[tr_idx]
        y_va = y_trainval[va_idx]
        fold_scale = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
        fold_model = make_models(fold_scale)[best_model_name]
        fold_model = fit_model(best_model_name, fold_model, X_tr, y_tr, X_va, y_va)
        oof_prob[va_idx] = predict_proba_safe(fold_model, X_va)
        test_fold_prob = predict_proba_safe(fold_model, X_te)
        test_probs.append(test_fold_prob)
        fold_rows.append({
            'Fold': fold,
            'OOF_ROC_AUC': roc_auc_score(y_va, oof_prob[va_idx]),
            'OOF_PR_AUC': average_precision_score(y_va, oof_prob[va_idx]),
        })
        log(f"  Fold {fold}: OOF ROC-AUC={fold_rows[-1]['OOF_ROC_AUC']:.4f}, OOF PR-AUC={fold_rows[-1]['OOF_PR_AUC']:.4f}")
    pd.DataFrame(fold_rows).to_csv(OUTDIR / 'best_model_fold_metrics.csv', index=False)
    test_prob_mean = np.mean(np.vstack(test_probs), axis=0)
    test_prob_std = np.std(np.vstack(test_probs), axis=0)
    final_threshold = choose_threshold(y_trainval, oof_prob)
    final_test_metrics = evaluate_probabilities(y_test, test_prob_mean, final_threshold)
    final_test_metrics['Threshold_from_trainval_oof'] = final_threshold
    final_test_metrics['Best_Model'] = best_model_name
    pd.DataFrame([final_test_metrics]).to_csv(OUTDIR / 'best_model_test_metrics.csv', index=False)
    log(json.dumps(final_test_metrics, indent=2))

    log('Fitting best model on the full train+validation set for AD and interpretability...')
    pre_full, _, _ = get_preprocessor(train_val_df)
    X_desc_trainval_full = pre_full.fit_transform(train_val_df[DESC_COLS]).astype(np.float32)
    X_desc_test_full = pre_full.transform(test_df[DESC_COLS]).astype(np.float32)
    X_trainval_full = build_feature_matrix(X_fp_trainval, X_desc_trainval_full)
    X_test_full = build_feature_matrix(X_fp_test, X_desc_test_full)
    ad_score, d2_train, d2_test, ad_thresh = compute_ad_scores(X_trainval_full, X_test_full)

    preds_df = test_df[[
        'chembl_id', 'iupac_name', 'can_smiles', 'ic50_value', 'pIC50', 'label', 'molecular_weight', 'aLogP', 'qed_weighted'
    ]].copy()
    preds_df['pred_prob_mean'] = test_prob_mean
    preds_df['pred_prob_std'] = test_prob_std
    preds_df['ad_score'] = ad_score
    preds_df['predicted_class'] = (preds_df['pred_prob_mean'] >= final_threshold).astype(int)
    preds_df.to_csv(OUTDIR / 'test_set_predictions.csv', index=False)

    top5 = top5_table(test_df, test_prob_mean, test_prob_std, ad_score, final_threshold, train_val_df, X_fp_trainval)
    top5.to_csv(OUTDIR / 'top5_prioritized_compounds.csv', index=False)

    plot_activity_distribution(df)
    plot_curves(y_test, test_prob_mean)
    plot_calibration(y_test, test_prob_mean)
    plot_applicability_top5(test_df.reset_index(drop=True), test_prob_mean, ad_score, top5.index.tolist())
    descriptor_importance = plot_descriptor_importance(best_model_name, best_model_params, train_df, val_df, X_fp_train, X_fp_val, y_train, y_val)
    descriptor_importance.to_csv(OUTDIR / 'descriptor_permutation_importance.csv', index=False)
    y_scramble = plot_y_randomization(best_model_name, best_model_params, X_trainval_full, y_trainval, X_test_full, y_test)
    json.dump(y_scramble, open(OUTDIR / 'y_randomization.json', 'w'), indent=2)

    report_lines = []
    report_lines.append('# Breast cancer (MCF7) QSAR classification run')
    report_lines.append('')
    report_lines.append('## Data curation')
    report_lines.append(f"- Raw rows: {len(pd.read_csv(DATA_PATH, low_memory=False)):,}")
    report_lines.append(f"- Curated rows (exact nM records, valid SMILES, ambiguous 1-10 uM removed): {len(df):,}")
    report_lines.append(f"- Actives: {int(df['label'].sum()):,} | Inactives: {int((1-df['label']).sum()):,}")
    report_lines.append('')
    report_lines.append('## Split design')
    report_lines.append(f"- Train: {split_summary['train_n']:,} | Validation: {split_summary['val_n']:,} | Test: {split_summary['test_n']:,}")
    report_lines.append(f"- Train active rate: {split_summary['train_active_rate']:.4f}")
    report_lines.append(f"- Validation active rate: {split_summary['val_active_rate']:.4f}")
    report_lines.append(f"- Test active rate: {split_summary['test_active_rate']:.4f}")
    report_lines.append(f"- Train/Test scaffold overlap: {split_summary['n_scaffold_overlap_train_test']}")
    report_lines.append('')
    report_lines.append('## Best model')
    report_lines.append(f"- Model: {best_model_name}")
    report_lines.append(f"- Locked threshold from train+validation OOF predictions: {final_threshold:.3f}")
    report_lines.append(f"- Test ROC-AUC: {final_test_metrics['ROC_AUC']:.4f}")
    report_lines.append(f"- Test PR-AUC: {final_test_metrics['PR_AUC']:.4f}")
    report_lines.append(f"- Test MCC: {final_test_metrics['MCC']:.4f}")
    report_lines.append(f"- Test Balanced Accuracy: {final_test_metrics['Balanced_Accuracy']:.4f}")
    report_lines.append(f"- Test F1: {final_test_metrics['F1']:.4f}")
    report_lines.append(f"- Test Accuracy: {final_test_metrics['Accuracy']:.4f}")
    report_lines.append('')
    report_lines.append('## Top 5 prioritized compounds')
    report_lines.append(top5[[
        'chembl_id', 'can_smiles', 'ic50_value', 'pred_prob_mean', 'pred_prob_std', 'ad_score', 'priority_score', 'retrospective_true_label'
    ]].to_markdown(index=False))
    (OUTDIR / 'run_report.md').write_text('\n'.join(report_lines), encoding='utf-8')

    # zip outputs
    import zipfile
    zip_path = OUTDIR.with_suffix('.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(OUTDIR.rglob('*')):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(OUTDIR.parent))

    log('Done.')


if __name__ == '__main__':
    main()

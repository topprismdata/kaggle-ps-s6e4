"""Round 02: Target Encoding + Frequency Encoding + Multi-model Ensemble

Based on R01 (LB=0.96499). Add:
1. K-Fold Target Encoding for all categorical features
2. Frequency Encoding for categorical features
3. LightGBM + XGBoost + CatBoost ensemble
4. Threshold optimization for balanced accuracy
"""
import warnings
warnings.filterwarnings("ignore")
import time
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from pathlib import Path
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (SUBMISSIONS, TARGET_COL, ID_COL, CLASSES,
                        CATEGORICAL_COLS, NUMERICAL_COLS, ModelConfig)

def log(msg=""):
    print(msg, flush=True)

start = time.time()
log("=" * 60)
log("Round 02: Target Encoding + Multi-model Ensemble")
log("=" * 60)

# ============================================================
# STEP 1: Load data
# ============================================================
log("\nSTEP 1: Load data")
from src.data.loader import load_data
train_df, test_df = load_data()

# ============================================================
# STEP 2: Feature Engineering
# ============================================================
log("\nSTEP 2: Feature Engineering")

# Encode target
target_le = LabelEncoder()
target_le.fit(CLASSES)
y_all = target_le.transform(train_df[TARGET_COL])
n_classes = len(CLASSES)

# Label encoding (baseline)
train = train_df.copy()
test = test_df.copy()
label_encoders = {}
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0)
    le.fit(combined)
    train[f"{col}_le"] = le.transform(train[col])
    test[f"{col}_le"] = le.transform(test[col])
    label_encoders[col] = le

# Frequency encoding
for col in CATEGORICAL_COLS:
    freq = train[col].value_counts(normalize=True)
    train[f"{col}_freq"] = train[col].map(freq)
    test[f"{col}_freq"] = test[col].map(freq).fillna(0)

# K-Fold Target Encoding (prevent leakage)
N_TE_FOLDS = 5
skf_te = StratifiedKFold(n_splits=N_TE_FOLDS, shuffle=True, random_state=42)

for col in CATEGORICAL_COLS:
    # Global target encoding (for test)
    global_mean = train.groupby(col)["_temp_target"].mean() if "_temp_target" in train.columns else None

# Need to set target first
train["_target_enc"] = y_all

te_cols = []
for col in CATEGORICAL_COLS:
    # For each class, compute target encoding
    for cls_idx in range(n_classes):
        cls_name = CLASSES[cls_idx]
        te_col = f"te_{col}_{cls_name}"
        te_cols.append(te_col)

        # OOF target encoding
        train[te_col] = np.nan
        for fold_i, (tr_idx, val_idx) in enumerate(skf_te.split(train, y_all)):
            # Compute mean of (y == cls_idx) grouped by col on training fold
            cls_target = (y_all[tr_idx] == cls_idx).astype(float)
            tr_col = train.iloc[tr_idx][col]
            te_map = pd.Series(cls_target, index=tr_col.values).groupby(level=0).mean()
            val_col_vals = train.iloc[val_idx][col].values
            train.iloc[val_idx, train.columns.get_loc(te_col)] = pd.Series(val_col_vals).map(te_map).values

        # Global target encoding for test
        cls_target_all = (y_all == cls_idx).astype(float)
        te_map_all = pd.Series(cls_target_all, index=train[col].values).groupby(level=0).mean()
        test[te_col] = test[col].map(te_map_all).fillna(cls_target_all.mean())

    # Also add target mean (regression-style TE)
    te_mean_col = f"te_{col}_mean"
    te_cols.append(te_mean_col)
    train[te_mean_col] = np.nan
    for fold_i, (tr_idx, val_idx) in enumerate(skf_te.split(train, y_all)):
        tr_target = y_all[tr_idx].astype(float)
        tr_col = train.iloc[tr_idx][col]
        te_map = pd.Series(tr_target, index=tr_col.values).groupby(level=0).mean()
        val_col_vals = train.iloc[val_idx][col].values
        train.iloc[val_idx, train.columns.get_loc(te_mean_col)] = pd.Series(val_col_vals).map(te_map).values
    te_map_all = pd.Series(y_all.astype(float), index=train[col].values).groupby(level=0).mean()
    test[te_mean_col] = test[col].map(te_map_all).fillna(y_all.mean())

train = train.drop(columns=["_target_enc"])

# Build feature columns
le_cols = [f"{col}_le" for col in CATEGORICAL_COLS]
freq_cols = [f"{col}_freq" for col in CATEGORICAL_COLS]
feat_cols = NUMERICAL_COLS + le_cols + freq_cols + te_cols

# Fill NaN
for col in feat_cols:
    if train[col].isna().any():
        train[col] = train[col].fillna(train[col].median())
    if test[col].isna().any():
        test[col] = test[col].fillna(train[col].median())

log(f"  Features: {len(feat_cols)} ({len(NUMERICAL_COLS)} num + {len(le_cols)} LE + {len(freq_cols)} freq + {len(te_cols)} TE)")

# ============================================================
# STEP 3: Train models with CV
# ============================================================
log("\nSTEP 3: Train LightGBM + XGBoost + CatBoost with 5-fold CV")
X = train[feat_cols].values.astype(np.float32)
X_test_final = test[feat_cols].values.astype(np.float32)

cfg = ModelConfig()
N_FOLDS = cfg.n_folds
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=cfg.random_state)

# --- LightGBM ---
log("\n  --- LightGBM ---")
lgb_params = {
    "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
    "learning_rate": 0.05, "n_estimators": 3000,
    "num_leaves": 64, "min_child_samples": 50,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
    "random_state": 42, "verbose": -1, "n_jobs": -1,
    "class_weight": "balanced",
}

lgb_oof = np.zeros((len(X), 3))
lgb_test = np.zeros((len(X_test_final), 3))
lgb_scores = []

for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X, y_all)):
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X[tr_idx], y_all[tr_idx],
              eval_set=[(X[val_idx], y_all[val_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    lgb_oof[val_idx] = model.predict_proba(X[val_idx])
    lgb_test += model.predict_proba(X_test_final) / N_FOLDS
    pred = model.predict(X[val_idx])
    lgb_scores.append(balanced_accuracy_score(y_all[val_idx], pred))
    log(f"    Fold {fold_i+1}: BA={lgb_scores[-1]:.5f}, iter={model.best_iteration_}")

log(f"  LGB Mean BA: {np.mean(lgb_scores):.5f}")

# --- XGBoost ---
log("\n  --- XGBoost ---")
from sklearn.utils.class_weight import compute_sample_weight
xgb_params = {
    "objective": "multi:softprob", "num_class": 3,
    "learning_rate": 0.05, "n_estimators": 3000,
    "max_depth": 6, "min_child_weight": 50,
    "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
    "random_state": 42, "verbosity": 0, "n_jobs": -1,
}

xgb_oof = np.zeros((len(X), 3))
xgb_test = np.zeros((len(X_test_final), 3))
xgb_scores = []

# Compute sample weights for XGBoost
sample_weights_all = compute_sample_weight("balanced", y_all)

for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X, y_all)):
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X[tr_idx], y_all[tr_idx],
              sample_weight=sample_weights_all[tr_idx],
              eval_set=[(X[val_idx], y_all[val_idx])],
              verbose=False)
    xgb_oof[val_idx] = model.predict_proba(X[val_idx])
    xgb_test += model.predict_proba(X_test_final) / N_FOLDS
    pred = model.predict(X[val_idx])
    xgb_scores.append(balanced_accuracy_score(y_all[val_idx], pred))
    bi = model.best_iteration if hasattr(model, 'best_iteration') and model.best_iteration else 3000
    log(f"    Fold {fold_i+1}: BA={xgb_scores[-1]:.5f}, iter={bi}")

log(f"  XGB Mean BA: {np.mean(xgb_scores):.5f}")

# ============================================================
# STEP 4: Ensemble and threshold optimization
# ============================================================
log("\nSTEP 4: Ensemble + Threshold Optimization")

# Simple average ensemble
ens_oof = 0.5 * lgb_oof + 0.5 * xgb_oof
ens_test = 0.5 * lgb_test + 0.5 * xgb_test

# Default argmax
ens_pred_default = np.argmax(ens_oof, axis=1)
ba_default = balanced_accuracy_score(y_all, ens_pred_default)
log(f"  Ensemble (50/50) default BA: {ba_default:.5f}")

# Threshold optimization: scale probabilities per class to maximize BA
from scipy.optimize import minimize

def ba_loss(scales, probs, y_true):
    scaled = probs * scales
    pred = np.argmax(scaled, axis=1)
    return -balanced_accuracy_score(y_true, pred)

result = minimize(ba_loss, np.ones(3), args=(ens_oof, y_all), method='Nelder-Mead',
                  options={'maxiter': 1000, 'xatol': 1e-4})
best_scales = result.x
log(f"  Optimized scales: {best_scales}")

ens_oof_scaled = ens_oof * best_scales
ens_pred_opt = np.argmax(ens_oof_scaled, axis=1)
ba_opt = balanced_accuracy_score(y_all, ens_pred_opt)
log(f"  Ensemble optimized BA: {ba_opt:.5f}")

# Apply to test
ens_test_scaled = ens_test * best_scales
test_classes_opt = np.argmax(ens_test_scaled, axis=1)
test_labels_opt = target_le.inverse_transform(test_classes_opt)

# ============================================================
# STEP 5: Save submissions
# ============================================================
log("\nSTEP 5: Save submissions")

def save_sub(preds, name):
    labels = target_le.inverse_transform(preds)
    sub = pd.DataFrame({ID_COL: test[ID_COL].values, TARGET_COL: labels})
    path = SUBMISSIONS / f"submission_{name}.csv"
    sub.to_csv(path, index=False)
    dist = dict(zip(*np.unique(labels, return_counts=True)))
    log(f"  {name}: {dist}")
    return path

save_sub(np.argmax(lgb_test, axis=1), "r02_lgb_only")
save_sub(np.argmax(xgb_test, axis=1), "r02_xgb_only")
save_sub(np.argmax(ens_test, axis=1), "r02_ens_avg")
save_sub(test_classes_opt, "r02_ens_opt")

# ============================================================
# SUMMARY
# ============================================================
log("\n" + "=" * 60)
log("SUMMARY")
log("=" * 60)
log(f"  Features: {len(feat_cols)} (num+LE+freq+TE)")
log(f"  LGB BA: {np.mean(lgb_scores):.5f}")
log(f"  XGB BA: {np.mean(xgb_scores):.5f}")
log(f"  Ensemble default BA: {ba_default:.5f}")
log(f"  Ensemble optimized BA: {ba_opt:.5f}")
log(f"  Total: {time.time() - start:.1f}s")

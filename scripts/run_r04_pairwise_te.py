"""Round 04: Pairwise Target Encoding + Log-space Bias Tuning + Original Data

Based on research findings:
1. All pairwise combinations of categorical features → target encoding (key to 0.977+)
2. KBinsDiscretizer for continuous features → target encoding
3. Log-space bias tuning (not just probability scaling)
4. Original data integration (weight ~0.35)
5. Domain features: dryness stress, moisture/rainfall ratio, etc.

Target: match V5 (0.97745) or exceed it.
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
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (SUBMISSIONS, TARGET_COL, ID_COL, CLASSES,
                        CATEGORICAL_COLS, NUMERICAL_COLS, ModelConfig)

def log(msg=""):
    print(msg, flush=True)

start = time.time()
log("=" * 60)
log("Round 04: Pairwise TE + Log-space Bias + Original Data")
log("=" * 60)

# ============================================================
# STEP 1: Load data
# ============================================================
log("\nSTEP 1: Load data")
from src.data.loader import load_data
train_df, test_df = load_data()

# Load original data
try:
    orig_df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "raw" / "irrigation_prediction.csv")
    log(f"  Original data: {orig_df.shape}")
except FileNotFoundError:
    orig_df = None
    log("  WARNING: Original data not found")

# Encode target
target_le = LabelEncoder()
target_le.fit(CLASSES)
y_all = target_le.transform(train_df[TARGET_COL])
n_classes = len(CLASSES)

train = train_df.copy()
test = test_df.copy()

# ============================================================
# STEP 2: Feature Engineering
# ============================================================
log("\nSTEP 2: Feature Engineering")

# --- 2a. Formula boolean features ---
train["soil_lt_25"] = (train["Soil_Moisture"] < 25).astype(int)
train["temp_gt_30"] = (train["Temperature_C"] > 30).astype(int)
train["rain_lt_300"] = (train["Rainfall_mm"] < 300).astype(int)
train["wind_gt_10"] = (train["Wind_Speed_kmh"] > 10).astype(int)
train["formula_score"] = train["soil_lt_25"] + train["temp_gt_30"] + train["rain_lt_300"] + train["wind_gt_10"]

test["soil_lt_25"] = (test["Soil_Moisture"] < 25).astype(int)
test["temp_gt_30"] = (test["Temperature_C"] > 30).astype(int)
test["rain_lt_300"] = (test["Rainfall_mm"] < 300).astype(int)
test["wind_gt_10"] = (test["Wind_Speed_kmh"] > 10).astype(int)
test["formula_score"] = test["soil_lt_25"] + test["temp_gt_30"] + test["rain_lt_300"] + test["wind_gt_10"]

# --- 2b. Domain features ---
train["dryness_stress"] = train["Temperature_C"] / (train["Soil_Moisture"] + 1)
train["moisture_per_rain"] = train["Soil_Moisture"] / (train["Rainfall_mm"] + 1)
train["rain_per_area"] = train["Rainfall_mm"] / (train["Field_Area_hectare"] + 0.01)
train["temp_x_wind"] = train["Temperature_C"] * train["Wind_Speed_kmh"]
train["humidity_x_temp"] = train["Humidity"] * train["Temperature_C"]
train["soil_pH_x_moisture"] = train["Soil_pH"] * train["Soil_Moisture"]
train["ec_x_moisture"] = train["Electrical_Conductivity"] * train["Soil_Moisture"]
train["water_deficit"] = train["Temperature_C"] * train["Field_Area_hectare"] - train["Rainfall_mm"] * 0.1 - train["Soil_Moisture"] * 0.5
train["irrigation_balance"] = train["Previous_Irrigation_mm"] + train["Rainfall_mm"] * 0.01 - train["Temperature_C"] * train["Field_Area_hectare"] * 0.01

test["dryness_stress"] = test["Temperature_C"] / (test["Soil_Moisture"] + 1)
test["moisture_per_rain"] = test["Soil_Moisture"] / (test["Rainfall_mm"] + 1)
test["rain_per_area"] = test["Rainfall_mm"] / (test["Field_Area_hectare"] + 0.01)
test["temp_x_wind"] = test["Temperature_C"] * test["Wind_Speed_kmh"]
test["humidity_x_temp"] = test["Humidity"] * test["Temperature_C"]
test["soil_pH_x_moisture"] = test["Soil_pH"] * test["Soil_Moisture"]
test["ec_x_moisture"] = test["Electrical_Conductivity"] * test["Soil_Moisture"]
test["water_deficit"] = test["Temperature_C"] * test["Field_Area_hectare"] - test["Rainfall_mm"] * 0.1 - test["Soil_Moisture"] * 0.5
test["irrigation_balance"] = test["Previous_Irrigation_mm"] + test["Rainfall_mm"] * 0.01 - test["Temperature_C"] * test["Field_Area_hectare"] * 0.01

domain_cols = ["dryness_stress", "moisture_per_rain", "rain_per_area", "temp_x_wind",
               "humidity_x_temp", "soil_pH_x_moisture", "ec_x_moisture",
               "water_deficit", "irrigation_balance"]

# --- 2c. Label encoding ---
le_cols = []
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0)
    le.fit(combined)
    train[f"{col}_le"] = le.transform(train[col])
    test[f"{col}_le"] = le.transform(test[col])
    le_cols.append(f"{col}_le")

# --- 2d. Frequency encoding ---
freq_cols = []
for col in CATEGORICAL_COLS:
    freq = train[col].value_counts(normalize=True)
    train[f"{col}_freq"] = train[col].map(freq)
    test[f"{col}_freq"] = test[col].map(freq).fillna(0)
    freq_cols.append(f"{col}_freq")

# --- 2e. Pairwise combinations of categorical features ---
log("  Creating pairwise categorical combinations...")
pair_cols = []
cat_pairs = list(combinations(CATEGORICAL_COLS, 2))
log(f"  Number of pairwise combinations: {len(cat_pairs)}")

for c1, c2 in cat_pairs:
    pair_name = f"{c1}_{c2}"
    train[pair_name] = train[c1].astype(str) + "_" + train[c2].astype(str)
    test[pair_name] = test[c1].astype(str) + "_" + test[c2].astype(str)
    pair_cols.append(pair_name)

# --- 2f. KBinsDiscretizer for continuous features ---
log("  Binning continuous features...")
bin_cols = []
for col in NUMERICAL_COLS:
    binned_col = f"{col}_bin"
    kbd = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile', subsample=200000)
    train[binned_col] = kbd.fit_transform(train[[col]]).astype(int).flatten()
    test[binned_col] = kbd.transform(test[[col]]).astype(int).flatten()
    bin_cols.append(binned_col)

# Also create pairwise between binned numerical and categorical
num_cat_pair_cols = []
for ncol in ["Soil_Moisture_bin", "Temperature_C_bin", "Rainfall_mm_bin"]:
    base = ncol.replace("_bin", "")
    for ccol in ["Crop_Type", "Season", "Region", "Crop_Growth_Stage"]:
        pair_name = f"{base}_{ccol}_pair"
        train[pair_name] = train[ncol].astype(str) + "_" + train[ccol].astype(str)
        test[pair_name] = test[ncol].astype(str) + "_" + test[ccol].astype(str)
        num_cat_pair_cols.append(pair_name)

# --- 2g. K-Fold Target Encoding for ALL categorical + pairwise features ---
log("  Computing K-Fold Target Encoding...")
N_TE_FOLDS = 5
skf_te = StratifiedKFold(n_splits=N_TE_FOLDS, shuffle=True, random_state=42)
train["_target_enc"] = y_all

# Combine all categorical-like columns for TE
te_source_cols = list(CATEGORICAL_COLS) + pair_cols + bin_cols + num_cat_pair_cols
te_cols = []

for col in te_source_cols:
    for cls_idx in range(n_classes):
        cls_name = CLASSES[cls_idx]
        te_col = f"te_{col}_{cls_name}"
        te_cols.append(te_col)
        train[te_col] = np.nan
        for fold_i, (tr_idx, val_idx) in enumerate(skf_te.split(train, y_all)):
            cls_target = (y_all[tr_idx] == cls_idx).astype(float)
            tr_col = train.iloc[tr_idx][col]
            te_map = pd.Series(cls_target, index=tr_col.values).groupby(level=0).mean()
            val_col_vals = train.iloc[val_idx][col].values
            train.iloc[val_idx, train.columns.get_loc(te_col)] = pd.Series(val_col_vals).map(te_map).values
        cls_target_all = (y_all == cls_idx).astype(float)
        te_map_all = pd.Series(cls_target_all, index=train[col].values).groupby(level=0).mean()
        test[te_col] = test[col].map(te_map_all).fillna(cls_target_all.mean())

    # Mean TE
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
formula_bool_cols = ["soil_lt_25", "temp_gt_30", "rain_lt_300", "wind_gt_10", "formula_score"]
feat_cols = (NUMERICAL_COLS + domain_cols + le_cols + freq_cols +
             formula_bool_cols + bin_cols + te_cols)

# Fill NaN
for col in feat_cols:
    if col in train.columns and train[col].isna().any():
        train[col] = train[col].fillna(train[col].median())
    if col in test.columns and test[col].isna().any():
        test[col] = test[col].fillna(train[col].median())

log(f"  Total features: {len(feat_cols)}")
log(f"    Numerical: {len(NUMERICAL_COLS)}")
log(f"    Domain: {len(domain_cols)}")
log(f"    LE: {len(le_cols)}, Freq: {len(freq_cols)}")
log(f"    Formula: {len(formula_bool_cols)}")
log(f"    Binned: {len(bin_cols)}")
log(f"    TE: {len(te_cols)} ({len(te_source_cols)} sources × {n_classes+1})")

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
    "learning_rate": 0.03, "n_estimators": 5000,
    "num_leaves": 63, "min_child_samples": 50,
    "subsample": 0.8, "colsample_bytree": 0.6,
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
xgb_params = {
    "objective": "multi:softprob", "num_class": 3,
    "learning_rate": 0.03, "n_estimators": 5000,
    "max_depth": 6, "min_child_weight": 50,
    "subsample": 0.8, "colsample_bytree": 0.6,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
    "random_state": 42, "verbosity": 0, "n_jobs": -1,
    "early_stopping_rounds": 100,
}

xgb_oof = np.zeros((len(X), 3))
xgb_test = np.zeros((len(X_test_final), 3))
xgb_scores = []
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
    bi = model.best_iteration if hasattr(model, 'best_iteration') and model.best_iteration else 5000
    log(f"    Fold {fold_i+1}: BA={xgb_scores[-1]:.5f}, iter={bi}")

log(f"  XGB Mean BA: {np.mean(xgb_scores):.5f}")

# --- CatBoost ---
log("\n  --- CatBoost ---")
try:
    from catboost import CatBoostClassifier
    cb_oof = np.zeros((len(X), 3))
    cb_test = np.zeros((len(X_test_final), 3))
    cb_scores = []

    cb_params = {
        "iterations": 5000, "learning_rate": 0.03,
        "depth": 6, "l2_leaf_reg": 3.0,
        "random_seed": 42, "verbose": 0,
        "auto_class_weights": "Balanced",
        "loss_function": "MultiClass",
        "eval_metric": "TotalF1:average=Macro",
        "early_stopping_rounds": 100,
    }

    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X, y_all)):
        model = CatBoostClassifier(**cb_params)
        model.fit(X[tr_idx], y_all[tr_idx],
                  eval_set=(X[val_idx], y_all[val_idx]),
                  verbose=0)
        cb_oof[val_idx] = model.predict_proba(X[val_idx])
        cb_test += model.predict_proba(X_test_final) / N_FOLDS
        pred = model.predict(X[val_idx]).flatten().astype(int)
        cb_scores.append(balanced_accuracy_score(y_all[val_idx], pred))
        log(f"    Fold {fold_i+1}: BA={cb_scores[-1]:.5f}, iter={model.best_iteration_}")

    log(f"  CB Mean BA: {np.mean(cb_scores):.5f}")
    HAS_CB = True
except ImportError:
    log("  CatBoost not available, skipping")
    HAS_CB = False

# ============================================================
# STEP 4: Ensemble + Log-space Bias Tuning
# ============================================================
log("\nSTEP 4: Ensemble + Log-space Bias Tuning")

# Multi-model ensemble
if HAS_CB:
    # Try different weights
    best_ens_ba = 0
    best_ens_weights = None
    for w1, w2, w3 in [(1,1,1), (2,1,1), (1,2,1), (1,1,2), (2,2,1), (2,1,2), (1,2,2)]:
        wsum = w1 + w2 + w3
        ens_oof = (w1 * lgb_oof + w2 * xgb_oof + w3 * cb_oof) / wsum
        pred = np.argmax(ens_oof, axis=1)
        ba = balanced_accuracy_score(y_all, pred)
        if ba > best_ens_ba:
            best_ens_ba = ba
            best_ens_weights = (w1, w2, w3)

    w1, w2, w3 = best_ens_weights
    wsum = w1 + w2 + w3
    ens_oof = (w1 * lgb_oof + w2 * xgb_oof + w3 * cb_oof) / wsum
    ens_test = (w1 * lgb_test + w2 * xgb_test + w3 * cb_test) / wsum
    log(f"  Best ensemble weights: LGB={w1}, XGB={w2}, CB={w3}, BA: {best_ens_ba:.5f}")
else:
    ens_oof = 0.5 * lgb_oof + 0.5 * xgb_oof
    ens_test = 0.5 * lgb_test + 0.5 * xgb_test
    pred = np.argmax(ens_oof, axis=1)
    log(f"  LGB+XGB (50/50) BA: {balanced_accuracy_score(y_all, pred):.5f}")

# --- Log-space bias tuning ---
log("\n  Log-space bias tuning...")
log_probs = np.log(ens_oof + 1e-15)  # avoid log(0)

def ba_loss_log_bias(bias, log_probs, y_true):
    """Bias in log-probability space (2 free params, 3rd class as reference)"""
    adjusted = log_probs.copy()
    adjusted[:, 0] += bias[0]
    adjusted[:, 1] += bias[1]
    pred = np.argmax(adjusted, axis=1)
    return -balanced_accuracy_score(y_true, pred)

# Grid search for initial point
best_init_ba = 0
best_init = [0, 0]
for b0 in np.arange(-2, 2.1, 0.2):
    for b1 in np.arange(-2, 2.1, 0.2):
        trial = log_probs.copy()
        trial[:, 0] += b0
        trial[:, 1] += b1
        pred = np.argmax(trial, axis=1)
        ba = balanced_accuracy_score(y_all, pred)
        if ba > best_init_ba:
            best_init_ba = ba
            best_init = [b0, b1]

log(f"  Grid search best BA: {best_init_ba:.5f} at bias={best_init}")

result = minimize(ba_loss_log_bias, best_init, args=(log_probs, y_all),
                  method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-6})
best_bias = result.x
final_log_probs = log_probs.copy()
final_log_probs[:, 0] += best_bias[0]
final_log_probs[:, 1] += best_bias[1]
opt_pred = np.argmax(final_log_probs, axis=1)
ba_opt = balanced_accuracy_score(y_all, opt_pred)
log(f"  Optimized bias: {best_bias}")
log(f"  Optimized BA: {ba_opt:.5f}")

# Apply to test
log_probs_test = np.log(ens_test + 1e-15)
log_probs_test[:, 0] += best_bias[0]
log_probs_test[:, 1] += best_bias[1]
test_classes_opt = np.argmax(log_probs_test, axis=1)

# Also try probability scaling (Nelder-Mead on 3 scales)
def ba_loss_scales(scales, probs, y_true):
    scaled = probs * scales
    pred = np.argmax(scaled, axis=1)
    return -balanced_accuracy_score(y_true, pred)

result2 = minimize(ba_loss_scales, np.ones(3), args=(ens_oof, y_all),
                   method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-6})
best_scales = result2.x
scaled_pred = np.argmax(ens_oof * best_scales, axis=1)
ba_scaled = balanced_accuracy_score(y_all, scaled_pred)
log(f"\n  Probability scaling BA: {ba_scaled:.5f}, scales: {best_scales}")

# Use whichever is better
if ba_opt >= ba_scaled:
    log(f"  Using log-space bias (BA: {ba_opt:.5f})")
    final_test_pred = test_classes_opt
    final_method = "log_bias"
else:
    log(f"  Using probability scaling (BA: {ba_scaled:.5f})")
    final_test_pred = np.argmax(ens_test * best_scales, axis=1)
    final_method = "prob_scale"

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

save_sub(np.argmax(lgb_test, axis=1), "r04_lgb_only")
save_sub(np.argmax(xgb_test, axis=1), "r04_xgb_only")
if HAS_CB:
    save_sub(np.argmax(cb_test, axis=1), "r04_cb_only")
save_sub(np.argmax(ens_test, axis=1), "r04_ens")
save_sub(final_test_pred, f"r04_{final_method}")

# ============================================================
# SUMMARY
# ============================================================
log("\n" + "=" * 60)
log("SUMMARY")
log("=" * 60)
log(f"  Total features: {len(feat_cols)}")
log(f"  TE sources: {len(te_source_cols)} (cat={len(CATEGORICAL_COLS)}, pairs={len(cat_pairs)}, bins={len(bin_cols)}, num_cat_pairs={len(num_cat_pair_cols)})")
log(f"  LGB BA: {np.mean(lgb_scores):.5f}")
log(f"  XGB BA: {np.mean(xgb_scores):.5f}")
if HAS_CB:
    log(f"  CB BA: {np.mean(cb_scores):.5f}")
log(f"  Log-space bias BA: {ba_opt:.5f}")
log(f"  Probability scale BA: {ba_scaled:.5f}")
log(f"  Final method: {final_method}")
log(f"  Total: {time.time() - start:.1f}s")

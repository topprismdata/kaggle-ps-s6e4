"""Round 03: Original Data Formula Features + Enhanced Ensemble

Key discovery: cdeotte found the exact formula for original data.
4 boolean thresholds + 2 categorical features with Logistic Regression = perfect BA.
We use these formula features + richer ensemble to improve on R02.

Features:
1. Formula boolean features (soil_lt_25, temp_gt_30, rain_lt_300, wind_gt_10)
2. One-hot Crop_Growth_Stage + Mulching_Used (formula categorical features)
3. Original numerical + label encoded features
4. Target encoding from R02
5. Frequency encoding from R02

Models: LightGBM + XGBoost + CatBoost ensemble with threshold optimization
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (SUBMISSIONS, TARGET_COL, ID_COL, CLASSES,
                        CATEGORICAL_COLS, NUMERICAL_COLS, ModelConfig)

def log(msg=""):
    print(msg, flush=True)

start = time.time()
log("=" * 60)
log("Round 03: Original Data Formula Features + Enhanced Ensemble")
log("=" * 60)

# ============================================================
# STEP 1: Load data
# ============================================================
log("\nSTEP 1: Load data")
from src.data.loader import load_data
train_df, test_df = load_data()

# Also load original data for logistic regression training
try:
    orig_df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "raw" / "irrigation_prediction.csv")
    log(f"  Original data loaded: {orig_df.shape}")
except FileNotFoundError:
    log("  WARNING: Original data not found, skipping formula features")
    orig_df = None

# ============================================================
# STEP 2: Feature Engineering
# ============================================================
log("\nSTEP 2: Feature Engineering")

# Encode target
target_le = LabelEncoder()
target_le.fit(CLASSES)
y_all = target_le.transform(train_df[TARGET_COL])
n_classes = len(CLASSES)

train = train_df.copy()
test = test_df.copy()

# --- Formula boolean features ---
train["soil_lt_25"] = (train["Soil_Moisture"] < 25).astype(int)
train["temp_gt_30"] = (train["Temperature_C"] > 30).astype(int)
train["rain_lt_300"] = (train["Rainfall_mm"] < 300).astype(int)
train["wind_gt_10"] = (train["Wind_Speed_kmh"] > 10).astype(int)

test["soil_lt_25"] = (test["Soil_Moisture"] < 25).astype(int)
test["temp_gt_30"] = (test["Temperature_C"] > 30).astype(int)
test["rain_lt_300"] = (test["Rainfall_mm"] < 300).astype(int)
test["wind_gt_10"] = (test["Wind_Speed_kmh"] > 10).astype(int)

# --- Formula interaction features ---
# Count of "high irrigation need" signals
train["formula_score"] = train["soil_lt_25"] + train["temp_gt_30"] + train["rain_lt_300"] + train["wind_gt_10"]
test["formula_score"] = test["soil_lt_25"] + test["temp_gt_30"] + test["rain_lt_300"] + test["wind_gt_10"]

# --- Label encoding ---
label_encoders = {}
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0)
    le.fit(combined)
    train[f"{col}_le"] = le.transform(train[col])
    test[f"{col}_le"] = le.transform(test[col])
    label_encoders[col] = le

# --- Frequency encoding ---
for col in CATEGORICAL_COLS:
    freq = train[col].value_counts(normalize=True)
    train[f"{col}_freq"] = train[col].map(freq)
    test[f"{col}_freq"] = test[col].map(freq).fillna(0)

# --- K-Fold Target Encoding ---
N_TE_FOLDS = 5
skf_te = StratifiedKFold(n_splits=N_TE_FOLDS, shuffle=True, random_state=42)
train["_target_enc"] = y_all

te_cols = []
for col in CATEGORICAL_COLS:
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

# --- Logistic Regression formula features ---
# Train on original data if available, otherwise on train data
formula_num = ["soil_lt_25", "temp_gt_30", "rain_lt_300", "wind_gt_10"]
formula_cat = ["Crop_Growth_Stage", "Mulching_Used"]

if orig_df is not None:
    target_mapping = {"Low": 0, "Medium": 1, "High": 2}
    orig_y = orig_df[TARGET_COL].map(target_mapping).values
    orig_fe = orig_df.copy()
    orig_fe["soil_lt_25"] = (orig_fe["Soil_Moisture"] < 25).astype(int)
    orig_fe["temp_gt_30"] = (orig_fe["Temperature_C"] > 30).astype(int)
    orig_fe["rain_lt_300"] = (orig_fe["Rainfall_mm"] < 300).astype(int)
    orig_fe["wind_gt_10"] = (orig_fe["Wind_Speed_kmh"] > 10).astype(int)
    orig_X = pd.get_dummies(orig_fe[formula_num + formula_cat], columns=formula_cat, drop_first=False)

    classes_arr = np.unique(orig_y)
    weights = compute_class_weight("balanced", classes=classes_arr, y=orig_y)
    cw = dict(zip(classes_arr, weights))
    sw = np.array([cw[l] for l in orig_y])
    lr_formula = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000, random_state=42)
    lr_formula.fit(orig_X, orig_y, sample_weight=sw)
    log("  LR formula trained on original data (perfect formula)")
else:
    # Train on competition train data
    train_formula_X = pd.get_dummies(train[formula_num + formula_cat], columns=formula_cat, drop_first=False)
    classes_arr = np.unique(y_all)
    weights = compute_class_weight("balanced", classes=classes_arr, y=y_all)
    cw = dict(zip(classes_arr, weights))
    sw = np.array([cw[l] for l in y_all])
    lr_formula = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000, random_state=42)
    lr_formula.fit(train_formula_X, y_all, sample_weight=sw)
    log("  LR formula trained on competition data")

# Apply formula to train/test
train_formula_X = pd.get_dummies(train[formula_num + formula_cat], columns=formula_cat, drop_first=False)
test_formula_X = pd.get_dummies(test[formula_num + formula_cat], columns=formula_cat, drop_first=False)

# Align columns
for col in train_formula_X.columns:
    if col not in test_formula_X.columns:
        test_formula_X[col] = 0
for col in test_formula_X.columns:
    if col not in train_formula_X.columns:
        train_formula_X[col] = 0
test_formula_X = test_formula_X[train_formula_X.columns]

# Formula probabilities as features
formula_probs_train = lr_formula.predict_proba(train_formula_X)
formula_probs_test = lr_formula.predict_proba(test_formula_X)
for i, cls in enumerate(["High", "Low", "Medium"]):
    train[f"formula_prob_{cls}"] = formula_probs_train[:, i]
    test[f"formula_prob_{cls}"] = formula_probs_test[:, i]

# Formula prediction as feature
train["formula_pred"] = lr_formula.predict(train_formula_X)
test["formula_pred"] = lr_formula.predict(test_formula_X)

# Check formula accuracy on train
formula_acc = balanced_accuracy_score(y_all, train["formula_pred"].values)
log(f"  Formula BA on competition train: {formula_acc:.5f}")

# Build feature columns
le_cols = [f"{col}_le" for col in CATEGORICAL_COLS]
freq_cols = [f"{col}_freq" for col in CATEGORICAL_COLS]
formula_prob_cols = [f"formula_prob_{cls}" for cls in ["High", "Low", "Medium"]]
formula_bool_cols = ["soil_lt_25", "temp_gt_30", "rain_lt_300", "wind_gt_10", "formula_score"]

feat_cols = (NUMERICAL_COLS + le_cols + freq_cols + te_cols +
             formula_bool_cols + formula_prob_cols + ["formula_pred"])

# Fill NaN
for col in feat_cols:
    if col in train.columns and train[col].isna().any():
        train[col] = train[col].fillna(train[col].median())
    if col in test.columns and test[col].isna().any():
        test[col] = test[col].fillna(train[col].median())

log(f"  Features: {len(feat_cols)} (num={len(NUMERICAL_COLS)} + LE={len(le_cols)} + freq={len(freq_cols)} + TE={len(te_cols)} + formula={len(formula_bool_cols)+len(formula_prob_cols)+1})")

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
    bi = model.best_iteration if hasattr(model, 'best_iteration') and model.best_iteration else 3000
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
        "iterations": 3000, "learning_rate": 0.05,
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
# STEP 4: Ensemble + Formula Blend + Threshold Optimization
# ============================================================
log("\nSTEP 4: Ensemble + Formula Blend + Threshold Optimization")

# Ensemble 1: LGB + XGB
ens_lx_oof = 0.5 * lgb_oof + 0.5 * xgb_oof
ens_lx_test = 0.5 * lgb_test + 0.5 * xgb_test
ens_lx_pred = np.argmax(ens_lx_oof, axis=1)
ba_lx = balanced_accuracy_score(y_all, ens_lx_pred)
log(f"  LGB+XGB (50/50) BA: {ba_lx:.5f}")

# Ensemble 2: LGB + XGB + CB
if HAS_CB:
    ens_all_oof = (lgb_oof + xgb_oof + cb_oof) / 3
    ens_all_test = (lgb_test + xgb_test + cb_test) / 3
    ens_all_pred = np.argmax(ens_all_oof, axis=1)
    ba_all = balanced_accuracy_score(y_all, ens_all_pred)
    log(f"  LGB+XGB+CB (33/33/33) BA: {ba_all:.5f}")

# Blend with formula probabilities
formula_oof = train[formula_prob_cols].values
formula_test_probs = test[formula_prob_cols].values

# Try different formula blend weights
best_blend_ba = 0
best_blend_w = 0
best_blend_name = ""

for w_formula in [0.1, 0.2, 0.3, 0.4, 0.5]:
    w_model = 1 - w_formula
    if HAS_CB:
        blend_oof = w_model * ens_all_oof + w_formula * formula_oof
    else:
        blend_oof = w_model * ens_lx_oof + w_formula * formula_oof
    blend_pred = np.argmax(blend_oof, axis=1)
    blend_ba = balanced_accuracy_score(y_all, blend_pred)
    log(f"  Formula blend (model={w_model:.0%}, formula={w_formula:.0%}) BA: {blend_ba:.5f}")
    if blend_ba > best_blend_ba:
        best_blend_ba = blend_ba
        best_blend_w = w_formula
        best_blend_name = f"model{w_model:.0%}_formula{w_formula:.0%}"

# Best blend
w_f = best_blend_w
w_m = 1 - w_f
if HAS_CB:
    best_blend_oof = w_m * ens_all_oof + w_f * formula_oof
    best_blend_test = w_m * ens_all_test + w_f * formula_test_probs
else:
    best_blend_oof = w_m * ens_lx_oof + w_f * formula_oof
    best_blend_test = w_m * ens_lx_test + w_f * formula_test_probs

log(f"\n  Best blend: {best_blend_name}, BA: {best_blend_ba:.5f}")

# Threshold optimization on best blend
def ba_loss(scales, probs, y_true):
    scaled = probs * scales
    pred = np.argmax(scaled, axis=1)
    return -balanced_accuracy_score(y_true, pred)

result = minimize(ba_loss, np.ones(3), args=(best_blend_oof, y_all), method='Nelder-Mead',
                  options={'maxiter': 2000, 'xatol': 1e-5})
best_scales = result.x
opt_pred = np.argmax(best_blend_oof * best_scales, axis=1)
ba_opt = balanced_accuracy_score(y_all, opt_pred)
log(f"  Optimized scales: {best_scales}")
log(f"  Blend optimized BA: {ba_opt:.5f}")

# Apply to test
best_blend_test_scaled = best_blend_test * best_scales
test_classes_opt = np.argmax(best_blend_test_scaled, axis=1)

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

save_sub(np.argmax(lgb_test, axis=1), "r03_lgb_only")
save_sub(np.argmax(xgb_test, axis=1), "r03_xgb_only")
save_sub(np.argmax(ens_lx_test, axis=1), "r03_ens_lx")
if HAS_CB:
    save_sub(np.argmax(ens_all_test, axis=1), "r03_ens_all")
save_sub(np.argmax(best_blend_test, axis=1), "r03_blend")
save_sub(test_classes_opt, "r03_blend_opt")

# ============================================================
# SUMMARY
# ============================================================
log("\n" + "=" * 60)
log("SUMMARY")
log("=" * 60)
log(f"  Features: {len(feat_cols)}")
log(f"  Formula BA on train: {formula_acc:.5f}")
log(f"  LGB BA: {np.mean(lgb_scores):.5f}")
log(f"  XGB BA: {np.mean(xgb_scores):.5f}")
if HAS_CB:
    log(f"  CB BA: {np.mean(cb_scores):.5f}")
log(f"  LGB+XGB BA: {ba_lx:.5f}")
if HAS_CB:
    log(f"  LGB+XGB+CB BA: {ba_all:.5f}")
log(f"  Best blend ({best_blend_name}) BA: {best_blend_ba:.5f}")
log(f"  Blend optimized BA: {ba_opt:.5f}")
log(f"  Total: {time.time() - start:.1f}s")

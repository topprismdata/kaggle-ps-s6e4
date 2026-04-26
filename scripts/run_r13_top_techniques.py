"""Round 13: Top Techniques — Logit Features + Expanded TE_ORIG + Digit Extraction

Target: Close gap from LB 0.97785 to 0.982+ (deadline 2026-04-30)

Key improvements over R09:
1. Logit features from original formula (LR on orig → 3-class logits for all data)
2. TE_ORIG for ALL columns (CATS + NUMS + pairwise TE columns), not just CATS+NUMS
3. Digit extraction features for numerical columns
4. Frequency encoding for categorical columns
5. Formula-derived binary indicators + composite scores
6. More diverse model set: XGB + LGB + CB + HGB
7. 5-fold CV (keeping same as R09 for speed)

Based on Chris Deotte's top solution (0.98219) and automatylicza (0.98215).
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
from sklearn.preprocessing import TargetEncoder
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (SUBMISSIONS, TARGET_COL, ID_COL, CLASSES,
                        CATEGORICAL_COLS, NUMERICAL_COLS)

def log(msg=""):
    print(msg, flush=True)

start = time.time()
log("=" * 60)
log("Round 13: Top Techniques")
log("=" * 60)

# ============================================================
# STEP 1: Load data
# ============================================================
log("\nSTEP 1: Load data")
DATA = Path(__file__).resolve().parent.parent / "data" / "raw"
train = pd.read_csv(DATA / "train.csv", index_col="id")
test = pd.read_csv(DATA / "test.csv", index_col="id")
orig = pd.read_csv(DATA / "irrigation_prediction.csv")
test_ids = pd.read_csv(DATA / "test.csv")[ID_COL].values

TARGET = TARGET_COL
tmap = {"Low": 0, "Medium": 1, "High": 2}
rmap = {0: "Low", 1: "Medium", 2: "High"}
train[TARGET] = train[TARGET].map(tmap)
orig[TARGET] = orig[TARGET].map(tmap)
log(f"  Train: {train.shape}, Test: {test.shape}, Orig: {orig.shape}")

NUMS = NUMERICAL_COLS
CATS = CATEGORICAL_COLS
NF = 5
SEEDS = [42, 123, 456]

# ============================================================
# STEP 2: Factorize categoricals
# ============================================================
log("\nSTEP 2: Factorizing categoricals...")
combined = pd.concat([train, test, orig])
for c in CATS:
    combined[c], _ = combined[c].factorize()
combined[CATS] = combined[CATS].astype("category")
train = combined[:len(train)].copy()
test = combined[len(train):len(train)+len(test)].copy().drop(TARGET, axis=1)
orig = combined[len(train)+len(test):].copy()
del combined
gc.collect()

# ============================================================
# STEP 3: Formula-derived features
# ============================================================
log("\nSTEP 3: Adding formula-derived features...")

def add_formula_features(df):
    """Binary indicators from the reverse-engineered irrigation formula."""
    df = df.copy()
    df['f_dry_soil']    = (df['Soil_Moisture'] < 25).astype(np.int8)
    df['f_low_rain']    = (df['Rainfall_mm'] < 300).astype(np.int8)
    df['f_hot']         = (df['Temperature_C'] > 30).astype(np.int8)
    df['f_windy']       = (df['Wind_Speed_kmh'] > 10).astype(np.int8)
    df['f_harvest']     = (df['Crop_Growth_Stage'] == 3).astype(np.int8)  # Harvest after factorize
    df['f_sowing']      = (df['Crop_Growth_Stage'] == 5).astype(np.int8)  # Sowing after factorize
    df['f_mulched']     = (df['Mulching_Used'] == 1).astype(np.int8)

    df['f_high_score']  = df['f_dry_soil'] * 2 + df['f_low_rain'] * 2 + df['f_hot'] + df['f_windy']
    df['f_low_score']   = df['f_harvest'] * 2 + df['f_sowing'] * 2 + df['f_mulched']
    df['f_net_score']   = df['f_high_score'] - df['f_low_score']
    df['f_formula_pred'] = np.where(df['f_net_score'] <= 0, 0,
                           np.where(df['f_net_score'] <= 3, 1, 2))
    return df

# Need to figure out factorized values for Harvest/Sowing/Yes
# Let's check from orig data before factorize
log("  Checking factorized values for formula features...")
# Re-read original to check factorize mapping
orig_raw = pd.read_csv(DATA / "irrigation_prediction.csv")
train_raw = pd.read_csv(DATA / "train.csv", index_col="id")

# Verify factorize mapping by checking unique values
for c in ['Crop_Growth_Stage', 'Mulching_Used']:
    vals = train[c].unique()
    log(f"  {c} factorized values: {sorted(vals)}")

# We need to map: Harvest, Sowing, Yes → factorized integers
# Let's find them properly by re-factorizing
stage_map = {}
for v in train_raw['Crop_Growth_Stage'].unique():
    idx = train_raw[train_raw['Crop_Growth_Stage'] == v].index[0]
    stage_map[v] = train.loc[idx, 'Crop_Growth_Stage']
log(f"  Crop_Growth_Stage mapping: {stage_map}")

mulch_map_val = None
for v in train_raw['Mulching_Used'].unique():
    idx = train_raw[train_raw['Mulching_Used'] == v].index[0]
    mulch_map_val_stage = train.loc[idx, 'Mulching_Used']
    log(f"  Mulching_Used '{v}' → factorized {mulch_map_val_stage}")
    if v == 'Yes':
        mulch_map_val = mulch_map_val_stage

# Now add formula features with correct factorized values
harvest_code = stage_map.get('Harvest', 3)
sowing_code = stage_map.get('Sowing', 5)

def add_formula_features_v2(df):
    """Binary indicators with correct factorized codes."""
    df = df.copy()
    df['f_dry_soil']    = (df['Soil_Moisture'] < 25).astype(np.int8)
    df['f_low_rain']    = (df['Rainfall_mm'] < 300).astype(np.int8)
    df['f_hot']         = (df['Temperature_C'] > 30).astype(np.int8)
    df['f_windy']       = (df['Wind_Speed_kmh'] > 10).astype(np.int8)
    df['f_harvest']     = (df['Crop_Growth_Stage'] == harvest_code).astype(np.int8)
    df['f_sowing']      = (df['Crop_Growth_Stage'] == sowing_code).astype(np.int8)
    df['f_mulched']     = (df['Mulching_Used'] == mulch_map_val).astype(np.int8)

    df['f_high_score']  = df['f_dry_soil'] * 2 + df['f_low_rain'] * 2 + df['f_hot'] + df['f_windy']
    df['f_low_score']   = df['f_harvest'] * 2 + df['f_sowing'] * 2 + df['f_mulched']
    df['f_net_score']   = df['f_high_score'] - df['f_low_score']
    df['f_formula_pred'] = np.where(df['f_net_score'] <= 0, 0,
                           np.where(df['f_net_score'] <= 3, 1, 2))
    return df

train = add_formula_features_v2(train)
test = add_formula_features_v2(test)
orig = add_formula_features_v2(orig)
FORMULA_FEATS = ['f_dry_soil', 'f_low_rain', 'f_hot', 'f_windy', 'f_harvest',
                 'f_sowing', 'f_mulched', 'f_high_score', 'f_low_score',
                 'f_net_score', 'f_formula_pred']
log(f"  Added {len(FORMULA_FEATS)} formula features")

# ============================================================
# STEP 4: Digit extraction features
# ============================================================
log("\nSTEP 4: Adding digit extraction features...")

def add_digit_features(df, num_cols):
    """For each numeric column, extract digits at different powers of 10."""
    df = df.copy()
    digit_cols = []
    for c in num_cols:
        vals = df[c].fillna(0).values
        # Extract ones digit (for integer-like values)
        # Use floor to handle float values
        int_vals = np.floor(vals).astype(int)
        for power, name in [(1, 'd1'), (10, 'd10'), (100, 'd100')]:
            digit = (int_vals // power) % 10
            col_name = f"digit_{c}_{name}"
            df[col_name] = digit.astype(np.int8)
            digit_cols.append(col_name)
    return df, digit_cols

# Apply to all datasets
train, digit_cols_train = add_digit_features(train, NUMS)
test, digit_cols_test = add_digit_features(test, NUMS)
orig, digit_cols_orig = add_digit_features(orig, NUMS)
DIGIT_FEATS = digit_cols_train
log(f"  Added {len(DIGIT_FEATS)} digit features")

# ============================================================
# STEP 5: Frequency encoding
# ============================================================
log("\nSTEP 5: Adding frequency encoding...")

def add_freq_encoding(train_df, test_df, orig_df, cols):
    """Add frequency encoding for categorical columns."""
    combined = pd.concat([train_df[cols], test_df[cols], orig_df[cols]])
    freq_cols = []
    for c in cols:
        freq = combined[c].value_counts(normalize=True)
        col_name = f"freq_{c}"
        train_df[col_name] = train_df[c].map(freq).astype(np.float32)
        test_df[col_name] = test_df[c].map(freq).astype(np.float32)
        orig_df[col_name] = orig_df[c].map(freq).astype(np.float32)
        freq_cols.append(col_name)
    return train_df, test_df, orig_df, freq_cols

train, test, orig, FREQ_FEATS = add_freq_encoding(train, test, orig, CATS)
log(f"  Added {len(FREQ_FEATS)} frequency features")

# ============================================================
# STEP 6: Pairwise TE features (same as R09)
# ============================================================
log("\nSTEP 6: Creating pairwise interaction features...")
TE_columns = []
columns = NUMS + list(CATS)
total_pairs = len(list(combinations(columns, 2)))

for cols in combinations(columns, 2):
    name = "-".join(cols)
    train[name] = train[cols[0]].astype(str)
    for col in cols[1:]:
        train[name] = train[name] + "_" + train[col].astype(str)
    test[name] = test[cols[0]].astype(str)
    for col in cols[1:]:
        test[name] = test[name] + "_" + test[col].astype(str)
    orig[name] = orig[cols[0]].astype(str)
    for col in cols[1:]:
        orig[name] = orig[name] + "_" + orig[col].astype(str)

    cv = pd.concat([train[name], test[name], orig[name]], ignore_index=True)
    cv, _ = cv.factorize()
    if pd.Series(cv).nunique() > len(cv) // 2:
        train.drop(name, axis=1, inplace=True)
        test.drop(name, axis=1, inplace=True)
        orig.drop(name, axis=1, inplace=True)
        continue
    train[name] = cv[:len(train)]
    test[name] = cv[len(train):len(train)+len(test)]
    orig[name] = cv[len(train)+len(test):]
    TE_columns.append(name)

log(f"  Created {len(TE_columns)} pairwise TE features (from {total_pairs} total pairs)")

# ============================================================
# STEP 7: TE_ORIG for CATS + NUMS (same as R09)
# ============================================================
log("\nSTEP 7: Computing TE_ORIG features (CATS + NUMS)...")
TE_ORIG_COLS = []

for c in CATS + NUMS:
    tmp = orig.groupby(c, observed=True)[TARGET].mean().astype("float32")
    tmp.name = f"TE_ORIG_{c}"
    train = train.merge(tmp, on=c, how="left")
    train[tmp.name] = train[tmp.name].fillna(0.5)
    test = test.merge(tmp, on=c, how="left")
    test[tmp.name] = test[tmp.name].fillna(0.5)
    TE_ORIG_COLS.append(tmp.name)

log(f"  TE_ORIG features: {len(TE_ORIG_COLS)} (same as R09)")

FEATURES = [c for c in train.columns if c != TARGET]
log(f"  Total features: {len(FEATURES)}")

# ============================================================
# STEP 8: Logit features from LR trained on original data
# ============================================================
log("\nSTEP 8: Computing logit features (LR on original data)...")

# Use NUMS + factorized CATS + formula features as input to LR
logit_feature_cols = [c for c in NUMS + list(CATS) + FORMULA_FEATS
                      if c in orig.columns and c != TARGET]

# Prepare original data for LR
X_orig_lr = orig[logit_feature_cols].copy()
y_orig_lr = orig[TARGET].values

# Convert categoricals to int for LR
for c in CATS:
    X_orig_lr[c] = X_orig_lr[c].astype(int)

# Train LR on original data
lr_orig = LogisticRegression(
    multi_class="multinomial", max_iter=5000, C=1.0, random_state=42,
    solver="lbfgs"
)
lr_orig.fit(X_orig_lr, y_orig_lr)

# Predict logits (log-probabilities) for train, test, and orig
for df, name in [(train, "train"), (test, "test"), (orig, "orig")]:
    X_lr = df[logit_feature_cols].copy()
    for c in CATS:
        X_lr[c] = X_lr[c].astype(int) if X_lr[c].dtype.name == 'category' else X_lr[c]
    logit_vals = lr_orig.predict_log_proba(X_lr)
    for cls_idx, cls_name in enumerate(["Low", "Medium", "High"]):
        df[f"logit_{cls_name}"] = logit_vals[:, cls_idx].astype(np.float32)

LOGIT_FEATS = ["logit_Low", "logit_Medium", "logit_High"]
log(f"  Logit features: {LOGIT_FEATS}")
orig_ba = balanced_accuracy_score(y_orig_lr, lr_orig.predict(X_orig_lr))
log(f"  LR on orig BA: {orig_ba:.4f}")

FEATURES = [c for c in train.columns if c != TARGET]
log(f"  Final total features: {len(FEATURES)}")

# ============================================================
# Helper functions (same as R09 but with expanded features)
# ============================================================
def apply_te(X_tr, X_va, X_te, te_cols, y_tr):
    enc = TargetEncoder(target_type="multiclass", cv=5, random_state=42)
    tr_enc = enc.fit_transform(X_tr[te_cols], y_tr)
    va_enc = enc.transform(X_va[te_cols])
    te_enc = enc.transform(X_te[te_cols])
    n_enc = tr_enc.shape[1]
    col_names = [f"TE_{i}" for i in range(n_enc)]
    X_tr = pd.concat([X_tr.drop(columns=te_cols).reset_index(drop=True),
                       pd.DataFrame(tr_enc, columns=col_names)], axis=1)
    X_va = pd.concat([X_va.drop(columns=te_cols).reset_index(drop=True),
                       pd.DataFrame(va_enc, columns=col_names)], axis=1)
    X_te = pd.concat([X_te.drop(columns=te_cols).reset_index(drop=True),
                       pd.DataFrame(te_enc, columns=col_names)], axis=1)
    return X_tr, X_va, X_te

def make_bal_acc():
    def f(y_true, y_pred):
        return balanced_accuracy_score(y_true.astype(int), np.argmax(y_pred.reshape(-1, 3), axis=1))
    f.__name__ = "bal_ACC"
    return f

def train_xgb(train_df, y_arr, test_df, te_cols, seed, nm, n_orig=None, pw=None):
    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=seed)
    oof_size = n_orig if n_orig is not None else len(train_df)
    oof = np.zeros((oof_size, 3))
    tp = np.zeros((len(test_df), 3))

    for fold, (tri, vai) in enumerate(skf.split(train_df, y_arr)):
        if n_orig is not None:
            orig_vai = vai[vai < n_orig]
            if len(orig_vai) == 0:
                continue
        else:
            orig_vai = vai

        X_tr = train_df.iloc[tri].copy()
        X_va = train_df.iloc[orig_vai].copy()
        X_te = test_df.copy()
        y_tr = y_arr[tri]

        X_tr, X_va, X_te = apply_te(X_tr, X_va, X_te, te_cols, y_tr)

        classes = np.unique(y_tr)
        cw = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=y_tr)))
        sw_class = np.array([cw[l] for l in y_tr])

        if pw is not None and n_orig is not None:
            sw_sample = np.ones(len(tri))
            sw_sample[tri >= n_orig] = pw
            sw = sw_class * sw_sample
        else:
            sw = sw_class

        feat = [c for c in X_tr.columns if c != TARGET]
        for c in CATS:
            for df_ in [X_tr, X_va, X_te]:
                df_[c] = df_[c].astype(int)

        model = xgb.XGBClassifier(
            max_depth=6, subsample=0.8, colsample_bytree=0.8,
            n_estimators=5000, objective="multi:softprob", learning_rate=0.03,
            callbacks=[xgb.callback.EarlyStopping(rounds=100, metric_name="bal_ACC",
                                                   maximize=True, save_best=True)],
            eval_metric=make_bal_acc(),
            max_bin=1024, random_state=seed, n_jobs=-1, tree_method="hist"
        )
        model.fit(X_tr[feat], y_tr, eval_set=[(X_va[feat], y_arr[orig_vai])],
                  sample_weight=sw, verbose=False)
        oof[orig_vai] = model.predict_proba(X_va[feat])
        tp += model.predict_proba(X_te[feat]) / NF

        fs = balanced_accuracy_score(y_arr[orig_vai], oof[orig_vai].argmax(1))
        log(f"    {nm} fold {fold+1}: {fs:.5f}")
        del X_tr, X_va, X_te, model
        gc.collect()

    sc = balanced_accuracy_score(y_arr[:oof_size], oof.argmax(1))
    log(f"  >> {nm}: {sc:.5f}")
    return oof, tp

def train_lgb(train_df, y_arr, test_df, te_cols, seed, nm, lgb_params, n_orig=None, pw=None):
    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=seed)
    oof_size = n_orig if n_orig is not None else len(train_df)
    oof = np.zeros((oof_size, 3))
    tp = np.zeros((len(test_df), 3))

    for fold, (tri, vai) in enumerate(skf.split(train_df, y_arr)):
        if n_orig is not None:
            orig_vai = vai[vai < n_orig]
            if len(orig_vai) == 0:
                continue
        else:
            orig_vai = vai

        X_tr = train_df.iloc[tri].copy()
        X_va = train_df.iloc[orig_vai].copy()
        X_te = test_df.copy()
        y_tr = y_arr[tri]

        X_tr, X_va, X_te = apply_te(X_tr, X_va, X_te, te_cols, y_tr)

        if pw is not None and n_orig is not None:
            sw_balanced = compute_sample_weight("balanced", y_tr)
            sw_sample = np.ones(len(tri))
            sw_sample[tri >= n_orig] = pw
            sw = sw_balanced * sw_sample
        else:
            sw = compute_sample_weight("balanced", y_tr)

        feat = [c for c in X_tr.columns if c != TARGET]
        for c in CATS:
            for df_ in [X_tr, X_va, X_te]:
                df_[c] = df_[c].astype("category")

        model = lgb.LGBMClassifier(**lgb_params, random_state=seed)
        model.fit(X_tr[feat], y_tr, sample_weight=sw,
                  eval_set=[(X_va[feat], y_arr[orig_vai])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[orig_vai] = model.predict_proba(X_va[feat])
        tp += model.predict_proba(X_te[feat]) / NF
        del X_tr, X_va, X_te, model
        gc.collect()

    sc = balanced_accuracy_score(y_arr[:oof_size], oof.argmax(1))
    log(f"  >> {nm}: {sc:.5f}")
    return oof, tp

def train_cb(train_df, y_arr, test_df, te_cols, seed, nm, n_orig=None, pw=None):
    import catboost as cb_mod
    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=seed)
    oof_size = n_orig if n_orig is not None else len(train_df)
    oof = np.zeros((oof_size, 3))
    tp = np.zeros((len(test_df), 3))

    for fold, (tri, vai) in enumerate(skf.split(train_df, y_arr)):
        if n_orig is not None:
            orig_vai = vai[vai < n_orig]
            if len(orig_vai) == 0:
                continue
        else:
            orig_vai = vai

        X_tr = train_df.iloc[tri].copy()
        X_va = train_df.iloc[orig_vai].copy()
        X_te = test_df.copy()
        y_tr = y_arr[tri]

        X_tr, X_va, X_te = apply_te(X_tr, X_va, X_te, te_cols, y_tr)

        classes = np.unique(y_tr)
        cw = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=y_tr)))
        sw_class = np.array([cw[l] for l in y_tr])

        if pw is not None and n_orig is not None:
            sw_sample = np.ones(len(tri))
            sw_sample[tri >= n_orig] = pw
            sw = sw_class * sw_sample
        else:
            sw = sw_class

        feat = [c for c in X_tr.columns if c != TARGET]
        for c in CATS:
            for df_ in [X_tr, X_va, X_te]:
                df_[c] = df_[c].astype(str).astype("category")

        model = cb_mod.CatBoostClassifier(
            task_type="CPU", iterations=800, learning_rate=0.05, depth=6,
            auto_class_weights="Balanced", cat_features=CATS, verbose=0,
            colsample_bylevel=0.8, l2_leaf_reg=3.0, min_data_in_leaf=50,
            random_seed=seed
        )
        model.fit(X_tr[feat], y_tr, sample_weight=sw,
                  eval_set=(X_va[feat], y_arr[orig_vai]), early_stopping_rounds=50)
        oof[orig_vai] = model.predict_proba(X_va[feat])
        tp += model.predict_proba(X_te[feat]) / NF

        fs = balanced_accuracy_score(y_arr[orig_vai], oof[orig_vai].argmax(1))
        log(f"    {nm} fold {fold+1}: {fs:.5f}")
        del X_tr, X_va, X_te, model
        gc.collect()

    sc = balanced_accuracy_score(y_arr[:oof_size], oof.argmax(1))
    log(f"  >> {nm}: {sc:.5f}")
    return oof, tp

def train_hgb(train_df, y_arr, test_df, te_cols, seed, nm, n_orig=None, pw=None):
    """HistGradientBoosting — adds model diversity."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=seed)
    oof_size = n_orig if n_orig is not None else len(train_df)
    oof = np.zeros((oof_size, 3))
    tp = np.zeros((len(test_df), 3))

    for fold, (tri, vai) in enumerate(skf.split(train_df, y_arr)):
        if n_orig is not None:
            orig_vai = vai[vai < n_orig]
            if len(orig_vai) == 0:
                continue
        else:
            orig_vai = vai

        X_tr = train_df.iloc[tri].copy()
        X_va = train_df.iloc[orig_vai].copy()
        X_te = test_df.copy()
        y_tr = y_arr[tri]

        X_tr, X_va, X_te = apply_te(X_tr, X_va, X_te, te_cols, y_tr)

        if pw is not None and n_orig is not None:
            sw_balanced = compute_sample_weight("balanced", y_tr)
            sw_sample = np.ones(len(tri))
            sw_sample[tri >= n_orig] = pw
            sw = sw_balanced * sw_sample
        else:
            sw = compute_sample_weight("balanced", y_tr)

        feat = [c for c in X_tr.columns if c != TARGET]
        # HGB can't handle category dtype directly — convert to int
        for c in CATS:
            for df_ in [X_tr, X_va, X_te]:
                df_[c] = df_[c].astype(int)

        model = HistGradientBoostingClassifier(
            max_iter=2000, learning_rate=0.05, max_depth=8,
            min_samples_leaf=50, l2_regularization=1.0,
            max_bins=255, random_state=seed,
            class_weight="balanced", early_stopping=True,
            validation_fraction=None, n_iter_no_change=50,
        )
        model.fit(X_tr[feat], y_tr, sample_weight=sw)
        oof[orig_vai] = model.predict_proba(X_va[feat])
        tp += model.predict_proba(X_te[feat]) / NF

        fs = balanced_accuracy_score(y_arr[orig_vai], oof[orig_vai].argmax(1))
        log(f"    {nm} fold {fold+1}: {fs:.5f}")
        del X_tr, X_va, X_te, model
        gc.collect()

    sc = balanced_accuracy_score(y_arr[:oof_size], oof.argmax(1))
    log(f"  >> {nm}: {sc:.5f}")
    return oof, tp

# ============================================================
# STEP 9: Stage 1 — Train models without pseudo labels
# ============================================================
y = train[TARGET].values

log("\nSTEP 9: Stage 1 — Train without pseudo labels...")
s1_oof = {}
s1_tp = {}

# 9a: XGBoost (3 seeds)
log("  --- XGBoost ---")
for SEED in SEEDS:
    nm = f"s1_xgb_s{SEED}"
    oof, tp = train_xgb(train, y, test, TE_columns, SEED, nm)
    s1_oof[nm] = oof
    s1_tp[nm] = tp

# 9b: LightGBM (2 configs × 3 seeds = 6 models)
log("  --- LightGBM ---")
LGB_CONFIGS = [
    dict(n_estimators=2000, learning_rate=0.02, num_leaves=127, max_depth=9,
         class_weight="balanced", verbose=-1, colsample_bytree=0.7, subsample=0.8,
         reg_alpha=0.05, reg_lambda=0.1, min_child_samples=50),
    dict(n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
         class_weight="balanced", verbose=-1, colsample_bytree=0.8, subsample=0.7,
         reg_alpha=0.2, reg_lambda=0.3, min_child_samples=30),
]
for SEED in SEEDS:
    for li, lp in enumerate(LGB_CONFIGS):
        nm = f"s1_lgb{li}_s{SEED}"
        oof, tp = train_lgb(train, y, test, TE_columns, SEED, nm, lp)
        s1_oof[nm] = oof
        s1_tp[nm] = tp

# 9c: CatBoost (1 model)
log("  --- CatBoost ---")
nm = "s1_cb_s42"
oof, tp = train_cb(train, y, test, TE_columns, 42, nm)
s1_oof[nm] = oof
s1_tp[nm] = tp

# 9d: HistGradientBoosting (1 model — new diversity)
log("  --- HistGradientBoosting ---")
nm = "s1_hgb_s42"
oof, tp = train_hgb(train, y, test, TE_columns, 42, nm)
s1_oof[nm] = oof
s1_tp[nm] = tp

log(f"\n  Stage 1 models: {len(s1_oof)}")

# ============================================================
# STEP 10: Pseudo-Labeling
# ============================================================
log("\nSTEP 10: Pseudo-Labeling...")
s1_avg_tp = sum(s1_tp.values()) / len(s1_tp)
s1_pred_labels = s1_avg_tp.argmax(axis=1)
s1_pred_conf = s1_avg_tp.max(axis=1)

PSEUDO_THRESHOLD = 0.90
pseudo_mask = s1_pred_conf >= PSEUDO_THRESHOLD
log(f"  Total test: {len(test)}, Pseudo-labeled: {pseudo_mask.sum()} ({pseudo_mask.mean()*100:.1f}%)")

pseudo_labels = s1_pred_labels[pseudo_mask]
for cls_id, cls_name in rmap.items():
    log(f"    {cls_name}: {(pseudo_labels == cls_id).sum()}")

pseudo_test = test[pseudo_mask].copy()
pseudo_test[TARGET] = pseudo_labels
pseudo_y = pseudo_labels.copy()

# ============================================================
# STEP 11: Stage 2 — Retrain WITH pseudo-labeled data
# ============================================================
log(f"\nSTEP 11: Stage 2 — Train WITH pseudo-labeled data...")
train_with_pseudo = pd.concat([train, pseudo_test], ignore_index=True)
y_with_pseudo = np.concatenate([y, pseudo_y])
log(f"  Train+Pseudo: {len(train_with_pseudo)} rows")

N_ORIG = len(train)
PW = 0.5

s2_oof = {}
s2_tp = {}

# 11a: XGBoost
log("  --- XGBoost ---")
for SEED in SEEDS:
    nm = f"s2_xgb_s{SEED}"
    oof, tp = train_xgb(train_with_pseudo, y_with_pseudo, test, TE_columns,
                         SEED, nm, n_orig=N_ORIG, pw=PW)
    s2_oof[nm] = oof
    s2_tp[nm] = tp

# 11b: LightGBM
log("  --- LightGBM ---")
for SEED in SEEDS:
    for li, lp in enumerate(LGB_CONFIGS):
        nm = f"s2_lgb{li}_s{SEED}"
        oof, tp = train_lgb(train_with_pseudo, y_with_pseudo, test, TE_columns,
                             SEED, nm, lp, n_orig=N_ORIG, pw=PW)
        s2_oof[nm] = oof
        s2_tp[nm] = tp

# 11c: CatBoost
log("  --- CatBoost ---")
nm = "s2_cb_s42"
oof, tp = train_cb(train_with_pseudo, y_with_pseudo, test, TE_columns,
                    42, nm, n_orig=N_ORIG, pw=PW)
s2_oof[nm] = oof
s2_tp[nm] = tp

# 11d: HistGradientBoosting
log("  --- HistGradientBoosting ---")
nm = "s2_hgb_s42"
oof, tp = train_hgb(train_with_pseudo, y_with_pseudo, test, TE_columns,
                     42, nm, n_orig=N_ORIG, pw=PW)
s2_oof[nm] = oof
s2_tp[nm] = tp

log(f"\n  Stage 2 models: {len(s2_oof)}")

# ============================================================
# STEP 12: Fast Ensemble
# ============================================================
log(f"\nSTEP 12: Fast Ensemble ({len(s2_oof)} models)...")
names = list(s2_oof.keys())

# Simple average
simple_avg = sum(s2_oof[n] for n in names) / len(names)
sa_score = balanced_accuracy_score(y, simple_avg.argmax(1))
log(f"  Simple avg OOF: {sa_score:.5f}")

# Individual scores
for n in names:
    s = balanced_accuracy_score(y, s2_oof[n].argmax(1))
    log(f"    {n}: {s:.5f}")

# Weighted average (fast: 2000 iterations)
bwa, bww = 0, None
rng = np.random.RandomState(42)
for i in range(2000):
    w = rng.dirichlet(np.ones(len(names)))
    combo = sum(w[j] * s2_oof[n] for j, n in enumerate(names))
    s = balanced_accuracy_score(y, combo.argmax(1))
    if s > bwa:
        bwa = s
        bww = w.copy()
log(f"  Best weighted OOF: {bwa:.5f}")

# Stacking with LR
log("  Running LR stacking...")
ostk = np.hstack([s2_oof[n] for n in names])
tstk = np.hstack([s2_tp[n] for n in names])

skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=42)
moof = np.zeros(len(y), dtype=int)
mtest = np.zeros((len(test), 3))
for fold, (tri, vai) in enumerate(skf.split(ostk, y)):
    lr = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, random_state=42)
    lr.fit(ostk[tri], y[tri])
    moof[vai] = lr.predict(ostk[vai])
    mtest += lr.predict_proba(tstk) / NF
ssc = balanced_accuracy_score(y, moof)
log(f"  Stacked OOF: {ssc:.5f}")

# Choose best ensemble method
if ssc >= bwa and ssc >= sa_score:
    log("  >>> Using Stacking")
    fm = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, random_state=42)
    fm.fit(ostk, y)
    bt = fm.predict_proba(tstk)
    bo = np.zeros((len(y), 3))
    for fold, (tri, vai) in enumerate(skf.split(ostk, y)):
        lr = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, random_state=42)
        lr.fit(ostk[tri], y[tri])
        bo[vai] = lr.predict_proba(ostk[vai])
elif bwa >= sa_score:
    log("  >>> Using weighted average")
    bt = sum(bww[j] * s2_tp[n] for j, n in enumerate(names))
    bo = sum(bww[j] * s2_oof[n] for j, n in enumerate(names))
else:
    log("  >>> Using simple average")
    bt = sum(s2_tp[n] for n in names) / len(names)
    bo = simple_avg

# ============================================================
# STEP 13: Threshold Optimization
# ============================================================
log("\nSTEP 13: Threshold optimization...")

def neg_ba(w):
    return -balanced_accuracy_score(y, (bo * np.array([1.0, w[0], w[1]])).argmax(1))

bg = (1.0, 1.0)
bgs = -1
for wm in np.arange(0.3, 2.0, 0.05):
    for wh in np.arange(0.3, 8.0, 0.1):
        s = -neg_ba([wm, wh])
        if s > bgs:
            bgs = s
            bg = (wm, wh)

res = minimize(neg_ba, list(bg), method="Nelder-Mead", options={"xatol": 0.001, "fatol": 1e-6, "maxiter": 1000})
bw = [1.0, res.x[0], res.x[1]]
fcv = -res.fun
log(f"  Weights: Low={bw[0]:.3f} Med={bw[1]:.3f} High={bw[2]:.3f}")
log(f"  FINAL CV: {fcv:.5f}")

# ============================================================
# STEP 14: Save submissions
# ============================================================
log("\nSTEP 14: Save submissions...")

# Threshold optimized
preds_thresh = (bt * np.array(bw)).argmax(1)
sub = pd.DataFrame({ID_COL: test_ids, TARGET_COL: [rmap[p] for p in preds_thresh]})
fname = SUBMISSIONS / "submission_r13_top_techniques.csv"
sub.to_csv(fname, index=False)
dist = pd.Series([rmap[p] for p in preds_thresh]).value_counts()
log(f"  r13_top_techniques: {dict(dist)}")

# Default (no threshold opt)
preds_default = bt.argmax(1)
sub2 = pd.DataFrame({ID_COL: test_ids, TARGET_COL: [rmap[p] for p in preds_default]})
fname2 = SUBMISSIONS / "submission_r13_ens_default.csv"
sub2.to_csv(fname2, index=False)
dist2 = pd.Series([rmap[p] for p in preds_default]).value_counts()
log(f"  r13_ens_default: {dict(dist2)}")

# Weighted avg (no stacking)
if bww is not None:
    preds_wavg = (sum(bww[j] * s2_tp[n] for j, n in enumerate(names))).argmax(1)
    sub3 = pd.DataFrame({ID_COL: test_ids, TARGET_COL: [rmap[p] for p in preds_wavg]})
    fname3 = SUBMISSIONS / "submission_r13_weighted_avg.csv"
    sub3.to_csv(fname3, index=False)
    dist3 = pd.Series([rmap[p] for p in preds_wavg]).value_counts()
    log(f"  r13_weighted_avg: {dict(dist3)}")

elapsed = time.time() - start
log("\n" + "=" * 60)
log(f"R13 Summary: {elapsed:.1f}s, {len(s2_oof)} models, {len(FEATURES)} features")
log(f"FINAL CV: {fcv:.5f}")
log(f"Submissions saved to {SUBMISSIONS}")
log("=" * 60)

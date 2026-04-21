"""Round 10: Diverse Model Types + Pseudo-Labeling

Adds MLP, LR, and HistGradientBoosting as diverse model types to R09.
Key insight: model TYPE diversity often beats seed diversity for ensembles.

Pipeline:
1. Same features as R09 (135 pairwise TE + 19 TE_ORIG)
2. Stage 1: 10 GBDT models + 3 MLP + 1 LR + 1 HGB = 15 models
3. Pseudo-label from Stage 1 average
4. Stage 2: Retrain all with pseudo data
5. Fast ensemble + stacking + threshold opt
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
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (SUBMISSIONS, TARGET_COL, ID_COL, CLASSES,
                        CATEGORICAL_COLS, NUMERICAL_COLS)

def log(msg=""):
    print(msg, flush=True)

start = time.time()
log("=" * 60)
log("Round 10: Diverse Model Types + Pseudo-Labeling")
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
# STEP 3: Pairwise TE features
# ============================================================
log("\nSTEP 3: Creating pairwise interaction features...")
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
# STEP 4: TE_ORIG features
# ============================================================
log("\nSTEP 4: Computing TE_ORIG features...")
for c in CATS + NUMS:
    tmp = orig.groupby(c, observed=True)[TARGET].mean().astype("float32")
    tmp.name = f"TE_ORIG_{c}"
    train = train.merge(tmp, on=c, how="left")
    train[tmp.name] = train[tmp.name].fillna(0.5)
    test = test.merge(tmp, on=c, how="left")
    test[tmp.name] = test[tmp.name].fillna(0.5)

FEATURES = [c for c in train.columns if c != TARGET]
log(f"  Total features before TE: {len(FEATURES)}")

# ============================================================
# Helper functions
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

def get_features_and_convert(X_tr, X_va, X_te, to_int_cats=True):
    feat = [c for c in X_tr.columns if c != TARGET]
    if to_int_cats:
        for c in CATS:
            for df_ in [X_tr, X_va, X_te]:
                df_[c] = df_[c].astype(int)
    return X_tr, X_va, X_te, feat

def train_xgb(train_df, y_arr, test_df, te_cols, seed, nm, n_orig=None, pw=None):
    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=seed)
    oof_size = n_orig if n_orig is not None else len(train_df)
    oof = np.zeros((oof_size, 3))
    tp = np.zeros((len(test_df), 3))

    for fold, (tri, vai) in enumerate(skf.split(train_df, y_arr)):
        orig_vai = vai[vai < n_orig] if n_orig is not None else vai
        if len(orig_vai) == 0:
            continue

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

        X_tr, X_va, X_te, feat = get_features_and_convert(X_tr, X_va, X_te)

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
        orig_vai = vai[vai < n_orig] if n_orig is not None else vai
        if len(orig_vai) == 0:
            continue

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
        orig_vai = vai[vai < n_orig] if n_orig is not None else vai
        if len(orig_vai) == 0:
            continue

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

def train_mlp(train_df, y_arr, test_df, te_cols, seed, nm, hidden_sizes=(256, 128),
              n_orig=None, pw=None):
    """Train MLP with 5-fold CV. Uses StandardScaler on numerical features."""
    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=seed)
    oof_size = n_orig if n_orig is not None else len(train_df)
    oof = np.zeros((oof_size, 3))
    tp = np.zeros((len(test_df), 3))

    for fold, (tri, vai) in enumerate(skf.split(train_df, y_arr)):
        orig_vai = vai[vai < n_orig] if n_orig is not None else vai
        if len(orig_vai) == 0:
            continue

        X_tr = train_df.iloc[tri].copy()
        X_va = train_df.iloc[orig_vai].copy()
        X_te = test_df.copy()
        y_tr = y_arr[tri]

        X_tr, X_va, X_te = apply_te(X_tr, X_va, X_te, te_cols, y_tr)
        X_tr, X_va, X_te, feat = get_features_and_convert(X_tr, X_va, X_te)

        # Scale features for MLP
        scaler = StandardScaler()
        X_tr_f = scaler.fit_transform(X_tr[feat])
        X_va_f = scaler.transform(X_va[feat])
        X_te_f = scaler.transform(X_te[feat])

        if pw is not None and n_orig is not None:
            sw_balanced = compute_sample_weight("balanced", y_tr)
            sw_sample = np.ones(len(tri))
            sw_sample[tri >= n_orig] = pw
            sw = sw_balanced * sw_sample
        else:
            sw = compute_sample_weight("balanced", y_tr)

        model = MLPClassifier(
            hidden_layer_sizes=hidden_sizes, max_iter=500, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=20, random_state=seed,
            learning_rate="adaptive", learning_rate_init=0.001
        )
        model.fit(X_tr_f, y_tr, sample_weight=sw)
        oof[orig_vai] = model.predict_proba(X_va_f)
        tp += model.predict_proba(X_te_f) / NF

        fs = balanced_accuracy_score(y_arr[orig_vai], oof[orig_vai].argmax(1))
        log(f"    {nm} fold {fold+1}: {fs:.5f}")
        del X_tr, X_va, X_te, model, scaler
        gc.collect()

    sc = balanced_accuracy_score(y_arr[:oof_size], oof.argmax(1))
    log(f"  >> {nm}: {sc:.5f}")
    return oof, tp

def train_lr(train_df, y_arr, test_df, te_cols, seed, nm, n_orig=None, pw=None):
    """Train LogisticRegression with 5-fold CV."""
    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=seed)
    oof_size = n_orig if n_orig is not None else len(train_df)
    oof = np.zeros((oof_size, 3))
    tp = np.zeros((len(test_df), 3))

    for fold, (tri, vai) in enumerate(skf.split(train_df, y_arr)):
        orig_vai = vai[vai < n_orig] if n_orig is not None else vai
        if len(orig_vai) == 0:
            continue

        X_tr = train_df.iloc[tri].copy()
        X_va = train_df.iloc[orig_vai].copy()
        X_te = test_df.copy()
        y_tr = y_arr[tri]

        X_tr, X_va, X_te = apply_te(X_tr, X_va, X_te, te_cols, y_tr)
        X_tr, X_va, X_te, feat = get_features_and_convert(X_tr, X_va, X_te)

        scaler = StandardScaler()
        X_tr_f = scaler.fit_transform(X_tr[feat])
        X_va_f = scaler.transform(X_va[feat])
        X_te_f = scaler.transform(X_te[feat])

        if pw is not None and n_orig is not None:
            sw_balanced = compute_sample_weight("balanced", y_tr)
            sw_sample = np.ones(len(tri))
            sw_sample[tri >= n_orig] = pw
            sw = sw_balanced * sw_sample
        else:
            sw = compute_sample_weight("balanced", y_tr)

        model = LogisticRegression(
            class_weight="balanced", max_iter=2000, C=1.0, random_state=seed
        )
        model.fit(X_tr_f, y_tr, sample_weight=sw)
        oof[orig_vai] = model.predict_proba(X_va_f)
        tp += model.predict_proba(X_te_f) / NF

        fs = balanced_accuracy_score(y_arr[orig_vai], oof[orig_vai].argmax(1))
        log(f"    {nm} fold {fold+1}: {fs:.5f}")
        del X_tr, X_va, X_te, model, scaler
        gc.collect()

    sc = balanced_accuracy_score(y_arr[:oof_size], oof.argmax(1))
    log(f"  >> {nm}: {sc:.5f}")
    return oof, tp

def train_hgb(train_df, y_arr, test_df, te_cols, seed, nm, n_orig=None, pw=None):
    """Train HistGradientBoosting with 5-fold CV."""
    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=seed)
    oof_size = n_orig if n_orig is not None else len(train_df)
    oof = np.zeros((oof_size, 3))
    tp = np.zeros((len(test_df), 3))

    for fold, (tri, vai) in enumerate(skf.split(train_df, y_arr)):
        orig_vai = vai[vai < n_orig] if n_orig is not None else vai
        if len(orig_vai) == 0:
            continue

        X_tr = train_df.iloc[tri].copy()
        X_va = train_df.iloc[orig_vai].copy()
        X_te = test_df.copy()
        y_tr = y_arr[tri]

        X_tr, X_va, X_te = apply_te(X_tr, X_va, X_te, te_cols, y_tr)
        X_tr, X_va, X_te, feat = get_features_and_convert(X_tr, X_va, X_te)

        if pw is not None and n_orig is not None:
            sw_balanced = compute_sample_weight("balanced", y_tr)
            sw_sample = np.ones(len(tri))
            sw_sample[tri >= n_orig] = pw
            sw = sw_balanced * sw_sample
        else:
            sw = compute_sample_weight("balanced", y_tr)

        model = HistGradientBoostingClassifier(
            max_iter=1000, learning_rate=0.05, max_depth=8, min_samples_leaf=50,
            l2_regularization=0.1, random_state=seed, early_stopping=True,
            n_iter_no_change=50, class_weight="balanced"
        )
        model.fit(X_tr[feat].values, y_tr, sample_weight=sw)
        oof[orig_vai] = model.predict_proba(X_va[feat].values)
        tp += model.predict_proba(X_te[feat].values) / NF

        fs = balanced_accuracy_score(y_arr[orig_vai], oof[orig_vai].argmax(1))
        log(f"    {nm} fold {fold+1}: {fs:.5f}")
        del X_tr, X_va, X_te, model
        gc.collect()

    sc = balanced_accuracy_score(y_arr[:oof_size], oof.argmax(1))
    log(f"  >> {nm}: {sc:.5f}")
    return oof, tp

# ============================================================
# STEP 5: Stage 1 — Train all models
# ============================================================
y = train[TARGET].values
LGB_CONFIGS = [
    dict(n_estimators=2000, learning_rate=0.02, num_leaves=127, max_depth=9,
         class_weight="balanced", verbose=-1, colsample_bytree=0.7, subsample=0.8,
         reg_alpha=0.05, reg_lambda=0.1, min_child_samples=50),
    dict(n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
         class_weight="balanced", verbose=-1, colsample_bytree=0.8, subsample=0.7,
         reg_alpha=0.2, reg_lambda=0.3, min_child_samples=30),
]

log("\nSTEP 5: Stage 1 — Train all models...")
s1_oof = {}
s1_tp = {}

# XGBoost
log("  --- XGBoost ---")
for SEED in SEEDS:
    nm = f"s1_xgb_s{SEED}"
    oof, tp = train_xgb(train, y, test, TE_columns, SEED, nm)
    s1_oof[nm] = oof
    s1_tp[nm] = tp

# LightGBM
log("  --- LightGBM ---")
for SEED in SEEDS:
    for li, lp in enumerate(LGB_CONFIGS):
        nm = f"s1_lgb{li}_s{SEED}"
        oof, tp = train_lgb(train, y, test, TE_columns, SEED, nm, lp)
        s1_oof[nm] = oof
        s1_tp[nm] = tp

# CatBoost
log("  --- CatBoost ---")
nm = "s1_cb_s42"
oof, tp = train_cb(train, y, test, TE_columns, 42, nm)
s1_oof[nm] = oof
s1_tp[nm] = tp

# MLP (2 configs × 1 seed = 2)
log("  --- MLP ---")
for mi, (hs, nm) in enumerate([((256, 128), "s1_mlp256"), ((512, 256, 128), "s1_mlp512")]):
    oof, tp = train_mlp(train, y, test, TE_columns, 42, nm, hidden_sizes=hs)
    s1_oof[nm] = oof
    s1_tp[nm] = tp

# LR
log("  --- LogisticRegression ---")
oof, tp = train_lr(train, y, test, TE_columns, 42, "s1_lr")
s1_oof["s1_lr"] = oof
s1_tp["s1_lr"] = tp

# HistGradientBoosting
log("  --- HistGradientBoosting ---")
oof, tp = train_hgb(train, y, test, TE_columns, 42, "s1_hgb")
s1_oof["s1_hgb"] = oof
s1_tp["s1_hgb"] = tp

log(f"\n  Stage 1 total models: {len(s1_oof)}")

# ============================================================
# STEP 6: Pseudo-Labeling
# ============================================================
log("\nSTEP 6: Pseudo-Labeling...")
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
# STEP 7: Stage 2 — Retrain with pseudo
# ============================================================
log(f"\nSTEP 7: Stage 2 — Train WITH pseudo-labeled data...")
train_with_pseudo = pd.concat([train, pseudo_test], ignore_index=True)
y_with_pseudo = np.concatenate([y, pseudo_y])
log(f"  Train+Pseudo: {len(train_with_pseudo)} rows")

N_ORIG = len(train)
PW = 0.5

s2_oof = {}
s2_tp = {}

# XGBoost
log("  --- XGBoost ---")
for SEED in SEEDS:
    nm = f"s2_xgb_s{SEED}"
    oof, tp = train_xgb(train_with_pseudo, y_with_pseudo, test, TE_columns,
                         SEED, nm, n_orig=N_ORIG, pw=PW)
    s2_oof[nm] = oof
    s2_tp[nm] = tp

# LightGBM
log("  --- LightGBM ---")
for SEED in SEEDS:
    for li, lp in enumerate(LGB_CONFIGS):
        nm = f"s2_lgb{li}_s{SEED}"
        oof, tp = train_lgb(train_with_pseudo, y_with_pseudo, test, TE_columns,
                             SEED, nm, lp, n_orig=N_ORIG, pw=PW)
        s2_oof[nm] = oof
        s2_tp[nm] = tp

# CatBoost
log("  --- CatBoost ---")
oof, tp = train_cb(train_with_pseudo, y_with_pseudo, test, TE_columns,
                    42, "s2_cb_s42", n_orig=N_ORIG, pw=PW)
s2_oof["s2_cb_s42"] = oof
s2_tp["s2_cb_s42"] = tp

# MLP
log("  --- MLP ---")
for mi, (hs, nm) in enumerate([((256, 128), "s2_mlp256"), ((512, 256, 128), "s2_mlp512")]):
    oof, tp = train_mlp(train_with_pseudo, y_with_pseudo, test, TE_columns,
                         42, nm, hidden_sizes=hs, n_orig=N_ORIG, pw=PW)
    s2_oof[nm] = oof
    s2_tp[nm] = tp

# LR
log("  --- LogisticRegression ---")
oof, tp = train_lr(train_with_pseudo, y_with_pseudo, test, TE_columns,
                    42, "s2_lr", n_orig=N_ORIG, pw=PW)
s2_oof["s2_lr"] = oof
s2_tp["s2_lr"] = tp

# HGB
log("  --- HistGradientBoosting ---")
oof, tp = train_hgb(train_with_pseudo, y_with_pseudo, test, TE_columns,
                     42, "s2_hgb", n_orig=N_ORIG, pw=PW)
s2_oof["s2_hgb"] = oof
s2_tp["s2_hgb"] = tp

log(f"\n  Stage 2 total models: {len(s2_oof)}")

# ============================================================
# STEP 8: Fast Ensemble
# ============================================================
log(f"\nSTEP 8: Fast Ensemble ({len(s2_oof)} models)...")
names = list(s2_oof.keys())

# Simple average
simple_avg = sum(s2_oof[n] for n in names) / len(names)
sa_score = balanced_accuracy_score(y, simple_avg.argmax(1))
log(f"  Simple avg OOF: {sa_score:.5f}")

# Individual scores
for n in names:
    s = balanced_accuracy_score(y, s2_oof[n].argmax(1))
    log(f"    {n}: {s:.5f}")

# Weighted average (2000 iter)
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

# Stacking
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

# Choose best
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
# STEP 9: Threshold Optimization
# ============================================================
log("\nSTEP 9: Threshold optimization...")

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
# STEP 10: Save submissions
# ============================================================
log("\nSTEP 10: Save submissions...")

preds_thresh = (bt * np.array(bw)).argmax(1)
sub = pd.DataFrame({ID_COL: test_ids, TARGET_COL: [rmap[p] for p in preds_thresh]})
sub.to_csv(SUBMISSIONS / "submission_r10_thresh_opt.csv", index=False)
dist = pd.Series([rmap[p] for p in preds_thresh]).value_counts()
log(f"  r10_thresh_opt: {dict(dist)}")

preds_default = bt.argmax(1)
sub2 = pd.DataFrame({ID_COL: test_ids, TARGET_COL: [rmap[p] for p in preds_default]})
sub2.to_csv(SUBMISSIONS / "submission_r10_ens_default.csv", index=False)
dist2 = pd.Series([rmap[p] for p in preds_default]).value_counts()
log(f"  r10_ens_default: {dict(dist2)}")

elapsed = int(time.time() - start)

log("\n" + "=" * 60)
log(f"SUMMARY - Round 10")
log("=" * 60)
log(f"  Total models: {len(s2_oof)}")
log(f"  Pseudo samples: {pseudo_mask.sum()} / {len(test)} ({pseudo_mask.mean()*100:.1f}%)")
log(f"  Simple avg: {sa_score:.5f}")
log(f"  Weighted avg: {bwa:.5f}")
log(f"  Stacked: {ssc:.5f}")
log(f"  FINAL CV: {fcv:.5f}")
log(f"  Total: {elapsed}s")

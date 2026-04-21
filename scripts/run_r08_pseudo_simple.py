"""Round 08: Pseudo-Labeling with Fast Ensemble

Same as R07 but with optimized ensemble:
1. Reduce weight search to 5000 iterations
2. Simple average as fallback
3. No stacking (LR is fast but the weighted avg search was the bottleneck)

Uses same 3 models (2 XGB + 1 CB), 2 stages.
"""
import warnings
warnings.filterwarnings("ignore")
import time
import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
import sys
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import TargetEncoder
from sklearn.utils.class_weight import compute_class_weight
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (SUBMISSIONS, TARGET_COL, ID_COL, CLASSES,
                        CATEGORICAL_COLS, NUMERICAL_COLS)

def log(msg=""):
    print(msg, flush=True)

start = time.time()
log("=" * 60)
log("Round 08: Pseudo-Labeling + Fast Ensemble")
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

# ============================================================
# STEP 2-4: Feature Engineering (same as R05)
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

log("\nSTEP 3: Creating pairwise interaction features...")
TE_columns = []
columns = NUMS + CATS
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

def train_xgb(train_df, y_arr, test_df, te_cols, seed, nm):
    """Train XGB with 5-fold CV."""
    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=seed)
    oof = np.zeros((len(train_df), 3))
    tp = np.zeros((len(test_df), 3))

    for fold, (tri, vai) in enumerate(skf.split(train_df, y_arr)):
        X_tr, X_va, X_te = train_df.iloc[tri].copy(), train_df.iloc[vai].copy(), test_df.copy()
        X_tr, X_va, X_te = apply_te(X_tr, X_va, X_te, te_cols, y_arr[tri])

        classes = np.unique(y_arr[tri])
        cw = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=y_arr[tri])))
        sw = np.array([cw[l] for l in y_arr[tri]])

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
        model.fit(X_tr[feat], y_arr[tri], eval_set=[(X_va[feat], y_arr[vai])],
                  sample_weight=sw, verbose=False)
        oof[vai] = model.predict_proba(X_va[feat])
        tp += model.predict_proba(X_te[feat]) / NF

        fs = balanced_accuracy_score(y_arr[vai], oof[vai].argmax(1))
        log(f"    {nm} fold {fold+1}: {fs:.5f}")
        del X_tr, X_va, X_te, model
        gc.collect()

    sc = balanced_accuracy_score(y_arr, oof.argmax(1))
    log(f"  >> {nm}: {sc:.5f}")
    return oof, tp

def train_cb(train_df, y_arr, test_df, te_cols, seed, nm):
    """Train CatBoost with 5-fold CV."""
    import catboost as cb_mod
    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=seed)
    oof = np.zeros((len(train_df), 3))
    tp = np.zeros((len(test_df), 3))

    for fold, (tri, vai) in enumerate(skf.split(train_df, y_arr)):
        X_tr, X_va, X_te = train_df.iloc[tri].copy(), train_df.iloc[vai].copy(), test_df.copy()
        X_tr, X_va, X_te = apply_te(X_tr, X_va, X_te, te_cols, y_arr[tri])

        classes = np.unique(y_arr[tri])
        cw = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=y_arr[tri])))
        sw = np.array([cw[l] for l in y_arr[tri]])

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
        model.fit(X_tr[feat], y_arr[tri], sample_weight=sw,
                  eval_set=(X_va[feat], y_arr[vai]), early_stopping_rounds=50)
        oof[vai] = model.predict_proba(X_va[feat])
        tp += model.predict_proba(X_te[feat]) / NF

        fs = balanced_accuracy_score(y_arr[vai], oof[vai].argmax(1))
        log(f"    {nm} fold {fold+1}: {fs:.5f}")
        del X_tr, X_va, X_te, model
        gc.collect()

    sc = balanced_accuracy_score(y_arr, oof.argmax(1))
    log(f"  >> {nm}: {sc:.5f}")
    return oof, tp

# ============================================================
# STEP 5: Stage 1 — Train without pseudo labels
# ============================================================
y = train[TARGET].values

log(f"\nSTEP 5: Stage 1 — Train without pseudo labels...")
s1_xgb42_oof, s1_xgb42_tp = train_xgb(train, y, test, TE_columns, 42, "s1_xgb42")
s1_xgb123_oof, s1_xgb123_tp = train_xgb(train, y, test, TE_columns, 123, "s1_xgb123")
s1_cb42_oof, s1_cb42_tp = train_cb(train, y, test, TE_columns, 42, "s1_cb42")

# ============================================================
# STEP 6: Pseudo-Labeling
# ============================================================
log("\nSTEP 6: Pseudo-Labeling...")
s1_avg_tp = (s1_xgb42_tp + s1_xgb123_tp + s1_cb42_tp) / 3
s1_pred_labels = s1_avg_tp.argmax(axis=1)
s1_pred_conf = s1_avg_tp.max(axis=1)

PSEUDO_THRESHOLD = 0.90
pseudo_mask = s1_pred_conf >= PSEUDO_THRESHOLD
log(f"  Total test: {len(test)}, Pseudo-labeled: {pseudo_mask.sum()} ({pseudo_mask.mean()*100:.1f}%)")

# Class distribution
pseudo_labels = s1_pred_labels[pseudo_mask]
for cls_id, cls_name in rmap.items():
    log(f"    {cls_name}: {(pseudo_labels == cls_id).sum()}")

# Create pseudo test df
pseudo_test = test[pseudo_mask].copy()
pseudo_test[TARGET] = pseudo_labels
pseudo_y = pseudo_labels.copy()

# ============================================================
# STEP 7: Stage 2 — Retrain with pseudo labels
# ============================================================
log(f"\nSTEP 7: Stage 2 — Train WITH pseudo-labeled data...")
train_with_pseudo = pd.concat([train, pseudo_test], ignore_index=True)
y_with_pseudo = np.concatenate([y, pseudo_y])
log(f"  Train+Pseudo: {len(train_with_pseudo)} rows")

PSEUDO_WEIGHT = 0.5

# For Stage 2, we need OOF only on original train rows
# Custom training function that handles pseudo weight
def train_xgb_pseudo(train_full, y_full, test_df, te_cols, seed, nm, n_orig, pw):
    """Train XGB with pseudo-labeled data, OOF only on original rows."""
    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=seed)
    oof = np.zeros((n_orig, 3))
    tp = np.zeros((len(test_df), 3))

    for fold, (tri, vai) in enumerate(skf.split(train_full, y_full)):
        # Only evaluate on original train indices
        orig_vai = vai[vai < n_orig]
        if len(orig_vai) == 0:
            continue

        X_tr = train_full.iloc[tri].copy()
        X_va = train_full.iloc[orig_vai].copy()
        X_te = test_df.copy()
        y_tr = y_full[tri]
        y_va = y_full[orig_vai]

        X_tr, X_va, X_te = apply_te(X_tr, X_va, X_te, te_cols, y_tr)

        classes = np.unique(y_tr)
        cw = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=y_tr)))
        sw_class = np.array([cw[l] for l in y_tr])
        sw_sample = np.ones(len(tri))
        sw_sample[tri >= n_orig] = pw
        sw = sw_class * sw_sample

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
        model.fit(X_tr[feat], y_tr, eval_set=[(X_va[feat], y_va)],
                  sample_weight=sw, verbose=False)
        oof[orig_vai] = model.predict_proba(X_va[feat])
        tp += model.predict_proba(X_te[feat]) / NF

        fs = balanced_accuracy_score(y_va, oof[orig_vai].argmax(1))
        log(f"    {nm} fold {fold+1}: {fs:.5f}")
        del X_tr, X_va, X_te, model
        gc.collect()

    sc = balanced_accuracy_score(y, oof.argmax(1))
    log(f"  >> {nm}: {sc:.5f}")
    return oof, tp

def train_cb_pseudo(train_full, y_full, test_df, te_cols, seed, nm, n_orig, pw):
    """Train CB with pseudo-labeled data."""
    import catboost as cb_mod
    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=seed)
    oof = np.zeros((n_orig, 3))
    tp = np.zeros((len(test_df), 3))

    for fold, (tri, vai) in enumerate(skf.split(train_full, y_full)):
        orig_vai = vai[vai < n_orig]
        if len(orig_vai) == 0:
            continue

        X_tr = train_full.iloc[tri].copy()
        X_va = train_full.iloc[orig_vai].copy()
        X_te = test_df.copy()
        y_tr = y_full[tri]
        y_va = y_full[orig_vai]

        X_tr, X_va, X_te = apply_te(X_tr, X_va, X_te, te_cols, y_tr)

        classes = np.unique(y_tr)
        cw = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=y_tr)))
        sw_class = np.array([cw[l] for l in y_tr])
        sw_sample = np.ones(len(tri))
        sw_sample[tri >= n_orig] = pw
        sw = sw_class * sw_sample

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
                  eval_set=(X_va[feat], y_va), early_stopping_rounds=50)
        oof[orig_vai] = model.predict_proba(X_va[feat])
        tp += model.predict_proba(X_te[feat]) / NF

        fs = balanced_accuracy_score(y_va, oof[orig_vai].argmax(1))
        log(f"    {nm} fold {fold+1}: {fs:.5f}")
        del X_tr, X_va, X_te, model
        gc.collect()

    sc = balanced_accuracy_score(y, oof.argmax(1))
    log(f"  >> {nm}: {sc:.5f}")
    return oof, tp

n_orig = len(train)
s2_xgb42_oof, s2_xgb42_tp = train_xgb_pseudo(train_with_pseudo, y_with_pseudo, test,
                                                TE_columns, 42, "s2_xgb42", n_orig, PSEUDO_WEIGHT)
s2_xgb123_oof, s2_xgb123_tp = train_xgb_pseudo(train_with_pseudo, y_with_pseudo, test,
                                                  TE_columns, 123, "s2_xgb123", n_orig, PSEUDO_WEIGHT)
s2_cb42_oof, s2_cb42_tp = train_cb_pseudo(train_with_pseudo, y_with_pseudo, test,
                                            TE_columns, 42, "s2_cb42", n_orig, PSEUDO_WEIGHT)

# ============================================================
# STEP 8: Fast Ensemble — Simple avg + threshold opt
# ============================================================
log("\nSTEP 8: Fast Ensemble...")

# Simple average of Stage 2 predictions
bo = (s2_xgb42_oof + s2_xgb123_oof + s2_cb42_oof) / 3
bt = (s2_xgb42_tp + s2_xgb123_tp + s2_cb42_tp) / 3

s2_avg_ba = balanced_accuracy_score(y, bo.argmax(1))
log(f"  Simple avg OOF: {s2_avg_ba:.5f}")

# Also try weighted: use individual model BAs as weights
xgb42_ba = balanced_accuracy_score(y, s2_xgb42_oof.argmax(1))
xgb123_ba = balanced_accuracy_score(y, s2_xgb123_oof.argmax(1))
cb42_ba = balanced_accuracy_score(y, s2_cb42_oof.argmax(1))
log(f"  Individual: xgb42={xgb42_ba:.5f}, xgb123={xgb123_ba:.5f}, cb42={cb42_ba:.5f}")

# Quick weight search (only 2000 iterations)
bwa = s2_avg_ba
bww = np.array([1/3, 1/3, 1/3])
oofs = [s2_xgb42_oof, s2_xgb123_oof, s2_cb42_oof]
tps = [s2_xgb42_tp, s2_xgb123_tp, s2_cb42_tp]

rng = np.random.RandomState(42)
for _ in range(2000):
    w = rng.dirichlet(np.ones(3))
    combo = sum(w[i] * oofs[i] for i in range(3))
    s = balanced_accuracy_score(y, combo.argmax(1))
    if s > bwa:
        bwa = s
        bww = w.copy()

log(f"  Best weighted OOF: {bwa:.5f} (weights: {bww})")
bo_weighted = sum(bww[i] * oofs[i] for i in range(3))
bt_weighted = sum(bww[i] * tps[i] for i in range(3))

# Use weighted if better
if bwa > s2_avg_ba:
    bo_final = bo_weighted
    bt_final = bt_weighted
    log("  >>> Using weighted average")
else:
    bo_final = bo
    bt_final = bt
    log("  >>> Using simple average")

# ============================================================
# STEP 9: Threshold Optimization
# ============================================================
log("\nSTEP 9: Threshold optimization...")

def neg_ba(w):
    return -balanced_accuracy_score(y, (bo_final * np.array([1.0, w[0], w[1]])).argmax(1))

bg = (1.0, 1.0)
bgs = -1
for wm in np.arange(0.5, 3.0, 0.05):
    for wh in np.arange(0.5, 3.0, 0.1):
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

preds_opt = (bt_final * np.array(bw)).argmax(axis=1)
sub_opt = pd.DataFrame({"id": test_ids, TARGET_COL: [rmap[p] for p in preds_opt]})
path_opt = SUBMISSIONS / "submission_r08_thresh_opt.csv"
sub_opt.to_csv(path_opt, index=False)
log(f"  r08_thresh_opt: {dict(sub_opt[TARGET_COL].value_counts())}")

preds_default = bt_final.argmax(axis=1)
sub_default = pd.DataFrame({"id": test_ids, TARGET_COL: [rmap[p] for p in preds_default]})
path_default = SUBMISSIONS / "submission_r08_ens_default.csv"
sub_default.to_csv(path_default, index=False)
log(f"  r08_ens_default: {dict(sub_default[TARGET_COL].value_counts())}")

# Also save Stage 1 simple avg for comparison
s1_avg_bt = (s1_xgb42_tp + s1_xgb123_tp + s1_cb42_tp) / 3
preds_s1 = s1_avg_bt.argmax(axis=1)
sub_s1 = pd.DataFrame({"id": test_ids, TARGET_COL: [rmap[p] for p in preds_s1]})
path_s1 = SUBMISSIONS / "submission_r08_stage1_avg.csv"
sub_s1.to_csv(path_s1, index=False)
log(f"  r08_stage1_avg: {dict(sub_s1[TARGET_COL].value_counts())}")

# ============================================================
# SUMMARY
# ============================================================
log("\n" + "=" * 60)
log("SUMMARY - Round 08")
log("=" * 60)
log(f"  Pseudo threshold: {PSEUDO_THRESHOLD}")
log(f"  Pseudo samples: {pseudo_mask.sum()} / {len(test)} ({pseudo_mask.mean()*100:.1f}%)")
log(f"  Pseudo weight: {PSEUDO_WEIGHT}x")
log(f"  --- Stage 1 ---")
log(f"    xgb42: {xgb42_ba:.5f}")
log(f"    xgb123: {xgb123_ba:.5f}")
log(f"    cb42: {cb42_ba:.5f}")
s1_avg = balanced_accuracy_score(y, ((s1_xgb42_oof + s1_xgb123_oof + s1_cb42_oof) / 3).argmax(1))
log(f"    Stage 1 avg: {s1_avg:.5f}")
log(f"  --- Stage 2 ---")
s2_xgb42_ba = balanced_accuracy_score(y, s2_xgb42_oof.argmax(1))
s2_xgb123_ba = balanced_accuracy_score(y, s2_xgb123_oof.argmax(1))
s2_cb42_ba = balanced_accuracy_score(y, s2_cb42_oof.argmax(1))
log(f"    xgb42: {s2_xgb42_ba:.5f}")
log(f"    xgb123: {s2_xgb123_ba:.5f}")
log(f"    cb42: {s2_cb42_ba:.5f}")
log(f"  Ensemble OOF: {bwa:.5f}")
log(f"  FINAL CV: {fcv:.5f}")
log(f"  Total: {time.time() - start:.0f}s")

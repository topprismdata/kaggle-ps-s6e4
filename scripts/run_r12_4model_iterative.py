"""Round 12: 4-Model Iterative Pseudo-Labeling (3 rounds)

Based on R09 but optimized:
- Only 4 models (3 XGB + 1 CB), dropped LGB (marginal contribution)
- 3 rounds of iterative pseudo-labeling (thresholds 0.90, 0.85, 0.80)
- Hill climbing ensemble instead of random weight search
- No stacking (avoids CV-LB gap inflation seen in R09)

Pipeline:
1. Load data + pairwise TE (135 cols) + TE_ORIG (19 cols)
2. Stage 1: Train 4 models (3 XGB + 1 CB) without pseudo labels
3. Pseudo-label round 1 (threshold=0.90, weight=0.5x)
4. Stage 2: Train 4 models WITH pseudo-labeled data
5. Pseudo-label round 2 (threshold=0.85, weight=0.3x)
6. Stage 3: Train 4 models WITH more pseudo data
7. Pseudo-label round 3 (threshold=0.80, weight=0.2x)
8. Stage 4: Train 4 models WITH all pseudo data
9. Hill climbing ensemble + threshold optimization
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
log("Round 12: 4-Model Iterative Pseudo-Labeling (3 rounds)")
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

def train_xgb(train_df, y_arr, test_df, te_cols, seed, nm, n_orig=None, pw=None):
    """Train XGB with 5-fold CV. If n_orig/pw given, use pseudo-label weights."""
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
    """Train LGB with 5-fold CV. If n_orig/pw given, use pseudo-label weights."""
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
    """Train CB with 5-fold CV. If n_orig/pw given, use pseudo-label weights."""
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

# ============================================================
# STEP 5: Stage 1 — Train 4 models without pseudo labels
# ============================================================
y = train[TARGET].values

log("\nSTEP 5: Stage 1 — Train without pseudo labels...")
s1_oof = {}
s1_tp = {}

# 5a: XGBoost (3 seeds)
log("  --- XGBoost ---")
for SEED in SEEDS:
    nm = f"s1_xgb_s{SEED}"
    oof, tp = train_xgb(train, y, test, TE_columns, SEED, nm)
    s1_oof[nm] = oof
    s1_tp[nm] = tp

# 5b: CatBoost (1 model)
log("  --- CatBoost ---")
nm = "s1_cb_s42"
oof, tp = train_cb(train, y, test, TE_columns, 42, nm)
s1_oof[nm] = oof
s1_tp[nm] = tp

log(f"\n  Stage 1 models: {len(s1_oof)}")

# ============================================================
# STEP 6-8: Iterative Pseudo-Labeling (3 rounds)
# ============================================================
PSEUDO_ROUNDS = [
    {"threshold": 0.90, "weight": 0.5, "label": "R1"},
    {"threshold": 0.85, "weight": 0.3, "label": "R2"},
    {"threshold": 0.80, "weight": 0.2, "label": "R3"},
]

all_oof = dict(s1_oof)
all_tp = dict(s1_tp)

# Initial test predictions from Stage 1 average
current_test_pred = sum(s1_tp.values()) / len(s1_tp)

for pr_idx, pr in enumerate(PSEUDO_ROUNDS):
    stage_label = pr["label"]
    threshold = pr["threshold"]
    pw = pr["weight"]
    stage_num = pr_idx + 2
    step_num = 5 + stage_num

    log(f"\n{'=' * 60}")
    log(f"STEP {step_num}: Pseudo-Label {stage_label} (threshold={threshold}, weight={pw})")
    log(f"{'=' * 60}")

    # Pseudo-label from current test predictions
    max_probs = current_test_pred.max(1)
    pseudo_mask = max_probs >= threshold
    pseudo_labels = current_test_pred[pseudo_mask].argmax(1)

    log(f"  Total test: {len(test)}, Pseudo-labeled: {pseudo_mask.sum()} ({pseudo_mask.sum()/len(test)*100:.1f}%)")
    for cls_id, cls_name in rmap.items():
        log(f"    {cls_name}: {(pseudo_labels == cls_id).sum()}")

    # Add pseudo-labeled test data to training
    pseudo_test = test[pseudo_mask].copy()
    pseudo_test[TARGET] = pseudo_labels
    pseudo_y = pseudo_labels.copy()

    train_with_pseudo = pd.concat([train, pseudo_test], ignore_index=True)
    y_with_pseudo = np.concatenate([y, pseudo_y])
    N_ORIG = len(train)

    log(f"  Train+Pseudo: {len(train_with_pseudo)} rows")

    # Train models for this stage
    stage_prefix = f"s{stage_num}"
    s_oof = {}
    s_tp = {}

    log("  --- XGBoost ---")
    for SEED in SEEDS:
        nm = f"{stage_prefix}_xgb_s{SEED}"
        oof, tp = train_xgb(train_with_pseudo, y_with_pseudo, test, TE_columns,
                             SEED, nm, n_orig=N_ORIG, pw=pw)
        s_oof[nm] = oof
        s_tp[nm] = tp

    log("  --- CatBoost ---")
    nm = f"{stage_prefix}_cb_s42"
    oof, tp = train_cb(train_with_pseudo, y_with_pseudo, test, TE_columns,
                        42, nm, n_orig=N_ORIG, pw=pw)
    s_oof[nm] = oof
    s_tp[nm] = tp

    # Update current test predictions for next round
    s_names = list(s_tp.keys())
    current_test_pred = sum(s_tp[n] for n in s_names) / len(s_names)

    # Add to all models
    all_oof.update(s_oof)
    all_tp.update(s_tp)

    log(f"  Stage {stage_num} models: {len(s_oof)}")

# ============================================================
# STEP 9: Hill Climbing Ensemble
# ============================================================
all_names = list(all_oof.keys())
log(f"\nSTEP 9: Hill Climbing Ensemble ({len(all_names)} models)...")

# Individual scores
for n in all_names:
    s = balanced_accuracy_score(y, all_oof[n].argmax(1))
    log(f"    {n}: {s:.5f}")

# Simple average of all models
simple_avg_oof = sum(all_oof[n] for n in all_names) / len(all_names)
sa_score = balanced_accuracy_score(y, simple_avg_oof.argmax(1))
log(f"  Simple avg OOF: {sa_score:.5f}")

# Hill climbing ensemble
def hill_climbing(y_true, oof_dict, model_names, n_repeats=5):
    best_score = 0
    best_w = None
    best_selected = None
    for r in range(n_repeats):
        remaining = list(model_names)
        selected = []
        weights = {}
        # Start with best single model
        best_single = max(remaining, key=lambda n: balanced_accuracy_score(
            y_true, oof_dict[n].argmax(1)))
        selected.append(best_single)
        weights[best_single] = 1.0
        remaining.remove(best_single)
        current_pred = oof_dict[best_single].copy()
        current_score = balanced_accuracy_score(y_true, current_pred.argmax(1))

        while remaining:
            best_addition = None
            best_new_score = current_score
            best_alpha = 0
            for cand in remaining:
                for alpha in [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3]:
                    new_pred = (1 - alpha) * current_pred + alpha * oof_dict[cand]
                    new_score = balanced_accuracy_score(y_true, new_pred.argmax(1))
                    if new_score > best_new_score:
                        best_new_score = new_score
                        best_addition = cand
                        best_alpha = alpha
            if best_addition is None:
                break
            current_pred = (1 - best_alpha) * current_pred + best_alpha * oof_dict[best_addition]
            for n in weights:
                weights[n] *= (1 - best_alpha)
            weights[best_addition] = best_alpha
            selected.append(best_addition)
            remaining.remove(best_addition)
            current_score = best_new_score

        if current_score > best_score:
            best_score = current_score
            best_w = weights
            best_selected = selected

    return best_w, best_selected, best_score

hc_w, hc_selected, hc_score = hill_climbing(y, all_oof, all_names)
log(f"  Hill climbing: {hc_score:.5f} with {len(hc_selected)} models")
log(f"    Selected: {hc_selected}")

# Choose best ensemble
if hc_score >= sa_score:
    log("  >>> Using Hill Climbing")
    bo = sum(hc_w[n] * all_oof[n] for n in hc_selected)
    bt = sum(hc_w[n] * all_tp[n] for n in hc_selected)
else:
    log("  >>> Using Simple Average")
    bo = simple_avg_oof
    bt = sum(all_tp[n] for n in all_names) / len(all_names)

# ============================================================
# STEP 10: Threshold Optimization
# ============================================================
log("\nSTEP 10: Threshold optimization...")

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
# STEP 11: Save submissions
# ============================================================
log("\nSTEP 11: Save submissions...")

# Threshold optimized
preds_thresh = (bt * np.array(bw)).argmax(1)
sub = pd.DataFrame({ID_COL: test_ids, TARGET_COL: [rmap[p] for p in preds_thresh]})
sub.to_csv(SUBMISSIONS / "submission_r12_thresh_opt.csv", index=False)
dist = pd.Series([rmap[p] for p in preds_thresh]).value_counts()
log(f"  r12_thresh_opt: {dict(dist)}")

# Default (no threshold opt)
preds_default = bt.argmax(1)
sub2 = pd.DataFrame({ID_COL: test_ids, TARGET_COL: [rmap[p] for p in preds_default]})
sub2.to_csv(SUBMISSIONS / "submission_r12_ens_default.csv", index=False)
dist2 = pd.Series([rmap[p] for p in preds_default]).value_counts()
log(f"  r12_ens_default: {dict(dist2)}")

# Simple avg of all stages
preds_simple = (sum(all_tp[n] for n in all_names) / len(all_names)).argmax(1)
sub3 = pd.DataFrame({ID_COL: test_ids, TARGET_COL: [rmap[p] for p in preds_simple]})
sub3.to_csv(SUBMISSIONS / "submission_r12_simple_avg.csv", index=False)
dist3 = pd.Series([rmap[p] for p in preds_simple]).value_counts()
log(f"  r12_simple_avg: {dict(dist3)}")

elapsed = int(time.time() - start)

# ============================================================
# SUMMARY
# ============================================================
log("\n" + "=" * 60)
log(f"SUMMARY - Round 12")
log("=" * 60)
for pr in PSEUDO_ROUNDS:
    log(f"  Pseudo {pr['label']}: threshold={pr['threshold']}, weight={pr['weight']}")
log(f"  Total models: {len(all_oof)}")
for stage_prefix in ["s1", "s2", "s3", "s4"]:
    stage_models = [n for n in all_names if n.startswith(stage_prefix + "_")]
    if stage_models:
        stage_avg_oof = sum(all_oof[n] for n in stage_models) / len(stage_models)
        stage_score = balanced_accuracy_score(y, stage_avg_oof.argmax(1))
        log(f"  {stage_prefix} avg: {stage_score:.5f} ({len(stage_models)} models)")
log(f"  Simple avg OOF: {sa_score:.5f}")
log(f"  Hill climbing OOF: {hc_score:.5f}")
log(f"  FINAL CV: {fcv:.5f}")
log(f"  Total: {elapsed}s")

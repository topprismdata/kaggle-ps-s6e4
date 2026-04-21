"""Round 07: Pseudo-Labeling + Diverse Ensemble

Based on research: Pseudo-Labeling is the #1 missing technique.
Public notebook 0.97959 uses XGB + Pseudo Labels.

Pipeline:
1. Same 135 pairwise TE + TE_ORIG features as R05
2. Train 4 models (3 XGB + 1 CB) without pseudo labels -> get test predictions
3. Add high-confidence test predictions (prob > threshold) as pseudo-labeled data
4. Retrain 4 models with pseudo-labeled data added
5. Stacking + threshold optimization

Time budget: ~5 hours
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
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (SUBMISSIONS, TARGET_COL, ID_COL, CLASSES,
                        CATEGORICAL_COLS, NUMERICAL_COLS)

def log(msg=""):
    print(msg, flush=True)

start = time.time()
log("=" * 60)
log("Round 07: Pseudo-Labeling + Diverse Ensemble")
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
SEEDS = [42, 123, 456]
NF = 5

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
# STEP 3: All-pairwise interactions
# ============================================================
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

# ============================================================
# STEP 4: TE_ORIG
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

def train_models(train_df, y_arr, test_df, te_cols, model_configs, label=""):
    """Train multiple models and return OOF + test predictions."""
    all_oof = {}
    all_tp = {}

    for mname, mtype, mparams, seed in model_configs:
        log(f"  Training {mname}...")
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

            if mtype == "xgb":
                for c in CATS:
                    for df_ in [X_tr, X_va, X_te]:
                        df_[c] = df_[c].astype(int)
                model = xgb.XGBClassifier(
                    **mparams, objective="multi:softprob",
                    callbacks=[xgb.callback.EarlyStopping(rounds=100, metric_name="bal_ACC",
                                                           maximize=True, save_best=True)],
                    eval_metric=make_bal_acc(),
                    random_state=seed, n_jobs=-1, tree_method="hist"
                )
                model.fit(X_tr[feat], y_arr[tri], eval_set=[(X_va[feat], y_arr[vai])],
                          sample_weight=sw, verbose=False)

            elif mtype == "cb":
                import catboost as cb_mod
                for c in CATS:
                    for df_ in [X_tr, X_va, X_te]:
                        df_[c] = df_[c].astype(str).astype("category")
                model = cb_mod.CatBoostClassifier(
                    **mparams, task_type="CPU", auto_class_weights="Balanced",
                    cat_features=CATS, verbose=0, random_seed=seed
                )
                model.fit(X_tr[feat], y_arr[tri], sample_weight=sw,
                          eval_set=(X_va[feat], y_arr[vai]), early_stopping_rounds=50)

            oof[vai] = model.predict_proba(X_va[feat])
            tp += model.predict_proba(X_te[feat]) / NF

            fs = balanced_accuracy_score(y_arr[vai], oof[vai].argmax(1))
            log(f"    {mname} fold {fold+1}: {fs:.5f}")
            del X_tr, X_va, X_te, model
            gc.collect()

        sc = balanced_accuracy_score(y_arr, oof.argmax(1))
        log(f"  >> {mname}: {sc:.5f}")
        all_oof[mname] = oof
        all_tp[mname] = tp

    return all_oof, all_tp

# ============================================================
# STEP 5: Stage 1 — Train without pseudo labels
# ============================================================
log(f"\nSTEP 5: Stage 1 — Train without pseudo labels...")
y = train[TARGET].values

stage1_configs = [
    ("xgb_s42", "xgb", {"max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8,
                         "n_estimators": 5000, "learning_rate": 0.03, "max_bin": 1024}, 42),
    ("xgb_s123", "xgb", {"max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8,
                          "n_estimators": 5000, "learning_rate": 0.03, "max_bin": 1024}, 123),
    ("cat_s42", "cb", {"iterations": 800, "learning_rate": 0.05, "depth": 6,
                       "colsample_bylevel": 0.8, "l2_leaf_reg": 3.0, "min_data_in_leaf": 50}, 42),
]

s1_oof, s1_tp = train_models(train, y, test, TE_columns, stage1_configs, label="Stage1")

# ============================================================
# STEP 6: Pseudo-Labeling
# ============================================================
log("\nSTEP 6: Pseudo-Labeling...")

# Average test predictions from all stage 1 models
s1_names = list(s1_tp.keys())
test_proba_avg = np.mean([s1_tp[n] for n in s1_names], axis=0)
test_pred_labels = test_proba_avg.argmax(axis=1)
test_pred_conf = test_proba_avg.max(axis=1)

# Select high-confidence samples
PSEUDO_THRESHOLD = 0.90
pseudo_mask = test_pred_conf >= PSEUDO_THRESHOLD

log(f"  Total test samples: {len(test)}")
log(f"  Pseudo-labeled (conf >= {PSEUDO_THRESHOLD}): {pseudo_mask.sum()} ({pseudo_mask.mean()*100:.1f}%)")

# Class distribution of pseudo-labeled samples
pseudo_labels = test_pred_labels[pseudo_mask]
for cls_id, cls_name in rmap.items():
    cnt = (pseudo_labels == cls_id).sum()
    log(f"    {cls_name}: {cnt}")

# Create pseudo-labeled test DataFrame
pseudo_test = test[pseudo_mask].copy()
pseudo_test[TARGET] = test_pred_labels[pseudo_mask]
pseudo_y = test_pred_labels[pseudo_mask]

# ============================================================
# STEP 7: Stage 2 — Retrain with pseudo-labeled data
# ============================================================
log(f"\nSTEP 7: Stage 2 — Train WITH pseudo-labeled data...")
log(f"  Train: {len(train)}, Pseudo: {len(pseudo_test)}, Total: {len(train) + len(pseudo_test)}")

# Combine train + pseudo-labeled data
train_with_pseudo = pd.concat([train, pseudo_test], ignore_index=True)
y_with_pseudo = np.concatenate([y, pseudo_y])

# Pseudo samples get 0.5x weight (lower than real data)
PSEUDO_WEIGHT = 0.5

s2_configs = [
    ("xgb_s42", "xgb", {"max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8,
                         "n_estimators": 5000, "learning_rate": 0.03, "max_bin": 1024}, 42),
    ("xgb_s123", "xgb", {"max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8,
                          "n_estimators": 5000, "learning_rate": 0.03, "max_bin": 1024}, 123),
    ("cat_s42", "cb", {"iterations": 800, "learning_rate": 0.05, "depth": 6,
                       "colsample_bylevel": 0.8, "l2_leaf_reg": 3.0, "min_data_in_leaf": 50}, 42),
]

# Train stage 2 models — OOF only on original train portion
s2_oof = {}
s2_tp = {}

for mname, mtype, mparams, seed in s2_configs:
    log(f"  Training {mname} (stage 2)...")
    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=seed)
    oof = np.zeros((len(train), 3))  # Only original train OOF
    tp = np.zeros((len(test), 3))

    for fold, (tri_full, vai_full) in enumerate(skf.split(train_with_pseudo, y_with_pseudo)):
        # Find which indices in vai_full are original train (first len(train) indices)
        orig_mask = vai_full < len(train)
        vai = vai_full[orig_mask]

        X_tr = train_with_pseudo.iloc[tri_full].copy()
        X_va = train_with_pseudo.iloc[vai].copy()
        X_te = test.copy()

        y_tr = y_with_pseudo[tri_full]
        y_va = y_with_pseudo[vai]

        X_tr, X_va, X_te = apply_te(X_tr, X_va, X_te, TE_columns, y_tr)

        classes = np.unique(y_tr)
        cw = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=y_tr)))
        sw_class = np.array([cw[l] for l in y_tr])

        # Pseudo samples get lower weight
        is_pseudo = tri_full >= len(train)
        sw_sample = np.ones(len(tri_full))
        sw_sample[is_pseudo] = PSEUDO_WEIGHT
        sw = sw_class * sw_sample

        feat = [c for c in X_tr.columns if c != TARGET]

        if mtype == "xgb":
            for c in CATS:
                for df_ in [X_tr, X_va, X_te]:
                    df_[c] = df_[c].astype(int)
            model = xgb.XGBClassifier(
                **mparams, objective="multi:softprob",
                callbacks=[xgb.callback.EarlyStopping(rounds=100, metric_name="bal_ACC",
                                                       maximize=True, save_best=True)],
                eval_metric=make_bal_acc(),
                random_state=seed, n_jobs=-1, tree_method="hist"
            )
            model.fit(X_tr[feat], y_tr, eval_set=[(X_va[feat], y_va)],
                      sample_weight=sw, verbose=False)

        elif mtype == "cb":
            import catboost as cb_mod
            for c in CATS:
                for df_ in [X_tr, X_va, X_te]:
                    df_[c] = df_[c].astype(str).astype("category")
            model = cb_mod.CatBoostClassifier(
                **mparams, task_type="CPU", auto_class_weights="Balanced",
                cat_features=CATS, verbose=0, random_seed=seed
            )
            model.fit(X_tr[feat], y_tr, sample_weight=sw,
                      eval_set=(X_va[feat], y_va), early_stopping_rounds=50)

        oof[vai] = model.predict_proba(X_va[feat])
        tp += model.predict_proba(X_te[feat]) / NF

        fs = balanced_accuracy_score(y_va, oof[vai].argmax(1))
        log(f"    {mname} fold {fold+1}: {fs:.5f} (val on {len(vai)} orig samples)")
        del X_tr, X_va, X_te, model
        gc.collect()

    sc = balanced_accuracy_score(y, oof.argmax(1))
    log(f"  >> {mname}: {sc:.5f}")
    s2_oof[mname] = oof
    s2_tp[mname] = tp

# ============================================================
# STEP 8: Ensemble Stage 2
# ============================================================
log(f"\nSTEP 8: Ensemble...")
names = list(s2_oof.keys())

# Weighted average search
bwa = 0
bww = None
rng = np.random.RandomState(42)
for _ in range(30000):
    w = rng.dirichlet(np.ones(len(names)))
    s = balanced_accuracy_score(y, sum(w[i]*s2_oof[n] for i, n in enumerate(names)).argmax(1))
    if s > bwa:
        bwa = s
        bww = w.copy()
log(f"  Weighted avg OOF: {bwa:.5f}")

# Stacking
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

if ssc >= bwa:
    log("  >>> Using Stacking")
    fm = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, random_state=42)
    fm.fit(ostk, y)
    bt = fm.predict_proba(tstk)
    bo = np.zeros((len(y), 3))
    for fold, (tri, vai) in enumerate(skf.split(ostk, y)):
        lr = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, random_state=42)
        lr.fit(ostk[tri], y[tri])
        bo[vai] = lr.predict_proba(ostk[vai])
else:
    log("  >>> Using Weighted avg")
    bt = sum(bww[i]*s2_tp[n] for i, n in enumerate(names))
    bo = sum(bww[i]*s2_oof[n] for i, n in enumerate(names))

# ============================================================
# STEP 9: Threshold Optimization
# ============================================================
log("\nSTEP 9: Threshold optimization...")

def neg_ba(w):
    return -balanced_accuracy_score(y, (bo * np.array([1.0, w[0], w[1]])).argmax(1))

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

preds_opt = (bt * np.array(bw)).argmax(axis=1)
sub_opt = pd.DataFrame({"id": test_ids, TARGET_COL: [rmap[p] for p in preds_opt]})
path_opt = SUBMISSIONS / "submission_r07_thresh_opt.csv"
sub_opt.to_csv(path_opt, index=False)
log(f"  r07_thresh_opt: {dict(sub_opt[TARGET_COL].value_counts())}")

preds_default = bt.argmax(axis=1)
sub_default = pd.DataFrame({"id": test_ids, TARGET_COL: [rmap[p] for p in preds_default]})
path_default = SUBMISSIONS / "submission_r07_ens_default.csv"
sub_default.to_csv(path_default, index=False)
log(f"  r07_ens_default: {dict(sub_default[TARGET_COL].value_counts())}")

# Also save stage 1 ensemble for comparison
log("\n  Also saving stage 1 predictions for comparison...")
s1_ostk = np.hstack([s1_oof[n] for n in s1_names])
s1_tstk = np.hstack([s1_tp[n] for n in s1_names])
fm1 = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, random_state=42)
fm1.fit(s1_ostk, y)
bt_s1 = fm1.predict_proba(s1_tstk)
preds_s1 = bt_s1.argmax(axis=1)
sub_s1 = pd.DataFrame({"id": test_ids, TARGET_COL: [rmap[p] for p in preds_s1]})
path_s1 = SUBMISSIONS / "submission_r07_stage1_ens.csv"
sub_s1.to_csv(path_s1, index=False)
log(f"  r07_stage1_ens: {dict(sub_s1[TARGET_COL].value_counts())}")

# ============================================================
# SUMMARY
# ============================================================
log("\n" + "=" * 60)
log("SUMMARY - Round 07 Pseudo-Labeling")
log("=" * 60)
log(f"  Pseudo threshold: {PSEUDO_THRESHOLD}")
log(f"  Pseudo samples: {pseudo_mask.sum()} / {len(test)} ({pseudo_mask.mean()*100:.1f}%)")
log(f"  Pseudo weight: {PSEUDO_WEIGHT}x")
log(f"  Pairwise TE features: {len(TE_columns)}")
log(f"  Models per stage: {len(s2_oof)}")
log(f"  --- Stage 1 (no pseudo) ---")
for nm in s1_names:
    sc = balanced_accuracy_score(y, s1_oof[nm].argmax(1))
    log(f"    {nm}: {sc:.5f}")
log(f"  --- Stage 2 (with pseudo) ---")
for nm in names:
    sc = balanced_accuracy_score(y, s2_oof[nm].argmax(1))
    log(f"    {nm}: {sc:.5f}")
log(f"  Weighted avg OOF: {bwa:.5f}")
log(f"  Stacked OOF: {ssc:.5f}")
log(f"  FINAL CV (with threshold opt): {fcv:.5f}")
log(f"  Total: {time.time() - start:.0f}s")

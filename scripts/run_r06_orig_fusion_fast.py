"""Round 06 Fast: R05 + Original Data Fusion

Key improvements over R05 (LB=0.97765):
1. Orig data fusion with weighted sample_weight (20x)
2. Same 135 pairwise TE features as R05 (no extra KBins to keep runtime down)
3. Same TE_ORIG features
4. Reduced models: 3 XGB + 1 CB = 4 models (removed LGB, minimal contribution)
5. Stacking + Threshold optimization (constrained range [0.5, 3.0])

Target runtime: ~3-4 hours (vs R05's 7.4h with 10 models)
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
log("Round 06 Fast: R05 + Orig Fusion")
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
ORIG_WEIGHT = 20

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
# STEP 3: All-pairwise interactions (same as R05)
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
# STEP 5: Training with orig fusion
# ============================================================
log(f"\nSTEP 5: Training ({NF} folds, orig fusion weight={ORIG_WEIGHT}x)...")
y = train[TARGET].values
orig_y = orig[TARGET].values

all_oof = {}
all_tp = {}

def apply_te(X_tr, X_va, X_te, te_cols, y_tr):
    """Apply TargetEncoding within fold using sklearn's TargetEncoder."""
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

# === 5a. XGBoost (3 seeds) ===
log("\n  --- XGBoost (3 seeds) ---")
for SEED in SEEDS:
    nm = f"xgb_s{SEED}"
    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=SEED)
    oof = np.zeros((len(train), 3))
    tp = np.zeros((len(test), 3))

    for fold, (tri, vai) in enumerate(skf.split(train, y)):
        # Fuse orig data into training fold
        X_tr = pd.concat([train.iloc[tri], orig], ignore_index=True)
        X_va = train.iloc[vai].copy()
        X_te = test.copy()

        y_tr = np.concatenate([y[tri], orig_y])
        orig_w = np.full(len(orig), ORIG_WEIGHT)
        train_w = np.ones(len(tri))
        sample_w = np.concatenate([train_w, orig_w])

        X_tr, X_va, X_te = apply_te(X_tr, X_va, X_te, TE_columns, y_tr)

        classes = np.unique(y_tr)
        cw = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=y_tr)))
        sw_class = np.array([cw[l] for l in y_tr])
        sw = sample_w * sw_class

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
            max_bin=1024, random_state=SEED, n_jobs=-1, tree_method="hist"
        )
        model.fit(X_tr[feat], y_tr, eval_set=[(X_va[feat], y[vai])],
                  sample_weight=sw, verbose=False)
        oof[vai] = model.predict_proba(X_va[feat])
        tp += model.predict_proba(X_te[feat]) / NF

        fs = balanced_accuracy_score(y[vai], oof[vai].argmax(1))
        log(f"    {nm} fold {fold+1}: {fs:.5f}")
        del X_tr, X_va, X_te, model
        gc.collect()

    sc = balanced_accuracy_score(y, oof.argmax(1))
    log(f"  >> {nm}: {sc:.5f}")
    all_oof[nm] = oof
    all_tp[nm] = tp

# === 5b. CatBoost (1 seed) ===
log("\n  --- CatBoost ---")
try:
    import catboost as cb_mod
    for SEED in [42]:
        nm = f"cat_s{SEED}"
        skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=SEED)
        oof = np.zeros((len(train), 3))
        tp = np.zeros((len(test), 3))

        for fold, (tri, vai) in enumerate(skf.split(train, y)):
            X_tr = pd.concat([train.iloc[tri], orig], ignore_index=True)
            X_va = train.iloc[vai].copy()
            X_te = test.copy()

            y_tr = np.concatenate([y[tri], orig_y])
            orig_w = np.full(len(orig), ORIG_WEIGHT)
            train_w = np.ones(len(tri))
            sample_w = np.concatenate([train_w, orig_w])

            X_tr, X_va, X_te = apply_te(X_tr, X_va, X_te, TE_columns, y_tr)

            classes = np.unique(y_tr)
            cw = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=y_tr)))
            sw_class = np.array([cw[l] for l in y_tr])
            sw = sample_w * sw_class

            feat = [c for c in X_tr.columns if c != TARGET]
            for c in CATS:
                for df_ in [X_tr, X_va, X_te]:
                    df_[c] = df_[c].astype(str).astype("category")

            model = cb_mod.CatBoostClassifier(
                task_type="CPU", iterations=800, learning_rate=0.05, depth=6,
                auto_class_weights="Balanced", cat_features=CATS, verbose=0,
                colsample_bylevel=0.8, l2_leaf_reg=3.0, min_data_in_leaf=50,
                random_seed=SEED
            )
            model.fit(X_tr[feat], y_tr, sample_weight=sw,
                      eval_set=(X_va[feat], y[vai]), early_stopping_rounds=50)
            oof[vai] = model.predict_proba(X_va[feat])
            tp += model.predict_proba(X_te[feat]) / NF

            fs = balanced_accuracy_score(y[vai], oof[vai].argmax(1))
            log(f"    {nm} fold {fold+1}: {fs:.5f}")
            del X_tr, X_va, X_te, model
            gc.collect()

        sc = balanced_accuracy_score(y, oof.argmax(1))
        log(f"  >> {nm}: {sc:.5f}")
        all_oof[nm] = oof
        all_tp[nm] = tp
except ImportError:
    log("  CatBoost not available, skipping")

# ============================================================
# STEP 6: Ensemble
# ============================================================
log(f"\nSTEP 6: Ensemble ({len(all_oof)} models)...")
names = list(all_oof.keys())

# Weighted average search
bwa = 0
bww = None
rng = np.random.RandomState(42)
for _ in range(30000):
    w = rng.dirichlet(np.ones(len(names)))
    s = balanced_accuracy_score(y, sum(w[i]*all_oof[n] for i, n in enumerate(names)).argmax(1))
    if s > bwa:
        bwa = s
        bww = w.copy()
log(f"  Weighted avg OOF: {bwa:.5f}")

# Stacking
ostk = np.hstack([all_oof[n] for n in names])
tstk = np.hstack([all_tp[n] for n in names])

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
    bt = sum(bww[i]*all_tp[n] for i, n in enumerate(names))
    bo = sum(bww[i]*all_oof[n] for i, n in enumerate(names))

# ============================================================
# STEP 7: Threshold Optimization (constrained [0.5, 3.0])
# ============================================================
log("\nSTEP 7: Threshold optimization (range [0.5, 3.0])...")

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
# STEP 8: Save submissions
# ============================================================
log("\nSTEP 8: Save submissions...")

preds_opt = (bt * np.array(bw)).argmax(axis=1)
sub_opt = pd.DataFrame({"id": test_ids, TARGET_COL: [rmap[p] for p in preds_opt]})
path_opt = SUBMISSIONS / "submission_r06_thresh_opt.csv"
sub_opt.to_csv(path_opt, index=False)
log(f"  r06_thresh_opt: {dict(sub_opt[TARGET_COL].value_counts())}")

preds_default = bt.argmax(axis=1)
sub_default = pd.DataFrame({"id": test_ids, TARGET_COL: [rmap[p] for p in preds_default]})
path_default = SUBMISSIONS / "submission_r06_ens_default.csv"
sub_default.to_csv(path_default, index=False)
log(f"  r06_ens_default: {dict(sub_default[TARGET_COL].value_counts())}")

# ============================================================
# SUMMARY
# ============================================================
log("\n" + "=" * 60)
log("SUMMARY - Round 06 Fast")
log("=" * 60)
log(f"  Orig data fusion: {len(orig)} rows x {ORIG_WEIGHT}x weight")
log(f"  Pairwise TE features: {len(TE_columns)}")
log(f"  Total features: {len(FEATURES)} + {len(TE_columns) * 3} TE = {len(FEATURES) + len(TE_columns) * 3}")
log(f"  Models: {len(all_oof)}")
for nm in names:
    sc = balanced_accuracy_score(y, all_oof[nm].argmax(1))
    log(f"    {nm}: {sc:.5f}")
log(f"  Weighted avg OOF: {bwa:.5f}")
log(f"  Stacked OOF: {ssc:.5f}")
log(f"  FINAL CV (with threshold opt): {fcv:.5f}")
log(f"  Total: {time.time() - start:.0f}s")

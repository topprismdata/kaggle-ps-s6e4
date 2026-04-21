"""Round 06: Original Data Fusion + DT-on-orig Features

Key improvements over R05 (LB=0.97765):
1. Orig data fusion (10000 rows, weight 20x)
2. DT-on-orig prediction features (3-class probabilities from DecisionTreeClassifier)
3. LR-on-orig prediction features (3-class probabilities from LogisticRegression)
4. KBinsDiscretizer on 11 numerical features + pairwise TE with categories
5. All-pairwise TE (same as R05, ~135 pairs after cardinality filter)
6. TE_ORIG features (orig data groupby mean)
7. 3 XGB + 1 CB = 4 models (removed LGB, minimal contribution)
8. XGB uses callbacks for early stopping (not deprecated parameter)
9. Threshold optimization range [0.5, 3.0] to prevent overfitting
10. Adversarial validation at end for train/test drift detection
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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import TargetEncoder, KBinsDiscretizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (SUBMISSIONS, TARGET_COL, ID_COL, CLASSES,
                        CATEGORICAL_COLS, NUMERICAL_COLS)

def log(msg=""):
    print(msg, flush=True)

start = time.time()
log("=" * 60)
log("Round 06: Orig Fusion + DT-on-orig Features")
log("=" * 60)

# ============================================================
# STEP 1: Load data
# ============================================================
log("\nSTEP 1: Load data")
DATA = Path(__file__).resolve().parent.parent / "data" / "raw"
train = pd.read_csv(DATA / "train.csv", index_col="id")
test = pd.read_csv(DATA / "test.csv", index_col="id")
orig = pd.read_csv(DATA / "irrigation_prediction.csv")
# Read test with original id column for submission
test_ids = pd.read_csv(DATA / "test.csv")[ID_COL].values

TARGET = TARGET_COL
tmap = {"Low": 0, "Medium": 1, "High": 2}
rmap = {0: "Low", 1: "Medium", 2: "High"}
train[TARGET] = train[TARGET].map(tmap)
orig[TARGET] = orig[TARGET].map(tmap)
log(f"  Train: {train.shape}, Test: {test.shape}, Orig: {orig.shape}")
log(f"  Train target distribution: {dict(train[TARGET].value_counts().sort_index())}")
log(f"  Orig target distribution: {dict(orig[TARGET].value_counts().sort_index())}")

NUMS = NUMERICAL_COLS
CATS = CATEGORICAL_COLS
SEEDS = [42, 123, 456]
NF = 5

# ============================================================
# STEP 2: Factorize categoricals across all datasets
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
# STEP 3: DT-on-orig and LR-on-orig prediction features
# ============================================================
log("\nSTEP 3: Creating DT/LR-on-orig prediction features...")
# Train DecisionTree on orig data, predict on train and test
orig_X = orig[NUMS + CATS].copy()
orig_y = orig[TARGET].values

# Ensure categorical columns are integer for sklearn
for c in CATS:
    orig_X[c] = orig_X[c].astype(int)

# DecisionTree with reasonable depth
dt_model = DecisionTreeClassifier(
    max_depth=12, min_samples_leaf=20, random_state=42,
    class_weight="balanced"
)
dt_model.fit(orig_X, orig_y)

# Prepare train/test for prediction
train_for_pred = train[NUMS + CATS].copy()
test_for_pred = test[NUMS + CATS].copy()
for c in CATS:
    train_for_pred[c] = train_for_pred[c].astype(int)
    test_for_pred[c] = test_for_pred[c].astype(int)

dt_train_proba = dt_model.predict_proba(train_for_pred)
dt_test_proba = dt_model.predict_proba(test_for_pred)

for i in range(3):
    train[f"DT_orig_p{i}"] = dt_train_proba[:, i]
    test[f"DT_orig_p{i}"] = dt_test_proba[:, i]

log(f"  DT train accuracy on orig: {dt_model.score(orig_X, orig_y):.5f}")

# LogisticRegression on orig data
lr_model = LogisticRegression(
    class_weight="balanced", max_iter=2000, C=1.0, random_state=42
)
lr_model.fit(orig_X, orig_y)
lr_train_proba = lr_model.predict_proba(train_for_pred)
lr_test_proba = lr_model.predict_proba(test_for_pred)

for i in range(3):
    train[f"LR_orig_p{i}"] = lr_train_proba[:, i]
    test[f"LR_orig_p{i}"] = lr_test_proba[:, i]

log(f"  LR train accuracy on orig: {lr_model.score(orig_X, orig_y):.5f}")
log(f"  Added 6 DT/LR probability features (3 DT + 3 LR)")

del orig_X, train_for_pred, test_for_pred, dt_model, lr_model
gc.collect()

# ============================================================
# STEP 4: KBinsDiscretizer + pairwise TE with categories
# ============================================================
log("\nSTEP 4: KBinsDiscretizer features + pairwise TE with categories...")
kbd = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile",
                       subsample=None, random_state=42)

# Fit on combined numerical data to ensure consistent bins
all_nums_combined = pd.concat([
    train[NUMS].copy(),
    test[NUMS].copy(),
    orig[NUMS].copy()
], ignore_index=True)
kbd.fit(all_nums_combined)
del all_nums_combined

# Transform each dataset
binned_col_names = [f"{c}_bin" for c in NUMS]
train_binned = pd.DataFrame(
    kbd.transform(train[NUMS]).astype(int),
    columns=binned_col_names, index=train.index
)
test_binned = pd.DataFrame(
    kbd.transform(test[NUMS]).astype(int),
    columns=binned_col_names, index=test.index
)
orig_binned = pd.DataFrame(
    kbd.transform(orig[NUMS]).astype(int),
    columns=binned_col_names, index=orig.index
)

train = pd.concat([train, train_binned], axis=1)
test = pd.concat([test, test_binned], axis=1)
orig = pd.concat([orig, orig_binned], axis=1)
del train_binned, test_binned, orig_binned
gc.collect()

log(f"  Added {len(binned_col_names)} binned numerical features")

# Pairwise TE: binned_numerical x category
kbd_te_columns = []
for num_col in binned_col_names:
    for cat_col in CATS:
        name = f"{num_col}-{cat_col}"
        train[name] = train[num_col].astype(str) + "_" + train[cat_col].astype(str)
        test[name] = test[num_col].astype(str) + "_" + test[cat_col].astype(str)
        orig[name] = orig[num_col].astype(str) + "_" + orig[cat_col].astype(str)

        # Factorize across all datasets
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
        kbd_te_columns.append(name)

log(f"  Added {len(kbd_te_columns)} KBins pairwise TE features")

# ============================================================
# STEP 5: All-pairwise interactions (same as R05)
# ============================================================
log("\nSTEP 5: Creating all-pairwise interaction features...")
TE_columns = []
columns = NUMS + CATS
total_pairs = len(list(combinations(columns, 2)))
kept = 0

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
    kept += 1

log(f"  Created {len(TE_columns)} pairwise TE features (from {total_pairs} total pairs)")

# Combine all TE columns: pairwise + KBins pairwise
ALL_TE_COLUMNS = TE_columns + kbd_te_columns
log(f"  Total TE columns: {len(ALL_TE_COLUMNS)} ({len(TE_columns)} pairwise + {len(kbd_te_columns)} KBins)")

# ============================================================
# STEP 6: TE_ORIG (orig data groupby mean)
# ============================================================
log("\nSTEP 6: Computing TE_ORIG features...")
for c in CATS + NUMS:
    tmp = orig.groupby(c, observed=True)[TARGET].mean().astype("float32")
    tmp.name = f"TE_ORIG_{c}"
    train = train.merge(tmp, on=c, how="left")
    train[tmp.name] = train[tmp.name].fillna(0.5)
    test = test.merge(tmp, on=c, how="left")
    test[tmp.name] = test[tmp.name].fillna(0.5)

FEATURES = [c for c in train.columns if c != TARGET]
log(f"  Total features before TE encoding: {len(FEATURES)}")

# ============================================================
# STEP 7: Orig data fusion with weight
# ============================================================
log("\nSTEP 7: Orig data fusion...")
ORIG_WEIGHT = 20  # Each orig row counts as 20x a train row

# Add DT/LR probability features to orig data.
# Since DT and LR were trained on orig, we retrain and use training predictions.
log("  Retraining DT/LR on orig for orig dataset features...")
orig_model_cols = NUMS + CATS
orig_for_model = orig[orig_model_cols].copy()
for c in CATS:
    orig_for_model[c] = orig_for_model[c].astype(int)

dt2 = DecisionTreeClassifier(max_depth=12, min_samples_leaf=20,
                             random_state=42, class_weight="balanced")
dt2.fit(orig_for_model[orig_model_cols], orig[TARGET])
dt_orig_proba = dt2.predict_proba(orig_for_model[orig_model_cols])

lr2 = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, random_state=42)
lr2.fit(orig_for_model[orig_model_cols], orig[TARGET])
lr_orig_proba = lr2.predict_proba(orig_for_model[orig_model_cols])

for i in range(3):
    orig[f"DT_orig_p{i}"] = dt_orig_proba[:, i]
    orig[f"LR_orig_p{i}"] = lr_orig_proba[:, i]

del orig_for_model, dt2, lr2, dt_orig_proba, lr_orig_proba
gc.collect()

log(f"  Orig rows: {len(orig)}, each weighted {ORIG_WEIGHT}x")
log(f"  Effective orig weight: {len(orig) * ORIG_WEIGHT} ({ORIG_WEIGHT}x boost)")

# ============================================================
# STEP 8: Training with orig fusion
# ============================================================
log(f"\nSTEP 8: Training ({NF} folds, with orig fusion)...")
y = train[TARGET].values
orig_y = orig[TARGET].values
tids = test.index.values

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

# === 8a. XGBoost (3 seeds) ===
log("\n  --- XGBoost (3 seeds) ---")
for SEED in SEEDS:
    nm = f"xgb_s{SEED}"
    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=SEED)
    oof = np.zeros((len(train), 3))
    tp = np.zeros((len(test), 3))

    for fold, (tri, vai) in enumerate(skf.split(train, y)):
        # Get train fold data + orig data
        X_tr = pd.concat([train.iloc[tri], orig], ignore_index=True)
        X_va = train.iloc[vai].copy()
        X_te = test.copy()

        y_tr = np.concatenate([y[tri], orig_y])
        orig_w = np.full(len(orig), ORIG_WEIGHT)
        train_w = np.ones(len(tri))
        sample_w = np.concatenate([train_w, orig_w])

        X_tr, X_va, X_te = apply_te(X_tr, X_va, X_te, ALL_TE_COLUMNS, y_tr)

        # Class weights
        classes = np.unique(y_tr)
        cw = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=y_tr)))
        sw_class = np.array([cw[l] for l in y_tr])
        # Combine sample weights (orig weight) with class weights
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

# === 8b. CatBoost (1 seed) ===
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

            X_tr, X_va, X_te = apply_te(X_tr, X_va, X_te, ALL_TE_COLUMNS, y_tr)

            # Class weights
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
# STEP 9: Ensemble
# ============================================================
log(f"\nSTEP 9: Ensemble ({len(all_oof)} models)...")
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
# STEP 10: Threshold Optimization (range [0.5, 3.0])
# ============================================================
log("\nSTEP 10: Threshold optimization (range [0.5, 3.0])...")

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

res = minimize(neg_ba, list(bg), method="Nelder-Mead",
               options={"xatol": 0.001, "fatol": 1e-6, "maxiter": 1000})
bw = [1.0, res.x[0], res.x[1]]
fcv = -res.fun
log(f"  Weights: Low={bw[0]:.3f} Med={bw[1]:.3f} High={bw[2]:.3f}")
log(f"  FINAL CV: {fcv:.5f}")

# ============================================================
# STEP 11: Save submissions
# ============================================================
log("\nSTEP 11: Save submissions...")

# Weighted ensemble with threshold opt
preds_opt = (bt * np.array(bw)).argmax(axis=1)
sub_opt = pd.DataFrame({"id": test_ids, TARGET_COL: [rmap[p] for p in preds_opt]})
path_opt = SUBMISSIONS / "submission_r06_thresh_opt.csv"
sub_opt.to_csv(path_opt, index=False)
log(f"  r06_thresh_opt: {dict(sub_opt[TARGET_COL].value_counts())}")

# Default ensemble (no threshold opt)
preds_default = bt.argmax(axis=1)
sub_default = pd.DataFrame({"id": test_ids, TARGET_COL: [rmap[p] for p in preds_default]})
path_default = SUBMISSIONS / "submission_r06_ens_default.csv"
sub_default.to_csv(path_default, index=False)
log(f"  r06_ens_default: {dict(sub_default[TARGET_COL].value_counts())}")

# ============================================================
# STEP 12: Adversarial Validation (train vs test drift detection)
# ============================================================
log("\nSTEP 12: Adversarial Validation (train vs test)...")

# Prepare adversarial data using numerical + basic categorical features
adv_features = NUMS + CATS

adv_train = train[adv_features].copy()
adv_train["is_test"] = 0
adv_test = test[adv_features].copy()
adv_test["is_test"] = 1

adv_combined = pd.concat([adv_train, adv_test], axis=0, ignore_index=True)

# Ensure categorical columns are int for XGB
for c in CATS:
    adv_combined[c] = adv_combined[c].astype(int)

adv_y = adv_combined["is_test"].values
adv_X = adv_combined.drop("is_test", axis=1)

skf_adv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
adv_oof = np.zeros(len(adv_combined))

for fold_adv, (tri_adv, vai_adv) in enumerate(skf_adv.split(adv_X, adv_y)):
    model_adv = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        objective="binary:logistic", eval_metric="auc",
        random_state=42, n_jobs=-1, tree_method="hist",
        subsample=0.8, colsample_bytree=0.8
    )
    model_adv.fit(
        adv_X.iloc[tri_adv], adv_y[tri_adv],
        eval_set=[(adv_X.iloc[vai_adv], adv_y[vai_adv])],
        verbose=False
    )
    adv_oof[vai_adv] = model_adv.predict_proba(adv_X.iloc[vai_adv])[:, 1]

    del model_adv
    gc.collect()

adv_auc = roc_auc_score(adv_y, adv_oof)
log(f"  Adversarial AUC (train vs test): {adv_auc:.5f}")

if adv_auc < 0.55:
    log("  Interpretation: Train and test distributions are very similar (good)")
elif adv_auc < 0.65:
    log("  Interpretation: Minor distribution differences (acceptable)")
elif adv_auc < 0.75:
    log("  Interpretation: Moderate distribution shift (consider adversarial filtering)")
else:
    log("  Interpretation: Significant distribution shift (adversarial filtering recommended)")

# Feature importance for drift detection
log("\n  Top features causing drift:")
model_adv_full = xgb.XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    objective="binary:logistic", eval_metric="auc",
    random_state=42, n_jobs=-1, tree_method="hist",
    subsample=0.8, colsample_bytree=0.8
)
model_adv_full.fit(adv_X, adv_y, verbose=False)
importances = model_adv_full.feature_importances_
feat_imp = sorted(zip(adv_features, importances), key=lambda x: -x[1])
for fname, fimp in feat_imp[:5]:
    log(f"    {fname}: {fimp:.4f}")

del adv_train, adv_test, adv_combined, adv_X, adv_y, adv_oof, model_adv_full
gc.collect()

# Also check orig vs synthetic (train) distribution
log("\n  Adversarial: orig vs train distribution check...")
adv_orig = orig[adv_features].copy()
adv_orig["is_orig"] = 1
adv_syn = train[adv_features].copy()
adv_syn["is_orig"] = 0

adv_combined2 = pd.concat([adv_orig, adv_syn], axis=0, ignore_index=True)
for c in CATS:
    adv_combined2[c] = adv_combined2[c].astype(int)

adv_y2 = adv_combined2["is_orig"].values
adv_X2 = adv_combined2.drop("is_orig", axis=1)

skf_adv2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
adv_oof2 = np.zeros(len(adv_combined2))

for fold_adv, (tri_adv, vai_adv) in enumerate(skf_adv2.split(adv_X2, adv_y2)):
    model_adv2 = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        objective="binary:logistic", eval_metric="auc",
        random_state=42, n_jobs=-1, tree_method="hist",
        subsample=0.8, colsample_bytree=0.8
    )
    model_adv2.fit(
        adv_X2.iloc[tri_adv], adv_y2[tri_adv],
        eval_set=[(adv_X2.iloc[vai_adv], adv_y2[vai_adv])],
        verbose=False
    )
    adv_oof2[vai_adv] = model_adv2.predict_proba(adv_X2.iloc[vai_adv])[:, 1]
    del model_adv2
    gc.collect()

adv_auc2 = roc_auc_score(adv_y2, adv_oof2)
log(f"  Adversarial AUC (orig vs train): {adv_auc2:.5f}")
if adv_auc2 < 0.55:
    log("  Interpretation: Orig and train distributions are very similar (safe to fuse)")
elif adv_auc2 < 0.65:
    log("  Interpretation: Minor differences between orig and train (safe with weighting)")
else:
    log("  Interpretation: Notable differences - consider adjusting orig weight")

# ============================================================
# SUMMARY
# ============================================================
log("\n" + "=" * 60)
log("SUMMARY - Round 06")
log("=" * 60)
log(f"  Orig data fusion: {len(orig)} rows x {ORIG_WEIGHT}x weight")
log(f"  DT-on-orig features: 3 probability columns")
log(f"  LR-on-orig features: 3 probability columns")
log(f"  KBins binned features: {len(binned_col_names)}")
log(f"  KBins pairwise TE features: {len(kbd_te_columns)}")
log(f"  All-pairwise TE features: {len(TE_columns)}")
log(f"  Total TE columns (all pairwise + KBins): {len(ALL_TE_COLUMNS)}")
log(f"  TE_ORIG features: {len(CATS) + len(NUMS)}")
log(f"  Total features: {len(FEATURES)} + {len(ALL_TE_COLUMNS) * 3} TE = "
    f"{len(FEATURES) + len(ALL_TE_COLUMNS) * 3}")
log(f"  Models: {len(all_oof)} ({len(names)})")
for nm in names:
    sc = balanced_accuracy_score(y, all_oof[nm].argmax(1))
    log(f"    {nm}: {sc:.5f}")
log(f"  Weighted avg OOF: {bwa:.5f}")
log(f"  Stacked OOF: {ssc:.5f}")
log(f"  FINAL CV (with threshold opt): {fcv:.5f}")
log(f"  Adversarial AUC (train vs test): {adv_auc:.5f}")
log(f"  Adversarial AUC (orig vs train): {adv_auc2:.5f}")
log(f"  Total: {time.time() - start:.0f}s")

#!/usr/bin/env python3
"""R18: MLOps Framework-Validated Multi-Model Ensemble

Replicates R15's best algorithm (13 models + pseudo-labeling + hill climbing)
using the MLOps framework for config, logging, MLflow tracking, and validation.

Architecture:
  - 3 XGBoost + 6 LightGBM + 3 CatBoost + 1 HGB = 13 models
  - 2-stage: Stage 1 (no pseudo) → Stage 2 (with pseudo, weight=0.5x)
  - Ensemble: Hill Climbing + LR stacking + simple avg → pick best
  - Threshold optimization on final predictions

Usage:
    python scripts/run_r18_framework.py
    python scripts/run_r18_framework.py --no-mlflow
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import gc
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import TargetEncoder
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

# ---- Framework imports ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHARED_SRC = Path("/Users/guohongbin/projects/fashion-lifecycle-pricing/src")

# Load LOCAL config FIRST (before any shared framework imports)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from config import (
    DATA_RAW, SUBMISSIONS, TARGET_COL, ID_COL, CLASSES,
    CATEGORICAL_COLS, NUMERICAL_COLS,
)

# Load shared framework utilities (insert after local config is loaded)
sys.path.insert(0, str(SHARED_SRC))
from utils.logging_utils import get_logger
from utils.submission import get_submission_filename

RUN_NAME = "R18_framework"
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

log = get_logger(RUN_NAME)
start_time = time.time()

# ============================================================
# Constants
# ============================================================
NF = 5
SEEDS = [42, 123, 456]
tmap = {"Low": 0, "Medium": 1, "High": 2}
rmap = {0: "Low", 1: "Medium", 2: "High"}
CATS = CATEGORICAL_COLS
NUMS = NUMERICAL_COLS
TARGET = TARGET_COL

LGB_CONFIGS = [
    dict(n_estimators=2000, learning_rate=0.02, num_leaves=127, max_depth=9,
         class_weight="balanced", verbose=-1, colsample_bytree=0.7, subsample=0.8,
         reg_alpha=0.05, reg_lambda=0.1, min_child_samples=50),
    dict(n_estimators=1500, learning_rate=0.03, num_leaves=63, max_depth=7,
         class_weight="balanced", verbose=-1, colsample_bytree=0.8, subsample=0.7,
         reg_alpha=0.2, reg_lambda=0.3, min_child_samples=30),
]
PSEUDO_THRESHOLD = 0.90
PSEUDO_WEIGHT = 0.5


# ============================================================
# Helper functions
# ============================================================
def apply_te(X_tr, X_va, X_te, te_cols, y_tr):
    enc = TargetEncoder(target_type="multiclass", cv=5, random_state=RANDOM_STATE)
    tr_enc = enc.fit_transform(X_tr[te_cols], y_tr)
    va_enc = enc.transform(X_va[te_cols])
    te_enc = enc.transform(X_te[te_cols])
    col_names = [f"TE_{i}" for i in range(tr_enc.shape[1])]
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
    import xgboost as xgb
    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=seed)
    oof_size = n_orig if n_orig is not None else len(train_df)
    oof = np.zeros((oof_size, 3))
    tp = np.zeros((len(test_df), 3))
    for fold, (tri, vai) in enumerate(skf.split(train_df, y_arr)):
        if n_orig is not None:
            orig_vai = vai[vai < n_orig]
            if len(orig_vai) == 0: continue
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
            for df_ in [X_tr, X_va, X_te]: df_[c] = df_[c].astype(int)
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
        del X_tr, X_va, X_te, model; gc.collect()
    sc = balanced_accuracy_score(y_arr[:oof_size], oof.argmax(1))
    log.info(f"  >> {nm}: {sc:.5f}")
    return oof, tp


def train_lgb(train_df, y_arr, test_df, te_cols, seed, nm, lgb_params, n_orig=None, pw=None):
    import lightgbm as lgb
    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=seed)
    oof_size = n_orig if n_orig is not None else len(train_df)
    oof = np.zeros((oof_size, 3))
    tp = np.zeros((len(test_df), 3))
    for fold, (tri, vai) in enumerate(skf.split(train_df, y_arr)):
        if n_orig is not None:
            orig_vai = vai[vai < n_orig]
            if len(orig_vai) == 0: continue
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
            for df_ in [X_tr, X_va, X_te]: df_[c] = df_[c].astype("category")
        model = lgb.LGBMClassifier(**lgb_params, random_state=seed)
        model.fit(X_tr[feat], y_tr, sample_weight=sw,
                  eval_set=[(X_va[feat], y_arr[orig_vai])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[orig_vai] = model.predict_proba(X_va[feat])
        tp += model.predict_proba(X_te[feat]) / NF
        del X_tr, X_va, X_te, model; gc.collect()
    sc = balanced_accuracy_score(y_arr[:oof_size], oof.argmax(1))
    log.info(f"  >> {nm}: {sc:.5f}")
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
            if len(orig_vai) == 0: continue
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
            for df_ in [X_tr, X_va, X_te]: df_[c] = df_[c].astype(str).astype("category")
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
        del X_tr, X_va, X_te, model; gc.collect()
    sc = balanced_accuracy_score(y_arr[:oof_size], oof.argmax(1))
    log.info(f"  >> {nm}: {sc:.5f}")
    return oof, tp


def train_hgb(train_df, y_arr, test_df, te_cols, seed, nm, n_orig=None, pw=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=seed)
    oof_size = n_orig if n_orig is not None else len(train_df)
    oof = np.zeros((oof_size, 3))
    tp = np.zeros((len(test_df), 3))
    for fold, (tri, vai) in enumerate(skf.split(train_df, y_arr)):
        if n_orig is not None:
            orig_vai = vai[vai < n_orig]
            if len(orig_vai) == 0: continue
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
            for df_ in [X_tr, X_va, X_te]: df_[c] = df_[c].astype(float)
        model = HistGradientBoostingClassifier(
            max_iter=2000, learning_rate=0.03, max_depth=8,
            min_samples_leaf=50, l2_regularization=0.1,
            class_weight="balanced", random_state=seed,
            early_stopping=True, n_iter_no_change=50,
            scoring="balanced_accuracy"
        )
        model.fit(X_tr[feat], y_tr, sample_weight=sw)
        oof[orig_vai] = model.predict_proba(X_va[feat])
        tp += model.predict_proba(X_te[feat]) / NF
        del X_tr, X_va, X_te, model; gc.collect()
    sc = balanced_accuracy_score(y_arr[:oof_size], oof.argmax(1))
    log.info(f"  >> {nm}: {sc:.5f}")
    return oof, tp


# ============================================================
# Main Pipeline
# ============================================================
def main():
    args = argparse.ArgumentParser()
    args.add_argument("--no-mlflow", action="store_true")
    args = args.parse_args()

    log.separator(f"{RUN_NAME}: MLOps Framework Multi-Model Ensemble")

    # Stage 1: Load data
    log.section("Stage 1: Data Loading")
    train = pd.read_csv(DATA_RAW / "train.csv", index_col="id")
    test = pd.read_csv(DATA_RAW / "test.csv", index_col="id")
    orig = pd.read_csv(DATA_RAW / "irrigation_prediction.csv")
    test_ids = pd.read_csv(DATA_RAW / "test.csv")[ID_COL].values

    train[TARGET] = train[TARGET].map(tmap)
    orig[TARGET] = orig[TARGET].map(tmap)
    log.info(f"Train: {train.shape}, Test: {test.shape}, Orig: {orig.shape}")

    # Stage 2: Feature Engineering
    log.section("Stage 2: Feature Engineering")

    # Factorize categoricals
    log.info("  Factorizing categoricals...")
    combined = pd.concat([train, test, orig])
    for c in CATS:
        combined[c], _ = combined[c].factorize()
    combined[CATS] = combined[CATS].astype("category")
    train = combined[:len(train)].copy()
    test = combined[len(train):len(train)+len(test)].copy().drop(TARGET, axis=1)
    orig = combined[len(train)+len(test):].copy()
    del combined; gc.collect()

    # Pairwise TE features
    log.info("  Creating pairwise interaction features...")
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

    log.info(f"  Created {len(TE_columns)} pairwise TE features (from {total_pairs} pairs)")

    # TE_ORIG features
    log.info("  Computing TE_ORIG features...")
    for c in CATS + NUMS:
        tmp = orig.groupby(c, observed=True)[TARGET].mean().astype("float32")
        tmp.name = f"TE_ORIG_{c}"
        train = train.merge(tmp, on=c, how="left")
        train[tmp.name] = train[tmp.name].fillna(0.5)
        test = test.merge(tmp, on=c, how="left")
        test[tmp.name] = test[tmp.name].fillna(0.5)

    FEATURES = [c for c in train.columns if c != TARGET]
    log.info(f"  Total features: {len(FEATURES)}")
    log.info(f"  TE columns: {len(TE_columns)}")

    y = train[TARGET].values

    # Stage 3: Stage 1 Training (no pseudo labels)
    log.section("Stage 3: Stage 1 Training (13 models, no pseudo)")
    s1_oof = {}
    s1_tp = {}

    log.info("  --- XGBoost (3 seeds) ---")
    for SEED in SEEDS:
        nm = f"s1_xgb_s{SEED}"
        oof, tp = train_xgb(train, y, test, TE_columns, SEED, nm)
        s1_oof[nm] = oof
        s1_tp[nm] = tp

    log.info("  --- LightGBM (2 configs x 3 seeds = 6 models) ---")
    for SEED in SEEDS:
        for li, lp in enumerate(LGB_CONFIGS):
            nm = f"s1_lgb{li}_s{SEED}"
            oof, tp = train_lgb(train, y, test, TE_columns, SEED, nm, lp)
            s1_oof[nm] = oof
            s1_tp[nm] = tp

    log.info("  --- CatBoost (3 seeds) ---")
    for SEED in SEEDS:
        nm = f"s1_cb_s{SEED}"
        oof, tp = train_cb(train, y, test, TE_columns, SEED, nm)
        s1_oof[nm] = oof
        s1_tp[nm] = tp

    log.info("  --- HistGradientBoosting (1 model) ---")
    nm = "s1_hgb_s42"
    oof, tp = train_hgb(train, y, test, TE_columns, 42, nm)
    s1_oof[nm] = oof
    s1_tp[nm] = tp

    log.info(f"  Stage 1 models: {len(s1_oof)}")

    # Stage 4: Pseudo-Labeling
    log.section("Stage 4: Pseudo-Labeling")
    s1_avg_tp = sum(s1_tp.values()) / len(s1_tp)
    s1_pred_labels = s1_avg_tp.argmax(axis=1)
    s1_pred_conf = s1_avg_tp.max(axis=1)

    pseudo_mask = s1_pred_conf >= PSEUDO_THRESHOLD
    log.info(f"  Total test: {len(test)}, Pseudo-labeled: {pseudo_mask.sum()} ({pseudo_mask.mean()*100:.1f}%)")
    for cls_id, cls_name in rmap.items():
        count = (s1_pred_labels[pseudo_mask] == cls_id).sum()
        log.info(f"    {cls_name}: {count}")

    pseudo_test = test[pseudo_mask].copy()
    pseudo_test[TARGET] = s1_pred_labels[pseudo_mask]
    pseudo_y = s1_pred_labels[pseudo_mask].copy()

    # Stage 5: Stage 2 Training (with pseudo labels)
    log.section("Stage 5: Stage 2 Training (with pseudo, weight=0.5x)")
    train_with_pseudo = pd.concat([train, pseudo_test], ignore_index=True)
    y_with_pseudo = np.concatenate([y, pseudo_y])
    N_ORIG = len(train)
    log.info(f"  Train+Pseudo: {len(train_with_pseudo)} rows")

    s2_oof = {}
    s2_tp = {}

    log.info("  --- XGBoost (3 seeds) ---")
    for SEED in SEEDS:
        nm = f"s2_xgb_s{SEED}"
        oof, tp = train_xgb(train_with_pseudo, y_with_pseudo, test, TE_columns,
                             SEED, nm, n_orig=N_ORIG, pw=PSEUDO_WEIGHT)
        s2_oof[nm] = oof
        s2_tp[nm] = tp

    log.info("  --- LightGBM (6 models) ---")
    for SEED in SEEDS:
        for li, lp in enumerate(LGB_CONFIGS):
            nm = f"s2_lgb{li}_s{SEED}"
            oof, tp = train_lgb(train_with_pseudo, y_with_pseudo, test, TE_columns,
                                 SEED, nm, lp, n_orig=N_ORIG, pw=PSEUDO_WEIGHT)
            s2_oof[nm] = oof
            s2_tp[nm] = tp

    log.info("  --- CatBoost (3 seeds) ---")
    for SEED in SEEDS:
        nm = f"s2_cb_s{SEED}"
        oof, tp = train_cb(train_with_pseudo, y_with_pseudo, test, TE_columns,
                            SEED, nm, n_orig=N_ORIG, pw=PSEUDO_WEIGHT)
        s2_oof[nm] = oof
        s2_tp[nm] = tp

    log.info("  --- HistGradientBoosting ---")
    nm = "s2_hgb_s42"
    oof, tp = train_hgb(train_with_pseudo, y_with_pseudo, test, TE_columns,
                         42, nm, n_orig=N_ORIG, pw=PSEUDO_WEIGHT)
    s2_oof[nm] = oof
    s2_tp[nm] = tp

    log.info(f"  Stage 2 models: {len(s2_oof)}")

    # Stage 6: Ensemble
    log.section("Stage 6: Ensemble")
    names = list(s2_oof.keys())

    # Simple average
    simple_avg = sum(s2_oof[n] for n in names) / len(names)
    sa_score = balanced_accuracy_score(y, simple_avg.argmax(1))
    log.metric("Simple avg OOF", sa_score)

    # Individual scores
    for n in names:
        s = balanced_accuracy_score(y, s2_oof[n].argmax(1))
        log.info(f"    {n}: {s:.5f}")

    # Hill Climbing
    log.info("  Running Hill Climbing...")
    hc_names = []
    hc_best_score = 0
    for n in names:
        s = balanced_accuracy_score(y, s2_oof[n].argmax(1))
        if s > hc_best_score:
            hc_best_score = s
            hc_names = [n]
    hc_remaining = [n for n in names if n not in hc_names]

    improved = True
    while improved and hc_remaining:
        improved = False
        best_add = None
        best_new_score = hc_best_score
        for n in hc_remaining:
            candidates = hc_names + [n]
            combo = sum(s2_oof[c] for c in candidates) / len(candidates)
            s = balanced_accuracy_score(y, combo.argmax(1))
            if s > best_new_score:
                best_new_score = s
                best_add = n
        if best_add is not None:
            hc_names.append(best_add)
            hc_remaining.remove(best_add)
            hc_best_score = best_new_score
            improved = True

    log.info(f"  Hill Climbing: {len(hc_names)} models, OOF: {hc_best_score:.5f}")

    # Optimize weights
    def neg_ba_ensemble(w):
        w = np.abs(w)
        w = w / w.sum()
        combo = sum(w[j] * s2_oof[n] for j, n in enumerate(hc_names))
        return -balanced_accuracy_score(y, combo.argmax(1))

    n_sel = len(hc_names)
    best_w = np.ones(n_sel) / n_sel
    best_wa = hc_best_score
    rng = np.random.RandomState(RANDOM_STATE)
    for _ in range(5000):
        w = rng.dirichlet(np.ones(n_sel))
        s = -neg_ba_ensemble(w)
        if s > best_wa:
            best_wa = s
            best_w = w.copy()

    res_w = minimize(neg_ba_ensemble, best_w, method="Nelder-Mead",
                     options={"xatol": 0.001, "fatol": 1e-6, "maxiter": 2000})
    opt_w = np.abs(res_w.x)
    opt_w = opt_w / opt_w.sum()
    opt_wa = -res_w.fun
    log.metric("HC+opt weights OOF", opt_wa)

    # Stacking
    log.info("  Running LR stacking...")
    ostk = np.hstack([s2_oof[n] for n in names])
    tstk = np.hstack([s2_tp[n] for n in names])
    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=RANDOM_STATE)
    moof = np.zeros(len(y), dtype=int)
    mtest = np.zeros((len(test), 3))
    for fold, (tri, vai) in enumerate(skf.split(ostk, y)):
        lr = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, random_state=RANDOM_STATE)
        lr.fit(ostk[tri], y[tri])
        moof[vai] = lr.predict(ostk[vai])
        mtest += lr.predict_proba(tstk) / NF
    ssc = balanced_accuracy_score(y, moof)
    log.metric("Stacked OOF", ssc)

    # Choose best
    scores = {"stacking": ssc, "hill_climbing": opt_wa, "simple_avg": sa_score}
    best_method = max(scores, key=scores.get)
    log.info(f"  >>> Best: {best_method} ({scores[best_method]:.5f})")

    if best_method == "stacking":
        fm = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, random_state=RANDOM_STATE)
        fm.fit(ostk, y)
        bt = fm.predict_proba(tstk)
        bo = np.zeros((len(y), 3))
        for fold, (tri, vai) in enumerate(skf.split(ostk, y)):
            lr = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, random_state=RANDOM_STATE)
            lr.fit(ostk[tri], y[tri])
            bo[vai] = lr.predict_proba(ostk[vai])
    elif best_method == "hill_climbing":
        bt = sum(opt_w[j] * s2_tp[n] for j, n in enumerate(hc_names))
        bo = sum(opt_w[j] * s2_oof[n] for j, n in enumerate(hc_names))
    else:
        bt = sum(s2_tp[n] for n in names) / len(names)
        bo = simple_avg

    # Stage 7: Threshold Optimization
    log.section("Stage 7: Threshold Optimization")
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

    res = minimize(neg_ba, list(bg), method="Nelder-Mead",
                   options={"xatol": 0.001, "fatol": 1e-6, "maxiter": 1000})
    bw = [1.0, res.x[0], res.x[1]]
    fcv = -res.fun
    log.info(f"  Weights: Low={bw[0]:.3f} Med={bw[1]:.3f} High={bw[2]:.3f}")
    log.metric("FINAL CV", fcv)

    # Stage 8: Submission
    log.section("Stage 8: Submission")

    # Threshold optimized
    preds_thresh = (bt * np.array(bw)).argmax(1)
    sub = pd.DataFrame({ID_COL: test_ids, TARGET_COL: [rmap[p] for p in preds_thresh]})
    fname = get_submission_filename("r18_framework", SUBMISSIONS)
    sub.to_csv(fname, index=False)
    dist = pd.Series([rmap[p] for p in preds_thresh]).value_counts()
    log.info(f"  r18_thresh_opt: {dict(dist)}")
    log.info(f"  Saved: {fname}")

    # Default (no threshold)
    preds_default = bt.argmax(1)
    sub2 = pd.DataFrame({ID_COL: test_ids, TARGET_COL: [rmap[p] for p in preds_default]})
    fname2 = get_submission_filename("r18_framework_default", SUBMISSIONS)
    sub2.to_csv(fname2, index=False)
    log.info(f"  r18_default: saved to {fname2}")

    # HC weighted only
    preds_hc = (sum(opt_w[j] * s2_tp[n] for j, n in enumerate(hc_names))).argmax(1)
    sub_hc = pd.DataFrame({ID_COL: test_ids, TARGET_COL: [rmap[p] for p in preds_hc]})
    fname_hc = get_submission_filename("r18_framework_hc", SUBMISSIONS)
    sub_hc.to_csv(fname_hc, index=False)
    log.info(f"  r18_hc: saved to {fname_hc}")

    elapsed = int(time.time() - start_time)

    # MLflow Tracking
    if not args.no_mlflow:
        log.section("MLflow Tracking")
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT}/mlflow.db")
        mlflow.set_experiment("ps_s6e4_irrigation")

        with mlflow.start_run(run_name=RUN_NAME) as run:
            run_id = run.info.run_id
            mlflow.log_params({
                "n_models_stage1": len(s1_oof),
                "n_models_stage2": len(s2_oof),
                "pseudo_threshold": PSEUDO_THRESHOLD,
                "pseudo_weight": PSEUDO_WEIGHT,
                "n_folds": NF,
                "seeds": str(SEEDS),
                "random_state": RANDOM_STATE,
                "n_features": len(FEATURES),
                "n_te_columns": len(TE_columns),
                "best_method": best_method,
                "elapsed_sec": elapsed,
            })
            mlflow.log_metrics({
                "cv_simple_avg": sa_score,
                "cv_hill_climbing": opt_wa,
                "cv_stacking": ssc,
                "cv_final": fcv,
            })
            mlflow.log_text("\n".join(FEATURES), "features.txt")
            mlflow.log_text("\n".join(names), "models.txt")
            for f in [fname, fname2, fname_hc]:
                mlflow.log_artifact(str(f), artifact_path="submissions")
            mlflow.set_tag("notes", "R18: MLOps framework-validated multi-model ensemble")
            log.info(f"  MLflow run_id: {run_id}")
    else:
        log.info("  MLflow disabled (--no-mlflow)")

    # Summary
    log.separator(f"{RUN_NAME} Complete")
    log.metric("FINAL CV", fcv)
    log.info(f"  Models: {len(s2_oof)} (S1: {len(s1_oof)} + S2: {len(s2_oof)})")
    log.info(f"  Best method: {best_method}")
    log.info(f"  Elapsed: {elapsed}s")
    log.separator()


if __name__ == "__main__":
    main()

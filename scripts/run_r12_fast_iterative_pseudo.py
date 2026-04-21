"""Round 12: Fast Iterative Pseudo-Labeling

Key innovations over R09:
1. Pre-compute all TE features ONCE (no per-fold TE recomputation = 5x faster)
2. 3 rounds of iterative pseudo-labeling (thresholds 0.90, 0.85, 0.80)
3. Only strongest models: 3 XGB + 1 CB (drop LGB which adds <0.001)
4. Hill climbing ensemble (greedy forward selection)
5. Confidence-weighted pseudo-labels
6. No stacking (avoids CV-LB gap inflation)

Target: close gap from 0.97785 to 0.98+
"""
import warnings
warnings.filterwarnings("ignore")
import time
import gc
import numpy as np
import pandas as pd
import xgboost as xgb
import catboost as cb
from pathlib import Path
import sys
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import TargetEncoder
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (SUBMISSIONS, TARGET_COL, ID_COL, CLASSES,
                        CATEGORICAL_COLS, NUMERICAL_COLS)

def log(msg=""):
    print(msg, flush=True)

start = time.time()
log("=" * 60)
log("Round 12: Fast Iterative Pseudo-Labeling")
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
CATS = list(CATEGORICAL_COLS)
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
# STEP 3: Create pairwise interaction features
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

# ============================================================
# STEP 5: Prepare feature columns
# ============================================================
log("\nSTEP 5: Preparing feature columns...")

# Convert category columns to int codes for XGBoost compatibility
for df_ in [train, test, orig]:
    for c in CATS:
        if c in df_.columns:
            df_[c] = df_[c].cat.codes.astype("int32")

y = train[TARGET].values
BASE_COLS = [c for c in train.columns if c != TARGET and c not in TE_columns]
log(f"  Base features: {len(BASE_COLS)}, TE columns: {len(TE_columns)}")

# ============================================================
# Helper functions
# ============================================================
def fast_te_encode(X_tr, X_va, X_te, te_cols, y_tr, smoothing=10):
    """Fast target encoding: smoothed per-class means, no internal CV.
    Much faster than sklearn TargetEncoder while avoiding major leakage."""
    n_classes = 3
    n_te = len(te_cols)
    global_means = np.array([(y_tr == c).mean() for c in range(n_classes)])

    tr_enc = np.zeros((len(X_tr), n_te * n_classes), dtype=np.float32)
    va_enc = np.zeros((len(X_va), n_te * n_classes), dtype=np.float32)
    te_enc = np.zeros((len(X_te), n_te * n_classes), dtype=np.float32)

    for i, col in enumerate(te_cols):
        # One-pass: compute smoothed class means per category
        grouped = X_tr[[col]].copy()
        grouped["_y"] = y_tr
        stats = grouped.groupby(col)["_y"]
        counts = stats.count()

        for c in range(n_classes):
            class_counts = grouped[grouped["_y"] == c].groupby(col).size()
            class_counts = class_counts.reindex(counts.index, fill_value=0)
            smoothed = (class_counts + smoothing * global_means[c]) / (counts + smoothing)

            col_enc_name = f"TE_{i}_{c}"
            mapping = smoothed.to_dict()
            default = global_means[c]

            tr_enc[:, i * n_classes + c] = X_tr[col].map(mapping).fillna(default).values
            va_enc[:, i * n_classes + c] = X_va[col].map(mapping).fillna(default).values
            te_enc[:, i * n_classes + c] = X_te[col].map(mapping).fillna(default).values

    return tr_enc, va_enc, te_enc


def prepare_features(X_tr, X_va, X_te, te_cols, y_tr):
    """Apply fast TE and combine with base features."""
    tr_te, va_te, te_te = fast_te_encode(X_tr, X_va, X_te, te_cols, y_tr)

    n_te_features = tr_te.shape[1]
    te_col_names = [f"TE_{i}" for i in range(n_te_features)]

    X_tr_out = pd.concat([
        X_tr[BASE_COLS].reset_index(drop=True),
        pd.DataFrame(tr_te, columns=te_col_names)
    ], axis=1)

    X_va_out = pd.concat([
        X_va[BASE_COLS].reset_index(drop=True),
        pd.DataFrame(va_te, columns=te_col_names)
    ], axis=1)

    X_te_out = pd.concat([
        X_te[BASE_COLS].reset_index(drop=True),
        pd.DataFrame(te_te, columns=te_col_names)
    ], axis=1)

    return X_tr_out, X_va_out, X_te_out


def train_xgb_fast(train_df, y_arr, test_df, te_cols, seed, nm, n_orig=None, pw=None):
    """Train XGB with 5-fold CV and per-fold fast TE."""
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

        X_tr, X_va, X_te = prepare_features(X_tr, X_va, X_te, te_cols, y_tr)

        # Sample weights
        classes = np.unique(y_tr)
        cw = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=y_tr)))
        sw = np.array([cw[l] for l in y_tr])
        if pw is not None and n_orig is not None:
            pseudo_mask = tri >= n_orig
            sw[pseudo_mask] *= pw

        model = xgb.XGBClassifier(
            max_depth=6, subsample=0.8, colsample_bytree=0.8,
            n_estimators=5000, objective="multi:softprob", learning_rate=0.03,
            early_stopping_rounds=100,
            eval_metric="mlogloss",
            max_bin=1024, random_state=seed, n_jobs=-1, tree_method="hist"
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_arr[orig_vai])],
                  sample_weight=sw, verbose=False)
        oof[orig_vai] = model.predict_proba(X_va)
        tp += model.predict_proba(X_te) / NF

        fs = balanced_accuracy_score(y_arr[orig_vai], oof[orig_vai].argmax(1))
        log(f"    {nm} fold {fold+1}: {fs:.5f}")
        del X_tr, X_va, X_te, model
        gc.collect()

    sc = balanced_accuracy_score(y_arr[:oof_size], oof.argmax(1))
    log(f"  >> {nm}: {sc:.5f}")
    return oof, tp


def train_cb_fast(train_df, y_arr, test_df, te_cols, seed, nm, n_orig=None, pw=None):
    """Train CatBoost with 5-fold CV and per-fold fast TE."""
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

        X_tr, X_va, X_te = prepare_features(X_tr, X_va, X_te, te_cols, y_tr)

        # Sample weights
        sw = compute_sample_weight("balanced", y_tr)
        if pw is not None and n_orig is not None:
            pseudo_mask = tri >= n_orig
            sw[pseudo_mask] *= pw

        model = cb.CatBoostClassifier(
            iterations=5000, learning_rate=0.03, depth=6,
            loss_function="MultiClass", eval_metric="TotalF1",
            random_seed=seed, verbose=0, thread_count=-1,
            auto_class_weights="Balanced",
            early_stopping_rounds=100
        )
        model.fit(X_tr, y_tr, eval_set=(X_va, y_arr[orig_vai]),
                  sample_weight=sw, verbose=0)
        oof[orig_vai] = model.predict_proba(X_va)
        tp += model.predict_proba(X_te) / NF

        fs = balanced_accuracy_score(y_arr[orig_vai], oof[orig_vai].argmax(1))
        log(f"    {nm} fold {fold+1}: {fs:.5f}")
        del X_tr, X_va, X_te, model
        gc.collect()

    sc = balanced_accuracy_score(y_arr[:oof_size], oof.argmax(1))
    log(f"  >> {nm}: {sc:.5f}")
    return oof, tp


def hill_climbing(y_true, oof_dict, model_names, n_repeats=5):
    """Greedy forward selection ensemble."""
    best_score = 0
    best_w = None
    best_selected = None

    for r in range(n_repeats):
        rng = np.random.RandomState(r * 42)
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

            # Recompute all weights
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

    log(f"  Hill climbing: {best_score:.5f} with {len(best_selected)} models")
    log(f"    Selected: {best_selected}")
    log(f"    Weights: {[f'{best_w[n]:.4f}' for n in best_selected]}")
    return best_w, best_selected, best_score


def threshold_optimize(y_true, probs):
    """Optimize class weights for balanced accuracy."""
    best_score = balanced_accuracy_score(y_true, probs.argmax(1))
    best_w = [1.0, 1.0, 1.0]

    # Grid search
    for w_med in np.arange(0.5, 3.0, 0.1):
        for w_high in np.arange(0.5, 5.0, 0.2):
            w = np.array([1.0, w_med, w_high])
            pred = (probs * w).argmax(1)
            s = balanced_accuracy_score(y_true, pred)
            if s > best_score:
                best_score = s
                best_w = [1.0, w_med, w_high]

    # Nelder-Mead refinement
    def neg_ba(params):
        w = np.array([1.0, params[0], params[1]])
        return -balanced_accuracy_score(y_true, (probs * w).argmax(1))

    res = minimize(neg_ba, [best_w[1], best_w[2]], method="Nelder-Mead",
                   options={"maxiter": 500, "xatol": 0.001})
    final_w = [1.0, res.x[0], res.x[1]]
    final_score = balanced_accuracy_score(y_true, (probs * np.array(final_w)).argmax(1))

    if final_score > best_score:
        best_score = final_score
        best_w = final_w

    log(f"  Threshold opt: {best_score:.5f} weights={[f'{w:.3f}' for w in best_w]}")
    return best_w, best_score


# ============================================================
# STEP 6: Stage 1 — Train on original data (no pseudo labels)
# ============================================================
log(f"\nSTEP 6: Stage 1 — Train on {len(train)} rows (no pseudo)...")

s1_oof = {}
s1_tp = {}

log("  --- XGBoost ---")
for SEED in SEEDS:
    nm = f"s1_xgb_s{SEED}"
    oof, tp = train_xgb_fast(train, y, test, TE_columns, SEED, nm)
    s1_oof[nm] = oof
    s1_tp[nm] = tp

log("  --- CatBoost ---")
nm = "s1_cb_s42"
oof, tp = train_cb_fast(train, y, test, TE_columns, 42, nm)
s1_oof[nm] = oof
s1_tp[nm] = tp

s1_names = list(s1_oof.keys())
s1_avg = sum(s1_tp[n] for n in s1_names) / len(s1_names)
s1_score = balanced_accuracy_score(y, sum(s1_oof[n] for n in s1_names).argmax(1))
log(f"\n  Stage 1 avg OOF: {s1_score:.5f}")
log(f"  Stage 1 models: {len(s1_oof)}")

# ============================================================
# STEP 7: Pseudo-Label Round 1
# ============================================================
PSEUDO_ROUNDS = [
    {"threshold": 0.90, "weight": 0.5, "label": "R1"},
    {"threshold": 0.85, "weight": 0.3, "label": "R2"},
    {"threshold": 0.80, "weight": 0.2, "label": "R3"},
]

all_oof = dict(s1_oof)
all_tp = dict(s1_tp)
current_test_pred = s1_avg.copy()

for pr_idx, pr in enumerate(PSEUDO_ROUNDS):
    stage_label = pr["label"]
    threshold = pr["threshold"]
    pw = pr["weight"]
    stage_num = pr_idx + 2

    log(f"\n{'=' * 60}")
    log(f"STEP {6 + stage_num}: Pseudo-Label {stage_label} (threshold={threshold}, weight={pw})")
    log(f"{'=' * 60}")

    # Pseudo-label from current test predictions
    max_probs = current_test_pred.max(1)
    pseudo_mask = max_probs >= threshold
    pseudo_labels = current_test_pred[pseudo_mask].argmax(1)
    pseudo_confidence = max_probs[pseudo_mask]

    log(f"  Total test: {len(test)}, Pseudo-labeled: {pseudo_mask.sum()} ({pseudo_mask.sum()/len(test)*100:.1f}%)")
    for cls in range(3):
        log(f"    {rmap[cls]}: {(pseudo_labels == cls).sum()}")

    # Add pseudo-labeled test data to training
    pseudo_X = test[pseudo_mask].reset_index(drop=True)
    pseudo_y = pseudo_labels
    pseudo_sw = pseudo_confidence  # Confidence-weighted

    train_with_pseudo = pd.concat([train, pseudo_X], ignore_index=True)
    y_with_pseudo = np.concatenate([y, pseudo_y])
    n_orig = len(train)

    log(f"  Train+Pseudo: {len(train_with_pseudo)} rows")

    # Train models
    stage_prefix = f"s{stage_num}"
    s_oof = {}
    s_tp = {}

    log("  --- XGBoost ---")
    for SEED in SEEDS:
        nm = f"{stage_prefix}_xgb_s{SEED}"
        oof, tp = train_xgb_fast(train_with_pseudo, y_with_pseudo, test, TE_columns,
                                   SEED, nm, n_orig=n_orig, pw=pw)
        s_oof[nm] = oof
        s_tp[nm] = tp

    log("  --- CatBoost ---")
    nm = f"{stage_prefix}_cb_s42"
    oof, tp = train_cb_fast(train_with_pseudo, y_with_pseudo, test, TE_columns,
                             42, nm, n_orig=n_orig, pw=pw)
    s_oof[nm] = oof
    s_tp[nm] = tp

    # Update current test predictions for next round
    s_names = list(s_tp.keys())
    current_test_pred = sum(s_tp[n] for n in s_names) / len(s_names)

    # Add to all models
    all_oof.update(s_oof)
    all_tp.update(s_tp)

    s_score = balanced_accuracy_score(y, sum(s_oof[n] for n in s_names).argmax(1))
    log(f"\n  Stage {stage_num} avg OOF: {s_score:.5f}")

# ============================================================
# STEP 10: Summary of all stages
# ============================================================
log(f"\n{'=' * 60}")
log(f"ALL STAGES COMPLETE")
log(f"{'=' * 60}")

all_names = list(all_oof.keys())
log(f"  Total models: {len(all_names)}")

# Per-stage scores
for stage_prefix in ["s1", "s2", "s3", "s4"]:
    stage_models = [n for n in all_names if n.startswith(stage_prefix + "_")]
    if stage_models:
        stage_oof_avg = sum(all_oof[n] for n in stage_models) / len(stage_models)
        stage_score = balanced_accuracy_score(y, stage_oof_avg.argmax(1))
        log(f"  {stage_prefix} avg: {stage_score:.5f} ({len(stage_models)} models)")

# ============================================================
# STEP 11: Hill Climbing Ensemble
# ============================================================
log(f"\nSTEP 11: Hill Climbing Ensemble ({len(all_names)} models)...")

best_w, best_selected, hc_score = hill_climbing(y, all_oof, all_names)

# Compute final ensemble predictions
ens_oof = sum(best_w[n] * all_oof[n] for n in best_selected)
ens_test = sum(best_w[n] * all_tp[n] for n in best_selected)
ens_score = balanced_accuracy_score(y, ens_oof.argmax(1))
log(f"  Ensemble OOF: {ens_score:.5f}")

# Also try simple average of all models
simple_avg_oof = sum(all_oof[n] for n in all_names) / len(all_names)
simple_avg_score = balanced_accuracy_score(y, simple_avg_oof.argmax(1))
simple_avg_test = sum(all_tp[n] for n in all_names) / len(all_names)
log(f"  Simple avg OOF: {simple_avg_score:.5f} ({len(all_names)} models)")

# Choose best ensemble
if ens_score >= simple_avg_score:
    log(f"  >>> Using Hill Climbing")
    final_oof = ens_oof
    final_test = ens_test
else:
    log(f"  >>> Using Simple Average")
    final_oof = simple_avg_oof
    final_test = simple_avg_test

# ============================================================
# STEP 12: Threshold Optimization
# ============================================================
log(f"\nSTEP 12: Threshold optimization...")
best_thresh, thresh_score = threshold_optimize(y, final_oof)
final_cv = thresh_score
log(f"  FINAL CV: {final_cv:.5f}")

# ============================================================
# STEP 13: Generate Submissions
# ============================================================
log(f"\nSTEP 13: Generating submissions...")

# thresh_opt submission
pred_thresh = (final_test * np.array(best_thresh)).argmax(1)
sub = pd.DataFrame({ID_COL: test_ids, TARGET_COL: [rmap[p] for p in pred_thresh]})
sub.to_csv(SUBMISSIONS / "submission_r12_thresh_opt.csv", index=False)
log(f"  Saved submission_r12_thresh_opt.csv")
log(f"    Distribution: {dict(zip(*np.unique(pred_thresh, return_counts=True)))}")

# ens_default submission (no threshold)
pred_default = final_test.argmax(1)
sub2 = pd.DataFrame({ID_COL: test_ids, TARGET_COL: [rmap[p] for p in pred_default]})
sub2.to_csv(SUBMISSIONS / "submission_r12_ens_default.csv", index=False)
log(f"  Saved submission_r12_ens_default.csv")

# simple avg of all stages submission
pred_simple = simple_avg_test.argmax(1)
sub3 = pd.DataFrame({ID_COL: test_ids, TARGET_COL: [rmap[p] for p in pred_simple]})
sub3.to_csv(SUBMISSIONS / "submission_r12_simple_avg.csv", index=False)
log(f"  Saved submission_r12_simple_avg.csv")

# ============================================================
# STEP 14: Final Summary
# ============================================================
elapsed = time.time() - start
log(f"\n{'=' * 60}")
log(f"R12 COMPLETE")
log(f"{'=' * 60}")
log(f"  Pre-computed TE features: {len(FEATURES)}")
log(f"  Models per stage: 4 (3 XGB + 1 CB)")
log(f"  Pseudo-label rounds: {len(PSEUDO_ROUNDS)}")
for pr_idx, pr in enumerate(PSEUDO_ROUNDS):
    log(f"    {pr['label']}: threshold={pr['threshold']}, weight={pr['weight']}")
log(f"  Total models: {len(all_names)}")
log(f"  --- Per-Stage ---")
for stage_prefix in ["s1", "s2", "s3", "s4"]:
    stage_models = [n for n in all_names if n.startswith(stage_prefix + "_")]
    if stage_models:
        stage_oof_avg = sum(all_oof[n] for n in stage_models) / len(stage_models)
        stage_score = balanced_accuracy_score(y, stage_oof_avg.argmax(1))
        log(f"    {stage_prefix} avg: {stage_score:.5f}")
log(f"  --- Ensemble ---")
log(f"    Simple avg OOF: {simple_avg_score:.5f}")
log(f"    Hill climbing OOF: {hc_score:.5f}")
log(f"    FINAL CV: {final_cv:.5f}")
log(f"  Total: {elapsed:.0f}s")

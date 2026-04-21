"""Round 01: Baseline Model

LightGBM multiclass baseline with label-encoded categorical features.
5-fold StratifiedKFold, class_weight='balanced'.
Target: establish baseline balanced accuracy.
"""
import warnings
warnings.filterwarnings("ignore")
import time
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_data
from src.features.builder import build_features
from src.config import SUBMISSIONS, TARGET_COL, ID_COL, CLASSES, ModelConfig
from src.utils.metrics import balanced_accuracy


def log(msg=""):
    print(msg, flush=True)


start = time.time()

log("=" * 60)
log("Round 01: Baseline LightGBM Multiclass")
log("=" * 60)

# ============================================================
# STEP 1: Load data
# ============================================================
log("\nSTEP 1: Load data")
train_df, test_df = load_data()

# ============================================================
# STEP 2: Build features
# ============================================================
log("\nSTEP 2: Build features (Label Encoding baseline)")
train, test, feat_cols, label_encoders, target_le = build_features(train_df, test_df)
del train_df, test_df
gc.collect()

# ============================================================
# STEP 3: Prepare training data
# ============================================================
log("\nSTEP 3: Prepare training data")
X = train[feat_cols].values.astype(np.float32)
y = train["_target_encoded"].values
X_test = test[feat_cols].values.astype(np.float32)
log(f"  Train: {X.shape}, Test: {X_test.shape}")
log(f"  Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# ============================================================
# STEP 4: 5-fold StratifiedKFold CV
# ============================================================
log("\nSTEP 4: 5-fold StratifiedKFold CV")
cfg = ModelConfig()
N_FOLDS = cfg.n_folds
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=cfg.random_state)

params = cfg.lgb_params.copy()
# Remove n_estimators from params (we'll use early stopping)
n_estimators = params.pop("n_estimators")

oof_preds = np.zeros((len(X), 3))
fold_scores = []
best_iters = []

for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    model = lgb.LGBMClassifier(n_estimators=n_estimators, **params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    oof_preds[val_idx] = model.predict_proba(X_val)
    y_pred = model.predict(X_val)
    score = balanced_accuracy_score(y_val, y_pred)
    fold_scores.append(score)
    best_iters.append(model.best_iteration_)
    log(f"  Fold {fold_i+1}: BA={score:.5f}, iter={model.best_iteration_}")

mean_ba = np.mean(fold_scores)
log(f"  Mean BA: {mean_ba:.5f}")

# OOF overall score
oof_classes = np.argmax(oof_preds, axis=1)
oof_ba = balanced_accuracy_score(y, oof_classes)
log(f"  OOF BA: {oof_ba:.5f}")

# Per-class metrics
log("\n  Classification Report (OOF):")
target_names = [CLASSES[i] for i in range(3)]
log(classification_report(y, oof_classes, target_names=target_names))

# ============================================================
# STEP 5: Train final model and predict
# ============================================================
log("\nSTEP 5: Train final model on all data")
avg_iter = int(np.mean(best_iters))
final_params = params.copy()
final_model = lgb.LGBMClassifier(n_estimators=max(avg_iter, 500), **final_params)
final_model.fit(X, y)
log(f"  Final model trained with {max(avg_iter, 500)} iterations")

# Predict test
test_probs = final_model.predict_proba(X_test)
test_classes = np.argmax(test_probs, axis=1)
test_labels = target_le.inverse_transform(test_classes)

log(f"  Test prediction distribution:")
for cls, cnt in zip(*np.unique(test_labels, return_counts=True)):
    log(f"    {cls}: {cnt} ({cnt/len(test_labels)*100:.1f}%)")

# ============================================================
# STEP 6: Feature importance
# ============================================================
log("\nSTEP 6: Top 15 feature importance")
importance = dict(zip(feat_cols, final_model.feature_importances_))
for fname, fimp in sorted(importance.items(), key=lambda x: -x[1])[:15]:
    log(f"  {fname:40s}: {fimp}")

# ============================================================
# STEP 7: Save submission
# ============================================================
log("\nSTEP 7: Save submission")
sub = pd.DataFrame({
    ID_COL: test[ID_COL].values,
    TARGET_COL: test_labels,
})
path = SUBMISSIONS / "submission_r01_baseline.csv"
sub.to_csv(path, index=False)
log(f"  Saved to {path}")
log(f"  Submission shape: {sub.shape}")

# ============================================================
# SUMMARY
# ============================================================
log("\n" + "=" * 60)
log("SUMMARY")
log("=" * 60)
log(f"  Experiment: R01 (Baseline)")
log(f"  Features: {len(feat_cols)} (label encoding)")
log(f"  Model: LightGBM multiclass (class_weight=balanced)")
log(f"  CV BA: {mean_ba:.5f}")
log(f"  OOF BA: {oof_ba:.5f}")
log(f"  Best iterations: {best_iters}, avg={avg_iter}")
log(f"  Total: {time.time() - start:.1f}s")

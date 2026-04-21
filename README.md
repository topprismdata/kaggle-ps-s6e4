# Kaggle Playground Series S6E4: Predicting Irrigation Need

3-class classification (Low/Medium/High) to predict irrigation requirements.
Evaluation metric: **Balanced Accuracy**.

## Competition Info

- **Duration**: 2026-04-01 to 2026-04-30
- **Train**: 630,000 rows × 20 features
- **Test**: 270,000 rows × 19 features
- **External data**: 10,000 rows (original dataset)
- **Class imbalance**: Low=58.7%, Medium=37.9%, High=3.3%

## Project Structure

```
kaggle-ps-s6e4/
├── scripts/                    # Training scripts (R01-R12)
│   ├── run_r01_baseline.py     # LightGBM baseline
│   ├── run_r02_target_encoding.py
│   ├── run_r03_formula.py      # Feature formula
│   ├── run_r04_pairwise_te.py  # Pairwise target encoding
│   ├── run_r05_v5style.py      # Multi-model approach
│   ├── run_r06_orig_fusion.py  # External data fusion
│   ├── run_r07_diverse.py      # Diverse model set
│   ├── run_r07_pseudo_label.py # First pseudo-labeling
│   ├── run_r08_pseudo_simple.py
│   ├── run_r09_10model_pseudo.py  # Best: LB=0.97785
│   ├── run_r10_diverse_models.py
│   ├── run_r11_iterative_pseudo.py
│   ├── run_r12_4model_iterative.py
│   └── run_r12_fast_iterative_pseudo.py
├── src/                        # Shared modules
│   ├── config.py
│   ├── data/
│   ├── features/
│   ├── models/
│   └── utils/
├── notebooks/                  # EDA notebooks
├── outputs/                    # Submissions + saved models
└── PLAN.md                     # Project plan
```

## Experiment Results

| Version | CV | LB | Key Technique |
|---------|-----|-----|---------------|
| R01 baseline | — | 0.97476 | LightGBM baseline |
| R02 | — | 0.97586 | Target encoding |
| R04 | — | 0.97656 | Pairwise TE (135 features) |
| R05 | — | 0.97730 | Multi-model + threshold opt |
| R08 | 0.97899 | 0.97782 | 3-model pseudo-labeling |
| **R09** | **0.97961** | **0.97785** | **10-model + pseudo + stacking** |
| R12 | 0.97913 | 0.97742 | 4-model × 3-round iterative pseudo |

**Best LB: 0.97785 (R09)**

## Key Findings

1. **Target encoding is the most impactful feature engineering** — Pairwise TE added 0.001-0.002 CV
2. **Pseudo-labeling helps** (+0.001 LB) but only with high confidence threshold (0.90+)
3. **Iterative pseudo-labeling hurts** — Multiple rounds with decreasing thresholds introduce noise
4. **Stacking > Hill climbing** — LR stacking on OOF predictions beats greedy hill climbing
5. **More models ≠ better LB** — R09 (10 models) ≈ R08 (3 models) on LB; R12 (16 models) worse
6. **CV-LB gap is significant** (~0.00176) — Overfitting to OOF predictions is a real concern

## Best Approach (R09)

1. Features: 38 base + 135 pairwise TE + TE_ORIG = 173 total features
2. Models: 3 XGB + 6 LGB + 1 CB = 10 models (stage 1: no pseudo, stage 2: with pseudo)
3. Pseudo-labeling: single round, threshold=0.9, weight=0.5x, 95.7% test coverage
4. Ensemble: LR stacking on OOF predictions
5. Post-processing: threshold optimization for balanced accuracy

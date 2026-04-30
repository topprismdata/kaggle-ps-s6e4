# Kaggle Playground Series S6E4: Predicting Irrigation Need

3-class classification (Low/Medium/High) to predict irrigation requirements.
Evaluation metric: **Balanced Accuracy**.

## Competition Info

- **URL**: https://www.kaggle.com/competitions/playground-series-s6e4
- **Duration**: 2026-04-01 to 2026-04-30
- **Train**: 630,000 rows x 20 features
- **Test**: 270,000 rows x 19 features
- **External data**: 10,000 rows (original dataset)
- **Class imbalance**: Low=58.7%, Medium=37.9%, High=3.3%

## Project Structure

```
kaggle-ps-s6e4/
├── scripts/
│   ├── run_r01_baseline.py          # LightGBM baseline
│   ├── run_r02_target_encoding.py   # Target encoding
│   ├── run_r03_formula.py           # Feature formula
│   ├── run_r04_pairwise_te.py       # Pairwise target encoding
│   ├── run_r05_v5style.py           # Multi-model approach
│   ├── run_r06_orig_fusion.py       # External data fusion
│   ├── run_r07_diverse.py           # Diverse model set
│   ├── run_r07_pseudo_label.py      # First pseudo-labeling
│   ├── run_r08_pseudo_simple.py     # Simplified pseudo-labeling
│   ├── run_r09_10model_pseudo.py    # 10-model + pseudo + stacking
│   ├── run_r10_diverse_models.py    # Diverse model experiments
│   ├── run_r11_iterative_pseudo.py  # Iterative pseudo-labeling
│   ├── run_r12_4model_iterative.py  # 4-model iterative pseudo
│   ├── run_r13_top_techniques.py    # Top technique combination
│   ├── run_r14_logit_formula.py     # Logit + formula features
│   ├── run_r15_model_diversity.py   # 13-model ensemble (best self-trained)
│   └── run_r18_framework.py         # MLOps framework-validated (incomplete)
├── src/
│   ├── config.py
│   ├── data/
│   ├── features/
│   ├── models/
│   └── utils/
├── outputs/
│   └── submissions/
└── notebooks/
```

## Experiment Results

| Version | CV | Public LB | Key Technique | Notes |
|---------|-----|-----------|---------------|-------|
| R01 | — | 0.97476 | LightGBM baseline | Starting point |
| R02 | — | 0.97586 | Target encoding | +0.001 |
| R04 | — | 0.97656 | Pairwise TE (135 features) | +0.001 |
| R05 | — | 0.97730 | Multi-model + threshold opt | +0.001 |
| R08 | 0.97899 | 0.97782 | 3-model pseudo-labeling | Pseudo helps |
| R09 | 0.97961 | 0.97785 | 10-model + pseudo + stacking | Best self-trained single-script |
| R12 | 0.97913 | 0.97742 | 4-model x 3-round iterative pseudo | Worse: iterative pseudo hurts |
| R13 | 0.97948 | 0.97750 | Stacking + thresh (top techniques) | |
| R13 | 0.97812 | 0.97671 | Weighted avg | |
| R14 | 0.97885 | 0.97720 | Logit + formula + thresh | |
| R15 | 0.97961 | 0.97847 | 13-model (3XGB+6LGB+3CB+1HGB) stacking+thresh | Best self-trained |
| R15 | 0.97902 | 0.97781 | Hill climbing ensemble | |
| R16 | — | 0.97901 | 5-way soft vote (R15x3 + 2 public) | First external blend |
| R16 | — | 0.97888 | 3-public weighted blend | |
| R16 | — | 0.97897 | Weighted blend (2 public) | |
| R16 | — | 0.97809 | Mega blend 7 sources | Too many sources |
| R17 | — | **0.98150** | Schema8 + formula prediction | **BEST** |
| R17 | — | 0.98148 | Nina schema8 (RealMLP + H overrides) | |
| R17 | — | 0.98145 | Top-4 vote + best single source | |
| R17 | — | 0.98144 | 8-source ensemble / formula enhanced | |
| R17 | — | 0.98141 | Top-3 diverse vote | |
| R17 | — | 0.98121 | Schema8 extended | |
| R17 | — | 0.98115 | 23-source majority vote | Too many sources |
| R17 | — | 0.98114 | Mohit transfer learning | |
| R17 | — | 0.98129 | Top-7 exp-weighted vote | |
| R17 | — | 0.98092 | NN-only (RealMLP) | Unique signal |

**Final Best: Public LB 0.98150 (R17 schema8_formula)**

### Progression Summary

```
Self-trained models:     R01 (0.97476) → R15 (0.97847)   [+0.00371, 15 experiments]
External integration:   R15 (0.97847) → R17 (0.98150)   [+0.00303, 2 experiments]
```

External prediction integration matched the entire self-trained improvement in 2 experiments vs 15.

## Key Findings

### 1. Pairwise Target Encoding is the Most Impactful Feature Engineering
- 8 categorical x 11 numerical = 171 interaction pairs
- 135 survived uniqueness filter (< 50% unique values)
- Added 0.001-0.002 CV improvement in a single step

### 2. Pseudo-Labeling Helps Only with High Confidence Threshold
- Single round, threshold=0.90, weight=0.5x: +0.001 LB
- Iterative multi-round (R12): WORSE — decreasing thresholds introduce noise
- Coverage: 95.7% of test set labeled at 0.90 threshold

### 3. External Prediction Integration is the Dominant Strategy
- R16 (first external blend): +0.0005 over best self-trained
- R17 (Nina's 23 prediction sources): +0.003 over best self-trained
- Self-trained models plateaued at ~0.9785; external sources jumped to 0.9815

### 4. Ensemble Source Quality > Quantity (Signal Dilution Principle)
- 23-source majority vote: 0.98115
- Top-4 vote: 0.98145 (+0.00030 with fewer sources)
- Best schema8 + formula: 0.98150 (2-3 high-quality sources)
- Adding low-quality sources dilutes the consensus signal

### 5. Self-Trained Models Contributed Minimally to Final Score
- R15 (best self-trained, 13 models): LB 0.97847
- When blended into R17's external ensemble: contributed ~0.00003
- The 14x larger training effort produced marginal improvement over external sources

### 6. Stacking > Hill Climbing for Self-Trained Models
- LR stacking on OOF predictions consistently beat greedy hill climbing
- But both were far below external ensemble quality

### 7. CV-LB Gap is Significant (~0.00176)
- R09 CV 0.97961 vs LB 0.97785
- Overfitting to OOF predictions is a real concern
- R17 bypassed this gap entirely by using external predictions

## Best Approaches

### Best Self-Trained (R15)

1. Features: 38 base + 135 pairwise TE + TE_ORIG = 173 total
2. Models: 3 XGB + 6 LGB + 3 CB + 1 HGB = 13 models
3. Two-stage: Stage 1 (no pseudo) -> Stage 2 (pseudo, weight=0.5x)
4. Ensemble: Hill climbing + LR stacking + simple avg -> pick best
5. Post-processing: threshold optimization for balanced accuracy

### Best Overall (R17 schema8_formula)

1. Source: Nina's collection of 23 high-quality prediction CSVs
2. Method: Label voting on top 2-3 sources with formula-based refinement
3. Key sources: RealMLP neural network (unique signal) + specific high-LB submissions
4. Formula prediction: Systematic override on low-confidence samples

## Lessons Learned

### For Tabular Classification Competitions

1. **Start with pairwise TE** — It's the single biggest feature engineering win
2. **Build multi-model ensemble early** — Model diversity > hyperparameter tuning
3. **Pseudo-label with caution** — Single round, high threshold only
4. **Integrate external predictions ASAP** — The biggest LB jumps come from external data
5. **Source selection matters** — 4 high-quality sources beat 23 mediocre ones
6. **Don't over-invest in self-training** — If external sources are available, leverage them

### For MLOps Framework Validation

- R18 framework script validated import patterns and stage structure
- Key gotcha: `sys.path` ordering matters when local and shared configs share names
- Framework components (logging, validation, submission) worked correctly
- `max_bin=1024` in XGBoost causes extreme slowdown; use 256 for iteration speed

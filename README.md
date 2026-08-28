<p align="center">
  <img src="https://raw.githubusercontent.com/topprismdata/.github/main/assets/brand/topprism-repo-header.png" alt="TopPrism dual-prism visual" width="100%" />
</p>

# Kaggle PS S6E4 --- Capability Formation Case

> **Language / 语言:** English primary · 中文概览如下。
>
> ### 中文概览
> 用于训练和评测 Cultivating ML Agent 的公开分类基准项目，记录实验、失败分析和能力沉淀。


**A public classification benchmark used to train and evaluate
TopPrism's Cultivating ML Agent.**

`LEARNING PROJECT` · `TABULAR ML` · `KAGGLE BENCHMARK`

> This is **not a customer product**. It is a measurable environment for
> experimentation, failure analysis, ensemble design, and knowledge
> crystallization.

------------------------------------------------------------------------

## Task

Three-class irrigation-need classification with Balanced Accuracy.

The repository records a progression from a LightGBM baseline through
target encoding, pseudo-labeling, diverse ensembles, and external
prediction integration.

------------------------------------------------------------------------

## Current result metadata --- fix immediately

The GitHub About text currently says **Best LB = 0.97785**, while the
README reports later results up to **0.98150**.

Unify:

-   README;
-   GitHub About;
-   badges;
-   any project index.

Recommended About:

> Learning project for Cultivating ML Agent --- Kaggle PS S6E4 tabular
> classification; experiment history through best reported Public LB
> 0.98150.

------------------------------------------------------------------------

## What this project taught

The README already contains a more valuable story than the leaderboard
number:

-   pairwise target encoding produced meaningful gains;
-   iterative pseudo-labeling can degrade performance;
-   ensemble source quality matters more than source count;
-   self-trained models plateaued before external-prediction
    integration;
-   diverse signals can matter more than adding more similar models.

These lessons are reported first on this page.

------------------------------------------------------------------------

## Experiment history

The full R01--R18 experiment log is preserved verbatim in
[`docs/competition-log.md`](docs/competition-log.md) as project history.
This README records only the headline progression:

``` text
Baseline (R01 LightGBM)
-> best self-trained (R09 10-model + pseudo + stacking, LB 0.97785)
-> key failed experiment (R12 iterative pseudo-labeling hurt performance)
-> best externally-assisted (R17 Schema8 + formula prediction, LB 0.98150)
-> reusable skills crystallized for the Cultivating ML Agent
```

------------------------------------------------------------------------

## External predictions

Because later results integrate external prediction sources, clearly
separate:

-   self-trained performance;
-   externally assisted / blended performance.

Do not imply the final LB is produced solely by the repository's own
trained models.

------------------------------------------------------------------------

## TopPrism metadata

``` yaml
topprism:
  purpose: learning-project
  capability: tabular-ml
  maturity: learning
  evidence:
    type: kaggle-benchmark
    best_reported_public_lb: 0.98150
  parent:
    - cultivating-ml-agent
```

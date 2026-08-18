# Kaggle PS S6E4 --- Capability Formation Case

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

These should become the first screen.

------------------------------------------------------------------------

## Experiment history

Do not keep R01--R18 as the main README body.

Move to:

`docs/experiment-log.md`

README should show:

``` text
Baseline
→ best self-trained
→ key failed experiment
→ best final
→ skills / principles extracted
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

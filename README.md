# Statistical Model Selection Pipeline

A three-stage statistical model-selection pipeline for identifying a best-performing machine learning model from repeated Monte Carlo Cross-Validation (MCCV) results.

Rather than selecting a model solely because it has the highest mean performance, the pipeline evaluates models through three sequential questions:

1. **Temporal/generalization stability:** Does performance on the held-out test split differ significantly from the corresponding validation performance?
2. **Global model differences:** Do the candidate models differ significantly overall?
3. **Compare-to-best superiority:** Is the numerically top-ranked model significantly better than the competing models after multiple-comparison correction?

The framework is **metric-agnostic** and can be applied to repeated-CV benchmarking studies involving multiple machine learning classifiers.

It was developed for a spatiotemporal public-health morbidity classification study involving seven candidate models:

* Logistic Regression (LR)
* Random Forest (RF)
* Support Vector Machine (SVM)
* Extreme Gradient Boosting (XGB)
* Light Gradient Boosting Machine (LGBM)
* CatBoost (CAT)
* Stacked Ensemble (STACK)

The original application used **AUC** as the primary model-selection metric.

---

## Why Not Simply Select the Model with the Highest Mean Score?

Selecting the model with the highest mean AUC, accuracy, F1-score, or another metric does not establish that the observed difference is statistically meaningful.

A model may achieve the highest average score because of resampling variability, while another model may have:

* similar performance,
* greater temporal stability,
* lower validation-to-test drift, or
* statistically indistinguishable performance.

This pipeline therefore separates model selection into three complementary stages:

> **Stability → Global comparison → Compare-to-best selection**

The objective is not merely to identify the numerically highest-performing model, but to provide a reproducible statistical procedure for selecting a model after considering stability and statistical evidence.

---

# Pipeline Overview

```text
                 Repeated MCCV Results
                         │
                         ▼
              ┌──────────────────────┐
              │ Step 1               │
              │ Temporal Stability   │
              │ Test vs. Validation  │
              └──────────┬───────────┘
                         │
              Stable candidate models
                         │
                         ▼
              ┌──────────────────────┐
              │ Step 2               │
              │ Friedman Omnibus Test│
              │ Global comparison    │
              └──────────┬───────────┘
                         │
                ┌────────┴─────────┐
                │                  │
              p ≥ α              p < α
                │                  │
                ▼                  ▼
       No significant       ┌──────────────────┐
       difference detected  │ Step 3           │
                │           │ Conover          │
                │           │ compare-to-best  │
                │           └────────┬─────────┘
                │                    │
                └──────────┬─────────┘
                           ▼
                     Selected Model
```

---

# Statistical Model-Selection Algorithm

## Input

Let

$$
M = {M_1, M_2, \ldots, M_k}
$$

denote the set of candidate machine learning models, where $k \ge 3$.

Let:

* $N$ = number of MCCV iterations, typically $N = 100$
* $\Phi$ = set of evaluated performance metrics
* $\alpha$ = significance level, typically $0.05$
* $S_{\text{val}}$ = validation performance
* $S_{\text{test}}$ = held-out test performance

The primary selection metric can be AUC, Accuracy, Kappa, or another appropriate metric.

---

## Step 1 — Temporal/Generalization Stability

For each model $M_i$, the pipeline compares the model's validation and held-out test performance across the same MCCV iterations.

For iteration $r$:

$$
\Delta_{ir} = S_{\text{test},ir} - S_{\text{val},ir}
$$

The paired validation and test scores are compared using a paired t-test.

### Decision Rule

If $p_i \ge \alpha$, the model is retained as a **stable candidate**, meaning that the analysis did not detect a statistically significant validation-to-test performance difference.

If $p_i < \alpha$, the model is excluded from the temporally stable candidate pool because statistically significant performance drift was detected.

### Important Interpretation

A non-significant result should **not** be interpreted as proof that the two performances are identical.

Instead, it indicates:

> No statistically significant validation-to-test performance drift was detected under the specified resampling procedure.

This distinction avoids interpreting failure to reject the null hypothesis as formal evidence of equivalence.

---

## Step 2 — Global Model Comparison

The temporally stable models are then compared using the **Friedman test**.

The test evaluates whether there is an overall difference in the performance ranks of the candidate models across the aligned MCCV iterations.

### Null Hypothesis

$$
H_0:\text{The candidate models have equivalent performance ranks.}
$$

### Alternative Hypothesis

$$
H_1:\text{At least one candidate model differs in performance.}
$$

If $p \ge \alpha$, the null hypothesis is not rejected.

In this case, the analysis reports that **no statistically significant difference was detected among the candidate models**, and the model with the highest mean value of the selected metric is chosen as the practical winner.

This is a **ranking-based selection**, not a claim that all models have been formally proven equivalent.

If $p < \alpha$, the analysis proceeds to Step 3.

---

## Step 3 — Compare-to-Best Post-Hoc Selection

When the Friedman test detects a statistically significant difference among the candidate models, the model with the highest mean validation performance is identified as the empirical top-ranked model:

$$
M_{\text{top}} =
\arg\max_{M_i} \overline{S}_{\text{val},i}
$$

The top-ranked model is then compared against every alternative candidate using the **Conover post-hoc test**.

Only the comparisons

$$
M_{\text{top}} \text{ vs. } M_j
$$

are performed.

This is a **compare-to-best** strategy rather than a full all-pairs post-hoc analysis.

### Multiple-Comparison Correction

If there are $k$ candidate models, the number of comparisons is:

$$
m = k - 1
$$

A Bonferroni-adjusted significance level is therefore used:

$$
\alpha_{\text{adj}} = \frac{\alpha}{k-1}
$$

For each comparison:

$$
p < \alpha_{\text{adj}}
$$

indicates that the top-ranked model is statistically distinguishable from that alternative.

If the top-ranked model is not significantly different from one or more alternatives, those models remain in the top-performing candidate set.

The final model is selected as the highest-ranked model among the remaining statistically indistinguishable candidates.

---

# Formal Algorithm

```text
Require:
    M = {M1, ..., Mk}: candidate ML models, k >= 3
    N: number of MCCV iterations
    Phi: evaluated performance metrics
    alpha: significance level (default = 0.05)

Step 1: Temporal/Generalization Stability

    M_temporal <- empty set

    For each model Mi in M:

        Compute Delta_i:
            Delta_i[r] = Score_test_i[r] - Score_val_i[r]
            for r = 1, ..., N

        Perform a paired t-test comparing:
            Score_test_i and Score_val_i

        If p_i >= alpha:
            Add Mi to M_temporal

    If |M_temporal| < 3:

        Rank all models in M by mean validation score
        Select the top 3 models
        Set M_temporal <- top 3 models

    Return M_temporal


Step 2: Global Comparison

    M_candidate <- M_temporal

    Perform a Friedman test using the
    iteration-aligned validation scores of
    M_candidate.

    If p >= alpha:

        No statistically significant difference
        is detected among candidate models.

        Select the model with the highest
        mean validation score.

        Return M_best

    Otherwise:

        Proceed to Step 3


Step 3: Compare-to-Best Post-Hoc Selection

    If |M_candidate| == 1:

        M_best <- the single candidate model

    Else:

        Identify M_top as the model with the
        highest mean validation score.

        m <- |M_candidate| - 1
        alpha_adj <- alpha / m

        For each M_x in M_candidate except M_top:

            Perform Conover post-hoc comparison
            between M_top and M_x.

            If p < alpha_adj:

                M_x is statistically distinguishable
                from M_top.

            Otherwise:

                Retain M_x in the top-performing set.

        M_best <- highest-ranked model among
                  the retained candidates.

    Return M_best
```

---

# Minimum Number of Models

The pipeline requires at least three candidate models.

| Number of Models | Status        | Interpretation                                                                                         |
| ---------------: | ------------- | ------------------------------------------------------------------------------------------------------ |
|            $k=2$ | Not supported | The global Friedman stage provides little additional information beyond a two-model paired comparison. |
|            $k=3$ | Supported     | Minimum operating configuration; candidate rankings are based on a small model pool.                   |
|            $k=4$ | Supported     | Valid, but still a relatively small comparison set.                                                    |
|         $k\geq5$ | Recommended   | Provides a broader benchmarking setting and a more informative compare-to-best analysis.               |

The implementation raises `InsufficientModelsError` when fewer than three models are supplied.

For $3 \leq k < 5$, the implementation may issue a `UserWarning` indicating that the candidate pool is relatively small.

The recommendation of $k \geq 5$ should be understood as a **practical benchmarking recommendation**, rather than a universal statistical requirement.

---

# Metric Selection

The pipeline is metric-agnostic.

The default metric in the original public-health application is **AUC** because the models produce continuous prediction scores and AUC provides threshold-independent discrimination.

For a binary classifier:

$$
AUC \in [0,1]
$$

and larger values indicate better discrimination.

However, the appropriate metric depends on the prediction task.

| Model/Application                | Possible Selection Metric                              |
| -------------------------------- | ------------------------------------------------------ |
| Probabilistic binary classifier  | AUC                                                    |
| Threshold-based classifier       | Accuracy                                               |
| Imbalanced classification        | F1, balanced accuracy, MCC, or AUC                     |
| Agreement-focused classification | Kappa                                                  |
| Rule-based/symbolic classifier   | Accuracy, Kappa, F1, or another threshold-based metric |

### Rule-Based Models

AUC should not automatically be used for a deterministic rule-set or symbolic classifier if the model does not produce meaningful continuous prediction scores.

For example, when evaluating:

* surrogate decision trees,
* symbolic rule systems,
* FastSRS,
* deterministic if-then classifiers,

a threshold-based metric such as Accuracy, Kappa, F1, or balanced accuracy may be more appropriate.

The metric can be specified through the `metric=` argument of `select_best_model()`.

---

# Installation

Clone the repository and install the required dependencies:

```bash
git clone <this-repo-url>
cd model-selection-pipeline
pip install -r requirements.txt
```

---

# Usage

## Quick Start

Run the synthetic demonstration:

```bash
python example_usage.py
```

The example generates synthetic MCCV results for seven models and executes the complete three-stage model-selection pipeline.

The example produces:

```text
example_output/
├── table1_tradeoff_analysis.csv
└── table2_model_selection.csv
```

---

## Using Your Own MCCV Results

The pipeline expects iteration-aligned validation and test results.

A typical input file contains:

```text
Split,AUC,Accuracy,Sensitivity,Specificity
Val,0.82,0.76,0.71,0.80
Val,0.84,0.78,0.74,0.81
...
Test,0.80,0.74,0.69,0.78
Test,0.83,0.77,0.72,0.80
...
```

There should be $N$ observations for each split.

The validation and test observations must correspond to the **same MCCV iteration index**.

For example:

```text
Iteration 1:
    Validation -> row 1
    Test       -> row 1

Iteration 2:
    Validation -> row 2
    Test       -> row 2

...

Iteration 100:
    Validation -> row 100
    Test       -> row 100
```

This iteration alignment is essential for the paired validation-test comparison.

---

## Example

```python
from data import build_csv_paths, load_model_data
from evaluation import build_tradeoff_table, select_best_model

MODELS = [
    "LR",
    "RF",
    "SVM",
    "XGB",
    "LGBM",
    "CAT",
    "STACK",
]

METRICS = [
    "AUC",
    "Accuracy",
    "Sensitivity",
    "Specificity",
]

csv_paths = build_csv_paths(
    metrics_dir="results/metrics",
    models=MODELS,
)

model_data = load_model_data(
    csv_paths,
    metrics=METRICS,
)

table1 = build_tradeoff_table(
    model_data,
    MODELS,
    METRICS,
)

result = select_best_model(
    model_data=model_data,
    models=MODELS,
    metric="AUC",
    n_iterations=100,
    alpha=0.05,
)

print("Selected model:", result.m_best)
print(result.table_selection_summary)
```

---

# Output

The model-selection function returns the selected model together with statistical and descriptive information used during the selection process.

The selection summary can be used to document:

* validation performance,
* test performance,
* validation-test differences,
* temporal stability results,
* Friedman test results,
* Conover post-hoc comparisons,
* adjusted significance levels,
* candidate-model status, and
* final model selection.

This makes the selection process reproducible and auditable rather than relying only on a manually reported maximum mean score.

---

# Interpretation of the Final Model

The pipeline distinguishes between two situations.

## Case 1 — No Significant Global Difference

If the Friedman test gives:

$$
p \geq \alpha
$$

the analysis reports:

> No statistically significant difference was detected among the candidate models.

The model with the highest mean performance is then selected as the **practical winner**.

This should not be described as proof that the selected model is statistically superior.

---

## Case 2 — Significant Global Difference

If:

$$
p < \alpha
$$

the Conover post-hoc analysis determines whether the empirical top-ranked model is statistically distinguishable from each alternative.

If the top model significantly outperforms all alternatives after correction, it can be described as a **statistically supported winner** under the specified procedure.

If one or more alternatives are not significantly different from the top model, the final selection is made from the highest-performing statistically indistinguishable candidate set.

---

# Why This Pipeline Is Useful

The pipeline is designed to avoid three common weaknesses in model benchmarking.

### 1. Ignoring Validation-to-Test Drift

A model can perform well during validation but deteriorate on the corresponding held-out test split.

Step 1 explicitly examines this difference.

### 2. Treating Numerical Differences as Statistical Differences

A model with AUC = 0.881 is not necessarily statistically superior to a model with AUC = 0.878.

Step 2 evaluates whether an overall difference exists.

### 3. Performing Unrestricted Pairwise Testing

Testing every model against every other model increases the number of comparisons.

Step 3 instead uses a predefined **compare-to-best** strategy and applies a Bonferroni correction across the $k-1$ comparisons involving the empirical top-ranked model.

---

# Statistical Tests

The pipeline combines the following procedures.

## Paired t-Test

Used to compare iteration-aligned validation and held-out test performance for each model.

For each model:

$$
\Delta_r = S_{\text{test},r} - S_{\text{val},r}
$$

The test evaluates whether the mean validation-test difference differs significantly from zero.

A non-significant result indicates that statistically significant drift was not detected; it does not formally establish equivalence.

---

## Friedman Test

The Friedman test is used as the global nonparametric comparison of candidate models.

It is appropriate when the same resampling iterations provide paired performance observations across models.

The test evaluates whether the candidate models differ in their performance ranks.

---

## Conover Post-Hoc Test

When the Friedman test is significant, Conover's post-hoc procedure is used to compare the empirical top-ranked model with each alternative.

The number of comparisons is:

$$
m = k-1
$$

and the Bonferroni-adjusted significance threshold is:

$$
\alpha_{\text{adj}} = \frac{\alpha}{k-1}
$$

---

# Assumptions and Requirements

The pipeline assumes:

### 1. Iteration Alignment

The same MCCV iteration index must correspond to the same resampling/split across models.

### 2. Paired Observations

Model performance must be evaluated on corresponding MCCV iterations so that model comparisons are paired.

### 3. Repeated Resampling

The procedure is designed for repeated MCCV or a similar repeated resampling framework.

### 4. Appropriate Metric

The selected metric must be meaningful for the underlying prediction task.

### 5. Predefined Selection Metric

The primary metric should be specified before model selection rather than chosen after observing the results.

---

# Important Statistical Considerations

## Non-Significance Is Not Equivalence

The pipeline intentionally avoids interpreting:

$$
p \geq 0.05
$$

as proof that two models are equivalent.

Instead, the correct interpretation is:

> The analysis did not detect a statistically significant difference at the specified significance level.

For applications where formal equivalence is important, an equivalence-testing framework with a predefined practical equivalence margin may be preferable.

---

## MCCV Iterations Are Resampled Observations

Although 100 MCCV iterations are commonly used to characterize performance variability, the resulting observations arise from overlapping resamples and should not automatically be interpreted as 100 independent experimental datasets.

Therefore, statistical results should be interpreted as **resampling-based evidence** rather than as evidence from 100 independent studies.

---

## Compare-to-Best Versus All-Pairs Testing

This pipeline does not perform every possible pairwise comparison.

For $k$ models:

* Full pairwise comparison requires

$$
\frac{k(k-1)}{2}
$$

comparisons.

* The compare-to-best strategy requires only

$$
k-1
$$

comparisons.

For seven models:

$$
\frac{7(7-1)}{2}=21
$$

full pairwise comparisons would be required, whereas the proposed approach performs:

$$
7-1=6
$$

top-versus-alternative comparisons.

A Bonferroni correction is applied across these six comparisons.

The procedure is structurally analogous to a **control-versus-all** comparison design, although it does not implement Dunnett's exact multivariate procedure.

---

# Recommended Reporting Format

A manuscript using this pipeline should report the selection process transparently.

A concise reporting structure is:

1. **MCCV design:** number of iterations and resampling strategy.
2. **Primary metric:** e.g., AUC.
3. **Temporal stability:** validation-test comparison for each model.
4. **Candidate pool:** models retained after stability screening.
5. **Global test:** Friedman statistic and p-value.
6. **Post-hoc test:** Conover comparisons and multiplicity adjustment.
7. **Final selection:** selected model and rationale.
8. **Descriptive performance:** mean ± SD or another appropriate summary.

For example:

```text
Model selection was performed in three stages. First, validation and
held-out test AUC values were compared across the 100 aligned MCCV
iterations using paired t-tests to identify models without statistically
significant validation-to-test drift. Second, the retained models were
compared using the Friedman test. When the omnibus test was significant,
Conover post-hoc comparisons were performed between the highest-ranked
model and each alternative, with Bonferroni adjustment for the k−1
comparisons. When no significant global difference was detected, the
model with the highest mean validation AUC was selected as the practical
winner.
```

---

# Repository Structure

```text
model-selection-pipeline/
│
├── data.py
│   └── Generic MCCV result loader
│
├── evaluation.py
│   └── Core statistical model-selection procedures
│
├── example_usage.py
│   └── Runnable synthetic-data demonstration
│
├── requirements.txt
│   └── Python dependencies
│
├── LICENSE
│   └── MIT License
│
└── README.md
    └── Documentation
```

---

# Reproducibility

For reproducible benchmarking, the MCCV procedure should use:

* a fixed random-seed strategy,
* predefined train/test proportions,
* iteration-aligned splits across candidate models,
* consistent preprocessing,
* consistent hyperparameter-tuning procedures, and
* a predefined primary performance metric.

For example, a 100-iteration MCCV protocol may use:

```python
random_state = 42 + iteration
```

where:

```text
iteration = 0, 1, ..., 99
```

This allows the same resampling sequence to be reconstructed.

---

# Scope

This repository is designed for **statistical validation and selection of already-trained/evaluated machine learning models**.

It does not:

* train machine learning models,
* automatically tune hyperparameters,
* perform feature selection,
* replace a complete cross-validation framework,
* perform AutoML,
* guarantee model generalizability, or
* establish formal equivalence between models.

Its purpose is to provide a structured statistical layer **after model evaluation**.

---

# Limitations

The current framework has several limitations.

### Temporal Stability Criterion

The paired t-test identifies statistically significant validation-test drift but does not constitute a formal equivalence test.

### MCCV Dependence

Repeated MCCV observations may be correlated because different iterations can contain overlapping observations.

### Candidate-Model Selection

The same performance results are used to rank the empirical top model and conduct the compare-to-best comparisons. Therefore, the resulting post-hoc evidence should be interpreted as part of a predefined model-selection procedure rather than as an independent confirmatory test.

### Fallback Procedure

When fewer than three models satisfy the temporal stability criterion, the implementation uses the top three models ranked by mean validation performance to maintain a viable candidate pool for the subsequent comparison stage.

These fallback models should therefore be interpreted as **fallback candidates**, not as models independently demonstrated to be temporally stable.

### Metric Dependence

Model-selection conclusions depend on the selected primary metric. A model that is optimal according to AUC may not be optimal according to accuracy, sensitivity, specificity, F1-score, or Kappa.

---

# Associated Publication

This pipeline was developed for and used in the following manuscript, currently **under review**:

```bibtex
@inproceedings{hossain2026-icece-2026-1,
  author    = {Maniruzzaman, M. and Hossain, M. G. and Asadujjaman, M.},
  title     = {Spatiotemporal Validation and Explainable AI for Childhood
               Morbidity Classification},
  booktitle = {IEEE ICECE 2026},
  year      = {2026},
  note      = {Under review}
}
```

The citation will be updated with the final proceedings information, volume, pages, and DOI following acceptance and publication.

---

# Statistical References

If you use this pipeline, please cite the statistical procedures on which it is based.

### Friedman Test

Friedman, M. (1937). The use of ranks to avoid the assumption of normality implicit in the analysis of variance. *Journal of the American Statistical Association*, 32(200), 675–701.

### Conover Post-Hoc Procedure

Conover, W. J., & Iman, R. L. (1979). *On Multiple-Comparisons Procedures*. Technical Report LA-7677-MS, Los Alamos Scientific Laboratory.

### Statistical Comparison of Classifiers

Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1–30.

---

# Citation

If this repository or the model-selection methodology is used in your research, please cite the associated manuscript and the underlying statistical methods.

```bibtex
@inproceedings{hossain2026-icece-2026-1,
  author    = {Maniruzzaman, M. and Hossain, M. G. and Asadujjaman, M.},
  title     = {Spatiotemporal Validation and Explainable AI for Childhood
               Morbidity Classification},
  booktitle = {IEEE ICECE 2026},
  year      = {2026},
  note      = {Under review}
}
```

---

# License

This project is released under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

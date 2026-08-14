from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import build_tradeoff_table, select_best_model

OUTPUT_DIR = Path("example_output")
OUTPUT_DIR.mkdir(exist_ok=True)

MODELS = ["LR", "RF", "SVM", "XGB", "LGBM", "CAT", "STACK"]
METRICS = ["AUC", "Accuracy", "Sensitivity", "Specificity"]
METRIC_LABELS = {"AUC": "AUC", "Accuracy": "ACC", "Sensitivity": "SE", "Specificity": "SPE"}
N_ITERATIONS = 100
SEED = 42


def make_synthetic_model_data(rng: np.random.Generator):
    base_auc = {
        "LR": 0.71, "RF": 0.76, "SVM": 0.73, "XGB": 0.77,
        "LGBM": 0.775, "CAT": 0.80, "STACK": 0.785,
    }
    model_data = {}
    for model in MODELS:
        mu = base_auc[model]
        val_auc = np.clip(rng.normal(mu, 0.015, N_ITERATIONS), 0, 1)
        # Small, model-specific test/val drift so Step 1 has something to filter on.
        drift = rng.normal(0.0, 0.004, N_ITERATIONS)
        test_auc = np.clip(val_auc + drift, 0, 1)

        def derive(scores, jitter):
            return np.clip(scores + rng.normal(0, jitter, N_ITERATIONS), 0, 1)

        model_data[model] = {
            "val": {
                "AUC": val_auc,
                "Accuracy": derive(val_auc, 0.01),
                "Sensitivity": derive(val_auc, 0.02),
                "Specificity": derive(val_auc, 0.02),
            },
            "test": {
                "AUC": test_auc,
                "Accuracy": derive(test_auc, 0.01),
                "Sensitivity": derive(test_auc, 0.02),
                "Specificity": derive(test_auc, 0.02),
            },
        }
    return model_data


def main():
    rng = np.random.default_rng(SEED)
    model_data = make_synthetic_model_data(rng)

    # Table 1: Test/Val tradeoff per model per metric
    table1 = build_tradeoff_table(model_data, MODELS, METRICS, METRIC_LABELS)
    table1.to_csv(OUTPUT_DIR / "table1_tradeoff_analysis.csv", index=False)

    # Run the full three-step selection pipeline on AUC
    result = select_best_model(
        model_data=model_data,
        models=MODELS,
        metric="AUC",
        n_iterations=N_ITERATIONS,
        alpha=0.05,
        min_stable_models=3,
    )
    result.table_selection_summary.to_csv(
        OUTPUT_DIR / "table2_model_selection.csv", index=False
    )

    print(f"M_temporal: {result.m_temporal}  (fallback used: {result.fallback_used})")
    print(f"Ranked M_candidate: {result.ranked_candidates}")
    print(f"Friedman p = {result.friedman_p:.4e}  (early return: {result.early_return})")
    print(f"M_best = {result.m_best}")
    print(f"\nTables written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

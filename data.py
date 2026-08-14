from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]


def build_csv_paths(
    metrics_dir: PathLike,
    models: Sequence[str],
    filename_template: str = "{model}_mc_results.csv",
) -> Dict[str, Path]:
    metrics_dir = Path(metrics_dir)
    return {
        model: metrics_dir / model / filename_template.format(model=model)
        for model in models
    }


def load_model_data(
    csv_paths: Dict[str, PathLike],
    metrics: Sequence[str],
    split_col: str = "Split",
    test_label: str = "Test",
    val_label: str = "Val",
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    model_data: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

    for model, path in csv_paths.items():
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No results file found for '{model}' at: {path}")

        df = pd.read_csv(path)

        missing_cols = [c for c in [split_col, *metrics] if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"'{path}' is missing required column(s): {missing_cols}. "
                f"Found columns: {list(df.columns)}"
            )

        test_df = df[df[split_col] == test_label]
        val_df = df[df[split_col] == val_label]

        if test_df.empty or val_df.empty:
            raise ValueError(
                f"'{path}' must contain rows for both split='{test_label}' "
                f"and split='{val_label}'. Found unique values: "
                f"{df[split_col].unique().tolist()}"
            )
        if len(test_df) != len(val_df):
            raise ValueError(
                f"'{path}': Test split has {len(test_df)} rows but Val split "
                f"has {len(val_df)} rows. Paired t-test requires equal-length, "
                f"iteration-aligned arrays."
            )

        model_data[model] = {
            "test": {metric: test_df[metric].to_numpy() for metric in metrics},
            "val": {metric: val_df[metric].to_numpy() for metric in metrics},
        }

    return model_data


def check_iteration_count(
    model_data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    expected_n: int,
) -> None:
    """Warn-free hard check that every model/split/metric array has length expected_n."""
    for model, splits in model_data.items():
        for split_name, metric_dict in splits.items():
            for metric, arr in metric_dict.items():
                if len(arr) != expected_n:
                    raise ValueError(
                        f"{model}/{split_name}/{metric} has {len(arr)} values, "
                        f"expected {expected_n} (N_ITERATIONS)."
                    )

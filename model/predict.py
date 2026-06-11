"""Inference helpers for the Global Tech Salary Predictor.

Loads the trained artifact bundle and turns a dictionary of user inputs into a
calibrated salary range (low / median / high in USD). Kept separate from the
training module so the app depends only on lightweight inference code.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config  # pylint: disable=wrong-import-position


def load_model_bundle(path=config.MODEL_PATH):
    """Load the serialised model bundle, or explain how to create it."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. "
            "Train it first with: python model/train_model.py"
        )
    return joblib.load(path)


def build_input_frame(user_input, feature_options):
    """Build a one-row DataFrame with the training feature columns.

    Unknown ``job_title`` or ``company_location`` values are mapped to
    ``OTHER_BUCKET`` so they match the categories the model was trained on.
    """
    row = dict(user_input)
    for column in ("job_title", "company_location"):
        if row.get(column) not in feature_options.get(column, []):
            row[column] = config.OTHER_BUCKET
    ordered = {feature: row[feature] for feature in config.FEATURES}
    return pd.DataFrame([ordered], columns=config.FEATURES)


def predict_salary_range(bundle, user_input):
    """Predict a sorted low / median / high salary range for one input.

    Each quantile model is independent, so their raw predictions can cross.
    The three values are clamped at zero and sorted to guarantee that
    low <= median <= high.
    """
    frame = build_input_frame(user_input, bundle["feature_options"])
    models = bundle["models"]
    predictions = [
        max(0.0, float(models[name].predict(frame)[0]))
        for name in ("low", "median", "high")
    ]
    predictions.sort()
    low, median, high = predictions
    return {"low": low, "median": median, "high": high}


def convert_to_pln(amount_usd):
    """Convert a USD amount to PLN using the rough display-only rate."""
    return amount_usd * config.USD_TO_PLN

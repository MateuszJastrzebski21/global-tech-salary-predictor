"""Model training for the Global Tech Salary Predictor.

Trains three independent quantile gradient-boosting regressors (low / median /
high) on the cleaned salary dataset, evaluates them, and serialises a single
artifact bundle that the app loads for inference. Run the module directly to
(re)train and write ``model/artifacts/salary_model.joblib``.
"""

import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config  # pylint: disable=wrong-import-position
from data.prepare_data import prepare_dataset  # pylint: disable=wrong-import-position


def load_training_data():
    """Return the processed dataset, building it first if it does not exist."""
    if config.PROCESSED_DATA_PATH.exists():
        return pd.read_csv(config.PROCESSED_DATA_PATH)
    return prepare_dataset(save=True)


def build_pipeline(quantile):
    """Build a one-hot encoding + quantile gradient-boosting pipeline."""
    preprocessor = ColumnTransformer(
        [(
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            config.CATEGORICAL_FEATURES,
        )],
        remainder="passthrough",
    )
    regressor = HistGradientBoostingRegressor(
        loss="quantile", quantile=quantile, **config.MODEL_PARAMS
    )
    return Pipeline([("preprocess", preprocessor), ("regressor", regressor)])


def train_quantile_models(features, target):
    """Fit one pipeline per configured quantile and return them in a dict."""
    models = {}
    for name, quantile in config.QUANTILES.items():
        pipeline = build_pipeline(quantile)
        pipeline.fit(features, target)
        models[name] = pipeline
    return models


def evaluate_models(models, features_test, target_test):
    """Compute headline metrics for the trained quantile models.

    Returns R^2, MAE and RMSE of the median model against the truth, plus the
    interval coverage: the fraction of test targets that fall inside the
    predicted [low, high] band (the model's calibration claim).
    """
    median_pred = models["median"].predict(features_test)
    low_pred = models["low"].predict(features_test)
    high_pred = models["high"].predict(features_test)
    within = (target_test >= low_pred) & (target_test <= high_pred)
    return {
        "r2": float(r2_score(target_test, median_pred)),
        "mae": float(mean_absolute_error(target_test, median_pred)),
        "rmse": float(root_mean_squared_error(target_test, median_pred)),
        "interval_coverage": float(np.mean(within)),
    }


def extract_feature_options(df):
    """Return sorted unique values for each categorical feature + remote_ratio.

    The app builds its dropdown menus from this mapping so the choices always
    match the categories the model was trained on.
    """
    options = {}
    for feature in config.CATEGORICAL_FEATURES:
        options[feature] = sorted(df[feature].unique().tolist())
    options["remote_ratio"] = sorted(df["remote_ratio"].unique().tolist())
    return options


def build_artifact(models, metrics, options, sample_count):
    """Bundle models, metrics and metadata into one serialisable dict."""
    return {
        "models": models,
        "metrics": metrics,
        "feature_options": options,
        "sample_count": sample_count,
        "trained_on": date.today().isoformat(),
        "quantiles": config.QUANTILES,
    }


def main():
    """Train, evaluate and persist the salary model artifact."""
    df = load_training_data()
    features = df[config.FEATURES]
    target = df[config.TARGET]
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    models = train_quantile_models(features_train, target_train)
    metrics = evaluate_models(models, features_test, target_test)
    options = extract_feature_options(df)
    artifact = build_artifact(models, metrics, options, len(df))
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, config.MODEL_PATH)
    print(
        f"Trained on {len(df)} rows | "
        f"coverage={metrics['interval_coverage']:.3f} "
        f"r2={metrics['r2']:.3f} "
        f"mae=${metrics['mae']:,.0f} "
        f"rmse=${metrics['rmse']:,.0f} -> {config.MODEL_PATH}"
    )


if __name__ == "__main__":
    main()

"""Central configuration for the Global Tech Salary Predictor.

This module is the single source of truth for the project: every path,
feature list, model hyper-parameter and human-readable UI label lives here.
The data, model and app layers all import from this module so that changing
a constant in one place updates the whole pipeline consistently.
"""

from pathlib import Path

# --- Filesystem paths (all derived from this file's location) ----------------
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "salaries.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "salaries_clean.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "artifacts" / "salary_model.joblib"

# --- Modelling target and feature definitions --------------------------------
TARGET = "salary_in_usd"
CATEGORICAL_FEATURES = [
    "experience_level",
    "employment_type",
    "job_title",
    "company_location",
    "company_size",
]
NUMERIC_FEATURES = ["work_year", "remote_ratio"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# --- Rare-category bucketing -------------------------------------------------
OTHER_BUCKET = "Other"
TOP_TITLES_COUNT = 25
TOP_LOCATIONS_COUNT = 12

# --- Human-readable labels for the Streamlit UI ------------------------------
EXPERIENCE_LABELS = {
    "EN": "Junior / Entry-level",
    "MI": "Mid-level / Intermediate",
    "SE": "Senior / Expert",
    "EX": "Executive / Director",
}
EMPLOYMENT_LABELS = {
    "FT": "Full-time",
    "PT": "Part-time",
    "CT": "Contract",
    "FL": "Freelance",
}
COMPANY_SIZE_LABELS = {
    "S": "Small (< 50 employees)",
    "M": "Medium (50-250 employees)",
    "L": "Large (> 250 employees)",
}
REMOTE_LABELS = {
    0: "On-site",
    50: "Hybrid",
    100: "Fully remote",
}
COUNTRY_LABELS = {
    "US": "United States",
    "CA": "Canada",
    "GB": "United Kingdom",
    "AU": "Australia",
    "NL": "Netherlands",
    "DE": "Germany",
    "FR": "France",
    "LT": "Lithuania",
    "AT": "Austria",
    "ES": "Spain",
    "SK": "Slovakia",
    "IN": "India",
    "PL": "Poland",
    "IE": "Ireland",
    "LV": "Latvia",
    OTHER_BUCKET: "Other country",
}

# --- Quantile / inference settings -------------------------------------------
QUANTILES = {"low": 0.1, "median": 0.5, "high": 0.9}
LATEST_WORK_YEAR = 2025
USD_TO_PLN = 4.0  # rough display-only conversion, not a live exchange rate

# --- Training settings -------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.08,
    "max_depth": 8,
    "random_state": RANDOM_STATE,
}

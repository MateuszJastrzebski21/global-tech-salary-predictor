"""Data preparation pipeline for the ai-jobs.net salary dataset.

This module turns the raw CSV export into a clean, model-ready dataset. The
pipeline removes exact duplicate records, keeps only the columns the model
needs (dropping rows with missing values or a non-positive salary), and
buckets rare job titles and company locations into an ``Other`` group. Run
the module directly to produce ``data/processed/salaries_clean.csv``.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config  # pylint: disable=wrong-import-position


def load_raw_data(path=config.RAW_DATA_PATH):
    """Load the raw salary CSV into a DataFrame."""
    return pd.read_csv(path)


def remove_duplicates(df):
    """Drop exact duplicate rows and reset the index.

    The raw ai-jobs.net export repeats identical records many times. Without
    deduplication the same row could land in both the train and test split,
    leaking information across the split and inflating the evaluation scores.
    """
    return df.drop_duplicates().reset_index(drop=True)


def keep_valid_rows(df):
    """Keep only the modelling columns and drop unusable rows.

    Selects ``FEATURES + [TARGET]``, removes rows with missing values, keeps
    only strictly positive salaries, then resets the index.
    """
    columns = config.FEATURES + [config.TARGET]
    valid = df[columns].dropna()
    valid = valid[valid[config.TARGET] > 0]
    return valid.reset_index(drop=True)


def bucket_rare_categories(df):
    """Collapse rare job titles and company locations into ``OTHER_BUCKET``.

    Keeps the most frequent ``TOP_TITLES_COUNT`` job titles and
    ``TOP_LOCATIONS_COUNT`` company locations; every other value is replaced
    with ``OTHER_BUCKET`` so the one-hot encoding stays small and stable.
    """
    df = df.copy()
    top_titles = df["job_title"].value_counts().nlargest(config.TOP_TITLES_COUNT).index
    top_locations = (
        df["company_location"].value_counts().nlargest(config.TOP_LOCATIONS_COUNT).index
    )
    df["job_title"] = df["job_title"].where(
        df["job_title"].isin(top_titles), config.OTHER_BUCKET
    )
    df["company_location"] = df["company_location"].where(
        df["company_location"].isin(top_locations), config.OTHER_BUCKET
    )
    return df


def prepare_dataset(save=True):
    """Run the full cleaning pipeline and optionally persist the result."""
    df = load_raw_data()
    df = remove_duplicates(df)
    df = keep_valid_rows(df)
    df = bucket_rare_categories(df)
    if save:
        config.PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.PROCESSED_DATA_PATH, index=False)
    return df


def main():
    """Prepare the dataset and report how many rows were saved and where."""
    clean = prepare_dataset(save=True)
    print(f"Saved {len(clean)} clean rows to {config.PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()

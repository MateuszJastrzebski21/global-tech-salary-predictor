# 💰 Global Tech Salary Predictor

Estimates an **annual salary range** (low / median / high, in USD) for a tech
role from a handful of inputs such as seniority, job title, company location
and size. Built as the final project for **SUML** ("ML runtime environments")
at **PJATK**.

## What it does

Instead of a single, falsely precise number, the app reports a **calibrated
range**. Three independent quantile models predict the 10th, 50th and 90th
percentiles, so the low–high band is designed to contain about **80%** of real
salaries for similar roles. The data comes from **ai-jobs.net** (public salary
export, 2020–2025, 150k+ records), giving the estimates a real-world basis.

## How to run

Only two commands are needed — on the **first launch** the app automatically
prepares the data and trains the model (about a minute), with no other setup:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Live version

Try it here: _TBD — link added after deployment to Streamlit Community Cloud._

## Input and output

| UI field         | Example value          | Maps to feature      |
|------------------|------------------------|----------------------|
| Seniority        | Senior / Expert        | `experience_level`   |
| Role             | Software Engineer      | `job_title`          |
| Company location | United States          | `company_location`   |
| Company size     | Medium (50-250)        | `company_size`       |
| Work mode        | On-site                | `remote_ratio`       |
| Employment type  | Full-time              | `employment_type`    |

**Output:** three USD figures — lower estimate, typical (median) and upper
estimate — plus a rough PLN/year hint for the median (display-only conversion).

## Project structure

```text
global-tech-salary-predictor/
├── config.py                 # single source of truth: paths, features, params
├── requirements.txt          # runtime dependencies
├── data/                     # DATA layer
│   ├── raw/salaries.csv      # committed raw dataset (clone-and-run)
│   ├── processed/            # generated clean CSV (gitignored)
│   └── prepare_data.py       # cleaning pipeline: dedup, filter, bucket
├── model/                    # MODEL layer
│   ├── artifacts/            # generated model bundle (gitignored)
│   ├── train_model.py        # train + evaluate + save quantile models
│   └── predict.py            # load bundle + inference helpers
└── app/                      # APP layer
    └── streamlit_app.py      # Streamlit UI
```

## The ML model

- **Type:** three `HistGradientBoostingRegressor` models (scikit-learn), each
  trained with `loss="quantile"` at quantile 0.1, 0.5 and 0.9.
- **Why three models:** one per quantile produces a calibrated low / median /
  high band rather than a single point estimate; the median is the typical
  guess and the low–high pair expresses the uncertainty.
- **Input vector:** `experience_level`, `employment_type`, `job_title`,
  `company_location`, `company_size` (one-hot encoded) plus `work_year` and
  `remote_ratio` (passed through numerically).
- **Output vector:** three real numbers — low / median / high annual salary in
  USD, clamped at zero and sorted so that low ≤ median ≤ high.

## Quality metrics

| Metric                 | Value      | Reading                                   |
|------------------------|------------|-------------------------------------------|
| Interval coverage      | ~80%       | share of real salaries inside low–high    |
| R² (median model)      | ~0.25      | variance explained by the median model    |
| Mean absolute error    | ~$47k      | typical error of the median estimate      |

A modest R² is expected for salary data: pay depends on many factors the
dataset does not capture (exact company, individual skills, negotiation). That
is exactly why the **calibrated range** — not the point estimate — is the
headline result.

## Data notes

- **Deduplication:** roughly half of the raw export rows are exact duplicates.
  They are removed first, otherwise identical records could land in both the
  train and test split and leak information, inflating the scores.
- **Rare-category bucketing:** only the most common job titles (top 25) and
  company locations (top 12) are kept; everything else is grouped into
  `"Other"` so the one-hot encoding stays small and stable.

## Team & contributions

| Member               | Index   | Contribution                                                        |
|----------------------|---------|---------------------------------------------------------------------|
| Mateusz Jastrzębski  | s27397  | Implementation: data / model / app layers, training, cloud deployment, code quality |
| Mateusz Stawarz      | s21871  | Project concept and scope, dataset selection, documentation, testing |

"""Streamlit user interface for the Global Tech Salary Predictor.

Presents a small form of role / seniority / location inputs and shows a
calibrated annual salary range (low / median / high in USD, with a rough PLN
hint). On a fresh clone the model trains itself on first launch, so the app is
self-bootstrapping with no manual setup step.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config  # pylint: disable=wrong-import-position
from model import train_model  # pylint: disable=wrong-import-position
from model.predict import (  # pylint: disable=wrong-import-position
    convert_to_pln,
    load_model_bundle,
    predict_salary_range,
)

# Student index shown in the app footer (course requirement).
STUDENT_INDEX = "s27397"


@st.cache_resource(show_spinner=False)
def get_model_bundle():
    """Load the model bundle, training it on first launch if it is missing."""
    try:
        return load_model_bundle()
    except FileNotFoundError:
        with st.spinner("Training the model for the first time, please wait..."):
            train_model.main()
        return load_model_bundle()


def _default_index(values, preferred):
    """Return the position of ``preferred`` in ``values``, or 0 if absent."""
    return values.index(preferred) if preferred in values else 0


def render_inputs(options):
    """Render the input widgets and return the collected user_input dict."""
    left, right = st.columns(2)
    with left:
        experience = st.selectbox(
            "Seniority",
            options["experience_level"],
            index=_default_index(options["experience_level"], "SE"),
            format_func=lambda code: config.EXPERIENCE_LABELS.get(code, code),
        )
        job_title = st.selectbox("Role", options["job_title"])
        company_location = st.selectbox(
            "Company location",
            options["company_location"],
            index=_default_index(options["company_location"], "US"),
            format_func=lambda code: config.COUNTRY_LABELS.get(code, code),
        )
    with right:
        company_size = st.selectbox(
            "Company size",
            options["company_size"],
            format_func=lambda code: config.COMPANY_SIZE_LABELS.get(code, code),
        )
        remote_ratio = st.selectbox(
            "Work mode",
            options["remote_ratio"],
            format_func=lambda value: config.REMOTE_LABELS.get(value, value),
        )
        employment_type = st.selectbox(
            "Employment type",
            options["employment_type"],
            index=_default_index(options["employment_type"], "FT"),
            format_func=lambda code: config.EMPLOYMENT_LABELS.get(code, code),
        )
    return {
        "experience_level": experience,
        "employment_type": employment_type,
        "job_title": job_title,
        "company_location": company_location,
        "company_size": company_size,
        "remote_ratio": remote_ratio,
        "work_year": config.LATEST_WORK_YEAR,
    }


def render_results(prediction):
    """Show the predicted salary range as metrics, a PLN hint and a chart."""
    st.success("Here is the estimated annual salary range:")
    low_col, median_col, high_col = st.columns(3)
    low_col.metric("Lower estimate", f"${prediction['low']:,.0f}")
    median_col.metric("Typical (median)", f"${prediction['median']:,.0f}")
    high_col.metric("Upper estimate", f"${prediction['high']:,.0f}")
    median_pln = convert_to_pln(prediction["median"])
    st.caption(
        f"Roughly {median_pln:,.0f} PLN/year at the typical estimate "
        f"(display-only conversion at {config.USD_TO_PLN:.1f} PLN/USD)."
    )
    chart = pd.DataFrame(
        {"USD": [prediction["low"], prediction["median"], prediction["high"]]},
        index=["Lower", "Median", "Upper"],
    )
    st.bar_chart(chart)


def render_model_quality(bundle):
    """Show an expander with the model's training stats and honest caveats."""
    metrics = bundle["metrics"]
    with st.expander("How good is this model?"):
        st.write(
            f"Trained on **{bundle['sample_count']:,} real records** "
            f"(last trained on {bundle['trained_on']})."
        )
        st.write(
            f"**Interval coverage: {metrics['interval_coverage']:.0%}** — that "
            "share of real salaries fell inside the predicted low-high band, so "
            "the range is well calibrated."
        )
        st.write(
            f"R-squared = {metrics['r2']:.2f}, "
            f"mean absolute error about ${metrics['mae']:,.0f}."
        )
        st.write(
            "Salaries depend on factors this data does not capture - the exact "
            "company, individual skills, interview performance and negotiation. "
            "That is why we report a calibrated range instead of a single, "
            "falsely precise number."
        )


def main():
    """Run the Streamlit app: collect inputs and show a salary estimate."""
    st.set_page_config(
        page_title="Global Tech Salary Predictor", page_icon="💰"
    )
    st.title("💰 Global Tech Salary Predictor")
    st.write(
        "Estimate an annual salary range for a tech role, learned from 150k+ "
        "real job records (ai-jobs.net, 2020-2025)."
    )
    st.info(
        "Pick the role and company details below, then press "
        "**Estimate salary** to see a low / median / high range."
    )
    try:
        bundle = get_model_bundle()
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Could not load or train the model: {error}")
        return
    user_input = render_inputs(bundle["feature_options"])
    if st.button("Estimate salary", type="primary"):
        with st.spinner("Crunching the numbers..."):
            try:
                prediction = predict_salary_range(bundle, user_input)
            except (ValueError, KeyError) as error:
                st.error(f"Prediction failed: {error}")
                return
        render_results(prediction)
    render_model_quality(bundle)
    st.divider()
    st.caption(f"SUML project | Student index: {STUDENT_INDEX}")


if __name__ == "__main__":
    main()

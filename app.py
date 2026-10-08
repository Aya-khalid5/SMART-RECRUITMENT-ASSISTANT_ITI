"""Smart Recruitment Assistant - Streamlit interface around the original ML project.

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import logging

import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import recruitment_pipeline as rp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart_recruitment")

st.set_page_config(
    page_title="Smart Recruitment Assistant",
    page_icon="🧭",
    layout="wide",
)

NOT_SPECIFIED = "Not specified"

# Friendly display labels -> the raw dataset values the model was trained on.
LAST_NEW_JOB_LABELS = {
    "Never changed jobs": "never",
    "1 year ago": "1",
    "2 years ago": "2",
    "3 years ago": "3",
    "4 years ago": "4",
    "More than 4 years ago": ">4",
}
RELEVANT_EXP_LABELS = {
    "Has relevant experience": "Has relevent experience",
    "No relevant experience": "No relevent experience",
}


# --------------------------------------------------------------------------- #
# Cached loaders (not re-run on every interaction)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Preparing preprocessing pipeline...")
def get_reference():
    return rp.load_reference_data()


@st.cache_resource(show_spinner="Loading trained model...")
def get_model(name: str):
    return rp.load_model(name)


@st.cache_data(show_spinner=False)
def get_validation_metrics(name: str) -> dict:
    ref = get_reference()
    model = get_model(name)
    X = ref["preprocessor"].transform(ref["X_val"])
    pred = model.predict(X)
    y = ref["y_val"]
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred),
        "Recall": recall_score(y, pred),
        "F1 score": f1_score(y, pred),
        "n": int(len(y)),
    }


def opt(values: list[str]) -> list[str]:
    return [NOT_SPECIFIED] + values


def none_if_unspecified(v):
    return None if v in (None, NOT_SPECIFIED) else v


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🧭 Smart Recruitment Assistant")
st.caption("Predictive analytics for employee job-change risk")

try:
    ref = get_reference()
except Exception:  # noqa: BLE001 - never show raw tracebacks to the user
    logger.exception("Failed to load reference data")
    st.error(
        "The application could not load its reference dataset. "
        "Please check that `data/aug_train.csv` exists in the repository."
    )
    st.stop()

with st.sidebar:
    st.header("Model")
    model_name = st.selectbox(
        "Prediction model",
        list(rp.MODEL_FILES.keys()),
        index=list(rp.MODEL_FILES.keys()).index(rp.DEFAULT_MODEL),
        help="All models are the original pre-trained models from the project. "
        "Gradient Boosting was selected as the final model in the notebook.",
    )
    st.markdown("---")
    st.caption(
        "Estimates how likely a candidate is to be **looking for a new job**, "
        "based on training history, education, experience and company context."
    )

# --------------------------------------------------------------------------- #
# 1. Input / Employee information
# --------------------------------------------------------------------------- #
st.header("1. Candidate information")

cities = sorted(ref["city_cdi"].keys(), key=lambda c: int(c.split("_")[-1]) if c.split("_")[-1].isdigit() else 0)
o = ref["options"]

with st.form("candidate_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Profile")
        city = st.selectbox("City", cities, help="City code from the dataset.")
        cdi = st.number_input(
            "City development index",
            min_value=0.0, max_value=1.0, step=0.001, format="%.3f",
            value=float(ref["city_cdi"][city]),
            key=f"cdi_{city}",
            help="Defaults to the index recorded for the selected city; you can adjust it.",
        )
        gender = st.selectbox("Gender", opt(o["gender"]))
        education = st.selectbox("Education level", opt(o["education_level"]))
        major = st.selectbox("Major discipline", opt(o["major_discipline"]))
        enrolled = st.selectbox("University enrolment", opt(o["enrolled_university"]))

    with c2:
        st.subheader("Experience")
        relevant_label = st.radio("Relevant experience", list(RELEVANT_EXP_LABELS), horizontal=False)
        exp_unknown = st.checkbox("Years of experience unknown")
        experience = st.number_input(
            "Years of experience",
            min_value=0, max_value=60, value=5, step=1,
            disabled=exp_unknown,
            help="Values below 1 are treated as 1 and above 20 as 20, as in the original data.",
        )
        last_job_label = st.selectbox("Last job change", list(LAST_NEW_JOB_LABELS), index=1)
        training_hours = st.number_input(
            "Training hours completed", min_value=0, max_value=1000, value=40, step=1
        )

    with c3:
        st.subheader("Current company")
        company_size = st.selectbox("Company size", opt(o["company_size"]))
        company_type = st.selectbox("Company type", opt(o["company_type"]))
        st.caption(
            "Fields marked *Not specified* are handled with the same rules used when the "
            "model was trained (e.g. 'Unknown' category, most frequent value, or average)."
        )

    submitted = st.form_submit_button("Predict job-change risk", type="primary", use_container_width=True)

# --------------------------------------------------------------------------- #
# 2-4. Prediction, probability, interpretation
# --------------------------------------------------------------------------- #
if submitted:
    try:
        exp_value = None if exp_unknown else int(min(max(int(experience), 1), 20))
        candidate = {
            "city": city,
            "city_development_index": cdi,
            "gender": none_if_unspecified(gender),
            "relevent_experience": RELEVANT_EXP_LABELS[relevant_label],
            "enrolled_university": none_if_unspecified(enrolled),
            "education_level": none_if_unspecified(education),
            "major_discipline": none_if_unspecified(major),
            "experience": exp_value,
            "company_size": none_if_unspecified(company_size),
            "company_type": none_if_unspecified(company_type),
            "last_new_job": LAST_NEW_JOB_LABELS[last_job_label],
            "training_hours": training_hours,
        }
        model = get_model(model_name)
        frame = rp.candidate_to_frame(candidate, ref["fills"])
        label, proba = rp.predict(model, ref["preprocessor"], frame)
        st.session_state["result"] = {"label": label, "proba": proba, "model": model_name, "frame": frame}
    except FileNotFoundError:
        logger.exception("Model file missing")
        st.session_state.pop("result", None)
        st.error("The selected model file could not be found in the `models/` folder.")
    except ModuleNotFoundError as e:
        logger.exception("Missing dependency")
        st.session_state.pop("result", None)
        st.error(f"A required package is not installed ({e.name}). Please check requirements.txt.")
    except Exception:  # noqa: BLE001
        logger.exception("Prediction failed")
        st.session_state.pop("result", None)
        st.error(
            "Sorry, a prediction could not be produced for these inputs. "
            "Please review the values and try again, or choose another model."
        )

result = st.session_state.get("result")
if result:
    st.header("2. Prediction")
    if result["label"] == 1:
        st.warning("**Likely to look for a new job**")
    else:
        st.success("**Likely to stay in the current role**")
    st.caption(f"Model used: {result['model']}")

    st.header("3. Prediction probability")
    p = result["proba"]
    m1, m2 = st.columns(2)
    m1.metric("Probability of looking for a job change", f"{p:.1%}")
    m2.metric("Probability of staying", f"{1 - p:.1%}")
    st.progress(min(max(p, 0.0), 1.0))

    st.header("4. Interpretation")
    if p >= 0.5:
        band = "high"
    elif p >= 0.3:
        band = "moderate"
    else:
        band = "low"
    st.markdown(
        f"The model estimates a **{band}** likelihood ({p:.1%}) that this candidate is "
        "looking for a new job. "
        + {
            "high": "From a recruiting perspective this candidate may be receptive to new opportunities.",
            "moderate": "The signal is mixed; the candidate may be open to the right opportunity.",
            "low": "The candidate looks relatively settled in their current situation.",
        }[band]
    )
    st.info(
        "This is a statistical estimate from historical data, not a certainty. "
        "The low/moderate/high bands are an indicative reading aid; the classification above "
        "uses the model's own decision rule. It should support, not replace, human judgement, "
        "and should not be used as the sole basis for decisions about individuals."
    )
    with st.expander("Show the processed input used for the prediction"):
        st.dataframe(result["frame"][rp.NUMERICAL_FEATURES + rp.CATEGORICAL_FEATURES], use_container_width=True)

# --------------------------------------------------------------------------- #
# 5. About the model
# --------------------------------------------------------------------------- #
st.header("5. About the model")
a1, a2 = st.columns([2, 3])
with a1:
    st.markdown(
        f"""
**Data:** HR analytics dataset of {ref['n_rows']:,} candidates; about {ref['target_rate']:.0%} are
looking for a job change (`target = 1`).

**Models:** Logistic Regression, Random Forest, XGBoost and Gradient Boosting, trained in the
original notebook. Gradient Boosting was chosen as the final model.

**Preprocessing:** numeric features standardised; categorical features one-hot encoded; missing
values handled with an 'Unknown' category, the most frequent value or the average, exactly as in
the notebook.
"""
    )
with a2:
    try:
        metrics = get_validation_metrics(model_name)
        st.markdown(f"**Hold-out validation performance - {model_name}** ({metrics['n']:,} candidates)")
        cols = st.columns(4)
        for col, key in zip(cols, ["Accuracy", "Precision", "Recall", "F1 score"]):
            col.metric(key, f"{metrics[key]:.1%}")
    except Exception:  # noqa: BLE001
        logger.exception("Could not compute validation metrics")
        st.caption("Validation metrics are unavailable for the selected model.")

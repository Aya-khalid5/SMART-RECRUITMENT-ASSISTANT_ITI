"""
Preprocessing + model loading for the Smart Recruitment Assistant.

This module reproduces, step by step, the cleaning / feature-engineering /
train-test split / ColumnTransformer logic from notebooks/smart_jop.ipynb.

WHY IT IS NEEDED
The original notebook saved only the four trained models (models/*.pkl). The
fitted ColumnTransformer (StandardScaler + OneHotEncoder) was NOT saved, but the
models were trained on its output. The transformer is deterministic: it is fit on
X_train from train_test_split(random_state=42, stratify=y) of the cleaned
dataset. Re-fitting it here on the same data and the same split gives exactly
the same scaler means and one-hot categories the models were trained with.
No model is retrained.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "aug_train.csv"
MODELS_DIR = BASE_DIR / "models"

MODEL_FILES = {
    "Gradient Boosting (final model)": "gradient_boosting_model.pkl",
    "XGBoost": "xgboost_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl",
}
DEFAULT_MODEL = "Gradient Boosting (final model)"

# Column groups used by the original ColumnTransformer (order matters).
NUMERICAL_FEATURES = [
    "city_development_index",
    "experience",
    "last_new_job",
    "training_hours",
    "relevent_experience",
]
CATEGORICAL_FEATURES = [
    "city",
    "gender",
    "enrolled_university",
    "education_level",
    "major_discipline",
    "company_size",
    "company_type",
]

# Same lists as the notebook's cleaning step.
CATEGORICAL_UNKNOWN = ["company_type", "company_size", "gender", "major_discipline"]
CATEGORICAL_MODE = ["education_level", "last_new_job", "enrolled_university"]

LAST_NEW_JOB_MAP = {"never": 0, "1": 1, "2": 2, "3": 3, "4": 4, ">4": 5}
RELEVANT_EXP_MAP = {"Has relevent experience": 1, "No relevent experience": 0}


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Notebook cells 13-24 + 36. Returns the cleaned frame and the fill values."""
    df = df.copy()

    for col in CATEGORICAL_UNKNOWN:
        df[col] = df[col].fillna("Unknown")

    modes = {}
    for col in CATEGORICAL_MODE:
        modes[col] = df[col].mode()[0]
        df[col] = df[col].fillna(modes[col])

    df["experience"] = df["experience"].replace({"<1": 1, ">20": 20})
    df["experience"] = pd.to_numeric(df["experience"], errors="coerce")
    experience_mean = df["experience"].mean()
    df["experience"] = df["experience"].fillna(experience_mean)

    df["relevent_experience"] = df["relevent_experience"].replace(RELEVANT_EXP_MAP)
    df["last_new_job"] = df["last_new_job"].replace(LAST_NEW_JOB_MAP)

    df["relevent_experience"] = df["relevent_experience"].astype(int)
    df["last_new_job"] = df["last_new_job"].astype(int)
    df["target"] = df["target"].astype(int)
    df["experience"] = df["experience"].astype(int)  # truncates, as in the notebook

    df = df.drop("enrollee_id", axis=1)
    fills = {"modes": modes, "experience_mean": float(experience_mean)}
    return df, fills


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Notebook cells 39-42. (These columns are dropped by the ColumnTransformer
    but are kept so X has the same columns as in the notebook.)"""
    df = df.copy()
    df["experience_level"] = pd.cut(
        df["experience"], bins=[0, 2, 5, 10, 20],
        labels=["Entry", "Junior", "Mid", "Senior"], include_lowest=True,
    )
    df["development_level"] = pd.cut(
        df["city_development_index"], bins=[0, 0.6, 0.8, 0.9, 1],
        labels=["Low", "Medium", "High", "Very High"], include_lowest=True,
    )
    for col in ["gender", "major_discipline", "company_size", "company_type"]:
        df[col + "_missing"] = (df[col] == "Unknown").astype(int)
    df["high_training"] = (df["training_hours"] >= 100).astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Notebook cell 47 (identical definition)."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def load_reference_data(data_path: Path = DATA_PATH) -> dict:
    """Clean the dataset, split it exactly as the notebook did and fit the
    preprocessor on X_train. Returns everything the app needs."""
    raw = pd.read_csv(data_path)
    df, fills = clean_dataset(raw)
    df = add_engineered_features(df)

    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    city_cdi = df.groupby("city")["city_development_index"].first().to_dict()
    options = {
        col: sorted(raw[col].dropna().unique().tolist())
        for col in [
            "gender", "enrolled_university", "education_level", "major_discipline",
            "company_size", "company_type",
        ]
    }
    return {
        "preprocessor": preprocessor,
        "fills": fills,
        "X_val": X_val,
        "y_val": y_val,
        "city_cdi": city_cdi,
        "options": options,
        "n_rows": int(len(raw)),
        "target_rate": float(y.mean()),
    }


def load_model(name: str):
    """Load one of the original pickled models (no retraining)."""
    path = MODELS_DIR / MODEL_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path.name}")
    with open(path, "rb") as f:
        return pickle.load(f)


def candidate_to_frame(candidate: dict, fills: dict) -> pd.DataFrame:
    """Turn the form inputs into a one-row frame with the notebook's cleaning
    rules applied (missing -> 'Unknown' / mode / mean, ordinal maps, int casts).

    Expected keys: city, city_development_index, gender, relevent_experience
    (the raw dataset label), enrolled_university, education_level,
    major_discipline, experience (number or None), company_size, company_type,
    last_new_job (raw dataset label), training_hours.
    """
    row = dict(candidate)

    for col in CATEGORICAL_UNKNOWN:
        if row.get(col) in (None, ""):
            row[col] = "Unknown"
    for col in CATEGORICAL_MODE:
        if row.get(col) in (None, ""):
            row[col] = fills["modes"][col]

    exp = row.get("experience")
    if exp is None or (isinstance(exp, float) and np.isnan(exp)):
        exp = fills["experience_mean"]
    row["experience"] = int(exp)

    row["relevent_experience"] = int(RELEVANT_EXP_MAP[row["relevent_experience"]])
    row["last_new_job"] = int(LAST_NEW_JOB_MAP[str(row["last_new_job"])])
    row["training_hours"] = int(row["training_hours"])
    row["city_development_index"] = float(row["city_development_index"])

    frame = pd.DataFrame([row])
    return add_engineered_features(frame)


def predict(model, preprocessor, frame: pd.DataFrame) -> tuple[int, float]:
    """Return (predicted class, probability of looking for a job change)."""
    processed = preprocessor.transform(frame)
    proba = float(model.predict_proba(processed)[0, 1])
    label = int(model.predict(processed)[0])
    return label, proba

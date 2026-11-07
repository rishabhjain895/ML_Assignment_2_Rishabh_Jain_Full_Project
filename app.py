from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

ROOT = Path(__file__).resolve().parent

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

FEATURES = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
]

st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🧬", layout="wide")
st.title("Wisconsin Breast Cancer Classification")
st.write("Explore five classification models using an uploaded test CSV.")
st.info("Target encoding: 1 = Malignant (positive class); 0 = Benign.")


def patch_estimator(estimator):
    """Dynamically backports missing attributes from sklearn 1.8.0 to 1.5.2."""
    cls_name = getattr(estimator, "__class__", None).__name__ if hasattr(estimator, "__class__") else ""
    if cls_name == "LogisticRegression":
        if not hasattr(estimator, "multi_class"):
            estimator.multi_class = "auto"
    return estimator


def get_model(model_name: str):
    """Loads model and applies version-compatibility patching."""
    model_path = ROOT / "models" / MODEL_FILES[model_name]
    if not model_path.exists():
        st.error(f"Model file not found: {model_path.name}")
        st.stop()

    model = joblib.load(model_path)

    # Patch final classifier if wrapped inside a Pipeline
    if hasattr(model, "steps") and len(model.steps) > 0:
        patch_estimator(model.steps[-1][1])
    else:
        patch_estimator(model)

    return model


uploaded = st.file_uploader("Upload test dataset (CSV)", type=["csv"])
choice = st.selectbox("Select classification model", list(MODEL_FILES.keys()))

if uploaded is not None:
    df = pd.read_csv(uploaded)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        st.error("Missing columns: " + ", ".join(missing))
        st.stop()

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        st.error("Features contain missing or non-numeric values.")
        st.stop()

    model = get_model(choice)

    try:
        pred = model.predict(X)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # Robust probability computation across all model families
    prob = None
    if hasattr(model, "predict_proba"):
        try:
            prob = model.predict_proba(X)[:, 1]
        except AttributeError:
            pass

    if prob is None:
        if hasattr(model, "decision_function"):
            try:
                df_vals = model.decision_function(X)
                # Apply min-max normalization to project decision values to [0, 1]
                prob = (df_vals - df_vals.min()) / (df_vals.max() - df_vals.min() + 1e-9)
            except Exception:
                prob = pred.astype(float)
        else:
            prob = pred.astype(float)

    out = X.copy()
    out["predicted_target"] = pred
    out["predicted_class"] = pd.Series(pred).map({0: "Benign", 1: "Malignant"}).values
    out["malignant_probability"] = prob

    st.subheader("Predictions")
    st.dataframe(out, use_container_width=True)

    if "target" not in df.columns:
        st.warning("No target column: predictions are shown, but evaluation requires true labels.")
    else:
        actual = pd.to_numeric(df["target"], errors="coerce")
        if actual.isna().any() or not set(actual.unique()).issubset({0, 1}):
            st.error("target must contain only 0 and 1.")
            st.stop()
        if actual.nunique() != 2:
            st.error("AUC requires both target classes.")
            st.stop()

        scores = {
            "Accuracy": accuracy_score(actual, pred),
            "AUC": roc_auc_score(actual, prob),
            "Precision": precision_score(actual, pred, zero_division=0),
            "Recall": recall_score(actual, pred, zero_division=0),
            "F1": f1_score(actual, pred, zero_division=0),
            "MCC": matthews_corrcoef(actual, pred),
        }

        st.subheader("Evaluation metrics")
        cols = st.columns(6)
        for col, (label, value) in zip(cols, scores.items()):
            col.metric(label, f"{value:.4f}")

        st.subheader("Confusion matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(
            confusion_matrix(actual, pred, labels=[0, 1]),
            display_labels=["Benign", "Malignant"],
        ).plot(ax=ax, colorbar=False)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Classification report")
        report = classification_report(
            actual,
            pred,
            labels=[0, 1],
            target_names=["Benign", "Malignant"],
            output_dict=True,
            zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
else:
    st.caption("Upload the included test_data.csv to evaluate a model.")

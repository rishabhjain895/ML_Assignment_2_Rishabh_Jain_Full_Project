from pathlib import Path
import numpy as np
import joblib, pandas as pd, streamlit as st, matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,roc_auc_score,precision_score,recall_score,f1_score,matthews_corrcoef,confusion_matrix,classification_report,ConfusionMatrixDisplay
ROOT=Path(__file__).resolve().parent
MODEL_FILES={"Logistic Regression":"logistic_regression.joblib","Decision Tree":"decision_tree.joblib","K-Nearest Neighbors":"k_nearest_neighbors.joblib","Gaussian Naive Bayes":"gaussian_naive_bayes.joblib","Support Vector Machine":"support_vector_machine.joblib"}
FEATURES=[np.str_('mean radius'), np.str_('mean texture'), np.str_('mean perimeter'), np.str_('mean area'), np.str_('mean smoothness'), np.str_('mean compactness'), np.str_('mean concavity'), np.str_('mean concave points'), np.str_('mean symmetry'), np.str_('mean fractal dimension'), np.str_('radius error'), np.str_('texture error'), np.str_('perimeter error'), np.str_('area error'), np.str_('smoothness error'), np.str_('compactness error'), np.str_('concavity error'), np.str_('concave points error'), np.str_('symmetry error'), np.str_('fractal dimension error'), np.str_('worst radius'), np.str_('worst texture'), np.str_('worst perimeter'), np.str_('worst area'), np.str_('worst smoothness'), np.str_('worst compactness'), np.str_('worst concavity'), np.str_('worst concave points'), np.str_('worst symmetry'), np.str_('worst fractal dimension')]
st.set_page_config(page_title="Breast Cancer Classifier",page_icon="🧬",layout="wide")
st.title("Wisconsin Breast Cancer Classification")
st.write("Explore five classification models using an uploaded test CSV.")
st.info("Target encoding: 1 = Malignant (positive class); 0 = Benign.")
uploaded=st.file_uploader("Upload test dataset (CSV)",type=["csv"])
choice=st.selectbox("Select classification model",list(MODEL_FILES))
if uploaded is not None:
    df=pd.read_csv(uploaded)
    missing=[c for c in FEATURES if c not in df.columns]
    if missing: st.error("Missing columns: "+", ".join(missing)); st.stop()
    X=df[FEATURES].apply(pd.to_numeric,errors="coerce")
    if X.isna().any().any(): st.error("Features contain missing or non-numeric values."); st.stop()
    model=joblib.load(ROOT/"models"/MODEL_FILES[choice])
    pred=model.predict(X); prob=model.predict_proba(X)[:,1]
    out=X.copy(); out["predicted_target"]=pred
    out["predicted_class"]=pd.Series(pred).map({0:"Benign",1:"Malignant"}).values
    out["malignant_probability"]=prob
    st.subheader("Predictions"); st.dataframe(out,use_container_width=True)
    if "target" not in df.columns: st.warning("No target column: predictions are shown, but evaluation requires true labels.")
    else:
        actual=pd.to_numeric(df["target"],errors="coerce")
        if actual.isna().any() or not set(actual.unique()).issubset({0,1}): st.error("target must contain only 0 and 1."); st.stop()
        if actual.nunique()!=2: st.error("AUC requires both target classes."); st.stop()
        scores={"Accuracy":accuracy_score(actual,pred),"AUC":roc_auc_score(actual,prob),"Precision":precision_score(actual,pred,zero_division=0),"Recall":recall_score(actual,pred,zero_division=0),"F1":f1_score(actual,pred,zero_division=0),"MCC":matthews_corrcoef(actual,pred)}
        st.subheader("Evaluation metrics"); cols=st.columns(6)
        for col,(label,value) in zip(cols,scores.items()): col.metric(label,f"{value:.4f}")
        st.subheader("Confusion matrix")
        fig,ax=plt.subplots(); ConfusionMatrixDisplay(confusion_matrix(actual,pred,labels=[0,1]),display_labels=["Benign","Malignant"]).plot(ax=ax,colorbar=False)
        st.pyplot(fig); plt.close(fig)
        st.subheader("Classification report")
        report=classification_report(actual,pred,labels=[0,1],target_names=["Benign","Malignant"],output_dict=True,zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose(),use_container_width=True)
else: st.caption("Upload the included test_data.csv to evaluate a model.")

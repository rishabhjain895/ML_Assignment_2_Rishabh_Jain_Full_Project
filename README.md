# Machine Learning Assignment 2 — Rishabh Jain
**BITS ID:** 2025NS04065  
**Programme:** M.Tech (DSE), Machine Learning

## A. Problem statement
Compare five supervised classification algorithms for predicting malignant versus benign breast tumours. Evaluate each model using six metrics and build an interactive Streamlit app.

## B. Dataset description
- Dataset: Wisconsin Breast Cancer Diagnostic.
- Source: UCI Machine Learning Repository, accessed through `sklearn.datasets.load_breast_cancer`.
- 569 instances and 30 numerical features; binary classification.
- Target: `1 = Malignant` (positive class), `0 = Benign`.
- Split: stratified 80:20 train/test, `random_state=42`.
- StandardScaler is fitted inside pipelines for Logistic Regression, KNN and SVM; Decision Tree and Gaussian Naive Bayes do not require scaling here.

## C. GitHub repository link
**Add your actual public GitHub URL here after uploading the project.**

## D. Models and evaluation
AUC uses predicted probabilities for malignant class. All models use the same held-out test data.

| Model                  |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression    |     0.9649 | 0.9960 |      0.9750 |   0.9286 | 0.9512 | 0.9245 |
| Decision Tree          |     0.9123 | 0.8770 |      0.9444 |   0.8095 | 0.8718 | 0.8113 |
| K-Nearest Neighbors    |     0.9561 | 0.9823 |      0.9744 |   0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes   |     0.9386 | 0.9934 |      1.0000 |   0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine |     0.9737 | 0.9947 |      1.0000 |   0.9286 | 0.9630 | 0.9442 |

### Observations
| Model | Observation |
|---|---|
| Logistic Regression | Acc=0.9649, AUC=0.9960, Precision=0.9750, Recall=0.9286, F1=0.9512, MCC=0.9245. |\n| Decision Tree | Acc=0.9123, AUC=0.8770, Precision=0.9444, Recall=0.8095, F1=0.8718, MCC=0.8113. |\n| K-Nearest Neighbors | Acc=0.9561, AUC=0.9823, Precision=0.9744, Recall=0.9048, F1=0.9383, MCC=0.9058. |\n| Gaussian Naive Bayes | Acc=0.9386, AUC=0.9934, Precision=1.0000, Recall=0.8333, F1=0.9091, MCC=0.8715. |\n| Support Vector Machine | Acc=0.9737, AUC=0.9947, Precision=1.0000, Recall=0.9286, F1=0.9630, MCC=0.9442. |
| Overall winner | Support Vector Machine has the highest F1 (0.9630) on this split. Logistic Regression has the highest AUC. In screening, malignant recall matters because false negatives are missed malignant cases. |

## E. Streamlit application
CSV upload, five-model dropdown, predictions and malignant probabilities, six evaluation metrics when `target` is provided, confusion matrix and classification report.

## F. Run
```bash
python -m pip install -r requirements.txt
python train.py
streamlit run app.py
```
Upload `test_data.csv`.

## G. Deployment
Push all files including `models/` to GitHub. In https://share.streamlit.io/, sign in with GitHub, create an app, choose repository, branch `main`, file `app.py`, and deploy. Add actual URLs below.
- GitHub: TO BE ADDED
- Streamlit: TO BE ADDED

## H. Repository contents
`app.py`, `train.py`, `requirements.txt`, `README.md`, `test_data.csv`, `metrics.csv`, `metrics.json`, `DATASET_SOURCE.txt`, and `models/` containing all five `.joblib` model files.

## I. Submission checklist
- [ ] Run on BITS Virtual Lab and capture your own screenshot.
- [ ] Push repository to GitHub.
- [ ] Deploy and verify Streamlit app.
- [ ] Replace both URL placeholders with real URLs.
- [ ] Include this complete README content in the single-PDF submission.

Academic integrity: the assignment permits AI for learning support, not direct copy-paste submissions. Understand, run, customize and verify the project. Your BITS Lab screenshot, GitHub history and deployment must be genuine.

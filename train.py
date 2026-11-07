from pathlib import Path
import json, joblib, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

ROOT=Path(__file__).resolve().parent
(ROOT/"models").mkdir(exist_ok=True)
data=load_breast_cancer()
X=pd.DataFrame(data.data,columns=data.feature_names)
y=(data.target==0).astype(int)  # 1=Malignant, 0=Benign
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)
test=X_test.copy(); test["target"]=y_test; test.to_csv(ROOT/"test_data.csv",index=False)
names=["Logistic Regression","Decision Tree","K-Nearest Neighbors","Gaussian Naive Bayes","Support Vector Machine"]
files=["logistic_regression.joblib","decision_tree.joblib","k_nearest_neighbors.joblib","gaussian_naive_bayes.joblib","support_vector_machine.joblib"]
models=[
Pipeline([("scaler",StandardScaler()),("classifier",LogisticRegression(max_iter=5000,random_state=42))]),
DecisionTreeClassifier(random_state=42,max_depth=4),
Pipeline([("scaler",StandardScaler()),("classifier",KNeighborsClassifier(n_neighbors=5))]),
GaussianNB(),
Pipeline([("scaler",StandardScaler()),("classifier",SVC(kernel="rbf",probability=True,random_state=42))])]
results=[]
for name,filename,model in zip(names,files,models):
    model.fit(X_train,y_train); pred=model.predict(X_test); prob=model.predict_proba(X_test)[:,1]
    results.append({"Model":name,"Accuracy":float(accuracy_score(y_test,pred)),"AUC":float(roc_auc_score(y_test,prob)),"Precision":float(precision_score(y_test,pred,zero_division=0)),"Recall":float(recall_score(y_test,pred,zero_division=0)),"F1":float(f1_score(y_test,pred,zero_division=0)),"MCC":float(matthews_corrcoef(y_test,pred))})
    joblib.dump(model,ROOT/"models"/filename)
pd.DataFrame(results).to_csv(ROOT/"metrics.csv",index=False)
(ROOT/"metrics.json").write_text(json.dumps(results,indent=2))
print(pd.DataFrame(results).to_string(index=False))

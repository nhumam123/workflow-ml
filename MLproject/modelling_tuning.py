# modelling_tuning.py

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# =========================
# 1. SETUP MLFLOW
# =========================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("XGBoost_Tuning_Experiment_Naufal_Humam")
# mlflow.autolog()


# =========================
# 2. LOAD DATASET
# =========================
df = pd.read_csv("creditcardfraud/creditcard.csv")

X = df.drop(columns=["Class"])
y = df["Class"]


# =========================
# 3. SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# 4. SMOTE HANYA UNTUK TRAIN DATA
# =========================
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)


# =========================
# 5. LOOP TUNING
# =========================
for n_estimators in [200, 300, 400, 500]:

    with mlflow.start_run(run_name=f"xgb_{n_estimators}"):

        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train_smote, y_train_smote)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_param("model", "XGBoost")
        mlflow.log_param("balancing", "SMOTE")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.1)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, "model")

        print(f"Run xgb_{n_estimators}")
        print(f"Accuracy : {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall   : {recall}")
        print(f"F1 Score : {f1}")
        print("-" * 40)
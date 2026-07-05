import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd



# --- 1. INISIALISASI MLFLOW ---
import mlflow
import mlflow.xgboost
import dagshub

# Atur tracking URI ke localhost
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_tracking_uri("file:./mlruns")
# Beri nama eksperimen sesuai format Dicoding
mlflow.set_experiment("Eksperimen_SML_Nama_Anda")

dagshub.init(repo_owner='nhumam123', repo_name='workflow-ml', mlflow=True)

df = pd.read_csv("creditcardfraud/creditcard.csv")
# 1. Pisahkan Fitur dan Target
X = df.drop(columns=['Class'])
y = df['Class']

# 2. Split Data (80% Train, 20% Test) - WAJIB SEBELUM SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Oversampling dengan SMOTE (Hanya pada Data Train)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# --- 2. MULAI RUN MLFLOW ---
with mlflow.start_run(run_name="XGBoost_SMOTE_Run"):
    
    # 4. Inisialisasi dan Training Model XGBoost
    model_xgb = XGBClassifier(random_state=42, eval_metric='logloss')
    model_xgb.fit(X_train_res, y_train_res)
    
    # [LOG PARAMETER] Catat parameter penting ke MLflow
    mlflow.log_param("model_type", "XGBClassifier")
    mlflow.log_param("sampling_strategy", "SMOTE")
    mlflow.log_param("random_state", 42)
    
    # 5. Prediksi Menggunakan Data Test yang ASLI
    y_pred = model_xgb.predict(X_test)
    
    # 6. Tampilkan Hasil Evaluasi & Ambil Metrik untuk dicatat
    report = classification_report(y_test, y_pred, output_dict=True)
    print("--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred))
    
    # [LOG METRICS] Catat metrik evaluasi utama ke MLflow
    mlflow.log_metric("accuracy", report["accuracy"])
    mlflow.log_metric("macro_f1", report["macro avg"]["f1-score"])
    mlflow.log_metric("fraud_recall", report["1"]["recall"])  # Recall untuk kelas Fraud (1)
    
    # 7. Plot Confusion Matrix
    print("--- CONFUSION MATRIX ---")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Fraud'])
    
    # Simpan plot ke file gambar terlebih dahulu agar bisa diunggah sebagai artefak
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    
    # [LOG ARTIFACTS] Unggah gambar Confusion Matrix ke MLflow
    mlflow.log_artifact(cm_path)
    
    # [LOG MODEL] Simpan struktur model XGBoost Anda ke MLflow Artifacts
    mlflow.xgboost.log_model(model_xgb, artifact_path="model")

print("Eksperimen berhasil dicatat ke MLflow!")
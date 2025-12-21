import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn

# --- 1. SETUP PATH & FOLDER (Supaya tidak error path) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
dataset_path = 'water_potability_clean.csv'

# --- 2. LOAD DATA ---
print(f"Loading data from: {os.path.abspath(dataset_path)}")
try:
    df = pd.read_csv(dataset_path)
except FileNotFoundError:
    print("❌ Error: File CSV tidak ditemukan.")
    exit()

X = df.drop('Potability', axis=1)
y = df['Potability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. SETUP MLFLOW ---
tracking_uri = "file:///" + os.path.join(script_dir, "mlruns").replace("\\", "/")
mlflow.set_tracking_uri(tracking_uri)

# Nama Eksperimen Spesifik
experiment_name = "Eksperimen_Advance_Tuning_Yustianas_Rombon"
mlflow.set_experiment(experiment_name)

print(f"Experiment: {experiment_name}")
print("--- Memulai Proses Tuning ---")

# --- 4. TUNING PROCESS (PARENT RUN) ---
with mlflow.start_run(run_name="Proses_Tuning_Utama") as parent_run:
    
    # -- A. Proses Tuning --
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestClassifier(random_state=42)
    
    # Kita buat 5 iterasi saja biar cepat tapi dashboard tetap ramai
    random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=5, cv=3, random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)
    
    # -- B. Log Child Runs (Agar Dashboard Ramai) --
    results = random_search.cv_results_
    for i in range(len(results['params'])):
        with mlflow.start_run(run_name=f"Trial_{i+1}", nested=True):
            mlflow.log_params(results['params'][i])
            mlflow.log_metric("val_accuracy", results['mean_test_score'][i])

    # -- C. Log Best Model & Artifacts (Bagian Penting!) --
    print("Logging Model Terbaik dan Artefak...")
    best_model = random_search.best_estimator_
    
    # 1. Log Parameter & Metrik
    mlflow.log_params(random_search.best_params_)
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric("test_f1", f1_score(y_test, y_pred))

    # 2. Log Model ke Folder bernama 'model' (Sesuai Request)
    # Ini yang akan membuat folder bernama "model" di UI
    mlflow.sklearn.log_model(best_model, "model")
    
    # 3. Log Gambar Confusion Matrix (Sesuai Nama Request)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Acc: {acc:.2f})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Nama file disamakan dengan contoh
    img_name = "training_confusion_matrix.png"
    plt.savefig(img_name)
    mlflow.log_artifact(img_name)
    plt.close()
    
    print(f"[SUKSES] Selesai!")
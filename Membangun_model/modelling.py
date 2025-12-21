import pandas as pd
import os  # <--- Library wajib untuk mengatur path file
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

# --- 1. SETUP PATH OTOMATIS (SOLUSI ERROR FILENOTFOUND) ---
# Mengambil lokasi folder dimana file modelling.py ini disimpan
script_dir = os.path.dirname(os.path.abspath(__file__))

# Menggabungkan lokasi folder script dengan nama file CSV
# Ini memastikan Python mencari csv di folder 'Membangun_model', bukan folder luar
dataset_path = os.path.join(script_dir, 'water_potability_clean.csv')

print(f"Sedang memuat dataset dari: {dataset_path}")

try:
    df = pd.read_csv(dataset_path)
    print("Dataset berhasil dimuat!")
except FileNotFoundError:
    print("\n❌ ERROR: File CSV tidak ditemukan.")
    print(f"Pastikan file 'water_potability_clean.csv' sudah ada di folder: {script_dir}\n")
    exit()

# --- 2. SPLIT DATA ---
X = df.drop('Potability', axis=1)
y = df['Potability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. SET MLFLOW LOKAL (PENTING UNTUK REVISI) ---
# Kita set agar folder 'mlruns' muncul di dalam folder Membangun_model
# Ini memudahkan Anda mencari artefak untuk screenshot
tracking_uri = "file:///" + os.path.join(script_dir, "mlruns").replace("\\", "/")
mlflow.set_tracking_uri(tracking_uri)

mlflow.set_experiment("Eksperimen_Basic_Yusti")
print(f"MLflow Tracking URI diset ke: {tracking_uri}")

# --- 4. TRAINING DENGAN AUTOLOG ---
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="Basic_RandomForest"):
    print("Memulai training model...")
    
    # Inisialisasi Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train Model
    model.fit(X_train, y_train)
    
    # Evaluasi Sederhana
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Model Basic Selesai Dilatih!")
    print(f"Accuracy: {acc}")
    print("Laporan Klasifikasi:")
    print(classification_report(y_test, y_pred))
    
    print("\n✅ Training Selesai.")
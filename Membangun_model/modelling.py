import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

# 1. Load Dataset (Sesuaikan path jika perlu)
# Pastikan file csv ada di folder yang sama atau sesuaikan path-nya
df = pd.read_csv('water_potability_clean.csv')

# 2. Split Data
X = df.drop('Potability', axis=1)
y = df['Potability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Set MLflow Experiment
# Gunakan nama eksperimen yang sama atau berbeda sedikit untuk membedakan
mlflow.set_experiment("Eksperimen_Basic_Yusti")

# 4. Training dengan Autolog
# Autolog akan otomatis merekam parameter, metrik, dan model tanpa kita suruh manual
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="Basic_RandomForest"):
    # Inisialisasi Model (Tanpa Tuning yang rumit)
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

    # Karena pakai autolog, kita tidak perlu kode 'log_metric' atau 'log_model' manual di sini.
    # Semuanya otomatis tersimpan.
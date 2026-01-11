# ⚙️ MLOps & Supervised Learning Experiments

![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue?style=for-the-badge&logo=mlflow&logoColor=white)
![DagsHub](https://img.shields.io/badge/DagsHub-Storage-black?style=for-the-badge&logo=dagshub&logoColor=white)
![CI/CD](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)

## 🎯 Project Overview

Repositori ini bukan sekadar kumpulan algoritma Machine Learning, melainkan implementasi **End-to-End ML Pipeline**. Fokus utama proyek ini adalah menerapkan prinsip **MLOps** dalam eksperimen Supervised Machine Learning (SML), mencakup:
1.  **Experiment Tracking**: Memantau metrik, parameter, dan artifak model secara sistematis.
2.  **Reproducibility**: Memastikan eksperimen dapat diulang dengan hasil yang konsisten.
3.  **Model Management**: Menyimpan dan mengelola versi model (Model Registry).

---

## 🏗️ Technical Architecture & Stack

Berikut adalah *tools* dan kerangka kerja teknis yang digunakan untuk mengelola siklus hidup Machine Learning dalam repositori ini:

| Komponen | Tools / Library | Fungsi Utama |
| :--- | :--- | :--- |
| **Orchestration & Tracking** | **MLflow** | Logging parameter (learning rate, epochs), metrik (RMSE, Accuracy), dan artifak model. |
| **Version Control** | **Git & DVC** (Optional) | Manajemen versi kode dan dataset (Data Version Control). |
| **Remote Storage** | **DagsHub** | Sentralisasi repositori MLflow dan penyimpanan data jarak jauh. |
| **Automation (CI/CD)** | **GitHub Actions** | Otomatisasi testing skrip dan *linting* kode saat *push*. |
| **Environment** | **Conda / venv** | Isolasi dependensi untuk mencegah konflik *library*. |
| **Modeling Core** | **Scikit-Learn** | Implementasi algoritma Regresi dan Klasifikasi. |

---

## 📊 MLOps Workflow

Setiap eksperimen dalam repositori ini mengikuti alur kerja standar MLOps:

1.  **Data Ingestion & Versioning**: Raw data diproses dan dilacak versinya.
2.  **Preprocessing Pipeline**: Penanganan *missing values*, *encoding*, dan *scaling*.
3.  **Training with Logging**:
    * Setiap *run* training dicatat otomatis ke MLflow server.
    * Mencatat Hyperparameters: `alpha`, `penalty`, `max_depth`.
    * Mencatat Metrics: `MAE`, `RMSE`, `R2 Score`.
4.  **Model Registration**: Model terbaik disimpan (diser) dalam format `.pkl` atau `mlflow.sklearn`.

---

## ⚡ Key Technical implementation

### 1. Feature Engineering & Scaling Pipeline
Menggunakan `Pipeline` dan `ColumnTransformer` dari Scikit-Learn untuk mencegah *data leakage* dan memastikan transformasi data yang konsisten antara training dan inference.

### 2. Handling Target Variable (Inverse Transform)
Salah satu aspek krusial dalam pipeline regresi yang diterapkan di sini adalah **Target Transformation**.
* **Case:** Melakukan scaling pada target variabel ($y$) menggunakan `MinMaxScaler` atau `StandardScaler` untuk mempercepat konvergensi gradien.
* **Implementation:** Memastikan metrik evaluasi dihitung pada **Real Values**, bukan Scaled Values.

```python
# Snippet Implementasi Evaluasi
y_pred_scaled = model.predict(X_test)

# MENGEMBALIKAN KE SKALA ASLI (CRITICAL STEP)
y_pred_real = scaler_target.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_test_real = scaler_target.inverse_transform(y_test_scaled.reshape(-1, 1))

# Logging ke MLflow dengan nilai asli
mlflow.log_metric("rmse_real", np.sqrt(mean_squared_error(y_test_real, y_pred_real)))

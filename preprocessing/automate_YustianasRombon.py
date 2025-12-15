import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os

def main():
    # 1. Definisikan Path (PENTING: Sesuaikan dengan struktur folder repo)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Naik satu level dari folder 'preprocessing' ke root, lalu masuk ke 'water_potability_raw'
    raw_data_path = os.path.join(base_dir, '..', 'water_potability_raw', 'water_potability.csv')
    # Output path di dalam folder 'preprocessing/water_potability_preprocessing'
    output_path = os.path.join(base_dir, 'water_potability_preprocessing', 'water_potability_clean.csv')

    # Cek apakah folder output ada, jika tidak buat baru
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Memulai proses preprocessing otomatis...")
    
    # 2. Load Dataset
    if not os.path.exists(raw_data_path):
        print(f"Error: File tidak ditemukan di {raw_data_path}")
        return

    df = pd.read_csv(raw_data_path)
    print(f"Data mentah dimuat: {df.shape}")

    # 3. Handling Missing Values (Imputasi Mean)
    imputer = SimpleImputer(strategy='mean')
    df_clean = df.copy()
    
    # Pisahkan target untuk sementara agar tidak ikut kena processing jika tidak perlu
    target_col = 'Potability'
    features = df_clean.drop(target_col, axis=1)
    target = df_clean[target_col]
    
    feature_cols = features.columns
    features_imputed = imputer.fit_transform(features)
    
    # 4. Scaling (Standarisasi)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_imputed)
    
    # Gabungkan kembali menjadi DataFrame
    df_features_final = pd.DataFrame(features_scaled, columns=feature_cols)
    df_final = pd.concat([df_features_final, target], axis=1)

    # 5. Simpan Hasil
    df_final.to_csv(output_path, index=False)
    print(f"Sukses! Data bersih disimpan di: {output_path}")

if __name__ == "__main__":
    main()
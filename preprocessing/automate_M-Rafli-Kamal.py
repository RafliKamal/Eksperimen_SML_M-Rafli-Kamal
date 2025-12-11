import pandas as pd
import numpy as np

def preprocess_data(file_path):
    """
    Automatisasi proses preprocessing untuk dataset pinjaman (loan dataset).

    Tahapan:
    1. Load dataset
    2. Bersihkan data (hapus duplikat, hapus nilai kosong)
    3. Tangani outlier (metode IQR)
    4. Encoding kategori (One-Hot Encoding)

    Args:
        file_path (str): Lokasi file CSV mentah.

    Returns:
        pd.DataFrame: DataFrame yang sudah dibersihkan dan dipreproses.
    """
    
    # 1. Load dataset
    print(f"Loading data dari {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {file_path}")
        return None

    initial_rows = df.shape[0]
    print(f"Jumlah baris awal: {initial_rows}")

    # 2. Bersihkan data dasar
    # Hapus data duplikat
    df = df.drop_duplicates()
    print(f"Baris setelah hapus duplikat: {df.shape[0]}")

    # Hapus baris yang punya missing value
    df = df.dropna()
    print(f"Baris setelah hapus missing values: {df.shape[0]}")

    # 3. Deteksi dan penanganan outlier (pakai metode IQR)
    print("Proses handle outlier...")
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Pastikan target variable nggak ikut difilter outlier
    if 'loan_status' in numerical_cols:
        numerical_cols.remove('loan_status')

    rows_before_outlier = df.shape[0]
    
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter data biar cuma yang dalam range aman
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    print(f"Baris setelah buang outlier: {df.shape[0]}")
    print(f"Total outlier yang dihapus: {rows_before_outlier - df.shape[0]}")

    # 4. Encoding kolom kategorikal
    print("Encoding kolom kategori...")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # One-Hot Encoding, drop_first biar lebih rapi dan hindari multicollinearity
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    print("Preprocessing selesai.")
    return df

if __name__ == "__main__":
    # Lokasi file input dan output
    input_file = "loan_data_raw/loan_data.csv"
    output_file = "loan_data_cleaned_automated.csv"
    
    print("Mulai menjalankan automation script...")

    processed_df = preprocess_data(input_file)
    
    if processed_df is not None:
        # Simpan hasilnya
        processed_df.to_csv(output_file, index=False)
        print(f"Data hasil preprocessing disimpan ke {output_file}")
        
        # Print info dan beberapa baris sebagai verifikasi
        print("\nInfo Data Setelah Diproses:")
        print(processed_df.info())
        print("\nContoh isi data:")
        print(processed_df.head())
    else:
        print("Preprocessing gagal.")

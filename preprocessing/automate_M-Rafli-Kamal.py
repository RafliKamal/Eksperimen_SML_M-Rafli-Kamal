import pandas as pd
import numpy as np

def preprocess_data(file_path):
    """
    Automates the preprocessing steps for the loan dataset.
    
    Steps:
    1. Load Dataset
    2. Data Cleaning (Drop duplicates, Drop NA)
    3. Outlier Handling (IQR Method)
    4. Categorical Encoding (One-Hot Encoding)
    
    Args:
        file_path (str): Path to the raw csv file.
        
    Returns:
        pd.DataFrame: Cleaning and preprocessed dataframe.
    """
    
    # 1. Memuat Dataset
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

    initial_rows = df.shape[0]
    print(f"Initial rows: {initial_rows}")

    # 2. Pembersihan Data Dasar
    # Menghapus duplikat
    df = df.drop_duplicates()
    print(f"Rows after dropping duplicates: {df.shape[0]}")

    # Menangani missing values
    df = df.dropna()
    print(f"Rows after dropping missing values: {df.shape[0]}")

    # 3. Deteksi dan Penanganan Outlier (Metode IQR)
    print("Handling outliers...")
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if 'loan_status' in numerical_cols:
        numerical_cols.remove('loan_status') # Exclude target variable

    rows_before_outlier = df.shape[0]
    
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter data
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    print(f"Rows after removing outliers: {df.shape[0]}")
    print(f"Outliers removed: {rows_before_outlier - df.shape[0]}")

    # 4. Encoding Data Kategorikal
    print("Encoding categorical variables...")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # One-Hot Encoding with drop_first=True to avoid multicollinearity
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    print("Preprocessing complete.")
    return df

if __name__ == "__main__":
    # Define file paths
    # Assuming script is run from 'preprocessing' directory, data is in parent dir
    input_file = "../loan_data.csv"
    output_file = "loan_data_cleaned_automated.csv"
    
    print("Starting automation script...")
    
    processed_df = preprocess_data(input_file)
    
    if processed_df is not None:
        # Save to current directory (preprocessing folder usually) or adjust as needed
        processed_df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
        
        # Verification: Print head and info
        print("\nProcessed Data Info:")
        print(processed_df.info())
        print("\nProcessed Data Head:")
        print(processed_df.head())
    else:
        print("Preprocessing failed.")

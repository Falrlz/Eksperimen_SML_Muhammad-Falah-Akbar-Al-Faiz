import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_path, output_path):
    """
    Function to perform the automated preprocessing pipeline.
    It reads raw excel file, cleans and preprocesses the features,
    and writes the final dataframe to a CSV file.
    """
    print(f"Loading raw data from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    df = pd.read_excel(input_path)
    print(f"Loaded dataset shape: {df.shape}")
    
    # 1. Clean Total Charges (convert string spaces to NaN, then fill with 0 and convert to float)
    df['Total Charges'] = df['Total Charges'].astype(str).str.strip()
    df['Total Charges'] = df['Total Charges'].replace('', '0')
    df['Total Charges'] = pd.to_numeric(df['Total Charges'])
    
    # 2. Drop unnecessary / leakage columns
    cols_to_drop = [
        'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 
        'Lat Long', 'Latitude', 'Longitude', 'Churn Label', 
        'Churn Score', 'CLTV', 'Churn Reason'
    ]
    # Check if any columns are present before dropping
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df_clean = df.drop(columns=cols_to_drop)
    
    # 3. Categorical Encoding (One-Hot Encoding)
    cat_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    num_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges']
    
    # One-hot encoding
    df_preprocessed = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True, dtype=int)
    
    # 4. Feature Scaling
    scaler = StandardScaler()
    df_preprocessed[num_cols] = scaler.fit_transform(df_preprocessed[num_cols])
    
    # 5. Save output
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df_preprocessed.to_csv(output_path, index=False)
    print(f"Preprocessed data successfully saved to: {output_path}")
    print(f"Output dataset shape: {df_preprocessed.shape}")
    return df_preprocessed

if __name__ == "__main__":
    # Default paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    default_input = os.path.join(script_dir, "..", "namadataset_raw", "Telco_customer_churn_raw.xlsx")
    default_output = os.path.join(script_dir, "namadataset_preprocessing", "Telco_customer_churn_preprocessing.csv")
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else default_input
    output_file = sys.argv[2] if len(sys.argv) > 2 else default_output
    
    preprocess_data(input_file, output_file)

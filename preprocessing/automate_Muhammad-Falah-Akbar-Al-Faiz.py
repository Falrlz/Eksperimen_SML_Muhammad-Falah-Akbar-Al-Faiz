import os
import sys
import logging
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Configure logging to write to both stdout and a log file
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, "preprocessing.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger("preprocessing")

def preprocess_data(input_path, output_dir):
    """
    Function to perform the automated preprocessing pipeline.
    It reads raw excel file, cleans features, splits train-test,
    and applies StandardScaling & OneHotEncoding to prevent leakage.
    Writes train.csv and test.csv into output_dir.
    """
    logger.info("==========================================")
    logger.info("Starting Data Preprocessing ETL Pipeline")
    logger.info("==========================================")
    
    logger.info(f"Loading raw data from: {input_path}")
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    try:
        df = pd.read_excel(input_path)
        logger.info(f"Successfully loaded dataset. Shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading Excel file: {str(e)}")
        raise e
    
    # 1. Clean Total Charges
    logger.info("Cleaning 'Total Charges' column...")
    df['Total Charges'] = df['Total Charges'].astype(str).str.strip()
    df['Total Charges'] = df['Total Charges'].replace('', '0')
    df['Total Charges'] = pd.to_numeric(df['Total Charges'])
    logger.info(f"Total Charges dtype: {df['Total Charges'].dtype}")
    logger.info(f"Number of empty Total Charges values resolved: {(df['Total Charges'] == 0).sum()}")
    
    # 2. Drop leakage & irrelevant columns
    cols_to_drop = [
        'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 
        'Lat Long', 'Latitude', 'Longitude', 'Churn Label', 
        'Churn Score', 'CLTV', 'Churn Reason'
    ]
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    logger.info(f"Dropping target leakage & constant columns: {cols_to_drop}")
    df_clean = df.drop(columns=cols_to_drop)
    logger.info(f"Shape after columns dropped: {df_clean.shape}")
    
    # 3. Separate features and target
    X = df_clean.drop(columns=['Churn Value'])
    y = df_clean['Churn Value']
    
    # 4. Train-Test Split (80% Train, 20% Test, stratified)
    logger.info("Performing stratified train-test split (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Split completed: Train shape {X_train.shape}, Test shape {X_test.shape}")
    
    # 5. Fit & Transform Preprocessing (Bebas Leakage)
    logger.info("Applying StandardScaler and OneHotEncoder to prevent leakage...")
    num_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges']
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    # Fit & transform StandardScaler on numerical features
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train[num_cols])
    X_test_num_scaled = scaler.transform(X_test[num_cols])
    logger.info("StandardScaler fitted and applied successfully.")
    
    # Fit & transform OneHotEncoder on categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    X_train_cat_encoded = encoder.fit_transform(X_train[cat_cols])
    X_test_cat_encoded = encoder.transform(X_test[cat_cols])
    logger.info("OneHotEncoder fitted and applied successfully.")
    
    # Combine back
    encoded_cat_names = encoder.get_feature_names_out(cat_cols)
    all_col_names = num_cols + list(encoded_cat_names)
    
    X_train_preprocessed = pd.DataFrame(
        np.hstack([X_train_num_scaled, X_train_cat_encoded]), 
        columns=all_col_names, 
        index=X_train.index
    )
    X_test_preprocessed = pd.DataFrame(
        np.hstack([X_test_num_scaled, X_test_cat_encoded]), 
        columns=all_col_names, 
        index=X_test.index
    )
    
    # Add target column back
    train_preprocessed = X_train_preprocessed.copy()
    train_preprocessed['Churn Value'] = y_train
    
    test_preprocessed = X_test_preprocessed.copy()
    test_preprocessed['Churn Value'] = y_test
    
    # 6. Save outputs
    logger.info(f"Saving preprocessed splits to directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_preprocessed.to_csv(train_path, index=False)
    test_preprocessed.to_csv(test_path, index=False)
    
    logger.info(f"Train dataset successfully saved to: {train_path} (Shape: {train_preprocessed.shape})")
    logger.info(f"Test dataset successfully saved to: {test_path} (Shape: {test_preprocessed.shape})")
    
    logger.info("==========================================")
    logger.info("Data Preprocessing ETL Pipeline Completed")
    logger.info("==========================================")
    
    return train_preprocessed, test_preprocessed

if __name__ == "__main__":
    # Default paths relative to this script
    default_input = os.path.join(script_dir, "..", "Telco_customer_churn_raw", "Telco_customer_churn_raw.xlsx")
    default_output_dir = os.path.join(script_dir, "Telco_customer_churn_preprocessing")
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else default_input
    output_directory = sys.argv[2] if len(sys.argv) > 2 else default_output_dir
    
    try:
        preprocess_data(input_file, output_directory)
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        sys.exit(1)

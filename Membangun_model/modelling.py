import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn

# Set up local paths relative to the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(script_dir, "..", "preprocessing", "Telco_customer_churn_preprocessing", "train.csv")
test_path = os.path.join(script_dir, "..", "preprocessing", "Telco_customer_churn_preprocessing", "test.csv")
log_file = os.path.join(script_dir, "modelling.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger("modelling")

def train_and_track():
    logger.info("==========================================")
    logger.info("Starting Baseline Model Training Pipeline")
    logger.info("==========================================")
    
    logger.info("Loading preprocessed dataset splits...")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        logger.error(f"Preprocessed train/test files not found at: {train_path} or {test_path}")
        raise FileNotFoundError("Preprocessed train/test files not found. Make sure ETL script has run successfully.")
        
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logger.info("Dataset splits loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise e
    
    # Separate features and target
    X_train = train_df.drop(columns=['Churn Value'])
    y_train = train_df['Churn Value']
    X_test = test_df.drop(columns=['Churn Value'])
    y_test = test_df['Churn Value']
    
    logger.info(f"Train features shape: {X_train.shape}, Test features shape: {X_test.shape}")
    
    # Set model hyperparameters
    params = {
        "n_estimators": 100,
        "max_depth": 8,
        "min_samples_split": 5,
        "random_state": 42
    }
    
    # Set local MLflow experiment
    mlflow.set_experiment("Telco_Customer_Churn_Eksperimen")
    
    logger.info("Starting local MLflow run...")
    with mlflow.start_run(run_name="baseline_random_forest"):
        # Log model parameters
        mlflow.log_params(params)
        logger.info(f"Logged parameters to MLflow: {params}")
        
        # Train baseline classifier
        logger.info("Training RandomForestClassifier...")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        logger.info("Model training completed.")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate model performance
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0))
        }
        
        # Log metrics
        mlflow.log_metrics(metrics)
        logger.info(f"Logged evaluation metrics to MLflow: {metrics}")
        
        # 1. Generate & save Confusion Matrix Plot
        logger.info("Generating Confusion Matrix plot...")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Retained', 'Churn'], 
                    yticklabels=['Retained', 'Churn'])
        plt.title('Confusion Matrix - Baseline RF')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        cm_plot_path = os.path.join(script_dir, "confusion_matrix.png")
        plt.savefig(cm_plot_path, dpi=100)
        plt.close()
        
        # Log confusion matrix artifact
        mlflow.log_artifact(cm_plot_path)
        logger.info("Logged Confusion Matrix plot as an MLflow artifact.")
        
        # 2. Generate & save Feature Importance Plot
        logger.info("Generating Feature Importance plot...")
        importances = model.feature_importances_
        feature_names = X_train.columns
        indices = np.argsort(importances)[::-1]
        
        # Top 15 features for clean visualization
        top_n = min(15, len(feature_names))
        plt.figure(figsize=(10, 6))
        plt.title(f"Top {top_n} Feature Importances - Baseline RF")
        plt.bar(range(top_n), importances[indices[:top_n]], align="center", color='teal')
        plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
        plt.xlim([-1, top_n])
        plt.tight_layout()
        
        feat_plot_path = os.path.join(script_dir, "feature_importance.png")
        plt.savefig(feat_plot_path, dpi=100)
        plt.close()
        
        # Log feature importance artifact
        mlflow.log_artifact(feat_plot_path)
        logger.info("Logged Feature Importance plot as an MLflow artifact.")
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        logger.info("Logged sklearn model object to MLflow.")
        
        # Clean up local image files
        if os.path.exists(cm_plot_path):
            os.remove(cm_plot_path)
        if os.path.exists(feat_plot_path):
            os.remove(feat_plot_path)
            
    logger.info("==========================================")
    logger.info("Baseline Model Training Completed Successfully")
    logger.info("==========================================")

if __name__ == "__main__":
    try:
        train_and_track()
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)

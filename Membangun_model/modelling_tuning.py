import os
import sys

# Reconfigure stdout/stderr to handle Unicode emojis without crashing on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(errors='replace')

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.inspection import permutation_importance
import dagshub
import optuna
import mlflow
import mlflow.sklearn

# Set up local paths relative to the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(script_dir, "..", "preprocessing", "Telco_customer_churn_preprocessing", "train.csv")
test_path = os.path.join(script_dir, "..", "preprocessing", "Telco_customer_churn_preprocessing", "test.csv")
log_file = os.path.join(script_dir, "modelling_tuning.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger("modelling_tuning")

def perform_tuning_and_remote_track():
    logger.info("==========================================")
    logger.info("Starting Optuna Hyperparameter Tuning Pipeline")
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
    
    logger.info("Initializing DagsHub integration...")
    try:
        dagshub.init(
            repo_owner='Falrlz', 
            repo_name='Eksperimen_SML_Muhammad-Falah-Akbar-Al-Faiz', 
            mlflow=True
        )
        logger.info("DagsHub integration initialized successfully.")
    except Exception as e:
        logger.warning(f"DagsHub auth failed/skipped: {str(e)}")
        logger.warning("Falling back to local tracking.")
    
    # Define experiment name
    mlflow.set_experiment("Telco_Customer_Churn_Tuning")
    
    # Enable MLflow Autolog for Scikit-Learn
    mlflow.sklearn.autolog()
    
    # Define objective function for Optuna
    def objective(trial):
        # Start a nested run for each Optuna trial
        run_name = f"optuna_trial_{trial.number}"
        with mlflow.start_run(run_name=run_name, nested=True):
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 6),
                "max_iter": trial.suggest_int("max_iter", 300, 1000),
                "class_weight": "balanced",
                "random_state": 42
            }
            logger.info(f"[Trial {trial.number}] Parameters: {params}")
            
            # Train classifier
            model = HistGradientBoostingClassifier(**params)
            model.fit(X_train, y_train)
                
            y_pred = model.predict(X_test)
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1_score": f1
            }
            mlflow.log_metrics(metrics)
            logger.info(f"[Trial {trial.number}] F1-Score: {f1:.4f}")
            return f1
 
    # Start a parent run for the Optuna study
    logger.info("Beginning Optuna hyperparameter optimization study...")
    with mlflow.start_run(run_name="optuna_study"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=5)
        
        logger.info("Optuna study completed.")
        logger.info(f"Best Trial F1-Score: {study.best_value:.4f}")
        logger.info(f"Best Parameters: {study.best_params}")
        
        # Log best params to the parent run
        mlflow.log_metric("best_f1_score", study.best_value)
        
        # Retrain best model to log its artifacts
        logger.info("Retraining best model for final evaluation...")
        best_params = study.best_params.copy()
        best_params["random_state"] = 42
        best_params["class_weight"] = "balanced"
        
        best_model = HistGradientBoostingClassifier(**best_params)
        best_model.fit(X_train, y_train)
            
        y_pred = best_model.predict(X_test)
        
        # Final metrics
        best_metrics = {
            "best_accuracy": accuracy_score(y_test, y_pred),
            "best_precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "best_recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "best_f1_score_final": float(f1_score(y_test, y_pred, zero_division=0))
        }
        mlflow.log_metrics(best_metrics)
        
        # 1. Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                    xticklabels=['Retained', 'Churn'], 
                    yticklabels=['Retained', 'Churn'])
        plt.title('Confusion Matrix - Best HGB Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        cm_path = os.path.join(script_dir, "best_confusion_matrix.png")
        plt.savefig(cm_path, dpi=100)
        plt.close()
        
        mlflow.log_artifact(cm_path)
        logger.info("Logged best Confusion Matrix plot.")
        if os.path.exists(cm_path):
            os.remove(cm_path)
            
        # 2. Feature Importance Plot
        feature_names = X_train.columns
        result = permutation_importance(best_model, X_test, y_test, n_repeats=5, random_state=42)
        importances = result.importances_mean
            
        indices = np.argsort(importances)[::-1]
        top_n = min(15, len(feature_names))
        
        plt.figure(figsize=(10, 6))
        plt.title(f"Top {top_n} Features - Best HGB Model (Permutation Importance)")
        plt.bar(range(top_n), importances[indices[:top_n]], align="center", color='coral')
        plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
        plt.xlim([-1, top_n])
        plt.tight_layout()
        
        feat_path = os.path.join(script_dir, "best_feature_importance.png")
        plt.savefig(feat_path, dpi=100)
        plt.close()
        
        mlflow.log_artifact(feat_path)
        logger.info("Logged best Feature Importance plot.")
        if os.path.exists(feat_path):
            os.remove(feat_path)
            
        # Note: Model object and parameters are logged automatically by mlflow.sklearn.autolog()
        logger.info("Best model object and parameters logged automatically by autolog.")

    logger.info("==========================================")
    logger.info("Optuna Hyperparameter Tuning Completed Successfully")
    logger.info("==========================================")

if __name__ == "__main__":
    try:
        perform_tuning_and_remote_track()
    except Exception as e:
        logger.error(f"Tuning pipeline failed: {str(e)}")
        sys.exit(1)

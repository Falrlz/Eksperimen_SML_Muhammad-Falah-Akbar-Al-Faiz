import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import dagshub
import mlflow
import mlflow.sklearn

# Set up local paths relative to the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(script_dir, "..", "preprocessing", "Telco_customer_churn_preprocessing", "train.csv")
test_path = os.path.join(script_dir, "..", "preprocessing", "Telco_customer_churn_preprocessing", "test.csv")

def perform_tuning_and_remote_track():
    print("Loading preprocessed dataset splits...")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Preprocessed train/test files not found. Make sure ETL script has run successfully.")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Separate features and target
    X_train = train_df.drop(columns=['Churn Value'])
    y_train = train_df['Churn Value']
    X_test = test_df.drop(columns=['Churn Value'])
    y_test = test_df['Churn Value']
    
    print("Initializing DagsHub integration...")
    try:
        dagshub.init(
            repo_owner='Falrlz', 
            repo_name='Eksperimen_SML_Muhammad-Falah-Akbar-Al-Faiz', 
            mlflow=True
        )
        print("DagsHub integration initialized successfully.")
    except Exception as e:
        print(f"Warning: DagsHub auth failed/skipped: {str(e)}")
        print("Falling back to local tracking.")
    
    # Define experiment name
    mlflow.set_experiment("Telco_Customer_Churn_Tuning")
    
    # Define parameter search space
    grid = [
        {"n_estimators": 50, "max_depth": 6, "min_samples_split": 5, "random_state": 42},
        {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5, "random_state": 42},
        {"n_estimators": 150, "max_depth": 12, "min_samples_split": 2, "random_state": 42},
    ]
    
    print(f"Beginning hyperparameter grid evaluation ({len(grid)} configurations)...")
    
    for idx, params in enumerate(grid):
        run_name = f"tuning_rf_config_{idx + 1}"
        print(f"\n[Config {idx + 1}/{len(grid)}] Training with parameters: {params}")
        
        with mlflow.start_run(run_name=run_name):
            # Log hyperparameters
            mlflow.log_params(params)
            
            # Train model
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            
            # Evaluate performance
            y_pred = model.predict(X_test)
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0)
            }
            
            # Log metrics
            mlflow.log_metrics(metrics)
            print(f"Metrics: {metrics}")
            
            # 1. Confusion Matrix Plot
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                        xticklabels=['Retained', 'Churn'], 
                        yticklabels=['Retained', 'Churn'])
            plt.title(f'Confusion Matrix - Config {idx + 1}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            cm_path = os.path.join(script_dir, f"cm_config_{idx + 1}.png")
            plt.savefig(cm_path, dpi=100)
            plt.close()
            
            mlflow.log_artifact(cm_path)
            if os.path.exists(cm_path):
                os.remove(cm_path)
                
            # 2. Feature Importance Plot
            importances = model.feature_importances_
            feature_names = X_train.columns
            indices = np.argsort(importances)[::-1]
            top_n = min(15, len(feature_names))
            
            plt.figure(figsize=(10, 6))
            plt.title(f"Top {top_n} Features - Config {idx + 1}")
            plt.bar(range(top_n), importances[indices[:top_n]], align="center", color='coral')
            plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
            plt.xlim([-1, top_n])
            plt.tight_layout()
            
            feat_path = os.path.join(script_dir, f"feat_config_{idx + 1}.png")
            plt.savefig(feat_path, dpi=100)
            plt.close()
            
            mlflow.log_artifact(feat_path)
            if os.path.exists(feat_path):
                os.remove(feat_path)
                
            # Log model
            mlflow.sklearn.log_model(model, "model")
            print(f"Finished and logged run '{run_name}' to MLflow.")

    print("\nAll tuning runs successfully completed and tracked!")

if __name__ == "__main__":
    perform_tuning_and_remote_track()

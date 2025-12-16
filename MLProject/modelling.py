import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import mlflow
import mlflow.sklearn
import joblib
import os
import sys

# =====================================================
# CONFIGURATION
# =====================================================
EXPERIMENT_NAME = "Diabetes_Classification_MLProject"

# =====================================================
# PARSE PARAMETERS
# =====================================================
def get_parameters():
    """Get parameters from environment or use defaults"""
    
    params = {
        'data_path': 'diabetes_preprocessing.csv',
        'model_type': 'RandomForest',
        'test_size': 0.2,
        'random_state': 42,
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5
    }
    
    return params

# =====================================================
# LOAD DATA
# =====================================================
def load_data(path):
    print(f"📂 Loading data from: {path}")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    
    df = pd.read_csv(path)
    print(f"✅ Data loaded: {df.shape}")
    return df

# =====================================================
# PREPARE DATA
# =====================================================
def prepare_data(df, test_size, random_state):
    print(f"\n🔄 Preparing data (test_size={test_size})...")
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✅ Train size: {X_train.shape}")
    print(f"✅ Test size: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

# =====================================================
# TRAIN MODEL WITH MLFLOW
# =====================================================
def train_model(X_train, X_test, y_train, y_test, params):
    print("\n🤖 Training model with MLflow...")
    
    # Set experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Enable autolog
    mlflow.sklearn.autolog()
    
    # Start MLflow run
    with mlflow.start_run(run_name="MLProject_Training"):
        
        # Model parameters
        model_params = {
            'n_estimators': params['n_estimators'],
            'max_depth': params['max_depth'],
            'min_samples_split': params['min_samples_split'],
            'random_state': params['random_state'],
            'n_jobs': -1
        }
        
        # Train
        print("⏳ Training RandomForest...")
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n✅ Model Training Complete!")
        print(f"📊 Accuracy: {accuracy:.4f}")
        print(f"📊 Precision: {precision:.4f}")
        print(f"📊 Recall: {recall:.4f}")
        print(f"📊 F1-Score: {f1:.4f}")
        print(f"📊 ROC-AUC: {roc_auc:.4f}")
        
        # Get run ID
        run_id = mlflow.active_run().info.run_id
        print(f"\n🔗 Run ID: {run_id}")
        
        return model, y_pred, y_test, X_test, {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }

# =====================================================
# SAVE VISUALIZATIONS
# =====================================================
def save_visualizations(model, X_test, y_test, y_pred):
    print("\n📊 Saving visualizations...")
    
    os.makedirs('artifacts', exist_ok=True)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - MLflow Project')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = 'artifacts/confusion_matrix.png'
    plt.savefig(cm_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved: {cm_path}")
    
    # Feature Importance
    feature_names = X_test.columns
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance - MLflow Project')
    plt.bar(range(X_test.shape[1]), importances[indices])
    plt.xticks(range(X_test.shape[1]), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    fi_path = 'artifacts/feature_importance.png'
    plt.savefig(fi_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"✅ Feature importance saved: {fi_path}")
    
    # Classification Report
    report = classification_report(y_test, y_pred)
    report_path = 'artifacts/classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✅ Classification report saved: {report_path}")

# =====================================================
# SAVE MODEL
# =====================================================
def save_model(model, path='artifacts/model.pkl'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"✅ Model saved: {path}")
    return path

# =====================================================
# MAIN EXECUTION
# =====================================================
if __name__ == "__main__":
    print("="*70)
    print("🚀 MLFLOW PROJECT - DIABETES CLASSIFICATION TRAINING")
    print("="*70)
    
    try:
        # Get parameters
        params = get_parameters()
        print(f"\n📋 Parameters:")
        for key, value in params.items():
            print(f"   {key}: {value}")
        
        # Load & prepare data
        df = load_data(params['data_path'])
        X_train, X_test, y_train, y_test = prepare_data(
            df, params['test_size'], params['random_state']
        )
        
        # Train model with MLflow
        model, y_pred, y_test_actual, X_test_actual, metrics = train_model(
            X_train, X_test, y_train, y_test, params
        )
        
        # Save visualizations & model
        save_visualizations(model, X_test_actual, y_test_actual, y_pred)
        model_path = save_model(model)
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"\n📁 Artifacts saved in: artifacts/")
        print(f"📁 Model saved at: {model_path}")
        print(f"\n🔗 View results with: mlflow ui")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
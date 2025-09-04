import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_error, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
from tabulate import tabulate
import argparse
import sys
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = None
    XGBRegressor = None

def classification_metrics(y_true, y_pred, y_proba=None):
    """Calculate comprehensive classification metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1_Score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    if y_proba is not None:
        try:
            metrics['ROC_AUC'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
        except:
            metrics['ROC_AUC'] = 0.0
    return metrics

def regression_metrics(y_true, y_pred):
    """Calculate comprehensive regression metrics"""
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2_Score': r2_score(y_true, y_pred)
    }

def detect_problem_type(y):
    """Detect if target is classification or regression"""
    unique_values = len(np.unique(y))
    is_numeric = np.issubdtype(y.dtype, np.number)
    return 'classification' if unique_values <= 10 or not is_numeric else 'regression'

def preprocess_data(X, y):
    """Preprocess data: handle missing values, encode categoricals, standardize"""
    # Convert to DataFrame if numpy array
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    
    # Handle missing values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    if len(numeric_cols) > 0:
        X[numeric_cols] = SimpleImputer(strategy='mean').fit_transform(X[numeric_cols])
    
    if len(categorical_cols) > 0:
        X[categorical_cols] = SimpleImputer(strategy='most_frequent').fit_transform(X[categorical_cols])
        # Encode categorical variables
        for col in categorical_cols:
            X[col] = LabelEncoder().fit_transform(X[col])
    
    return X.values, y

def get_classification_models():
    """Return classification models grouped by scaling requirement"""
    scaled_models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'Neural Network': MLPClassifier(random_state=42, max_iter=2000)
    }
    
    unscaled_models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=10, random_state=42),
        'Naive Bayes': GaussianNB(),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    if XGBClassifier:
        unscaled_models['XGBoost'] = XGBClassifier(random_state=42, eval_metric='logloss')
    
    return scaled_models, unscaled_models

def get_regression_models():
    """Return regression models grouped by scaling requirement"""
    scaled_models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(random_state=42),
        'Lasso': Lasso(random_state=42),
        'SVR': SVR(),
        'Neural Network': MLPRegressor(random_state=42, max_iter=2000)
    }
    
    unscaled_models = {
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    
    if XGBRegressor:
        unscaled_models['XGBoost'] = XGBRegressor(random_state=42)
    
    return scaled_models, unscaled_models

def train_and_evaluate_models(models, X_train, X_test, y_train, y_test, problem_type):
    """Train models and evaluate performance"""
    results = {}
    predictions = {}
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        trained_models[name] = model
        
        if problem_type == 'classification':
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') and len(np.unique(y_test)) == 2 else None
            results[name] = classification_metrics(y_test, y_pred, y_proba)
        else:
            results[name] = regression_metrics(y_test, y_pred)
    
    return results, predictions, trained_models

def plot_feature_importance(trained_models, feature_names=None):
    """Plot feature importance for tree-based models"""
    tree_models = {}
    for name, model in trained_models.items():
        if hasattr(model, 'feature_importances_'):
            tree_models[name] = model.feature_importances_
    
    if not tree_models:
        return
    
    n_features = len(list(tree_models.values())[0])
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    n_models = len(tree_models)
    cols = 2
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (model_name, importance) in enumerate(tree_models.items()):
        row, col = idx // cols, idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        indices = np.argsort(importance)[::-1]
        ax.bar(range(len(importance)), importance[indices])
        ax.set_title(f'{model_name} - Feature Importance')
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance')
        ax.set_xticks(range(len(importance)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45)
    
    # Hide empty subplots
    for idx in range(n_models, rows * cols):
        row, col = idx // cols, idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrices(y_test, predictions):
    """Plot confusion matrices for classification models"""
    n_models = len(predictions)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (model_name, y_pred) in enumerate(predictions.items()):
        row, col = idx // cols, idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{model_name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    # Hide empty subplots
    for idx in range(n_models, rows * cols):
        row, col = idx // cols, idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def print_results_table(results, problem_type):
    """Print formatted results table with best model"""
    if problem_type == 'classification':
        # Print accuracy only
        print(f"\n{problem_type.upper()} RESULTS - ACCURACY:")
        print("=" * 40)
        accuracies = {}
        for model, metrics in results.items():
            accuracy = metrics['Accuracy']
            accuracies[model] = accuracy
            print(f"{model:<20}: {accuracy:.3f}")
        
        # Find best model
        best_model = max(accuracies, key=accuracies.get)
        print(f"\n🏆 BEST MODEL: {best_model} (Accuracy: {accuracies[best_model]:.3f})")
        
    else:
        # Print R² score only for regression
        print(f"\n{problem_type.upper()} RESULTS - R² SCORE:")
        print("=" * 40)
        r2_scores = {}
        for model, metrics in results.items():
            r2 = metrics['R2_Score']
            r2_scores[model] = r2
            print(f"{model:<20}: {r2:.3f}")
        
        # Find best model
        best_model = max(r2_scores, key=r2_scores.get)
        print(f"\n🏆 BEST MODEL: {best_model} (R² Score: {r2_scores[best_model]:.3f})")

def test_ml_algorithms(X, y):
    """Main function to test all ML algorithms"""
    problem_type = detect_problem_type(y)
    original_X = X.copy() if hasattr(X, 'copy') else X
    feature_names = list(X.columns) if hasattr(X, 'columns') else None
    
    X, y = preprocess_data(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if problem_type == 'classification':
        scaled_models, unscaled_models = get_classification_models()
    else:
        scaled_models, unscaled_models = get_regression_models()
    
    # Evaluate scaled models
    scaled_results, scaled_predictions, scaled_trained = train_and_evaluate_models(scaled_models, X_train_scaled, X_test_scaled, y_train, y_test, problem_type)
    
    # Evaluate unscaled models
    unscaled_results, unscaled_predictions, unscaled_trained = train_and_evaluate_models(unscaled_models, X_train, X_test, y_train, y_test, problem_type)
    
    # Combine results
    all_results = {**scaled_results, **unscaled_results}
    all_predictions = {**scaled_predictions, **unscaled_predictions}
    all_trained = {**scaled_trained, **unscaled_trained}
    
    # Skip visualizations for batch file usage
    # Visualizations commented out for faster execution
    
    return problem_type, all_results

def load_data_from_file(File_path, target_column=None):
    """Load data from CSV file"""
    try:
        data = pd.read_csv(File_path)
        if target_column:
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            X = data.drop(columns=[target_column])
            y = data[target_column].values
        else:
            # Assume last column is target
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1].values
        
        # Auto-encode categorical target
        if y.dtype == 'object' or not np.issubdtype(y.dtype, np.number):
            y = LabelEncoder().fit_transform(y)
            print(f"Target column auto-encoded (categorical to numeric)")
        
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

# Configuration - Set your file path here
FILE_PATH = r"C:\Users\aditi\Downloads\regression_test.csv"
TARGET_COLUMN = None

def main():
    parser = argparse.ArgumentParser(description='ML Algorithm Accuracy Testing')
    parser.add_argument('--file', '-f', type=str, default=FILE_PATH, help='Path to CSV file')
    parser.add_argument('--target', '-t', type=str, default=TARGET_COLUMN, help='Target column name (default: last column)')
    
    args = parser.parse_args()
    
    # Use configured file path if it exists
    import os
    if os.path.exists(args.file):
        X, y = load_data_from_file(args.file, args.target)
        print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
        problem_type, results = test_ml_algorithms(X, y)
        print_results_table(results, problem_type)
    else:
        print("No file provided. Running examples with synthetic data...")
        
        # Test classification
        X_class, y_class = make_classification(n_samples=200, n_features=4, n_classes=2, random_state=42)
        problem_type, results = test_ml_algorithms(X_class, y_class)
        print_results_table(results, problem_type)
        
        # Test regression
        X_reg, y_reg = make_regression(n_samples=200, n_features=4, random_state=42)
        problem_type, results = test_ml_algorithms(X_reg, y_reg)
        print_results_table(results, problem_type)

if __name__ == "__main__":
    main()
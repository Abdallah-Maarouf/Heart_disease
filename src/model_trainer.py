"""
Supervised Learning Model Training Infrastructure

This module provides comprehensive training functionality for multiple classification models
including Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine.
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from typing import Dict, Tuple, Any, List
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


class SupervisedTrainer:
    """
    Comprehensive supervised learning model trainer for heart disease classification.
    
    This class handles data preparation, model training, validation, and persistence
    for multiple classification algorithms.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the SupervisedTrainer.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.training_log = {
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'data_info': {},
            'cross_validation_scores': {}
        }
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        
    def prepare_data(self, data_path: str, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data with stratified train-test split.
        
        Args:
            data_path (str): Path to the processed dataset
            test_size (float): Proportion of data for testing (default: 0.2)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print(f"Loading data from {data_path}...")
        
        # Load the feature-selected dataset
        df = pd.read_csv(data_path)
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Log data information (convert numpy types to native Python types for JSON serialization)
        class_counts = y.value_counts()
        self.training_log['data_info'] = {
            'total_samples': int(len(df)),
            'n_features': int(X.shape[1]),
            'feature_names': list(X.columns),
            'class_distribution': {str(k): int(v) for k, v in class_counts.to_dict().items()},
            'class_balance': {
                'class_0': int((y == 0).sum()),
                'class_1': int((y == 1).sum()),
                'balance_ratio': float((y == 1).sum() / (y == 0).sum())
            }
        }
        
        print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Class distribution: {dict(y.value_counts())}")
        
        # Stratified train-test split to maintain class balance
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state, 
            stratify=y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Train set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_logistic_regression(self, C: float = 1.0, penalty: str = 'l2', max_iter: int = 1000) -> LogisticRegression:
        """
        Train Logistic Regression model with regularization parameters.
        
        Args:
            C (float): Regularization strength (default: 1.0)
            penalty (str): Regularization penalty ('l1', 'l2', 'elasticnet', 'none')
            max_iter (int): Maximum iterations for convergence
            
        Returns:
            Trained LogisticRegression model
        """
        print(f"Training Logistic Regression (C={C}, penalty={penalty})...")
        
        model = LogisticRegression(
            C=C,
            penalty=penalty,
            random_state=self.random_state,
            max_iter=max_iter,
            solver='liblinear' if penalty in ['l1', 'l2'] else 'lbfgs'
        )
        
        model.fit(self.X_train_scaled, self.y_train)
        
        # Store model and parameters
        self.models['logistic_regression'] = model
        self.training_log['models']['logistic_regression'] = {
            'parameters': {'C': C, 'penalty': penalty, 'max_iter': max_iter},
            'training_time': datetime.now().isoformat()
        }
        
        print("Logistic Regression training completed.")
        return model
    
    def train_decision_tree(self, max_depth: int = None, min_samples_split: int = 2, 
                          min_samples_leaf: int = 1, criterion: str = 'gini') -> DecisionTreeClassifier:
        """
        Train Decision Tree model with pruning parameters.
        
        Args:
            max_depth (int): Maximum depth of the tree
            min_samples_split (int): Minimum samples required to split a node
            min_samples_leaf (int): Minimum samples required at a leaf node
            criterion (str): Split quality measure ('gini' or 'entropy')
            
        Returns:
            Trained DecisionTreeClassifier model
        """
        print(f"Training Decision Tree (max_depth={max_depth}, min_samples_split={min_samples_split})...")
        
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=self.random_state
        )
        
        model.fit(self.X_train_scaled, self.y_train)
        
        # Store model and parameters
        self.models['decision_tree'] = model
        self.training_log['models']['decision_tree'] = {
            'parameters': {
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'criterion': criterion
            },
            'training_time': datetime.now().isoformat()
        }
        
        print("Decision Tree training completed.")
        return model
    
    def train_random_forest(self, n_estimators: int = 100, max_features: str = 'sqrt', 
                          max_depth: int = None, min_samples_split: int = 2) -> RandomForestClassifier:
        """
        Train Random Forest model with ensemble parameters.
        
        Args:
            n_estimators (int): Number of trees in the forest
            max_features (str): Number of features to consider for best split
            max_depth (int): Maximum depth of trees
            min_samples_split (int): Minimum samples required to split a node
            
        Returns:
            Trained RandomForestClassifier model
        """
        print(f"Training Random Forest (n_estimators={n_estimators}, max_features={max_features})...")
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=self.random_state,
            n_jobs=-1  # Use all available cores
        )
        
        model.fit(self.X_train_scaled, self.y_train)
        
        # Store model and parameters
        self.models['random_forest'] = model
        self.training_log['models']['random_forest'] = {
            'parameters': {
                'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split
            },
            'training_time': datetime.now().isoformat()
        }
        
        print("Random Forest training completed.")
        return model
    
    def train_svm(self, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale', 
                  degree: int = 3) -> SVC:
        """
        Train Support Vector Machine model with kernel options.
        
        Args:
            kernel (str): Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            C (float): Regularization parameter
            gamma (str): Kernel coefficient ('scale', 'auto', or float)
            degree (int): Degree for polynomial kernel
            
        Returns:
            Trained SVC model
        """
        print(f"Training SVM (kernel={kernel}, C={C})...")
        
        model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            degree=degree,
            random_state=self.random_state,
            probability=True  # Enable probability estimates for ROC-AUC
        )
        
        model.fit(self.X_train_scaled, self.y_train)
        
        # Store model and parameters
        self.models['svm'] = model
        self.training_log['models']['svm'] = {
            'parameters': {
                'kernel': kernel,
                'C': C,
                'gamma': gamma,
                'degree': degree
            },
            'training_time': datetime.now().isoformat()
        }
        
        print("SVM training completed.")
        return model
    
    def train_all_models(self) -> Dict[str, Any]:
        """
        Train all classification models with default parameters.
        
        Returns:
            Dictionary containing all trained models
        """
        print("Training all models with default parameters...")
        
        # Train each model with default parameters
        self.train_logistic_regression()
        self.train_decision_tree()
        self.train_random_forest()
        self.train_svm()
        
        print(f"All {len(self.models)} models trained successfully.")
        return self.models
    
    def evaluate_model(self, model_name: str, model: Any) -> Dict[str, float]:
        """
        Evaluate a single model and return performance metrics.
        
        Args:
            model_name (str): Name of the model
            model (Any): Trained model object
            
        Returns:
            Dictionary containing performance metrics
        """
        # Make predictions
        y_pred = model.predict(self.X_test_scaled)
        y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, average='binary'),
            'recall': recall_score(self.y_test, y_pred, average='binary'),
            'f1_score': f1_score(self.y_test, y_pred, average='binary')
        }
        
        # Add ROC-AUC if probability predictions are available
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(self.y_test, y_pred_proba)
        
        return metrics
    
    def cross_validate_models(self, cv_folds: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Perform cross-validation for all trained models.
        
        Args:
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            Dictionary containing cross-validation scores for each model
        """
        print(f"Performing {cv_folds}-fold cross-validation...")
        
        cv_results = {}
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Combine training and test data for cross-validation
        X_full = np.vstack([self.X_train_scaled, self.X_test_scaled])
        y_full = np.hstack([self.y_train, self.y_test])
        
        for model_name, model in self.models.items():
            print(f"Cross-validating {model_name}...")
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X_full, y_full, cv=skf, scoring='accuracy')
            
            cv_results[model_name] = {
                'mean_accuracy': cv_scores.mean(),
                'std_accuracy': cv_scores.std(),
                'individual_scores': cv_scores.tolist()
            }
            
            print(f"{model_name} CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.training_log['cross_validation_scores'] = cv_results
        return cv_results
    
    def save_models(self, models_dir: str = 'models/supervised') -> List[str]:
        """
        Save all trained models to disk with descriptive names.
        
        Args:
            models_dir (str): Directory to save models
            
        Returns:
            List of saved model file paths
        """
        print(f"Saving models to {models_dir}...")
        
        # Create directory if it doesn't exist
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        for model_name, model in self.models.items():
            # Create descriptive filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_{timestamp}.pkl"
            filepath = Path(models_dir) / filename
            
            # Save model
            joblib.dump(model, filepath)
            saved_files.append(str(filepath))
            
            print(f"Saved {model_name} to {filepath}")
        
        # Also save the scaler
        scaler_path = Path(models_dir) / f"scaler_{timestamp}.pkl"
        joblib.dump(self.scaler, scaler_path)
        saved_files.append(str(scaler_path))
        
        return saved_files
    
    def save_training_log(self, log_path: str = 'results/training_log.json') -> None:
        """
        Save training log with model parameters and basic metrics.
        
        Args:
            log_path (str): Path to save the training log
        """
        # Create directory if it doesn't exist
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Add basic evaluation metrics to the log
        for model_name, model in self.models.items():
            metrics = self.evaluate_model(model_name, model)
            self.training_log['models'][model_name]['test_metrics'] = metrics
        
        # Save log to JSON file
        with open(log_path, 'w') as f:
            json.dump(self.training_log, f, indent=2)
        
        print(f"Training log saved to {log_path}")
    
    def get_model_summary(self) -> pd.DataFrame:
        """
        Get a summary of all trained models and their performance.
        
        Returns:
            DataFrame containing model performance summary
        """
        summary_data = []
        
        for model_name, model in self.models.items():
            metrics = self.evaluate_model(model_name, model)
            
            summary_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'ROC-AUC': f"{metrics.get('roc_auc', 'N/A')}"
            })
        
        return pd.DataFrame(summary_data)


def main():
    """
    Main function to demonstrate the SupervisedTrainer usage.
    """
    # Initialize trainer
    trainer = SupervisedTrainer(random_state=42)
    
    # Prepare data
    trainer.prepare_data('data/processed/heart_disease_selected.csv')
    
    # Train all models
    models = trainer.train_all_models()
    
    # Perform cross-validation
    cv_results = trainer.cross_validate_models()
    
    # Save models and training log
    saved_files = trainer.save_models()
    trainer.save_training_log()
    
    # Display summary
    print("\n" + "="*60)
    print("MODEL TRAINING SUMMARY")
    print("="*60)
    summary_df = trainer.get_model_summary()
    print(summary_df.to_string(index=False))
    
    print(f"\nModels saved: {len(saved_files)} files")
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
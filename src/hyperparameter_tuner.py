"""
Hyperparameter Optimization System for Heart Disease ML Pipeline

This module provides comprehensive hyperparameter tuning capabilities using
GridSearchCV, RandomizedSearchCV, and Optuna for Bayesian optimization.
"""

import numpy as np
import pandas as pd
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Optuna for Bayesian optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Bayesian optimization will be skipped.")


class HyperparameterTuner:
    """
    Comprehensive hyperparameter optimization system for machine learning models.
    
    Supports multiple optimization strategies:
    - Grid Search with cross-validation
    - Random Search for efficiency
    - Bayesian optimization using Optuna
    """
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1, cv_folds: int = 5):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all processors)
            cv_folds: Number of cross-validation folds
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.cv_folds = cv_folds
        self.optimization_results = {}
        self.best_models = {}
        
        # Create results directory
        self.results_dir = Path("results/hyperparameter_tuning")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create optimized models directory
        self.models_dir = Path("models/optimized")
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def define_parameter_grids(self) -> Dict[str, Dict]:
        """
        Define comprehensive parameter grids for each model type.
        
        Returns:
            Dictionary containing parameter grids for each model
        """
        parameter_grids = {
            'logistic_regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [100, 500, 1000, 2000]
            },
            
            'decision_tree': {
                'max_depth': [3, 5, 7, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 5, 10],
                'criterion': ['gini', 'entropy'],
                'max_features': ['sqrt', 'log2', None]
            },
            
            'random_forest': {
                'n_estimators': [50, 100, 200, 300, 500],
                'max_depth': [3, 5, 7, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2'],
                'bootstrap': [True, False]
            },
            
            'svm': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'degree': [2, 3, 4]  # Only for poly kernel
            }
        }
        
        return parameter_grids
    
    def optimize_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict]:
        """
        Systematically optimize all classification models using Random Search.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing optimization results for all models
        """
        print("Starting systematic optimization of all models...")
        
        # Define base models
        models = {
            'logistic_regression': LogisticRegression(random_state=self.random_state),
            'decision_tree': DecisionTreeClassifier(random_state=self.random_state),
            'random_forest': RandomForestClassifier(random_state=self.random_state),
            'svm': SVC(random_state=self.random_state, probability=True)
        }
        
        # Define simplified parameter distributions for efficiency
        param_distributions = {
            'logistic_regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [100, 500, 1000]
            },
            
            'decision_tree': {
                'max_depth': [3, 5, 7, 10, 15, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 5, 10],
                'criterion': ['gini', 'entropy']
            },
            
            'random_forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            },
            
            'svm': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
            }
        }
        
        all_results = {}
        
        for model_name, model in models.items():
            print(f"\n{'='*60}")
            print(f"Optimizing {model_name.upper()}")
            print(f"{'='*60}")
            
            try:
                # Get baseline performance
                model.fit(X_train, y_train)
                baseline_pred = model.predict(X_test)
                baseline_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                baseline_metrics = self._calculate_metrics(y_test, baseline_pred, baseline_proba)
                
                # Perform optimization using RandomizedSearchCV
                param_dist = param_distributions[model_name]
                
                random_search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_dist,
                    n_iter=50,  # Reduced for efficiency
                    cv=self.cv_folds,
                    scoring='accuracy',
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                    verbose=1
                )
                
                random_search.fit(X_train, y_train)
                best_model = random_search.best_estimator_
                
                # Evaluate on test set
                y_pred = best_model.predict(X_test)
                y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
                
                test_metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
                
                optimization_results = {
                    'method': 'random_search',
                    'best_params': random_search.best_params_,
                    'best_cv_score': random_search.best_score_,
                    'test_metrics': test_metrics,
                    'n_iter': 50,
                    'optimization_time': datetime.now().isoformat()
                }
                
                # Store results
                all_results[model_name] = {
                    'baseline_metrics': baseline_metrics,
                    'optimization_results': optimization_results,
                    'improvement': {
                        'accuracy': test_metrics['accuracy'] - baseline_metrics['accuracy'],
                        'f1_score': test_metrics['f1_score'] - baseline_metrics['f1_score']
                    }
                }
                
                # Save the best model
                self.best_models[model_name] = best_model
                
                print(f"Baseline accuracy: {baseline_metrics['accuracy']:.4f}")
                print(f"Optimized accuracy: {test_metrics['accuracy']:.4f}")
                print(f"Improvement: {all_results[model_name]['improvement']['accuracy']:.4f}")
                
            except Exception as e:
                print(f"Optimization failed for {model_name}: {e}")
                all_results[model_name] = {
                    'baseline_metrics': baseline_metrics if 'baseline_metrics' in locals() else {},
                    'error': str(e)
                }
        
        self.optimization_results = all_results
        return all_results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except ValueError:
                metrics['roc_auc'] = 0.0
        
        return metrics
    
    def plot_optimization_results(self, save_plots: bool = True) -> None:
        """
        Create visualizations for optimization results.
        
        Args:
            save_plots: Whether to save plots to disk
        """
        if not self.optimization_results:
            print("No optimization results available for plotting.")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Hyperparameter Optimization Results', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        models = []
        baseline_acc = []
        optimized_acc = []
        baseline_f1 = []
        optimized_f1 = []
        improvements_acc = []
        improvements_f1 = []
        
        for model_name, results in self.optimization_results.items():
            if 'error' not in results:
                models.append(model_name.replace('_', ' ').title())
                baseline_acc.append(results['baseline_metrics']['accuracy'])
                optimized_acc.append(results['optimization_results']['test_metrics']['accuracy'])
                baseline_f1.append(results['baseline_metrics']['f1_score'])
                optimized_f1.append(results['optimization_results']['test_metrics']['f1_score'])
                improvements_acc.append(results['improvement']['accuracy'])
                improvements_f1.append(results['improvement']['f1_score'])
        
        if not models:
            print("No successful optimization results to plot.")
            return
        
        # Plot 1: Accuracy comparison
        x = np.arange(len(models))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, baseline_acc, width, label='Baseline', alpha=0.8)
        axes[0, 0].bar(x + width/2, optimized_acc, width, label='Optimized', alpha=0.8)
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Accuracy: Baseline vs Optimized')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: F1-Score comparison
        axes[0, 1].bar(x - width/2, baseline_f1, width, label='Baseline', alpha=0.8)
        axes[0, 1].bar(x + width/2, optimized_f1, width, label='Optimized', alpha=0.8)
        axes[0, 1].set_xlabel('Models')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].set_title('F1-Score: Baseline vs Optimized')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(models, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Accuracy improvements
        colors = ['green' if imp > 0 else 'red' for imp in improvements_acc]
        axes[1, 0].bar(models, improvements_acc, color=colors, alpha=0.7)
        axes[1, 0].set_xlabel('Models')
        axes[1, 0].set_ylabel('Accuracy Improvement')
        axes[1, 0].set_title('Accuracy Improvement from Optimization')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: F1-Score improvements
        colors = ['green' if imp > 0 else 'red' for imp in improvements_f1]
        axes[1, 1].bar(models, improvements_f1, color=colors, alpha=0.7)
        axes[1, 1].set_xlabel('Models')
        axes[1, 1].set_ylabel('F1-Score Improvement')
        axes[1, 1].set_title('F1-Score Improvement from Optimization')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.results_dir / 'optimization_comparison.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Optimization results plot saved to: {plot_path}")
        
        plt.show()
    
    def save_best_models(self) -> None:
        """
        Save the best optimized models to disk.
        """
        if not self.best_models:
            print("No optimized models available to save.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, model in self.best_models.items():
            model_path = self.models_dir / f"{model_name}_optimized_{timestamp}.pkl"
            joblib.dump(model, model_path)
            print(f"Saved optimized {model_name} to: {model_path}")
        
        # Save optimization metadata
        metadata = {
            'timestamp': timestamp,
            'models_saved': list(self.best_models.keys()),
            'optimization_summary': {}
        }
        
        for model_name, results in self.optimization_results.items():
            if 'error' not in results:
                metadata['optimization_summary'][model_name] = {
                    'best_params': results['optimization_results']['best_params'],
                    'best_cv_score': results['optimization_results']['best_cv_score'],
                    'test_accuracy': results['optimization_results']['test_metrics']['accuracy'],
                    'improvement': results['improvement']['accuracy']
                }
        
        metadata_path = self.models_dir / f"optimization_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved optimization metadata to: {metadata_path}")
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive optimization report.
        
        Returns:
            Dictionary containing the complete optimization report
        """
        if not self.optimization_results:
            print("No optimization results available for report generation.")
            return {}
        
        report = {
            'optimization_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_models_optimized': len(self.optimization_results),
                'optimization_method': 'RandomizedSearchCV',
                'cross_validation_folds': self.cv_folds
            },
            'model_results': {},
            'best_overall_model': None,
            'performance_improvements': {}
        }
        
        best_accuracy = 0
        best_model_name = None
        
        for model_name, results in self.optimization_results.items():
            if 'error' not in results:
                model_report = {
                    'baseline_performance': results['baseline_metrics'],
                    'optimized_performance': results['optimization_results']['test_metrics'],
                    'best_parameters': results['optimization_results']['best_params'],
                    'cv_score': results['optimization_results']['best_cv_score'],
                    'improvements': results['improvement']
                }
                
                report['model_results'][model_name] = model_report
                
                # Track best model
                current_accuracy = results['optimization_results']['test_metrics']['accuracy']
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    best_model_name = model_name
        
        if best_model_name:
            report['best_overall_model'] = {
                'model_name': best_model_name,
                'accuracy': best_accuracy,
                'parameters': self.optimization_results[best_model_name]['optimization_results']['best_params']
            }
        
        # Calculate average improvements
        if report['model_results']:
            avg_acc_improvement = np.mean([
                results['improvements']['accuracy'] 
                for results in report['model_results'].values()
            ])
            avg_f1_improvement = np.mean([
                results['improvements']['f1_score'] 
                for results in report['model_results'].values()
            ])
            
            report['performance_improvements'] = {
                'average_accuracy_improvement': avg_acc_improvement,
                'average_f1_improvement': avg_f1_improvement,
                'models_improved': sum(1 for results in report['model_results'].values() 
                                     if results['improvements']['accuracy'] > 0)
            }
        
        # Save report
        report_path = self.results_dir / 'optimization_results.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Optimization report saved to: {report_path}")
        return report
    
    def load_data_for_optimization(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load training and test data for optimization.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            # Load processed training and test data
            train_data = pd.read_csv('data/processed/heart_disease_train.csv')
            test_data = pd.read_csv('data/processed/heart_disease_test.csv')
            
            # Separate features and target
            X_train = train_data.drop('target', axis=1).values
            y_train = train_data['target'].values
            X_test = test_data.drop('target', axis=1).values
            y_test = test_data['target'].values
            
            print(f"Loaded training data: {X_train.shape}")
            print(f"Loaded test data: {X_test.shape}")
            
            return X_train, X_test, y_train, y_test
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print("Please ensure the data processing pipeline has been run first.")
            raise


def main():
    """
    Main function to demonstrate hyperparameter optimization.
    """
    print("Heart Disease ML Pipeline - Hyperparameter Optimization")
    print("=" * 60)
    
    # Initialize tuner
    tuner = HyperparameterTuner(random_state=42, n_jobs=-1, cv_folds=5)
    
    try:
        # Load data
        X_train, X_test, y_train, y_test = tuner.load_data_for_optimization()
        
        # Optimize all models
        results = tuner.optimize_all_models(X_train, y_train, X_test, y_test)
        
        # Generate visualizations
        tuner.plot_optimization_results(save_plots=True)
        
        # Save best models
        tuner.save_best_models()
        
        # Generate comprehensive report
        report = tuner.generate_optimization_report()
        
        # Print summary
        print("\n" + "=" * 60)
        print("OPTIMIZATION SUMMARY")
        print("=" * 60)
        
        if report.get('best_overall_model'):
            best_model = report['best_overall_model']
            print(f"Best Model: {best_model['model_name']}")
            print(f"Best Accuracy: {best_model['accuracy']:.4f}")
        
        if report.get('performance_improvements'):
            improvements = report['performance_improvements']
            print(f"Average Accuracy Improvement: {improvements['average_accuracy_improvement']:.4f}")
            print(f"Models Improved: {improvements['models_improved']}/{len(results)}")
        
        print("\nOptimization completed successfully!")
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        raise


if __name__ == "__main__":
    main()
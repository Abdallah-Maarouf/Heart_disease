"""
Model Evaluation and Performance Metrics System

This module provides comprehensive model evaluation functionality including
classification metrics, ROC curves, confusion matrices, cross-validation,
and performance comparison visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, confusion_matrix, classification_report,
    average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler


class ModelEvaluator:
    """
    Comprehensive model evaluation system for heart disease classification models.
    
    This class provides functionality to evaluate multiple models, generate
    performance metrics, create visualizations, and compare model performance.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the ModelEvaluator.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'comparison': {},
            'recommendations': {}
        }
        self.X_test = None
        self.y_test = None
        self.X_train = None
        self.y_train = None
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_models_and_data(self, models_dir: str = 'models/supervised', 
                           data_path: str = 'data/processed/heart_disease_selected.csv') -> None:
        """
        Load trained models and test data for evaluation.
        
        Args:
            models_dir (str): Directory containing trained models
            data_path (str): Path to the dataset
        """
        print(f"Loading models from {models_dir}...")
        
        # Load data
        df = pd.read_csv(data_path)
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Handle scaling - create a new scaler for the selected features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Features: {list(X.columns)}")
        
        # Split data (using same split as training)
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Load models
        model_files = list(Path(models_dir).glob('*.pkl'))
        model_files = [f for f in model_files if 'scaler' not in f.name]
        
        for model_file in model_files:
            model_name = model_file.stem.split('_')[0] + '_' + model_file.stem.split('_')[1]
            if model_name not in self.models:  # Avoid duplicates
                try:
                    model = joblib.load(model_file)
                    self.models[model_name] = model
                    print(f"Loaded {model_name} from {model_file.name}")
                except Exception as e:
                    print(f"Error loading {model_file.name}: {e}")
        
        print(f"Loaded {len(self.models)} models and data with {len(self.X_test)} test samples")
        
    def calculate_classification_metrics(self, model_name: str, model: Any) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics for a model.
        
        Args:
            model_name (str): Name of the model
            model (Any): Trained model object
            
        Returns:
            Dictionary containing classification metrics
        """
        # Make predictions
        y_pred = model.predict(self.X_test)
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_pred_proba = model.decision_function(self.X_test)
        else:
            y_pred_proba = None
        
        # Calculate basic metrics
        metrics = {
            'accuracy': float(accuracy_score(self.y_test, y_pred)),
            'precision': float(precision_score(self.y_test, y_pred, average='binary', zero_division=0)),
            'recall': float(recall_score(self.y_test, y_pred, average='binary', zero_division=0)),
            'f1_score': float(f1_score(self.y_test, y_pred, average='binary', zero_division=0)),
            'specificity': float(precision_score(self.y_test, y_pred, pos_label=0, average='binary', zero_division=0))
        }
        
        # Add ROC-AUC and Average Precision if probabilities are available
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = float(roc_auc_score(self.y_test, y_pred_proba))
                metrics['average_precision'] = float(average_precision_score(self.y_test, y_pred_proba))
            except ValueError as e:
                print(f"Warning: Could not calculate ROC-AUC for {model_name}: {e}")
                metrics['roc_auc'] = 0.0
                metrics['average_precision'] = 0.0
        else:
            metrics['roc_auc'] = 0.0
            metrics['average_precision'] = 0.0
        
        # Calculate additional metrics
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        metrics.update({
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            'positive_predictive_value': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            'negative_predictive_value': float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        })
        
        return metrics
    
    def generate_roc_curves(self, save_dir: str = 'results/model_evaluation/plots') -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate ROC curves with AUC calculation for all models.
        
        Args:
            save_dir (str): Directory to save ROC curve plots
            
        Returns:
            Dictionary containing ROC curve data for each model
        """
        print("Generating ROC curves...")
        
        # Create directory if it doesn't exist
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        roc_data = {}
        
        # Create figure for combined ROC curves
        plt.figure(figsize=(12, 8))
        
        for model_name, model in self.models.items():
            try:
                # Get prediction probabilities
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_pred_proba = model.decision_function(self.X_test)
                else:
                    print(f"Warning: {model_name} does not support probability predictions")
                    continue
                
                # Calculate ROC curve
                fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
                auc_score = roc_auc_score(self.y_test, y_pred_proba)
                
                # Store ROC data
                roc_data[model_name] = {
                    'fpr': fpr,
                    'tpr': tpr,
                    'thresholds': thresholds,
                    'auc': auc_score
                }
                
                # Plot ROC curve
                plt.plot(fpr, tpr, linewidth=2, 
                        label=f'{model_name.replace("_", " ").title()} (AUC = {auc_score:.3f})')
                
            except Exception as e:
                print(f"Error generating ROC curve for {model_name}: {e}")
        
        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier (AUC = 0.500)')
        
        # Customize plot
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison - Heart Disease Classification', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Save combined ROC plot
        plt.tight_layout()
        plt.savefig(Path(save_dir) / 'roc_curves_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ROC curves saved to {save_dir}/roc_curves_comparison.png")
        return roc_data
    
    def plot_confusion_matrices(self, save_dir: str = 'results/model_evaluation/plots') -> Dict[str, np.ndarray]:
        """
        Generate confusion matrix heatmaps for all models.
        
        Args:
            save_dir (str): Directory to save confusion matrix plots
            
        Returns:
            Dictionary containing confusion matrices for each model
        """
        print("Generating confusion matrices...")
        
        # Create directory if it doesn't exist
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        confusion_matrices = {}
        
        # Calculate number of subplots needed
        n_models = len(self.models)
        n_cols = min(2, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        # Create figure for all confusion matrices
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Make predictions and calculate confusion matrix
            y_pred = model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)
            confusion_matrices[model_name] = cm
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['No Disease', 'Disease'],
                       yticklabels=['No Disease', 'Disease'])
            
            ax.set_title(f'{model_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted Label', fontsize=10)
            ax.set_ylabel('True Label', fontsize=10)
        
        # Hide empty subplots
        for idx in range(n_models, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.suptitle('Confusion Matrices - Heart Disease Classification', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(Path(save_dir) / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrices saved to {save_dir}/confusion_matrices.png")
        return confusion_matrices    

    def cross_validation_scores(self, cv_folds: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Perform robust cross-validation assessment for all models.
        
        Args:
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            Dictionary containing cross-validation scores for each model
        """
        print(f"Performing {cv_folds}-fold cross-validation...")
        
        cv_results = {}
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Combine training and test data for cross-validation
        X_full = np.vstack([self.X_train, self.X_test])
        y_full = np.hstack([self.y_train, self.y_test])
        
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        for model_name, model in self.models.items():
            print(f"Cross-validating {model_name}...")
            
            model_cv_results = {}
            
            for metric in scoring_metrics:
                try:
                    cv_scores = cross_val_score(model, X_full, y_full, cv=skf, scoring=metric)
                    model_cv_results[metric] = {
                        'mean': float(cv_scores.mean()),
                        'std': float(cv_scores.std()),
                        'scores': cv_scores.tolist()
                    }
                except Exception as e:
                    print(f"Warning: Could not calculate {metric} for {model_name}: {e}")
                    model_cv_results[metric] = {
                        'mean': 0.0,
                        'std': 0.0,
                        'scores': [0.0] * cv_folds
                    }
            
            cv_results[model_name] = model_cv_results
            print(f"{model_name} CV accuracy: {model_cv_results['accuracy']['mean']:.4f} "
                  f"(+/- {model_cv_results['accuracy']['std'] * 2:.4f})")
        
        return cv_results
    
    def compare_model_performance(self) -> pd.DataFrame:
        """
        Create side-by-side metrics comparison for all models.
        
        Returns:
            DataFrame containing performance comparison
        """
        print("Comparing model performance...")
        
        comparison_data = []
        
        for model_name, model in self.models.items():
            metrics = self.calculate_classification_metrics(model_name, model)
            
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}",
                'Specificity': f"{metrics['specificity']:.4f}",
                'Avg Precision': f"{metrics['average_precision']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Store in evaluation results
        self.evaluation_results['comparison']['performance_table'] = comparison_df.to_dict('records')
        
        return comparison_df
    
    def generate_classification_report(self, save_dir: str = 'results/model_evaluation') -> Dict[str, Dict]:
        """
        Generate detailed per-class classification reports for all models.
        
        Args:
            save_dir (str): Directory to save classification reports
            
        Returns:
            Dictionary containing classification reports for each model
        """
        print("Generating detailed classification reports...")
        
        # Create directory if it doesn't exist
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        classification_reports = {}
        
        for model_name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            
            # Generate classification report
            report = classification_report(
                self.y_test, y_pred,
                target_names=['No Disease', 'Disease'],
                output_dict=True,
                zero_division=0
            )
            
            classification_reports[model_name] = report
            
            # Save individual report as text file
            report_text = classification_report(
                self.y_test, y_pred,
                target_names=['No Disease', 'Disease'],
                zero_division=0
            )
            
            with open(Path(save_dir) / f'{model_name}_classification_report.txt', 'w') as f:
                f.write(f"Classification Report - {model_name.replace('_', ' ').title()}\n")
                f.write("=" * 60 + "\n\n")
                f.write(report_text)
        
        print(f"Classification reports saved to {save_dir}/")
        return classification_reports
    
    def plot_precision_recall_curves(self, save_dir: str = 'results/model_evaluation/plots') -> Dict[str, Dict]:
        """
        Generate precision-recall curves for additional model assessment.
        
        Args:
            save_dir (str): Directory to save precision-recall curve plots
            
        Returns:
            Dictionary containing precision-recall curve data for each model
        """
        print("Generating precision-recall curves...")
        
        # Create directory if it doesn't exist
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        pr_data = {}
        
        # Create figure for combined PR curves
        plt.figure(figsize=(12, 8))
        
        for model_name, model in self.models.items():
            try:
                # Get prediction probabilities
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_pred_proba = model.decision_function(self.X_test)
                else:
                    print(f"Warning: {model_name} does not support probability predictions")
                    continue
                
                # Calculate precision-recall curve
                precision, recall, thresholds = precision_recall_curve(self.y_test, y_pred_proba)
                avg_precision = average_precision_score(self.y_test, y_pred_proba)
                
                # Store PR data
                pr_data[model_name] = {
                    'precision': precision,
                    'recall': recall,
                    'thresholds': thresholds,
                    'average_precision': avg_precision
                }
                
                # Plot PR curve
                plt.plot(recall, precision, linewidth=2,
                        label=f'{model_name.replace("_", " ").title()} (AP = {avg_precision:.3f})')
                
            except Exception as e:
                print(f"Error generating PR curve for {model_name}: {e}")
        
        # Plot baseline (random classifier)
        baseline = len(self.y_test[self.y_test == 1]) / len(self.y_test)
        plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1,
                   label=f'Random Classifier (AP = {baseline:.3f})')
        
        # Customize plot
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves - Heart Disease Classification', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Save combined PR plot
        plt.tight_layout()
        plt.savefig(Path(save_dir) / 'precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Precision-recall curves saved to {save_dir}/precision_recall_curves.png")
        return pr_data
    
    def plot_model_comparison_metrics(self, save_dir: str = 'results/model_evaluation/plots') -> None:
        """
        Create comprehensive model comparison visualizations.
        
        Args:
            save_dir (str): Directory to save comparison plots
        """
        print("Creating model comparison visualizations...")
        
        # Create directory if it doesn't exist
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Collect metrics for all models
        metrics_data = []
        for model_name, model in self.models.items():
            metrics = self.calculate_classification_metrics(model_name, model)
            metrics['Model'] = model_name.replace('_', ' ').title()
            metrics_data.append(metrics)
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Create comparison bar plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy, Precision, Recall, F1-Score
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for idx, (metric, title) in enumerate(zip(metrics_to_plot, metric_titles)):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            bars = ax.bar(metrics_df['Model'], metrics_df[metric], alpha=0.7)
            ax.set_title(f'{title} Comparison', fontsize=12, fontweight='bold')
            ax.set_ylabel(title, fontsize=10)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            # Rotate x-axis labels if needed
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.suptitle('Model Performance Metrics Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(Path(save_dir) / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create ROC-AUC comparison plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics_df['Model'], metrics_df['roc_auc'], alpha=0.7, color='skyblue')
        plt.title('ROC-AUC Score Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('ROC-AUC Score', fontsize=12)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(Path(save_dir) / 'roc_auc_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plots saved to {save_dir}/")
    
    def model_performance_summary(self, save_dir: str = 'results/model_evaluation') -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report with recommendations.
        
        Args:
            save_dir (str): Directory to save the performance summary
            
        Returns:
            Dictionary containing comprehensive performance summary
        """
        print("Generating comprehensive model performance summary...")
        
        # Create directory if it doesn't exist
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Calculate metrics for all models
        all_metrics = {}
        for model_name, model in self.models.items():
            all_metrics[model_name] = self.calculate_classification_metrics(model_name, model)
        
        # Perform cross-validation
        cv_results = self.cross_validation_scores()
        
        # Find best performing models
        best_models = {
            'accuracy': max(all_metrics.items(), key=lambda x: x[1]['accuracy']),
            'precision': max(all_metrics.items(), key=lambda x: x[1]['precision']),
            'recall': max(all_metrics.items(), key=lambda x: x[1]['recall']),
            'f1_score': max(all_metrics.items(), key=lambda x: x[1]['f1_score']),
            'roc_auc': max(all_metrics.items(), key=lambda x: x[1]['roc_auc'])
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_metrics, best_models)
        
        # Create comprehensive summary
        summary = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'test_samples': int(len(self.y_test)),
                'positive_class_ratio': float(self.y_test.sum() / len(self.y_test)),
                'class_distribution': {
                    'no_disease': int((self.y_test == 0).sum()),
                    'disease': int((self.y_test == 1).sum())
                }
            },
            'model_metrics': all_metrics,
            'cross_validation_results': cv_results,
            'best_performing_models': {
                metric: {'model': name, 'score': score[metric]}
                for metric, (name, score) in best_models.items()
            },
            'recommendations': recommendations
        }
        
        # Store in evaluation results
        self.evaluation_results.update(summary)
        
        # Create text summary report
        self._create_text_summary_report(summary, save_dir)
        
        return summary
    
    def _generate_recommendations(self, all_metrics: Dict[str, Dict], 
                                best_models: Dict[str, Tuple[str, Dict]]) -> Dict[str, Any]:
        """
        Generate model recommendations based on performance analysis.
        
        Args:
            all_metrics (Dict): All model metrics
            best_models (Dict): Best performing models for each metric
            
        Returns:
            Dictionary containing recommendations
        """
        recommendations = {
            'overall_best_model': None,
            'use_case_recommendations': {},
            'performance_insights': [],
            'improvement_suggestions': []
        }
        
        # Calculate overall score (weighted average of key metrics)
        weights = {'accuracy': 0.25, 'precision': 0.25, 'recall': 0.25, 'f1_score': 0.25}
        overall_scores = {}
        
        for model_name, metrics in all_metrics.items():
            overall_score = sum(weights[metric] * metrics[metric] for metric in weights.keys())
            overall_scores[model_name] = overall_score
        
        best_overall = max(overall_scores.items(), key=lambda x: x[1])
        recommendations['overall_best_model'] = {
            'model': best_overall[0],
            'score': best_overall[1]
        }
        
        # Use case specific recommendations
        recommendations['use_case_recommendations'] = {
            'high_precision_needed': {
                'model': best_models['precision'][0],
                'reason': 'Best for minimizing false positives (avoiding unnecessary treatments)'
            },
            'high_recall_needed': {
                'model': best_models['recall'][0],
                'reason': 'Best for minimizing false negatives (catching all disease cases)'
            },
            'balanced_performance': {
                'model': best_models['f1_score'][0],
                'reason': 'Best balance between precision and recall'
            },
            'probability_ranking': {
                'model': best_models['roc_auc'][0],
                'reason': 'Best for ranking patients by disease probability'
            }
        }
        
        # Performance insights
        avg_accuracy = np.mean([metrics['accuracy'] for metrics in all_metrics.values()])
        if avg_accuracy > 0.85:
            recommendations['performance_insights'].append("Excellent overall model performance (>85% accuracy)")
        elif avg_accuracy > 0.75:
            recommendations['performance_insights'].append("Good overall model performance (>75% accuracy)")
        else:
            recommendations['performance_insights'].append("Model performance needs improvement (<75% accuracy)")
        
        # Check for class imbalance issues
        precision_recall_diff = abs(
            np.mean([metrics['precision'] for metrics in all_metrics.values()]) -
            np.mean([metrics['recall'] for metrics in all_metrics.values()])
        )
        
        if precision_recall_diff > 0.1:
            recommendations['performance_insights'].append(
                "Significant precision-recall imbalance detected - consider class balancing techniques"
            )
        
        # Improvement suggestions
        if avg_accuracy < 0.9:
            recommendations['improvement_suggestions'].extend([
                "Consider hyperparameter tuning for better performance",
                "Explore feature engineering and selection techniques",
                "Try ensemble methods to combine model strengths"
            ])
        
        return recommendations
    
    def _create_text_summary_report(self, summary: Dict[str, Any], save_dir: str) -> None:
        """
        Create a human-readable text summary report.
        
        Args:
            summary (Dict): Summary data
            save_dir (str): Directory to save the report
        """
        report_path = Path(save_dir) / 'model_evaluation_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("HEART DISEASE CLASSIFICATION - MODEL EVALUATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Evaluation Date: {summary['evaluation_timestamp']}\n")
            f.write(f"Test Samples: {summary['dataset_info']['test_samples']}\n")
            f.write(f"Positive Class Ratio: {summary['dataset_info']['positive_class_ratio']:.3f}\n\n")
            
            f.write("MODEL PERFORMANCE COMPARISON\n")
            f.write("-" * 30 + "\n")
            
            for model_name, metrics in summary['model_metrics'].items():
                f.write(f"\n{model_name.replace('_', ' ').title()}:\n")
                f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall:    {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score:  {metrics['f1_score']:.4f}\n")
                f.write(f"  ROC-AUC:   {metrics['roc_auc']:.4f}\n")
            
            f.write(f"\n\nBEST PERFORMING MODELS\n")
            f.write("-" * 25 + "\n")
            
            for metric, info in summary['best_performing_models'].items():
                f.write(f"{metric.upper()}: {info['model'].replace('_', ' ').title()} "
                       f"({info['score']:.4f})\n")
            
            f.write(f"\n\nRECOMMENDations\n")
            f.write("-" * 15 + "\n")
            
            f.write(f"Overall Best Model: {summary['recommendations']['overall_best_model']['model'].replace('_', ' ').title()}\n\n")
            
            f.write("Use Case Recommendations:\n")
            for use_case, rec in summary['recommendations']['use_case_recommendations'].items():
                f.write(f"  {use_case.replace('_', ' ').title()}: {rec['model'].replace('_', ' ').title()}\n")
                f.write(f"    Reason: {rec['reason']}\n")
            
            f.write(f"\nPerformance Insights:\n")
            for insight in summary['recommendations']['performance_insights']:
                f.write(f"  • {insight}\n")
            
            if summary['recommendations']['improvement_suggestions']:
                f.write(f"\nImprovement Suggestions:\n")
                for suggestion in summary['recommendations']['improvement_suggestions']:
                    f.write(f"  • {suggestion}\n")
        
        print(f"Summary report saved to {report_path}")
    
    def save_evaluation_results(self, filepath: str = 'results/model_evaluation/evaluation_metrics.json') -> None:
        """
        Save all evaluation results to JSON file.
        
        Args:
            filepath (str): Path to save the evaluation results
        """
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(self.evaluation_results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Evaluation results saved to {filepath}")
    
    def evaluate_all_models(self) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline for all models.
        
        Returns:
            Dictionary containing all evaluation results
        """
        print("Starting comprehensive model evaluation...")
        
        # Generate all evaluations
        roc_data = self.generate_roc_curves()
        confusion_matrices = self.plot_confusion_matrices()
        cv_results = self.cross_validation_scores()
        comparison_df = self.compare_model_performance()
        classification_reports = self.generate_classification_report()
        pr_data = self.plot_precision_recall_curves()
        self.plot_model_comparison_metrics()
        summary = self.model_performance_summary()
        
        # Save results
        self.save_evaluation_results()
        
        print("Model evaluation completed successfully!")
        return self.evaluation_results


def main():
    """
    Main function to demonstrate the ModelEvaluator usage.
    """
    # Initialize evaluator
    evaluator = ModelEvaluator(random_state=42)
    
    # Load models and data
    evaluator.load_models_and_data()
    
    # Run complete evaluation
    results = evaluator.evaluate_all_models()
    
    # Display summary
    print("\n" + "="*60)
    print("MODEL EVALUATION SUMMARY")
    print("="*60)
    
    comparison_df = evaluator.compare_model_performance()
    print(comparison_df.to_string(index=False))
    
    print(f"\nBest Overall Model: {results['recommendations']['overall_best_model']['model']}")
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
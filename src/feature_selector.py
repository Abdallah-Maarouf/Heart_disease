"""
Comprehensive feature selection module for heart disease dataset.

This module provides the FeatureSelector class with multiple feature selection
techniques including Random Forest importance, RFE, Chi-Square tests, correlation
analysis, and univariate selection with comparison and ranking capabilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import json
import logging
from sklearn.feature_selection import (
    SelectKBest, f_classif, chi2, RFE, RFECV,
    mutual_info_classif, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

from utils import setup_logging, ensure_directory_exists


class FeatureSelector:
    """
    Comprehensive feature selection system with multiple techniques.
    
    This class implements various feature selection methods including:
    - Random Forest feature importance
    - Recursive Feature Elimination (RFE)
    - Chi-Square statistical tests
    - Correlation-based selection
    - Univariate statistical tests
    - Combined selection methods with ranking
    """
    
    def __init__(self, log_level: str = "INFO"):
        """
        Initialize the FeatureSelector.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.logger = setup_logging(__name__, log_level)
        self.feature_rankings = {}
        self.selection_results = {}
        self.feature_names = None
        
    def random_forest_importance(self, X: pd.DataFrame, y: pd.Series, 
                                n_estimators: int = 100, random_state: int = 42) -> Dict[str, Any]:
        """
        Rank features using Random Forest feature importance.
        
        Args:
            X: Feature matrix
            y: Target vector
            n_estimators: Number of trees in the forest
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary containing feature importance rankings and scores
        """
        self.logger.info("Starting Random Forest feature importance analysis")
        
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        feature_names = X.columns.tolist()
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Calculate standard deviations from tree importances
        std_importances = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
        importance_df['std'] = std_importances[importance_df.index]
        
        # Rank features
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        result = {
            'method': 'random_forest',
            'feature_rankings': importance_df,
            'model': rf,
            'top_features': importance_df.head(10)['feature'].tolist(),
            'importance_threshold': np.mean(importances)
        }
        
        self.feature_rankings['random_forest'] = importance_df
        self.logger.info(f"Random Forest analysis complete. Top feature: {importance_df.iloc[0]['feature']}")
        
        return result
    
    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series,
                                    estimator=None, n_features_to_select: Optional[int] = None,
                                    step: int = 1, cv: int = 5) -> Dict[str, Any]:
        """
        Perform Recursive Feature Elimination with cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            estimator: Base estimator for RFE (default: LogisticRegression)
            n_features_to_select: Number of features to select (None for optimal)
            step: Number of features to remove at each iteration
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary containing RFE results and selected features
        """
        self.logger.info("Starting Recursive Feature Elimination with CV")
        
        if estimator is None:
            estimator = LogisticRegression(random_state=42, max_iter=1000)
        
        feature_names = X.columns.tolist()
        
        if n_features_to_select is None:
            # Use RFECV to find optimal number of features
            selector = RFECV(
                estimator=estimator,
                step=step,
                cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
                scoring='accuracy',
                n_jobs=-1
            )
        else:
            # Use RFE with specified number of features
            selector = RFE(
                estimator=estimator,
                n_features_to_select=n_features_to_select,
                step=step
            )
        
        # Fit the selector
        selector.fit(X, y)
        
        # Create results DataFrame
        rfe_df = pd.DataFrame({
            'feature': feature_names,
            'selected': selector.support_,
            'ranking': selector.ranking_
        }).sort_values('ranking')
        
        # Get selected features
        selected_features = rfe_df[rfe_df['selected']]['feature'].tolist()
        
        result = {
            'method': 'rfe',
            'feature_rankings': rfe_df,
            'selector': selector,
            'selected_features': selected_features,
            'n_features_selected': len(selected_features),
            'optimal_score': getattr(selector, 'grid_scores_', None)
        }
        
        if hasattr(selector, 'grid_scores_'):
            result['cv_scores'] = selector.grid_scores_
            result['optimal_n_features'] = selector.n_features_
        
        self.feature_rankings['rfe'] = rfe_df
        self.logger.info(f"RFE complete. Selected {len(selected_features)} features")
        
        return result
    
    def chi_square_selection(self, X: pd.DataFrame, y: pd.Series, 
                           k: int = 10) -> Dict[str, Any]:
        """
        Select features using Chi-Square statistical test.
        
        Args:
            X: Feature matrix (should be non-negative for chi2)
            y: Target vector
            k: Number of top features to select
            
        Returns:
            Dictionary containing Chi-Square test results
        """
        self.logger.info("Starting Chi-Square feature selection")
        
        # Ensure non-negative values for chi2 test
        X_positive = X.copy()
        
        # Scale to positive values if needed
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Shift to ensure all values are positive
        X_positive = X_scaled - X_scaled.min() + 1e-8
        X_positive = pd.DataFrame(X_positive, columns=X.columns, index=X.index)
        
        # Perform Chi-Square test
        selector = SelectKBest(score_func=chi2, k=k)
        X_selected = selector.fit_transform(X_positive, y)
        
        # Get scores and p-values
        scores = selector.scores_
        pvalues = selector.pvalues_
        feature_names = X.columns.tolist()
        
        # Create results DataFrame
        chi2_df = pd.DataFrame({
            'feature': feature_names,
            'chi2_score': scores,
            'p_value': pvalues,
            'selected': selector.get_support()
        }).sort_values('chi2_score', ascending=False)
        
        chi2_df['rank'] = range(1, len(chi2_df) + 1)
        
        selected_features = chi2_df[chi2_df['selected']]['feature'].tolist()
        
        result = {
            'method': 'chi_square',
            'feature_rankings': chi2_df,
            'selector': selector,
            'selected_features': selected_features,
            'k': k,
            'scaler': scaler
        }
        
        self.feature_rankings['chi_square'] = chi2_df
        self.logger.info(f"Chi-Square selection complete. Selected {len(selected_features)} features")
        
        return result
    
    def correlation_based_selection(self, X: pd.DataFrame, 
                                  threshold: float = 0.95) -> Dict[str, Any]:
        """
        Remove highly correlated features to reduce multicollinearity.
        
        Args:
            X: Feature matrix
            threshold: Correlation threshold for feature removal
            
        Returns:
            Dictionary containing correlation analysis results
        """
        self.logger.info(f"Starting correlation-based selection with threshold {threshold}")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        features_to_remove = set()
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    feature1 = corr_matrix.columns[i]
                    feature2 = corr_matrix.columns[j]
                    high_corr_pairs.append((feature1, feature2, corr_matrix.iloc[i, j]))
                    
                    # Remove the feature with lower variance (less informative)
                    if X[feature1].var() < X[feature2].var():
                        features_to_remove.add(feature1)
                    else:
                        features_to_remove.add(feature2)
        
        # Create results DataFrame
        remaining_features = [col for col in X.columns if col not in features_to_remove]
        
        corr_df = pd.DataFrame({
            'feature': X.columns,
            'removed': [col in features_to_remove for col in X.columns],
            'max_correlation': [corr_matrix[col].drop(col).max() for col in X.columns]
        }).sort_values('max_correlation', ascending=False)
        
        result = {
            'method': 'correlation',
            'feature_rankings': corr_df,
            'correlation_matrix': corr_matrix,
            'high_corr_pairs': high_corr_pairs,
            'features_to_remove': list(features_to_remove),
            'remaining_features': remaining_features,
            'threshold': threshold,
            'n_removed': len(features_to_remove)
        }
        
        self.feature_rankings['correlation'] = corr_df
        self.logger.info(f"Correlation analysis complete. Removed {len(features_to_remove)} features")
        
        return result
    
    def univariate_selection(self, X: pd.DataFrame, y: pd.Series,
                           score_func=f_classif, k: int = 10) -> Dict[str, Any]:
        """
        Select features using univariate statistical tests.
        
        Args:
            X: Feature matrix
            y: Target vector
            score_func: Scoring function (f_classif, mutual_info_classif, etc.)
            k: Number of top features to select
            
        Returns:
            Dictionary containing univariate selection results
        """
        self.logger.info(f"Starting univariate selection with {score_func.__name__}")
        
        # Perform univariate selection
        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get scores
        scores = selector.scores_
        feature_names = X.columns.tolist()
        
        # Handle p-values if available
        pvalues = getattr(selector, 'pvalues_', [np.nan] * len(feature_names))
        
        # Create results DataFrame
        univariate_df = pd.DataFrame({
            'feature': feature_names,
            'score': scores,
            'p_value': pvalues,
            'selected': selector.get_support()
        }).sort_values('score', ascending=False)
        
        univariate_df['rank'] = range(1, len(univariate_df) + 1)
        
        selected_features = univariate_df[univariate_df['selected']]['feature'].tolist()
        
        result = {
            'method': f'univariate_{score_func.__name__}',
            'feature_rankings': univariate_df,
            'selector': selector,
            'selected_features': selected_features,
            'score_function': score_func.__name__,
            'k': k
        }
        
        self.feature_rankings[f'univariate_{score_func.__name__}'] = univariate_df
        self.logger.info(f"Univariate selection complete. Selected {len(selected_features)} features")
        
        return result
    
    def mutual_information_selection(self, X: pd.DataFrame, y: pd.Series,
                                   k: int = 10, random_state: int = 42) -> Dict[str, Any]:
        """
        Select features using mutual information.
        
        Args:
            X: Feature matrix
            y: Target vector
            k: Number of top features to select
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary containing mutual information results
        """
        self.logger.info("Starting mutual information feature selection")
        
        # Calculate mutual information scores
        mi_scores = mutual_info_classif(X, y, random_state=random_state)
        feature_names = X.columns.tolist()
        
        # Create results DataFrame
        mi_df = pd.DataFrame({
            'feature': feature_names,
            'mi_score': mi_scores,
            'selected': False
        }).sort_values('mi_score', ascending=False)
        
        # Select top k features
        mi_df.iloc[:k, mi_df.columns.get_loc('selected')] = True
        mi_df['rank'] = range(1, len(mi_df) + 1)
        
        selected_features = mi_df[mi_df['selected']]['feature'].tolist()
        
        result = {
            'method': 'mutual_information',
            'feature_rankings': mi_df,
            'selected_features': selected_features,
            'k': k
        }
        
        self.feature_rankings['mutual_information'] = mi_df
        self.logger.info(f"Mutual information selection complete. Selected {len(selected_features)} features")
        
        return result
    
    def compare_selection_methods(self, X: pd.DataFrame, y: pd.Series,
                                methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare different feature selection methods and their performance.
        
        Args:
            X: Feature matrix
            y: Target vector
            methods: List of methods to compare (None for all available)
            
        Returns:
            Dictionary containing comparison results and performance metrics
        """
        self.logger.info("Starting feature selection methods comparison")
        
        if methods is None:
            methods = ['random_forest', 'rfe', 'chi_square', 'univariate', 'mutual_information']
        
        comparison_results = {}
        performance_results = {}
        
        # Store original feature names
        self.feature_names = X.columns.tolist()
        
        # Run each method
        for method in methods:
            try:
                if method == 'random_forest':
                    result = self.random_forest_importance(X, y)
                elif method == 'rfe':
                    result = self.recursive_feature_elimination(X, y)
                elif method == 'chi_square':
                    result = self.chi_square_selection(X, y, k=8)
                elif method == 'univariate':
                    result = self.univariate_selection(X, y, k=8)
                elif method == 'mutual_information':
                    result = self.mutual_information_selection(X, y, k=8)
                elif method == 'correlation':
                    result = self.correlation_based_selection(X)
                else:
                    self.logger.warning(f"Unknown method: {method}")
                    continue
                
                comparison_results[method] = result
                
                # Evaluate performance if features are selected
                if 'selected_features' in result and result['selected_features']:
                    performance = self._evaluate_feature_subset(
                        X, y, result['selected_features'], method
                    )
                    performance_results[method] = performance
                
            except Exception as e:
                self.logger.error(f"Error in {method}: {e}")
                continue
        
        # Create summary comparison
        summary = self._create_comparison_summary(comparison_results, performance_results)
        
        result = {
            'methods_results': comparison_results,
            'performance_comparison': performance_results,
            'summary': summary,
            'feature_rankings': self.feature_rankings
        }
        
        self.selection_results = result
        self.logger.info(f"Comparison complete for {len(comparison_results)} methods")
        
        return result
    
    def _evaluate_feature_subset(self, X: pd.DataFrame, y: pd.Series,
                               selected_features: List[str], method_name: str) -> Dict[str, float]:
        """
        Evaluate the performance of a feature subset using cross-validation.
        
        Args:
            X: Full feature matrix
            y: Target vector
            selected_features: List of selected feature names
            method_name: Name of the selection method
            
        Returns:
            Dictionary containing performance metrics
        """
        # Select subset of features
        X_subset = X[selected_features]
        
        # Use Logistic Regression for evaluation
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            model, X_subset, y, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy'
        )
        
        # Calculate additional metrics
        model.fit(X_subset, y)
        y_pred = model.predict(X_subset)
        train_accuracy = accuracy_score(y, y_pred)
        
        performance = {
            'n_features': len(selected_features),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_accuracy': train_accuracy,
            'feature_ratio': len(selected_features) / len(X.columns)
        }
        
        return performance
    
    def _create_comparison_summary(self, comparison_results: Dict[str, Any],
                                 performance_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Create a summary of all feature selection methods.
        
        Args:
            comparison_results: Results from different selection methods
            performance_results: Performance metrics for each method
            
        Returns:
            Dictionary containing summary statistics and rankings
        """
        summary_data = []
        
        for method, perf in performance_results.items():
            summary_data.append({
                'method': method,
                'n_features': perf['n_features'],
                'cv_accuracy': perf['cv_mean'],
                'cv_std': perf['cv_std'],
                'train_accuracy': perf['train_accuracy'],
                'feature_ratio': perf['feature_ratio']
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        if not summary_df.empty:
            # Rank methods by CV accuracy
            summary_df = summary_df.sort_values('cv_accuracy', ascending=False)
            summary_df['rank'] = range(1, len(summary_df) + 1)
        
        # Find most frequently selected features across methods
        feature_frequency = {}
        for method, result in comparison_results.items():
            if 'selected_features' in result:
                for feature in result['selected_features']:
                    feature_frequency[feature] = feature_frequency.get(feature, 0) + 1
        
        # Sort features by selection frequency
        frequent_features = sorted(
            feature_frequency.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        summary = {
            'method_comparison': summary_df,
            'best_method': summary_df.iloc[0]['method'] if not summary_df.empty else None,
            'feature_frequency': feature_frequency,
            'most_frequent_features': frequent_features[:10],
            'total_methods': len(comparison_results),
            'avg_features_selected': np.mean([perf['n_features'] for perf in performance_results.values()]) if performance_results else 0
        }
        
        return summary
    
    def select_best_features(self, X: pd.DataFrame, y: pd.Series,
                           strategy: str = 'ensemble', n_features: Optional[int] = None) -> Dict[str, Any]:
        """
        Select the best features using a combined approach from multiple methods.
        
        Args:
            X: Feature matrix
            y: Target vector
            strategy: Selection strategy ('ensemble', 'best_method', 'intersection', 'union')
            n_features: Target number of features (None for automatic selection)
            
        Returns:
            Dictionary containing final selected features and methodology
        """
        self.logger.info(f"Selecting best features using {strategy} strategy")
        
        # Run comparison if not already done
        if not self.selection_results:
            self.compare_selection_methods(X, y)
        
        if strategy == 'ensemble':
            selected_features = self._ensemble_selection(n_features)
        elif strategy == 'best_method':
            selected_features = self._best_method_selection()
        elif strategy == 'intersection':
            selected_features = self._intersection_selection()
        elif strategy == 'union':
            selected_features = self._union_selection(n_features)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Evaluate final selection
        if selected_features:
            final_performance = self._evaluate_feature_subset(X, y, selected_features, f'final_{strategy}')
        else:
            final_performance = {}
        
        result = {
            'strategy': strategy,
            'selected_features': selected_features,
            'n_features': len(selected_features),
            'performance': final_performance,
            'feature_names': selected_features
        }
        
        self.logger.info(f"Final selection: {len(selected_features)} features using {strategy} strategy")
        return result
    
    def _ensemble_selection(self, n_features: Optional[int] = None) -> List[str]:
        """Select features based on ensemble voting across methods."""
        feature_votes = {}
        
        # Count votes from each method
        for method, result in self.selection_results['methods_results'].items():
            if 'selected_features' in result:
                for feature in result['selected_features']:
                    feature_votes[feature] = feature_votes.get(feature, 0) + 1
        
        # Sort by votes
        sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        
        if n_features is None:
            # Select features with at least 2 votes
            selected = [feature for feature, votes in sorted_features if votes >= 2]
            if not selected:  # Fallback to top voted features
                selected = [feature for feature, votes in sorted_features[:8]]
        else:
            selected = [feature for feature, votes in sorted_features[:n_features]]
        
        return selected
    
    def _best_method_selection(self) -> List[str]:
        """Select features from the best performing method."""
        if not self.selection_results['performance_comparison']:
            return []
        
        best_method = max(
            self.selection_results['performance_comparison'].items(),
            key=lambda x: x[1]['cv_mean']
        )[0]
        
        return self.selection_results['methods_results'][best_method].get('selected_features', [])
    
    def _intersection_selection(self) -> List[str]:
        """Select features that appear in all methods."""
        all_selections = []
        for method, result in self.selection_results['methods_results'].items():
            if 'selected_features' in result:
                all_selections.append(set(result['selected_features']))
        
        if not all_selections:
            return []
        
        # Find intersection
        intersection = all_selections[0]
        for selection in all_selections[1:]:
            intersection = intersection.intersection(selection)
        
        return list(intersection)
    
    def _union_selection(self, n_features: Optional[int] = None) -> List[str]:
        """Select features that appear in any method."""
        all_features = set()
        for method, result in self.selection_results['methods_results'].items():
            if 'selected_features' in result:
                all_features.update(result['selected_features'])
        
        features_list = list(all_features)
        
        if n_features is not None and len(features_list) > n_features:
            # Prioritize by frequency across methods
            feature_frequency = self.selection_results['summary']['feature_frequency']
            features_list = sorted(features_list, key=lambda x: feature_frequency.get(x, 0), reverse=True)
            features_list = features_list[:n_features]
        
        return features_list
    
    def plot_feature_importance(self, method: str = 'random_forest', 
                              top_n: int = 15, figsize: Tuple[int, int] = (12, 8),
                              save_path: Optional[str] = None) -> None:
        """
        Plot feature importance rankings for a specific method.
        
        Args:
            method: Feature selection method to plot
            top_n: Number of top features to display
            figsize: Figure size (width, height)
            save_path: Path to save the plot (optional)
        """
        if method not in self.feature_rankings:
            self.logger.error(f"Method {method} not found in rankings")
            return
        
        df = self.feature_rankings[method].head(top_n)
        
        plt.figure(figsize=figsize)
        
        if method == 'random_forest':
            plt.barh(range(len(df)), df['importance'], color='skyblue', alpha=0.7)
            plt.xlabel('Feature Importance')
            plt.title(f'Random Forest Feature Importance (Top {top_n})')
            
        elif method == 'chi_square':
            plt.barh(range(len(df)), df['chi2_score'], color='lightcoral', alpha=0.7)
            plt.xlabel('Chi-Square Score')
            plt.title(f'Chi-Square Feature Scores (Top {top_n})')
            
        elif method.startswith('univariate'):
            plt.barh(range(len(df)), df['score'], color='lightgreen', alpha=0.7)
            plt.xlabel('Univariate Score')
            plt.title(f'Univariate Feature Scores (Top {top_n})')
            
        elif method == 'mutual_information':
            plt.barh(range(len(df)), df['mi_score'], color='gold', alpha=0.7)
            plt.xlabel('Mutual Information Score')
            plt.title(f'Mutual Information Scores (Top {top_n})')
            
        elif method == 'correlation':
            plt.barh(range(len(df)), df['max_correlation'], color='plum', alpha=0.7)
            plt.xlabel('Maximum Correlation')
            plt.title(f'Feature Correlation Analysis (Top {top_n})')
        
        plt.yticks(range(len(df)), df['feature'])
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_method_comparison(self, figsize: Tuple[int, int] = (14, 10),
                             save_path: Optional[str] = None) -> None:
        """
        Create comprehensive comparison plots for all feature selection methods.
        
        Args:
            figsize: Figure size (width, height)
            save_path: Path to save the plot (optional)
        """
        if not self.selection_results:
            self.logger.error("No selection results available. Run compare_selection_methods first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Method Performance Comparison
        perf_data = self.selection_results['performance_comparison']
        if perf_data:
            methods = list(perf_data.keys())
            cv_scores = [perf_data[m]['cv_mean'] for m in methods]
            cv_stds = [perf_data[m]['cv_std'] for m in methods]
            
            axes[0, 0].bar(methods, cv_scores, yerr=cv_stds, capsize=5, alpha=0.7, color='skyblue')
            axes[0, 0].set_title('Cross-Validation Accuracy by Method')
            axes[0, 0].set_ylabel('CV Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Plot 2: Number of Features Selected
        if perf_data:
            n_features = [perf_data[m]['n_features'] for m in methods]
            axes[0, 1].bar(methods, n_features, alpha=0.7, color='lightcoral')
            axes[0, 1].set_title('Number of Features Selected')
            axes[0, 1].set_ylabel('Number of Features')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Plot 3: Feature Selection Frequency
        feature_freq = self.selection_results['summary']['feature_frequency']
        if feature_freq:
            top_features = sorted(feature_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            features, frequencies = zip(*top_features)
            
            axes[1, 0].barh(range(len(features)), frequencies, alpha=0.7, color='lightgreen')
            axes[1, 0].set_yticks(range(len(features)))
            axes[1, 0].set_yticklabels(features)
            axes[1, 0].set_title('Feature Selection Frequency Across Methods')
            axes[1, 0].set_xlabel('Selection Frequency')
            axes[1, 0].grid(axis='x', alpha=0.3)
        
        # Plot 4: Performance vs Number of Features
        if perf_data:
            n_features = [perf_data[m]['n_features'] for m in methods]
            cv_scores = [perf_data[m]['cv_mean'] for m in methods]
            
            axes[1, 1].scatter(n_features, cv_scores, alpha=0.7, s=100, color='gold')
            for i, method in enumerate(methods):
                axes[1, 1].annotate(method, (n_features[i], cv_scores[i]), 
                                  xytext=(5, 5), textcoords='offset points', fontsize=8)
            axes[1, 1].set_title('Performance vs Number of Features')
            axes[1, 1].set_xlabel('Number of Features')
            axes[1, 1].set_ylabel('CV Accuracy')
            axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def save_results(self, output_dir: str = "results/feature_selection") -> Dict[str, str]:
        """
        Save feature selection results to files.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Dictionary with paths to saved files
        """
        output_path = Path(output_dir)
        ensure_directory_exists(output_path)
        
        saved_files = {}
        
        # Save feature rankings for each method
        for method, rankings in self.feature_rankings.items():
            rankings_path = output_path / f"{method}_rankings.csv"
            rankings.to_csv(rankings_path, index=False)
            saved_files[f'{method}_rankings'] = str(rankings_path)
        
        # Save comparison results
        if self.selection_results:
            # Save performance comparison
            if 'performance_comparison' in self.selection_results:
                perf_df = pd.DataFrame(self.selection_results['performance_comparison']).T
                perf_path = output_path / "performance_comparison.csv"
                perf_df.to_csv(perf_path)
                saved_files['performance_comparison'] = str(perf_path)
            
            # Save summary
            if 'summary' in self.selection_results:
                summary_path = output_path / "selection_summary.json"
                summary_data = self.selection_results['summary'].copy()
                
                # Convert DataFrames to dictionaries for JSON serialization
                if 'method_comparison' in summary_data and isinstance(summary_data['method_comparison'], pd.DataFrame):
                    summary_data['method_comparison'] = summary_data['method_comparison'].to_dict('records')
                
                with open(summary_path, 'w') as f:
                    json.dump(summary_data, f, indent=2, default=str)
                saved_files['summary'] = str(summary_path)
        
        self.logger.info(f"Saved feature selection results to {len(saved_files)} files")
        return saved_files
    
    def create_selected_dataset(self, X: pd.DataFrame, y: pd.Series,
                              selected_features: List[str],
                              output_path: str = "data/processed/heart_disease_selected.csv") -> str:
        """
        Create a new dataset with only the selected features.
        
        Args:
            X: Original feature matrix
            y: Target vector
            selected_features: List of selected feature names
            output_path: Path to save the selected dataset
            
        Returns:
            Path to the saved dataset
        """
        # Create dataset with selected features
        X_selected = X[selected_features].copy()
        X_selected['target'] = y
        
        # Ensure output directory exists
        output_file = Path(output_path)
        ensure_directory_exists(output_file.parent)
        
        # Save the dataset
        X_selected.to_csv(output_path, index=False)
        
        self.logger.info(f"Created selected dataset with {len(selected_features)} features")
        self.logger.info(f"Selected features: {selected_features}")
        self.logger.info(f"Dataset saved to: {output_path}")
        
        return output_path
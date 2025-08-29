"""
Principal Component Analysis (PCA) module for dimensionality reduction and variance analysis.

This module provides comprehensive PCA functionality including:
- PCA fitting with configurable components
- Explained variance analysis
- Optimal component selection
- Data transformation
- Visualization of PCA results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class PCAAnalyzer:
    """
    A comprehensive PCA analysis class for dimensionality reduction and variance analysis.
    """
    
    def __init__(self):
        """Initialize the PCA analyzer."""
        self.pca_model = None
        self.scaler = None
        self.feature_names = None
        self.explained_variance_ratio_ = None
        self.cumulative_variance_ = None
        self.n_components_optimal_ = None
        
    def fit_pca(self, X: np.ndarray, n_components: Optional[int] = None, 
                feature_names: Optional[list] = None) -> PCA:
        """
        Fit PCA model to the data with configurable number of components.
        
        Args:
            X: Input feature matrix
            n_components: Number of components to keep (if None, keeps all)
            feature_names: List of feature names for reference
            
        Returns:
            Fitted PCA model
        """
        # Store feature names for later use
        self.feature_names = feature_names if feature_names else [f'feature_{i}' for i in range(X.shape[1])]
        
        # Standardize the data before PCA
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine number of components
        if n_components is None:
            n_components = min(X.shape[0], X.shape[1])
        
        # Fit PCA model
        self.pca_model = PCA(n_components=n_components, random_state=42)
        self.pca_model.fit(X_scaled)
        
        # Store variance information
        self.explained_variance_ratio_ = self.pca_model.explained_variance_ratio_
        self.cumulative_variance_ = np.cumsum(self.explained_variance_ratio_)
        
        print(f"PCA fitted with {n_components} components")
        print(f"Total explained variance: {self.cumulative_variance_[-1]:.4f}")
        
        return self.pca_model
    
    def calculate_explained_variance(self) -> Dict[str, np.ndarray]:
        """
        Calculate and return explained variance ratios and cumulative variance.
        
        Returns:
            Dictionary containing variance information
        """
        if self.pca_model is None:
            raise ValueError("PCA model must be fitted first. Call fit_pca() method.")
        
        variance_info = {
            'explained_variance_ratio': self.explained_variance_ratio_,
            'cumulative_variance': self.cumulative_variance_,
            'explained_variance': self.pca_model.explained_variance_,
            'singular_values': self.pca_model.singular_values_
        }
        
        return variance_info
    
    def find_optimal_components(self, variance_threshold: float = 0.95) -> int:
        """
        Find optimal number of components using variance threshold.
        
        Args:
            variance_threshold: Minimum cumulative variance to retain (default: 0.95)
            
        Returns:
            Optimal number of components
        """
        if self.cumulative_variance_ is None:
            raise ValueError("PCA model must be fitted first. Call fit_pca() method.")
        
        # Find the number of components that explain the desired variance
        optimal_components = np.argmax(self.cumulative_variance_ >= variance_threshold) + 1
        self.n_components_optimal_ = optimal_components
        
        print(f"Optimal number of components for {variance_threshold*100}% variance: {optimal_components}")
        print(f"Actual variance explained: {self.cumulative_variance_[optimal_components-1]:.4f}")
        
        return optimal_components
    
    def transform_data(self, X: np.ndarray, n_components: Optional[int] = None) -> np.ndarray:
        """
        Transform data using fitted PCA model.
        
        Args:
            X: Input data to transform
            n_components: Number of components to use (if None, uses all fitted components)
            
        Returns:
            PCA-transformed data
        """
        if self.pca_model is None or self.scaler is None:
            raise ValueError("PCA model must be fitted first. Call fit_pca() method.")
        
        # Scale the data using fitted scaler
        X_scaled = self.scaler.transform(X)
        
        # Transform using PCA
        X_pca = self.pca_model.transform(X_scaled)
        
        # Select specific number of components if specified
        if n_components is not None:
            X_pca = X_pca[:, :n_components]
        
        return X_pca
    
    def plot_explained_variance(self, save_path: Optional[str] = None, 
                              figsize: Tuple[int, int] = (12, 5)) -> None:
        """
        Plot explained variance ratio for each component.
        
        Args:
            save_path: Path to save the plot (optional)
            figsize: Figure size tuple
        """
        if self.explained_variance_ratio_ is None:
            raise ValueError("PCA model must be fitted first. Call fit_pca() method.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot individual explained variance
        components = range(1, len(self.explained_variance_ratio_) + 1)
        ax1.bar(components, self.explained_variance_ratio_, alpha=0.7, color='steelblue')
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Explained Variance by Component')
        ax1.grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        for i, v in enumerate(self.explained_variance_ratio_):
            ax1.text(i + 1, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot cumulative explained variance
        ax2.plot(components, self.cumulative_variance_, 'o-', color='red', linewidth=2, markersize=6)
        ax2.axhline(y=0.95, color='green', linestyle='--', alpha=0.7, label='95% Variance')
        ax2.axhline(y=0.90, color='orange', linestyle='--', alpha=0.7, label='90% Variance')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Explained Variance')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(0, 1.05)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Explained variance plot saved to: {save_path}")
        
        plt.show()
    
    def plot_cumulative_variance(self, save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot cumulative explained variance with threshold lines.
        
        Args:
            save_path: Path to save the plot (optional)
            figsize: Figure size tuple
        """
        if self.cumulative_variance_ is None:
            raise ValueError("PCA model must be fitted first. Call fit_pca() method.")
        
        plt.figure(figsize=figsize)
        
        components = range(1, len(self.cumulative_variance_) + 1)
        plt.plot(components, self.cumulative_variance_, 'o-', linewidth=3, markersize=8, color='darkblue')
        
        # Add threshold lines
        thresholds = [0.80, 0.85, 0.90, 0.95, 0.99]
        colors = ['red', 'orange', 'yellow', 'green', 'purple']
        
        for threshold, color in zip(thresholds, colors):
            plt.axhline(y=threshold, color=color, linestyle='--', alpha=0.7, 
                       label=f'{threshold*100}% Variance')
            
            # Find and mark the component number for this threshold
            if np.any(self.cumulative_variance_ >= threshold):
                comp_idx = np.argmax(self.cumulative_variance_ >= threshold)
                plt.axvline(x=comp_idx + 1, color=color, linestyle=':', alpha=0.5)
                plt.text(comp_idx + 1, threshold + 0.02, f'PC{comp_idx + 1}', 
                        ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')
        
        plt.xlabel('Number of Principal Components', fontsize=12)
        plt.ylabel('Cumulative Explained Variance Ratio', fontsize=12)
        plt.title('Cumulative Explained Variance Analysis', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.ylim(0, 1.05)
        plt.xlim(0.5, len(components) + 0.5)
        
        # Add text box with summary
        if self.n_components_optimal_:
            textstr = f'Optimal Components (95%): {self.n_components_optimal_}\n'
            textstr += f'Variance Explained: {self.cumulative_variance_[self.n_components_optimal_-1]:.3f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cumulative variance plot saved to: {save_path}")
        
        plt.show()    

    def plot_pca_scatter(self, X_pca: np.ndarray, y: np.ndarray, 
                        save_path: Optional[str] = None,
                        figsize: Tuple[int, int] = (15, 5)) -> None:
        """
        Create 2D and 3D PCA scatter plots with target coloring.
        
        Args:
            X_pca: PCA-transformed data
            y: Target labels for coloring
            save_path: Path to save the plot (optional)
            figsize: Figure size tuple
        """
        if X_pca.shape[1] < 2:
            raise ValueError("Need at least 2 principal components for scatter plot")
        
        # Create subplots
        if X_pca.shape[1] >= 3:
            fig = plt.figure(figsize=figsize)
            
            # 2D scatter plot (PC1 vs PC2)
            ax1 = plt.subplot(1, 3, 1)
            scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7, s=50)
            ax1.set_xlabel(f'PC1 ({self.explained_variance_ratio_[0]:.3f} variance)')
            ax1.set_ylabel(f'PC2 ({self.explained_variance_ratio_[1]:.3f} variance)')
            ax1.set_title('PCA: PC1 vs PC2')
            ax1.grid(True, alpha=0.3)
            plt.colorbar(scatter1, ax=ax1, label='Target')
            
            # 2D scatter plot (PC1 vs PC3)
            ax2 = plt.subplot(1, 3, 2)
            scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 2], c=y, cmap='viridis', alpha=0.7, s=50)
            ax2.set_xlabel(f'PC1 ({self.explained_variance_ratio_[0]:.3f} variance)')
            ax2.set_ylabel(f'PC3 ({self.explained_variance_ratio_[2]:.3f} variance)')
            ax2.set_title('PCA: PC1 vs PC3')
            ax2.grid(True, alpha=0.3)
            plt.colorbar(scatter2, ax=ax2, label='Target')
            
            # 3D scatter plot
            ax3 = fig.add_subplot(1, 3, 3, projection='3d')
            scatter3 = ax3.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                                 c=y, cmap='viridis', alpha=0.7, s=50)
            ax3.set_xlabel(f'PC1 ({self.explained_variance_ratio_[0]:.3f})')
            ax3.set_ylabel(f'PC2 ({self.explained_variance_ratio_[1]:.3f})')
            ax3.set_zlabel(f'PC3 ({self.explained_variance_ratio_[2]:.3f})')
            ax3.set_title('3D PCA Visualization')
            plt.colorbar(scatter3, ax=ax3, label='Target', shrink=0.8)
            
        else:
            # Only 2D plot available
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7, s=50)
            ax.set_xlabel(f'PC1 ({self.explained_variance_ratio_[0]:.3f} variance)')
            ax.set_ylabel(f'PC2 ({self.explained_variance_ratio_[1]:.3f} variance)')
            ax.set_title('PCA: PC1 vs PC2')
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Target')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PCA scatter plot saved to: {save_path}")
        
        plt.show()
    
    def plot_component_loadings(self, save_path: Optional[str] = None,
                              n_components: int = 4, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot component loadings (feature contributions) for the first n components.
        
        Args:
            save_path: Path to save the plot (optional)
            n_components: Number of components to plot
            figsize: Figure size tuple
        """
        if self.pca_model is None:
            raise ValueError("PCA model must be fitted first. Call fit_pca() method.")
        
        n_components = min(n_components, self.pca_model.n_components_)
        
        # Get component loadings
        loadings = self.pca_model.components_[:n_components]
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()
        
        for i in range(n_components):
            ax = axes[i]
            
            # Create bar plot of loadings
            y_pos = np.arange(len(self.feature_names))
            colors = ['red' if x < 0 else 'blue' for x in loadings[i]]
            
            bars = ax.barh(y_pos, loadings[i], color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(self.feature_names, fontsize=10)
            ax.set_xlabel('Loading Value')
            ax.set_title(f'PC{i+1} Loadings ({self.explained_variance_ratio_[i]:.3f} variance)')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels on bars
            for j, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                       f'{width:.3f}', ha='left' if width >= 0 else 'right', va='center', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Component loadings plot saved to: {save_path}")
        
        plt.show()
    
    def generate_pca_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive PCA analysis report.
        
        Returns:
            Dictionary containing complete PCA analysis results
        """
        if self.pca_model is None:
            raise ValueError("PCA model must be fitted first. Call fit_pca() method.")
        
        # Calculate additional metrics
        variance_info = self.calculate_explained_variance()
        
        # Find components for different variance thresholds
        thresholds = [0.80, 0.85, 0.90, 0.95, 0.99]
        components_for_threshold = {}
        
        for threshold in thresholds:
            if np.any(self.cumulative_variance_ >= threshold):
                comp_idx = np.argmax(self.cumulative_variance_ >= threshold) + 1
                components_for_threshold[f'{threshold*100}%'] = comp_idx
        
        # Create comprehensive report
        report = {
            'pca_summary': {
                'total_components': self.pca_model.n_components_,
                'total_features': len(self.feature_names),
                'total_variance_explained': float(self.cumulative_variance_[-1]),
                'optimal_components_95pct': self.n_components_optimal_
            },
            'variance_analysis': {
                'explained_variance_ratio': self.explained_variance_ratio_.tolist(),
                'cumulative_variance': self.cumulative_variance_.tolist(),
                'explained_variance': variance_info['explained_variance'].tolist(),
                'singular_values': variance_info['singular_values'].tolist()
            },
            'component_thresholds': components_for_threshold,
            'top_components_summary': {
                f'PC{i+1}': {
                    'variance_explained': float(self.explained_variance_ratio_[i]),
                    'cumulative_variance': float(self.cumulative_variance_[i]),
                    'top_features': self._get_top_features_for_component(i, top_n=3)
                }
                for i in range(min(5, len(self.explained_variance_ratio_)))
            },
            'feature_names': self.feature_names,
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _get_top_features_for_component(self, component_idx: int, top_n: int = 3) -> Dict[str, float]:
        """
        Get top contributing features for a specific component.
        
        Args:
            component_idx: Index of the component
            top_n: Number of top features to return
            
        Returns:
            Dictionary of top features and their loadings
        """
        if self.pca_model is None:
            return {}
        
        loadings = self.pca_model.components_[component_idx]
        abs_loadings = np.abs(loadings)
        top_indices = np.argsort(abs_loadings)[-top_n:][::-1]
        
        top_features = {}
        for idx in top_indices:
            feature_name = self.feature_names[idx]
            loading_value = float(loadings[idx])
            top_features[feature_name] = loading_value
        
        return top_features
    
    def _generate_recommendations(self) -> Dict[str, str]:
        """
        Generate recommendations based on PCA analysis.
        
        Returns:
            Dictionary of recommendations
        """
        recommendations = {}
        
        if self.cumulative_variance_ is not None:
            # Dimensionality reduction recommendation
            if self.n_components_optimal_ and self.n_components_optimal_ < len(self.feature_names):
                reduction_pct = (1 - self.n_components_optimal_ / len(self.feature_names)) * 100
                recommendations['dimensionality_reduction'] = (
                    f"Consider reducing dimensions from {len(self.feature_names)} to "
                    f"{self.n_components_optimal_} components ({reduction_pct:.1f}% reduction) "
                    f"while retaining 95% of variance."
                )
            
            # Variance concentration
            first_two_variance = self.cumulative_variance_[1] if len(self.cumulative_variance_) > 1 else 0
            if first_two_variance > 0.6:
                recommendations['variance_concentration'] = (
                    f"First two components explain {first_two_variance:.1f}% of variance. "
                    f"2D visualization should be highly informative."
                )
            
            # Component selection
            if len(self.cumulative_variance_) > 5:
                five_comp_variance = self.cumulative_variance_[4]
                recommendations['component_selection'] = (
                    f"First 5 components explain {five_comp_variance:.1f}% of variance. "
                    f"Consider using 5 components for most analyses."
                )
        
        return recommendations
    
    def save_model(self, filepath: str) -> None:
        """
        Save the fitted PCA model and scaler.
        
        Args:
            filepath: Path to save the model
        """
        if self.pca_model is None or self.scaler is None:
            raise ValueError("PCA model must be fitted first. Call fit_pca() method.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model components
        model_data = {
            'pca_model': self.pca_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'explained_variance_ratio_': self.explained_variance_ratio_,
            'cumulative_variance_': self.cumulative_variance_,
            'n_components_optimal_': self.n_components_optimal_
        }
        
        joblib.dump(model_data, filepath)
        print(f"PCA model saved to: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a previously saved PCA model.
        
        Args:
            filepath: Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load model components
        model_data = joblib.load(filepath)
        
        self.pca_model = model_data['pca_model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.explained_variance_ratio_ = model_data['explained_variance_ratio_']
        self.cumulative_variance_ = model_data['cumulative_variance_']
        self.n_components_optimal_ = model_data.get('n_components_optimal_')
        
        print(f"PCA model loaded from: {filepath}")


def main():
    """
    Example usage of PCAAnalyzer class.
    """
    # Load the cleaned heart disease data
    data_path = 'data/processed/heart_disease_cleaned.csv'
    
    if os.path.exists(data_path):
        # Load data
        df = pd.read_csv(data_path)
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        feature_names = X.columns.tolist()
        
        # Initialize PCA analyzer
        pca_analyzer = PCAAnalyzer()
        
        # Fit PCA
        pca_analyzer.fit_pca(X.values, feature_names=feature_names)
        
        # Find optimal components
        optimal_components = pca_analyzer.find_optimal_components(variance_threshold=0.95)
        
        # Transform data
        X_pca = pca_analyzer.transform_data(X.values)
        
        # Generate visualizations
        pca_analyzer.plot_explained_variance()
        pca_analyzer.plot_cumulative_variance()
        pca_analyzer.plot_pca_scatter(X_pca, y.values)
        pca_analyzer.plot_component_loadings()
        
        # Generate report
        report = pca_analyzer.generate_pca_report()
        print("\nPCA Analysis Report:")
        print(f"Total variance explained: {report['pca_summary']['total_variance_explained']:.4f}")
        print(f"Optimal components (95%): {report['pca_summary']['optimal_components_95pct']}")
        
        # Save model
        pca_analyzer.save_model('models/pca_model.pkl')
        
        print("\nPCA analysis completed successfully!")
    else:
        print(f"Data file not found: {data_path}")
        print("Please ensure the data preprocessing step has been completed.")


if __name__ == "__main__":
    main()
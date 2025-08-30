"""
Clustering Analysis Module for Heart Disease ML Pipeline

This module implements K-Means and Hierarchical clustering algorithms
with comprehensive evaluation and visualization capabilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import joblib
import os
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class ClusteringAnalyzer:
    """
    Comprehensive clustering analysis system for heart disease data.
    
    Implements K-Means and Hierarchical clustering with evaluation metrics,
    visualization, and comparison with true labels.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the clustering analyzer.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.kmeans_models = {}
        self.hierarchical_models = {}
        self.scaler = StandardScaler()
        self.results = {}
        
    def load_data(self, pca_file: str, cleaned_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load PCA-transformed and cleaned datasets.
        
        Args:
            pca_file: Path to PCA-transformed data
            cleaned_file: Path to cleaned original data
            
        Returns:
            Tuple of (pca_data, cleaned_data)
        """
        try:
            pca_data = pd.read_csv(pca_file)
            cleaned_data = pd.read_csv(cleaned_file)
            
            print(f"Loaded PCA data: {pca_data.shape}")
            print(f"Loaded cleaned data: {cleaned_data.shape}")
            
            return pca_data, cleaned_data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def prepare_clustering_data(self, data: pd.DataFrame, use_pca: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for clustering analysis.
        
        Args:
            data: Input dataframe
            use_pca: Whether to use PCA-transformed features
            
        Returns:
            Tuple of (features, target)
        """
        if 'target' in data.columns:
            target = data['target'].values
            features = data.drop('target', axis=1).values
        else:
            target = None
            features = data.values
            
        # Scale features for clustering
        features_scaled = self.scaler.fit_transform(features)
        
        return features_scaled, target
    
    def kmeans_clustering(self, X: np.ndarray, k_range: Tuple[int, int] = (2, 10)) -> Dict[str, Any]:
        """
        Perform K-Means clustering with elbow method for optimal K selection.
        
        Args:
            X: Feature matrix
            k_range: Range of K values to test
            
        Returns:
            Dictionary containing clustering results and metrics
        """
        print("Performing K-Means clustering analysis...")
        
        k_min, k_max = k_range
        inertias = []
        silhouette_scores = []
        k_values = list(range(k_min, k_max + 1))
        
        # Test different K values
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            inertias.append(kmeans.inertia_)
            
            # Calculate silhouette score
            if k > 1:  # Silhouette score requires at least 2 clusters
                sil_score = silhouette_score(X, cluster_labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)
            
            # Store model
            self.kmeans_models[k] = kmeans
        
        # Find optimal K using elbow method (simplified)
        optimal_k = self._find_optimal_k_elbow(k_values, inertias)
        
        # Find optimal K using silhouette score
        optimal_k_silhouette = k_values[np.argmax(silhouette_scores)]
        
        results = {
            'k_values': k_values,
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'optimal_k_elbow': optimal_k,
            'optimal_k_silhouette': optimal_k_silhouette,
            'best_model': self.kmeans_models[optimal_k_silhouette]
        }
        
        print(f"Optimal K (elbow method): {optimal_k}")
        print(f"Optimal K (silhouette): {optimal_k_silhouette}")
        
        return results
    
    def _find_optimal_k_elbow(self, k_values: List[int], inertias: List[float]) -> int:
        """
        Find optimal K using elbow method.
        
        Args:
            k_values: List of K values
            inertias: List of inertia values
            
        Returns:
            Optimal K value
        """
        # Calculate the rate of change in inertia
        if len(inertias) < 3:
            return k_values[0]
        
        # Simple elbow detection using second derivative
        diffs = np.diff(inertias)
        diffs2 = np.diff(diffs)
        
        if len(diffs2) > 0:
            elbow_idx = np.argmax(diffs2) + 2  # +2 because of double diff
            if elbow_idx < len(k_values):
                return k_values[elbow_idx]
        
        # Fallback: return K with highest silhouette score
        return k_values[len(k_values)//2]
    
    def hierarchical_clustering(self, X: np.ndarray, n_clusters: int = 3, 
                               linkage_method: str = 'ward') -> Dict[str, Any]:
        """
        Perform Hierarchical clustering with dendrogram analysis.
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            linkage_method: Linkage method for clustering
            
        Returns:
            Dictionary containing clustering results
        """
        print(f"Performing Hierarchical clustering with {n_clusters} clusters...")
        
        # Perform hierarchical clustering
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters, 
            linkage=linkage_method
        )
        cluster_labels = hierarchical.fit_predict(X)
        
        # Calculate linkage matrix for dendrogram
        linkage_matrix = linkage(X, method=linkage_method)
        
        # Calculate silhouette score
        sil_score = silhouette_score(X, cluster_labels) if n_clusters > 1 else 0
        
        results = {
            'model': hierarchical,
            'cluster_labels': cluster_labels,
            'linkage_matrix': linkage_matrix,
            'silhouette_score': sil_score,
            'n_clusters': n_clusters,
            'linkage_method': linkage_method
        }
        
        self.hierarchical_models[n_clusters] = hierarchical
        
        print(f"Hierarchical clustering silhouette score: {sil_score:.3f}")
        
        return results
    
    def evaluate_clustering(self, X: np.ndarray, cluster_labels: np.ndarray, 
                          y_true: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate clustering performance using multiple metrics.
        
        Args:
            X: Feature matrix
            cluster_labels: Predicted cluster labels
            y_true: True labels (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Silhouette score
        if len(np.unique(cluster_labels)) > 1:
            metrics['silhouette_score'] = silhouette_score(X, cluster_labels)
        else:
            metrics['silhouette_score'] = 0
        
        # Inertia (for K-means-like evaluation)
        centroids = []
        for cluster_id in np.unique(cluster_labels):
            cluster_points = X[cluster_labels == cluster_id]
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)
        
        inertia = 0
        for i, cluster_id in enumerate(np.unique(cluster_labels)):
            cluster_points = X[cluster_labels == cluster_id]
            distances = np.sum((cluster_points - centroids[i]) ** 2, axis=1)
            inertia += np.sum(distances)
        
        metrics['inertia'] = inertia
        
        # If true labels are available
        if y_true is not None:
            metrics['adjusted_rand_score'] = adjusted_rand_score(y_true, cluster_labels)
        
        # Number of clusters
        metrics['n_clusters'] = len(np.unique(cluster_labels))
        
        return metrics
    
    def plot_elbow_curve(self, kmeans_results: Dict[str, Any], save_path: str = None) -> None:
        """
        Plot elbow curve for K-Means optimal cluster selection.
        
        Args:
            kmeans_results: Results from kmeans_clustering
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 5))
        
        # Elbow curve (Inertia)
        plt.subplot(1, 2, 1)
        plt.plot(kmeans_results['k_values'], kmeans_results['inertias'], 'bo-')
        plt.axvline(x=kmeans_results['optimal_k_elbow'], color='r', linestyle='--', 
                   label=f'Optimal K (Elbow): {kmeans_results["optimal_k_elbow"]}')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal K')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Silhouette scores
        plt.subplot(1, 2, 2)
        plt.plot(kmeans_results['k_values'], kmeans_results['silhouette_scores'], 'go-')
        plt.axvline(x=kmeans_results['optimal_k_silhouette'], color='r', linestyle='--',
                   label=f'Optimal K (Silhouette): {kmeans_results["optimal_k_silhouette"]}')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score vs Number of Clusters')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Elbow curve saved to: {save_path}")
        
        plt.show()
    
    def plot_dendrogram(self, hierarchical_results: Dict[str, Any], 
                       save_path: str = None, max_display: int = 30) -> None:
        """
        Plot dendrogram for hierarchical clustering visualization.
        
        Args:
            hierarchical_results: Results from hierarchical_clustering
            save_path: Path to save the plot
            max_display: Maximum number of leaves to display
        """
        plt.figure(figsize=(15, 8))
        
        # Create dendrogram
        dendrogram(
            hierarchical_results['linkage_matrix'],
            truncate_mode='lastp',
            p=max_display,
            leaf_rotation=90,
            leaf_font_size=10,
            show_contracted=True
        )
        
        plt.title(f'Hierarchical Clustering Dendrogram\n'
                 f'Linkage: {hierarchical_results["linkage_method"]}, '
                 f'Clusters: {hierarchical_results["n_clusters"]}')
        plt.xlabel('Sample Index or (Cluster Size)')
        plt.ylabel('Distance')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dendrogram saved to: {save_path}")
        
        plt.show()
    
    def plot_cluster_scatter(self, X: np.ndarray, cluster_labels: np.ndarray, 
                           y_true: Optional[np.ndarray] = None, 
                           save_path: str = None) -> None:
        """
        Plot 2D cluster visualization using first two principal components.
        
        Args:
            X: Feature matrix (should be PCA-transformed for best visualization)
            cluster_labels: Predicted cluster labels
            y_true: True labels for comparison (optional)
            save_path: Path to save the plot
        """
        if X.shape[1] < 2:
            print("Need at least 2 features for scatter plot")
            return
        
        n_plots = 2 if y_true is not None else 1
        plt.figure(figsize=(6 * n_plots, 5))
        
        # Plot predicted clusters
        plt.subplot(1, n_plots, 1)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter)
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title(f'Predicted Clusters (K={len(np.unique(cluster_labels))})')
        
        # Plot true labels if available
        if y_true is not None:
            plt.subplot(1, n_plots, 2)
            scatter = plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='coolwarm', alpha=0.7)
            plt.colorbar(scatter)
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
            plt.title('True Labels')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cluster scatter plot saved to: {save_path}")
        
        plt.show()
    
    def compare_with_true_labels(self, cluster_labels: np.ndarray, 
                               y_true: np.ndarray) -> Dict[str, Any]:
        """
        Analyze cluster-target relationships and create comparison visualizations.
        
        Args:
            cluster_labels: Predicted cluster labels
            y_true: True target labels
            
        Returns:
            Dictionary containing comparison analysis
        """
        print("Analyzing cluster-target relationships...")
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'cluster': cluster_labels,
            'true_label': y_true
        })
        
        # Cross-tabulation
        crosstab = pd.crosstab(comparison_df['cluster'], comparison_df['true_label'])
        
        # Calculate cluster purity
        cluster_purity = {}
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_true_labels = y_true[cluster_mask]
            
            if len(cluster_true_labels) > 0:
                # Most common true label in this cluster
                most_common_label = np.bincount(cluster_true_labels).argmax()
                purity = np.sum(cluster_true_labels == most_common_label) / len(cluster_true_labels)
                cluster_purity[cluster_id] = {
                    'dominant_label': most_common_label,
                    'purity': purity,
                    'size': len(cluster_true_labels)
                }
        
        # Overall metrics
        adjusted_rand = adjusted_rand_score(y_true, cluster_labels)
        
        results = {
            'crosstab': crosstab,
            'cluster_purity': cluster_purity,
            'adjusted_rand_score': adjusted_rand,
            'comparison_df': comparison_df
        }
        
        # Print summary
        print(f"Adjusted Rand Score: {adjusted_rand:.3f}")
        print("\nCluster Purity Analysis:")
        for cluster_id, purity_info in cluster_purity.items():
            print(f"Cluster {cluster_id}: {purity_info['purity']:.3f} purity, "
                  f"size: {purity_info['size']}, dominant label: {purity_info['dominant_label']}")
        
        return results
    
    def cluster_analysis_report(self, X: np.ndarray, y_true: np.ndarray, 
                              save_dir: str = "results/clustering") -> Dict[str, Any]:
        """
        Generate comprehensive clustering analysis report.
        
        Args:
            X: Feature matrix
            y_true: True labels
            save_dir: Directory to save results
            
        Returns:
            Complete analysis results
        """
        print("Generating comprehensive clustering analysis report...")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Perform K-Means analysis
        kmeans_results = self.kmeans_clustering(X)
        
        # Perform Hierarchical clustering
        hierarchical_results = self.hierarchical_clustering(X, n_clusters=3)
        
        # Get best K-Means model predictions
        best_kmeans = kmeans_results['best_model']
        kmeans_labels = best_kmeans.predict(X)
        
        # Evaluate both methods
        kmeans_metrics = self.evaluate_clustering(X, kmeans_labels, y_true)
        hierarchical_metrics = self.evaluate_clustering(X, hierarchical_results['cluster_labels'], y_true)
        
        # Compare with true labels
        kmeans_comparison = self.compare_with_true_labels(kmeans_labels, y_true)
        hierarchical_comparison = self.compare_with_true_labels(hierarchical_results['cluster_labels'], y_true)
        
        # Generate visualizations
        self.plot_elbow_curve(kmeans_results, os.path.join(save_dir, 'elbow_curve.png'))
        self.plot_dendrogram(hierarchical_results, os.path.join(save_dir, 'dendrogram.png'))
        self.plot_cluster_scatter(X, kmeans_labels, y_true, 
                                os.path.join(save_dir, 'kmeans_clusters.png'))
        self.plot_cluster_scatter(X, hierarchical_results['cluster_labels'], y_true,
                                os.path.join(save_dir, 'hierarchical_clusters.png'))
        
        # Compile comprehensive results
        comprehensive_results = {
            'kmeans': {
                'results': kmeans_results,
                'metrics': kmeans_metrics,
                'comparison': kmeans_comparison,
                'labels': kmeans_labels
            },
            'hierarchical': {
                'results': hierarchical_results,
                'metrics': hierarchical_metrics,
                'comparison': hierarchical_comparison,
                'labels': hierarchical_results['cluster_labels']
            },
            'data_info': {
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'n_true_classes': len(np.unique(y_true))
            }
        }
        
        # Save cluster assignments
        assignments_df = pd.DataFrame({
            'sample_id': range(len(y_true)),
            'true_label': y_true,
            'kmeans_cluster': kmeans_labels,
            'hierarchical_cluster': hierarchical_results['cluster_labels']
        })
        
        assignments_path = os.path.join(save_dir, 'cluster_assignments.csv')
        assignments_df.to_csv(assignments_path, index=False)
        print(f"Cluster assignments saved to: {assignments_path}")
        
        # Print summary report
        self._print_analysis_summary(comprehensive_results)
        
        return comprehensive_results
    
    def _print_analysis_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of the clustering analysis."""
        print("\n" + "="*60)
        print("CLUSTERING ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Dataset: {results['data_info']['n_samples']} samples, "
              f"{results['data_info']['n_features']} features, "
              f"{results['data_info']['n_true_classes']} true classes")
        
        print("\nK-MEANS CLUSTERING:")
        kmeans = results['kmeans']
        print(f"  Optimal K (silhouette): {kmeans['results']['optimal_k_silhouette']}")
        print(f"  Silhouette Score: {kmeans['metrics']['silhouette_score']:.3f}")
        print(f"  Adjusted Rand Score: {kmeans['metrics']['adjusted_rand_score']:.3f}")
        
        print("\nHIERARCHICAL CLUSTERING:")
        hierarchical = results['hierarchical']
        print(f"  Number of Clusters: {hierarchical['results']['n_clusters']}")
        print(f"  Silhouette Score: {hierarchical['metrics']['silhouette_score']:.3f}")
        print(f"  Adjusted Rand Score: {hierarchical['metrics']['adjusted_rand_score']:.3f}")
        
        print("\nRECOMMENDATION:")
        if kmeans['metrics']['silhouette_score'] > hierarchical['metrics']['silhouette_score']:
            print("  K-Means clustering shows better silhouette score")
        else:
            print("  Hierarchical clustering shows better silhouette score")
        
        print("="*60)
    
    def save_models(self, save_dir: str = "models/unsupervised") -> None:
        """
        Save trained clustering models.
        
        Args:
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save K-Means models
        for k, model in self.kmeans_models.items():
            model_path = os.path.join(save_dir, f'kmeans_k{k}.pkl')
            joblib.dump(model, model_path)
        
        # Save Hierarchical models
        for n_clusters, model in self.hierarchical_models.items():
            model_path = os.path.join(save_dir, f'hierarchical_{n_clusters}clusters.pkl')
            joblib.dump(model, model_path)
        
        # Save scaler
        scaler_path = os.path.join(save_dir, 'clustering_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Models saved to: {save_dir}")


def main():
    """Main function to demonstrate clustering analysis."""
    # Initialize analyzer
    analyzer = ClusteringAnalyzer()
    
    # Load data
    try:
        pca_data, cleaned_data = analyzer.load_data(
            'data/processed/heart_disease_pca.csv',
            'data/processed/heart_disease_cleaned.csv'
        )
        
        # Prepare data for clustering (using PCA data)
        X_pca, y_true = analyzer.prepare_clustering_data(pca_data, use_pca=True)
        
        # Generate comprehensive analysis report
        results = analyzer.cluster_analysis_report(X_pca, y_true)
        
        # Save models
        analyzer.save_models()
        
        print("\nClustering analysis completed successfully!")
        
    except Exception as e:
        print(f"Error in clustering analysis: {e}")
        raise


if __name__ == "__main__":
    main()
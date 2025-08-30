"""
Interactive Data Visualization Dashboard for Heart Disease ML Pipeline
Provides comprehensive data exploration and model performance visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class VisualizationDashboard:
    """Main class for interactive data visualization dashboard"""
    
    def __init__(self):
        self.data_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "heart_disease_cleaned.csv")
        self.metrics_path = os.path.join(os.path.dirname(__file__), "..", "results", "model_evaluation", "evaluation_metrics.json")
        self.clustering_path = os.path.join(os.path.dirname(__file__), "..", "results", "clustering", "cluster_assignments.csv")
        
        self.feature_labels = {
            'age': 'Age (years)',
            'sex': 'Sex (0=Female, 1=Male)',
            'cp': 'Chest Pain Type',
            'trestbps': 'Resting Blood Pressure (mmHg)',
            'chol': 'Cholesterol (mg/dl)',
            'fbs': 'Fasting Blood Sugar >120 mg/dl',
            'restecg': 'Resting ECG Results',
            'thalach': 'Max Heart Rate Achieved',
            'exang': 'Exercise Induced Angina',
            'oldpeak': 'ST Depression',
            'slope': 'ST Slope',
            'ca': 'Major Vessels (0-3)',
            'thal': 'Thalassemia',
            'target': 'Heart Disease (0=No, 1=Yes)'
        }
        
        self.load_data()
    
    def load_data(self):
        """Load all required datasets for visualization"""
        try:
            # Load main dataset
            if os.path.exists(self.data_path):
                self.df = pd.read_csv(self.data_path)
                st.success(f"✅ Dataset loaded: {len(self.df)} samples")
            else:
                st.error(f"❌ Dataset not found at: {self.data_path}")
                self.df = None
            
            # Load model evaluation metrics
            if os.path.exists(self.metrics_path):
                with open(self.metrics_path, 'r') as f:
                    self.metrics_data = json.load(f)
                st.success("✅ Model evaluation metrics loaded")
            else:
                st.warning("⚠️ Model evaluation metrics not found")
                self.metrics_data = None
            
            # Load clustering results
            if os.path.exists(self.clustering_path):
                self.clustering_df = pd.read_csv(self.clustering_path)
                st.success("✅ Clustering results loaded")
            else:
                st.warning("⚠️ Clustering results not found")
                self.clustering_df = None
                
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
    
    def render_data_explorer_page(self):
        """Render comprehensive data exploration page with dataset overview and statistics"""
        st.title("📊 Data Explorer")
        
        if self.df is None:
            st.error("❌ No dataset available for exploration")
            return
        
        # Dataset Overview Section
        st.header("📋 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(self.df))
        with col2:
            st.metric("Features", len(self.df.columns) - 1)  # Excluding target
        with col3:
            disease_count = self.df['target'].sum()
            st.metric("Disease Cases", disease_count)
        with col4:
            disease_rate = (disease_count / len(self.df)) * 100
            st.metric("Disease Rate", f"{disease_rate:.1f}%")
        
        # Dataset Statistics
        st.subheader("📈 Statistical Summary")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Descriptive Stats", "🔍 Missing Values", "📋 Data Types"])
        
        with tab1:
            st.dataframe(self.df.describe(), use_container_width=True)
        
        with tab2:
            missing_data = self.df.isnull().sum()
            if missing_data.sum() == 0:
                st.success("✅ No missing values found in the dataset")
            else:
                fig = px.bar(x=missing_data.index, y=missing_data.values,
                           title="Missing Values by Feature",
                           labels={'x': 'Features', 'y': 'Missing Count'})
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            dtype_info = pd.DataFrame({
                'Feature': self.df.columns,
                'Data Type': self.df.dtypes.values,
                'Non-Null Count': self.df.count().values,
                'Unique Values': [self.df[col].nunique() for col in self.df.columns]
            })
            st.dataframe(dtype_info, use_container_width=True)
        
        # Interactive Feature Distribution
        st.header("📊 Feature Distributions")
        self.interactive_feature_distribution()
        
        # Correlation Analysis
        st.header("🔗 Feature Correlations")
        self.correlation_matrix_interactive()
    
    def interactive_feature_distribution(self):
        """Create interactive feature distribution plots with selectable features and filtering"""
        if self.df is None:
            return
        
        # Feature selection
        numeric_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_features:
            numeric_features.remove('target')
        
        selected_feature = st.selectbox(
            "Select feature to explore:",
            numeric_features,
            format_func=lambda x: self.feature_labels.get(x, x),
            key="feature_distribution_selector"
        )
        
        # Filtering options
        col1, col2 = st.columns(2)
        with col1:
            filter_by_target = st.checkbox("Split by Heart Disease Status", value=True, key="filter_by_target_dist")
        with col2:
            filter_by_sex = st.checkbox("Split by Sex", key="filter_by_sex_dist")
        
        # Create distribution plot
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Distribution', 'Box Plot', 'Violin Plot', 'Statistics'),
                           specs=[[{"secondary_y": False}, {"secondary_y": False}],
                                  [{"secondary_y": False}, {"type": "table"}]])
        
        # Prepare data based on filters
        plot_data = self.df.copy()
        color_col = None
        
        if filter_by_target and filter_by_sex:
            plot_data['group'] = plot_data['target'].astype(str) + '_' + plot_data['sex'].astype(str)
            color_col = 'group'
        elif filter_by_target:
            color_col = 'target'
        elif filter_by_sex:
            color_col = 'sex'
        
        # Histogram
        if color_col:
            for group in plot_data[color_col].unique():
                group_data = plot_data[plot_data[color_col] == group]
                fig.add_trace(
                    go.Histogram(x=group_data[selected_feature], name=f'Group {group}',
                               opacity=0.7, nbinsx=20),
                    row=1, col=1
                )
        else:
            fig.add_trace(
                go.Histogram(x=plot_data[selected_feature], name='Distribution',
                           opacity=0.7, nbinsx=20),
                row=1, col=1
            )
        
        # Box plot
        if color_col:
            fig.add_trace(
                go.Box(y=plot_data[selected_feature], x=plot_data[color_col],
                      name='Box Plot'),
                row=1, col=2
            )
        else:
            fig.add_trace(
                go.Box(y=plot_data[selected_feature], name='Box Plot'),
                row=1, col=2
            )
        
        # Violin plot
        if color_col:
            fig.add_trace(
                go.Violin(y=plot_data[selected_feature], x=plot_data[color_col],
                         name='Violin Plot'),
                row=2, col=1
            )
        else:
            fig.add_trace(
                go.Violin(y=plot_data[selected_feature], name='Violin Plot'),
                row=2, col=1
            )
        
        # Statistics table
        stats_data = plot_data[selected_feature].describe()
        fig.add_trace(
            go.Table(
                header=dict(values=['Statistic', 'Value']),
                cells=dict(values=[stats_data.index, stats_data.values.round(2)])
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title=f"Distribution Analysis: {self.feature_labels.get(selected_feature, selected_feature)}")
        st.plotly_chart(fig, use_container_width=True)
    
    def correlation_matrix_interactive(self):
        """Create interactive correlation matrix with hover information and feature selection"""
        if self.df is None:
            return
        
        # Feature selection for correlation
        numeric_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        selected_features = st.multiselect(
            "Select features for correlation analysis:",
            numeric_features,
            default=numeric_features[:8],  # Default to first 8 features
            format_func=lambda x: self.feature_labels.get(x, x),
            key="correlation_features_selector"
        )
        
        if len(selected_features) < 2:
            st.warning("Please select at least 2 features for correlation analysis")
            return
        
        # Calculate correlation matrix
        corr_matrix = self.df[selected_features].corr()
        
        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=[self.feature_labels.get(col, col) for col in corr_matrix.columns],
            y=[self.feature_labels.get(col, col) for col in corr_matrix.index],
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(3),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate='<b>%{x}</b><br><b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Interactive Correlation Matrix",
            xaxis_title="Features",
            yaxis_title="Features",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def model_performance_comparison(self):
        """Create interactive charts for model metrics comparison"""
        if self.metrics_data is None:
            st.error("❌ Model evaluation metrics not available")
            return
        
        st.header("🤖 Model Performance Comparison")
        
        # Extract performance data
        if 'comparison' in self.metrics_data and 'performance_table' in self.metrics_data['comparison']:
            performance_df = pd.DataFrame(self.metrics_data['comparison']['performance_table'])
            
            # Convert string metrics to float
            metric_columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Specificity', 'Avg Precision']
            for col in metric_columns:
                if col in performance_df.columns:
                    performance_df[col] = pd.to_numeric(performance_df[col], errors='coerce')
            
            # Model performance radar chart
            st.subheader("📊 Model Performance Radar Chart")
            
            selected_models = st.multiselect(
                "Select models to compare:",
                performance_df['Model'].tolist(),
                default=performance_df['Model'].tolist()[:3],
                key="radar_chart_model_selector"
            )
            
            if selected_models:
                fig = go.Figure()
                
                for model in selected_models:
                    model_data = performance_df[performance_df['Model'] == model].iloc[0]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=[model_data[col] for col in metric_columns if col in performance_df.columns],
                        theta=metric_columns,
                        fill='toself',
                        name=model,
                        opacity=0.6
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title="Model Performance Comparison (Radar Chart)",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Performance table
            st.subheader("📋 Complete Performance Table")
            st.dataframe(performance_df, use_container_width=True)
    
    def feature_importance_visualization(self):
        """Create sortable and filterable importance plots"""
        st.header("🎯 Feature Importance Analysis")
        
        st.info("📝 Feature importance visualization requires trained models with feature importance attributes.")
        st.info("This would typically show Random Forest or other tree-based model feature importances.")
        
        # Create a mock feature importance for demonstration
        features = ['age', 'cp', 'thalach', 'oldpeak', 'ca', 'thal', 'chol', 'trestbps', 'exang', 'slope', 'sex', 'fbs', 'restecg']
        
        # Mock importance scores (in real implementation, these would come from trained models)
        mock_importance = np.random.random(len(features))
        mock_importance = mock_importance / mock_importance.sum()  # Normalize
        
        importance_df = pd.DataFrame({
            'Feature': [self.feature_labels.get(f, f) for f in features],
            'Importance': mock_importance,
            'Feature_Code': features
        }).sort_values('Importance', ascending=False)
        
        # Interactive importance plot
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance (Mock Data)',
            color='Importance',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance table
        st.subheader("📋 Feature Importance Rankings")
        importance_df['Rank'] = range(1, len(importance_df) + 1)
        st.dataframe(importance_df[['Rank', 'Feature', 'Importance']], use_container_width=True)
    
    def prediction_distribution_analysis(self):
        """Show model prediction patterns and distribution"""
        st.header("📈 Prediction Distribution Analysis")
        
        if self.df is None:
            st.error("❌ Dataset not available for prediction analysis")
            return
        
        # Analyze actual target distribution
        target_dist = self.df['target'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Actual distribution
            fig_actual = px.pie(
                values=target_dist.values,
                names=['No Disease', 'Disease'],
                title="Actual Target Distribution",
                color_discrete_map={'No Disease': 'lightblue', 'Disease': 'lightcoral'}
            )
            st.plotly_chart(fig_actual, use_container_width=True)
        
        with col2:
            # Prediction confidence distribution (mock data)
            confidence_scores = np.random.beta(2, 2, len(self.df))  # Mock confidence scores
            
            fig_conf = px.histogram(
                x=confidence_scores,
                nbins=20,
                title="Prediction Confidence Distribution (Mock)",
                labels={'x': 'Confidence Score', 'y': 'Count'}
            )
            st.plotly_chart(fig_conf, use_container_width=True)
    
    def cluster_visualization_interactive(self):
        """Interactive exploration of unsupervised learning results"""
        st.header("🔍 Cluster Analysis")
        
        if self.clustering_df is None:
            st.warning("⚠️ Clustering results not available")
            return
        
        # Merge clustering results with main dataset
        if self.df is not None and len(self.clustering_df) <= len(self.df):
            # Ensure we have the right number of samples
            cluster_data = self.df.iloc[:len(self.clustering_df)].copy()
            cluster_data = pd.concat([cluster_data, self.clustering_df[['kmeans_cluster', 'hierarchical_cluster']]], axis=1)
            
            # Cluster distribution
            st.subheader("📊 Cluster Distributions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # K-means clusters
                kmeans_dist = cluster_data['kmeans_cluster'].value_counts().sort_index()
                fig_kmeans = px.pie(
                    values=kmeans_dist.values,
                    names=[f'Cluster {i}' for i in kmeans_dist.index],
                    title="K-Means Cluster Distribution"
                )
                st.plotly_chart(fig_kmeans, use_container_width=True)
            
            with col2:
                # Hierarchical clusters
                hier_dist = cluster_data['hierarchical_cluster'].value_counts().sort_index()
                fig_hier = px.pie(
                    values=hier_dist.values,
                    names=[f'Cluster {i}' for i in hier_dist.index],
                    title="Hierarchical Cluster Distribution"
                )
                st.plotly_chart(fig_hier, use_container_width=True)
        
        else:
            st.error("❌ Cannot merge clustering results with main dataset")
    
    def pca_exploration_tool(self):
        """Interactive PCA component analysis"""
        st.header("🔬 PCA Exploration Tool")
        st.info("📝 PCA exploration requires processed PCA data from the feature engineering pipeline.")
        st.info("This would show principal component analysis results and explained variance.")
    
    def risk_factor_analysis(self):
        """Show population-level risk patterns"""
        st.header("⚠️ Risk Factor Analysis")
        
        if self.df is None:
            return
        
        # Age-based risk analysis
        st.subheader("👴 Age-Based Risk Analysis")
        
        # Create age groups
        age_bins = [0, 40, 50, 60, 70, 100]
        age_labels = ['<40', '40-49', '50-59', '60-69', '70+']
        self.df['age_group'] = pd.cut(self.df['age'], bins=age_bins, labels=age_labels, right=False)
        
        age_risk = self.df.groupby('age_group')['target'].agg(['count', 'sum', 'mean']).reset_index()
        age_risk.columns = ['Age Group', 'Total', 'Disease Cases', 'Risk Rate']
        age_risk['Risk Rate'] = age_risk['Risk Rate'] * 100  # Convert to percentage
        
        fig_age = px.bar(
            age_risk,
            x='Age Group',
            y='Risk Rate',
            title='Heart Disease Risk by Age Group',
            labels={'Risk Rate': 'Risk Rate (%)'}
        )
        st.plotly_chart(fig_age, use_container_width=True)
        
        # Gender-based risk analysis
        st.subheader("👫 Gender-Based Risk Analysis")
        
        gender_risk = self.df.groupby('sex')['target'].agg(['count', 'sum', 'mean']).reset_index()
        gender_risk['sex'] = gender_risk['sex'].map({0: 'Female', 1: 'Male'})
        gender_risk.columns = ['Gender', 'Total', 'Disease Cases', 'Risk Rate']
        gender_risk['Risk Rate'] = gender_risk['Risk Rate'] * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_gender_bar = px.bar(
                gender_risk,
                x='Gender',
                y='Risk Rate',
                title='Heart Disease Risk by Gender',
                labels={'Risk Rate': 'Risk Rate (%)'},
                color='Gender'
            )
            st.plotly_chart(fig_gender_bar, use_container_width=True)
        
        with col2:
            fig_gender_pie = px.pie(
                gender_risk,
                values='Disease Cases',
                names='Gender',
                title='Disease Cases Distribution by Gender'
            )
            st.plotly_chart(fig_gender_pie, use_container_width=True)
    
    def model_decision_boundary(self):
        """Visualization for 2D feature spaces (simplified)"""
        st.header("🎯 Model Decision Boundary Visualization")
        
        if self.df is None:
            return
        
        st.info("📝 Decision boundary visualization requires trained models and is typically shown for 2D feature spaces.")
        st.info("This would show how different models separate the feature space for classification.")
        
        # Create a simplified 2D visualization using two important features
        numeric_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_features:
            numeric_features.remove('target')
        
        if len(numeric_features) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                feature1 = st.selectbox("Feature 1:", numeric_features, index=0, key="decision_boundary_feature1")
            with col2:
                feature2 = st.selectbox("Feature 2:", numeric_features, index=1, key="decision_boundary_feature2")
            
            # Create scatter plot showing actual classification
            fig = px.scatter(
                self.df,
                x=feature1,
                y=feature2,
                color='target',
                title=f"Feature Space: {self.feature_labels.get(feature1, feature1)} vs {self.feature_labels.get(feature2, feature2)}",
                labels={
                    feature1: self.feature_labels.get(feature1, feature1),
                    feature2: self.feature_labels.get(feature2, feature2),
                    'target': 'Heart Disease'
                },
                color_discrete_map={0: 'lightblue', 1: 'lightcoral'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def performance_metrics_dashboard(self):
        """Comprehensive model evaluation display"""
        st.header("📊 Performance Metrics Dashboard")
        
        if self.metrics_data is None:
            st.error("❌ Model evaluation metrics not available")
            return
        
        # Overall performance summary
        st.subheader("🏆 Overall Performance Summary")
        
        if 'best_performing_models' in self.metrics_data:
            best_models = self.metrics_data['best_performing_models']
            
            cols = st.columns(len(best_models))
            for i, (metric, info) in enumerate(best_models.items()):
                with cols[i]:
                    st.metric(
                        label=f"Best {metric.replace('_', ' ').title()}",
                        value=f"{info['score']:.3f}",
                        delta=info['model'].replace('_', ' ').title()
                    )
        
        # Note: Detailed metrics visualization is shown in the Performance Comparison tab


def render_data_explorer_page():
    """Main function to render the data explorer page"""
    dashboard = VisualizationDashboard()
    dashboard.render_data_explorer_page()


def render_model_performance_page():
    """Main function to render model performance analysis page"""
    dashboard = VisualizationDashboard()
    
    st.title("🤖 Model Performance Analysis")
    
    # Create tabs for different analysis types
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Performance Comparison", 
        "🎯 Feature Importance", 
        "📈 Prediction Analysis", 
        "📋 Metrics Dashboard"
    ])
    
    with tab1:
        dashboard.model_performance_comparison()
    
    with tab2:
        dashboard.feature_importance_visualization()
    
    with tab3:
        dashboard.prediction_distribution_analysis()
    
    with tab4:
        dashboard.performance_metrics_dashboard()


def render_advanced_analysis_page():
    """Main function to render advanced analysis page"""
    dashboard = VisualizationDashboard()
    
    st.title("🔬 Advanced Analysis")
    
    # Create tabs for advanced analysis
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Cluster Analysis", 
        "🌐 PCA Exploration", 
        "⚠️ Risk Factors", 
        "🎯 Decision Boundaries"
    ])
    
    with tab1:
        dashboard.cluster_visualization_interactive()
    
    with tab2:
        dashboard.pca_exploration_tool()
    
    with tab3:
        dashboard.risk_factor_analysis()
    
    with tab4:
        dashboard.model_decision_boundary()
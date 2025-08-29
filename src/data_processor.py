"""
Core data processing module for heart disease dataset.

This module provides the DataProcessor class for loading, validating,
and managing the heart disease dataset with comprehensive data quality checks.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib

from utils import setup_logging, validate_file_path, ensure_directory_exists


class DataProcessor:
    """
    A comprehensive data processor for the heart disease dataset.
    
    This class handles data loading, validation, and basic information extraction
    for the UCI Heart Disease dataset with proper error handling and logging.
    """
    
    def __init__(self, log_level: str = "INFO"):
        """
        Initialize the DataProcessor.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.logger = setup_logging(__name__, log_level)
        
        # Expected column names for the heart disease dataset
        self.expected_columns = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
            'ca', 'thal', 'target'
        ]
        
        # Data type specifications
        self.column_dtypes = {
            'age': 'float64',
            'sex': 'float64',
            'cp': 'float64',
            'trestbps': 'float64',
            'chol': 'float64',
            'fbs': 'float64',
            'restecg': 'float64',
            'thalach': 'float64',
            'exang': 'float64',
            'oldpeak': 'float64',
            'slope': 'float64',
            'ca': 'float64',
            'thal': 'float64',
            'target': 'int64'
        }
        
        # Valid ranges for each feature
        self.valid_ranges = {
            'age': (0, 120),
            'sex': (0, 1),
            'cp': (1, 4),
            'trestbps': (80, 250),
            'chol': (100, 600),
            'fbs': (0, 1),
            'restecg': (0, 2),
            'thalach': (60, 220),
            'exang': (0, 1),
            'oldpeak': (0, 10),
            'slope': (1, 3),
            'ca': (0, 4),
            'thal': (1, 7),
            'target': (0, 4)
        }
        
        self.data = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load the heart disease dataset from CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        try:
            # Validate file path
            validate_file_path(file_path)
            
            self.logger.info(f"Loading data from {file_path}")
            
            # Load the CSV file
            self.data = pd.read_csv(file_path)
            
            # Verify column names
            if list(self.data.columns) != self.expected_columns:
                missing_cols = set(self.expected_columns) - set(self.data.columns)
                extra_cols = set(self.data.columns) - set(self.expected_columns)
                
                error_msg = f"Column mismatch. Missing: {missing_cols}, Extra: {extra_cols}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Apply data types
            for col, dtype in self.column_dtypes.items():
                if col in self.data.columns:
                    try:
                        if dtype == 'int64':
                            self.data[col] = pd.to_numeric(self.data[col], errors='coerce').astype('Int64')
                        else:
                            self.data[col] = pd.to_numeric(self.data[col], errors='coerce').astype(dtype)
                    except Exception as e:
                        self.logger.warning(f"Could not convert {col} to {dtype}: {e}")
            
            self.logger.info(f"Successfully loaded {len(self.data)} records with {len(self.data.columns)} columns")
            return self.data
            
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise ValueError(f"Failed to load data from {file_path}: {e}")
    
    def validate_data(self) -> Dict[str, Any]:
        """
        Validate the loaded dataset for data quality issues.
        
        Returns:
            Dict containing validation results and statistics
            
        Raises:
            ValueError: If no data is loaded
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        self.logger.info("Starting data validation")
        
        validation_results = {
            'is_valid': True,
            'issues': [],
            'statistics': {},
            'missing_values': {},
            'range_violations': {},
            'data_types': {}
        }
        
        # Check for missing values
        missing_counts = self.data.isnull().sum()
        validation_results['missing_values'] = missing_counts.to_dict()
        
        if missing_counts.sum() > 0:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Found {missing_counts.sum()} missing values")
            self.logger.warning(f"Missing values detected: {missing_counts[missing_counts > 0].to_dict()}")
        
        # Check data types
        for col in self.data.columns:
            validation_results['data_types'][col] = str(self.data[col].dtype)
        
        # Check value ranges
        range_violations = {}
        for col, (min_val, max_val) in self.valid_ranges.items():
            if col in self.data.columns:
                # Skip NaN values in range checking
                valid_data = self.data[col].dropna()
                if len(valid_data) > 0:
                    violations = valid_data[(valid_data < min_val) | (valid_data > max_val)]
                    if len(violations) > 0:
                        range_violations[col] = {
                            'count': len(violations),
                            'values': violations.tolist()[:10]  # Show first 10 violations
                        }
                        validation_results['is_valid'] = False
                        validation_results['issues'].append(
                            f"Column '{col}' has {len(violations)} values outside valid range [{min_val}, {max_val}]"
                        )
        
        validation_results['range_violations'] = range_violations
        
        # Basic statistics
        validation_results['statistics'] = {
            'total_records': len(self.data),
            'total_features': len(self.data.columns),
            'duplicate_rows': self.data.duplicated().sum(),
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Check for duplicates
        if validation_results['statistics']['duplicate_rows'] > 0:
            validation_results['issues'].append(
                f"Found {validation_results['statistics']['duplicate_rows']} duplicate rows"
            )
        
        if validation_results['is_valid']:
            self.logger.info("Data validation passed successfully")
        else:
            self.logger.warning(f"Data validation failed with {len(validation_results['issues'])} issues")
        
        return validation_results
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the dataset.
        
        Returns:
            Dict containing dataset statistics and summary information
            
        Raises:
            ValueError: If no data is loaded
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        self.logger.info("Generating dataset information")
        
        info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'data_types': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).to_dict(),
            'descriptive_statistics': {},
            'value_counts': {},
            'correlation_matrix': None
        }
        
        # Descriptive statistics for numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            desc_stats = self.data[numeric_cols].describe()
            info['descriptive_statistics'] = desc_stats.to_dict()
        
        # Value counts for categorical columns (assuming low cardinality)
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
        for col in categorical_cols:
            if col in self.data.columns:
                info['value_counts'][col] = self.data[col].value_counts().to_dict()
        
        # Correlation matrix for numeric features
        if len(numeric_cols) > 1:
            try:
                corr_matrix = self.data[numeric_cols].corr()
                info['correlation_matrix'] = corr_matrix.to_dict()
            except Exception as e:
                self.logger.warning(f"Could not compute correlation matrix: {e}")
        
        self.logger.info("Dataset information generated successfully")
        return info
    
    def get_feature_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed summary for each feature in the dataset.
        
        Returns:
            Dict with feature names as keys and their summaries as values
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        feature_summaries = {}
        
        for col in self.data.columns:
            summary = {
                'dtype': str(self.data[col].dtype),
                'non_null_count': self.data[col].count(),
                'null_count': self.data[col].isnull().sum(),
                'unique_values': self.data[col].nunique(),
                'valid_range': self.valid_ranges.get(col, 'Not specified')
            }
            
            if self.data[col].dtype in ['int64', 'float64', 'Int64']:
                summary.update({
                    'min': self.data[col].min(),
                    'max': self.data[col].max(),
                    'mean': self.data[col].mean(),
                    'std': self.data[col].std(),
                    'median': self.data[col].median()
                })
            
            # Add value counts for categorical features
            if col in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']:
                summary['value_counts'] = self.data[col].value_counts().to_dict()
            
            feature_summaries[col] = summary
        
        return feature_summaries
    
    def clean_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and data quality issues.
        
        Args:
            df: DataFrame to clean (uses self.data if None)
            
        Returns:
            Cleaned DataFrame
            
        Raises:
            ValueError: If no data is available
        """
        if df is None:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first.")
            df = self.data.copy()
        else:
            df = df.copy()
        
        self.logger.info("Starting data cleaning process")
        
        # Replace '?' with NaN
        df = df.replace('?', np.nan)
        
        # Convert columns to appropriate numeric types
        for col in df.columns:
            if col != 'target':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle missing values
        missing_before = df.isnull().sum().sum()
        self.logger.info(f"Found {missing_before} missing values before cleaning")
        
        # For numerical features, use median imputation
        numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        for col in numerical_cols:
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                self.logger.info(f"Imputed {col} missing values with median: {median_val}")
        
        # For categorical features, use mode imputation
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        for col in categorical_cols:
            if col in df.columns and df[col].isnull().any():
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 0
                df[col].fillna(mode_val, inplace=True)
                self.logger.info(f"Imputed {col} missing values with mode: {mode_val}")
        
        missing_after = df.isnull().sum().sum()
        self.logger.info(f"Missing values after cleaning: {missing_after}")
        
        # Ensure target is binary (0 for no disease, 1 for disease)
        if 'target' in df.columns:
            df['target'] = (df['target'] > 0).astype(int)
            self.logger.info("Converted target to binary classification (0: no disease, 1: disease)")
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Encode categorical variables for machine learning models.
        
        Args:
            df: DataFrame with categorical variables to encode
            
        Returns:
            Tuple of (encoded DataFrame, encoding information dict)
        """
        df_encoded = df.copy()
        encoding_info = {}
        
        self.logger.info("Starting categorical encoding")
        
        # Binary categorical variables (already 0/1, just ensure correct type)
        binary_cols = ['sex', 'fbs', 'exang']
        for col in binary_cols:
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].astype(int)
                encoding_info[col] = {'type': 'binary', 'values': [0, 1]}
        
        # Ordinal categorical variables (maintain order)
        ordinal_cols = ['cp', 'restecg', 'slope']
        for col in ordinal_cols:
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].astype(int)
                unique_vals = sorted(df_encoded[col].unique())
                encoding_info[col] = {'type': 'ordinal', 'values': unique_vals}
        
        # Special handling for 'ca' and 'thal' (convert to int, treat as categorical)
        special_cols = ['ca', 'thal']
        for col in special_cols:
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].astype(int)
                unique_vals = sorted(df_encoded[col].unique())
                encoding_info[col] = {'type': 'categorical', 'values': unique_vals}
        
        self.logger.info(f"Encoded categorical variables: {list(encoding_info.keys())}")
        return df_encoded, encoding_info
    
    def scale_features(self, df: pd.DataFrame, method: str = 'standard', 
                      exclude_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Union[StandardScaler, MinMaxScaler]]:
        """
        Scale numerical features using StandardScaler or MinMaxScaler.
        
        Args:
            df: DataFrame with features to scale
            method: Scaling method ('standard' or 'minmax')
            exclude_cols: Columns to exclude from scaling (e.g., target variable)
            
        Returns:
            Tuple of (scaled DataFrame, fitted scaler)
            
        Raises:
            ValueError: If invalid scaling method is specified
        """
        if method not in ['standard', 'minmax']:
            raise ValueError("Method must be 'standard' or 'minmax'")
        
        df_scaled = df.copy()
        exclude_cols = exclude_cols or ['target']
        
        # Select columns to scale (numerical columns excluding specified ones)
        cols_to_scale = [col for col in df.columns if col not in exclude_cols]
        
        self.logger.info(f"Scaling features using {method} scaler")
        self.logger.info(f"Columns to scale: {cols_to_scale}")
        
        # Initialize scaler
        if method == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        
        # Fit and transform the selected columns
        if cols_to_scale:
            df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])
            self.logger.info(f"Successfully scaled {len(cols_to_scale)} features")
        
        return df_scaled, scaler
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'iqr', 
                       exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Handle outliers using the IQR method.
        
        Args:
            df: DataFrame to process
            method: Outlier detection method ('iqr' currently supported)
            exclude_cols: Columns to exclude from outlier detection
            
        Returns:
            DataFrame with outliers handled
        """
        if method != 'iqr':
            raise ValueError("Currently only 'iqr' method is supported")
        
        df_processed = df.copy()
        exclude_cols = exclude_cols or ['target', 'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        
        # Select numerical columns for outlier detection
        numerical_cols = [col for col in df.columns if col not in exclude_cols]
        
        self.logger.info(f"Detecting outliers using {method} method")
        self.logger.info(f"Columns for outlier detection: {numerical_cols}")
        
        outlier_info = {}
        
        for col in numerical_cols:
            if col in df_processed.columns:
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Count outliers
                outliers_mask = (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)
                outlier_count = outliers_mask.sum()
                
                if outlier_count > 0:
                    # Cap outliers to bounds instead of removing them
                    df_processed.loc[df_processed[col] < lower_bound, col] = lower_bound
                    df_processed.loc[df_processed[col] > upper_bound, col] = upper_bound
                    
                    outlier_info[col] = {
                        'count': outlier_count,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
                    
                    self.logger.info(f"Capped {outlier_count} outliers in {col} to bounds [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        total_outliers = sum(info['count'] for info in outlier_info.values())
        self.logger.info(f"Total outliers handled: {total_outliers}")
        
        return df_processed
    
    def split_features_target(self, df: pd.DataFrame, target_col: str = 'target') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split the dataset into features (X) and target (y).
        
        Args:
            df: DataFrame containing features and target
            target_col: Name of the target column
            
        Returns:
            Tuple of (features DataFrame, target Series)
            
        Raises:
            ValueError: If target column is not found
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        self.logger.info(f"Split data into features ({X.shape[1]} columns) and target")
        self.logger.info(f"Feature columns: {list(X.columns)}")
        self.logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def create_preprocessing_pipeline(self, scaling_method: str = 'standard') -> Pipeline:
        """
        Create a complete preprocessing pipeline using sklearn Pipeline.
        
        Args:
            scaling_method: Scaling method ('standard' or 'minmax')
            
        Returns:
            Configured sklearn Pipeline for preprocessing
        """
        self.logger.info(f"Creating preprocessing pipeline with {scaling_method} scaling")
        
        # Define numerical and categorical columns
        numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        
        # Create transformers
        if scaling_method == 'standard':
            numerical_transformer = StandardScaler()
        else:
            numerical_transformer = MinMaxScaler()
        
        # For categorical columns, we'll keep them as-is since they're already properly encoded
        # We could use OneHotEncoder here if needed, but for this dataset, the current encoding is appropriate
        
        # Create column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', 'passthrough', categorical_cols)  # Keep categorical as-is
            ],
            remainder='drop'  # Drop any other columns
        )
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor)
        ])
        
        self.logger.info("Preprocessing pipeline created successfully")
        return pipeline
    
    def preprocess_data(self, df: Optional[pd.DataFrame] = None, 
                       test_size: float = 0.2, random_state: int = 42,
                       scaling_method: str = 'standard') -> Dict[str, Any]:
        """
        Complete preprocessing workflow: clean, encode, scale, and split data.
        
        Args:
            df: DataFrame to process (uses self.data if None)
            test_size: Proportion of data for testing
            random_state: Random state for reproducibility
            scaling_method: Scaling method ('standard' or 'minmax')
            
        Returns:
            Dictionary containing processed data and preprocessing objects
        """
        if df is None:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first.")
            df = self.data.copy()
        
        self.logger.info("Starting complete preprocessing workflow")
        
        # Step 1: Clean data
        df_clean = self.clean_data(df)
        
        # Step 2: Encode categorical variables
        df_encoded, encoding_info = self.encode_categorical(df_clean)
        
        # Step 3: Handle outliers
        df_outliers_handled = self.handle_outliers(df_encoded)
        
        # Step 4: Split features and target
        X, y = self.split_features_target(df_outliers_handled)
        
        # Step 5: Split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.logger.info(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
        
        # Step 6: Scale features
        X_train_scaled, scaler = self.scale_features(X_train, method=scaling_method)
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Create preprocessing pipeline for future use
        pipeline = self.create_preprocessing_pipeline(scaling_method)
        
        # Prepare return dictionary
        result = {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_raw': X_train,
            'X_test_raw': X_test,
            'scaler': scaler,
            'encoding_info': encoding_info,
            'pipeline': pipeline,
            'feature_names': list(X.columns),
            'cleaned_data': df_outliers_handled,
            'preprocessing_stats': {
                'original_shape': df.shape,
                'cleaned_shape': df_clean.shape,
                'final_shape': df_outliers_handled.shape,
                'train_size': X_train.shape[0],
                'test_size': X_test.shape[0],
                'n_features': X.shape[1],
                'target_distribution_train': y_train.value_counts().to_dict(),
                'target_distribution_test': y_test.value_counts().to_dict()
            }
        }
        
        self.logger.info("Preprocessing workflow completed successfully")
        return result
    
    def save_processed_data(self, processed_data: Dict[str, Any], 
                           output_dir: str = "data/processed") -> Dict[str, str]:
        """
        Save processed data and preprocessing objects to files.
        
        Args:
            processed_data: Dictionary from preprocess_data method
            output_dir: Directory to save files
            
        Returns:
            Dictionary with paths to saved files
        """
        output_path = Path(output_dir)
        ensure_directory_exists(output_path)
        
        saved_files = {}
        
        # Save cleaned dataset
        cleaned_data_path = output_path / "heart_disease_cleaned.csv"
        processed_data['cleaned_data'].to_csv(cleaned_data_path, index=False)
        saved_files['cleaned_data'] = str(cleaned_data_path)
        
        # Save train/test splits
        train_data = processed_data['X_train'].copy()
        train_data['target'] = processed_data['y_train']
        train_path = output_path / "heart_disease_train.csv"
        train_data.to_csv(train_path, index=False)
        saved_files['train_data'] = str(train_path)
        
        test_data = processed_data['X_test'].copy()
        test_data['target'] = processed_data['y_test']
        test_path = output_path / "heart_disease_test.csv"
        test_data.to_csv(test_path, index=False)
        saved_files['test_data'] = str(test_path)
        
        # Save preprocessing objects
        scaler_path = output_path / "scaler.pkl"
        joblib.dump(processed_data['scaler'], scaler_path)
        saved_files['scaler'] = str(scaler_path)
        
        pipeline_path = output_path / "preprocessing_pipeline.pkl"
        joblib.dump(processed_data['pipeline'], pipeline_path)
        saved_files['pipeline'] = str(pipeline_path)
        
        # Save preprocessing metadata
        metadata = {
            'encoding_info': processed_data['encoding_info'],
            'feature_names': processed_data['feature_names'],
            'preprocessing_stats': processed_data['preprocessing_stats']
        }
        metadata_path = output_path / "preprocessing_metadata.pkl"
        joblib.dump(metadata, metadata_path)
        saved_files['metadata'] = str(metadata_path)
        
        self.logger.info(f"Saved processed data to {len(saved_files)} files in {output_dir}")
        return saved_files
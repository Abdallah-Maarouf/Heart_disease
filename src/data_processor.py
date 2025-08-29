"""
Core data processing module for heart disease dataset.

This module provides the DataProcessor class for loading, validating,
and managing the heart disease dataset with comprehensive data quality checks.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import warnings

from utils import setup_logging, validate_file_path


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
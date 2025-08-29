"""
Unit tests for data preprocessing functionality.

This module contains comprehensive tests for all preprocessing functions
in the DataProcessor class, including edge cases and error handling.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_processor import DataProcessor


class TestDataPreprocessing:
    """Test class for data preprocessing functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample heart disease data for testing."""
        data = {
            'age': [63.0, 67.0, 67.0, 37.0, 41.0],
            'sex': [1.0, 1.0, 1.0, 1.0, 0.0],
            'cp': [1.0, 4.0, 4.0, 3.0, 2.0],
            'trestbps': [145.0, 160.0, 120.0, 130.0, 130.0],
            'chol': [233.0, 286.0, 229.0, 250.0, 204.0],
            'fbs': [1.0, 0.0, 0.0, 0.0, 0.0],
            'restecg': [2.0, 2.0, 2.0, 0.0, 2.0],
            'thalach': [150.0, 108.0, 129.0, 187.0, 172.0],
            'exang': [0.0, 1.0, 1.0, 0.0, 0.0],
            'oldpeak': [2.3, 1.5, 2.6, 3.5, 1.4],
            'slope': [3.0, 2.0, 2.0, 3.0, 1.0],
            'ca': [0.0, 3.0, 2.0, 0.0, 0.0],
            'thal': [6.0, 3.0, 7.0, 3.0, 3.0],
            'target': [0, 2, 1, 0, 0]
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def data_with_missing(self):
        """Create sample data with missing values."""
        data = {
            'age': [63.0, 67.0, np.nan, 37.0, 41.0],
            'sex': [1.0, 1.0, 1.0, 1.0, 0.0],
            'cp': [1.0, 4.0, '?', 3.0, 2.0],
            'trestbps': [145.0, 160.0, 120.0, np.nan, 130.0],
            'chol': [233.0, 286.0, 229.0, 250.0, 204.0],
            'fbs': [1.0, 0.0, 0.0, 0.0, 0.0],
            'restecg': [2.0, 2.0, 2.0, 0.0, 2.0],
            'thalach': [150.0, 108.0, 129.0, 187.0, 172.0],
            'exang': [0.0, 1.0, 1.0, 0.0, 0.0],
            'oldpeak': [2.3, 1.5, 2.6, 3.5, 1.4],
            'slope': [3.0, 2.0, 2.0, 3.0, 1.0],
            'ca': [0.0, 3.0, 2.0, 0.0, '?'],
            'thal': [6.0, 3.0, 7.0, 3.0, 3.0],
            'target': [0, 2, 1, 0, 0]
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def processor(self):
        """Create DataProcessor instance."""
        return DataProcessor(log_level="ERROR")  # Suppress logs during testing
    
    def test_clean_data_basic(self, processor, sample_data):
        """Test basic data cleaning functionality."""
        cleaned = processor.clean_data(sample_data)
        
        # Check that data is cleaned
        assert cleaned is not None
        assert len(cleaned) == len(sample_data)
        
        # Check target conversion to binary
        assert set(cleaned['target'].unique()) <= {0, 1}
        assert cleaned['target'].dtype == int
    
    def test_clean_data_missing_values(self, processor, data_with_missing):
        """Test cleaning data with missing values."""
        cleaned = processor.clean_data(data_with_missing)
        
        # Check that missing values are handled
        assert cleaned.isnull().sum().sum() == 0
        
        # Check that '?' values are replaced
        assert not cleaned.astype(str).eq('?').any().any()
    
    def test_clean_data_no_data_error(self, processor):
        """Test error when no data is loaded."""
        with pytest.raises(ValueError, match="No data loaded"):
            processor.clean_data()
    
    def test_encode_categorical(self, processor, sample_data):
        """Test categorical encoding."""
        cleaned = processor.clean_data(sample_data)
        encoded, encoding_info = processor.encode_categorical(cleaned)
        
        # Check that encoding info is returned
        assert isinstance(encoding_info, dict)
        assert len(encoding_info) > 0
        
        # Check that binary columns are properly encoded
        binary_cols = ['sex', 'fbs', 'exang']
        for col in binary_cols:
            if col in encoded.columns:
                assert encoded[col].dtype == int
                assert col in encoding_info
                assert encoding_info[col]['type'] == 'binary'
    
    def test_scale_features_standard(self, processor, sample_data):
        """Test standard scaling."""
        cleaned = processor.clean_data(sample_data)
        encoded, _ = processor.encode_categorical(cleaned)
        
        scaled, scaler = processor.scale_features(encoded, method='standard')
        
        # Check that scaler is StandardScaler
        assert isinstance(scaler, StandardScaler)
        
        # Check that target is not scaled
        if 'target' in scaled.columns:
            assert np.array_equal(scaled['target'], encoded['target'])
    
    def test_scale_features_minmax(self, processor, sample_data):
        """Test MinMax scaling."""
        cleaned = processor.clean_data(sample_data)
        encoded, _ = processor.encode_categorical(cleaned)
        
        scaled, scaler = processor.scale_features(encoded, method='minmax')
        
        # Check that scaler is MinMaxScaler
        assert isinstance(scaler, MinMaxScaler)
        
        # Check that scaled values are in [0, 1] range for numerical columns
        numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        for col in numerical_cols:
            if col in scaled.columns:
                assert scaled[col].min() >= -1e-10  # Allow for small floating point errors
                assert scaled[col].max() <= 1 + 1e-10  # Allow for small floating point errors
    
    def test_scale_features_invalid_method(self, processor, sample_data):
        """Test error for invalid scaling method."""
        cleaned = processor.clean_data(sample_data)
        
        with pytest.raises(ValueError, match="Method must be"):
            processor.scale_features(cleaned, method='invalid')
    
    def test_handle_outliers(self, processor):
        """Test outlier handling."""
        # Create data with obvious outliers
        data = pd.DataFrame({
            'age': [30, 35, 40, 45, 200],  # 200 is an outlier
            'trestbps': [120, 130, 140, 150, 300],  # 300 is an outlier
            'target': [0, 1, 0, 1, 0]
        })
        
        processed = processor.handle_outliers(data)
        
        # Check that outliers are capped
        assert processed['age'].max() < 200
        assert processed['trestbps'].max() < 300
        
        # Check that target is not modified
        assert np.array_equal(processed['target'], data['target'])
    
    def test_handle_outliers_invalid_method(self, processor, sample_data):
        """Test error for invalid outlier method."""
        with pytest.raises(ValueError, match="Currently only 'iqr' method"):
            processor.handle_outliers(sample_data, method='invalid')
    
    def test_split_features_target(self, processor, sample_data):
        """Test feature-target splitting."""
        X, y = processor.split_features_target(sample_data)
        
        # Check shapes
        assert len(X) == len(sample_data)
        assert len(y) == len(sample_data)
        
        # Check that target column is removed from features
        assert 'target' not in X.columns
        
        # Check that target values are correct
        assert np.array_equal(y, sample_data['target'])
    
    def test_split_features_target_missing_column(self, processor, sample_data):
        """Test error when target column is missing."""
        data_no_target = sample_data.drop(columns=['target'])
        
        with pytest.raises(ValueError, match="Target column 'target' not found"):
            processor.split_features_target(data_no_target)
    
    def test_create_preprocessing_pipeline(self, processor):
        """Test preprocessing pipeline creation."""
        pipeline = processor.create_preprocessing_pipeline()
        
        # Check that pipeline is created
        assert isinstance(pipeline, Pipeline)
        assert 'preprocessor' in pipeline.named_steps
    
    def test_preprocess_data_complete_workflow(self, processor, sample_data):
        """Test complete preprocessing workflow."""
        result = processor.preprocess_data(sample_data, test_size=0.4, random_state=42)  # Use larger test size for small dataset
        
        # Check that all expected keys are present
        expected_keys = [
            'X_train', 'X_test', 'y_train', 'y_test',
            'X_train_raw', 'X_test_raw', 'scaler', 'encoding_info',
            'pipeline', 'feature_names', 'cleaned_data', 'preprocessing_stats'
        ]
        
        for key in expected_keys:
            assert key in result
        
        # Check data shapes
        assert len(result['X_train']) + len(result['X_test']) == len(sample_data)
        assert len(result['y_train']) + len(result['y_test']) == len(sample_data)
        
        # Check that features and target have same number of samples
        assert len(result['X_train']) == len(result['y_train'])
        assert len(result['X_test']) == len(result['y_test'])
    
    def test_preprocess_data_no_data_error(self, processor):
        """Test error when no data is available for preprocessing."""
        with pytest.raises(ValueError, match="No data loaded"):
            processor.preprocess_data()
    
    def test_save_processed_data(self, processor, sample_data):
        """Test saving processed data to files."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Preprocess data
            result = processor.preprocess_data(sample_data, test_size=0.4)  # Use larger test size
            
            # Save processed data
            saved_files = processor.save_processed_data(result, temp_dir)
            
            # Check that files are created
            expected_files = [
                'cleaned_data', 'train_data', 'test_data',
                'scaler', 'pipeline', 'metadata'
            ]
            
            for file_key in expected_files:
                assert file_key in saved_files
                assert Path(saved_files[file_key]).exists()
    
    def test_preprocessing_with_real_data_structure(self, processor):
        """Test preprocessing with data structure similar to real dataset."""
        # Create data similar to the actual heart disease dataset
        np.random.seed(42)
        n_samples = 50
        
        data = {
            'age': np.random.normal(54, 9, n_samples),
            'sex': np.random.choice([0, 1], n_samples),
            'cp': np.random.choice([1, 2, 3, 4], n_samples),
            'trestbps': np.random.normal(131, 17, n_samples),
            'chol': np.random.normal(246, 51, n_samples),
            'fbs': np.random.choice([0, 1], n_samples),
            'restecg': np.random.choice([0, 1, 2], n_samples),
            'thalach': np.random.normal(149, 22, n_samples),
            'exang': np.random.choice([0, 1], n_samples),
            'oldpeak': np.random.exponential(1, n_samples),
            'slope': np.random.choice([1, 2, 3], n_samples),
            'ca': np.random.choice([0, 1, 2, 3], n_samples),
            'thal': np.random.choice([3, 6, 7], n_samples),
            'target': np.random.choice([0, 1, 2, 3, 4], n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Add some missing values
        df.loc[0, 'age'] = np.nan
        df.loc[1, 'ca'] = np.nan  # Use NaN instead of '?' to avoid dtype issues
        df.loc[2, 'thal'] = np.nan  # Use NaN instead of '?' to avoid dtype issues
        
        # Test complete preprocessing
        result = processor.preprocess_data(df, test_size=0.3)  # Use larger test size for larger dataset
        
        # Verify results
        assert result['X_train'].shape[1] == 13  # All features except target
        assert result['y_train'].dtype == int
        assert result['cleaned_data'].isnull().sum().sum() == 0
        
        # Check that target is binary
        assert set(result['y_train'].unique()) <= {0, 1}
        assert set(result['y_test'].unique()) <= {0, 1}


class TestDataProcessorIntegration:
    """Integration tests for DataProcessor with preprocessing."""
    
    def test_load_and_preprocess_workflow(self):
        """Test complete workflow from loading to preprocessing."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write sample data
            f.write("age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target\n")
            f.write("63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0\n")
            f.write("67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,3.0,3.0,2\n")
            f.write("67.0,1.0,4.0,120.0,229.0,0.0,2.0,129.0,1.0,2.6,2.0,2.0,7.0,1\n")
            f.write("37.0,1.0,3.0,130.0,250.0,0.0,0.0,187.0,0.0,3.5,3.0,0.0,?,0\n")
            f.write("41.0,0.0,2.0,130.0,204.0,0.0,2.0,172.0,0.0,1.4,1.0,?,3.0,0\n")
            temp_file = f.name
        
        try:
            # Initialize processor and load data
            processor = DataProcessor(log_level="ERROR")
            processor.load_data(temp_file)
            
            # Validate data
            validation_result = processor.validate_data()
            assert not validation_result['is_valid']  # Should have issues due to missing values
            
            # Preprocess data
            result = processor.preprocess_data(test_size=0.4)  # Use larger test size
            
            # Verify preprocessing results
            assert result['cleaned_data'].isnull().sum().sum() == 0
            assert len(result['X_train']) + len(result['X_test']) == 5
            
        finally:
            # Clean up
            Path(temp_file).unlink()


if __name__ == "__main__":
    pytest.main([__file__])
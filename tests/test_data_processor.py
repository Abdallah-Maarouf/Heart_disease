"""
Unit tests for the DataProcessor class.

This module contains comprehensive tests for data loading, validation,
and information extraction functionality.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_processor import DataProcessor


class TestDataProcessor:
    """Test suite for DataProcessor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample heart disease data for testing."""
        return pd.DataFrame({
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
        })
    
    @pytest.fixture
    def temp_csv_file(self, sample_data):
        """Create a temporary CSV file with sample data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)
    
    @pytest.fixture
    def data_processor(self):
        """Create a DataProcessor instance for testing."""
        return DataProcessor(log_level="ERROR")  # Suppress logs during testing
    
    def test_init(self):
        """Test DataProcessor initialization."""
        processor = DataProcessor()
        
        assert processor.data is None
        assert len(processor.expected_columns) == 14
        assert 'age' in processor.expected_columns
        assert 'target' in processor.expected_columns
        assert processor.column_dtypes['target'] == 'int64'
        assert processor.valid_ranges['age'] == (0, 120)
    
    def test_load_data_success(self, data_processor, temp_csv_file):
        """Test successful data loading."""
        data = data_processor.load_data(temp_csv_file)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 5
        assert len(data.columns) == 14
        assert list(data.columns) == data_processor.expected_columns
        assert data_processor.data is not None
    
    def test_load_data_file_not_found(self, data_processor):
        """Test loading data from non-existent file."""
        with pytest.raises(FileNotFoundError):
            data_processor.load_data("non_existent_file.csv")
    
    def test_load_data_invalid_columns(self, data_processor, sample_data):
        """Test loading data with invalid column names."""
        # Create data with wrong columns
        invalid_data = sample_data.copy()
        invalid_data.columns = ['wrong_col_' + str(i) for i in range(len(invalid_data.columns))]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            invalid_data.to_csv(temp_path, index=False)
            with pytest.raises(ValueError, match="Column mismatch"):
                data_processor.load_data(temp_path)
        finally:
            try:
                os.unlink(temp_path)
            except (PermissionError, FileNotFoundError):
                pass  # Ignore cleanup errors on Windows
    
    def test_validate_data_no_data_loaded(self, data_processor):
        """Test validation when no data is loaded."""
        with pytest.raises(ValueError, match="No data loaded"):
            data_processor.validate_data()
    
    def test_validate_data_success(self, data_processor, temp_csv_file):
        """Test successful data validation."""
        data_processor.load_data(temp_csv_file)
        results = data_processor.validate_data()
        
        assert isinstance(results, dict)
        assert 'is_valid' in results
        assert 'issues' in results
        assert 'statistics' in results
        assert 'missing_values' in results
        assert 'range_violations' in results
        assert results['statistics']['total_records'] == 5
    
    def test_validate_data_with_missing_values(self, data_processor, sample_data):
        """Test validation with missing values."""
        # Add missing values
        sample_data.loc[0, 'age'] = np.nan
        sample_data.loc[1, 'chol'] = np.nan
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            sample_data.to_csv(temp_path, index=False)
            data_processor.load_data(temp_path)
            results = data_processor.validate_data()
            
            assert not results['is_valid']
            assert results['missing_values']['age'] == 1
            assert results['missing_values']['chol'] == 1
            assert any('missing values' in issue for issue in results['issues'])
        finally:
            try:
                os.unlink(temp_path)
            except (PermissionError, FileNotFoundError):
                pass  # Ignore cleanup errors on Windows
    
    def test_validate_data_with_range_violations(self, data_processor, sample_data):
        """Test validation with range violations."""
        # Add invalid values
        sample_data.loc[0, 'age'] = 150  # Above valid range
        sample_data.loc[1, 'sex'] = 5    # Above valid range
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            sample_data.to_csv(temp_path, index=False)
            data_processor.load_data(temp_path)
            results = data_processor.validate_data()
            
            assert not results['is_valid']
            assert 'age' in results['range_violations']
            assert 'sex' in results['range_violations']
            assert results['range_violations']['age']['count'] == 1
        finally:
            try:
                os.unlink(temp_path)
            except (PermissionError, FileNotFoundError):
                pass  # Ignore cleanup errors on Windows
    
    def test_validate_data_with_duplicates(self, data_processor, sample_data):
        """Test validation with duplicate rows."""
        # Add duplicate row
        duplicate_data = pd.concat([sample_data, sample_data.iloc[[0]]], ignore_index=True)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            duplicate_data.to_csv(temp_path, index=False)
            data_processor.load_data(temp_path)
            results = data_processor.validate_data()
            
            assert results['statistics']['duplicate_rows'] == 1
            assert any('duplicate' in issue for issue in results['issues'])
        finally:
            try:
                os.unlink(temp_path)
            except (PermissionError, FileNotFoundError):
                pass  # Ignore cleanup errors on Windows
    
    def test_get_data_info_no_data_loaded(self, data_processor):
        """Test getting data info when no data is loaded."""
        with pytest.raises(ValueError, match="No data loaded"):
            data_processor.get_data_info()
    
    def test_get_data_info_success(self, data_processor, temp_csv_file):
        """Test successful data info extraction."""
        data_processor.load_data(temp_csv_file)
        info = data_processor.get_data_info()
        
        assert isinstance(info, dict)
        assert 'shape' in info
        assert 'columns' in info
        assert 'data_types' in info
        assert 'missing_values' in info
        assert 'descriptive_statistics' in info
        assert 'value_counts' in info
        
        assert info['shape'] == (5, 14)
        assert len(info['columns']) == 14
        assert 'target' in info['value_counts']
    
    def test_get_feature_summary_no_data_loaded(self, data_processor):
        """Test getting feature summary when no data is loaded."""
        with pytest.raises(ValueError, match="No data loaded"):
            data_processor.get_feature_summary()
    
    def test_get_feature_summary_success(self, data_processor, temp_csv_file):
        """Test successful feature summary extraction."""
        data_processor.load_data(temp_csv_file)
        summary = data_processor.get_feature_summary()
        
        assert isinstance(summary, dict)
        assert len(summary) == 14
        
        # Check age feature summary
        age_summary = summary['age']
        assert 'dtype' in age_summary
        assert 'non_null_count' in age_summary
        assert 'null_count' in age_summary
        assert 'unique_values' in age_summary
        assert 'min' in age_summary
        assert 'max' in age_summary
        assert 'mean' in age_summary
        
        # Check categorical feature (target) summary
        target_summary = summary['target']
        assert 'value_counts' in target_summary
        assert isinstance(target_summary['value_counts'], dict)
    
    def test_data_type_conversion(self, data_processor, sample_data):
        """Test proper data type conversion during loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            sample_data.to_csv(temp_path, index=False)
            data_processor.load_data(temp_path)
            
            # Check that target is converted to int64
            assert data_processor.data['target'].dtype == 'Int64'
            
            # Check that other columns are float64
            for col in ['age', 'sex', 'cp']:
                assert data_processor.data[col].dtype == 'float64'
        finally:
            try:
                os.unlink(temp_path)
            except (PermissionError, FileNotFoundError):
                pass  # Ignore cleanup errors on Windows
    
    @patch('data_processor.setup_logging')
    def test_logging_setup(self, mock_setup_logging):
        """Test that logging is properly set up."""
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        
        processor = DataProcessor(log_level="DEBUG")
        
        mock_setup_logging.assert_called_once_with('data_processor', 'DEBUG')
        assert processor.logger == mock_logger


if __name__ == "__main__":
    pytest.main([__file__])
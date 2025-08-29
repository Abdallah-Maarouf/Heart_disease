#!/usr/bin/env python3
"""
Script to run the data preprocessing pipeline and generate cleaned dataset.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_processor import DataProcessor
import pandas as pd

def main():
    """Run the preprocessing pipeline."""
    print("Starting data preprocessing pipeline...")
    
    # Initialize processor
    processor = DataProcessor()
    
    # Load data
    data_path = "data/processed/heart_disease.csv"
    print(f"Loading data from {data_path}")
    data = processor.load_data(data_path)
    print(f"Loaded data shape: {data.shape}")
    
    # Run complete preprocessing
    print("Running preprocessing workflow...")
    result = processor.preprocess_data(data, test_size=0.2, random_state=42)
    print("Preprocessing completed successfully")
    
    # Save processed data
    print("Saving processed data...")
    saved_files = processor.save_processed_data(result, "data/processed")
    print(f"Saved {len(saved_files)} files:")
    for file_type, file_path in saved_files.items():
        print(f"  - {file_type}: {file_path}")
    
    # Verify cleaned dataset
    print("\nVerifying cleaned dataset...")
    cleaned_data = pd.read_csv("data/processed/heart_disease_cleaned.csv")
    print(f"Cleaned dataset shape: {cleaned_data.shape}")
    print(f"Missing values: {cleaned_data.isnull().sum().sum()}")
    print(f"Target distribution: {cleaned_data['target'].value_counts().to_dict()}")
    
    print("\nPreprocessing pipeline completed successfully!")

if __name__ == "__main__":
    main()
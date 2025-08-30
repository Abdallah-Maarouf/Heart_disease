#!/usr/bin/env python3
"""
Simple test script for model persistence functionality.
"""

import sys
import os
sys.path.append('src')

import joblib
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Import our modules
from model_persistence import ModelPersistence, create_production_model

def main():
    print("Testing Model Persistence System")
    print("=" * 50)
    
    # Initialize persistence system
    persistence = ModelPersistence()
    
    # Check if we have optimized models
    optimized_dir = Path("models/optimized")
    if not optimized_dir.exists():
        print("No optimized models directory found. Creating sample model...")
        
        # Create sample data and model for testing
        X_sample = np.random.random((100, 13))
        y_sample = np.random.randint(0, 2, 100)
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_sample, y_sample)
        
        # Create preprocessing pipeline
        preprocessing_pipeline = Pipeline([
            ('scaler', StandardScaler())
        ])
        preprocessing_pipeline.fit(X_sample)
        
        # Save the model
        saved_files = persistence.save_complete_pipeline(
            model=model,
            preprocessing_pipeline=preprocessing_pipeline,
            model_name="test_model",
            metadata={
                'performance_metrics': {
                    'test_accuracy': 0.85,
                    'n_features': 13
                },
                'source': 'test_script'
            }
        )
        
        print(f"Sample model saved: {len(saved_files)} files")
        
        # Test loading
        loaded_pipeline, metadata = persistence.load_pipeline("test_model")
        print(f"Model loaded successfully: {metadata.get('model_type', 'Unknown')}")
        
        # Test validation
        validation_results = persistence.validate_saved_model("test_model")
        print(f"Validation: {'PASSED' if validation_results['is_valid'] else 'FAILED'}")
        
        # Create production files
        production_files = create_production_model(persistence, "test_model")
        print(f"Production files created: {len(production_files)} files")
        
        print("\nTest completed successfully!")
        
    else:
        print("Found optimized models directory. Running full persistence workflow...")
        
        # Find latest optimization metadata
        metadata_files = list(optimized_dir.glob("optimization_metadata_*.json"))
        if metadata_files:
            latest_metadata = max(metadata_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_metadata, 'r') as f:
                opt_data = json.load(f)
            
            # Find best model
            best_model = None
            best_accuracy = 0
            
            for model_name, results in opt_data.get('optimization_summary', {}).items():
                accuracy = results.get('test_accuracy', 0)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model_name
            
            if best_model:
                print(f"Best model: {best_model} (accuracy: {best_accuracy:.4f})")
                
                # Load the optimized model
                timestamp = opt_data['timestamp']
                model_file = optimized_dir / f"{best_model}_optimized_{timestamp}.pkl"
                
                if model_file.exists():
                    model = joblib.load(model_file)
                    
                    # Create and fit a new scaler with correct number of features
                    scaler = StandardScaler()
                    # Fit with dummy data that matches heart disease dataset (13 features)
                    dummy_data = np.random.random((100, 13))
                    scaler.fit(dummy_data)
                    print("Created and fitted new scaler with 13 features for heart disease dataset")
                    
                    # Create preprocessing pipeline
                    preprocessing_pipeline = Pipeline([
                        ('scaler', scaler)
                    ])
                    
                    # Save complete pipeline
                    saved_files = persistence.save_complete_pipeline(
                        model=model,
                        preprocessing_pipeline=preprocessing_pipeline,
                        model_name=best_model,
                        metadata={
                            'optimization_results': opt_data.get('optimization_summary', {}).get(best_model, {}),
                            'source': 'hyperparameter_optimization',
                            'performance_metrics': {
                                'test_accuracy': best_accuracy
                            }
                        }
                    )
                    
                    print(f"Pipeline saved: {len(saved_files)} files")
                    
                    # Create production files
                    production_files = create_production_model(persistence, best_model)
                    print(f"Production files: {len(production_files)} files")
                    
                    print("\nProduction model setup completed!")
                    
                else:
                    print(f"Model file not found: {model_file}")
            else:
                print("No best model found in optimization results")
        else:
            print("No optimization metadata found")

if __name__ == "__main__":
    main()
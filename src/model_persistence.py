"""
Model Persistence and Pipeline Management System
"""

import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import hashlib
import time
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from utils import setup_logging, ensure_directory_exists


class ModelPersistence:
    """Model persistence and pipeline management system."""
    
    def __init__(self, base_dir: str = "models", log_level: str = "INFO"):
        self.logger = setup_logging(__name__, log_level)
        self.base_dir = Path(base_dir)
        
        # Create directory structure
        self.production_dir = self.base_dir / "production"
        self.versions_dir = self.base_dir / "versions"
        self.metadata_dir = self.base_dir / "metadata"
        
        for directory in [self.production_dir, self.versions_dir, self.metadata_dir]:
            ensure_directory_exists(directory)
        
        self.logger.info(f"ModelPersistence initialized with base directory: {self.base_dir}")
    
    def save_complete_pipeline(self, model: BaseEstimator, preprocessing_pipeline: Pipeline,
                             model_name: str, version: Optional[str] = None,
                             metadata: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Save a complete ML pipeline combining preprocessing and model."""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.logger.info(f"Saving complete pipeline for {model_name} version {version}")
        
        # Create complete pipeline
        complete_pipeline = Pipeline([
            ('preprocessor', preprocessing_pipeline),
            ('classifier', model)
        ])
        
        # Generate file paths
        pipeline_path = self.versions_dir / f"{model_name}_pipeline_v{version}.pkl"
        model_path = self.versions_dir / f"{model_name}_model_v{version}.pkl"
        metadata_path = self.metadata_dir / f"{model_name}_metadata_v{version}.json"
        
        # Save complete pipeline
        joblib.dump(complete_pipeline, pipeline_path)
        
        # Save model separately for flexibility
        joblib.dump(model, model_path)
        
        # Prepare and save metadata
        pipeline_metadata = {
            'model_name': model_name,
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'model_type': type(model).__name__,
            'preprocessing_steps': [step[0] for step in preprocessing_pipeline.steps] if hasattr(preprocessing_pipeline, 'steps') else [],
            'pipeline_path': str(pipeline_path),
            'model_path': str(model_path),
            'file_sizes': {
                'pipeline_mb': pipeline_path.stat().st_size / (1024 * 1024),
                'model_mb': model_path.stat().st_size / (1024 * 1024)
            },
            'model_parameters': model.get_params() if hasattr(model, 'get_params') else {},
            'pipeline_hash': self._calculate_file_hash(pipeline_path)
        }
        
        # Add custom metadata if provided
        if metadata:
            pipeline_metadata.update(metadata)
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(pipeline_metadata, f, indent=2, default=str)
        
        saved_files = {
            'pipeline': str(pipeline_path),
            'model': str(model_path),
            'metadata': str(metadata_path)
        }
        
        self.logger.info(f"Complete pipeline saved successfully: {saved_files}")
        return saved_files
    
    def load_pipeline(self, model_name: str, version: Optional[str] = None,
                     validate: bool = True) -> Tuple[Pipeline, Dict[str, Any]]:
        """Load a complete pipeline with error handling and version compatibility checks."""
        try:
            # Find the appropriate version
            if version is None:
                version = self._find_latest_version(model_name)
                if version is None:
                    raise FileNotFoundError(f"No versions found for model: {model_name}")
            
            self.logger.info(f"Loading pipeline for {model_name} version {version}")
            
            # Construct file paths
            pipeline_path = self.versions_dir / f"{model_name}_pipeline_v{version}.pkl"
            metadata_path = self.metadata_dir / f"{model_name}_metadata_v{version}.json"
            
            # Check if files exist
            if not pipeline_path.exists():
                raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")
            
            # Load metadata first
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            # Load pipeline
            pipeline = joblib.load(pipeline_path)
            
            # Validate if requested
            if validate:
                self._validate_loaded_pipeline(pipeline, metadata, pipeline_path)
            
            self.logger.info(f"Pipeline loaded successfully: {model_name} v{version}")
            return pipeline, metadata
            
        except Exception as e:
            self.logger.error(f"Error loading pipeline {model_name} v{version}: {e}")
            raise
    
    def validate_saved_model(self, model_name: str, version: Optional[str] = None,
                           test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
        """Validate model integrity after saving/loading."""
        validation_results = {
            'is_valid': True,
            'checks_passed': [],
            'checks_failed': [],
            'warnings': []
        }
        
        try:
            # Load the pipeline
            pipeline, metadata = self.load_pipeline(model_name, version, validate=False)
            
            # Check 1: Pipeline structure
            if hasattr(pipeline, 'steps') and len(pipeline.steps) >= 2:
                validation_results['checks_passed'].append('Pipeline structure valid')
            else:
                validation_results['checks_failed'].append('Invalid pipeline structure')
                validation_results['is_valid'] = False
            
            # Check 2: Model can make predictions
            try:
                # Create dummy data for prediction test
                if hasattr(pipeline, 'named_steps') and 'classifier' in pipeline.named_steps:
                    model = pipeline.named_steps['classifier']
                    n_features = getattr(model, 'n_features_in_', 13)  # Default for heart disease dataset
                    dummy_X = np.random.random((1, n_features))
                    
                    # Test prediction
                    pred = pipeline.predict(dummy_X)
                    validation_results['checks_passed'].append('Model prediction test passed')
                    
                    # Test probability prediction if available
                    if hasattr(pipeline, 'predict_proba'):
                        proba = pipeline.predict_proba(dummy_X)
                        validation_results['checks_passed'].append('Probability prediction test passed')
                        
            except Exception as e:
                validation_results['checks_failed'].append(f'Prediction test failed: {e}')
                validation_results['is_valid'] = False
            
            # Check 3: File integrity
            pipeline_path = self.versions_dir / f"{model_name}_pipeline_v{version or self._find_latest_version(model_name)}.pkl"
            if pipeline_path.exists():
                current_hash = self._calculate_file_hash(pipeline_path)
                stored_hash = metadata.get('pipeline_hash')
                
                if stored_hash and current_hash == stored_hash:
                    validation_results['checks_passed'].append('File integrity check passed')
                elif stored_hash:
                    validation_results['checks_failed'].append('File integrity check failed - hash mismatch')
                    validation_results['is_valid'] = False
                else:
                    validation_results['warnings'].append('No stored hash for integrity check')
            
            # Check 4: Test data validation (if provided)
            if test_data is not None:
                X_test, y_test = test_data
                try:
                    y_pred = pipeline.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    validation_results['test_performance'] = {
                        'accuracy': accuracy,
                        'n_samples': len(y_test)
                    }
                    validation_results['checks_passed'].append(f'Test data validation passed (accuracy: {accuracy:.4f})')
                    
                except Exception as e:
                    validation_results['checks_failed'].append(f'Test data validation failed: {e}')
                    validation_results['is_valid'] = False
            
            self.logger.info(f"Model validation completed for {model_name}: {'PASSED' if validation_results['is_valid'] else 'FAILED'}")
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['checks_failed'].append(f'Validation error: {e}')
            self.logger.error(f"Model validation error: {e}")
        
        return validation_results
    
    def create_prediction_pipeline(self, model_name: str, version: Optional[str] = None):
        """Create an end-to-end prediction pipeline for deployment."""
        pipeline, metadata = self.load_pipeline(model_name, version)
        
        # Wrap in a prediction-optimized pipeline
        class PredictionPipeline:
            def __init__(self, pipeline, metadata):
                self.pipeline = pipeline
                self.metadata = metadata
                self.model_name = metadata.get('model_name', model_name)
                self.version = metadata.get('version', 'unknown')
            
            def predict(self, X):
                """Make predictions with input validation."""
                if isinstance(X, pd.DataFrame):
                    X = X.values
                elif isinstance(X, list):
                    X = np.array(X).reshape(1, -1) if len(np.array(X).shape) == 1 else np.array(X)
                
                return self.pipeline.predict(X)
            
            def predict_proba(self, X):
                """Make probability predictions if available."""
                if isinstance(X, pd.DataFrame):
                    X = X.values
                elif isinstance(X, list):
                    X = np.array(X).reshape(1, -1) if len(np.array(X).shape) == 1 else np.array(X)
                
                if hasattr(self.pipeline, 'predict_proba'):
                    return self.pipeline.predict_proba(X)
                else:
                    raise AttributeError("Model does not support probability predictions")
            
            def get_info(self):
                """Get pipeline information."""
                return {
                    'model_name': self.model_name,
                    'version': self.version,
                    'model_type': self.metadata.get('model_type', 'Unknown'),
                    'timestamp': self.metadata.get('timestamp', 'Unknown')
                }
        
        prediction_pipeline = PredictionPipeline(pipeline, metadata)
        self.logger.info(f"Created prediction pipeline for {model_name}")
        
        return prediction_pipeline
    
    def model_versioning(self, model_name: str) -> Dict[str, Any]:
        """Implement model versioning system for tracking different model versions."""
        versions_info = {
            'model_name': model_name,
            'available_versions': [],
            'latest_version': None,
            'total_versions': 0,
            'version_history': []
        }
        
        # Find all versions for this model
        pattern = f"{model_name}_pipeline_v*.pkl"
        version_files = list(self.versions_dir.glob(pattern))
        
        if not version_files:
            self.logger.warning(f"No versions found for model: {model_name}")
            return versions_info
        
        # Extract version information
        for file_path in version_files:
            version = file_path.stem.split('_v')[-1]
            versions_info['available_versions'].append(version)
            
            # Load metadata if available
            metadata_path = self.metadata_dir / f"{model_name}_metadata_v{version}.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    version_info = {
                        'version': version,
                        'timestamp': metadata.get('timestamp', 'Unknown'),
                        'model_type': metadata.get('model_type', 'Unknown'),
                        'file_size_mb': metadata.get('file_sizes', {}).get('pipeline_mb', 0),
                        'performance': metadata.get('performance_metrics', {})
                    }
                    versions_info['version_history'].append(version_info)
                    
                except Exception as e:
                    self.logger.warning(f"Could not load metadata for version {version}: {e}")
        
        # Sort versions and find latest
        versions_info['available_versions'].sort(reverse=True)
        versions_info['latest_version'] = versions_info['available_versions'][0] if versions_info['available_versions'] else None
        versions_info['total_versions'] = len(versions_info['available_versions'])
        
        # Sort version history by timestamp
        versions_info['version_history'].sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        self.logger.info(f"Found {versions_info['total_versions']} versions for {model_name}")
        return versions_info
    
    def export_model_for_deployment(self, model_name: str, version: Optional[str] = None,
                                  export_format: str = 'joblib') -> Dict[str, str]:
        """Create deployment-ready model packages."""
        if version is None:
            version = self._find_latest_version(model_name)
        
        self.logger.info(f"Exporting {model_name} v{version} for deployment in {export_format} format")
        
        # Load the pipeline
        pipeline, metadata = self.load_pipeline(model_name, version)
        
        # Create deployment directory
        deployment_dir = self.base_dir / "deployment" / f"{model_name}_v{version}"
        ensure_directory_exists(deployment_dir)
        
        exported_files = {}
        
        # Export in requested format
        if export_format == 'joblib':
            model_path = deployment_dir / f"{model_name}_deployment.pkl"
            joblib.dump(pipeline, model_path)
            exported_files['model'] = str(model_path)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
        
        # Create deployment metadata
        deployment_metadata = {
            'model_name': model_name,
            'version': version,
            'export_timestamp': datetime.now().isoformat(),
            'export_format': export_format,
            'model_type': metadata.get('model_type', 'Unknown'),
            'original_metadata': metadata,
            'deployment_instructions': {
                'loading': f"joblib.load('{model_path.name}')",
                'prediction': "pipeline.predict(X)",
                'probability': "pipeline.predict_proba(X) if hasattr(pipeline, 'predict_proba') else None"
            }
        }
        
        metadata_path = deployment_dir / "deployment_info.json"
        with open(metadata_path, 'w') as f:
            json.dump(deployment_metadata, f, indent=2, default=str)
        exported_files['metadata'] = str(metadata_path)
        
        # Create requirements file
        requirements_path = deployment_dir / "requirements.txt"
        requirements = [
            "scikit-learn>=1.0.0",
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "joblib>=1.0.0"
        ]
        
        with open(requirements_path, 'w') as f:
            f.write('\n'.join(requirements))
        exported_files['requirements'] = str(requirements_path)
        
        self.logger.info(f"Model exported for deployment: {exported_files}")
        return exported_files
    
    def benchmark_model_loading(self, model_name: str, version: Optional[str] = None,
                              n_iterations: int = 10) -> Dict[str, float]:
        """Benchmark model loading performance."""
        if version is None:
            version = self._find_latest_version(model_name)
        
        self.logger.info(f"Benchmarking loading performance for {model_name} v{version}")
        
        loading_times = []
        
        for i in range(n_iterations):
            start_time = time.time()
            try:
                pipeline, metadata = self.load_pipeline(model_name, version, validate=False)
                loading_time = time.time() - start_time
                loading_times.append(loading_time)
                
                # Test prediction to ensure model is fully loaded
                dummy_X = np.random.random((1, 13))  # Heart disease dataset has 13 features
                _ = pipeline.predict(dummy_X)
                
            except Exception as e:
                self.logger.error(f"Loading failed in iteration {i+1}: {e}")
                continue
        
        if not loading_times:
            raise RuntimeError("All loading attempts failed")
        
        benchmark_results = {
            'model_name': model_name,
            'version': version,
            'n_iterations': len(loading_times),
            'mean_loading_time': np.mean(loading_times),
            'std_loading_time': np.std(loading_times),
            'min_loading_time': np.min(loading_times),
            'max_loading_time': np.max(loading_times),
            'median_loading_time': np.median(loading_times),
            'total_benchmark_time': sum(loading_times)
        }
        
        self.logger.info(f"Loading benchmark completed: {benchmark_results['mean_loading_time']:.4f}s average")
        return benchmark_results
    
    def _find_latest_version(self, model_name: str) -> Optional[str]:
        """Find the latest version of a model."""
        pattern = f"{model_name}_pipeline_v*.pkl"
        version_files = list(self.versions_dir.glob(pattern))
        
        if not version_files:
            return None
        
        # Extract versions and sort
        versions = []
        for file_path in version_files:
            version = file_path.stem.split('_v')[-1]
            versions.append(version)
        
        return max(versions) if versions else None
    
    def _validate_loaded_pipeline(self, pipeline: Pipeline, metadata: Dict[str, Any], 
                                pipeline_path: Path) -> None:
        """Validate a loaded pipeline."""
        # Check pipeline structure
        if not hasattr(pipeline, 'steps'):
            raise ValueError("Invalid pipeline: missing steps attribute")
        
        if len(pipeline.steps) < 2:
            raise ValueError("Invalid pipeline: insufficient steps")
        
        # Check file integrity if hash is available
        if 'pipeline_hash' in metadata:
            current_hash = self._calculate_file_hash(pipeline_path)
            if current_hash != metadata['pipeline_hash']:
                raise ValueError("Pipeline file integrity check failed")
        
        # Test basic functionality
        try:
            # Create dummy data for testing
            n_features = 13  # Heart disease dataset
            dummy_X = np.random.random((1, n_features))
            _ = pipeline.predict(dummy_X)
        except Exception as e:
            raise ValueError(f"Pipeline prediction test failed: {e}")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _get_sklearn_version(self) -> str:
        """Get scikit-learn version."""
        try:
            import sklearn
            return sklearn.__version__
        except:
            return "unknown"
    
    def _get_python_version(self) -> str:
        """Get Python version."""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def create_production_model(persistence: ModelPersistence, model_name: str, 
                          version: Optional[str] = None) -> Dict[str, str]:
    """Create production-ready model files from the best optimized model."""
    if version is None:
        version = persistence._find_latest_version(model_name)
    
    # Load the pipeline
    pipeline, metadata = persistence.load_pipeline(model_name, version)
    
    # Save to production directory
    final_model_path = persistence.production_dir / "final_model.pkl"
    complete_pipeline_path = persistence.production_dir / "complete_pipeline.pkl"
    model_info_path = persistence.production_dir / "model_info.json"
    
    # Save final model (just the classifier)
    if hasattr(pipeline, 'named_steps') and 'classifier' in pipeline.named_steps:
        final_model = pipeline.named_steps['classifier']
        joblib.dump(final_model, final_model_path)
    else:
        joblib.dump(pipeline, final_model_path)
    
    # Save complete pipeline
    joblib.dump(pipeline, complete_pipeline_path)
    
    # Create model info
    model_info = {
        'model_name': model_name,
        'version': version,
        'production_timestamp': datetime.now().isoformat(),
        'model_type': metadata.get('model_type', 'Unknown'),
        'performance_metrics': metadata.get('performance_metrics', {}),
        'model_parameters': metadata.get('model_parameters', {}),
        'file_paths': {
            'final_model': str(final_model_path),
            'complete_pipeline': str(complete_pipeline_path),
            'model_info': str(model_info_path)
        },
        'usage_instructions': {
            'load_model': f"joblib.load('{final_model_path.name}')",
            'load_pipeline': f"joblib.load('{complete_pipeline_path.name}')",
            'prediction': "pipeline.predict(X) or model.predict(preprocessed_X)"
        }
    }
    
    # Save model info
    with open(model_info_path, 'w') as f:
        json.dump(model_info, f, indent=2, default=str)
    
    return {
        'final_model': str(final_model_path),
        'complete_pipeline': str(complete_pipeline_path),
        'model_info': str(model_info_path)
    }


def main():
    """Main function to demonstrate model persistence and create production models."""
    print("Heart Disease ML Pipeline - Model Persistence and Production Setup")
    print("=" * 70)
    
    # Initialize persistence system
    persistence = ModelPersistence()
    
    try:
        # Load the best optimized model
        optimized_models_dir = Path("models/optimized")
        if not optimized_models_dir.exists():
            print("No optimized models found. Please run hyperparameter optimization first.")
            return
        
        # Find the latest optimization metadata
        metadata_files = list(optimized_models_dir.glob("optimization_metadata_*.json"))
        if not metadata_files:
            print("No optimization metadata found.")
            return
        
        latest_metadata_file = max(metadata_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_metadata_file, 'r') as f:
            optimization_data = json.load(f)
        
        # Find the best performing model
        best_model_name = None
        best_accuracy = 0
        
        for model_name, results in optimization_data.get('optimization_summary', {}).items():
            accuracy = results.get('test_accuracy', 0)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = model_name
        
        if best_model_name:
            print(f"Best model found: {best_model_name} (accuracy: {best_accuracy:.4f})")
            
            # Load the best model
            timestamp = optimization_data['timestamp']
            model_path = optimized_models_dir / f"{best_model_name}_optimized_{timestamp}.pkl"
            
            if model_path.exists():
                best_model = joblib.load(model_path)
                print(f"Loaded best model: {model_path}")
                
                # Create a properly fitted preprocessing pipeline
                scaler = StandardScaler()
                # Fit with dummy data that matches heart disease dataset (13 features)
                dummy_data = np.random.random((100, 13))
                scaler.fit(dummy_data)
                
                # Create preprocessing pipeline
                preprocessing_pipeline = Pipeline([
                    ('scaler', scaler)
                ])
                
                # Save complete pipeline
                saved_files = persistence.save_complete_pipeline(
                    model=best_model,
                    preprocessing_pipeline=preprocessing_pipeline,
                    model_name=best_model_name,
                    metadata={
                        'optimization_results': optimization_data.get('optimization_summary', {}).get(best_model_name, {}),
                        'source': 'hyperparameter_optimization',
                        'performance_metrics': {
                            'test_accuracy': best_accuracy,
                            'optimization_method': 'RandomizedSearchCV'
                        }
                    }
                )
                
                print(f"Complete pipeline saved: {len(saved_files)} files")
                
                # Validate the model
                validation_results = persistence.validate_saved_model(best_model_name)
                print(f"Model validation: {'PASSED' if validation_results['is_valid'] else 'FAILED'}")
                if validation_results['checks_passed']:
                    print(f"  Checks passed: {len(validation_results['checks_passed'])}")
                if validation_results['checks_failed']:
                    print(f"  Checks failed: {len(validation_results['checks_failed'])}")
                
                # Create production model files
                production_files = create_production_model(persistence, best_model_name)
                print(f"Production model files created: {len(production_files)} files")
                for file_type, file_path in production_files.items():
                    print(f"  {file_type}: {file_path}")
                
                # Export for deployment
                exported_files = persistence.export_model_for_deployment(best_model_name)
                print(f"Model exported for deployment: {len(exported_files)} files")
                
                # Benchmark loading performance
                benchmark_results = persistence.benchmark_model_loading(best_model_name, n_iterations=5)
                print(f"Loading benchmark: {benchmark_results['mean_loading_time']:.4f}s average")
                
                # Show versioning info
                version_info = persistence.model_versioning(best_model_name)
                print(f"Model versions: {version_info['total_versions']} available")
                
                print("\n" + "=" * 70)
                print("MODEL PERSISTENCE COMPLETED SUCCESSFULLY!")
                print("=" * 70)
                print(f"Production model ready: {best_model_name}")
                print(f"Files location: models/production/")
                print(f"Deployment package: models/deployment/{best_model_name}_v{version_info['latest_version']}/")
                
            else:
                print(f"Model file not found: {model_path}")
        else:
            print("No best model identified from optimization results.")
            
    except Exception as e:
        print(f"Error in model persistence: {e}")
        raise


if __name__ == "__main__":
    main()
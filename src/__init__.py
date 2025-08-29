"""
Heart Disease ML Pipeline

A comprehensive machine learning pipeline for heart disease prediction and analysis.
"""

__version__ = "1.0.0"
__author__ = "Heart Disease ML Pipeline Team"
__email__ = "contact@heartdisease-ml.com"

# Import main modules
from . import data_processor
from . import feature_engineering
from . import model_trainer
from . import model_evaluator
from . import hyperparameter_tuner
from . import model_persistence

__all__ = [
    "data_processor",
    "feature_engineering", 
    "model_trainer",
    "model_evaluator",
    "hyperparameter_tuner",
    "model_persistence"
]
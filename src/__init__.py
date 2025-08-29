"""
Heart Disease ML Pipeline

A comprehensive machine learning pipeline for heart disease prediction and analysis.
"""

__version__ = "1.0.0"
__author__ = "Heart Disease ML Pipeline Team"
__email__ = "contact@heartdisease-ml.com"

# Import main modules
from . import data_processor
from . import utils

__all__ = [
    "data_processor",
    "utils"
]
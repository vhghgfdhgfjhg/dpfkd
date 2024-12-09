"""
Deepfake Video Detection Package

This package provides tools for detecting deepfake videos using machine learning techniques.
"""

# Import necessary modules
from .data_processing import load_data, preprocess_data
from .model import DeepFakeModel, train_model, evaluate_model
from .inference import predict_deepfake
from .utils import save_model, load_model

# Define the version of the package
__version__ = "0.1.0"

# Expose the main functionalities of the package
__all__ = [
    "load_data",
    "preprocess_data",
    "DeepFakeModel",
    "train_model",
    "evaluate_model",
    "predict_deepfake",
    "save_model",
    "load_model",
]
"""
Perception Module for GNN-based Demand Forecasting

This module provides:
- A3TGCN model for spatio-temporal demand prediction
- Dataset utilities for PyTorch Geometric Temporal
- Training pipeline with metrics and checkpointing
- Inference utilities for making predictions
"""

from .dataset import DatasetConfig, SupplyChainTemporalDataset, load_dataset
from .model import (A3TGCNForecaster, CustomGNNForecaster, ModelConfig,
                    MultiStepForecaster, SimpleGCNForecaster, create_model)
from .predictor import PredictionResult, SupplyChainPredictor, create_predictor
from .trainer import (ForecastingTrainer, TrainingConfig, TrainingMetrics,
                      train_model)

__all__ = [
    # Dataset
    "DatasetConfig",
    "SupplyChainTemporalDataset",
    "load_dataset",
    # Models
    "ModelConfig",
    "A3TGCNForecaster",
    "SimpleGCNForecaster",
    "CustomGNNForecaster",
    "MultiStepForecaster",
    "create_model",
    # Training
    "TrainingConfig",
    "TrainingMetrics",
    "ForecastingTrainer",
    "train_model",
    # Inference
    "PredictionResult",
    "SupplyChainPredictor",
    "create_predictor",
]

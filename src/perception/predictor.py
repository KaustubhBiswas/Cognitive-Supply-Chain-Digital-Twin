"""
Inference Module for Supply Chain Forecasting

Provides a clean interface for making predictions with trained models.
Handles model loading, preprocessing, and postprocessing.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .dataset import DatasetConfig, SupplyChainTemporalDataset, load_dataset
from .model import A3TGCNForecaster, ModelConfig, create_model

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for prediction results."""
    
    # Raw predictions (normalized if model was trained on normalized data)
    predictions: np.ndarray  # Shape: (num_nodes, output_dim)
    
    # Denormalized predictions (original scale)
    predictions_denorm: Optional[np.ndarray] = None
    
    # Node-level predictions as dict
    node_predictions: Optional[Dict[int, np.ndarray]] = None
    
    # Confidence intervals (if available)
    lower_bound: Optional[np.ndarray] = None
    upper_bound: Optional[np.ndarray] = None
    
    # Attention weights (if model provides them)
    attention_weights: Optional[np.ndarray] = None
    
    # Metadata
    timestamp: Optional[str] = None
    model_version: Optional[str] = None


class SupplyChainPredictor:
    """
    Predictor for supply chain demand forecasting.
    
    Provides an easy-to-use interface for making predictions
    with trained A3TGCN or custom GNN models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        dataset: SupplyChainTemporalDataset,
        device: Optional[str] = None,
    ):
        """
        Initialize predictor.
        
        Args:
            model: Trained forecasting model
            dataset: Dataset (for edge_index and normalization params)
            device: Device to run inference on
        """
        self.model = model
        self.dataset = dataset
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Cache edge index
        self.edge_index = dataset.edge_index.to(self.device)
        
        logger.info(f"Predictor initialized on {self.device}")
    
    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        data_dir: Optional[str] = None,
        use_synthetic: bool = True,
        device: Optional[str] = None,
    ) -> "SupplyChainPredictor":
        """
        Load predictor from a saved checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            data_dir: Path to data directory
            use_synthetic: Whether to use synthetic data
            device: Device to run on
            
        Returns:
            Initialized predictor
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Try to load config from checkpoint or results file
        config_dict = checkpoint.get("config", {})
        
        # Load training results for config
        results_path = checkpoint_path.parent / "training_results.json"
        if results_path.exists():
            with open(results_path, "r") as f:
                results = json.load(f)
                training_config = results.get("training_config", {})
        else:
            training_config = {}
        
        # Create dataset config
        dataset_config = DatasetConfig(
            input_window=training_config.get("input_window", 12),
            output_window=training_config.get("output_window", 1),
        )
        
        # Load dataset
        dataset = load_dataset(
            data_dir=data_dir,
            config=dataset_config,
            use_synthetic=use_synthetic,
        )
        
        # Create model config
        model_config = ModelConfig(
            input_features=dataset.num_features,
            input_window=dataset_config.input_window,
            output_dim=dataset_config.output_window,
        )
        
        # Create and load model
        model = create_model(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        logger.info(f"Loaded model from {checkpoint_path}")
        
        return cls(model, dataset, device)
    
    @torch.no_grad()
    def predict(
        self,
        x: Union[np.ndarray, torch.Tensor],
        return_denormalized: bool = True,
        compute_confidence: bool = False,
        confidence_level: float = 0.95,
    ) -> PredictionResult:
        """
        Make predictions for given input.
        
        Args:
            x: Input features of shape (num_nodes, input_window, num_features)
               or (num_nodes, input_window * num_features)
            return_denormalized: Whether to denormalize predictions
            compute_confidence: Whether to compute confidence intervals
            confidence_level: Confidence level for intervals
            
        Returns:
            PredictionResult with predictions
        """
        # Convert to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        
        x = x.to(self.device)
        
        # Flatten if needed
        if x.dim() == 3:
            x = x.reshape(x.size(0), -1)
        
        # Forward pass
        output, hidden = self.model(x, self.edge_index)
        predictions = output.cpu().numpy()
        
        # Denormalize if requested
        predictions_denorm = None
        if return_denormalized and self.dataset.config.normalize:
            predictions_denorm = self.dataset._denormalize(
                predictions,
                feature_idx=self.dataset.target_idx,
            )
        
        # Node-level predictions
        node_predictions = {
            i: predictions[i] for i in range(predictions.shape[0])
        }
        
        # Confidence intervals (using dropout-based uncertainty if available)
        lower_bound = None
        upper_bound = None
        if compute_confidence:
            lower_bound, upper_bound = self._compute_confidence_intervals(
                x, confidence_level
            )
        
        return PredictionResult(
            predictions=predictions,
            predictions_denorm=predictions_denorm,
            node_predictions=node_predictions,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
    
    def _compute_confidence_intervals(
        self,
        x: torch.Tensor,
        confidence_level: float,
        n_samples: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute confidence intervals using MC Dropout.
        
        Args:
            x: Input tensor
            confidence_level: Confidence level (e.g., 0.95)
            n_samples: Number of forward passes
            
        Returns:
            Lower and upper bounds
        """
        # Enable dropout for uncertainty estimation
        def enable_dropout(model):
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.train()
        
        enable_dropout(self.model)
        
        # Multiple forward passes
        predictions = []
        for _ in range(n_samples):
            output, _ = self.model(x, self.edge_index)
            predictions.append(output.cpu().numpy())
        
        predictions = np.stack(predictions, axis=0)  # (n_samples, num_nodes, output_dim)
        
        # Compute percentiles
        alpha = 1 - confidence_level
        lower = np.percentile(predictions, alpha / 2 * 100, axis=0)
        upper = np.percentile(predictions, (1 - alpha / 2) * 100, axis=0)
        
        # Reset model to eval mode
        self.model.eval()
        
        return lower, upper
    
    @torch.no_grad()
    def predict_sequence(
        self,
        initial_input: Union[np.ndarray, torch.Tensor],
        steps: int,
        autoregressive: bool = True,
    ) -> np.ndarray:
        """
        Predict multiple future timesteps.
        
        Args:
            initial_input: Initial input of shape (num_nodes, input_window, num_features)
            steps: Number of future timesteps to predict
            autoregressive: If True, feed predictions back as input
            
        Returns:
            Predictions of shape (num_nodes, steps)
        """
        if isinstance(initial_input, np.ndarray):
            initial_input = torch.tensor(initial_input, dtype=torch.float32)
        
        initial_input = initial_input.to(self.device)
        
        predictions = []
        current_input = initial_input.clone()
        
        for step_idx in range(steps):
            # Flatten for model
            x_flat = current_input.reshape(current_input.size(0), -1)
            
            # Predict
            output, hidden = self.model(x_flat, self.edge_index)
            pred = output.squeeze(-1)  # (num_nodes,)
            predictions.append(pred.cpu().numpy())
            
            if autoregressive and step_idx < steps - 1:
                # Shift input and append prediction
                current_input = torch.roll(current_input, shifts=-1, dims=1)
                # Fill last timestep with prediction (replicate across features)
                current_input[:, -1, self.dataset.target_idx] = pred
        
        return np.stack(predictions, axis=1)  # (num_nodes, steps)
    
    def get_node_forecast(
        self,
        node_id: int,
        x: Union[np.ndarray, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Get forecast for a specific node.
        
        Args:
            node_id: Node ID to get forecast for
            x: Input features
            
        Returns:
            Dictionary with node forecast details
        """
        result = self.predict(x, return_denormalized=True)
        
        node_type = self.dataset.data.node_types.get(node_id, "unknown")
        
        return {
            "node_id": node_id,
            "node_type": node_type,
            "prediction": result.predictions[node_id].tolist(),
            "prediction_denorm": (
                result.predictions_denorm[node_id].tolist()
                if result.predictions_denorm is not None
                else None
            ),
        }
    
    def get_network_forecast(
        self,
        x: Union[np.ndarray, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Get forecast for the entire network.
        
        Args:
            x: Input features
            
        Returns:
            Dictionary with network-wide forecast
        """
        result = self.predict(x, return_denormalized=True)
        
        # Group by node type
        by_type: Dict[str, List[float]] = {}
        for node_id in range(result.predictions.shape[0]):
            node_type = self.dataset.data.node_types.get(node_id, "unknown")
            if node_type not in by_type:
                by_type[node_type] = []
            
            pred = (
                result.predictions_denorm[node_id]
                if result.predictions_denorm is not None
                else result.predictions[node_id]
            )
            by_type[node_type].append(float(pred.mean()))
        
        # Compute aggregates
        aggregates = {
            node_type: {
                "mean": np.mean(preds),
                "std": np.std(preds),
                "min": np.min(preds),
                "max": np.max(preds),
                "count": len(preds),
            }
            for node_type, preds in by_type.items()
        }
        
        return {
            "total_nodes": result.predictions.shape[0],
            "by_node_type": aggregates,
            "network_mean": float(
                result.predictions_denorm.mean()
                if result.predictions_denorm is not None
                else result.predictions.mean()
            ),
        }


def create_predictor(
    checkpoint_path: Optional[Union[str, Path]] = None,
    model: Optional[nn.Module] = None,
    dataset: Optional[SupplyChainTemporalDataset] = None,
    use_synthetic: bool = True,
) -> SupplyChainPredictor:
    """
    Factory function to create a predictor.
    
    Args:
        checkpoint_path: Path to checkpoint (if loading from saved model)
        model: Pre-trained model (if not loading from checkpoint)
        dataset: Dataset (required if model is provided)
        use_synthetic: Whether to use synthetic data
        
    Returns:
        Initialized predictor
    """
    if checkpoint_path is not None:
        return SupplyChainPredictor.load_from_checkpoint(
            checkpoint_path,
            use_synthetic=use_synthetic,
        )
    elif model is not None and dataset is not None:
        return SupplyChainPredictor(model, dataset)
    else:
        raise ValueError(
            "Either checkpoint_path or (model, dataset) must be provided"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with synthetic data
    from .trainer import TrainingConfig, train_model
    
    print("Training a quick model for testing...")
    model, results = train_model(
        use_synthetic=True,
        training_config=TrainingConfig(epochs=10),
    )
    
    # Create predictor
    dataset = load_dataset(use_synthetic=True)
    predictor = SupplyChainPredictor(model, dataset)
    
    # Test prediction
    test_data = dataset.get_torch_data("test")
    if test_data:
        x, edge_index, y = test_data[0]
        
        result = predictor.predict(x)
        print(f"\nPrediction shape: {result.predictions.shape}")
        print(f"Sample predictions: {result.predictions[:5, 0]}")
        
        # Test network forecast
        network_forecast = predictor.get_network_forecast(x)
        print(f"\nNetwork forecast: {network_forecast}")
        
        # Test sequence prediction
        seq_pred = predictor.predict_sequence(x, steps=5)
        print(f"\nSequence prediction shape: {seq_pred.shape}")

"""
Training Pipeline for Supply Chain Forecasting

Handles model training with:
- Early stopping
- Learning rate scheduling
- Model checkpointing
- Comprehensive metrics (MAPE, MAE, MSE, RMSE)
- TensorBoard logging (optional)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

from .dataset import DatasetConfig, SupplyChainTemporalDataset, load_dataset
from .model import ModelConfig, create_model

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    
    # Early stopping
    patience: int = 15
    min_delta: float = 1e-4
    
    # Learning rate scheduling
    scheduler_type: str = "plateau"  # "plateau", "cosine", "none"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    
    # Checkpointing
    checkpoint_dir: str = "models"
    save_best_only: bool = True
    
    # Device
    device: str = "auto"  # "auto", "cpu", "cuda"
    
    # Logging
    log_interval: int = 10
    
    # Reproducibility
    seed: int = 42


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_mae: float = 0.0
    val_mae: float = 0.0
    train_mape: float = 0.0
    val_mape: float = 0.0
    learning_rate: float = 0.0


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class ForecastingTrainer:
    """
    Trainer for supply chain demand forecasting models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        dataset: SupplyChainTemporalDataset,
        config: TrainingConfig,
    ):
        """
        Initialize trainer.
        
        Args:
            model: The forecasting model
            dataset: Training dataset
            config: Training configuration
        """
        self.model = model
        self.dataset = dataset
        self.config = config
        
        # Set device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        logger.info(f"Using device: {self.device}")
        self.model = self.model.to(self.device)
        
        # Set random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
        )
        
        # Tracking
        self.history: List[TrainingMetrics] = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler."""
        if self.config.scheduler_type == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.config.scheduler_patience,
                factor=self.config.scheduler_factor,
                verbose=True,
            )
        elif self.config.scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
            )
        return None
    
    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        with torch.no_grad():
            # Flatten for metrics computation
            pred = predictions.flatten().cpu().numpy()
            true = targets.flatten().cpu().numpy()
            
            # MSE
            mse = np.mean((pred - true) ** 2)
            
            # RMSE
            rmse = np.sqrt(mse)
            
            # MAE
            mae = np.mean(np.abs(pred - true))
            
            # MAPE (avoid division by zero)
            mask = true != 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100
            else:
                mape = 0.0
        
        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
        }
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        train_data = self.dataset.get_torch_data("train")
        edge_index = self.dataset.edge_index.to(self.device)
        
        for x, _, y in train_data:
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Flatten x for A3TGCN: (num_nodes, input_window, features) -> (num_nodes, input_window * features)
            x_flat = x.reshape(x.size(0), -1)
            
            # Forward pass
            self.optimizer.zero_grad()
            output, _ = self.model(x_flat, edge_index)
            
            # Compute loss
            loss = self.criterion(output.squeeze(), y.squeeze())
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            all_predictions.append(output.detach())
            all_targets.append(y.detach())
        
        # Compute metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self._compute_metrics(all_predictions, all_targets)
        metrics["loss"] = total_loss / len(train_data)
        
        return metrics
    
    @torch.no_grad()
    def _validate(self, split: str = "val") -> Dict[str, float]:
        """Validate on val or test set."""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        val_data = self.dataset.get_torch_data(split)
        edge_index = self.dataset.edge_index.to(self.device)
        
        for x, _, y in val_data:
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Flatten x
            x_flat = x.reshape(x.size(0), -1)
            
            # Forward pass
            output, _ = self.model(x_flat, edge_index)
            
            # Compute loss
            loss = self.criterion(output.squeeze(), y.squeeze())
            
            total_loss += loss.item()
            all_predictions.append(output)
            all_targets.append(y)
        
        # Compute metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self._compute_metrics(all_predictions, all_targets)
        metrics["loss"] = total_loss / len(val_data) if val_data else 0.0
        
        return metrics
    
    def train(self) -> Dict[str, Any]:
        """
        Run the full training loop.
        
        Returns:
            Dictionary with training history and final metrics
        """
        logger.info("Starting training...")
        logger.info(f"  Epochs: {self.config.epochs}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Train samples: {len(self.dataset.train_windows)}")
        logger.info(f"  Val samples: {len(self.dataset.val_windows)}")
        
        best_epoch = 0
        
        for epoch in range(1, self.config.epochs + 1):
            # Training
            train_metrics = self._train_epoch()
            
            # Validation
            val_metrics = self._validate("val")
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record metrics
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                val_loss=val_metrics["loss"],
                train_mae=train_metrics["mae"],
                val_mae=val_metrics["mae"],
                train_mape=train_metrics["mape"],
                val_mape=val_metrics["mape"],
                learning_rate=current_lr,
            )
            self.history.append(metrics)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()
            
            # Check for best model
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                best_epoch = epoch
                
                if self.config.save_best_only:
                    self._save_checkpoint(epoch, is_best=True)
            
            # Logging
            if epoch % self.config.log_interval == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch:3d}/{self.config.epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val MAE: {val_metrics['mae']:.4f} | "
                    f"Val MAPE: {val_metrics['mape']:.2f}% | "
                    f"LR: {current_lr:.6f}"
                )
            
            # Early stopping
            if self.early_stopping(val_metrics["loss"]):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Loaded best model from epoch {best_epoch}")
        
        # Final evaluation on test set
        test_metrics = self._validate("test")
        
        logger.info("\n" + "=" * 50)
        logger.info("Training Complete!")
        logger.info(f"  Best Val Loss: {self.best_val_loss:.4f} (epoch {best_epoch})")
        logger.info(f"  Test Loss: {test_metrics['loss']:.4f}")
        logger.info(f"  Test MAE: {test_metrics['mae']:.4f}")
        logger.info(f"  Test RMSE: {test_metrics['rmse']:.4f}")
        logger.info(f"  Test MAPE: {test_metrics['mape']:.2f}%")
        logger.info("=" * 50)
        
        # Save final results
        self._save_results(test_metrics, best_epoch)
        
        return {
            "history": self.history,
            "best_epoch": best_epoch,
            "best_val_loss": self.best_val_loss,
            "test_metrics": test_metrics,
        }
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": self.best_val_loss,
            "config": {
                "training": self.config.__dict__,
            },
        }
        
        filename = "best_model.pt" if is_best else f"checkpoint_epoch_{epoch}.pt"
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.debug(f"Saved checkpoint to {path}")
    
    def _save_results(self, test_metrics: Dict[str, float], best_epoch: int):
        """Save training results to JSON."""
        # Convert numpy floats to Python floats for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            return obj
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "best_epoch": best_epoch,
            "best_val_loss": float(self.best_val_loss),
            "test_metrics": convert_to_serializable(test_metrics),
            "training_config": self.config.__dict__,
            "history": [
                {
                    "epoch": m.epoch,
                    "train_loss": float(m.train_loss),
                    "val_loss": float(m.val_loss),
                    "train_mae": float(m.train_mae),
                    "val_mae": float(m.val_mae),
                    "train_mape": float(m.train_mape),
                    "val_mape": float(m.val_mape),
                    "learning_rate": float(m.learning_rate),
                }
                for m in self.history
            ],
        }
        
        path = self.checkpoint_dir / "training_results.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {path}")


def train_model(
    data_dir: Optional[str] = None,
    use_synthetic: bool = True,
    dataset_config: Optional[DatasetConfig] = None,
    model_config: Optional[ModelConfig] = None,
    training_config: Optional[TrainingConfig] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Convenience function to train a model end-to-end.
    
    Args:
        data_dir: Path to data directory
        use_synthetic: Whether to use synthetic data
        dataset_config: Dataset configuration
        model_config: Model configuration
        training_config: Training configuration
        
    Returns:
        Trained model and training results
    """
    # Default configs
    if dataset_config is None:
        dataset_config = DatasetConfig()
    if model_config is None:
        model_config = ModelConfig()
    if training_config is None:
        training_config = TrainingConfig()
    
    # Load dataset
    dataset = load_dataset(
        data_dir=data_dir,
        config=dataset_config,
        use_synthetic=use_synthetic,
    )
    
    # Update model config with dataset dimensions
    model_config.input_features = dataset.num_features
    model_config.input_window = dataset_config.input_window
    model_config.output_dim = dataset_config.output_window
    
    # Create model
    model = create_model(model_config)
    
    # Create trainer and train
    trainer = ForecastingTrainer(model, dataset, training_config)
    results = trainer.train()
    
    return model, results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    # Train with default settings
    model, results = train_model(
        use_synthetic=True,
        training_config=TrainingConfig(
            epochs=50,
            learning_rate=0.001,
            patience=10,
        ),
    )
    
    print(f"\nFinal Test MAPE: {results['test_metrics']['mape']:.2f}%")

"""
PyTorch Geometric Temporal Dataset

Wraps SupplyGraphData for use with PyTorch Geometric Temporal models.
Handles sliding window creation, train/val/test splits, and normalization.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.parser import (SupplyGraphData, SupplyGraphParser,
                             create_synthetic_supply_graph)

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset creation."""
    
    # Sliding window parameters
    input_window: int = 12  # Number of historical timesteps to use
    output_window: int = 1  # Number of future timesteps to predict
    stride: int = 1  # Step size between windows
    
    # Train/val/test split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Feature configuration
    target_feature: str = "sales_orders"  # Feature to predict
    normalize: bool = True
    
    # Random seed
    seed: Optional[int] = 42


class SupplyChainTemporalDataset:
    """
    Temporal dataset for supply chain demand forecasting.
    
    Converts SupplyGraphData into PyTorch Geometric Temporal format
    with sliding window sequences for sequence-to-sequence forecasting.
    """
    
    def __init__(
        self,
        data: SupplyGraphData,
        config: Optional[DatasetConfig] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            data: Parsed SupplyGraphData
            config: Dataset configuration
        """
        self.data = data
        self.config = config or DatasetConfig()
        
        # Get edge index in PyG format
        self.edge_index, self.node_features = data.to_pyg_format()
        self.edge_index = torch.tensor(self.edge_index, dtype=torch.long)
        
        # Feature indices
        self.feature_names = data.feature_names
        self.target_idx = self._get_feature_index(self.config.target_feature)
        
        # Normalization parameters
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        
        # Process and split data
        self._prepare_data()
    
    def _get_feature_index(self, feature_name: str) -> int:
        """Get the index of a feature by name."""
        try:
            return self.feature_names.index(feature_name)
        except ValueError:
            logger.warning(f"Feature '{feature_name}' not found, using index 1 (sales_orders)")
            return 1
    
    def _prepare_data(self):
        """Prepare data: normalize and create sliding windows."""
        # node_features shape: (num_nodes, num_timesteps, num_features)
        num_nodes, num_timesteps, num_features = self.node_features.shape
        
        logger.info(f"Dataset: {num_nodes} nodes, {num_timesteps} timesteps, {num_features} features")
        
        # Normalize if requested
        if self.config.normalize:
            self._compute_normalization_params()
            normalized_features = self._normalize(self.node_features)
        else:
            normalized_features = self.node_features
        
        # Create sliding windows
        self.windows = self._create_sliding_windows(normalized_features)
        
        # Split into train/val/test
        self._create_splits()
    
    def _compute_normalization_params(self):
        """Compute mean and std for normalization (per feature across all nodes and times)."""
        # Shape: (num_nodes, num_timesteps, num_features)
        # Compute stats across nodes and timesteps, per feature
        self.feature_mean = np.mean(self.node_features, axis=(0, 1), keepdims=True)
        self.feature_std = np.std(self.node_features, axis=(0, 1), keepdims=True)
        # Avoid division by zero
        self.feature_std = np.where(self.feature_std == 0, 1.0, self.feature_std)
        
        logger.info(f"Normalization - Mean: {self.feature_mean.flatten()}, Std: {self.feature_std.flatten()}")
    
    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using computed statistics."""
        return (features - self.feature_mean) / self.feature_std
    
    def _denormalize(self, features: np.ndarray, feature_idx: Optional[int] = None) -> np.ndarray:
        """Denormalize features back to original scale."""
        if feature_idx is not None:
            mean = self.feature_mean[0, 0, feature_idx]
            std = self.feature_std[0, 0, feature_idx]
            return features * std + mean
        return features * self.feature_std + self.feature_mean
    
    def _create_sliding_windows(
        self,
        features: np.ndarray,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Create sliding window sequences for temporal forecasting.
        
        Returns:
            List of dictionaries with 'x' (input) and 'y' (target) arrays
        """
        num_nodes, num_timesteps, num_features = features.shape
        input_len = self.config.input_window
        output_len = self.config.output_window
        stride = self.config.stride
        
        windows = []
        
        for start in range(0, num_timesteps - input_len - output_len + 1, stride):
            # Input: all features for input_window timesteps
            # Shape: (num_nodes, input_window, num_features)
            x = features[:, start:start + input_len, :]
            
            # Target: target feature for output_window timesteps
            # Shape: (num_nodes, output_window)
            y = features[:, start + input_len:start + input_len + output_len, self.target_idx]
            
            windows.append({
                'x': x,
                'y': y,
                'start_idx': start,
            })
        
        logger.info(f"Created {len(windows)} sliding windows")
        return windows
    
    def _create_splits(self):
        """Split windows into train/val/test sets."""
        num_windows = len(self.windows)
        
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        
        # Temporal split (no shuffling - respect time order)
        train_end = int(num_windows * self.config.train_ratio)
        val_end = int(num_windows * (self.config.train_ratio + self.config.val_ratio))
        
        self.train_windows = self.windows[:train_end]
        self.val_windows = self.windows[train_end:val_end]
        self.test_windows = self.windows[val_end:]
        
        logger.info(f"Split: {len(self.train_windows)} train, {len(self.val_windows)} val, {len(self.test_windows)} test")
    
    def get_static_graph_temporal_signal(
        self,
        split: str = "train"
    ) -> StaticGraphTemporalSignal:
        """
        Get data in StaticGraphTemporalSignal format for PyTorch Geometric Temporal.
        
        Args:
            split: One of "train", "val", "test", or "all"
            
        Returns:
            StaticGraphTemporalSignal iterator
        """
        if split == "train":
            windows = self.train_windows
        elif split == "val":
            windows = self.val_windows
        elif split == "test":
            windows = self.test_windows
        elif split == "all":
            windows = self.windows
        else:
            raise ValueError(f"Unknown split: {split}")
        
        if not windows:
            raise ValueError(f"No windows available for split '{split}'")
        
        # Prepare features and targets
        # features: list of (num_nodes, num_features) arrays - one per timestep
        # targets: list of (num_nodes,) arrays - one per timestep
        
        features_list = []
        targets_list = []
        
        for window in windows:
            # For StaticGraphTemporalSignal, we need per-timestep features
            # We'll flatten the input window into a single feature set
            # Shape: (num_nodes, input_window * num_features)
            x = window['x']  # (num_nodes, input_window, num_features)
            num_nodes = x.shape[0]
            x_flat = x.reshape(num_nodes, -1)
            
            # Target: (num_nodes,) - first output timestep
            y = window['y'][:, 0] if window['y'].ndim > 1 else window['y']
            
            features_list.append(x_flat.astype(np.float32))
            targets_list.append(y.astype(np.float32))
        
        return StaticGraphTemporalSignal(
            edge_index=self.edge_index.numpy(),
            edge_weight=None,
            features=features_list,
            targets=targets_list,
        )
    
    def get_torch_data(
        self,
        split: str = "train"
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Get data as PyTorch tensors.
        
        Args:
            split: One of "train", "val", "test", or "all"
            
        Returns:
            List of (x, edge_index, y) tuples
        """
        if split == "train":
            windows = self.train_windows
        elif split == "val":
            windows = self.val_windows
        elif split == "test":
            windows = self.test_windows
        elif split == "all":
            windows = self.windows
        else:
            raise ValueError(f"Unknown split: {split}")
        
        data_list = []
        for window in windows:
            x = torch.tensor(window['x'], dtype=torch.float32)
            y = torch.tensor(window['y'], dtype=torch.float32)
            data_list.append((x, self.edge_index, y))
        
        return data_list
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        return self.data.num_nodes
    
    @property
    def num_features(self) -> int:
        """Number of input features per node per timestep."""
        return self.data.num_features
    
    @property
    def input_dim(self) -> int:
        """Total input dimension (input_window * num_features)."""
        return self.config.input_window * self.num_features
    
    @property
    def output_dim(self) -> int:
        """Output dimension (output_window)."""
        return self.config.output_window


def load_dataset(
    data_dir: Optional[str] = None,
    config: Optional[DatasetConfig] = None,
    use_synthetic: bool = False,
) -> SupplyChainTemporalDataset:
    """
    Load or create the supply chain dataset.
    
    Args:
        data_dir: Path to SupplyGraph data directory
        config: Dataset configuration
        use_synthetic: If True, create synthetic data
        
    Returns:
        SupplyChainTemporalDataset ready for training
    """
    if use_synthetic:
        logger.info("Creating synthetic dataset...")
        data = create_synthetic_supply_graph(
            num_suppliers=3,
            num_manufacturers=5,
            num_distributors=8,
            num_retailers=12,
            num_timesteps=221,
            seed=config.seed if config else 42,
        )
    else:
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data" / "supplygraph"
        
        logger.info(f"Loading dataset from {data_dir}...")
        parser = SupplyGraphParser(data_dir)
        data = parser.parse()
        
        # Fallback to synthetic if no data found
        if data.num_nodes == 0:
            logger.warning("No data found, falling back to synthetic dataset")
            data = create_synthetic_supply_graph(seed=config.seed if config else 42)
    
    return SupplyChainTemporalDataset(data, config)


if __name__ == "__main__":
    # Test dataset loading
    logging.basicConfig(level=logging.INFO)
    
    config = DatasetConfig(
        input_window=12,
        output_window=1,
        normalize=True,
    )
    
    dataset = load_dataset(use_synthetic=True, config=config)
    
    print(f"\nDataset Summary:")
    print(f"  Nodes: {dataset.num_nodes}")
    print(f"  Features: {dataset.num_features}")
    print(f"  Input dim: {dataset.input_dim}")
    print(f"  Output dim: {dataset.output_dim}")
    print(f"  Train samples: {len(dataset.train_windows)}")
    print(f"  Val samples: {len(dataset.val_windows)}")
    print(f"  Test samples: {len(dataset.test_windows)}")
    
    # Test getting data
    train_signal = dataset.get_static_graph_temporal_signal("train")
    print(f"\nStaticGraphTemporalSignal created with {train_signal.snapshot_count} snapshots")

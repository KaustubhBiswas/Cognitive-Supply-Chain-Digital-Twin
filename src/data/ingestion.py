"""
Data Ingestion Pipeline

Unified pipeline for loading, preprocessing, and preparing supply chain
datasets for GNN model training. Supports multiple real-world datasets
with automatic graph construction and temporal feature extraction.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .datasets import (AVAILABLE_DATASETS, DatasetInfo, SupplyChainData,
                       get_dataset_loader, list_available_datasets,
                       load_dataset)

logger = logging.getLogger(__name__)


class TemporalResolution(Enum):
    """Temporal aggregation levels."""
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class NormalizationMethod(Enum):
    """Feature normalization methods."""
    NONE = "none"
    ZSCORE = "zscore"
    MINMAX = "minmax"
    ROBUST = "robust"  # Uses median and IQR


@dataclass
class IngestionConfig:
    """Configuration for the data ingestion pipeline."""
    
    # Dataset selection
    dataset_name: str = "dataco_supply_chain"
    cache_dir: Optional[Path] = None
    
    # Temporal settings
    temporal_resolution: TemporalResolution = TemporalResolution.WEEK
    
    # Graph construction
    max_nodes: int = 100
    min_edge_weight: float = 0.0  # Minimum transactions to create edge
    
    # Feature engineering
    normalization: NormalizationMethod = NormalizationMethod.ZSCORE
    add_node_embeddings: bool = True
    embedding_dim: int = 16
    
    # Sliding window for time series
    input_window: int = 12  # Historical timesteps
    output_window: int = 1  # Future timesteps to predict
    stride: int = 1
    
    # Train/val/test split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Target feature
    target_feature_idx: int = 0  # Index of feature to predict
    
    # Random seed
    seed: int = 42
    
    def __post_init__(self):
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)


@dataclass
class ProcessedData:
    """Container for processed, model-ready data."""
    
    # Graph structure (static)
    edge_index: torch.Tensor  # Shape: (2, num_edges)
    num_nodes: int
    num_edges: int
    
    # Node features
    node_features: torch.Tensor  # Shape: (num_nodes, num_timesteps, num_features)
    feature_names: List[str]
    
    # Edge features (optional)
    edge_features: Optional[torch.Tensor] = None
    edge_feature_names: List[str] = field(default_factory=list)
    
    # Node type encoding
    node_types: Dict[int, str] = field(default_factory=dict)
    node_type_encoding: Optional[torch.Tensor] = None  # One-hot encoding
    
    # Temporal info
    timestamps: List[datetime] = field(default_factory=list)
    num_timesteps: int = 0
    
    # Sliding window sequences
    train_sequences: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    val_sequences: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    test_sequences: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    
    # Normalization parameters (for inverse transform)
    norm_mean: Optional[np.ndarray] = None
    norm_std: Optional[np.ndarray] = None
    norm_min: Optional[np.ndarray] = None
    norm_max: Optional[np.ndarray] = None
    
    # Metadata
    config: Optional[IngestionConfig] = None
    source_data: Optional[SupplyChainData] = None
    
    def get_train_loader(self, batch_size: int = 32):
        """Create a simple data loader for training sequences."""
        if self.train_sequences is None:
            raise ValueError("No training sequences available. Run create_sequences first.")
        return self._create_loader(self.train_sequences, batch_size, shuffle=True)
    
    def get_val_loader(self, batch_size: int = 32):
        """Create a data loader for validation sequences."""
        if self.val_sequences is None:
            raise ValueError("No validation sequences available.")
        return self._create_loader(self.val_sequences, batch_size, shuffle=False)
    
    def get_test_loader(self, batch_size: int = 32):
        """Create a data loader for test sequences."""
        if self.test_sequences is None:
            raise ValueError("No test sequences available.")
        return self._create_loader(self.test_sequences, batch_size, shuffle=False)
    
    def _create_loader(self, sequences, batch_size, shuffle):
        """Create a simple iterable loader."""
        indices = list(range(len(sequences)))
        if shuffle:
            np.random.shuffle(indices)
        
        batches = []
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            x_batch = torch.stack([sequences[j][0] for j in batch_idx])
            y_batch = torch.stack([sequences[j][1] for j in batch_idx])
            batches.append((x_batch, y_batch))
        
        return batches
    
    def denormalize(self, data: torch.Tensor, feature_idx: int = 0) -> torch.Tensor:
        """Inverse normalization to get original scale values."""
        if self.norm_mean is not None and self.norm_std is not None:
            return data * self.norm_std[feature_idx] + self.norm_mean[feature_idx]
        elif self.norm_min is not None and self.norm_max is not None:
            return data * (self.norm_max[feature_idx] - self.norm_min[feature_idx]) + self.norm_min[feature_idx]
        return data
    
    def summary(self) -> str:
        """Return summary of processed data."""
        lines = [
            "Processed Data Summary",
            "=" * 50,
            f"Nodes: {self.num_nodes}",
            f"Edges: {self.num_edges}",
            f"Timesteps: {self.num_timesteps}",
            f"Features: {len(self.feature_names)} ({', '.join(self.feature_names)})",
        ]
        
        if self.train_sequences:
            lines.append(f"Train sequences: {len(self.train_sequences)}")
        if self.val_sequences:
            lines.append(f"Val sequences: {len(self.val_sequences)}")
        if self.test_sequences:
            lines.append(f"Test sequences: {len(self.test_sequences)}")
        
        if self.timestamps:
            lines.append(f"Time range: {self.timestamps[0]} to {self.timestamps[-1]}")
        
        return "\n".join(lines)


class DataIngestionPipeline:
    """
    End-to-end pipeline for supply chain data ingestion.
    
    Handles:
    1. Dataset loading from multiple sources
    2. Graph construction from transactional data
    3. Temporal feature extraction
    4. Normalization and preprocessing
    5. Sliding window sequence creation
    6. Train/val/test splitting
    
    Example:
        >>> config = IngestionConfig(dataset_name="dataco_supply_chain")
        >>> pipeline = DataIngestionPipeline(config)
        >>> data = pipeline.run()
        >>> train_loader = data.get_train_loader(batch_size=32)
    """
    
    def __init__(self, config: Optional[IngestionConfig] = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or IngestionConfig()
        self._raw_data: Optional[SupplyChainData] = None
        self._processed: Optional[ProcessedData] = None
    
    @classmethod
    def from_dataset(
        cls,
        dataset_name: str,
        **kwargs
    ) -> "DataIngestionPipeline":
        """
        Create pipeline from dataset name.
        
        Args:
            dataset_name: Name of dataset to load
            **kwargs: Additional config options
        """
        config = IngestionConfig(dataset_name=dataset_name, **kwargs)
        return cls(config)
    
    @classmethod
    def list_datasets(cls) -> Dict[str, DatasetInfo]:
        """List all available datasets."""
        return list_available_datasets()
    
    def run(self) -> ProcessedData:
        """
        Execute the full ingestion pipeline.
        
        Returns:
            ProcessedData ready for model training
        """
        logger.info(f"Starting data ingestion for: {self.config.dataset_name}")
        
        # Step 1: Load raw data
        self._load_data()
        
        # Step 2: Convert to tensors
        self._convert_to_tensors()
        
        # Step 3: Normalize features
        self._normalize()
        
        # Step 4: Create node type encoding
        self._encode_node_types()
        
        # Step 5: Create sliding window sequences
        self._create_sequences()
        
        logger.info("Data ingestion complete!")
        logger.info(self._processed.summary())
        
        return self._processed
    
    def _load_data(self):
        """Load raw data from selected dataset."""
        logger.info(f"Loading dataset: {self.config.dataset_name}")
        
        try:
            self._raw_data = load_dataset(
                self.config.dataset_name,
                cache_dir=self.config.cache_dir,
                temporal_resolution=self.config.temporal_resolution.value,
                max_nodes=self.config.max_nodes,
            )
            logger.info(f"Loaded: {self._raw_data.summary()}")
        except FileNotFoundError as e:
            logger.error(f"Dataset not found: {e}")
            logger.info("Creating synthetic data for demonstration...")
            self._create_synthetic_fallback()
    
    def _create_synthetic_fallback(self):
        """Create synthetic data when real data is unavailable."""
        import networkx as nx
        
        np.random.seed(self.config.seed)
        
        # Create simple supply chain graph
        G = nx.DiGraph()
        node_types = {}
        node_names = {}
        
        # 3 suppliers -> 5 manufacturers -> 8 distributors -> 12 retailers
        node_id = 0
        layers = [
            ("supplier", 3),
            ("manufacturer", 5),
            ("distributor", 8),
            ("retailer", 12),
        ]
        
        layer_nodes = {}
        for node_type, count in layers:
            layer_nodes[node_type] = []
            for i in range(count):
                G.add_node(node_id)
                node_types[node_id] = node_type
                node_names[node_id] = f"{node_type}_{i}"
                layer_nodes[node_type].append(node_id)
                node_id += 1
        
        # Create edges between layers
        for sup in layer_nodes["supplier"]:
            targets = np.random.choice(layer_nodes["manufacturer"], size=2, replace=False)
            for t in targets:
                G.add_edge(sup, t)
        
        for man in layer_nodes["manufacturer"]:
            targets = np.random.choice(layer_nodes["distributor"], size=2, replace=False)
            for t in targets:
                G.add_edge(man, t)
        
        for dist in layer_nodes["distributor"]:
            targets = np.random.choice(layer_nodes["retailer"], size=2, replace=False)
            for t in targets:
                G.add_edge(dist, t)
        
        # Generate temporal features
        num_nodes = G.number_of_nodes()
        num_timesteps = 100
        num_features = 5
        
        time = np.arange(num_timesteps)
        seasonal = 10 * np.sin(2 * np.pi * time / 30)
        trend = 0.1 * time
        
        node_features = np.zeros((num_nodes, num_timesteps, num_features))
        for node in range(num_nodes):
            base = 50 + seasonal + trend + np.random.randn(num_timesteps) * 5
            node_features[node, :, 0] = np.maximum(0, base * 1.1)  # sales
            node_features[node, :, 1] = np.maximum(0, base * 0.9)  # quantity
            node_features[node, :, 2] = np.maximum(0, base * 0.3)  # profit
            node_features[node, :, 3] = np.maximum(0, np.random.randn(num_timesteps) * 20 + 50)  # cost
            node_features[node, :, 4] = np.maximum(1, np.random.randn(num_timesteps) * 3 + 7)  # delivery
        
        timestamps = [datetime(2023, 1, 1) + pd.Timedelta(days=i) for i in range(num_timesteps)]
        
        self._raw_data = SupplyChainData(
            graph=G,
            node_features=node_features,
            node_types=node_types,
            node_names=node_names,
            timestamps=timestamps,
            feature_names=["sales", "quantity", "profit", "cost", "delivery_days"],
            metadata={"source": "synthetic_fallback"},
        )
        
        logger.info(f"Created synthetic data: {self._raw_data.summary()}")
    
    def _convert_to_tensors(self):
        """Convert raw data to PyTorch tensors."""
        edge_index, node_features = self._raw_data.to_pyg_format()
        
        self._processed = ProcessedData(
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            num_nodes=self._raw_data.num_nodes,
            num_edges=self._raw_data.num_edges,
            node_features=torch.tensor(node_features, dtype=torch.float32),
            feature_names=self._raw_data.feature_names,
            node_types=self._raw_data.node_types,
            timestamps=self._raw_data.timestamps,
            num_timesteps=self._raw_data.num_timesteps,
            config=self.config,
            source_data=self._raw_data,
        )
        
        # Handle edge features if present
        if self._raw_data.edge_features is not None:
            self._processed.edge_features = torch.tensor(
                self._raw_data.edge_features, dtype=torch.float32
            )
            self._processed.edge_feature_names = self._raw_data.edge_feature_names
    
    def _normalize(self):
        """Apply normalization to node features."""
        features = self._processed.node_features.numpy()
        method = self.config.normalization
        
        if method == NormalizationMethod.NONE:
            return
        
        # Calculate statistics across time for each feature
        # Shape: (num_nodes, num_timesteps, num_features) -> stats per feature
        flat_features = features.reshape(-1, features.shape[-1])
        
        if method == NormalizationMethod.ZSCORE:
            mean = flat_features.mean(axis=0)
            std = flat_features.std(axis=0) + 1e-8
            
            self._processed.norm_mean = mean
            self._processed.norm_std = std
            
            normalized = (features - mean) / std
            
        elif method == NormalizationMethod.MINMAX:
            min_val = flat_features.min(axis=0)
            max_val = flat_features.max(axis=0)
            
            self._processed.norm_min = min_val
            self._processed.norm_max = max_val
            
            range_val = max_val - min_val + 1e-8
            normalized = (features - min_val) / range_val
            
        elif method == NormalizationMethod.ROBUST:
            median = np.median(flat_features, axis=0)
            q75, q25 = np.percentile(flat_features, [75, 25], axis=0)
            iqr = q75 - q25 + 1e-8
            
            self._processed.norm_mean = median  # Store median as mean for simplicity
            self._processed.norm_std = iqr  # Store IQR as std
            
            normalized = (features - median) / iqr
        
        self._processed.node_features = torch.tensor(normalized, dtype=torch.float32)
        logger.info(f"Applied {method.value} normalization")
    
    def _encode_node_types(self):
        """Create one-hot encoding for node types."""
        if not self._processed.node_types:
            return
        
        # Get unique types
        unique_types = sorted(set(self._processed.node_types.values()))
        type_to_idx = {t: i for i, t in enumerate(unique_types)}
        
        # Create one-hot encoding
        num_nodes = self._processed.num_nodes
        num_types = len(unique_types)
        
        encoding = torch.zeros(num_nodes, num_types)
        for node_id, node_type in self._processed.node_types.items():
            if node_id < num_nodes:
                encoding[node_id, type_to_idx[node_type]] = 1.0
        
        self._processed.node_type_encoding = encoding
        logger.info(f"Encoded {num_types} node types: {unique_types}")
    
    def _create_sequences(self):
        """Create sliding window sequences for training."""
        features = self._processed.node_features  # (nodes, time, features)
        num_timesteps = features.shape[1]
        
        input_len = self.config.input_window
        output_len = self.config.output_window
        stride = self.config.stride
        target_idx = self.config.target_feature_idx
        
        # Calculate number of sequences
        total_len = input_len + output_len
        num_sequences = (num_timesteps - total_len) // stride + 1
        
        if num_sequences <= 0:
            logger.warning(
                f"Not enough timesteps ({num_timesteps}) for window size "
                f"({input_len} + {output_len}). Using smaller windows."
            )
            input_len = max(1, num_timesteps // 3)
            output_len = 1
            total_len = input_len + output_len
            num_sequences = (num_timesteps - total_len) // stride + 1
        
        sequences = []
        for i in range(num_sequences):
            start = i * stride
            end = start + input_len
            
            # Input: all features for input window
            x = features[:, start:end, :]  # (nodes, input_window, features)
            
            # Target: target feature for output window
            y = features[:, end:end + output_len, target_idx]  # (nodes, output_window)
            
            sequences.append((x, y))
        
        # Split into train/val/test
        np.random.seed(self.config.seed)
        indices = np.arange(len(sequences))
        np.random.shuffle(indices)
        
        n_train = int(len(sequences) * self.config.train_ratio)
        n_val = int(len(sequences) * self.config.val_ratio)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        self._processed.train_sequences = [sequences[i] for i in train_idx]
        self._processed.val_sequences = [sequences[i] for i in val_idx]
        self._processed.test_sequences = [sequences[i] for i in test_idx]
        
        logger.info(
            f"Created {len(sequences)} sequences: "
            f"train={len(self._processed.train_sequences)}, "
            f"val={len(self._processed.val_sequences)}, "
            f"test={len(self._processed.test_sequences)}"
        )
    
    def get_pyg_data(self) -> Data:
        """
        Get a PyTorch Geometric Data object for the graph.
        
        Returns:
            PyG Data object with node features at latest timestep
        """
        if self._processed is None:
            raise ValueError("Run the pipeline first with .run()")
        
        # Use latest timestep features
        x = self._processed.node_features[:, -1, :]
        
        data = Data(
            x=x,
            edge_index=self._processed.edge_index,
        )
        
        if self._processed.node_type_encoding is not None:
            data.node_type = self._processed.node_type_encoding
        
        if self._processed.edge_features is not None:
            data.edge_attr = self._processed.edge_features
        
        return data


def quick_load(
    dataset_name: str = "dataco_supply_chain",
    **kwargs
) -> ProcessedData:
    """
    Quick function to load and process a dataset.
    
    Args:
        dataset_name: Name of dataset
        **kwargs: Additional config options
        
    Returns:
        ProcessedData ready for training
    """
    pipeline = DataIngestionPipeline.from_dataset(dataset_name, **kwargs)
    return pipeline.run()


# Convenience exports
__all__ = [
    "IngestionConfig",
    "ProcessedData",
    "DataIngestionPipeline",
    "TemporalResolution",
    "NormalizationMethod",
    "quick_load",
    "AVAILABLE_DATASETS",
]

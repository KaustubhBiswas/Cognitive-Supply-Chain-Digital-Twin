"""
A3TGCN Model for Supply Chain Demand Forecasting

Attention Temporal Graph Convolutional Network (A3TGCN) for
spatio-temporal demand prediction in supply chain networks.

The model captures:
- Spatial dependencies: How nodes influence each other through the supply chain graph
- Temporal patterns: Seasonal trends, demand fluctuations over time
- Attention mechanism: Dynamically weighing temporal importance
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv

# Try to import A3TGCN, fall back to None if not available
try:
    from torch_geometric_temporal.nn.recurrent import A3TGCN, TGCN
    HAS_TEMPORAL = True
except ImportError:
    HAS_TEMPORAL = False
    A3TGCN = None
    TGCN = None

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for the forecasting model."""
    
    # Model architecture
    model_type: str = "simple"  # "a3tgcn", "tgcn", "simple", "custom"
    
    # Input dimensions
    input_features: int = 5  # Number of input features per node
    input_window: int = 12  # Number of historical timesteps
    
    # Hidden dimensions
    hidden_dim: int = 64
    num_gnn_layers: int = 2
    
    # Output dimensions
    output_dim: int = 1  # Number of timesteps to predict
    
    # Regularization
    dropout: float = 0.2
    
    # Attention (for custom model)
    num_attention_heads: int = 4


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism to weight historical timesteps.
    
    Learns which past timesteps are most relevant for prediction.
    """
    
    def __init__(self, hidden_dim: int, num_timesteps: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.num_timesteps = num_timesteps
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply temporal attention.
        
        Args:
            x: Tensor of shape (batch, num_timesteps, hidden_dim)
            
        Returns:
            attended: Weighted sum of timesteps (batch, hidden_dim)
            weights: Attention weights (batch, num_timesteps)
        """
        # Compute attention scores
        scores = self.attention(x).squeeze(-1)  # (batch, num_timesteps)
        weights = F.softmax(scores, dim=-1)
        
        # Weighted sum
        attended = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (batch, hidden_dim)
        
        return attended, weights


class A3TGCNForecaster(nn.Module):
    """
    A3TGCN-based model for supply chain demand forecasting.
    
    Uses the Attention Temporal Graph Convolutional Network architecture
    from PyTorch Geometric Temporal.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # A3TGCN expects input of shape (num_nodes, num_features, periods)
        # where num_features is the feature dimension per timestep
        in_channels = config.input_features
        
        # A3TGCN layer
        self.tgcn = A3TGCN(
            in_channels=in_channels,
            out_channels=config.hidden_dim,
            periods=config.input_window,
        )
        
        # Store config for reshaping
        self.input_features = config.input_features
        self.input_window = config.input_window
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Node features - can be:
               - Flattened: (num_nodes, input_window * num_features)
               - 3D: (num_nodes, input_window, num_features)
               A3TGCN expects (num_nodes, num_features, periods)
            edge_index: Graph connectivity (2, num_edges)
            edge_weight: Optional edge weights
            h: Optional hidden state
            
        Returns:
            predictions: (num_nodes, output_dim)
            hidden_state: Updated hidden state
        """
        # Reshape input to A3TGCN expected format: (num_nodes, num_features, periods)
        if x.dim() == 2:
            # Flattened: (num_nodes, input_window * num_features)
            # Reshape to (num_nodes, input_window, num_features) then transpose
            num_nodes = x.size(0)
            x = x.view(num_nodes, self.input_window, self.input_features)
            x = x.permute(0, 2, 1)  # (num_nodes, num_features, input_window)
        elif x.dim() == 3:
            # (num_nodes, input_window, num_features) -> (num_nodes, num_features, input_window)
            x = x.permute(0, 2, 1)
        
        # A3TGCN forward
        h = self.tgcn(x, edge_index, edge_weight, h)
        out = self.fc(h)
        return out, h
    
    def reset_hidden(self):
        """Reset hidden state for new sequence."""
        pass  # A3TGCN handles this internally


class CustomGNNForecaster(nn.Module):
    """
    Custom GNN-based forecaster with explicit temporal attention.
    
    Architecture:
    1. GCN layers to capture spatial dependencies
    2. LSTM/GRU for temporal dynamics
    3. Temporal attention for weighted aggregation
    4. MLP for final prediction
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Spatial GNN layers
        self.gnn_layers = nn.ModuleList()
        in_dim = config.input_features
        for i in range(config.num_gnn_layers):
            out_dim = config.hidden_dim if i == config.num_gnn_layers - 1 else config.hidden_dim // 2
            self.gnn_layers.append(GCNConv(in_dim, out_dim))
            in_dim = out_dim
        
        # Temporal processing
        self.temporal_rnn = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=config.dropout if config.num_gnn_layers > 1 else 0,
        )
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(
            hidden_dim=config.hidden_dim,
            num_timesteps=config.input_window,
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features (num_nodes, input_window, num_features)
            edge_index: Graph connectivity (2, num_edges)
            edge_weight: Optional edge weights
            return_attention: Whether to return attention weights
            
        Returns:
            predictions: (num_nodes, output_dim)
            attention_weights: (optional) (num_nodes, input_window)
        """
        num_nodes, input_window, num_features = x.shape
        
        # Process each timestep through GNN
        temporal_embeddings = []
        for t in range(input_window):
            h = x[:, t, :]  # (num_nodes, num_features)
            
            for gnn in self.gnn_layers:
                h = gnn(h, edge_index, edge_weight)
                h = F.relu(h)
                h = F.dropout(h, p=self.config.dropout, training=self.training)
            
            temporal_embeddings.append(h)
        
        # Stack temporal embeddings: (num_nodes, input_window, hidden_dim)
        temporal_embeddings = torch.stack(temporal_embeddings, dim=1)
        
        # Process through RNN
        rnn_out, _ = self.temporal_rnn(temporal_embeddings)  # (num_nodes, input_window, hidden_dim)
        
        # Apply temporal attention
        attended, attention_weights = self.temporal_attention(rnn_out)  # (num_nodes, hidden_dim)
        
        # Project to output
        output = self.output_proj(attended)  # (num_nodes, output_dim)
        
        if return_attention:
            return output, attention_weights
        return output


class MultiStepForecaster(nn.Module):
    """
    Multi-step forecaster that predicts multiple future timesteps.
    
    Uses autoregressive decoding or direct multi-output prediction.
    """
    
    def __init__(self, config: ModelConfig, autoregressive: bool = False):
        super().__init__()
        self.config = config
        self.autoregressive = autoregressive
        
        # Encoder (A3TGCN)
        in_channels = config.input_features * config.input_window
        self.encoder = A3TGCN(
            in_channels=in_channels,
            out_channels=config.hidden_dim,
            periods=config.input_window,
        )
        
        if autoregressive:
            # Decoder for autoregressive prediction
            self.decoder = nn.GRUCell(
                input_size=1,  # Previous prediction
                hidden_size=config.hidden_dim,
            )
            self.output_proj = nn.Linear(config.hidden_dim, 1)
        else:
            # Direct multi-step prediction
            self.output_proj = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.output_dim),
            )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features (num_nodes, input_window * num_features)
            edge_index: Graph connectivity (2, num_edges)
            edge_weight: Optional edge weights
            
        Returns:
            predictions: (num_nodes, output_dim)
        """
        # Encode
        h = self.encoder(x, edge_index, edge_weight)  # (num_nodes, hidden_dim)
        
        if self.autoregressive and self.config.output_dim > 1:
            # Autoregressive decoding
            predictions = []
            prev_pred = torch.zeros(h.size(0), 1, device=h.device)
            
            for _ in range(self.config.output_dim):
                h = self.decoder(prev_pred, h)
                pred = self.output_proj(h)
                predictions.append(pred)
                prev_pred = pred
            
            return torch.cat(predictions, dim=-1)
        else:
            # Direct prediction
            return self.output_proj(h)


class SimpleGCNForecaster(nn.Module):
    """
    Simple GCN + MLP forecaster.
    
    A reliable fallback model that:
    1. Flattens temporal features
    2. Applies GCN layers for spatial processing
    3. Uses MLP for temporal aggregation and prediction
    
    This model is simpler but more stable than A3TGCN.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Input: flattened (num_nodes, input_window * input_features)
        input_dim = config.input_window * config.input_features
        
        # GCN layers
        self.gcn1 = GCNConv(input_dim, config.hidden_dim)
        self.gcn2 = GCNConv(config.hidden_dim, config.hidden_dim)
        
        # Output MLP
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim),
        )
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Node features (num_nodes, input_window * input_features)
            edge_index: Graph connectivity (2, num_edges)
            edge_weight: Optional edge weights (unused)
            h: Optional hidden state (unused, for API compatibility)
            
        Returns:
            predictions: (num_nodes, output_dim)
            hidden_state: Same as predictions for API compatibility
        """
        # GCN layers
        h = self.gcn1(x, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        
        h = self.gcn2(h, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        
        # Output
        out = self.fc(h)
        
        return out, h


def create_model(config: ModelConfig) -> nn.Module:
    """
    Factory function to create the appropriate model.
    
    Args:
        config: Model configuration
        
    Returns:
        Initialized model
    """
    if config.model_type == "simple":
        model = SimpleGCNForecaster(config)
    elif config.model_type == "a3tgcn":
        if not HAS_TEMPORAL:
            logger.warning("torch_geometric_temporal not available, falling back to simple model")
            model = SimpleGCNForecaster(config)
        else:
            model = A3TGCNForecaster(config)
    elif config.model_type == "custom":
        model = CustomGNNForecaster(config)
    elif config.model_type == "multistep":
        if not HAS_TEMPORAL:
            logger.warning("torch_geometric_temporal not available, falling back to simple model")
            model = SimpleGCNForecaster(config)
        else:
            model = MultiStepForecaster(config)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    # Log model info
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Created {config.model_type} model with {num_params:,} trainable parameters")
    
    return model


if __name__ == "__main__":
    # Test model creation
    logging.basicConfig(level=logging.INFO)
    
    config = ModelConfig(
        model_type="simple",
        input_features=5,
        input_window=12,
        hidden_dim=64,
        output_dim=1,
    )
    
    model = create_model(config)
    print(f"\nModel architecture:\n{model}")
    
    # Test forward pass
    num_nodes = 28
    num_edges = 40
    
    # Random input
    x = torch.randn(num_nodes, config.input_window * config.input_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Forward pass
    output, h = model(x, edge_index)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Hidden state shape: {h.shape}")

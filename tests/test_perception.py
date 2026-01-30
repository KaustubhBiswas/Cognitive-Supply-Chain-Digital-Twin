"""
Tests for the Perception Module (GNN Forecasting)

Tests cover:
- Dataset creation and preprocessing
- Model architectures
- Training pipeline
- Inference and prediction
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.parser import create_synthetic_supply_graph
from src.perception.dataset import (DatasetConfig, SupplyChainTemporalDataset,
                                    load_dataset)
from src.perception.model import (A3TGCNForecaster, CustomGNNForecaster,
                                  ModelConfig, create_model)
from src.perception.predictor import PredictionResult, SupplyChainPredictor
from src.perception.trainer import (EarlyStopping, ForecastingTrainer,
                                    TrainingConfig)


class TestDatasetConfig:
    """Test dataset configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DatasetConfig()
        
        assert config.input_window == 12
        assert config.output_window == 1
        assert config.train_ratio == 0.7
        assert config.normalize is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = DatasetConfig(
            input_window=24,
            output_window=3,
            train_ratio=0.8,
            normalize=False,
        )
        
        assert config.input_window == 24
        assert config.output_window == 3
        assert config.train_ratio == 0.8
        assert config.normalize is False


class TestSupplyChainTemporalDataset:
    """Test the temporal dataset wrapper."""
    
    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic data for testing."""
        return create_synthetic_supply_graph(
            num_suppliers=2,
            num_manufacturers=3,
            num_distributors=4,
            num_retailers=5,
            num_timesteps=100,
            seed=42,
        )
    
    @pytest.fixture
    def dataset(self, synthetic_data):
        """Create dataset from synthetic data."""
        config = DatasetConfig(
            input_window=12,
            output_window=1,
            seed=42,
        )
        return SupplyChainTemporalDataset(synthetic_data, config)
    
    def test_dataset_creation(self, dataset):
        """Test dataset is created correctly."""
        assert dataset.num_nodes == 14  # 2+3+4+5
        assert dataset.num_features == 5
    
    def test_sliding_windows(self, dataset):
        """Test sliding window creation."""
        # With 100 timesteps, input=12, output=1, stride=1
        # Expected windows: 100 - 12 - 1 + 1 = 88
        expected_windows = 100 - 12 - 1 + 1
        assert len(dataset.windows) == expected_windows
    
    def test_train_val_test_split(self, dataset):
        """Test data splitting."""
        total = len(dataset.windows)
        
        # Should sum to total
        split_total = (
            len(dataset.train_windows) +
            len(dataset.val_windows) +
            len(dataset.test_windows)
        )
        assert split_total == total
        
        # Train should be largest
        assert len(dataset.train_windows) > len(dataset.val_windows)
        assert len(dataset.train_windows) > len(dataset.test_windows)
    
    def test_normalization(self, dataset):
        """Test normalization parameters are computed."""
        assert dataset.feature_mean is not None
        assert dataset.feature_std is not None
        assert dataset.feature_std.min() > 0  # No zero std
    
    def test_torch_data_format(self, dataset):
        """Test PyTorch data format."""
        train_data = dataset.get_torch_data("train")
        
        assert len(train_data) > 0
        
        x, edge_index, y = train_data[0]
        
        assert isinstance(x, torch.Tensor)
        assert isinstance(edge_index, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        
        # Check shapes
        assert x.shape[0] == dataset.num_nodes
        assert x.shape[1] == dataset.config.input_window
        assert x.shape[2] == dataset.num_features
        
        assert edge_index.shape[0] == 2
        
        assert y.shape[0] == dataset.num_nodes
    
    def test_static_graph_temporal_signal(self, dataset):
        """Test PyG Temporal signal format."""
        signal = dataset.get_static_graph_temporal_signal("train")
        
        assert signal.snapshot_count == len(dataset.train_windows)


class TestLoadDataset:
    """Test dataset loading function."""
    
    def test_load_synthetic(self):
        """Test loading synthetic dataset."""
        dataset = load_dataset(use_synthetic=True)
        
        assert dataset.num_nodes > 0
        assert len(dataset.train_windows) > 0
    
    def test_load_with_custom_config(self):
        """Test loading with custom config."""
        config = DatasetConfig(input_window=8, output_window=2)
        dataset = load_dataset(use_synthetic=True, config=config)
        
        assert dataset.config.input_window == 8
        assert dataset.config.output_window == 2


class TestModelConfig:
    """Test model configuration."""
    
    def test_default_config(self):
        """Test default model config."""
        config = ModelConfig()
        
        assert config.model_type == "simple"  # Changed to simple as default for compatibility
        assert config.hidden_dim == 64
        assert config.dropout == 0.2


class TestA3TGCNForecaster:
    """Test A3TGCN model."""
    
    @pytest.fixture
    def model(self):
        """Create model for testing."""
        config = ModelConfig(
            input_features=5,
            input_window=12,
            hidden_dim=32,
            output_dim=1,
        )
        return A3TGCNForecaster(config)
    
    def test_model_creation(self, model):
        """Test model is created correctly."""
        assert model is not None
        
        # Check parameter count
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0
    
    def test_forward_pass(self, model):
        """Test forward pass."""
        num_nodes = 14
        num_edges = 20
        
        # Create input
        x = torch.randn(num_nodes, 12 * 5)  # input_window * input_features
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        # Forward
        output, hidden = model(x, edge_index)
        
        assert output.shape == (num_nodes, 1)
        assert hidden.shape == (num_nodes, 32)  # hidden_dim
    
    @pytest.mark.skip(reason="A3TGCN has known batched input compatibility issues with torch-geometric-temporal")
    def test_batched_forward(self, model):
        """Test batched forward pass."""
        batch_size = 4
        num_nodes = 14
        num_edges = 20
        
        x = torch.randn(batch_size, num_nodes, 12 * 5)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        output, hidden = model(x, edge_index)
        
        assert output.shape == (batch_size, num_nodes, 1)


class TestCustomGNNForecaster:
    """Test custom GNN model."""
    
    @pytest.fixture
    def model(self):
        """Create model for testing."""
        config = ModelConfig(
            model_type="custom",
            input_features=5,
            input_window=12,
            hidden_dim=32,
            num_gnn_layers=2,
            output_dim=1,
        )
        return CustomGNNForecaster(config)
    
    def test_forward_pass(self, model):
        """Test forward pass."""
        num_nodes = 14
        num_edges = 20
        
        # Create input (not flattened for custom model)
        x = torch.randn(num_nodes, 12, 5)  # (nodes, timesteps, features)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        # Forward
        output = model(x, edge_index)
        
        assert output.shape == (num_nodes, 1)
    
    def test_attention_weights(self, model):
        """Test attention weight output."""
        num_nodes = 14
        num_edges = 20
        
        x = torch.randn(num_nodes, 12, 5)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        output, attention = model(x, edge_index, return_attention=True)
        
        assert attention.shape == (num_nodes, 12)  # (nodes, input_window)
        
        # Attention should sum to 1
        assert torch.allclose(attention.sum(dim=1), torch.ones(num_nodes), atol=1e-5)


class TestCreateModel:
    """Test model factory function."""
    
    def test_create_a3tgcn(self):
        """Test creating A3TGCN model."""
        config = ModelConfig(model_type="a3tgcn")
        model = create_model(config)
        
        assert isinstance(model, A3TGCNForecaster)
    
    def test_create_custom(self):
        """Test creating custom model."""
        config = ModelConfig(model_type="custom")
        model = create_model(config)
        
        assert isinstance(model, CustomGNNForecaster)
    
    def test_invalid_type(self):
        """Test invalid model type raises error."""
        config = ModelConfig(model_type="invalid")
        
        with pytest.raises(ValueError):
            create_model(config)


class TestEarlyStopping:
    """Test early stopping handler."""
    
    def test_no_improvement(self):
        """Test stopping after no improvement."""
        es = EarlyStopping(patience=3, min_delta=0.01)
        
        # Decreasing loss - should not stop
        assert not es(1.0)
        assert not es(0.9)
        assert not es(0.8)
        
        # No improvement
        assert not es(0.85)  # counter = 1
        assert not es(0.85)  # counter = 2
        assert es(0.85)      # counter = 3 -> stop
    
    def test_continued_improvement(self):
        """Test no stop with continued improvement."""
        es = EarlyStopping(patience=3, min_delta=0.01)
        
        for val in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
            assert not es(val)


class TestForecastingTrainer:
    """Test the training pipeline."""
    
    @pytest.fixture
    def trainer_setup(self):
        """Set up trainer for testing."""
        # Create dataset
        dataset = load_dataset(
            use_synthetic=True,
            config=DatasetConfig(input_window=8, output_window=1, seed=42),
        )
        
        # Create model
        model_config = ModelConfig(
            input_features=dataset.num_features,
            input_window=8,
            hidden_dim=16,
            output_dim=1,
        )
        model = create_model(model_config)
        
        # Create trainer
        with tempfile.TemporaryDirectory() as tmpdir:
            training_config = TrainingConfig(
                epochs=3,
                learning_rate=0.01,
                patience=5,
                checkpoint_dir=tmpdir,
                log_interval=1,
            )
            trainer = ForecastingTrainer(model, dataset, training_config)
            yield trainer, tmpdir
    
    def test_trainer_creation(self, trainer_setup):
        """Test trainer is created correctly."""
        trainer, _ = trainer_setup
        
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.criterion is not None
    
    def test_compute_metrics(self, trainer_setup):
        """Test metric computation."""
        trainer, _ = trainer_setup
        
        pred = torch.tensor([1.0, 2.0, 3.0])
        true = torch.tensor([1.1, 2.2, 2.8])
        
        metrics = trainer._compute_metrics(pred, true)
        
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "mape" in metrics
        
        assert metrics["mse"] >= 0
        assert metrics["mae"] >= 0
    
    def test_train_epoch(self, trainer_setup):
        """Test single training epoch."""
        trainer, _ = trainer_setup
        
        metrics = trainer._train_epoch()
        
        assert "loss" in metrics
        assert metrics["loss"] > 0
    
    def test_validate(self, trainer_setup):
        """Test validation."""
        trainer, _ = trainer_setup
        
        metrics = trainer._validate("val")
        
        assert "loss" in metrics
        assert metrics["loss"] >= 0
    
    def test_full_training(self, trainer_setup):
        """Test full training loop."""
        trainer, tmpdir = trainer_setup
        
        results = trainer.train()
        
        assert "history" in results
        assert "test_metrics" in results
        assert len(results["history"]) == 3  # epochs
        
        # Check checkpoint was saved
        checkpoint_path = Path(tmpdir) / "best_model.pt"
        assert checkpoint_path.exists()


class TestSupplyChainPredictor:
    """Test the prediction module."""
    
    @pytest.fixture
    def predictor_setup(self):
        """Set up predictor for testing."""
        dataset = load_dataset(
            use_synthetic=True,
            config=DatasetConfig(input_window=8, seed=42),
        )
        
        model_config = ModelConfig(
            input_features=dataset.num_features,
            input_window=8,
            hidden_dim=16,
            output_dim=1,
        )
        model = create_model(model_config)
        
        predictor = SupplyChainPredictor(model, dataset)
        return predictor, dataset
    
    def test_predictor_creation(self, predictor_setup):
        """Test predictor is created correctly."""
        predictor, _ = predictor_setup
        
        assert predictor.model is not None
        assert predictor.edge_index is not None
    
    def test_predict(self, predictor_setup):
        """Test basic prediction."""
        predictor, dataset = predictor_setup
        
        # Get test input
        test_data = dataset.get_torch_data("test")
        x, _, y = test_data[0]
        
        result = predictor.predict(x)
        
        assert isinstance(result, PredictionResult)
        assert result.predictions.shape[0] == dataset.num_nodes
        assert result.node_predictions is not None
    
    def test_predict_denormalized(self, predictor_setup):
        """Test denormalized predictions."""
        predictor, dataset = predictor_setup
        
        test_data = dataset.get_torch_data("test")
        x, _, _ = test_data[0]
        
        result = predictor.predict(x, return_denormalized=True)
        
        assert result.predictions_denorm is not None
    
    def test_get_node_forecast(self, predictor_setup):
        """Test node-level forecast."""
        predictor, dataset = predictor_setup
        
        test_data = dataset.get_torch_data("test")
        x, _, _ = test_data[0]
        
        forecast = predictor.get_node_forecast(0, x)
        
        assert "node_id" in forecast
        assert "node_type" in forecast
        assert "prediction" in forecast
    
    def test_get_network_forecast(self, predictor_setup):
        """Test network-level forecast."""
        predictor, dataset = predictor_setup
        
        test_data = dataset.get_torch_data("test")
        x, _, _ = test_data[0]
        
        forecast = predictor.get_network_forecast(x)
        
        assert "total_nodes" in forecast
        assert "by_node_type" in forecast
        assert "network_mean" in forecast
    
    def test_predict_sequence(self, predictor_setup):
        """Test multi-step sequence prediction."""
        predictor, dataset = predictor_setup
        
        test_data = dataset.get_torch_data("test")
        x, _, _ = test_data[0]
        
        seq_pred = predictor.predict_sequence(x, steps=5)
        
        assert seq_pred.shape == (dataset.num_nodes, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

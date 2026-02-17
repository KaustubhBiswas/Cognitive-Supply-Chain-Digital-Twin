"""
ARIMA Baseline Model

Traditional time-series forecasting baseline for comparison with GNN models.
Provides single-node and multi-node ARIMA forecasting capabilities.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    ARIMA = None

logger = logging.getLogger(__name__)


@dataclass
class ARIMAConfig:
    """Configuration for ARIMA baseline model."""
    order: Tuple[int, int, int] = (2, 1, 2)  # (p, d, q)
    seasonal_order: Optional[Tuple[int, int, int, int]] = None  # (P, D, Q, s)
    trend: Optional[str] = None  # 'n', 'c', 't', 'ct'
    enforce_stationarity: bool = True
    enforce_invertibility: bool = True
    auto_order: bool = False  # Auto-select order using AIC
    max_p: int = 5
    max_d: int = 2
    max_q: int = 5


@dataclass 
class ForecastResult:
    """Container for forecast results."""
    predictions: np.ndarray
    confidence_intervals: Optional[np.ndarray] = None  # Shape: (steps, 2) for lower/upper
    model_info: Dict[str, Any] = field(default_factory=dict)


class ARIMABaseline:
    """
    ARIMA baseline for demand forecasting benchmark.
    
    Fits individual ARIMA models for each node in the supply chain
    and provides forecasting capabilities for comparison with GNN models.
    """
    
    def __init__(self, config: Optional[ARIMAConfig] = None):
        """
        Initialize ARIMA baseline.
        
        Args:
            config: ARIMA configuration
        """
        if not HAS_STATSMODELS:
            raise ImportError(
                "statsmodels is required for ARIMA baseline. "
                "Install with: pip install statsmodels"
            )
        
        self.config = config or ARIMAConfig()
        self.models: Dict[int, Any] = {}  # node_id -> fitted ARIMA model
        self.model_orders: Dict[int, Tuple[int, int, int]] = {}  # node_id -> order
        self._training_data: Dict[int, np.ndarray] = {}  # Store for diagnostics
    
    def fit(
        self, 
        time_series: np.ndarray, 
        node_id: int,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Fit ARIMA model for a specific node.
        
        Args:
            time_series: 1D array of historical values
            node_id: Identifier for the node
            verbose: Whether to print fitting progress
            
        Returns:
            Dictionary with fitting statistics
        """
        # Ensure we have enough data
        if len(time_series) < 10:
            raise ValueError(f"Need at least 10 data points, got {len(time_series)}")
        
        # Store training data
        self._training_data[node_id] = time_series.copy()
        
        # Determine order
        if self.config.auto_order:
            order = self._auto_select_order(time_series, node_id, verbose)
        else:
            order = self.config.order
        
        self.model_orders[node_id] = order
        
        # Fit model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                model = ARIMA(
                    time_series,
                    order=order,
                    seasonal_order=self.config.seasonal_order,
                    trend=self.config.trend,
                    enforce_stationarity=self.config.enforce_stationarity,
                    enforce_invertibility=self.config.enforce_invertibility,
                )
                fitted = model.fit()
                self.models[node_id] = fitted
                
                if verbose:
                    logger.info(f"Node {node_id}: Fitted ARIMA{order}, AIC={fitted.aic:.2f}")
                
                return {
                    "node_id": node_id,
                    "order": order,
                    "aic": fitted.aic,
                    "bic": fitted.bic,
                    "converged": True,
                }
                
            except Exception as e:
                logger.warning(f"Node {node_id}: ARIMA fitting failed: {e}")
                # Fallback to simpler model
                return self._fit_fallback(time_series, node_id)
    
    def _auto_select_order(
        self, 
        time_series: np.ndarray, 
        node_id: int,
        verbose: bool = False
    ) -> Tuple[int, int, int]:
        """
        Auto-select ARIMA order using AIC.
        
        Uses grid search over (p, d, q) combinations.
        """
        # Check stationarity to determine d
        d = self._determine_differencing(time_series)
        
        best_aic = float('inf')
        best_order = (1, d, 1)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for p in range(self.config.max_p + 1):
                for q in range(self.config.max_q + 1):
                    if p == 0 and q == 0:
                        continue
                    
                    try:
                        order = (p, d, q)
                        model = ARIMA(time_series, order=order)
                        fitted = model.fit()
                        
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = order
                            
                    except Exception:
                        continue
        
        if verbose:
            logger.info(f"Node {node_id}: Auto-selected ARIMA{best_order} (AIC={best_aic:.2f})")
        
        return best_order
    
    def _determine_differencing(self, time_series: np.ndarray) -> int:
        """Determine differencing order using ADF test."""
        for d in range(self.config.max_d + 1):
            series = time_series.copy()
            for _ in range(d):
                series = np.diff(series)
            
            if len(series) < 10:
                return d
            
            try:
                result = adfuller(series, autolag='AIC')
                p_value = result[1]
                
                # If stationary (p < 0.05), return current d
                if p_value < 0.05:
                    return d
            except Exception:
                continue
        
        return 1  # Default to d=1
    
    def _fit_fallback(
        self, 
        time_series: np.ndarray, 
        node_id: int
    ) -> Dict[str, Any]:
        """Fallback to simple random walk with drift model."""
        try:
            model = ARIMA(time_series, order=(0, 1, 0), trend='c')
            fitted = model.fit()
            self.models[node_id] = fitted
            self.model_orders[node_id] = (0, 1, 0)
            
            return {
                "node_id": node_id,
                "order": (0, 1, 0),
                "aic": fitted.aic,
                "bic": fitted.bic,
                "converged": True,
                "fallback": True,
            }
        except Exception as e:
            logger.error(f"Node {node_id}: Fallback also failed: {e}")
            return {
                "node_id": node_id,
                "converged": False,
                "error": str(e),
            }
    
    def fit_all(
        self, 
        node_time_series: Dict[int, np.ndarray],
        verbose: bool = False
    ) -> Dict[int, Dict[str, Any]]:
        """
        Fit ARIMA models for all nodes.
        
        Args:
            node_time_series: Dictionary mapping node_id to time series
            verbose: Whether to print progress
            
        Returns:
            Dictionary with fitting results for each node
        """
        results = {}
        
        for node_id, series in node_time_series.items():
            results[node_id] = self.fit(series, node_id, verbose)
        
        return results
    
    def predict(
        self, 
        node_id: int, 
        horizon: int = 1,
        return_conf_int: bool = False,
        alpha: float = 0.05
    ) -> Union[np.ndarray, ForecastResult]:
        """
        Generate predictions for specified horizon.
        
        Args:
            node_id: Node to predict for
            horizon: Number of future timesteps
            return_conf_int: Whether to return confidence intervals
            alpha: Significance level for confidence intervals
            
        Returns:
            Predictions array or ForecastResult with confidence intervals
        """
        if node_id not in self.models:
            raise ValueError(f"No model fitted for node {node_id}")
        
        fitted = self.models[node_id]
        
        # Get forecast
        forecast = fitted.get_forecast(steps=horizon)
        predicted_mean = forecast.predicted_mean
        # Handle both pandas Series and numpy array returns
        predictions = predicted_mean.values if hasattr(predicted_mean, 'values') else np.asarray(predicted_mean)
        
        if not return_conf_int:
            return predictions
        
        # Get confidence intervals
        conf_int_raw = forecast.conf_int(alpha=alpha)
        conf_int = conf_int_raw.values if hasattr(conf_int_raw, 'values') else np.asarray(conf_int_raw)
        
        return ForecastResult(
            predictions=predictions,
            confidence_intervals=conf_int,
            model_info={
                "order": self.model_orders.get(node_id),
                "alpha": alpha,
            }
        )
    
    def predict_all(
        self, 
        horizon: int = 1,
        node_ids: Optional[List[int]] = None
    ) -> Dict[int, np.ndarray]:
        """
        Generate predictions for all fitted nodes.
        
        Args:
            horizon: Number of future timesteps
            node_ids: Specific nodes to predict (None = all)
            
        Returns:
            Dictionary mapping node_id to predictions
        """
        if node_ids is None:
            node_ids = list(self.models.keys())
        
        return {
            node_id: self.predict(node_id, horizon)
            for node_id in node_ids
            if node_id in self.models
        }
    
    def evaluate(
        self, 
        test_data: Dict[int, np.ndarray],
        horizon: int = 1
    ) -> Dict[str, float]:
        """
        Evaluate ARIMA predictions against test data.
        
        Args:
            test_data: Dictionary mapping node_id to actual values
            horizon: Forecast horizon used
            
        Returns:
            Dictionary with evaluation metrics
        """
        all_predictions = []
        all_actuals = []
        
        for node_id, actuals in test_data.items():
            if node_id not in self.models:
                continue
            
            predictions = self.predict(node_id, horizon=len(actuals))
            
            # Align lengths
            min_len = min(len(predictions), len(actuals))
            all_predictions.extend(predictions[:min_len])
            all_actuals.extend(actuals[:min_len])
        
        if not all_predictions:
            return {"error": "No predictions available"}
        
        predictions = np.array(all_predictions)
        actuals = np.array(all_actuals)
        
        # Compute metrics
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(mse)
        
        # MAPE with small epsilon to avoid division by zero
        mape = np.mean(np.abs((actuals - predictions) / (np.abs(actuals) + 1e-8))) * 100
        
        # Symmetric MAPE
        smape = np.mean(
            2 * np.abs(predictions - actuals) / 
            (np.abs(predictions) + np.abs(actuals) + 1e-8)
        ) * 100
        
        return {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
            "smape": float(smape),
            "num_nodes": len(test_data),
            "num_predictions": len(predictions),
        }
    
    def get_residuals(self, node_id: int) -> Optional[np.ndarray]:
        """Get residuals from fitted model for diagnostics."""
        if node_id not in self.models:
            return None
        resid = self.models[node_id].resid
        return resid.values if hasattr(resid, 'values') else np.asarray(resid)
    
    def summary(self, node_id: int) -> Optional[str]:
        """Get model summary for a specific node."""
        if node_id not in self.models:
            return None
        return str(self.models[node_id].summary())


class MultiNodeARIMA:
    """
    Wrapper for fitting ARIMA models to supply chain temporal data.
    
    Provides convenient interface for working with SupplyGraphData
    and comparison with GNN models.
    """
    
    def __init__(self, config: Optional[ARIMAConfig] = None):
        """Initialize multi-node ARIMA."""
        self.config = config or ARIMAConfig()
        self.baseline = ARIMABaseline(self.config)
    
    def fit_from_features(
        self,
        node_features: np.ndarray,
        feature_idx: int = 0,
        verbose: bool = False
    ) -> Dict[int, Dict[str, Any]]:
        """
        Fit ARIMA models from node feature array.
        
        Args:
            node_features: Array of shape (num_nodes, num_timesteps, num_features)
            feature_idx: Which feature to use for forecasting (default: 0 = sales_orders)
            verbose: Whether to print progress
            
        Returns:
            Fitting results for all nodes
        """
        num_nodes = node_features.shape[0]
        
        node_time_series = {
            node_id: node_features[node_id, :, feature_idx]
            for node_id in range(num_nodes)
        }
        
        return self.baseline.fit_all(node_time_series, verbose)
    
    def predict_all(
        self, 
        horizon: int = 1
    ) -> np.ndarray:
        """
        Predict for all nodes.
        
        Args:
            horizon: Number of future timesteps
            
        Returns:
            Array of shape (num_nodes, horizon)
        """
        predictions = self.baseline.predict_all(horizon)
        
        if not predictions:
            return np.array([])
        
        num_nodes = max(predictions.keys()) + 1
        result = np.zeros((num_nodes, horizon))
        
        for node_id, preds in predictions.items():
            result[node_id, :] = preds
        
        return result
    
    def evaluate_against_gnn(
        self,
        gnn_predictions: np.ndarray,
        actual_values: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare ARIMA predictions against GNN predictions.
        
        Args:
            gnn_predictions: GNN predictions (num_nodes, horizon)
            actual_values: Actual values (num_nodes, horizon)
            
        Returns:
            Dictionary with metrics for both models
        """
        num_nodes, horizon = actual_values.shape
        
        # ARIMA predictions
        arima_preds = self.predict_all(horizon)
        
        # Compute metrics for both
        results = {}
        
        for name, preds in [("arima", arima_preds), ("gnn", gnn_predictions)]:
            if preds.shape != actual_values.shape:
                continue
            
            mse = np.mean((preds - actual_values) ** 2)
            mae = np.mean(np.abs(preds - actual_values))
            rmse = np.sqrt(mse)
            mape = np.mean(
                np.abs((actual_values - preds) / (np.abs(actual_values) + 1e-8))
            ) * 100
            
            results[name] = {
                "mse": float(mse),
                "mae": float(mae),
                "rmse": float(rmse),
                "mape": float(mape),
            }
        
        # Compute improvement
        if "arima" in results and "gnn" in results:
            results["improvement"] = {
                metric: (results["arima"][metric] - results["gnn"][metric]) / 
                        (results["arima"][metric] + 1e-8) * 100
                for metric in ["mse", "mae", "rmse", "mape"]
            }
        
        return results


if __name__ == "__main__":
    # Test ARIMA baseline
    logging.basicConfig(level=logging.INFO)
    
    # Create synthetic data
    np.random.seed(42)
    num_nodes = 5
    num_timesteps = 100
    
    # Generate seasonal + trend + noise
    t = np.arange(num_timesteps)
    node_features = np.zeros((num_nodes, num_timesteps, 1))
    
    for i in range(num_nodes):
        trend = 0.1 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 12)
        noise = np.random.randn(num_timesteps) * 2
        node_features[i, :, 0] = 50 + trend + seasonal + noise
    
    # Test MultiNodeARIMA
    print("Testing MultiNodeARIMA...")
    multi_arima = MultiNodeARIMA(ARIMAConfig(auto_order=True))
    
    # Fit on training data
    train_data = node_features[:, :80, :]
    results = multi_arima.fit_from_features(train_data, verbose=True)
    
    print(f"\nFitted {len(results)} models")
    
    # Predict
    predictions = multi_arima.predict_all(horizon=20)
    print(f"Predictions shape: {predictions.shape}")
    
    # Evaluate
    test_data = {i: node_features[i, 80:, 0] for i in range(num_nodes)}
    eval_results = multi_arima.baseline.evaluate(test_data, horizon=20)
    
    print(f"\nEvaluation metrics:")
    for metric, value in eval_results.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")

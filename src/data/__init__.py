"""Data module for dataset parsing, loading, and preprocessing."""

from .datasets import (AVAILABLE_DATASETS, DataCoSupplyChainLoader,
                       DatasetInfo, DatasetLoader, OlistEcommerceLoader,
                       SupplyChainData, SupplyChainShipmentLoader,
                       get_dataset_loader, list_available_datasets,
                       load_dataset)
from .ingestion import (DataIngestionPipeline, IngestionConfig,
                        NormalizationMethod, ProcessedData, TemporalResolution,
                        quick_load)
from .parser import (SupplyGraphData, SupplyGraphParser,
                     create_synthetic_supply_graph)
from .visualization import (visualize_bullwhip_effect,
                            visualize_inventory_timeseries, visualize_topology)

__all__ = [
    # Legacy parser
    "SupplyGraphParser",
    "SupplyGraphData",
    "create_synthetic_supply_graph",
    # Visualization
    "visualize_topology",
    "visualize_inventory_timeseries",
    "visualize_bullwhip_effect",
    # Dataset loaders
    "AVAILABLE_DATASETS",
    "DatasetInfo",
    "SupplyChainData",
    "DatasetLoader",
    "SupplyChainShipmentLoader",
    "DataCoSupplyChainLoader", 
    "OlistEcommerceLoader",
    "list_available_datasets",
    "get_dataset_loader",
    "load_dataset",
    # Ingestion pipeline
    "IngestionConfig",
    "ProcessedData",
    "DataIngestionPipeline",
    "TemporalResolution",
    "NormalizationMethod",
    "quick_load",
]

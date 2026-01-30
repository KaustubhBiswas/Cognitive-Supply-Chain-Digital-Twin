"""
SupplyGraph Dataset Parser

Parses the SupplyGraph benchmark dataset for FMCG supply chain analysis.
Dataset source: https://github.com/CIOL-SUST/SupplyGraph

The SupplyGraph dataset contains:
- 221 time points (January 2023 - August 2023)
- Temporal node features: Production, Sales Orders, Delivery, Factory Issues, Storage
- Graph structure: Supplier → Manufacturer → Distributor → Customer relationships
"""

import json
import pandas as pd
import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class SupplyGraphData:
    """Container for parsed SupplyGraph data."""
    
    graph: nx.DiGraph
    """Network topology as a directed graph."""
    
    node_features: np.ndarray
    """Temporal node features. Shape: (num_nodes, num_timesteps, num_features)"""
    
    edge_features: Optional[np.ndarray] = None
    """Edge features if available. Shape: (num_edges, num_features)"""
    
    node_types: Dict[int, str] = field(default_factory=dict)
    """Mapping of node ID to type (supplier, manufacturer, distributor, retailer)."""
    
    node_names: Dict[int, str] = field(default_factory=dict)
    """Mapping of node ID to human-readable name."""
    
    timesteps: List[str] = field(default_factory=list)
    """Timestamp labels for temporal features."""
    
    feature_names: List[str] = field(default_factory=list)
    """Names of the temporal features."""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional dataset metadata."""
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        return self.graph.number_of_nodes()
    
    @property
    def num_edges(self) -> int:
        """Number of edges in the graph."""
        return self.graph.number_of_edges()
    
    @property
    def num_timesteps(self) -> int:
        """Number of temporal observations."""
        return self.node_features.shape[1] if self.node_features.size > 0 else 0
    
    @property
    def num_features(self) -> int:
        """Number of features per node per timestep."""
        return self.node_features.shape[2] if self.node_features.size > 0 else 0
    
    def get_node_timeseries(self, node_id: int, feature_idx: int = 0) -> np.ndarray:
        """Get time series for a specific node and feature."""
        return self.node_features[node_id, :, feature_idx]
    
    def get_timestep_snapshot(self, timestep: int) -> np.ndarray:
        """Get all node features at a specific timestep."""
        return self.node_features[:, timestep, :]
    
    def to_pyg_format(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert to PyTorch Geometric format.
        
        Returns:
            edge_index: Shape (2, num_edges)
            node_features: Shape (num_nodes, num_timesteps, num_features)
        """
        edges = list(self.graph.edges())
        edge_index = np.array(edges).T if edges else np.zeros((2, 0), dtype=np.int64)
        return edge_index, self.node_features


class SupplyGraphParser:
    """
    Parser for SupplyGraph dataset files.
    
    Expected directory structure:
    data/supplygraph/
    ├── edges.csv           # Edge list
    ├── node_types.csv      # Node type mappings
    ├── temporal_features/  # Temporal feature files
    │   ├── production.csv
    │   ├── sales_orders.csv
    │   ├── delivery.csv
    │   ├── factory_issues.csv
    │   └── storage.csv
    └── metadata.json       # Optional metadata
    """
    
    TEMPORAL_FEATURES = [
        "production",
        "sales_orders",
        "delivery",
        "factory_issues",
        "storage",
    ]
    
    NODE_TYPES = ["supplier", "manufacturer", "distributor", "retailer"]
    
    def __init__(self, data_dir: Path | str):
        """
        Initialize parser.
        
        Args:
            data_dir: Path to the SupplyGraph data directory
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")
    
    def parse(self) -> SupplyGraphData:
        """
        Parse all dataset files and return structured data.
        
        Returns:
            SupplyGraphData containing all parsed information
        """
        # Load graph structure
        graph = self._parse_topology()
        
        # Load node types
        node_types = self._parse_node_types(graph)
        
        # Load temporal node features
        node_features, timesteps = self._parse_temporal_features(graph)
        
        # Load edge features if available
        edge_features = self._parse_edge_features()
        
        # Load metadata
        metadata = self._parse_metadata()
        
        return SupplyGraphData(
            graph=graph,
            node_features=node_features,
            edge_features=edge_features,
            node_types=node_types,
            timesteps=timesteps,
            feature_names=self.TEMPORAL_FEATURES,
            metadata=metadata,
        )
    
    def _parse_topology(self) -> nx.DiGraph:
        """Parse edge list to create directed graph."""
        edges_file = self.data_dir / "edges.csv"
        
        if edges_file.exists():
            df = pd.read_csv(edges_file)
            G = nx.DiGraph()
            
            # Determine column names
            if "source" in df.columns and "target" in df.columns:
                for _, row in df.iterrows():
                    G.add_edge(int(row["source"]), int(row["target"]))
            elif len(df.columns) >= 2:
                # Assume first two columns are source and target
                for _, row in df.iterrows():
                    G.add_edge(int(row.iloc[0]), int(row.iloc[1]))
            
            logger.info(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            return G
        else:
            logger.warning(f"Edges file not found: {edges_file}")
            return nx.DiGraph()
    
    def _parse_node_types(self, graph: nx.DiGraph) -> Dict[int, str]:
        """Parse node type mappings."""
        types_file = self.data_dir / "node_types.csv"
        
        if types_file.exists():
            df = pd.read_csv(types_file)
            node_types = {}
            
            if "node_id" in df.columns and "type" in df.columns:
                for _, row in df.iterrows():
                    node_types[int(row["node_id"])] = row["type"]
            elif len(df.columns) >= 2:
                for _, row in df.iterrows():
                    node_types[int(row.iloc[0])] = str(row.iloc[1])
            
            return node_types
        else:
            # Infer types from graph structure
            logger.info("Inferring node types from graph topology")
            return self._infer_node_types(graph)
    
    def _infer_node_types(self, graph: nx.DiGraph) -> Dict[int, str]:
        """Infer node types from graph structure using topological position."""
        if graph.number_of_nodes() == 0:
            return {}
        
        node_types = {}
        
        # Calculate distance from sources
        sources = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        max_distances = {}
        
        for node in graph.nodes():
            distances = []
            for source in sources:
                try:
                    dist = nx.shortest_path_length(graph, source, node)
                    distances.append(dist)
                except nx.NetworkXNoPath:
                    pass
            max_distances[node] = max(distances) if distances else 0
        
        # Assign types based on position
        max_dist = max(max_distances.values()) if max_distances else 0
        
        for node, dist in max_distances.items():
            if dist == 0:
                node_types[node] = "supplier"
            elif dist == max_dist:
                node_types[node] = "retailer"
            elif dist <= max_dist / 3:
                node_types[node] = "manufacturer"
            else:
                node_types[node] = "distributor"
        
        return node_types
    
    def _parse_temporal_features(
        self, graph: nx.DiGraph
    ) -> Tuple[np.ndarray, List[str]]:
        """Parse temporal node features across all timesteps."""
        temporal_dir = self.data_dir / "temporal_features"
        num_nodes = graph.number_of_nodes()
        
        if num_nodes == 0:
            return np.array([]), []
        
        all_features = []
        timesteps = None
        
        for feature_name in self.TEMPORAL_FEATURES:
            feature_file = temporal_dir / f"{feature_name}.csv"
            
            if feature_file.exists():
                df = pd.read_csv(feature_file, index_col=0)
                
                if timesteps is None:
                    timesteps = list(df.columns)
                
                # Ensure all nodes are present
                feature_data = np.zeros((num_nodes, len(timesteps)))
                for node_id in graph.nodes():
                    if node_id in df.index:
                        feature_data[node_id] = df.loc[node_id].values
                
                all_features.append(feature_data)
            else:
                logger.warning(f"Feature file not found: {feature_file}")
                # Add zeros for missing features
                all_features.append(np.zeros((num_nodes, len(timesteps) if timesteps else 1)))
        
        if not all_features:
            return np.array([]), []
        
        # Stack features: (num_nodes, num_timesteps, num_features)
        node_features = np.stack(all_features, axis=-1)
        
        return node_features, timesteps or []
    
    def _parse_edge_features(self) -> Optional[np.ndarray]:
        """Parse edge features if available."""
        edge_features_file = self.data_dir / "edge_features.csv"
        
        if edge_features_file.exists():
            df = pd.read_csv(edge_features_file)
            return df.values
        
        return None
    
    def _parse_metadata(self) -> Dict[str, Any]:
        """Parse metadata JSON if available."""
        metadata_file = self.data_dir / "metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                return json.load(f)
        
        return {
            "source": "SupplyGraph",
            "description": "FMCG Supply Chain Dataset",
        }


def create_synthetic_supply_graph(
    num_suppliers: int = 3,
    num_manufacturers: int = 5,
    num_distributors: int = 8,
    num_retailers: int = 12,
    num_timesteps: int = 100,
    seed: Optional[int] = None,
) -> SupplyGraphData:
    """
    Create synthetic SupplyGraph-like data for testing.
    
    This generates realistic temporal patterns including:
    - Seasonal demand variations
    - Random demand spikes
    - Correlated supply chain dynamics
    
    Args:
        num_suppliers: Number of supplier nodes
        num_manufacturers: Number of manufacturer nodes
        num_distributors: Number of distributor nodes
        num_retailers: Number of retailer nodes
        num_timesteps: Number of temporal observations
        seed: Random seed for reproducibility
        
    Returns:
        SupplyGraphData with synthetic data
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create graph
    G = nx.DiGraph()
    node_types = {}
    node_id = 0
    
    # Create node layers
    supplier_ids = []
    for _ in range(num_suppliers):
        G.add_node(node_id)
        node_types[node_id] = "supplier"
        supplier_ids.append(node_id)
        node_id += 1
    
    manufacturer_ids = []
    for _ in range(num_manufacturers):
        G.add_node(node_id)
        node_types[node_id] = "manufacturer"
        manufacturer_ids.append(node_id)
        node_id += 1
    
    distributor_ids = []
    for _ in range(num_distributors):
        G.add_node(node_id)
        node_types[node_id] = "distributor"
        distributor_ids.append(node_id)
        node_id += 1
    
    retailer_ids = []
    for _ in range(num_retailers):
        G.add_node(node_id)
        node_types[node_id] = "retailer"
        retailer_ids.append(node_id)
        node_id += 1
    
    # Create edges
    for sup in supplier_ids:
        targets = np.random.choice(manufacturer_ids, size=min(2, len(manufacturer_ids)), replace=False)
        for t in targets:
            G.add_edge(sup, t)
    
    for man in manufacturer_ids:
        targets = np.random.choice(distributor_ids, size=min(2, len(distributor_ids)), replace=False)
        for t in targets:
            G.add_edge(man, t)
    
    for dist in distributor_ids:
        targets = np.random.choice(retailer_ids, size=min(2, len(retailer_ids)), replace=False)
        for t in targets:
            G.add_edge(dist, t)
    
    num_nodes = G.number_of_nodes()
    num_features = 5  # production, sales_orders, delivery, factory_issues, storage
    
    # Generate temporal features
    time = np.arange(num_timesteps)
    
    # Base patterns
    seasonal = 10 * np.sin(2 * np.pi * time / 30)  # Monthly seasonality
    trend = 0.1 * time  # Slight upward trend
    
    node_features = np.zeros((num_nodes, num_timesteps, num_features))
    
    for node in range(num_nodes):
        node_type = node_types[node]
        
        # Base demand signal
        base_signal = 50 + seasonal + trend + np.random.randn(num_timesteps) * 5
        
        if node_type == "retailer":
            # Retailers have highest demand variability
            node_features[node, :, 1] = base_signal * 1.2  # sales_orders
            node_features[node, :, 4] = np.maximum(0, 100 - base_signal)  # storage
        elif node_type == "distributor":
            node_features[node, :, 1] = base_signal * 1.0
            node_features[node, :, 2] = base_signal * 0.95  # delivery
            node_features[node, :, 4] = np.maximum(0, 150 - base_signal)
        elif node_type == "manufacturer":
            node_features[node, :, 0] = base_signal * 1.1  # production
            node_features[node, :, 1] = base_signal * 0.9
            node_features[node, :, 3] = np.random.rand(num_timesteps) < 0.02  # factory_issues (2% chance)
            node_features[node, :, 4] = np.maximum(0, 200 - base_signal)
        else:  # supplier
            node_features[node, :, 0] = base_signal * 1.3
            node_features[node, :, 4] = np.maximum(0, 300 - base_signal)
    
    # Ensure non-negative
    node_features = np.maximum(0, node_features)
    
    # Generate timestep labels
    timesteps = [f"day_{i}" for i in range(num_timesteps)]
    
    return SupplyGraphData(
        graph=G,
        node_features=node_features,
        node_types=node_types,
        timesteps=timesteps,
        feature_names=["production", "sales_orders", "delivery", "factory_issues", "storage"],
        metadata={
            "source": "synthetic",
            "seed": seed,
        },
    )

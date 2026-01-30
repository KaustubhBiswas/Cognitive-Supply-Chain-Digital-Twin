"""
Dataset Download Script

Downloads supply chain datasets from various sources:
1. SupplyGraph (CIOL-SUST GitHub)
2. DataCo Global Supply Chain (Kaggle)
3. Supply Chain Shipment Pricing (Kaggle)
4. Brazilian E-Commerce - Olist (Kaggle)

Usage:
    python scripts/download_dataset.py [dataset_name] [--synthetic]
    
Examples:
    python scripts/download_dataset.py supplygraph
    python scripts/download_dataset.py dataco
    python scripts/download_dataset.py olist
    python scripts/download_dataset.py all
    python scripts/download_dataset.py --synthetic  # Create synthetic data
"""

import argparse
import json
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import httpx
    import numpy as np
    import pandas as pd
except ImportError:
    print("Installing required packages...")
    os.system("pip install httpx pandas numpy")
    import httpx
    import numpy as np
    import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Base data directory
DATA_DIR = Path(__file__).parent.parent / "data"

# Dataset configurations
DATASETS = {
    "supplygraph": {
        "name": "SupplyGraph FMCG Dataset",
        "source": "GitHub",
        "url": "https://github.com/CIOL-SUST/SupplyGraph",
        "raw_url": "https://raw.githubusercontent.com/CIOL-SUST/SupplyGraph/main",
        "local_dir": DATA_DIR / "supplygraph",
        "kaggle_id": None,
    },
    "dataco": {
        "name": "DataCo Global Supply Chain",
        "source": "Kaggle",
        "url": "https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis",
        "kaggle_id": "shashwatwork/dataco-smart-supply-chain-for-big-data-analysis",
        "local_dir": DATA_DIR / "cache",
    },
    "shipment": {
        "name": "Supply Chain Shipment Pricing",
        "source": "Kaggle",
        "url": "https://www.kaggle.com/datasets/divyeshardeshana/supply-chain-shipment-pricing-data",
        "kaggle_id": "divyeshardeshana/supply-chain-shipment-pricing-data",
        "local_dir": DATA_DIR / "cache",
    },
    "olist": {
        "name": "Brazilian E-Commerce (Olist)",
        "source": "Kaggle",
        "url": "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce",
        "kaggle_id": "olistbr/brazilian-ecommerce",
        "local_dir": DATA_DIR / "cache" / "olist",
    },
}


def download_file(url: str, dest: Path, timeout: int = 30) -> bool:
    """Download a file from URL to destination."""
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            dest.write_bytes(response.content)
            logger.info(f"Downloaded: {dest.name}")
            return True
    except Exception as e:
        logger.warning(f"Failed to download {url}: {e}")
        return False


def download_kaggle_dataset(kaggle_id: str, dest_dir: Path) -> bool:
    """Download dataset from Kaggle using the Kaggle API."""
    try:
        import kaggle
        
        dest_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading from Kaggle: {kaggle_id}")
        
        kaggle.api.dataset_download_files(
            kaggle_id,
            path=str(dest_dir),
            unzip=True
        )
        logger.info(f"Successfully downloaded to {dest_dir}")
        return True
        
    except ImportError:
        logger.error(
            "Kaggle package not installed. Install with: pip install kaggle\n"
            "Then set up API credentials: https://www.kaggle.com/docs/api"
        )
        return False
    except Exception as e:
        logger.error(f"Kaggle download failed: {e}")
        logger.info(
            f"Please download manually from: https://www.kaggle.com/datasets/{kaggle_id}\n"
            f"Extract files to: {dest_dir}"
        )
        return False


def download_supplygraph() -> bool:
    """Download the SupplyGraph dataset from GitHub."""
    config = DATASETS["supplygraph"]
    local_dir = config["local_dir"]
    raw_url = config["raw_url"]
    
    logger.info("Downloading SupplyGraph dataset from GitHub...")
    
    local_dir.mkdir(parents=True, exist_ok=True)
    temporal_dir = local_dir / "temporal_features"
    temporal_dir.mkdir(exist_ok=True)
    
    success = True
    
    # Files to download
    files_to_download = [
        ("data/edges.csv", local_dir / "edges.csv"),
        ("data/node_types.csv", local_dir / "node_types.csv"),
    ]
    
    temporal_files = [
        "production.csv",
        "sales_orders.csv",
        "delivery.csv",
        "factory_issues.csv",
        "storage.csv",
    ]
    
    for temporal_file in temporal_files:
        files_to_download.append(
            (f"data/temporal_features/{temporal_file}", temporal_dir / temporal_file)
        )
    
    for remote_path, local_path in files_to_download:
        url = f"{raw_url}/{remote_path}"
        if not download_file(url, local_path):
            success = False
            break
    
    if success:
        # Create metadata
        metadata = {
            "source": "SupplyGraph",
            "url": config["url"],
            "downloaded": True,
        }
        with open(local_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    return success


def create_synthetic_dataset(name: str = "default") -> bool:
    """Create synthetic dataset when real data is unavailable."""
    logger.info("Creating synthetic SupplyGraph-like dataset...")
    
    synthetic_dir = DATA_DIR / "supplygraph"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    temporal_dir = synthetic_dir / "temporal_features"
    temporal_dir.mkdir(exist_ok=True)
    
    np.random.seed(42)
    
    # Network structure
    # 3 suppliers → 5 manufacturers → 8 distributors → 12 retailers
    num_suppliers = 3
    num_manufacturers = 5
    num_distributors = 8
    num_retailers = 12
    num_timesteps = 221  # Match real dataset
    
    # Create edges
    edges = []
    node_id = 0
    
    supplier_ids = list(range(node_id, node_id + num_suppliers))
    node_id += num_suppliers
    
    manufacturer_ids = list(range(node_id, node_id + num_manufacturers))
    node_id += num_manufacturers
    
    distributor_ids = list(range(node_id, node_id + num_distributors))
    node_id += num_distributors
    
    retailer_ids = list(range(node_id, node_id + num_retailers))
    
    # Connect layers
    for sup in supplier_ids:
        targets = np.random.choice(manufacturer_ids, size=2, replace=False)
        for t in targets:
            edges.append((sup, t))
    
    for man in manufacturer_ids:
        targets = np.random.choice(distributor_ids, size=2, replace=False)
        for t in targets:
            edges.append((man, t))
    
    for dist in distributor_ids:
        targets = np.random.choice(retailer_ids, size=2, replace=False)
        for t in targets:
            edges.append((dist, t))
    
    edges_df = pd.DataFrame(edges, columns=["source", "target"])
    edges_df.to_csv(synthetic_dir / "edges.csv", index=False)
    logger.info(f"Created edges.csv with {len(edges)} edges")
    
    # Create node types
    node_types = []
    for nid in supplier_ids:
        node_types.append((nid, "supplier"))
    for nid in manufacturer_ids:
        node_types.append((nid, "manufacturer"))
    for nid in distributor_ids:
        node_types.append((nid, "distributor"))
    for nid in retailer_ids:
        node_types.append((nid, "retailer"))
    
    types_df = pd.DataFrame(node_types, columns=["node_id", "type"])
    types_df.to_csv(synthetic_dir / "node_types.csv", index=False)
    logger.info(f"Created node_types.csv with {len(node_types)} nodes")
    
    # Create temporal features
    total_nodes = num_suppliers + num_manufacturers + num_distributors + num_retailers
    time = np.arange(num_timesteps)
    
    # Generate realistic patterns
    seasonal = 10 * np.sin(2 * np.pi * time / 30)  # Monthly cycle
    weekly = 5 * np.sin(2 * np.pi * time / 7)      # Weekly cycle
    trend = 0.02 * time                             # Slight trend
    
    features = {
        "production": np.zeros((total_nodes, num_timesteps)),
        "sales_orders": np.zeros((total_nodes, num_timesteps)),
        "delivery": np.zeros((total_nodes, num_timesteps)),
        "factory_issues": np.zeros((total_nodes, num_timesteps)),
        "storage": np.zeros((total_nodes, num_timesteps)),
    }
    
    for node_id in range(total_nodes):
        base = 50 + seasonal + weekly + trend + np.random.randn(num_timesteps) * 8
        base = np.maximum(0, base)
        
        if node_id in supplier_ids:
            features["production"][node_id] = base * 1.5 + np.random.randn(num_timesteps) * 5
            features["storage"][node_id] = np.maximum(0, 300 - base)
        elif node_id in manufacturer_ids:
            features["production"][node_id] = base * 1.2
            features["sales_orders"][node_id] = base * 1.1
            features["factory_issues"][node_id] = (np.random.rand(num_timesteps) < 0.02).astype(float)
            features["storage"][node_id] = np.maximum(0, 200 - base)
        elif node_id in distributor_ids:
            features["sales_orders"][node_id] = base * 1.0
            features["delivery"][node_id] = base * 0.95
            features["storage"][node_id] = np.maximum(0, 150 - base)
        else:  # retailer
            features["sales_orders"][node_id] = base * 1.3 + np.random.randn(num_timesteps) * 10
            features["storage"][node_id] = np.maximum(0, 100 - base)
    
    # Save temporal features
    timestep_labels = [f"2023-01-01_day{i}" for i in range(num_timesteps)]
    
    for feature_name, data in features.items():
        df = pd.DataFrame(
            data,
            index=range(total_nodes),
            columns=timestep_labels,
        )
        df.index.name = "node_id"
        df.to_csv(temporal_dir / f"{feature_name}.csv")
        logger.info(f"Created {feature_name}.csv")
    
    # Create metadata
    metadata = {
        "source": "synthetic",
        "description": "Synthetic SupplyGraph-like dataset for development",
        "num_nodes": total_nodes,
        "num_edges": len(edges),
        "num_timesteps": num_timesteps,
        "feature_names": list(features.keys()),
        "created_by": "download_dataset.py",
    }
    
    with open(synthetic_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Synthetic dataset created successfully!")
    return True


def download_dataset(dataset_name: str) -> bool:
    """Download a specific dataset."""
    if dataset_name not in DATASETS:
        logger.error(f"Unknown dataset: {dataset_name}")
        logger.info(f"Available datasets: {list(DATASETS.keys())}")
        return False
    
    config = DATASETS[dataset_name]
    logger.info(f"Downloading: {config['name']}")
    
    if dataset_name == "supplygraph":
        return download_supplygraph()
    elif config.get("kaggle_id"):
        return download_kaggle_dataset(config["kaggle_id"], config["local_dir"])
    else:
        logger.error(f"No download method available for {dataset_name}")
        return False


def list_datasets():
    """Print information about available datasets."""
    print("\n" + "=" * 70)
    print("Available Supply Chain Datasets")
    print("=" * 70)
    
    for key, config in DATASETS.items():
        status = "✓ Downloaded" if config["local_dir"].exists() and any(config["local_dir"].glob("*")) else "✗ Not downloaded"
        print(f"\n[{key}]")
        print(f"  Name:   {config['name']}")
        print(f"  Source: {config['source']}")
        print(f"  URL:    {config['url']}")
        print(f"  Status: {status}")
    
    print("\n" + "=" * 70)


def setup_dataset(dataset_name: Optional[str] = None, synthetic: bool = False):
    """Main function to set up datasets."""
    print("=" * 60)
    print("Supply Chain Dataset Setup")
    print("=" * 60)
    
    if synthetic:
        create_synthetic_dataset()
        return
    
    if dataset_name is None:
        list_datasets()
        print("\nUsage:")
        print("  python download_dataset.py <dataset_name>  # Download specific dataset")
        print("  python download_dataset.py all              # Download all datasets")
        print("  python download_dataset.py --synthetic      # Create synthetic data")
        return
    
    if dataset_name == "all":
        for name in DATASETS.keys():
            print(f"\n--- Downloading {name} ---")
            download_dataset(name)
    else:
        success = download_dataset(dataset_name)
        if not success:
            logger.warning("Download failed. Creating synthetic data as fallback...")
            create_synthetic_dataset()
    
    print("\n" + "=" * 60)
    print("Dataset setup complete!")
    print(f"Data location: {DATA_DIR}")
    print("=" * 60)


def main():
    """Entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Download supply chain datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download_dataset.py supplygraph    # Download SupplyGraph from GitHub
    python download_dataset.py dataco         # Download DataCo from Kaggle
    python download_dataset.py olist          # Download Olist from Kaggle
    python download_dataset.py all            # Download all datasets
    python download_dataset.py --synthetic    # Create synthetic data only
    python download_dataset.py --list         # List available datasets
        """
    )
    
    parser.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Dataset to download (supplygraph, dataco, shipment, olist, all)"
    )
    parser.add_argument(
        "--synthetic", "-s",
        action="store_true",
        help="Create synthetic dataset"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available datasets"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_datasets()
    elif args.synthetic:
        create_synthetic_dataset()
    else:
        setup_dataset(args.dataset, args.synthetic)


if __name__ == "__main__":
    main()

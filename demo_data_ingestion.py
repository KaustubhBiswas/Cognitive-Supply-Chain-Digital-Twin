"""
Data Ingestion Demo

Demonstrates the data ingestion pipeline for real-world supply chain datasets.
This is the entry point for loading data before model training.

Usage:
    python demo_data_ingestion.py [--dataset DATASET_NAME] [--download]
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


def show_available_datasets():
    """Display information about available datasets."""
    from src.data import AVAILABLE_DATASETS
    
    console.print("\n[bold blue]📊 Available Supply Chain Datasets[/bold blue]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Dataset Key", style="cyan", width=25)
    table.add_column("Name", style="green", width=30)
    table.add_column("Source", style="yellow", width=15)
    table.add_column("Features", style="white", width=50)
    
    for key, info in AVAILABLE_DATASETS.items():
        features = ", ".join(info.features[:3]) + ("..." if len(info.features) > 3 else "")
        table.add_row(
            key,
            info.name,
            info.source,
            features,
        )
    
    console.print(table)
    
    console.print("\n[dim]Use --dataset <key> to load a specific dataset[/dim]")
    console.print("[dim]Use --download to attempt automatic download from Kaggle[/dim]\n")


def demo_ingestion_pipeline(dataset_name: str = None):
    """Demonstrate the data ingestion pipeline."""
    from src.data import (DataIngestionPipeline, IngestionConfig,
                          NormalizationMethod, TemporalResolution)
    
    console.print(Panel.fit(
        "[bold green]Supply Chain Data Ingestion Pipeline[/bold green]\n"
        "Loads real-world datasets and prepares them for GNN training",
        border_style="green"
    ))
    
    # Configuration
    config = IngestionConfig(
        dataset_name=dataset_name or "dataco_supply_chain",
        temporal_resolution=TemporalResolution.WEEK,
        normalization=NormalizationMethod.ZSCORE,
        max_nodes=50,  # Limit for demo
        input_window=12,
        output_window=1,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )
    
    console.print("\n[bold]Pipeline Configuration:[/bold]")
    config_table = Table(show_header=False, box=None)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="yellow")
    
    config_table.add_row("Dataset", config.dataset_name)
    config_table.add_row("Temporal Resolution", config.temporal_resolution.value)
    config_table.add_row("Normalization", config.normalization.value)
    config_table.add_row("Max Nodes", str(config.max_nodes))
    config_table.add_row("Input Window", f"{config.input_window} timesteps")
    config_table.add_row("Output Window", f"{config.output_window} timesteps")
    config_table.add_row("Train/Val/Test Split", f"{config.train_ratio}/{config.val_ratio}/{config.test_ratio}")
    
    console.print(config_table)
    
    # Run pipeline
    console.print("\n[bold]Running Ingestion Pipeline...[/bold]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading and processing data...", total=None)
        
        try:
            pipeline = DataIngestionPipeline(config)
            data = pipeline.run()
            progress.update(task, description="✅ Pipeline complete!")
        except Exception as e:
            progress.update(task, description=f"⚠️ Using synthetic fallback: {e}")
            pipeline = DataIngestionPipeline(config)
            data = pipeline.run()
    
    # Display results
    console.print("\n[bold green]✓ Data Ingestion Complete![/bold green]\n")
    
    # Summary table
    summary_table = Table(title="Processed Data Summary", show_header=True, header_style="bold")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Total Nodes", str(data.num_nodes))
    summary_table.add_row("Total Edges", str(data.num_edges))
    summary_table.add_row("Timesteps", str(data.num_timesteps))
    summary_table.add_row("Features", str(len(data.feature_names)))
    summary_table.add_row("Feature Names", ", ".join(data.feature_names))
    
    if data.train_sequences:
        summary_table.add_row("Training Sequences", str(len(data.train_sequences)))
    if data.val_sequences:
        summary_table.add_row("Validation Sequences", str(len(data.val_sequences)))
    if data.test_sequences:
        summary_table.add_row("Test Sequences", str(len(data.test_sequences)))
    
    if data.timestamps:
        summary_table.add_row("Time Range", f"{data.timestamps[0].date()} to {data.timestamps[-1].date()}")
    
    console.print(summary_table)
    
    # Node type distribution
    if data.node_types:
        console.print("\n[bold]Node Type Distribution:[/bold]")
        type_counts = {}
        for t in data.node_types.values():
            type_counts[t] = type_counts.get(t, 0) + 1
        
        type_table = Table(show_header=True)
        type_table.add_column("Node Type", style="cyan")
        type_table.add_column("Count", style="yellow")
        type_table.add_column("Percentage", style="green")
        
        total = sum(type_counts.values())
        for node_type, count in sorted(type_counts.items()):
            pct = count / total * 100
            type_table.add_row(node_type.capitalize(), str(count), f"{pct:.1f}%")
        
        console.print(type_table)
    
    # Feature statistics
    console.print("\n[bold]Feature Statistics (after normalization):[/bold]")
    stats_table = Table(show_header=True)
    stats_table.add_column("Feature", style="cyan")
    stats_table.add_column("Mean", style="yellow")
    stats_table.add_column("Std", style="yellow")
    stats_table.add_column("Min", style="green")
    stats_table.add_column("Max", style="green")
    
    features_np = data.node_features.numpy()
    for i, name in enumerate(data.feature_names):
        feat_data = features_np[:, :, i].flatten()
        stats_table.add_row(
            name,
            f"{feat_data.mean():.3f}",
            f"{feat_data.std():.3f}",
            f"{feat_data.min():.3f}",
            f"{feat_data.max():.3f}",
        )
    
    console.print(stats_table)
    
    # Sample sequence info
    if data.train_sequences:
        console.print("\n[bold]Sample Training Sequence Shape:[/bold]")
        x, y = data.train_sequences[0]
        console.print(f"  Input (X):  {tuple(x.shape)} → (nodes, input_window, features)")
        console.print(f"  Target (Y): {tuple(y.shape)} → (nodes, output_window)")
    
    # Return data for further use
    return data


def demo_quick_load():
    """Demonstrate the quick_load convenience function."""
    from src.data import quick_load
    
    console.print("\n[bold blue]Quick Load Demo[/bold blue]")
    console.print("Using quick_load() for one-line data loading:\n")
    
    console.print("[dim]>>> from src.data import quick_load[/dim]")
    console.print("[dim]>>> data = quick_load('dataco_supply_chain', max_nodes=30)[/dim]\n")
    
    data = quick_load(max_nodes=30)
    
    console.print(f"[green]✓[/green] Loaded {data.num_nodes} nodes, {data.num_timesteps} timesteps")
    console.print(f"[green]✓[/green] {len(data.train_sequences)} training sequences ready")
    
    return data


def demo_data_loader():
    """Demonstrate using the data loaders for model training."""
    from src.data import quick_load
    
    console.print("\n[bold blue]Data Loader Demo[/bold blue]")
    console.print("Creating PyTorch-compatible data loaders:\n")
    
    data = quick_load(max_nodes=20, input_window=8)
    
    # Get data loaders
    train_loader = data.get_train_loader(batch_size=4)
    val_loader = data.get_val_loader(batch_size=4)
    test_loader = data.get_test_loader(batch_size=4)
    
    console.print(f"[green]✓[/green] Train loader: {len(train_loader)} batches")
    console.print(f"[green]✓[/green] Val loader: {len(val_loader)} batches")
    console.print(f"[green]✓[/green] Test loader: {len(test_loader)} batches")
    
    # Show sample batch
    console.print("\n[bold]Sample Batch:[/bold]")
    x_batch, y_batch = train_loader[0]
    console.print(f"  Input batch shape:  {tuple(x_batch.shape)} → (batch, nodes, time, features)")
    console.print(f"  Target batch shape: {tuple(y_batch.shape)} → (batch, nodes, output_window)")
    
    return train_loader, val_loader, test_loader


def demo_pyg_conversion():
    """Demonstrate converting to PyTorch Geometric format."""
    from src.data import DataIngestionPipeline, IngestionConfig
    
    console.print("\n[bold blue]PyTorch Geometric Conversion[/bold blue]")
    console.print("Converting to PyG Data object for GNN models:\n")
    
    config = IngestionConfig(max_nodes=20)
    pipeline = DataIngestionPipeline(config)
    data = pipeline.run()
    
    pyg_data = pipeline.get_pyg_data()
    
    console.print(f"[green]✓[/green] PyG Data object created")
    console.print(f"  - x (node features): {tuple(pyg_data.x.shape)}")
    console.print(f"  - edge_index: {tuple(pyg_data.edge_index.shape)}")
    
    if hasattr(pyg_data, 'node_type') and pyg_data.node_type is not None:
        console.print(f"  - node_type encoding: {tuple(pyg_data.node_type.shape)}")
    
    return pyg_data


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Supply Chain Data Ingestion Demo"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=None,
        help="Dataset to load (e.g., dataco_supply_chain, olist_ecommerce)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available datasets"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Attempt to download dataset from Kaggle"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick load demo only"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    console.print(Panel.fit(
        "[bold cyan]🔗 Supply Chain Cognitive Digital Twin[/bold cyan]\n"
        "[dim]Data Ingestion Module Demo[/dim]",
        border_style="cyan"
    ))
    
    if args.list:
        show_available_datasets()
        return
    
    if args.quick:
        demo_quick_load()
        return
    
    try:
        # Main ingestion demo
        data = demo_ingestion_pipeline(args.dataset)
        
        # Additional demos
        console.print("\n" + "=" * 60 + "\n")
        demo_data_loader()
        
        console.print("\n" + "=" * 60 + "\n")
        demo_pyg_conversion()
        
        # Summary
        console.print("\n" + "=" * 60)
        console.print(Panel.fit(
            "[bold green]✅ Data Ingestion Demo Complete![/bold green]\n\n"
            "Next steps:\n"
            "1. Download real dataset: pip install kaggle && kaggle datasets download ...\n"
            "2. Use data for model training: from src.data import quick_load\n"
            "3. Create GNN model with the processed data",
            border_style="green"
        ))
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    main()

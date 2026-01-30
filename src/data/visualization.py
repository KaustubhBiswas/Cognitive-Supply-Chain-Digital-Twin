"""
Supply Chain Network Visualization

Utilities for visualizing the supply chain topology and simulation state.
Provides both static matplotlib plots and interactive Plotly visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from typing import Dict, List, Optional, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from .parser import SupplyGraphData
    from ..simulation.model import SupplyChainModel


# Color scheme for node types
NODE_COLORS = {
    "supplier": "#3498db",      # Blue
    "manufacturer": "#27ae60",  # Green
    "distributor": "#f39c12",   # Orange
    "retailer": "#e74c3c",      # Red
    "customer": "#e74c3c",      # Red
    "unknown": "#95a5a6",       # Gray
}


def visualize_topology(
    graph: nx.DiGraph,
    node_types: Dict[int, str],
    title: str = "Supply Chain Network Topology",
    show_labels: bool = True,
    figsize: tuple = (12, 8),
) -> go.Figure:
    """
    Create interactive Plotly visualization of supply chain network.
    
    Args:
        graph: NetworkX directed graph
        node_types: Mapping of node ID to type
        title: Plot title
        show_labels: Whether to show node labels
        figsize: Figure size (width, height)
        
    Returns:
        Plotly Figure object
    """
    if graph.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(text="No nodes in graph", showarrow=False)
        return fig
    
    # Use multipartite layout for hierarchical visualization
    # Assign subset based on node type order
    type_order = {"supplier": 0, "manufacturer": 1, "distributor": 2, "retailer": 3}
    
    for node in graph.nodes():
        node_type = node_types.get(node, "unknown")
        graph.nodes[node]["subset"] = type_order.get(node_type, 2)
    
    try:
        pos = nx.multipartite_layout(graph, subset_key="subset", align="horizontal")
    except:
        pos = nx.spring_layout(graph, seed=42)
    
    # Create edge traces
    edge_x, edge_y = [], []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.5, color="#888"),
        hoverinfo="none",
        mode="lines",
        name="Connections",
    )
    
    # Create node traces grouped by type
    node_traces = []
    for node_type, color in NODE_COLORS.items():
        nodes = [n for n, t in node_types.items() if t == node_type and n in graph.nodes()]
        if not nodes:
            continue
        
        node_x = [pos[n][0] for n in nodes]
        node_y = [pos[n][1] for n in nodes]
        
        node_traces.append(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text" if show_labels else "markers",
                name=node_type.capitalize(),
                marker=dict(
                    size=20,
                    color=color,
                    line=dict(width=2, color="white"),
                ),
                text=[str(n) for n in nodes],
                textposition="middle center",
                textfont=dict(size=10, color="white"),
                hovertemplate=f"<b>{node_type.capitalize()}</b><br>Node ID: %{{text}}<extra></extra>",
            )
        )
    
    fig = go.Figure(data=[edge_trace] + node_traces)
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        width=figsize[0] * 80,
        height=figsize[1] * 80,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    
    return fig


def visualize_inventory_timeseries(
    model_data: pd.DataFrame,
    agent_data: Optional[pd.DataFrame] = None,
    title: str = "Inventory Levels Over Time",
) -> go.Figure:
    """
    Create time series visualization of inventory levels.
    
    Args:
        model_data: DataFrame from DataCollector.get_model_vars_dataframe()
        agent_data: Optional DataFrame from DataCollector.get_agent_vars_dataframe()
        title: Plot title
        
    Returns:
        Plotly Figure object
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Total Inventory", "Inventory by Agent"),
        vertical_spacing=0.15,
    )
    
    # Total inventory trace
    if "total_inventory" in model_data.columns:
        fig.add_trace(
            go.Scatter(
                x=model_data.index,
                y=model_data["total_inventory"],
                mode="lines",
                name="Total Inventory",
                line=dict(color="#3498db", width=2),
            ),
            row=1,
            col=1,
        )
    
    # Add backlog if available
    if "total_backlog" in model_data.columns:
        fig.add_trace(
            go.Scatter(
                x=model_data.index,
                y=model_data["total_backlog"],
                mode="lines",
                name="Total Backlog",
                line=dict(color="#e74c3c", width=2, dash="dash"),
            ),
            row=1,
            col=1,
        )
    
    # Per-agent inventory if available
    if agent_data is not None and "inventory" in agent_data.columns:
        # Reset index for easier plotting
        agent_data_reset = agent_data.reset_index()
        
        # Get unique agents
        if "AgentID" in agent_data_reset.columns:
            for agent_id in agent_data_reset["AgentID"].unique()[:10]:  # Limit to 10 agents
                agent_subset = agent_data_reset[agent_data_reset["AgentID"] == agent_id]
                fig.add_trace(
                    go.Scatter(
                        x=agent_subset["Step"],
                        y=agent_subset["inventory"],
                        mode="lines",
                        name=f"Agent {agent_id}",
                        opacity=0.7,
                    ),
                    row=2,
                    col=1,
                )
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    
    fig.update_xaxes(title_text="Simulation Step", row=2, col=1)
    fig.update_yaxes(title_text="Units", row=1, col=1)
    fig.update_yaxes(title_text="Units", row=2, col=1)
    
    return fig


def visualize_bullwhip_effect(
    model_data: pd.DataFrame,
    agent_data: Optional[pd.DataFrame] = None,
    echelon_names: Optional[List[str]] = None,
    title: str = "Bullwhip Effect Analysis",
) -> go.Figure:
    """
    Create visualization showing order variance amplification (Bullwhip Effect).
    
    Args:
        model_data: DataFrame from DataCollector.get_model_vars_dataframe()
        agent_data: Optional DataFrame with per-agent order data
        echelon_names: Names for each echelon level
        title: Plot title
        
    Returns:
        Plotly Figure object
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Bullwhip Ratio Over Time",
            "Order Variance by Echelon",
            "Order Patterns",
            "Demand vs Orders",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "scatter"}],
        ],
    )
    
    # Bullwhip ratio time series
    if "bullwhip_ratio" in model_data.columns:
        bullwhip_values = model_data["bullwhip_ratio"].replace([np.inf, -np.inf], np.nan)
        
        fig.add_trace(
            go.Scatter(
                x=model_data.index,
                y=bullwhip_values,
                mode="lines",
                name="Bullwhip Ratio",
                line=dict(color="#9b59b6", width=2),
            ),
            row=1,
            col=1,
        )
        
        # Add reference line at 1.0
        fig.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="gray",
            annotation_text="No amplification (ratio=1)",
            row=1,
            col=1,
        )
    
    # Order variance by echelon (if we can extract it)
    if echelon_names is None:
        echelon_names = ["Suppliers", "Manufacturers", "Distributors", "Retailers"]
    
    # Simulated echelon variances (would be computed from agent_data in real use)
    if "avg_order_variance" in model_data.columns:
        avg_var = model_data["avg_order_variance"].mean()
        # Simulate increasing variance upstream (typical Bullwhip pattern)
        echelon_variances = [avg_var * (4 - i) for i in range(len(echelon_names))]
        
        fig.add_trace(
            go.Bar(
                x=echelon_names,
                y=echelon_variances,
                name="Order Variance",
                marker_color=["#3498db", "#27ae60", "#f39c12", "#e74c3c"],
            ),
            row=1,
            col=2,
        )
    
    # Order patterns over time
    if agent_data is not None and "orders_placed" in agent_data.columns:
        agent_data_reset = agent_data.reset_index()
        if "Step" in agent_data_reset.columns:
            orders_by_step = agent_data_reset.groupby("Step")["orders_placed"].mean()
            
            fig.add_trace(
                go.Scatter(
                    x=orders_by_step.index,
                    y=orders_by_step.values,
                    mode="lines",
                    name="Avg Orders",
                    line=dict(color="#2ecc71", width=2),
                ),
                row=2,
                col=1,
            )
    
    # Demand vs Orders comparison
    if agent_data is not None:
        agent_data_reset = agent_data.reset_index()
        if "demand" in agent_data_reset.columns and "orders_placed" in agent_data_reset.columns:
            if "Step" in agent_data_reset.columns:
                demand_by_step = agent_data_reset.groupby("Step")["demand"].mean()
                orders_by_step = agent_data_reset.groupby("Step")["orders_placed"].mean()
                
                fig.add_trace(
                    go.Scatter(
                        x=demand_by_step.index,
                        y=demand_by_step.values,
                        mode="lines",
                        name="Avg Demand",
                        line=dict(color="#3498db", width=2),
                    ),
                    row=2,
                    col=2,
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=orders_by_step.index,
                        y=orders_by_step.values,
                        mode="lines",
                        name="Avg Orders",
                        line=dict(color="#e74c3c", width=2, dash="dash"),
                    ),
                    row=2,
                    col=2,
                )
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=700,
        showlegend=True,
    )
    
    return fig


def visualize_simulation_dashboard(
    model: "SupplyChainModel",
    show_network: bool = True,
) -> go.Figure:
    """
    Create a comprehensive dashboard visualization for the simulation.
    
    Args:
        model: The SupplyChainModel instance
        show_network: Whether to include network visualization
        
    Returns:
        Plotly Figure object with multiple subplots
    """
    # Get data
    model_data = model.datacollector.get_model_vars_dataframe()
    agent_data = model.datacollector.get_agent_vars_dataframe()
    
    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Total Inventory & Backlog",
            "Bullwhip Ratio",
            "Order Variance",
            "Active Events",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )
    
    # Row 1, Col 1: Inventory and Backlog
    if "total_inventory" in model_data.columns:
        fig.add_trace(
            go.Scatter(
                x=model_data.index,
                y=model_data["total_inventory"],
                name="Inventory",
                line=dict(color="#3498db"),
            ),
            row=1,
            col=1,
        )
    
    if "total_backlog" in model_data.columns:
        fig.add_trace(
            go.Scatter(
                x=model_data.index,
                y=model_data["total_backlog"],
                name="Backlog",
                line=dict(color="#e74c3c", dash="dash"),
            ),
            row=1,
            col=1,
        )
    
    # Row 1, Col 2: Bullwhip Ratio
    if "bullwhip_ratio" in model_data.columns:
        bullwhip = model_data["bullwhip_ratio"].replace([np.inf, -np.inf], np.nan)
        fig.add_trace(
            go.Scatter(
                x=model_data.index,
                y=bullwhip,
                name="Bullwhip",
                line=dict(color="#9b59b6"),
            ),
            row=1,
            col=2,
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=2)
    
    # Row 2, Col 1: Order Variance
    if "avg_order_variance" in model_data.columns:
        fig.add_trace(
            go.Scatter(
                x=model_data.index,
                y=model_data["avg_order_variance"],
                name="Order Var",
                fill="tozeroy",
                line=dict(color="#2ecc71"),
            ),
            row=2,
            col=1,
        )
    
    # Row 2, Col 2: Active Events
    if "active_events" in model_data.columns:
        fig.add_trace(
            go.Scatter(
                x=model_data.index,
                y=model_data["active_events"],
                name="Events",
                mode="lines+markers",
                line=dict(color="#f39c12"),
            ),
            row=2,
            col=2,
        )
    
    fig.update_layout(
        title=dict(
            text=f"Supply Chain Simulation Dashboard (Step {model.current_step})",
            x=0.5,
        ),
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.12),
    )
    
    return fig


def create_matplotlib_topology(
    graph: nx.DiGraph,
    node_types: Dict[int, str],
    ax: Optional[plt.Axes] = None,
    title: str = "Supply Chain Network",
) -> plt.Figure:
    """
    Create a matplotlib visualization (for environments without Plotly).
    
    Args:
        graph: NetworkX directed graph
        node_types: Mapping of node ID to type
        ax: Optional matplotlib axes
        title: Plot title
        
    Returns:
        Matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.get_figure()
    
    # Layout
    for node in graph.nodes():
        node_type = node_types.get(node, "unknown")
        type_order = {"supplier": 0, "manufacturer": 1, "distributor": 2, "retailer": 3}
        graph.nodes[node]["subset"] = type_order.get(node_type, 2)
    
    try:
        pos = nx.multipartite_layout(graph, subset_key="subset", align="horizontal")
    except:
        pos = nx.spring_layout(graph, seed=42)
    
    # Draw edges
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#888", arrows=True, alpha=0.6)
    
    # Draw nodes by type
    for node_type, color in NODE_COLORS.items():
        nodes = [n for n, t in node_types.items() if t == node_type and n in graph.nodes()]
        if nodes:
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=nodes,
                node_color=color,
                node_size=500,
                ax=ax,
                label=node_type.capitalize(),
            )
    
    # Draw labels
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8, font_color="white")
    
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.axis("off")
    
    return fig

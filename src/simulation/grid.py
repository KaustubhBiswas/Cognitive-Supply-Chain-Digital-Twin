"""
Supply Network Grid

Custom Mesa space implementation for network-based supply chain topology.
Provides efficient queries for upstream/downstream relationships and echelon levels.
"""

import networkx as nx
from typing import List, Dict, Optional, Set, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from .agents import SupplyChainAgent


class SupplyNetworkGrid:
    """
    Network grid specialized for supply chain topology.
    
    Unlike Mesa's built-in NetworkGrid, this provides:
    - Directed graph support (upstream/downstream)
    - Echelon-level grouping
    - Path-based distance calculations
    - Efficient neighbor queries
    """
    
    def __init__(self, graph: nx.DiGraph):
        """
        Initialize the network grid.
        
        Args:
            graph: Directed graph representing supply chain topology
        """
        self.graph = graph
        self._agents: Dict[int, "SupplyChainAgent"] = {}
        self._echelon_cache: Dict[int, List[int]] = {}
        self._node_echelon: Dict[int, int] = {}
        self._build_echelon_cache()
    
    def _build_echelon_cache(self):
        """Pre-compute echelon levels for efficient queries."""
        # Find source nodes (no predecessors = suppliers)
        sources = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
        
        # BFS to assign echelon levels
        visited: Set[int] = set()
        current_level = 0
        current_nodes = sources
        
        while current_nodes:
            self._echelon_cache[current_level] = list(current_nodes)
            for node in current_nodes:
                self._node_echelon[node] = current_level
                visited.add(node)
            
            # Get next level (all successors of current level)
            next_nodes = set()
            for node in current_nodes:
                for successor in self.graph.successors(node):
                    if successor not in visited:
                        next_nodes.add(successor)
            
            current_nodes = list(next_nodes)
            current_level += 1
    
    def place_agent(self, agent: "SupplyChainAgent", node_id: int):
        """Place an agent at a node in the network."""
        if node_id not in self.graph.nodes():
            raise ValueError(f"Node {node_id} not in graph")
        self._agents[node_id] = agent
    
    def get_agent(self, node_id: int) -> Optional["SupplyChainAgent"]:
        """Get the agent at a specific node."""
        return self._agents.get(node_id)
    
    def get_all_agents(self) -> List["SupplyChainAgent"]:
        """Get all agents in the network."""
        return list(self._agents.values())
    
    def get_upstream_neighbors(self, node_id: int) -> List[int]:
        """Get all upstream nodes (suppliers/predecessors)."""
        return list(self.graph.predecessors(node_id))
    
    def get_downstream_neighbors(self, node_id: int) -> List[int]:
        """Get all downstream nodes (customers/successors)."""
        return list(self.graph.successors(node_id))
    
    def get_upstream_agents(self, node_id: int) -> List["SupplyChainAgent"]:
        """Get all upstream agents."""
        return [
            self._agents[n] 
            for n in self.get_upstream_neighbors(node_id) 
            if n in self._agents
        ]
    
    def get_downstream_agents(self, node_id: int) -> List["SupplyChainAgent"]:
        """Get all downstream agents."""
        return [
            self._agents[n] 
            for n in self.get_downstream_neighbors(node_id) 
            if n in self._agents
        ]
    
    def get_nodes_at_echelon(self, echelon: int) -> List[int]:
        """Get all node IDs at a specific echelon level."""
        return self._echelon_cache.get(echelon, [])
    
    def get_agents_at_echelon(self, echelon: int) -> List["SupplyChainAgent"]:
        """Get all agents at a specific echelon level."""
        nodes = self.get_nodes_at_echelon(echelon)
        return [self._agents[n] for n in nodes if n in self._agents]
    
    def get_node_echelon(self, node_id: int) -> int:
        """Get the echelon level of a specific node."""
        return self._node_echelon.get(node_id, -1)
    
    def get_num_echelons(self) -> int:
        """Get the total number of echelon levels."""
        return len(self._echelon_cache)
    
    def get_path_length(self, source: int, target: int) -> int:
        """Get the shortest path length between two nodes."""
        try:
            return nx.shortest_path_length(self.graph, source, target)
        except nx.NetworkXNoPath:
            return -1
    
    def get_all_paths(self, source: int, target: int) -> List[List[int]]:
        """Get all simple paths between two nodes."""
        try:
            return list(nx.all_simple_paths(self.graph, source, target))
        except nx.NetworkXNoPath:
            return []
    
    def get_subgraph_for_node(self, node_id: int, depth: int = 2) -> nx.DiGraph:
        """
        Get a subgraph centered on a node with specified depth.
        
        Args:
            node_id: Center node
            depth: Number of hops in each direction
            
        Returns:
            Subgraph containing nodes within depth hops
        """
        nodes_to_include = {node_id}
        
        # Upstream traversal
        current = {node_id}
        for _ in range(depth):
            next_level = set()
            for n in current:
                next_level.update(self.graph.predecessors(n))
            nodes_to_include.update(next_level)
            current = next_level
        
        # Downstream traversal
        current = {node_id}
        for _ in range(depth):
            next_level = set()
            for n in current:
                next_level.update(self.graph.successors(n))
            nodes_to_include.update(next_level)
            current = next_level
        
        return self.graph.subgraph(nodes_to_include).copy()
    
    def to_dict(self) -> dict:
        """Serialize grid state for visualization."""
        return {
            "nodes": list(self.graph.nodes()),
            "edges": list(self.graph.edges()),
            "echelons": {k: v for k, v in self._echelon_cache.items()},
            "agent_positions": {
                node_id: agent.node_type 
                for node_id, agent in self._agents.items()
            },
        }

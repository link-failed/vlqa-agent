"""
Directed Graph Analysis for Tasks 1681, 1464, 1753, and 1871 - Fee Matching Systems
This script constructs a directed graph showing relationships between all columns/fields used in these tasks.
Task 1681: Complex fee matching with monthly volume/fraud calculations
Task 1464: Simple fee matching based on account_type and aci only
Task 1753: Fee matching for specific merchant (Belles_cookbook_store) in March 2023
Task 1871: Fee computation with rate changes for specifi        print(f"Graph Statistics:")
        print(f"- Total nodes: {self.G.number_of_nodes()}")
        print(f"- Total edges: {self.G.number_of_edges()}")
        print(f"- Is connected: {nx.is_connected(self.G)}")rchant and time period
"""

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from typing import Dict, List, Set, Tuple, Any


class FieldDependencyGraph:
    """
    A class to create and analyze field dependency graphs for fee matching systems.
    
    This class encapsulates the creation, analysis, and visualization of undirected graphs
    showing relationships between data fields used in various fee matching tasks.
    """
    
    def __init__(self):
        """Initialize the FieldDependencyGraph with empty graph and nodes."""
        self.G = nx.Graph()
        self.nodes = {}
        self.colors = {
            'payments': '#90EE90',      # Light green for payments data
            'merchant_data': '#FFB6C1', # Light pink for merchant_data
            'fees': '#E8F4FD',          # Light blue for fees data
            'merchant_category_codes': '#DDA0DD',  # Plum for merchant category codes
            'compute': '#FFF2CC',      # Light yellow for compute fields
        }
        self._initialize_graph()
    
    def _initialize_graph(self):
        """Initialize the graph with all nodes and edges for the fee matching system."""
        self._define_nodes()
        self._add_nodes_to_graph()
        self._define_edges()
        self._add_edges_to_graph()
    
    def _define_nodes(self):
        """Define all nodes (data_file.field_name format) with their attributes."""
        self.nodes = {
            # Fields from payments data file
            'payments.day_of_year': {'type': 'data', 'file': 'payments'},
            'payments.year': {'type': 'data', 'file': 'payments'},
            'payments.merchant': {'type': 'data', 'file': 'payments'},
            'payments.aci': {'type': 'data', 'file': 'payments'},
            'payments.card_scheme': {'type': 'data', 'file': 'payments'},
            'payments.is_credit': {'type': 'data', 'file': 'payments'},
            'payments.ip_country': {'type': 'data', 'file': 'payments'},
            'payments.issuing_country': {'type': 'data', 'file': 'payments'},
            'payments.acquirer_country': {'type': 'data', 'file': 'payments'},
            'payments.eur_amount': {'type': 'data', 'file': 'payments'},
            'payments.has_fraudulent_dispute': {'type': 'data', 'file': 'payments'},
            
            # Fields from merchant_data file
            'merchant_data.merchant': {'type': 'data', 'file': 'merchant_data'},
            'merchant_data.capture_delay': {'type': 'data', 'file': 'merchant_data'},
            'merchant_data.merchant_category_code': {'type': 'data', 'file': 'merchant_data'},
            'merchant_data.account_type': {'type': 'data', 'file': 'merchant_data'},
            
            # Fields from fees file
            'fees.card_scheme': {'type': 'data', 'file': 'fees'},
            'fees.is_credit': {'type': 'data', 'file': 'fees'},
            'fees.intracountry': {'type': 'data', 'file': 'fees'},
            'fees.capture_delay': {'type': 'data', 'file': 'fees'},
            'fees.aci': {'type': 'data', 'file': 'fees'},
            'fees.account_type': {'type': 'data', 'file': 'fees'},
            'fees.merchant_category_code': {'type': 'data', 'file': 'fees'},
            'fees.monthly_fraud_level': {'type': 'data', 'file': 'fees'},
            'fees.monthly_volume': {'type': 'data', 'file': 'fees'},
            'fees.rate': {'type': 'data', 'file': 'fees'},
            'fees.fixed_amount': {'type': 'data', 'file': 'fees'},
            
            # Fields from merchant_category_codes file  
            'merchant_category_codes.mcc': {'type': 'data', 'file': 'merchant_category_codes'},
            'merchant_category_codes.description': {'type': 'data', 'file': 'merchant_category_codes'},
            
            # compute/intermediate fields
            'compute.target_month': {'type': 'compute', 'file': 'compute'},
            'compute.transaction_intracountry': {'type': 'compute', 'file': 'compute'},
            'compute.total_volume': {'type': 'compute', 'file': 'compute'},
            'compute.monthly_fraud_level': {'type': 'compute', 'file': 'compute'},
            'compute.fee': {'type': 'compute', 'file': 'compute'},
            
            # Output field - the answer to both tasks
            'fees.ID': {'type': 'data', 'file': 'fees'},
        }
    
    def _add_nodes_to_graph(self):
        """Add all nodes to the NetworkX graph."""
        for node, attrs in self.nodes.items():
            self.G.add_node(node, **attrs)
    
    def _define_edges(self):
        """Define undirected edges (connections between related fields)."""
        self.edges = [
            # Task 1681 & 1753 edges (complex fee matching)
            # Month calculation: day_of_year and year are used to compute target_month
            ('payments.day_of_year', 'compute.target_month'),
            ('payments.year', 'compute.target_month'),
            
            # Merchant info lookup: merchant field is used to lookup merchant data
            ('payments.merchant', 'merchant_data.merchant'),

            # Payments info lookup
            ('payments.merchant', 'payments.aci'),
            
            # Get merchant attributes for fee matching
            ('merchant_data.merchant', 'merchant_data.capture_delay'),
            ('merchant_data.merchant', 'merchant_data.merchant_category_code'),
            ('merchant_data.merchant', 'merchant_data.account_type'),
            
            # Merchant category code lookup and validation
            ('merchant_data.merchant_category_code', 'merchant_category_codes.mcc'),
            ('merchant_category_codes.mcc', 'merchant_category_codes.description'),
            
            # Monthly fraud level calculation: compute from merchant, day_of_year, year and fraud dispute data
            ('payments.merchant', 'compute.monthly_fraud_level'),
            ('compute.target_month', 'compute.monthly_fraud_level'),
            ('payments.has_fraudulent_dispute', 'compute.monthly_fraud_level'),
            
            # Transaction volume calculation for monthly totals: merchant, time period, and amount
            ('payments.merchant', 'compute.total_volume'),
            ('compute.target_month', 'compute.total_volume'),
            ('payments.eur_amount', 'compute.total_volume'),
            
            # Intracountry calculation: issuing_country and acquirer_country used to compute intracountry status
            ('payments.issuing_country', 'compute.transaction_intracountry'),
            ('payments.acquirer_country', 'compute.transaction_intracountry'),
            
            ('fees.aci', 'fees.ID'),
            ('payments.aci', 'fees.aci'),
            
            ('payments.card_scheme', 'fees.card_scheme'),
            ('fees.card_scheme', 'fees.ID'),
            
            ('payments.is_credit', 'fees.is_credit'),
            ('fees.is_credit', 'fees.ID'),
            
            ('compute.transaction_intracountry', 'fees.intracountry'),
            ('fees.intracountry', 'fees.ID'),
            
            ('merchant_data.capture_delay', 'fees.capture_delay'),
            ('fees.capture_delay', 'fees.ID'),
            
            ('merchant_data.account_type', 'fees.account_type'),
            ('fees.account_type', 'fees.ID'),
            
            ('merchant_data.merchant_category_code', 'fees.ID'),
            ('fees.merchant_category_code', 'fees.ID'),
            
            ('compute.monthly_fraud_level', 'fees.monthly_fraud_level'),
            ('fees.monthly_fraud_level', 'fees.ID'),

            ('fees.monthly_volume', 'fees.ID'),
            
            ('fees.rate', 'compute.fee'),
            ('fees.fixed_amount', 'compute.fee'),
            ('payments.eur_amount', 'compute.fee'),  # Used for transaction count and value calculation
        ]
    
    def _add_edges_to_graph(self):
        """Add all edges to the NetworkX graph."""
        for source, target in self.edges:
            self.G.add_edge(source, target)

    def create_node_layout(self) -> Dict[str, Tuple[float, float]]:
        """
        Create layout positions for nodes in the graph.
        
        Returns:
            Dictionary mapping node names to (x, y) positions
        """
        # Group nodes by type for positioning
        data_nodes = [n for n, attrs in self.nodes.items() if attrs['type'] == 'data']
        compute_nodes = [n for n, attrs in self.nodes.items() if attrs['type'] == 'compute']
        
        # Use a layered layout
        pos = {}
        
        # Top rows: data fields from different files
        payments_data = [n for n in data_nodes if n.startswith('payments.')]
        merchant_data = [n for n in data_nodes if n.startswith('merchant_data.')]
        fees_data = [n for n in data_nodes if n.startswith('fees.')]
        merchant_category_codes_data = [n for n in data_nodes if n.startswith('merchant_category_codes.')]
        
        # Position data nodes in layers by file with more spacing
        for i, node in enumerate(payments_data):
            pos[node] = (i * 2, 5)
        
        for i, node in enumerate(merchant_data):
            pos[node] = (i * 3 + 25, 5)
        
        for i, node in enumerate(fees_data):
            pos[node] = (i * 2 + 5, 1)
        
        for i, node in enumerate(merchant_category_codes_data):
            pos[node] = (i * 3 + 35, 4)
        
        # Middle row: compute fields (positioned vertically in the middle)
        for i, node in enumerate(compute_nodes):
            pos[node] = (i * 4 + 10, 3)
        
        return pos

    def draw_graph_nodes(self, pos: Dict[str, Tuple[float, float]]):
        """
        Draw nodes in the graph with appropriate colors.
        
        Args:
            pos: Node positions
        """
        # Draw nodes by file type with different colors
        for file_type in ['payments', 'merchant_data', 'fees', 'merchant_category_codes']:
            node_list = [node for node, attrs in self.nodes.items() if attrs.get('file') == file_type]
            if node_list:
                nx.draw_networkx_nodes(self.G, pos, nodelist=node_list, 
                                      node_color=self.colors[file_type], node_size=2500, 
                                      edgecolors='black', linewidths=1)
        
        # Draw compute nodes
        compute_node_list = [node for node, attrs in self.nodes.items() if attrs['type'] == 'compute']
        if compute_node_list:
            nx.draw_networkx_nodes(self.G, pos, nodelist=compute_node_list, 
                                  node_color=self.colors['compute'], node_size=2500, 
                                  edgecolors='black', linewidths=1)

    def draw_graph_edges(self, pos: Dict[str, Tuple[float, float]]):
        """
        Draw curved edges between nodes in the graph.
        
        Args:
            pos: Node positions
        """
        import matplotlib.patches as patches
        from matplotlib.path import Path
        
        for edge in self.G.edges():
            x1, y1 = pos[edge[0]]
            x2, y2 = pos[edge[1]]
            
            # Calculate control points for curved connections
            dx = x2 - x1
            dy = y2 - y1
            
            # Check if nodes are at the same level (horizontal connections)
            same_level = abs(dy) < 0.5  # Threshold for considering nodes at same level
            
            if same_level:
                # For same-level connections, create moderately curved connections
                curve_height = 0.8 if dx > 0 else -0.8  # Curve upward for left-to-right, downward for right-to-left
                control1_x = x1 + dx * 0.3
                control1_y = y1 + curve_height
                
                control2_x = x1 + dx * 0.7
                control2_y = y1 + curve_height
            else:
                # For vertical connections, use gentle curves
                control1_x = x1 + dx * 0.4
                control1_y = y1 + dy * 0.2 + 0.2  # Subtle curve
                
                control2_x = x1 + dx * 0.6
                control2_y = y1 + dy * 0.8 + 0.2  # Subtle curve
            
            # Create a curved path using cubic Bezier
            vertices = [(x1, y1), (control1_x, control1_y), (control2_x, control2_y), (x2, y2)]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
            path = Path(vertices, codes)
            patch = patches.PathPatch(path, facecolor='none', edgecolor='#2E86AB', 
                                    linewidth=1.5, alpha=0.7)
            plt.gca().add_patch(patch)

    def draw_graph_labels(self, pos: Dict[str, Tuple[float, float]]):
        """
        Draw node labels with better formatting.
        
        Args:
            pos: Node positions
        """
        labels = {}
        for node in self.G.nodes():
            # Split long labels for better readability
            parts = node.split('.')
            if len(parts) == 2:
                labels[node] = f"{parts[0]}\n.{parts[1]}"
            else:
                labels[node] = node
        
        nx.draw_networkx_labels(self.G, pos, labels, font_size=8, font_weight='bold')

    def draw_graph_legend(self):
        """Draw legend for the graph."""
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=self.colors['payments'], edgecolor='black', label='payments.csv'),
            plt.Rectangle((0, 0), 1, 1, facecolor=self.colors['merchant_data'], edgecolor='black', label='merchant_data.json'),
            plt.Rectangle((0, 0), 1, 1, facecolor=self.colors['fees'], edgecolor='black', label='fees.json'),
            plt.Rectangle((0, 0), 1, 1, facecolor=self.colors['merchant_category_codes'], edgecolor='black', label='merchant_category_codes.csv'),
            plt.Rectangle((0, 0), 1, 1, facecolor=self.colors['compute'], edgecolor='black', label='computation')
        ]
        plt.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1, 0), fontsize=12)

    def visualize_graph(self, save_path: str = None):
        """
        Create a visualization of the field dependency graph with color coding and curved edges.
        
        Args:
            save_path: Optional path to save the visualization
        """
        plt.figure(figsize=(16, 12))  # Better size for MacBook display
        
        # Create layout and draw components
        pos = self.create_node_layout()
        self.draw_graph_nodes(pos)
        self.draw_graph_edges(pos)
        self.draw_graph_labels(pos)
        self.draw_graph_legend()
        
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    def find_reachable_nodes(self, input_fields: List[str], output_fields: List[str] = None) -> Set[str]:
        """
        Find all nodes connected to input fields, optionally filtering by output fields.
        
        Args:
            input_fields: List of input node names (guaranteed to be non-empty)
            output_fields: List of output node names (can be empty)
            
        Returns:
            Set of all relevant node names:
            - If output_fields is empty: only the input_fields
            - If output_fields is not empty: all nodes on any path between input_fields and output_fields
        """
        if not input_fields:
            return set()
        
        # Ensure input_fields is a list/set
        if isinstance(input_fields, str):
            input_fields = [input_fields]
        
        # If output_fields is empty, only return input_fields
        if not output_fields:
            return set(input_fields)
        
        # Ensure output_fields is a list/set
        if isinstance(output_fields, str):
            output_fields = [output_fields]
        
        # Find all nodes that can reach any output field
        nodes_connected_to_outputs = set()
        
        # For each output field, find all connected nodes
        for output_field in output_fields:
            if output_field in self.G:
                # Add the output field itself
                nodes_connected_to_outputs.add(output_field)
                
                # Use BFS to find all connected nodes
                stack = [output_field]
                visited = {output_field}
                
                while stack:
                    current = stack.pop()
                    
                    # Get all neighbors (connected nodes)
                    for neighbor in self.G.neighbors(current):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            nodes_connected_to_outputs.add(neighbor)
                            stack.append(neighbor)
        
        # Find all nodes connected to input fields
        nodes_connected_to_inputs = set()
        
        for input_field in input_fields:
            if input_field in self.G:
                # Add the input field itself
                nodes_connected_to_inputs.add(input_field)
                
                # Use BFS to find all connected nodes
                stack = [input_field]
                visited = {input_field}
                
                while stack:
                    current = stack.pop()
                    
                    # Get all neighbors (connected nodes)
                    for neighbor in self.G.neighbors(current):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            nodes_connected_to_inputs.add(neighbor)
                            stack.append(neighbor)
        
        # Return intersection: nodes that are connected to both inputs and outputs
        return nodes_connected_to_inputs.intersection(nodes_connected_to_outputs)

    def get_subgraph_adjacency_list(self, reachable_nodes: Set[str]) -> Dict[str, List[str]]:
        """
        Compute the adjacency list of the subgraph containing only the reachable nodes.
        
        Args:
            reachable_nodes: Set of nodes to include in the subgraph
            
        Returns:
            Dictionary mapping each node to its list of adjacent nodes in the subgraph
        """
        adjacency_list = {}
        
        # Initialize adjacency list for all reachable nodes
        for node in reachable_nodes:
            adjacency_list[node] = []
        
        # Build adjacency list by checking connections between reachable nodes
        for node in reachable_nodes:
            if node in self.G:
                # Get all neighbors of this node
                for neighbor in self.G.neighbors(node):
                    # Only include neighbors that are also in the reachable set
                    if neighbor in reachable_nodes:
                        adjacency_list[node].append(neighbor)
                
                # Sort the adjacency list for consistent output
                adjacency_list[node].sort()
        
        return adjacency_list

    def print_subgraph_adjacency_list(self, reachable_nodes: Set[str], adjacency_list: Dict[str, List[str]]):
        """
        Print the adjacency list of the subgraph in a readable format.
        
        Args:
            reachable_nodes: Set of nodes in the subgraph
            adjacency_list: Adjacency list dictionary
        """
        print("\n" + "="*80)
        print("SUBGRAPH ADJACENCY LIST")
        print("="*80)
        
        print(f"\nSubgraph contains {len(reachable_nodes)} nodes")
        print(f"Edges in subgraph: {sum(len(neighbors) for neighbors in adjacency_list.values()) // 2}")
        
        # Group by file type for better organization
        nodes_by_file = {}
        for node in reachable_nodes:
            if node in self.nodes:
                file_type = self.nodes[node]['file']
                if file_type not in nodes_by_file:
                    nodes_by_file[file_type] = []
                nodes_by_file[file_type].append(node)
        
        # Print adjacency list organized by file type
        for file_type in sorted(nodes_by_file.keys()):
            print(f"\n{file_type.upper()} NODES:")
            for node in sorted(nodes_by_file[file_type]):
                neighbors = adjacency_list.get(node, [])
                degree = len(neighbors)
                print(f"  {node} (degree: {degree}):")
                if neighbors:
                    for neighbor in neighbors:
                        print(f"    -> {neighbor}")
                else:
                    print(f"    (no connections within subgraph)")
        
        # Print summary statistics
        degrees = [len(neighbors) for neighbors in adjacency_list.values()]
        if degrees:
            avg_degree = sum(degrees) / len(degrees)
            max_degree = max(degrees)
            min_degree = min(degrees)
            
            print(f"\nSubgraph Statistics:")
            print(f"  Average degree: {avg_degree:.2f}")
            print(f"  Maximum degree: {max_degree}")
            print(f"  Minimum degree: {min_degree}")
            
            # Find most connected nodes in subgraph
            most_connected = [(node, len(neighbors)) for node, neighbors in adjacency_list.items()]
            most_connected.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\nMost connected nodes in subgraph:")
            for node, degree in most_connected[:5]:
                if degree > 0:
                    print(f"  - {node}: {degree} connections")

    def analyze_reachable_subgraph(self, input_fields: List[str], output_fields: List[str] = None) -> Dict[str, Any]:
        """
        Comprehensive analysis of the reachable subgraph including adjacency list.
        
        Args:
            input_fields: List of input node names
            output_fields: List of output node names (can be empty)
            
        Returns:
            Dictionary containing:
            - reachable_nodes: Set of reachable nodes
            - adjacency_list: Adjacency list of the subgraph
            - subgraph_stats: Statistics about the subgraph
        """
        reachable_nodes = self.find_reachable_nodes(input_fields, output_fields)
        adjacency_list = self.get_subgraph_adjacency_list(reachable_nodes)
        
        # Compute statistics
        degrees = [len(neighbors) for neighbors in adjacency_list.values()]
        edge_count = sum(degrees) // 2  # Each edge counted twice in undirected graph
        
        stats = {
            'node_count': len(reachable_nodes),
            'edge_count': edge_count,
            'avg_degree': sum(degrees) / len(degrees) if degrees else 0,
            'max_degree': max(degrees) if degrees else 0,
            'min_degree': min(degrees) if degrees else 0,
            'density': (2 * edge_count) / (len(reachable_nodes) * (len(reachable_nodes) - 1)) if len(reachable_nodes) > 1 else 0
        }
        
        return {
            'reachable_nodes': reachable_nodes,
            'adjacency_list': adjacency_list,
            'subgraph_stats': stats,
            'input_fields': input_fields,
            'output_fields': output_fields or []
        }

    def print_reachable_analysis(self, input_fields: List[str], output_fields: List[str], reachable_nodes: Set[str]):
        """
        Print detailed analysis of reachable nodes from given input fields to output fields.
        
        Args:
            input_fields: List of input nodes
            output_fields: List of output nodes (can be empty)
            reachable_nodes: Set of reachable nodes
        """
        print("="*80)
        print("REACHABLE NODES ANALYSIS")
        print("="*80)
        
        print(f"\nInput fields: {sorted(input_fields)}")
        if output_fields:
            print(f"Output fields: {sorted(output_fields)}")
            print(f"Analysis: Nodes connected to both inputs and outputs")
        else:
            print(f"Output fields: (empty)")
            print(f"Analysis: Only input fields returned")
        
        print(f"Total reachable nodes: {len(reachable_nodes)}")
        print(f"Percentage of graph reachable: {len(reachable_nodes) / len(self.nodes) * 100:.1f}%")
        
        # Group reachable nodes by type and file
        reachable_by_type = {}
        reachable_by_file = {}
        
        for node in reachable_nodes:
            if node in self.nodes:
                node_type = self.nodes[node]['type']
                node_file = self.nodes[node]['file']
                
                if node_type not in reachable_by_type:
                    reachable_by_type[node_type] = []
                reachable_by_type[node_type].append(node)
                
                if node_file not in reachable_by_file:
                    reachable_by_file[node_file] = []
                reachable_by_file[node_file].append(node)
        
        print(f"\nReachable nodes by type:")
        for node_type, node_list in reachable_by_type.items():
            print(f"\n{node_type.upper()} ({len(node_list)}):")
            for node in sorted(node_list):
                print(f"  - {node}")
        
        print(f"\nReachable nodes by file:")
        for file_name, node_list in reachable_by_file.items():
            print(f"\n{file_name} ({len(node_list)}):")
            for node in sorted(node_list):
                print(f"  - {node}")
        
        # Find nodes NOT reachable
        all_nodes = set(self.nodes.keys())
        unreachable = all_nodes - reachable_nodes
        
        if unreachable:
            print(f"\nNodes NOT connected to both inputs and outputs ({len(unreachable)}):")
            for node in sorted(unreachable):
                print(f"  - {node}")
        
        # Compute and display adjacency list of the subgraph
        if reachable_nodes:
            adjacency_list = self.get_subgraph_adjacency_list(reachable_nodes)
            self.print_subgraph_adjacency_list(reachable_nodes, adjacency_list)

    def print_graph_analysis(self):
        """Print detailed analysis of the graph structure."""
        print("="*80)
        print("FIELD DEPENDENCY GRAPH ANALYSIS FOR TASKS 1681, 1464, 1753, AND 1871")
        print("="*80)
        
        print(f"\nGraph Statistics:")
        print(f"- Total nodes: {self.G.number_of_nodes()}")
        print(f"- Total edges: {self.G.number_of_edges()}")
        print(f"- Is DAG (Directed Acyclic Graph): {nx.is_directed_acyclic_graph(self.G)}")
        
        # Group nodes by type
        node_types = {}
        for node, attrs in self.nodes.items():
            node_type = attrs['type']
            if node_type not in node_types:
                node_types[node_type] = []
            node_types[node_type].append(node)
        
        print(f"\nNodes by Category:")
        for node_type, node_list in node_types.items():
            print(f"\n{node_type.upper()} FIELDS ({len(node_list)}):")
            for node in sorted(node_list):
                degree = self.G.degree(node)
                print(f"  - {node} (degree: {degree})")
        
        # Find key nodes
        print(f"\nKey Nodes Analysis:")
        
        # Nodes with highest degree (most connected)
        degrees = [(node, self.G.degree(node)) for node in self.G.nodes()]
        degrees.sort(key=lambda x: x[1], reverse=True)
        print(f"\nMost Connected Fields (highest degree):")
        for node, degree in degrees[:10]:
            if degree > 0:
                print(f"  - {node}: {degree} connections")
        
        # Isolated nodes (no connections)
        isolated = [node for node in self.G.nodes() if self.G.degree(node) == 0]
        print(f"\nIsolated Fields (no connections): {len(isolated)}")
        for isolated_node in sorted(isolated):
            print(f"  - {isolated_node}")

    @property
    def graph(self) -> nx.Graph:
        """Get the NetworkX graph."""
        return self.G
    
    @property
    def node_count(self) -> int:
        """Get the number of nodes in the graph."""
        return self.G.number_of_nodes()
    
    @property
    def edge_count(self) -> int:
        """Get the number of edges in the graph."""
        return self.G.number_of_edges()
    
    def get_node_info(self, node_name: str) -> Dict[str, Any]:
        """
        Get information about a specific node.
        
        Args:
            node_name: Name of the node
            
        Returns:
            Dictionary with node attributes and graph metrics
        """
        if node_name not in self.nodes:
            return {}
        
        return {
            'attributes': self.nodes[node_name],
            'degree': self.G.degree(node_name),
            'neighbors': list(self.G.neighbors(node_name))
        }


# Legacy functions for backward compatibility
def create_field_dependency_graph():
    """Legacy function - creates FieldDependencyGraph instance and returns graph and nodes."""
    graph_obj = FieldDependencyGraph()
    return graph_obj.G, graph_obj.nodes

def create_node_layout(nodes):
    """Legacy function - use FieldDependencyGraph.create_node_layout() instead."""
    graph_obj = FieldDependencyGraph()
    return graph_obj.create_node_layout()

def draw_graph_nodes(G, nodes, pos, colors):
    """Legacy function - use FieldDependencyGraph.draw_graph_nodes() instead."""
    pass  # Implementation moved to class

def draw_graph_edges(G, pos):
    """Legacy function - use FieldDependencyGraph.draw_graph_edges() instead."""
    pass  # Implementation moved to class

def draw_graph_labels(G, pos):
    """Legacy function - use FieldDependencyGraph.draw_graph_labels() instead."""
    pass  # Implementation moved to class

def draw_graph_legend(colors):
    """Legacy function - use FieldDependencyGraph.draw_graph_legend() instead."""
    pass  # Implementation moved to class

def visualize_graph(G, nodes, save_path=None):
    """Legacy function - use FieldDependencyGraph.visualize_graph() instead."""
    graph_obj = FieldDependencyGraph()
    graph_obj.visualize_graph(save_path)

def find_reachable_nodes(G, input_fields, output_fields=None):
    """Legacy function - use FieldDependencyGraph.find_reachable_nodes() instead."""
    graph_obj = FieldDependencyGraph()
    return graph_obj.find_reachable_nodes(input_fields, output_fields)

def print_reachable_analysis(G, nodes, input_fields, output_fields, reachable_nodes):
    """Legacy function - use FieldDependencyGraph.print_reachable_analysis() instead."""
    graph_obj = FieldDependencyGraph()
    graph_obj.print_reachable_analysis(input_fields, output_fields, reachable_nodes)

def print_graph_analysis(G, nodes):
    """Legacy function - use FieldDependencyGraph.print_graph_analysis() instead."""
    graph_obj = FieldDependencyGraph()
    graph_obj.print_graph_analysis()

def main():
    """
    Main function to create and analyze the field dependency graph.
    """
    print("Creating field dependency graph for Tasks 1681, 1464, 1753, and 1871...")
    
    # Create the graph using the new class-based approach
    field_graph = FieldDependencyGraph()
    
    # Print analysis
    field_graph.print_graph_analysis()
    
    # Create visualization
    print(f"\nGenerating visualization...")
    field_graph.visualize_graph('task_1681_field_dependency_graph.png')
    
    # Demonstrate reachable nodes functionality
    print(f"\n" + "="*80)
    print("DEMONSTRATING REACHABLE NODES FUNCTIONALITY")
    print("="*80)
    
    # Example 1: What nodes are involved in paths from basic payment fields to fee calculation?
    input_fields = ['payments.merchant', 'payments.day_of_year', 'payments.year']
    output_fields = ['fees.ID', 'compute.fee']
    reachable_nodes = field_graph.find_reachable_nodes(input_fields, output_fields)
    field_graph.print_reachable_analysis(input_fields, output_fields, reachable_nodes)
    
    # Example 2: What if we only provide input fields (output_fields is empty)?
    input_fields_only = ['payments.has_fraudulent_dispute']
    reachable_from_inputs_only = field_graph.find_reachable_nodes(input_fields_only, [])
    field_graph.print_reachable_analysis(input_fields_only, [], reachable_from_inputs_only)
    
    # Example 3: Comprehensive analysis with adjacency list
    print(f"\n" + "="*80)
    print("COMPREHENSIVE SUBGRAPH ANALYSIS")
    print("="*80)
    
    country_fields = ['payments.issuing_country', 'payments.acquirer_country']
    fee_outputs = ['fees.ID']
    analysis_result = field_graph.analyze_reachable_subgraph(country_fields, fee_outputs)
    
    print(f"\nComprehensive analysis for country fields -> fee calculation:")
    print(f"Input fields: {analysis_result['input_fields']}")
    print(f"Output fields: {analysis_result['output_fields']}")
    print(f"Subgraph statistics: {analysis_result['subgraph_stats']}")
    
    # Print a subset of the adjacency list
    print(f"\nSample adjacency list (first 5 nodes):")
    sample_nodes = list(analysis_result['adjacency_list'].keys())[:5]
    for node in sample_nodes:
        neighbors = analysis_result['adjacency_list'][node]
        print(f"  {node} -> {neighbors}")
    
if __name__ == "__main__":
    main()

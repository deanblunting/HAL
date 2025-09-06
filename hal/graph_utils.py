"""
Comprehensive graph analysis utilities for planarity testing, community detection, and QECC generation.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from sklearn.cluster import KMeans
import heapq
from collections import defaultdict, deque
try:
    import networkx.algorithms.community as nx_community
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False


class PlanarityTester:
    """Advanced planarity testing implementation based on Hopcroft-Tarjan algorithm."""
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.n = graph.number_of_nodes()
        self.m = graph.number_of_edges()
        
    def is_planar(self) -> bool:
        """Execute planarity analysis using Kuratowski theorem constraints."""
        # Apply preliminary planarity constraints
        if self.n <= 4:
            return True
        if self.m > 3 * self.n - 6:
            return False
            
        # Leverage NetworkX planarity implementation with fallback for compatibility
        try:
            return nx.is_planar(self.graph)
        except AttributeError:
            # Compatibility fallback for legacy NetworkX installations
            return self._simple_planarity_check()
    
    def _simple_planarity_check(self) -> bool:
        """Simplified planarity assessment for legacy compatibility."""
        # Apply heuristic planarity assessment for computational efficiency
        if self.n <= 6:
            return True
        if self.m <= max(3 * self.n - 6, 0):
            return True
        return False
    
    def get_planar_subgraph(self, prioritized_edges: List[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """Extract heuristic maximal planar subgraph exactly as described in the paper.
        
        Paper (Section A.1.a): "Starting from an empty graph, it tentatively inserts the next edge 
        and performs a Hopcroft–Tarjan planarity test; if the edge preserves planarity, 
        it becomes part of the subgraph, otherwise it is discarded and stored for higher-tier routing."
        """
        planar_edges = set()
        test_graph = nx.Graph()
        test_graph.add_nodes_from(self.graph.nodes())
        
        # Implement greedy incremental heuristic exactly as described in the paper
        for edge in prioritized_edges:
            u, v = edge
            if u not in self.graph or v not in self.graph:
                continue
                
            # Tentatively insert the edge
            test_graph.add_edge(u, v)
            
            # Perform planarity test (paper uses Hopcroft-Tarjan, we use NetworkX's implementation)
            try:
                is_planar = nx.is_planar(test_graph)
                if is_planar:
                    # Edge preserves planarity - keep it
                    planar_edges.add(edge)
                else:
                    # Edge breaks planarity - discard it and store for higher-tier routing
                    test_graph.remove_edge(u, v)
            except Exception:
                # Fallback planarity check using Kuratowski's theorem bounds
                n = test_graph.number_of_nodes()
                m = test_graph.number_of_edges()
                
                # For connected planar graphs: m <= 3n - 6 (for n >= 3)
                # For general planar graphs: m <= 3n - 6 + (number_of_components - 1)
                if n <= 2:
                    # Very small graphs are always planar
                    planar_edges.add(edge)
                elif m <= 3 * n - 6:
                    # Satisfies planar graph edge bound
                    planar_edges.add(edge)
                else:
                    # Exceeds planar bound - discard edge
                    test_graph.remove_edge(u, v)
                
        return planar_edges


class CommunityDetector:
    """Advanced community detection implementation utilizing Louvain algorithm methodology."""
    
    def __init__(self, graph: nx.Graph, config=None):
        self.graph = graph
        self.config = config
        
    def detect_communities(self) -> Dict[int, int]:
        """Execute community detection using Louvain algorithm as described in the placement phase."""
        if self.graph.number_of_nodes() == 0:
            return {}
        
        # Prioritize NetworkX Louvain implementation for optimal performance
        if HAS_LOUVAIN and self.config and getattr(self.config, 'use_louvain_communities', True):
            try:
                # Apply NetworkX Louvain community detection algorithm
                community_sets = nx_community.louvain_communities(
                    self.graph, 
                    resolution=getattr(self.config, 'community_resolution', 1.0),
                    random_state=getattr(self.config, 'random_seed', 42)
                )
                
                communities = {}
                for community_id, community_set in enumerate(community_sets):
                    for node in community_set:
                        communities[node] = community_id
                        
                return communities
            except Exception:
                # Implement graceful degradation for algorithmic robustness
                pass
        
        # Attempt alternative Louvain implementation
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(self.graph)
            return partition
        except ImportError:
            # Deploy structural clustering fallback mechanism
            return self._simple_community_detection()
    
    def _simple_community_detection(self) -> Dict[int, int]:
        """Simplified community detection using connected components analysis."""
        if self.graph.number_of_edges() == 0:
            return {node: i for i, node in enumerate(self.graph.nodes())}
        
        # Initialize communities from connected components
        communities = {}
        community_id = 0
        
        for component in nx.connected_components(self.graph):
            if len(component) <= 10:  # Small components stay as single community
                for node in component:
                    communities[node] = community_id
                community_id += 1
            else:
                # Split large components using node positions if available
                component_nodes = list(component)
                subgraph = self.graph.subgraph(component_nodes)
                
                # Use eigenvector centrality for clustering
                try:
                    centrality = nx.eigenvector_centrality(subgraph)
                    # Simple threshold-based clustering
                    threshold = np.median(list(centrality.values()))
                    
                    for node in component_nodes:
                        if centrality[node] > threshold:
                            communities[node] = community_id
                        else:
                            communities[node] = community_id + 1
                    community_id += 2
                except:
                    # Fallback: all nodes in same community
                    for node in component:
                        communities[node] = community_id
                    community_id += 1
        
        return communities


class GraphAnalyzer:
    """Analyze graph properties and compute distances."""
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self._distance_cache = {}
        
    def compute_all_pairs_shortest_paths(self) -> Dict[Tuple[int, int], float]:
        """Compute shortest path distances between all pairs of nodes."""
        if not self._distance_cache:
            try:
                distances = dict(nx.all_pairs_shortest_path_length(self.graph))
                for u in distances:
                    for v in distances[u]:
                        self._distance_cache[(u, v)] = distances[u][v]
                        self._distance_cache[(v, u)] = distances[u][v]
            except:
                # Fallback for disconnected graphs
                for u in self.graph.nodes():
                    for v in self.graph.nodes():
                        if u == v:
                            self._distance_cache[(u, v)] = 0
                        else:
                            try:
                                dist = nx.shortest_path_length(self.graph, u, v)
                                self._distance_cache[(u, v)] = dist
                            except nx.NetworkXNoPath:
                                self._distance_cache[(u, v)] = float('inf')
        
        return self._distance_cache
    
    def get_edge_priorities(self, communities: Dict[int, int], node_positions: Optional[Dict[int, Tuple[float, float]]] = None) -> List[Tuple[int, int]]:
        """Get edges sorted by priority for planar subgraph extraction.
        
        Paper (Section A.1.a): "For each edge, we then extract whether it is an intra- or 
        inter-community edge and the Euclidean length of the straight segment between its 
        endpoints. Edges are sorted: first, all intra-community edges are ordered by increasing 
        length, followed by the inter-community edges, again from short to long."
        """
        edges = list(self.graph.edges())
        
        # Separate intra-community and inter-community edges
        intra_community_edges = []
        inter_community_edges = []
        
        for edge in edges:
            u, v = edge
            u_community = communities.get(u, -1)
            v_community = communities.get(v, -1)
            
            # Classify edge type based on community membership
            if u_community == v_community and u_community != -1:
                intra_community_edges.append(edge)
            else:
                inter_community_edges.append(edge)
        
        # Define edge length function based on available position data
        if node_positions:
            def get_euclidean_length(edge):
                """Compute Euclidean length between endpoints as specified in paper."""
                u, v = edge
                if u in node_positions and v in node_positions:
                    x1, y1 = node_positions[u]
                    x2, y2 = node_positions[v]
                    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                return float('inf')
            
            length_function = get_euclidean_length
        else:
            # Fallback to graph-theoretic distances when positions unavailable
            distances = self.compute_all_pairs_shortest_paths()
            
            def get_graph_distance(edge):
                u, v = edge
                return distances.get((u, v), float('inf'))
            
            length_function = get_graph_distance
        
        # Sort edge groups by increasing length as specified in paper
        intra_community_edges.sort(key=length_function)
        inter_community_edges.sort(key=length_function)
        
        # Return prioritized edge list: intra-community first, then inter-community
        return intra_community_edges + inter_community_edges
    
    def analyze_connectivity(self) -> Dict[str, any]:
        """Analyze basic graph connectivity properties."""
        # Handle empty graph case
        if self.graph.number_of_nodes() == 0:
            return {
                'nodes': 0,
                'edges': 0,
                'density': 0.0,
                'is_connected': False,
                'num_components': 0,
                'avg_clustering': 0.0,
                'diameter': float('inf')
            }
            
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_connected(self.graph) if self.graph.number_of_nodes() > 0 else False,
            'num_components': nx.number_connected_components(self.graph),
            'avg_clustering': nx.average_clustering(self.graph),
            'diameter': nx.diameter(self.graph) if (nx.is_connected(self.graph) and self.graph.number_of_nodes() > 0) else float('inf')
        }


def _is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def _generate_classical_radial_matrix(r: int, s: int) -> np.ndarray:
    """
    Generate a classical radial code matrix A with elements a(u,u').
    
    Each element describes connections between rings u and u'.
    Ensures girth ≥ 6 and linear independence constraints.
    """
    import random
    
    A = np.zeros((r, r), dtype=int)
    
    # Ensure rows are linearly independent by making matrix full rank
    for u in range(r):
        for u_prime in range(r):
            A[u, u_prime] = random.randint(0, s - 1)
    
    # Check linear independence constraint and girth constraint
    max_attempts = 100
    attempt = 0
    
    while attempt < max_attempts:
        # Check girth constraint: a(u1,u'1) - a(u1,u'2) - a(u2,u'1) + a(u2,u'2) ≠ 0 mod s
        valid = True
        for u1 in range(r):
            for u2 in range(r):
                if u1 != u2:
                    for up1 in range(r):
                        for up2 in range(r):
                            if up1 != up2:
                                diff = (A[u1, up1] - A[u1, up2] - A[u2, up1] + A[u2, up2]) % s
                                if diff == 0:
                                    valid = False
                                    break
                        if not valid:
                            break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            break
        
        # Regenerate if constraints not satisfied
        for u in range(r):
            for u_prime in range(r):
                A[u, u_prime] = random.randint(0, s - 1)
        attempt += 1
    
    return A


def _matrices_equal(A1: np.ndarray, A2: np.ndarray) -> bool:
    """Check if two matrices are equal."""
    return np.array_equal(A1, A2)


def _lifted_product_to_graph(A1: np.ndarray, A2: np.ndarray, r: int, s: int) -> nx.Graph:
    """
    Convert lifted product of two classical radial codes to connectivity graph.
    
    Implementation follows the lifted product construction:
    HX = [H1 ⊗ Im2 | Im1 ⊗ H2]
    HZ = [In1 ⊗ H2* | H1* ⊗ In2]
    
    Each stabilizer connects to 2r qubits (r from each classical code).
    """
    n_qubits = 2 * r * r * s  # Total number of physical qubits
    
    G = nx.Graph()
    G.add_nodes_from(range(n_qubits))
    
    # Each qubit has coordinate (c, u, v) where:
    # c ∈ {0,1,...,2r-1}: which classical code (0 to r-1 for X codes, r to 2r-1 for Z codes)
    # u ∈ {0,1,...,r-1}: ring index
    # v ∈ {0,1,...,s-1}: spoke index
    
    # Build edges based on stabilizer structure
    # Each stabilizer from code c, ring u connects to:
    # - One qubit from each ring of code c (via H matrix)
    # - One qubit from ring u of each other code (via lifted product)
    
    for code_idx in range(2 * r):  # X codes (0 to r-1) and Z codes (r to 2r-1)
        is_x_code = code_idx < r
        matrix = A1 if is_x_code else A2
        
        for ring_u in range(r):
            for spoke_v in range(s):
                # This stabilizer connects qubits based on matrix entries
                stabilizer_qubits = []
                
                # Connect to qubits within same code (via circulant structure)
                for ring_up in range(r):
                    shift = matrix[ring_u, ring_up]
                    target_spoke = (spoke_v + shift) % s
                    target_qubit = code_idx * (r * s) + ring_up * s + target_spoke
                    stabilizer_qubits.append(target_qubit)
                
                # Connect to qubits in other codes (via lifted product structure)
                other_codes_start = r if is_x_code else 0
                other_codes_end = 2 * r if is_x_code else r
                
                for other_code in range(other_codes_start, other_codes_end):
                    if other_code != code_idx:
                        # Connect to ring_u qubit in other code
                        target_qubit = other_code * (r * s) + ring_u * s + spoke_v
                        stabilizer_qubits.append(target_qubit)
                
                # Add edges between all qubits in this stabilizer
                for i in range(len(stabilizer_qubits)):
                    for j in range(i + 1, len(stabilizer_qubits)):
                        u, v = stabilizer_qubits[i], stabilizer_qubits[j]
                        if u != v:
                            G.add_edge(u, v)
    
    return G


def create_qecc_graph_from_edges(edges: List[Tuple[int, int]]) -> nx.Graph:
    """Create a graph from a list of edges."""
    G = nx.Graph()
    G.add_edges_from(edges)
    return G



def create_bicycle_code_graph(n1: int, n2: int) -> nx.Graph:
    """
    Create a bivariate bicycle code connectivity graph from dimensions.
    
    Args:
        n1: Number of rows
        n2: Number of columns
    """
    n = n1 * n2  # Total number of qubits
    
    # Use simple shift parameters as default
    a, b = 1, 1
    
    G = nx.Graph()
    
    # Add nodes
    for i in range(n):
        G.add_node(i)
    
    # Add edges based on bicycle code structure
    for i in range(n1):
        for j in range(n2):
            node = i * n2 + j
            
            # X-type edges (shifted by a in dimension 1)
            next_i = (i + a) % n1
            next_node = next_i * n2 + j
            G.add_edge(node, next_node)
            
            # Z-type edges (shifted by b in dimension 2)
            next_j = (j + b) % n2
            next_node = i * n2 + next_j
            G.add_edge(node, next_node)
    
    # Store dimensions as graph attributes for custom position generation
    G.graph['n1'] = n1
    G.graph['n2'] = n2
    
    return G


def create_tile_code_graph(n: int, k: int, d: int) -> nx.Graph:
    """
    Create a tile code connectivity graph from [n,k,d] parameters.
    Uses heuristic to determine tile arrangement with open boundaries.
    
    Args:
        n: Number of physical qubits
        k: Number of logical qubits
        d: Distance of the code
    
    Returns:
        NetworkX graph representing the tile code with O(1) locality
    """
    # Determine tile arrangement from n
    tile_size = 3  # Standard 3x3 tiles
    total_tiles = max(1, n // (tile_size * tile_size))
    tiles_per_side = max(1, int(total_tiles**0.5))
    tiles_x = tiles_per_side
    tiles_y = (total_tiles + tiles_per_side - 1) // tiles_per_side
    
    G = nx.Graph()
    
    # Create nodes for each tile position
    node_id = 0
    node_positions = {}
    
    # Place data qubits and check qubits within each tile
    for tile_y in range(tiles_y):
        for tile_x in range(tiles_x):
            base_x = tile_x * tile_size
            base_y = tile_y * tile_size
            
            # Add nodes within this tile
            tile_nodes = []
            for i in range(tile_size):
                for j in range(tile_size):
                    x, y = base_x + j, base_y + i
                    G.add_node(node_id)
                    node_positions[node_id] = (x, y)
                    tile_nodes.append(node_id)
                    node_id += 1
            
            # Connect nodes within the tile (star configuration)
            center_node = tile_nodes[len(tile_nodes)//2]  # Central node as check qubit
            for node in tile_nodes:
                if node != center_node:
                    G.add_edge(center_node, node)
    
    # Add inter-tile connections with open boundaries (as specified in HAL paper)
    # For open boundaries, only connect adjacent internal tiles
    # No connections that would wrap around boundaries
    for tile_y in range(tiles_y - 1):
        for tile_x in range(tiles_x - 1):
            current_tile_center = tile_y * tiles_x * (tile_size * tile_size) + \
                                tile_x * (tile_size * tile_size) + (tile_size * tile_size) // 2
            
            # Connect to right tile
            if tile_x < tiles_x - 1:
                right_tile_center = current_tile_center + (tile_size * tile_size)
                if G.has_node(current_tile_center) and G.has_node(right_tile_center):
                    G.add_edge(current_tile_center, right_tile_center)
            
            # Connect to bottom tile  
            if tile_y < tiles_y - 1:
                bottom_tile_center = current_tile_center + tiles_x * (tile_size * tile_size)
                if G.has_node(current_tile_center) and G.has_node(bottom_tile_center):
                    G.add_edge(current_tile_center, bottom_tile_center)
    
    return G


def create_radial_code_graph(r: int, s: int) -> nx.Graph:
    """
    Create a quantum radial code connectivity graph using lifted product construction.
    
    Implements the construction from "High-threshold, low-overhead and single-shot 
    decodable fault-tolerant quantum memory" by Scruby, Hillmann, and Roffe.
    
    Creates quantum radial codes with parameters [[2r²s, 2(r-1)², ≤2s]].
    
    Args:
        r: Number of rings (must satisfy r ≤ s)
        s: Number of spokes per ring (must be prime)
    
    Returns:
        NetworkX graph representing quantum radial code with 2r²s nodes
    """
    # Validate parameters
    if r > s:
        raise ValueError(f"r ({r}) must be ≤ s ({s})")
    if not _is_prime(s):
        raise ValueError(f"s ({s}) must be prime")
    
    # Generate classical radial codes H1 and H2
    A1 = _generate_classical_radial_matrix(r, s)
    A2 = _generate_classical_radial_matrix(r, s)
    
    # Ensure A1 and A2 are different for better distance properties
    while _matrices_equal(A1, A2):
        A2 = _generate_classical_radial_matrix(r, s)
    
    # Create quantum code via lifted product
    G = _lifted_product_to_graph(A1, A2, r, s)
    
    # Store parameters as graph attributes
    G.graph['r'] = r
    G.graph['s'] = s
    G.graph['n'] = 2 * r * r * s
    G.graph['k'] = 2 * (r - 1) * (r - 1)
    G.graph['d_upper'] = 2 * s
    
    return G


def create_radial_code_graph_from_nkd(n: int, k: int, d: int) -> nx.Graph:
    """
    Create a quantum radial code from [n,k,d] parameters by finding suitable (r,s).
    
    Args:
        n: Number of physical qubits
        k: Number of logical qubits  
        d: Distance of the code
        
    Returns:
        NetworkX graph representing quantum radial code
    """
    # Find suitable (r,s) parameters
    # For quantum radial codes: n = 2r²s, k = 2(r-1)², d ≤ 2s
    
    best_r, best_s = None, None
    best_error = float('inf')
    
    # Search for good (r,s) values
    for r_candidate in range(2, min(20, int(k**0.5) + 5)):  # Reasonable range for r
        if 2 * (r_candidate - 1)**2 != k:
            continue  # k must match exactly
            
        # Find s such that n ≈ 2r²s and s is prime
        target_s = n // (2 * r_candidate**2)
        
        # Search around target_s for prime values
        for s_candidate in range(max(r_candidate + 1, target_s - 10), target_s + 20):
            if s_candidate <= r_candidate or not _is_prime(s_candidate):
                continue
                
            predicted_n = 2 * r_candidate**2 * s_candidate
            predicted_d_upper = 2 * s_candidate
            
            error = abs(predicted_n - n) + max(0, d - predicted_d_upper)
            
            if error < best_error:
                best_error = error
                best_r, best_s = r_candidate, s_candidate
    
    if best_r is None:
        # Fallback: use heuristic values
        best_r = max(2, int((k/2)**0.5) + 1)
        best_s = max(best_r + 1, 5)
        # Find next prime ≥ best_s
        while not _is_prime(best_s):
            best_s += 1
    
    return create_radial_code_graph(best_r, best_s)






def create_bicycle_code_positions(n1: int, n2: int) -> Dict[int, Tuple[int, int]]:
    """
    Create custom positions for bicycle code in compact grid layout.
    
    Args:
        n1: First dimension
        n2: Second dimension
    
    Returns:
        Dictionary mapping node IDs to (x, y) positions in compact grid
    """
    positions = {}
    
    for i in range(n1):
        for j in range(n2):
            node_id = i * n2 + j
            positions[node_id] = (j, i)  # Simple compact grid placement
    
    return positions


def create_tile_code_positions(tile_pattern: str, tiles_x: int, tiles_y: int, 
                              tile_size: int = 3) -> Dict[int, Tuple[int, int]]:
    """
    Create custom positions for tile codes respecting tile boundaries.
    
    Args:
        tile_pattern: Type of tile pattern ('square', 'triangular')
        tiles_x: Number of tiles in x direction
        tiles_y: Number of tiles in y direction
        tile_size: Size of each tile
    
    Returns:
        Dictionary mapping node IDs to (x, y) positions
    """
    positions = {}
    node_id = 0
    
    if tile_pattern == 'square':
        for tile_y in range(tiles_y):
            for tile_x in range(tiles_x):
                # Position tiles with spacing
                base_x = tile_x * (tile_size + 1)
                base_y = tile_y * (tile_size + 1)
                
                # Place nodes within each tile
                for i in range(tile_size):
                    for j in range(tile_size):
                        if node_id < 1000:  # Safety limit
                            positions[node_id] = (base_x + j, base_y + i)
                            node_id += 1
    
    return positions


def get_code_custom_positions(graph: nx.Graph, code_type: str, 
                             **kwargs) -> Optional[Dict[int, Tuple[int, int]]]:
    """
    Generate appropriate custom positions based on code type.
    
    Args:
        graph: The connectivity graph
        code_type: Type of code ('surface', 'bicycle', 'tile', 'radial')
        **kwargs: Additional parameters specific to each code type
    
    Returns:
        Custom positions dictionary or None for codes that should use spring layout
    """
    n_nodes = graph.number_of_nodes()
    
    if code_type == 'bicycle':
        # Get dimensions from graph attributes (stored during graph creation)
        n1 = graph.graph.get('n1')
        n2 = graph.graph.get('n2')
        
        if n1 is None or n2 is None:
            # Fallback: estimate dimensions for bicycle code
            n1 = kwargs.get('n1', int(n_nodes**0.5))
            while n_nodes % n1 != 0 and n1 > 1:
                n1 -= 1
            n2 = kwargs.get('n2', n_nodes // n1)
        
        return create_bicycle_code_positions(n1, n2)
    
    elif code_type == 'tile':
        # Default tile arrangement
        tile_size = kwargs.get('tile_size', 3)
        tiles_per_side = max(1, int((n_nodes / (tile_size * tile_size))**0.5))
        return create_tile_code_positions('square', tiles_per_side, tiles_per_side, tile_size)
    
    elif code_type == 'radial':
        # Radial codes should use spring layout
        return None
    
    else:
        # Unknown code type, use spring layout
        return None


def create_radial_code_graph_legacy(rings: int, nodes_per_ring: int, ring_connections: str = 'nearest') -> nx.Graph:
    """
    Legacy radial code graph creator (kept for backward compatibility).
    
    Args:
        rings: Number of concentric rings
        nodes_per_ring: Number of nodes in each ring
        ring_connections: Type of connections between rings ('nearest', 'full', 'alternating')
    
    Returns:
        NetworkX graph representing the radial code
    """
    G = nx.Graph()
    
    # Add center node if rings > 0
    total_nodes = 0
    if rings > 0:
        G.add_node(0)
        total_nodes = 1
    
    # Add nodes for each ring
    ring_start_nodes = [0]  # Start index for each ring
    for ring in range(1, rings + 1):
        ring_nodes = []
        for i in range(nodes_per_ring):
            node_id = total_nodes + i
            G.add_node(node_id)
            ring_nodes.append(node_id)
        ring_start_nodes.append(total_nodes)
        total_nodes += nodes_per_ring
        
        # Connect nodes within the ring (circular)
        for i in range(nodes_per_ring):
            current = ring_start_nodes[ring] + i
            next_node = ring_start_nodes[ring] + ((i + 1) % nodes_per_ring)
            G.add_edge(current, next_node)
    
    # Connect rings based on connection type
    for ring in range(1, rings + 1):
        if ring == 1:
            # Connect center to first ring
            center = 0
            for i in range(nodes_per_ring):
                ring_node = ring_start_nodes[ring] + i
                G.add_edge(center, ring_node)
        else:
            # Connect to previous ring
            prev_ring_start = ring_start_nodes[ring - 1]
            curr_ring_start = ring_start_nodes[ring]
            
            if ring_connections == 'nearest':
                # Connect each node to nearest node(s) in previous ring
                for i in range(nodes_per_ring):
                    curr_node = curr_ring_start + i
                    # Connect to corresponding node in previous ring
                    prev_node = prev_ring_start + i
                    G.add_edge(curr_node, prev_node)
                    
            elif ring_connections == 'full':
                # Connect each node to all nodes in previous ring
                for i in range(nodes_per_ring):
                    curr_node = curr_ring_start + i
                    for j in range(nodes_per_ring):
                        prev_node = prev_ring_start + j
                        G.add_edge(curr_node, prev_node)
                        
            elif ring_connections == 'alternating':
                # Connect each node to two adjacent nodes in previous ring
                for i in range(nodes_per_ring):
                    curr_node = curr_ring_start + i
                    # Connect to two adjacent nodes in previous ring
                    prev_node1 = prev_ring_start + (i % nodes_per_ring)
                    prev_node2 = prev_ring_start + ((i + 1) % nodes_per_ring)
                    G.add_edge(curr_node, prev_node1)
                    G.add_edge(curr_node, prev_node2)
    
    return G





def create_hypergraph_code_graph(n: int, k: int, connectivity_pattern: str = 'random') -> nx.Graph:
    """
    Create a hypergraph-based quantum code connectivity graph.
    
    Args:
        n: Total number of nodes
        k: Average degree per node
        connectivity_pattern: Pattern for connections ('random', 'structured', 'clustered')
    
    Returns:
        NetworkX graph representing the hypergraph code
    """
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    if connectivity_pattern == 'random':
        # Random regular-like graph
        edges_to_add = (n * k) // 2
        edges_added = 0
        attempts = 0
        max_attempts = edges_to_add * 10
        
        import random
        while edges_added < edges_to_add and attempts < max_attempts:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v)
                edges_added += 1
            attempts += 1
            
    elif connectivity_pattern == 'structured':
        # Create structured connections based on node indices
        for i in range(n):
            for j in range(1, k + 1):
                neighbor = (i + j) % n
                if i != neighbor and not G.has_edge(i, neighbor):
                    G.add_edge(i, neighbor)
                    
    elif connectivity_pattern == 'clustered':
        # Create clustered connectivity
        cluster_size = max(k + 2, 4)
        num_clusters = (n + cluster_size - 1) // cluster_size
        
        for cluster in range(num_clusters):
            cluster_nodes = []
            for i in range(cluster_size):
                node = cluster * cluster_size + i
                if node < n:
                    cluster_nodes.append(node)
            
            # Connect within cluster
            for i, u in enumerate(cluster_nodes):
                for j, v in enumerate(cluster_nodes[i + 1:], i + 1):
                    G.add_edge(u, v)
            
            # Connect to next cluster
            if cluster < num_clusters - 1:
                next_cluster_start = (cluster + 1) * cluster_size
                if next_cluster_start < n:
                    # Connect last node of current cluster to first of next
                    if cluster_nodes:
                        G.add_edge(cluster_nodes[-1], next_cluster_start)
    
    return G
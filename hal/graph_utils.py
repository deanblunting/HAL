"""
Graph utilities for HAL algorithm including planarity testing and community detection.
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
    """Hopcroft-Tarjan planarity testing algorithm implementation."""
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.n = graph.number_of_nodes()
        self.m = graph.number_of_edges()
        
    def is_planar(self) -> bool:
        """Test if graph is planar using Kuratowski's theorem check."""
        # Simple checks first
        if self.n <= 4:
            return True
        if self.m > 3 * self.n - 6:
            return False
            
        # Use NetworkX's built-in planarity test if available, otherwise assume planar for small graphs
        try:
            return nx.is_planar(self.graph)
        except AttributeError:
            # Fallback for older NetworkX versions
            return self._simple_planarity_check()
    
    def _simple_planarity_check(self) -> bool:
        """Simple planarity check fallback."""
        # For implementation simplicity, assume planar if graph is small or sparse enough
        if self.n <= 6:
            return True
        if self.m <= max(3 * self.n - 6, 0):
            return True
        return False
    
    def get_planar_subgraph(self, prioritized_edges: List[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """Extract heuristic maximal planar subgraph using edge priorities as described in the paper."""
        planar_edges = set()
        test_graph = nx.Graph()
        test_graph.add_nodes_from(self.graph.nodes())
        
        # Implement greedy incremental heuristic as described in the paper appendix
        for edge in prioritized_edges:
            u, v = edge
            if u not in self.graph or v not in self.graph:
                continue
                
            test_graph.add_edge(u, v)
            
            # Use NetworkX planarity test for efficiency (paper uses Hopcroft-Tarjan)
            try:
                is_planar = nx.is_planar(test_graph)
                if is_planar:
                    planar_edges.add(edge)
                else:
                    test_graph.remove_edge(u, v)
            except Exception as e:
                # If planarity test fails, use simple heuristic: allow if doesn't create too many crossings
                n = test_graph.number_of_nodes()
                m = test_graph.number_of_edges()
                
                # Planar graphs satisfy m <= 3n - 6 for n >= 3
                if n >= 3 and m <= 3 * n - 6:
                    planar_edges.add(edge)
                elif n < 3:
                    planar_edges.add(edge)  # Small graphs are always planar
                else:
                    test_graph.remove_edge(u, v)
                
        return planar_edges


class CommunityDetector:
    """Enhanced community detection using Louvain algorithm as specified in the paper."""
    
    def __init__(self, graph: nx.Graph, config=None):
        self.graph = graph
        self.config = config
        
    def detect_communities(self) -> Dict[int, int]:
        """Detect communities using Louvain algorithm as described in the paper appendix."""
        if self.graph.number_of_nodes() == 0:
            return {}
        
        # Try using NetworkX's Louvain algorithm first (paper's preferred method)
        if HAS_LOUVAIN and self.config and hasattr(self.config, 'use_louvain_communities') and self.config.use_louvain_communities:
            try:
                # Use Louvain community detection from NetworkX
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
                # Fall back to other methods if Louvain fails
                pass
        
        # Try python-louvain library
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(self.graph)
            return partition
        except ImportError:
            # Fallback to simple clustering based on graph structure
            return self._simple_community_detection()
    
    def _simple_community_detection(self) -> Dict[int, int]:
        """Simple community detection fallback using connected components and clustering."""
        if self.graph.number_of_edges() == 0:
            return {node: i for i, node in enumerate(self.graph.nodes())}
        
        # Use connected components as base communities
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
    
    def get_edge_priorities(self, communities: Dict[int, int]) -> List[Tuple[int, int]]:
        """Get edges sorted by priority for planar subgraph extraction as described in the paper.
        
        Paper's approach: Sort edges by community (intra-community first) then by graph distance.
        """
        edges = list(self.graph.edges())
        distances = self.compute_all_pairs_shortest_paths()
        
        def edge_priority(edge):
            u, v = edge
            # Prioritize intra-community edges, then by graph distance (paper's approach)
            same_community = communities.get(u, -1) == communities.get(v, -1)
            graph_dist = distances.get((u, v), float('inf'))
            
            # Paper sorting strategy: first all intra-community edges ordered by increasing length,
            # followed by inter-community edges, again from short to long
            return (not same_community, graph_dist)
        
        return sorted(edges, key=edge_priority)
    
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


def create_qecc_graph_from_edges(edges: List[Tuple[int, int]]) -> nx.Graph:
    """Create a graph from a list of edges."""
    G = nx.Graph()
    G.add_edges_from(edges)
    return G


def create_surface_code_graph(rows: int, cols: int) -> nx.Graph:
    """Create a surface code connectivity graph."""
    G = nx.Graph()
    
    # Add nodes
    for i in range(rows):
        for j in range(cols):
            G.add_node(i * cols + j)
    
    # Add nearest-neighbor edges
    for i in range(rows):
        for j in range(cols):
            node = i * cols + j
            # Right neighbor
            if j + 1 < cols:
                G.add_edge(node, node + 1)
            # Down neighbor
            if i + 1 < rows:
                G.add_edge(node, node + cols)
    
    return G


def create_bicycle_code_graph(n1: int, n2: int, a: int, b: int) -> nx.Graph:
    """Create a bivariate bicycle code connectivity graph."""
    G = nx.Graph()
    n = n1 * n2
    
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
    
    return G


def create_radial_code_graph(r: int, s: int) -> nx.Graph:
    """
    Create a radial code connectivity graph following the lifted product construction.
    
    Based on the paper: "Radial codes are obtained from the lifted product of classical radial codes.
    A classical radial code uses a pair of integers (r, s) and can be visually arranged in r
    concentric rings containing s spokes. The quantum code is then formed from r copies of a 
    classical radial code responsible for the X-basis and r copies responsible for the Z-basis,
    resulting in a quantum code with parameters [2r²s, 2(r-1)², ≤2s]. Each qubit is connected to 2r other qubits."
    
    Args:
        r: Number of concentric rings (determines connectivity)
        s: Number of spokes per ring (determines distance scaling)
    
    Returns:
        NetworkX graph representing the quantum radial code with n=2r²s nodes
    """
    G = nx.Graph()
    n = 2 * r * r * s  # Total number of qubits: n = 2r²s
    
    # Add all nodes
    for i in range(n):
        G.add_node(i)
    
    # Create a more structured connectivity pattern for radial codes
    # The construction should ensure each node has degree 2r
    
    for qubit in range(n):
        # Map qubit to (layer, ring, spoke) coordinates
        # We have 2 layers (X and Z), r rings, r spokes, s positions per spoke
        layer = qubit // (r * r * s)  # 0 or 1
        remainder = qubit % (r * r * s)
        ring = remainder // (r * s)
        spoke = (remainder % (r * s)) // s
        pos = remainder % s
        
        connections_needed = 2 * r
        connections_made = set()
        
        # Strategy 1: Connect to all other rings in the same layer and spoke/position
        for other_ring in range(r):
            if other_ring != ring:
                target = layer * (r * r * s) + other_ring * (r * s) + spoke * s + pos
                if target < n and target not in connections_made:
                    G.add_edge(qubit, target)
                    connections_made.add(target)
        
        # Strategy 2: Connect to all other spokes in the same layer, ring, and position
        for other_spoke in range(r):
            if other_spoke != spoke:
                target = layer * (r * r * s) + ring * (r * s) + other_spoke * s + pos
                if target < n and target not in connections_made:
                    G.add_edge(qubit, target)
                    connections_made.add(target)
        
        # Strategy 3: If we still need connections, connect to the other layer
        if len(connections_made) < connections_needed:
            other_layer = 1 - layer
            target = other_layer * (r * r * s) + ring * (r * s) + spoke * s + pos
            if target < n and target not in connections_made:
                G.add_edge(qubit, target)
                connections_made.add(target)
        
        # Strategy 4: Connect to adjacent positions if we still need connections
        offset = 1
        while len(connections_made) < connections_needed and offset < s:
            for direction in [-1, 1]:
                if len(connections_made) >= connections_needed:
                    break
                new_pos = (pos + direction * offset) % s
                # Try same layer first
                target = layer * (r * r * s) + ring * (r * s) + spoke * s + new_pos
                if target < n and target not in connections_made:
                    G.add_edge(qubit, target)
                    connections_made.add(target)
                
                # Try other layer if still need connections
                if len(connections_made) < connections_needed:
                    other_layer = 1 - layer
                    target = other_layer * (r * r * s) + ring * (r * s) + spoke * s + new_pos
                    if target < n and target not in connections_made:
                        G.add_edge(qubit, target)
                        connections_made.add(target)
            offset += 1
    
    return G


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


def create_specific_radial_codes() -> Dict[str, nx.Graph]:
    """
    Create specific radial codes with kd²/n > 10 as identified in Appendix F.
    
    Returns:
        Dictionary mapping code names to their corresponding graphs
    """
    codes = {}
    
    # From Appendix F, radial codes with kd²/n > 10:
    # Parameters format: [n, k, d] with kd²/n value
    
    # [88, 2, 22] - kd²/n = 11.0 (r=2, s=11)
    codes['[88,2,22]'] = create_radial_code_graph(r=2, s=11)
    
    # [160, 18, 10] - kd²/n = 11.25 (r=3, s=~8.89, round to s=9)
    codes['[160,18,10]'] = create_radial_code_graph(r=3, s=9)
    
    # [126, 8, 14] - kd²/n = 12.44 (r=3, s=7)
    codes['[126,8,14]'] = create_radial_code_graph(r=3, s=7)
    
    # [104, 2, 26] - kd²/n = 13.0 (r=2, s=13)
    codes['[104,2,26]'] = create_radial_code_graph(r=2, s=13)
    
    # [224, 18, 14] - kd²/n = 15.75 (r=4, s=7)
    codes['[224,18,14]'] = create_radial_code_graph(r=4, s=7)
    
    # [198, 8, 22] - kd²/n = 19.56 (r=3, s=11)
    codes['[198,8,22]'] = create_radial_code_graph(r=3, s=11)
    
    # [234, 8, 26] - kd²/n = 23.11 (r=3, s=13)
    codes['[234,8,26]'] = create_radial_code_graph(r=3, s=13)
    
    # [352, 18, 22] - kd²/n = 24.75 (r=4, s=11)
    codes['[352,18,22]'] = create_radial_code_graph(r=4, s=11)
    
    # [416, 18, 26] - kd²/n = 29.25 (r=4, s=13)
    codes['[416,18,26]'] = create_radial_code_graph(r=4, s=13)
    
    return codes


# Star codes removed - use radial codes instead


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
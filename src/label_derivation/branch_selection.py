"""
Find the single skeleton branch (segment) nearest a tumour mask,
within a larger branching vessel skeleton graph.

This is a replacement for when the vessel is
a branching tree (e.g. veins) rather than a single tube.
"""

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from scipy.ndimage import label
from angle_quantify import skeleton_to_graph


# ------------------------- Classify nodes -------------------------

def classify_nodes(G: nx.Graph):
    """
    Returns:
        structural_nodes: list of nodes with degree != 2
                           (tips = degree 1, branch points = degree >= 3)
    """
    structural_nodes = [n for n in G.nodes if G.degree(n) != 2]
    return structural_nodes


# ------------------------- Nearest node to tumour -------------------------

def nearest_skeleton_node_to_tumour(G: nx.Graph, tumour_mask: np.ndarray):
    """
    Find the skeleton graph node closest to any tumour voxel.
    Uses a KD-tree for speed (tumour masks can have thousands of voxels).
    """
    tumour_coords = np.argwhere(tumour_mask > 0)
    if len(tumour_coords) == 0:
        raise ValueError("Tumour mask is empty.")

    tree = cKDTree(tumour_coords)

    node_ids = list(G.nodes)
    node_coords = np.array([G.nodes[n]['xyz'] for n in node_ids])

    dists, _ = tree.query(node_coords)

    best_idx = np.argmin(dists)
    best_node = node_ids[best_idx]
    best_dist = dists[best_idx]

    return best_node, best_dist


# ------------------------- Walk outward to structural nodes -------------------------

def walk_to_structural_node(G: nx.Graph, start_node, next_node, structural_set):
    """
    Walk along the graph starting at start_node -> next_node,
    continuing until a structural node (tip or branch point) is reached.
    Returns the structural node found.
    """
    prev, curr = start_node, next_node
    while curr not in structural_set:
        neighbors = [n for n in G.neighbors(curr) if n != prev]
        if not neighbors:
            return curr
        prev, curr = curr, neighbors[0]
    return curr


def enclosing_segment_endpoints(G: nx.Graph, tumour_near_node, structural_nodes):
    """
    Given the node closest to the tumour, find the two structural nodes
    (tip or branch point) that bound the segment it lies on.

    Handles the edge case where tumour_near_node is itself structural
    (e.g. the tumour sits right at a bifurcation) by just returning
    that node twice
    """
    structural_set = set(structural_nodes)

    if tumour_near_node in structural_set:
        return None

    endpoints = []
    for neighbor in G.neighbors(tumour_near_node):
        endpoint = walk_to_structural_node(G, tumour_near_node, neighbor, structural_set)
        endpoints.append(endpoint)

    return endpoints


# ------------------------- Extract path coords -------------------------

def get_branch_path_coords(G: nx.Graph, tumour_mask: np.ndarray):
    """
    Main entry point. Returns (N,3) array of voxel coords (z,y,x)
    for the single branch nearest the tumour
    """
    structural_nodes = classify_nodes(G)
    tumour_near_node, dist = nearest_skeleton_node_to_tumour(G, tumour_mask)

    endpoints = enclosing_segment_endpoints(G, tumour_near_node, structural_nodes)

    if endpoints is None:
        neighbor_endpoints = []
        for neighbor in G.neighbors(tumour_near_node):
            ep = walk_to_structural_node(G, tumour_near_node, neighbor, set(structural_nodes))
            neighbor_endpoints.append(ep)
        if len(neighbor_endpoints) < 2:
            raise ValueError("Tumour-nearest node has fewer than 2 branches; cannot form a path.")
        a, b = neighbor_endpoints[0], neighbor_endpoints[1]
    else:
        a, b = endpoints[0], endpoints[1]

    path_nodes = nx.shortest_path(G, a, b, weight='weight')
    coords = np.array([G.nodes[n]['xyz'] for n in path_nodes], dtype=float)

    print(f"[branch_selection] tumour-nearest node at distance {dist:.2f} voxels; "
          f"segment length = {len(coords)} points")

    return coords

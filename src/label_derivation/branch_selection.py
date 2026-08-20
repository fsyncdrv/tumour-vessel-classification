"""
The PanTS dataset does not have per-vessel labels for the veins. As a result, it is
not possible to use Zhang et al's centreline approach which assumes you already know
which vessel is being measured.

Since the veins segmentation mask is a continous structure with multiple branches, we first need to determine
which branch is more connected to the tumour before running the angle quantification.
"""
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from scipy.ndimage import label
from angle_quantify import skeleton_to_graph


## Determine how many tumour blobs are present on a mask. (During a check (isotropic_resampling.ipynb),
# it was discovered that some tumour masks were showing multiple lesions).
# This function was added to account for that
def get_tumour_components(tumour_mask: np.ndarray, min_voxels=50):
    labeled, n_components = label(tumour_mask)
    components = []
    for i in range(1, n_components + 1):
        component = (labeled == i).astype(np.uint8)
        if component.sum() >= min_voxels:
            components.append(component)
    return components


## degree 1 = end point
## degree 2 = middle points
## degree >= 3 = connection with another branch
## This keeps only 1 and 3 to determine the shape of the veins structure
def classify_nodes(G: nx.Graph):
    structural_nodes = [n for n in G.nodes if G.degree(n) != 2]
    return structural_nodes


def nearest_skeleton_node_to_tumour(G: nx.Graph, tumour_mask: np.ndarray):
    """
    Find the skeleton graph node closest to any tumour voxel.
    Uses a KD-tree to find nearby points faster
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


def walk_to_structural_node(G: nx.Graph, start_node, next_node, structural_set):
    """
    Moves outward from the tumour-nearest point until it reaches the nearest
    endpoint or connection to another branch.
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
    Returns the endpoints of the branch closest to tumour
    """
    structural_set = set(structural_nodes)

    if tumour_near_node in structural_set:
        return None

    endpoints = []
    for neighbor in G.neighbors(tumour_near_node):
        endpoint = walk_to_structural_node(G, tumour_near_node, neighbor, structural_set)
        endpoints.append(endpoint)

    return endpoints



def get_branch_path_coords(G: nx.Graph, tumour_mask: np.ndarray):
    """
    Finds the vessel branch nearestr to tumour and returns its (z,y,x) coordinates
    """
    structural_nodes = classify_nodes(G)
    tumour_near_node, dist = nearest_skeleton_node_to_tumour(G, tumour_mask)
    endpoints = enclosing_segment_endpoints(G, tumour_near_node, structural_nodes)

    if endpoints is None:
        neighbor_endpoints = []
        for neighbor in G.neighbors(tumour_near_node):
            ep = walk_to_structural_node(G, tumour_near_node, neighbor, set(structural_nodes))
            neighbor_endpoints.append(ep)
        if len(neighbor_endpoints) >= 2:
            a, b = neighbor_endpoints[0], neighbor_endpoints[1]
        elif len(neighbor_endpoints) == 1:
            a = tumour_near_node
            b = neighbor_endpoints[0]
        else:
            raise ValueError("Tumour-nearest node is isolated; cannot form a path.")
    else:
        a, b = endpoints[0], endpoints[1]

    path_nodes = nx.shortest_path(G, a, b, weight='weight')
    coords = np.array([G.nodes[n]['xyz'] for n in path_nodes], dtype=float)

    print(f"[branch_selection] tumour-nearest node at distance {dist:.2f} voxels; "
          f"segment length = {len(coords)} points")

    return coords

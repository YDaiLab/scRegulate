import numpy as np
import math
import networkx as nx
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm, colors as mcolors
from scipy.spatial import KDTree


def run_affinity_propagation(W_normalized, damping=0.99, preference=-90.0, random_state=42):
    """
    Run Affinity Propagation clustering on the given normalized matrix.

    Parameters:
    - W_normalized (pd.DataFrame): Matrix with TFs as columns.
    - damping (float): Damping factor for AP (default: 0.99).
    - preference (float): Preference for AP (default: -90.0).
    - random_state (int): Random seed (default: 42).

    Returns:
    - ap_clusters (dict): Clusters with TFs grouped.
    """
    if W_normalized.isnull().values.any():
        raise ValueError("Input matrix contains NaN values. Please preprocess the data.")

    tf_similarity = cosine_similarity(W_normalized.T)
    ap = AffinityPropagation(
        affinity='precomputed',
        damping=damping,
        preference=preference,
        random_state=random_state
    )
    ap.fit(tf_similarity)
    #np.fill_diagonal(tf_similarity, 0)

    numeric_labels = ap.labels_
    unique_ids = np.unique(numeric_labels)
    id_to_label = {old_id: f"(co)regulatory cluster {i + 1}" for i, old_id in enumerate(unique_ids)}
    ap_labels = np.array([id_to_label[lab] for lab in numeric_labels], dtype=object)

    ap_clusters = {}
    for idx, cluster_label in enumerate(ap_labels):
        if cluster_label not in ap_clusters:
            ap_clusters[cluster_label] = []
        ap_clusters[cluster_label].append(W_normalized.columns[idx])

    return ap_clusters


def apply_cluster_repulsion(pos, clusters, repulsion_factor):
    """
    Apply repulsion between clusters to prevent overlap.

    Parameters:
    - pos (dict): Node positions.
    - clusters (dict): Cluster assignments.
    - repulsion_factor (float): Strength of the repulsion.

    Returns:
    - dict: Updated positions.
    """
    cluster_keys = list(clusters.keys())
    cluster_centroids = {
        key: np.mean([pos[node] for node in clusters[key]], axis=0)
        for key in cluster_keys
    }

    centroids = np.array(list(cluster_centroids.values()))
    tree = KDTree(centroids)

    for cluster1_idx, centroid1 in enumerate(centroids):
        neighbors = tree.query_ball_point(centroid1, r=repulsion_factor)
        for cluster2_idx in neighbors:
            if cluster1_idx != cluster2_idx:
                cluster1_key = cluster_keys[cluster1_idx]
                cluster2_key = cluster_keys[cluster2_idx]
                centroid2 = centroids[cluster2_idx]
                dx, dy = centroid2 - centroid1
                distance = np.sqrt(dx**2 + dy**2)
                if distance < repulsion_factor:
                    force = repulsion_factor - distance
                    dx_norm, dy_norm = dx / distance, dy / distance
                    for node in clusters[cluster1_key]:
                        pos[node][0] -= force * dx_norm / len(clusters[cluster1_key])
                        pos[node][1] -= force * dy_norm / len(clusters[cluster1_key])
                    for node in clusters[cluster2_key]:
                        pos[node][0] += force * dx_norm / len(clusters[cluster2_key])
                        pos[node][1] += force * dy_norm / len(clusters[cluster2_key])
    return pos


def apply_within_cluster_repulsion(pos, clusters, repulsion_factor=0.2):
    """
    Apply repulsion within clusters to increase spacing.

    Parameters:
    - pos (dict): Node positions.
    - clusters (dict): Cluster assignments.
    - repulsion_factor (float): Strength of the repulsion within clusters.

    Returns:
    - dict: Updated positions.
    """
    for cluster_nodes in clusters.values():
        for i, node1 in enumerate(cluster_nodes):
            for j, node2 in enumerate(cluster_nodes):
                if i >= j:
                    continue
                x1, y1 = pos[node1]
                x2, y2 = pos[node2]
                dx, dy = x2 - x1, y2 - y1
                distance = math.sqrt(dx**2 + dy**2)
                if distance < repulsion_factor:
                    force = (repulsion_factor - distance) / repulsion_factor
                    dx_norm, dy_norm = dx / distance, dy / distance
                    pos[node1][0] -= force * dx_norm / 2
                    pos[node1][1] -= force * dy_norm / 2
                    pos[node2][0] += force * dx_norm / 2
                    pos[node2][1] += force * dy_norm / 2
    return pos


def apply_node_overlap_repulsion(pos, repulsion_factor=0.05):
    """
    Apply node-level repulsion to prevent overlapping nodes.

    Parameters:
    - pos (dict): Node positions.
    - repulsion_factor (float): Strength of the repulsion between nodes.

    Returns:
    - dict: Updated positions.
    """
    nodes = list(pos.keys())
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if i >= j:
                continue
            x1, y1 = pos[node1]
            x2, y2 = pos[node2]
            dx = x2 - x1
            dy = y2 - y1
            distance = math.sqrt(dx**2 + dy**2)
            if distance < repulsion_factor:
                force = (repulsion_factor - distance) / repulsion_factor
                dx_norm, dy_norm = dx / distance, dy / distance
                pos[node1][0] -= force * dx_norm / 2
                pos[node1][1] -= force * dy_norm / 2
                pos[node2][0] += force * dx_norm / 2
                pos[node2][1] += force * dy_norm / 2
    return pos


def compute_layout_quality(pos, G):
    """
    Compute a quality metric for the layout.

    Parameters:
    - pos (dict): Node positions.
    - G (networkx.Graph): Graph object.

    Returns:
    - float: Quality of the layout.
    """
    compactness = 0
    for u, v in G.edges():
        dx = pos[v][0] - pos[u][0]
        dy = pos[v][1] - pos[u][1]
        compactness += math.sqrt(dx**2 + dy**2)
    return -compactness


def optimize_layout(G, layout_function, iterations=10, **kwargs):
    """
    Optimize the layout by iterating and finding the best representation.

    Parameters:
    - G (networkx.Graph): Graph object.
    - layout_function (function): Layout function to use.
    - iterations (int): Number of iterations to optimize.
    - kwargs: Additional arguments for the layout function.

    Returns:
    - dict: Best layout positions.
    """
    best_pos = None
    best_quality = float("-inf")
    for _ in range(iterations):
        pos = layout_function(G, **kwargs)
        quality = compute_layout_quality(pos, G)
        if quality > best_quality:
            best_quality = quality
            best_pos = pos
    return best_pos


def get_top_nodes(G, clusters, top_n=5):
    """
    Select the top-n most important nodes from each cluster based on degree centrality.

    Parameters:
    - G (networkx.Graph): Graph object.
    - clusters (dict): Cluster assignments.
    - top_n (int): Number of top nodes to select per cluster.

    Returns:
    - dict: Top nodes per cluster.
    """
    node_importance = nx.degree_centrality(G)
    top_nodes = {}
    for cluster, nodes in clusters.items():
        ranked_nodes = sorted(nodes, key=lambda node: node_importance[node], reverse=True)
        top_nodes[cluster] = ranked_nodes[:top_n]
    return top_nodes




def visualize_tf_network(
    W_normalized,
    ap_clusters,
    threshold=0.3,
    figsize=(15, 12),
    edge_cmap="coolwarm",
    edge_alpha_scale=0.7,
    node_alpha=0.8,
    min_node_size=10,
    max_node_size=1000,
    thickness_scale=2,
    node_label_mode=4,
    specific_nodes_to_label=None,
    cluster_to_label=None,
    label_repel=True,
    repel_strength=0.2,
    max_displacement=0.05,
    node_repulsion_factor=0.05,
    within_cluster_repulsion=0.2,
    repulsion_factor=4,
    layout_iterations=10,  # Number of iterations for layout optimization
    top_n_nodes_per_cluster=5,  # Number of top nodes to label per cluster
    colorbar=True,
    cluster_colors=None,
    save_path=None,
):
    """
    Visualize a TF-TF network with optimized layout, cluster-based repulsion, and improved node/label handling.

    Parameters:
    - W_normalized (pd.DataFrame): Normalized matrix with TFs as columns.
    - ap_clusters (dict): Affinity Propagation clusters (output of `run_affinity_propagation`).
    - threshold (float): Minimum cosine similarity to include an edge.
    - figsize (tuple): Size of the figure (default: (15, 12)).
    - edge_cmap (str): Colormap for edges based on weight.
    - edge_alpha_scale (float): Scaling factor for edge transparency.
    - node_alpha (float): Transparency for nodes.
    - min_node_size (int): Minimum node size.
    - max_node_size (int): Maximum node size.
    - thickness_scale (float): Scaling factor for edge thickness.
    - node_label_mode (int): Mode for node labeling:
        1. Label all nodes.
        2. Label only specified nodes (`specific_nodes_to_label`).
        3. Label nodes in a specified cluster (`cluster_to_label`).
        4. Label top-n nodes based on degree centrality in each cluster.
    - specific_nodes_to_label (list): List of nodes to label if `node_label_mode=2`.
    - cluster_to_label (str): Cluster to label if `node_label_mode=3`.
    - label_repel (bool): Whether to apply label repulsion.
    - repel_strength (float): Strength of label repulsion.
    - max_displacement (float): Maximum displacement for label repulsion.
    - node_repulsion_factor (float): Strength of node-level repulsion.
    - within_cluster_repulsion (float): Strength of repulsion within clusters.
    - repulsion_factor (float): Strength of cluster-to-cluster repulsion.
    - layout_iterations (int): Number of iterations for layout optimization.
    - top_n_nodes_per_cluster (int): Number of top nodes to label per cluster if `node_label_mode=4`.
    - colorbar (bool): Whether to display the colorbar.
    - save_path (str): Path to save the output figure (default: None).

    Returns:
    - None: Displays the TF network plot.
    """
    G = nx.Graph()
    tf_similarity = cosine_similarity(W_normalized.T)

    # Add edges based on cosine similarity threshold
    for i, tf1 in enumerate(W_normalized.columns):
        for j, tf2 in enumerate(W_normalized.columns):
            if tf1 != tf2 and tf_similarity[i, j] > threshold:
                G.add_edge(tf1, tf2, weight=tf_similarity[i, j])

    # Filter clusters to include only nodes in the graph
    filtered_ap_clusters = {
        cluster_id: [node for node in nodes if node in G.nodes()]
        for cluster_id, nodes in ap_clusters.items()
    }

    # Define cluster colors
    if cluster_colors is None:
        cluster_colors = {
            cluster: color
            for cluster, color in zip(filtered_ap_clusters.keys(), sns.color_palette("tab10", len(filtered_ap_clusters)))
        }

    # Compute node sizes based on degree
    node_degrees = dict(G.degree())
    log_degrees = {node: math.log10(deg + 1) for node, deg in node_degrees.items()}
    log_min, log_max = min(log_degrees.values()), max(log_degrees.values())
    log_range = log_max - log_min if log_max != log_min else 1.0
    node_sizes_dict = {
        node: min_node_size + (log_degrees[node] - log_min) / log_range * (max_node_size - min_node_size)
        for node in G.nodes()
    }

    # Optimize layout
    pos = optimize_layout(
        G,
        nx.kamada_kawai_layout,
        iterations=layout_iterations,
    )
    pos = apply_cluster_repulsion(pos, filtered_ap_clusters, repulsion_factor)
    pos = apply_within_cluster_repulsion(pos, filtered_ap_clusters, within_cluster_repulsion)
    pos = apply_node_overlap_repulsion(pos, node_repulsion_factor)

    plt.figure(figsize=figsize)

    # Compute edge properties
    weights = [data["weight"] for _, _, data in G.edges(data=True)]
    norm = mcolors.Normalize(vmin=min(weights), vmax=max(weights))
    cmap = cm.get_cmap(edge_cmap)
    edge_thickness = [(math.exp(data["weight"] * thickness_scale) - 1) for _, _, data in G.edges(data=True)]
    edge_alpha = [
        edge_alpha_scale * (data["weight"] - min(weights)) / (max(weights) - min(weights))
        for _, _, data in G.edges(data=True)
    ]
    edge_colors = [cmap(norm(data["weight"])) for _, _, data in G.edges(data=True)]

    # Draw edges
    for (u, v, data), alpha, thickness, color in zip(G.edges(data=True), edge_alpha, edge_thickness, edge_colors):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], alpha=alpha, edge_color=[color], width=thickness)

    # Draw nodes
    for cluster_id, nodes in filtered_ap_clusters.items():
        cluster_node_sizes = [node_sizes_dict[n] for n in nodes]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes,
            node_size=cluster_node_sizes,
            alpha=node_alpha,
            node_color=[cluster_colors[cluster_id]] * len(nodes),
        )

    # Label nodes
    if node_label_mode in {1, 2, 3, 4}:
        labels = {}
        if node_label_mode == 1:
            labels = {node: node for node in G.nodes()}
        elif node_label_mode == 2 and specific_nodes_to_label:
            labels = {node: node for node in specific_nodes_to_label if node in G.nodes()}
        elif node_label_mode == 3 and cluster_to_label is not None:
            labels = {node: node for node in filtered_ap_clusters.get(cluster_to_label, [])}
        elif node_label_mode == 4:
            top_nodes = get_top_nodes(G, filtered_ap_clusters, top_n=top_n_nodes_per_cluster)
            labels = {node: node for nodes in top_nodes.values() for node in nodes}

        label_positions = {node: list(pos[node]) for node in labels.keys()}

        for node, label in labels.items():
            x, y = pos[node]
            if label_repel:
                delta_x, delta_y = 0, 0
                for other_node in labels:
                    if other_node == node:
                        continue
                    ox, oy = label_positions[other_node]
                    distance = np.sqrt((ox - x) ** 2 + (oy - y) ** 2)
                    if distance < repel_strength:
                        displacement = (repel_strength - distance) / repel_strength
                        delta_x += (x - ox) * displacement
                        delta_y += (y - oy) * displacement

                delta_x = max(-max_displacement, min(max_displacement, delta_x))
                delta_y = max(-max_displacement, min(max_displacement, delta_y))

                label_positions[node][0] += delta_x
                label_positions[node][1] += delta_y

                plt.plot(
                    [x, label_positions[node][0]],
                    [y, label_positions[node][1]],
                    color="gray",
                    linestyle="--",
                    linewidth=0.5,
                    alpha=0.7,
                )

            plt.text(
                label_positions[node][0],
                label_positions[node][1],
                label,
                fontsize=10,
                color="black",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.2),
            )

    # Add colorbar
    if colorbar:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, shrink=0.5)
        cbar.set_label("Cosine Similarity (Regulatory Strength)", fontsize=12)

    #plt.title("Cluster-Optimized TF-TF Network with Node and Label Repulsion", fontsize=16)
    if save_path:
        plt.savefig(save_path, format="png", dpi=300)
    plt.show()

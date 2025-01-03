import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


size = 5
n_clusters = size * size
adj_matrix = [
    [0, 1, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 0],
]
node_position = {
    1: (1.5, 2),
    2: (0.5, 1),
    3: (1.5, 1),
    4: (2.5, 1),
    5: (0.5, 0),
    6: (1.5, 0),
    7: (2.5, 0),
}


def build_cluster(G, offset):
    length = len(adj_matrix)
    for i in range(length):
        G.add_node(i + 1 + offset)
    for i in range(length):
        for j in range(length):
            if adj_matrix[i][j] == 1:
                G.add_edge(i + 1 + offset, j + 1 + offset)


def connect_clusters_regular(G, cluster_offsets):
    connections = []
    for i in range(n_clusters - 1):
        if (i + 1) % size != 0:
            connections.append(
                (cluster_offsets[i] + 4, cluster_offsets[i + 1] + 2)
            )
            connections.append(
                (cluster_offsets[i] + 7, cluster_offsets[i + 1] + 5)
            )
        if i < n_clusters - size:
            connections.append(
                (cluster_offsets[i] + 6, cluster_offsets[i + size] + 1)
            )

    for u, v in connections:
        G.add_edge(u, v, color="red")


def connect_clusters_irregular_green(G, cluster_offsets):
    connections = []
    for i in range(n_clusters // 2):
        connections.append(
            (cluster_offsets[i] + 1, cluster_offsets[-i - 1] + 7)
        )

    for u, v in connections:
        G.add_edge(u, v, color="green")


def connect_clusters_irregular_blue(G, cluster_offsets):
    connections = []
    for i in range(1, size):
        connections.append(
            (cluster_offsets[i] + 3, cluster_offsets[size * i] + 3)
        )
        connections.append(
            (
                cluster_offsets[n_clusters - size + i] + 3,
                cluster_offsets[n_clusters - size * (i + 1)] + 3,
            )
        )
    for i in range(size - 2, 0, -1):
        connections.append(
            (
                cluster_offsets[i] + 3,
                cluster_offsets[size * (size - i) - 1] + 3,
            )
        )
        connections.append(
            (
                cluster_offsets[n_clusters - size + i] + 3,
                cluster_offsets[size * i - 1 + size] + 3,
            )
        )
    for u, v in connections:
        G.add_edge(u, v, color="blue")


def generate_grid_topology(num_clusters, cluster_size):
    pos = {}
    cluster_position = {}
    i = 0
    for x, y in nx.grid_2d_graph(size, size):
        cluster_position[i] = np.array([y * (size + 1), -x * (size + 1)])
        i += 1
    cluster_centers = nx.spring_layout(
        range(num_clusters),
        pos=cluster_position,
        fixed=cluster_position,
        scale=cluster_size,
    )
    for i in range(len(cluster_centers)):
        shift = i * cluster_size
        node_pos_array = np.array(list(node_position.values()))
        for idx, node in enumerate(range(shift, shift + cluster_size)):
            pos[node + 1] = node_pos_array[idx] + cluster_centers[i]
    return pos


def calculate_topological_metrics(G):
    diameter = nx.diameter(G)
    average_diameter = round(nx.average_shortest_path_length(G), 3)
    degree = max(deg for node, deg in G.degree())
    cost = G.number_of_edges()
    topological_traffic = round((2 * average_diameter) / degree, 3)
    print(f"Number of nodes: {len(G.nodes())}")
    print(f"Diameter: {diameter}")
    print(f"Average Path Length: {average_diameter}")
    print(f"Degrees: {degree}")
    print(f"Cost (number of edges): {cost}")
    print(f"Topological Traffic: {topological_traffic}")


def build_multiple_clusters(num_clusters):
    G = nx.Graph()
    offset = 0
    cluster_offsets = []
    for cluster_num in range(num_clusters):
        build_cluster(G, offset)
        cluster_offsets.append(offset)
        offset += len(adj_matrix)

    connect_clusters_regular(G, cluster_offsets)
    connect_clusters_irregular_green(G, cluster_offsets)
    connect_clusters_irregular_blue(G, cluster_offsets)

    cluster_size = len(adj_matrix)
    pos = generate_grid_topology(num_clusters, cluster_size)

    if size < 6:
        edge_colors = [
            "black" if not G[u][v].get("color") else G[u][v]["color"]
            for u, v in G.edges()
        ]
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color="lightblue",
            node_size=300,
            font_size=12,
            font_weight="bold",
            alpha=1.0,
        )
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2)
        plt.show()

    calculate_topological_metrics(G)


if __name__ == "__main__":
    build_multiple_clusters(n_clusters)

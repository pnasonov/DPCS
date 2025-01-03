import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

cluster_count = 1
base_matrix = [
    [0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0],
    [1, 0, 0, 1, 0, 1, 0],
    [1, 1, 1, 0, 1, 1, 1],
    [0, 1, 0, 1, 0, 0, 1],
    [0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0],
]
node_positions = {
    1: (-4, -1.5),
    2: (4, -1.5),
    3: (-5, 0),
    4: (0, -0.5),
    5: (5, 0),
    6: (-4, 1),
    7: (4, 1),
}


def create_cluster(graph, offset, matrix):
    size = len(matrix)
    for i in range(size):
        graph.add_node(i + 1 + offset)
    for i in range(size):
        for j in range(size):
            if matrix[i][j] == 1:
                graph.add_edge(i + 1 + offset, j + 1 + offset)


def add_connections(graph, offsets, rules, color):
    edges = [
        (offsets[i] + rule[0], offsets[(i + rule[1]) % len(offsets)] + rule[2])
        for i in range(len(offsets) - 1)
        for rule in rules
    ]
    for u, v in edges:
        graph.add_edge(u, v, color=color)


def layout_clusters(num_clusters, cluster_size):
    pos = {}
    cluster_shift = {
        i: np.array([i * cluster_size * 2.5, 0]) for i in range(num_clusters)
    }
    centers = nx.spring_layout(
        range(cluster_count),
        pos=cluster_shift,
        fixed=cluster_shift,
        scale=cluster_size,
    )
    for i in range(num_clusters):
        base_pos = node_positions.copy()
        base_pos[4] = (0, -0.5) if i % 2 == 0 else (0, -1)
        nodes = np.array(list(base_pos.values()))
        offset = i * cluster_size
        for idx, node_id in enumerate(range(offset, offset + cluster_size)):
            pos[node_id + 1] = nodes[idx] + centers[i]
    return pos


def analyze_graph(graph):
    metrics = {
        "Diameter": nx.diameter(graph),
        "Average Path Length": round(
            nx.average_shortest_path_length(graph), 3
        ),
        "Max Degree": max(degree for _, degree in graph.degree()),
        "Edge Count": graph.number_of_edges(),
        "Traffic": round(
            (2 * nx.average_shortest_path_length(graph))
            / max(degree for _, degree in graph.degree()),
            3,
        ),
    }
    for key, value in metrics.items():
        print(f"{key}: {value}")


def build_network(num_clusters):
    graph = nx.Graph()
    offset = 0
    cluster_offsets = []

    for _ in range(num_clusters):
        create_cluster(graph, offset, base_matrix)
        cluster_offsets.append(offset)
        offset += len(base_matrix)

    add_connections(
        graph, cluster_offsets, [(5, 1, 3), (7, 1, 6), (2, 1, 1)], "red"
    )
    add_connections(graph, cluster_offsets, [(4, 2, 4)], "green")
    add_connections(
        graph, cluster_offsets, [(7, cluster_count // 2, 1)], "blue"
    )

    cluster_size = len(base_matrix)
    positions = layout_clusters(num_clusters, cluster_size)

    if cluster_count < 10:
        colors = [
            "black" if not graph[u][v].get("color") else graph[u][v]["color"]
            for u, v in graph.edges()
        ]
        nx.draw(
            graph,
            positions,
            with_labels=True,
            node_color="lightblue",
            node_size=300,
            font_size=12,
            font_weight="bold",
        )
        nx.draw_networkx_edges(graph, positions, edge_color=colors, width=2)
        plt.show()

    analyze_graph(graph)


if __name__ == "__main__":
    build_network(cluster_count)

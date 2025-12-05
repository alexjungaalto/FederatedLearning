import subprocess
import time
import networkx as nx
import matplotlib.pyplot as plt

# Global TV regularization factor (same for all workers)
alpha = 10

# Define worker topology and parameters
topology = {
    1: {"a": 1.0, "neighbors": [2], "lr": 0.01, "speed": 10},
    2: {"a": 2.0, "neighbors": [1], "lr": 0.01, "speed": 10}#,
  #  3: {"a": 3.0, "neighbors": [1, 2, 4], "lr": 0.1, "speed": 2},
  #  4: {"a": 4.0, "neighbors": [3], "lr": 0.1, "speed": 2}
}

# Launch workers
processes = []
for worker_id, data in topology.items():
    cmd = ["python", "worker.py", str(worker_id), str(data["a"]), str(data["lr"]), str(alpha), str(data["speed"])] + list(map(str, data["neighbors"]))   
    print(cmd)
    p = subprocess.Popen(cmd)
    processes.append(p)
    time.sleep(1)  # Staggered start

# Visualize topology
def visualize_topology(topology):
    G = nx.Graph()

    # Add nodes
    for worker_id, data in topology.items():
        G.add_node(worker_id, size=data["speed"] * 300)  # Scale node size by speed

    # Add edges
    for worker_id, data in topology.items():
        for neighbor in data["neighbors"]:
            if not G.has_edge(worker_id, neighbor):  # Avoid duplicate edges
                G.add_edge(worker_id, neighbor)

    # Get node sizes based on speed
    node_sizes = [nx.get_node_attributes(G, 'size')[n] for n in G.nodes()]

    # Draw graph
    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(G)  # Layout for visualization
    nx.draw(G, pos, with_labels=True, node_size=node_sizes, edge_color="gray", font_weight="bold", alpha=0.8)

    plt.title("Worker Topology (Node Size = Speed)")
    plt.show()

# Call visualization function
visualize_topology(topology)

# Wait for all processes
for p in processes:
    p.wait()

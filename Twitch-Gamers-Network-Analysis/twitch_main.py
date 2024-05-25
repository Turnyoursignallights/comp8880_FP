
import networkx as nx
import random
import pandas as pd
import matplotlib.pyplot as plt
# Read the edges CSV file
#edges_df = pd.read_csv('large_twitch_edges.csv')
edges_df_sampled = pd.read_csv('large_twitch_edges.csv', nrows=None)
sample_percentage = 0.002
edges_df = edges_df_sampled.sample(frac=sample_percentage, random_state=1)
# Create a graph from the edges
g = nx.from_pandas_edgelist(edges_df, source='numeric_id_1', target='numeric_id_2')

# Identify connected components in the graph
connected_components = list(nx.connected_components(g))

# Find the largest connected component
largest_component = max(connected_components, key=len)

# Create a subgraph from the largest connected component
largest_subgraph = g.subgraph(largest_component)


# define Watts Strogatz graph
def create_watts_strogatz_graph(num_nodes, mean_degree, probability):
    graph = nx.Graph()

    # A ring lattice with num_nodes and degree
    for node in range(num_nodes):
        if node % 100 == 0:
            print(f"Creating ring lattice: Node {node}/{num_nodes}")
        for neighbor in range(1, mean_degree // 2 + 1):
            graph.add_edge(node, (node + neighbor) % num_nodes)

    # Rewire edges with probability p
    nodes = set(graph.nodes())
    for node in nodes:
        neighbors = list(graph.neighbors(node))
        for neighbor in neighbors:
            if random.random() < probability:
                possible_new_neighbors = nodes - set(graph.neighbors(node)) - {node}
                if possible_new_neighbors:  # 确保还有可选的新邻居
                    new_neighbor = random.choice(list(possible_new_neighbors))
                    graph.remove_edge(node, neighbor)
                    graph.add_edge(node, new_neighbor)

    return graph


# Number of nodes
n = largest_subgraph.order()

# Each node is connected to k nearest neighbors
k = largest_subgraph.size()

# Probability of rewiring each edge
p = 0.2

# Calculating Original Network size, degree, average path length, and clustering
print("Nodes: ", n)
print("Edges (Size): ", k)
print("Average degree: ", float(k * 2) / n)
print("Average clustering: ", nx.average_clustering(largest_subgraph))
print("Average path length: ", nx.average_shortest_path_length(largest_subgraph))

# Create a Watts-Strogatz model network
watts_strogatz_graph = create_watts_strogatz_graph(n, k, p)

# Calculating average path length, and clustering of the watts_strogatz graph
print("Watts Strogatz Graph")
print("Average clustering: ", nx.average_clustering(watts_strogatz_graph))
print("Average path length: ", nx.average_shortest_path_length(watts_strogatz_graph))


# define Barabasi-Albert method
def create_barabasi_albert_graph(initial_number_of_nodes, number_of_expected_connections, time_to_run):
    # Create an initial complete graph with m0 nodes
    graph = nx.complete_graph(initial_number_of_nodes)

    # Instead of checking graph.degree(), we should ensure the initial graph is connected
    # By using a complete graph, we are already ensuring that each node has at least m0 - 1 edges
    # Thus, there is no need for the if graph.degree() < 1 check

    # Start adding new nodes to the graph one by one
    for time in range(time_to_run):
        if time % 100 == 0:  # Print progress every 100 iterations
            print(f"Adding nodes: {time}/{time_to_run}")

        new_node = graph.number_of_nodes()  # Get the number for the new node
        graph.add_node(new_node)  # Add this new node to the graph

        # The sum of the degrees of all nodes in the graph
        sum_degree = sum(dict(graph.degree()).values())

        # The probability for each existing node to be connected to the new node
        probabilities = [graph.degree(node) / sum_degree for node in graph.nodes()]
        # Select m random existing nodes based on the calculated probabilities
        # It's important to allow nodes to be picked more than once to match m connections
        target_nodes = random.choices(list(graph.nodes()), weights=probabilities, k=number_of_expected_connections)

        # Add an edge to each selected target node
        for target_node in target_nodes:
            graph.add_edge(new_node, target_node)

    print(f"Barabasi-Albert model completed with {graph.number_of_nodes()} nodes.")
    return graph


# Now update the parameters according to your largest connected component
m0 = max(len(component) for component in connected_components)  # Number of nodes
m = round(sum(dict(largest_subgraph.degree()).values()) / m0)  # Average degree

# Time to run
t = 1000

# Generate a scale-free network using Preferential Attachment
barabasi_albert_graph = create_barabasi_albert_graph(m0, m, t)

# Calculating average path length, and clustering of the watts_strogatz graph
print("Barabasi Albert Graph")
print("Average clustering: ", nx.average_clustering(barabasi_albert_graph))
print("Average path length: ", nx.average_shortest_path_length(barabasi_albert_graph))


# ===============================================================================================================
# Visualize the network
def draw_watts_strogatz(ws_graph, filename='Watts-Strogatz_graph.png'):
    plt.figure(figsize=(16, 16))
    pos = nx.spectral_layout(ws_graph)
    nx.draw_networkx_nodes(ws_graph, pos, node_size=500, node_color='skyblue', edgecolors='darkblue')
    nx.draw_networkx_edges(ws_graph, pos, alpha=0.5, edge_color='gray')
    nx.draw_networkx_labels(ws_graph, pos, font_size=10, font_color='black', font_weight='bold')
    plt.title("Watts-Strogatz Model")
    plt.axis('off')
    plt.savefig('watts_strogatz_graph.png')
    plt.close()

# Visualize the network using circular layout
def draw_barabasi_albert(ba_graph, filename='Barabasi-Albert_graph.png'):
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(ba_graph)
    nx.draw_networkx_nodes(ba_graph, pos, node_size=500, node_color='skyblue', edgecolors='black')
    nx.draw_networkx_edges(ba_graph, pos, alpha=0.5, edge_color='gray')
    nx.draw_networkx_labels(ba_graph, pos, font_size=10, font_color='black', font_weight='bold')
    plt.title("Barabasi-Albert Model")
    plt.axis('off')
    plt.savefig('barabasi_albert_graph.png')
    plt.close()

draw_watts_strogatz(watts_strogatz_graph)
draw_barabasi_albert(barabasi_albert_graph)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
s

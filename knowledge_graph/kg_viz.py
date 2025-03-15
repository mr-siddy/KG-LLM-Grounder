import sys
import networkx as nx
import matplotlib.pyplot as plt

def visualize_gml(gml_path):
    try:
        # Load the graph from GML file
        G = nx.read_gml(gml_path)
        
        # Check if the graph is empty
        if G.number_of_nodes() == 0:
            print("The graph is empty. Please check your GML file.")
            return

        # # Draw the full graph
        # plt.figure(figsize=(12, 8))
        # pos = nx.spring_layout(G, seed=42)  # Use a fixed seed for consistent layout
        # nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10)
        # plt.title("Knowledge Graph Visualization")
        # plt.show()
        # plt.close()

        # Extract connected components and plot the second and third largest components
        components = list(nx.connected_components(G.to_undirected()))
        if len(components) < 3:
            print("The graph has fewer than three connected components.")
            return

        largest_components = sorted(components, key=len, reverse=True)[1:3]

        for i, component in enumerate(largest_components, start=2):
            subgraph = G.subgraph(component)
            plt.figure(figsize=(8, 6))
            pos = nx.spring_layout(subgraph, seed=42)
            nx.draw(subgraph, pos, with_labels=True, node_color='lightgreen', edge_color='black', node_size=10, font_size=8)
            plt.title(f"Subgraph Visualization - Component {i}")
            plt.show()
    except Exception as e:
        print(f"Error loading or visualizing the graph: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python kg_visualizer.py <input_gml_path>")
        sys.exit(1)

    input_gml_path = sys.argv[1]
    visualize_gml(input_gml_path)

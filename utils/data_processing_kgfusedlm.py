import json
import os
import logging
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_data_for_training(input_file, output_dir):
    """
    Process the dataset for training with KGFusedLM:
    1. Ensures all required fields are present
    2. Creates an enriched JSON format compatible with training pipeline
    3. Generates a simple KG visualization
    
    Args:
        input_file (str): Path to the input JSON file
        output_dir (str): Directory to save processed files
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if data is a list or a single dictionary
    if not isinstance(data, list):
        data = [data]
    
    # Process each entry
    processed_data = []
    for entry in data:
        processed_entry = {
            "chunk_text": entry.get("chunk_text", ""),
            "qa_pairs": entry.get("qa_pairs", []),
            "ner_entities": entry.get("ner_entities", []),
            "relation_triplets": entry.get("relation_triplets", [])
        }
        processed_data.append(processed_entry)
    
    # Save processed data
    output_file = os.path.join(output_dir, "processed_data.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2)
    
    logger.info(f"Processed {len(processed_data)} records")
    logger.info(f"Processed data saved to: {output_file}")
    
    # Create a simple KG visualization
    try:
        # Build a graph from the first few entries (to keep visualization manageable)
        G = nx.DiGraph()
        
        # Limit to first 10 entries or all if less than 10
        for entry in data[:min(10, len(data))]:
            # Add entities as nodes
            for entity in entry.get("ner_entities", []):
                entity_name = entity.get("entity")
                entity_label = entity.get("label")
                if entity_name:
                    G.add_node(entity_name, type=entity_label)
            
            # Add relations as edges
            for triplet in entry.get("relation_triplets", []):
                source = triplet.get("source")
                relation = triplet.get("relation")
                target = triplet.get("target")
                if source and target:
                    G.add_edge(source, target, relation=relation)
        
        # Save graph as GML file
        gml_file = os.path.join(output_dir, "knowledge_graph.gml")
        nx.write_gml(G, gml_file)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color="lightblue")
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10)
        
        # Save the visualization
        viz_file = os.path.join(output_dir, "kg_visualization.png")
        plt.title("Knowledge Graph Visualization")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(viz_file, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Knowledge graph visualization saved to: {viz_file}")
        logger.info(f"Knowledge graph GML file saved to: {gml_file}")
    except Exception as e:
        logger.warning(f"Could not create visualization: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process data for KGFusedLM training")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output", type=str, required=True, help="Directory to save processed files")
    
    args = parser.parse_args()
    
    process_data_for_training(args.input, args.output)
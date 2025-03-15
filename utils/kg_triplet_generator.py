import json
import os
import logging
from pathlib import Path
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_kg_triples(input_file, output_file):
    """
    Extract knowledge graph triples from the dataset.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to save the KG triples in TSV format
    
    Returns:
        str: Path to the output TSV file
    """
    logger.info("Extracting knowledge graph triples...")
    
    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if data is a list or a single dictionary
    if not isinstance(data, list):
        data = [data]
    
    # Extract triples from all entries
    triples = []
    entity_labels = {}  # To store entity types
    
    for entry in tqdm(data, desc="Processing entries"):
        # Extract NER entities and their labels
        for entity in entry.get("ner_entities", []):
            entity_name = entity.get("entity")
            entity_label = entity.get("label")
            if entity_name and entity_label:
                entity_labels[entity_name] = entity_label
        
        # Extract relation triples
        for triplet in entry.get("relation_triplets", []):
            source = triplet.get("source")
            relation = triplet.get("relation")
            target = triplet.get("target")
            
            if source and relation and target:
                triples.append((source, relation, target))
    
    # Add entity type triples
    for entity, label in entity_labels.items():
        triples.append((entity, "has_type", label))
    
    # Remove duplicates while preserving order
    unique_triples = []
    seen = set()
    for triple in triples:
        triple_str = f"{triple[0]}\t{triple[1]}\t{triple[2]}"
        if triple_str not in seen:
            unique_triples.append(triple_str)
            seen.add(triple_str)
    
    # Save the triples to TSV file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("source\trelation\ttarget\n")  # Header
        for triple_str in unique_triples:
            f.write(f"{triple_str}\n")
    
    logger.info(f"Extracted {len(unique_triples)} unique triples from {len(data)} records")
    
    return output_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract KG triples from dataset")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output", type=str, required=True, help="Path to output TSV file")
    
    args = parser.parse_args()
    
    extract_kg_triples(args.input, args.output)
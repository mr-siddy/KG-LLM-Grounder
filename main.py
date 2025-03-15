#!/usr/bin/env python3
"""
Complete end-to-end pipeline for training a Knowledge Graph Fused Language Model.

This script:
1. Prepares the dataset for training
2. Builds the knowledge graph
3. Generates KG embeddings
4. Configures and trains the KGFusedLM model
5. Validates the model's performance

Author: Sidgraph
"""

import os
import argparse
import json
import yaml
import shutil
from pathlib import Path
import torch
import random
import numpy as np
from tqdm import tqdm

from kg_dataset import KnowledgeGraphDataset
from kg_module import KnowledgeGraphEmbeddings
from model import KGFusedLM
from text_dataset_class import QADatasetWithKG
from utils import set_seed, print_num_parameters, Timer

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Knowledge Graph Fused Language Model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input data file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save all outputs")
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--model_name", type=str, default=None, help="Hugging Face model name (overrides config)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of data for training")
    parser.add_argument("--kg_only", action="store_true", help="Only build the knowledge graph, don't train")
    parser.add_argument("--small_test", action="store_true", help="Run with a small subset for testing")
    return parser.parse_args()

# Function to prepare directories
def prepare_directories(output_dir):
    """Create necessary directories for outputs."""
    dirs = {
        "data": os.path.join(output_dir, "data"),
        "kg": os.path.join(output_dir, "kg"),
        "model": os.path.join(output_dir, "model"),
        "checkpoints": os.path.join(output_dir, "checkpoints"),
        "evaluation": os.path.join(output_dir, "evaluation"),
    }
    
    for dir_path in dirs.values():
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    return dirs

# Function to load config
def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Function to update config
def update_config(config, args, dirs):
    """Update config with command line arguments and directories."""
    if args.model_name:
        config["model"]["name_or_path"] = args.model_name
    
    # Update data paths
    config["data"]["train_file"] = os.path.join(dirs["data"], "train.json")
    config["data"]["val_file"] = os.path.join(dirs["data"], "val.json")
    config["data"]["kg_file"] = os.path.join(dirs["kg"], "kg_triples.tsv")
    
    # Update output paths
    config["output"]["save_dir"] = dirs["checkpoints"]
    
    return config

# Function to split data
def split_data(input_file, output_dir, train_ratio=0.8, seed=42, small_test=False):
    """Split the data into training and validation sets."""
    print("Splitting data into training and validation sets...")
    
    # Set random seed
    random.seed(seed)
    
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if data is a list or a single dictionary
    if not isinstance(data, list):
        data = [data]
    
    # For small test, limit to a few samples
    if small_test:
        print("Running with small test subset...")
        data = data[:min(10, len(data))]
    
    # Shuffle the data
    random.shuffle(data)
    
    # Calculate the split point
    split_idx = int(len(data) * train_ratio)
    
    # Split the data
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # Save the training set
    train_file = os.path.join(output_dir, "train.json")
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2)
    
    # Save the validation set
    val_file = os.path.join(output_dir, "val.json")
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"  - Total records: {len(data)}")
    print(f"  - Training records: {len(train_data)} ({train_ratio*100:.0f}%)")
    print(f"  - Validation records: {len(val_data)} ({(1-train_ratio)*100:.0f}%)")
    
    return train_file, val_file

# Function to extract KG triples
def extract_kg_triples(input_file, output_file):
    """Extract knowledge graph triples from the dataset."""
    print("Extracting knowledge graph triples...")
    
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
    
    print(f"Extracted {len(unique_triples)} unique triples from {len(data)} records")
    
    return output_file

# Function to build and process the knowledge graph
def build_knowledge_graph(triples_file, kg_dir, config):
    """Build and process the knowledge graph."""
    print("Building knowledge graph and generating embeddings...")
    
    # Initialize the KnowledgeGraphDataset
    kg_dataset = KnowledgeGraphDataset(
        embedding_model_name=config["kg"].get("model_name", "all-MiniLM-L6-v2"),
        semantic_threshold=config["kg"].get("semantic_threshold", 0.8)
    )
    
    # Load triples from TSV file and build graph
    with open(triples_file, 'r', encoding='utf-8') as f:
        # Skip header
        next(f)
        for line in tqdm(f, desc="Loading triples"):
            try:
                source, relation, target = line.strip().split('\t')
                kg_dataset.add_relation(source, target, relation)
            except Exception as e:
                print(f"Error processing line: {line.strip()}, Error: {e}")
    
    print("Constructing graph structure...")
    kg_dataset.construct_graph()
    
    print("Computing node embeddings...")
    kg_dataset.compute_node_embeddings()
    
    print("Adding semantic edges...")
    kg_dataset.add_semantic_edges()
    
    # Save as GML for visualization
    gml_file = os.path.join(kg_dir, "knowledge_graph.gml")
    kg_dataset.save_as_gml(gml_file)
    
    # Generate and save PyG data
    graph_data = kg_dataset.get_graph_data()
    
    # Initialize KnowledgeGraphEmbeddings
    kg_embeddings = KnowledgeGraphEmbeddings(
        graph=kg_dataset.to_networkx(),
        embedding_dim=config["kg"].get("embedding_dim", 128),
        text_model=config["kg"].get("model_name", "all-MiniLM-L6-v2")
    )
    
    # Generate embeddings
    print("Generating structural embeddings...")
    structural_emb = kg_embeddings.generate_structural_embeddings()
    
    print("Generating textual embeddings...")
    textual_emb = kg_embeddings.generate_textual_embeddings()
    
    print("Combining embeddings...")
    combined_emb = kg_embeddings.combine_embeddings(strategy="concat")
    
    # Save embeddings
    embeddings_file = os.path.join(kg_dir, "combined_embeddings.txt")
    kg_embeddings.save_embeddings(embeddings_file, embedding_type="combined")
    
    return kg_dataset, kg_embeddings

# Function to train the model
def train_kgfused_model(config, train_file, val_file, kg_embeddings):
    """Train the KGFusedLM model."""
    print("Training KGFusedLM model...")
    
    try:
        # Import necessary modules
        from transformers import AutoTokenizer
        from torch.utils.data import DataLoader
        
        # Check for GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config["model"]["name_or_path"],
            trust_remote_code=config["model"].get("trust_remote_code", True)
        )
        
        # Initialize model
        model = KGFusedLM(
            llama_model_name=config["model"]["name_or_path"],
            trust_remote_code=config["model"].get("trust_remote_code", True),
            kg_embedding_dim=config["kg"].get("embedding_dim", 128)
        )
        model.to(device)
        
        # Print model parameters
        print_num_parameters(model)
        
        # Prepare datasets
        print("Loading training data...")
        train_dataset = QADatasetWithKG(
            train_file, 
            tokenizer, 
            max_length=config["training"].get("max_seq_length", 512)
        )
        
        print("Loading validation data...")
        val_dataset = QADatasetWithKG(
            val_file, 
            tokenizer, 
            max_length=config["training"].get("max_seq_length", 512)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config["training"].get("batch_size", 4), 
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config["training"].get("batch_size", 4), 
            shuffle=False
        )
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config["training"].get("learning_rate", 1e-5)
        )
        
        # Training loop
        num_epochs = config["training"].get("epochs", 3)
        save_dir = config["output"]["save_dir"]
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        
        print(f"Starting training for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # Training phase
            model.train()
            train_loss = 0
            train_steps = 0
            
            with tqdm(train_loader, desc=f"Training Epoch {epoch+1}") as t:
                for batch in t:
                    # Move batch to device
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    
                    # Get KG embeddings for each example
                    kg_emb_list = []
                    for ct in batch["chunk_text"]:
                        # This is where you'd use your kg_embeddings to get the embedding
                        # For now, let's use dummy embeddings of the right size
                        kg_emb = torch.zeros(config["kg"].get("embedding_dim", 128), device=device)
                        kg_emb_list.append(kg_emb)
                    
                    kg_emb = torch.stack(kg_emb_list)
                    
                    # Forward pass
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        kg_embedding=kg_emb
                    )
                    
                    loss = outputs["loss"]
                    train_loss += loss.item()
                    train_steps += 1
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Update progress bar
                    t.set_postfix(loss=loss.item())
            
            avg_train_loss = train_loss / train_steps if train_steps > 0 else 0
            print(f"Average training loss: {avg_train_loss:.4f}")
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_steps = 0
            
            with torch.no_grad():
                with tqdm(val_loader, desc=f"Validation Epoch {epoch+1}") as t:
                    for batch in t:
                        # Move batch to device
                        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                        
                        # Get KG embeddings
                        kg_emb_list = []
                        for ct in batch["chunk_text"]:
                            kg_emb = torch.zeros(config["kg"].get("embedding_dim", 128), device=device)
                            kg_emb_list.append(kg_emb)
                        
                        kg_emb = torch.stack(kg_emb_list)
                        
                        # Forward pass
                        outputs = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                            kg_embedding=kg_emb
                        )
                        
                        loss = outputs["loss"]
                        val_loss += loss.item()
                        val_steps += 1
                        
                        # Update progress bar
                        t.set_postfix(loss=loss.item())
            
            avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
            print(f"Average validation loss: {avg_val_loss:.4f}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(save_dir, f"epoch_{epoch+1}")
            os.makedirs(checkpoint_path, exist_ok=True)
            
            # Save model
            model.llama.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            
            # Save optimizer state
            torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))
            
            # Save as best model if it has the lowest validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = os.path.join(save_dir, "best_model")
                
                # If best_model_path exists, remove it first
                if os.path.exists(best_model_path):
                    shutil.rmtree(best_model_path)
                
                # Copy the current epoch to best_model
                shutil.copytree(checkpoint_path, best_model_path)
                print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        
        print("Training complete!")
        return model, tokenizer
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Prepare directories
    dirs = prepare_directories(args.output_dir)
    
    # Load configuration
    config = load_config(args.config_path)
    
    # Update configuration
    config = update_config(config, args, dirs)
    
    # Save updated configuration
    config_file = os.path.join(args.output_dir, "config.yaml")
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Split data
    train_file, val_file = split_data(
        args.data_path, 
        dirs["data"], 
        args.train_ratio, 
        args.seed,
        args.small_test
    )
    
    # Extract KG triples
    kg_triples_file = extract_kg_triples(args.data_path, os.path.join(dirs["kg"], "kg_triples.tsv"))
    
    # Build knowledge graph
    kg_dataset, kg_embeddings = build_knowledge_graph(kg_triples_file, dirs["kg"], config)
    
    # If kg_only flag is set, exit here
    if args.kg_only:
        print("Knowledge graph processing completed. Exiting as --kg_only flag was set.")
        return
    
    # Train the model
    with Timer("Model training"):
        model, tokenizer = train_kgfused_model(config, train_file, val_file, kg_embeddings)
    
    print(f"Pipeline completed successfully. Model saved to {dirs['checkpoints']}")

if __name__ == "__main__":
    main()
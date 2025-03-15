import torch
import random
import numpy as np
import yaml
import time
from math import sqrt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_num_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total}, Trainable: {trainable}")

def load_config(config_file):
    """
    Load a YAML configuration file.
    """
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config

class Timer:
    """
    Simple context manager for timing code blocks.
    Usage:
        with Timer("Training step"):
            # your code
    """
    def __init__(self, name="Block"):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        print(f"{self.name} took {elapsed:.4f} seconds.")

def cosine_similarity(vec1, vec2):
    """
    Compute the cosine similarity between two 1D numpy arrays.
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

# Example usage:
if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)
    
    # Print a dummy model's parameters (replace with your model)
    dummy_model = torch.nn.Linear(10, 2)
    print_num_parameters(dummy_model)
    
    # Load a configuration file
    config = load_config("configs/config.yaml")
    print("Configuration loaded:", config)
    
    # Time a simple operation
    with Timer("Sleep for demonstration"):
        time.sleep(1)
    
    # Test cosine similarity function
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([4, 5, 6])
    sim = cosine_similarity(vec1, vec2)
    print("Cosine similarity:", sim)

import json
import random
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_data(input_file, output_dir, train_ratio=0.8, seed=42, small_test=False):
    """
    Split the dataset into training and validation sets.
    
    Args:
        input_file (str): Path to the input JSON file
        output_dir (str): Directory to save the output files
        train_ratio (float): Ratio of examples to use for training (default: 0.8)
        seed (int): Random seed for reproducibility
        small_test (bool): If True, use a small subset of data for testing
    
    Returns:
        tuple: Paths to the training and validation files
    """

    random.seed(seed)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]
    
    if small_test:
        logger.info("Running with small test subset...")
        data = data[:min(10, len(data))]
    
    random.shuffle(data)
    
    split_idx = int(len(data) * train_ratio)
    
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    train_file = os.path.join(output_dir, "train.json")
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2)
    
    val_file = os.path.join(output_dir, "val.json")
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2)
    
    logger.info(f"Total records: {len(data)}")
    logger.info(f"Training records: {len(train_data)} ({train_ratio*100:.0f}%)")
    logger.info(f"Validation records: {len(val_data)} ({(1-train_ratio)*100:.0f}%)")
    
    return train_file, val_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Split dataset into training and validation sets")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output", type=str, required=True, help="Directory to save output files")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of data for training (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--small_test", action="store_true", help="Use a small subset for testing")
    
    args = parser.parse_args()
    
    split_data(args.input, args.output, args.train_ratio, args.seed, args.small_test)
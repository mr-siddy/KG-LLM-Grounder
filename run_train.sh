#!/bin/bash

# Example usage: bash run_train.sh
#!/bin/bash

# Default values
DATA_PATH="scripts/prompt_tuning.json"
OUTPUT_DIR="output"
CONFIG_PATH="configs/config.yaml"
MODEL_NAME=""
SEED=42
TRAIN_RATIO=0.8
KG_ONLY=false
SMALL_TEST=false
INFERENCE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_path)
      DATA_PATH="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --config_path)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --model_name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --train_ratio)
      TRAIN_RATIO="$2"
      shift 2
      ;;
    --kg_only)
      KG_ONLY=true
      shift
      ;;
    --small_test)
      SMALL_TEST=true
      shift
      ;;
    --inference)
      INFERENCE=true
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Create command
CMD="python main.py --data_path $DATA_PATH --output_dir $OUTPUT_DIR --config_path $CONFIG_PATH --seed $SEED --train_ratio $TRAIN_RATIO"

# Add optional arguments
if [ ! -z "$MODEL_NAME" ]; then
  CMD="$CMD --model_name $MODEL_NAME"
fi

if [ "$KG_ONLY" = true ]; then
  CMD="$CMD --kg_only"
fi

if [ "$SMALL_TEST" = true ]; then
  CMD="$CMD --small_test"
fi

if [ "$INFERENCE" = true ]; then
  CMD="$CMD --inference"
fi

# Print and execute the command
echo "Running: $CMD"
eval $CMD
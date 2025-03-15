import argparse
from datasets import load_dataset
from transformers import AutoTokenizer

def generate_prompt(example):
    """
    For each record, generate one (or more) prompt–completion pairs.
    Each record contains a "chunk_text" and a list of "qa_pairs" (each with a "question" and "answer").
    
    The prompt is built as:
        Q: {question}
        Context: {chunk_text}
        A:
    And the target (completion) is: " {answer}"
    We return a new field "text" which is a list of such full texts.
    """
    prompts = []
    chunk = example.get("chunk_text", "").strip()
    for qa in example.get("qa_pairs", []):
        question = qa.get("question", "").strip()
        answer = qa.get("answer", "").strip()
        # You can adjust the format below as needed.
        full_text = f"Q: {question}\nContext: {chunk}\nA: {answer}"
        prompts.append(full_text)
    return {"text": prompts}

def tokenize_function(example, tokenizer, max_length):
    # The "text" field here is a string (after exploding)
    tokenized = tokenizer(example["text"], truncation=True, max_length=max_length, padding="max_length")
    # For causal LM fine-tuning, the labels are usually the same as input_ids.
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def main(json_file, model_name, max_length):
    # Load the raw dataset from the JSON file.
    # (The JSON file is assumed to be a list of records.)
    raw_dataset = load_dataset("json", data_files=json_file)["train"]

    # For each record, create one or more prompt–completion examples.
    # This adds a new column "text" that is a list of strings.
    dataset_with_prompts = raw_dataset.map(generate_prompt, batched=False)

    # Explode the "text" field so that each example is one prompt–completion pair.
    dataset_exploded = dataset_with_prompts.explode("text")

    # Load the tokenizer for your chosen model (e.g., LLaMA‑3.3‑8B or DeepSeek‑v3‑r1)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize the text field. We use a lambda to pass the tokenizer and max_length.
    tokenized_dataset = dataset_exploded.map(lambda x: tokenize_function(x, tokenizer, max_length), batched=True)

    # Set format to PyTorch tensors so that they can be fed to the Trainer.
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Optionally, print out a sample.
    print("Sample tokenized example:")
    print(tokenized_dataset[0])
    
    # Save the tokenized dataset to disk for later use.
    tokenized_dataset.save_to_disk("qa_tokenized_dataset")
    print("Tokenized dataset saved to disk at 'qa_tokenized_dataset'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a tokenized dataset for LLM fine-tuning (QA task) from a JSON file.")
    parser.add_argument("--json_file", type=str, required=True, help="Path to the input JSON file (e.g., prompt_tuning.json).")
    parser.add_argument("--model_name", type=str, default="your-organization/llama-3.3-8b", help="Hugging Face model name or path.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum token length for each example.")
    args = parser.parse_args()
    main(args.json_file, args.model_name, args.max_length)

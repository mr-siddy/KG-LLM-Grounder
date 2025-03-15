import os
import json
import time
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_qa_pairs(prompt, model, tokenizer, max_new_tokens=300, temperature=0.7):
    """
    Generates question-answer pairs from the given prompt using a Hugging Face model.
    
    Args:
        prompt (str): The prompt containing instructions and the text.
        model: The Hugging Face causal LM.
        tokenizer: The associated tokenizer.
        max_new_tokens (int): Maximum tokens to generate.
        temperature (float): Sampling temperature.
    
    Returns:
        A list of QA pair dictionaries, e.g. [{"question": "...", "answer": "..."}, ...]
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    try:
        qa_pairs = json.loads(generated_text)
        # In case a single object is returned, wrap it in a list.
        if isinstance(qa_pairs, dict):
            qa_pairs = [qa_pairs]
    except Exception as e:
        print("Error parsing generated output:", e)
        print("Generated text:", generated_text)
        qa_pairs = []
    return qa_pairs

def main(input_file, output_file, model_name, sleep_time=1.0):
    # Load the Hugging Face model and tokenizer.
    print(f"Loading model '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}.")

    # Load your enriched data.
    with open(input_file, "r", encoding="utf-8") as f:
        records = json.load(f)
    
    qa_dataset = []
    for idx, record in enumerate(records):
        print(f"Processing record {idx+1}/{len(records)}...")
        chunk_text = record.get("chunk_text", "")
        ner_entities = record.get("ner_entities", [])
        relation_triplets = record.get("relation_triplets", [])

        # Construct a prompt for QA generation.
        prompt = (
            "You are an expert in creating question answering datasets for fine-tuning language models. "
            "Given the following text chunk, generate a list of clear, concise question and answer pairs that "
            "test comprehension of the text. Ensure that the answers are directly supported by the text. "
            "If the text does not contain sufficient details, generate one QA pair that captures the main idea.\n\n"
            "Text:\n" + chunk_text + "\n\n"
            "Named Entities: " + ", ".join([e['entity'] for e in ner_entities]) + "\n\n"
            "Relation Triplets: " + "; ".join([f"{t['source']} {t['relation']} {t['target']}" for t in relation_triplets]) + "\n\n"
            "Output the result strictly as a JSON list of objects, where each object has keys 'question' and 'answer'."
        )
        
        qa_pairs = generate_qa_pairs(prompt, model, tokenizer)
        qa_record = {
            "chunk_text": chunk_text,
            "qa_pairs": qa_pairs,
            "ner_entities": ner_entities,
            "relation_triplets": relation_triplets
        }
        qa_dataset.append(qa_record)
        time.sleep(sleep_time)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(qa_dataset, f, indent=2)
    print(f"QA dataset generated and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a question answering dataset from enriched data using a Hugging Face model."
    )
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file (enriched data).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file (QA dataset).")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name or path (e.g., llama-3.3-70B or deepseek-v3-r1).")
    parser.add_argument("--sleep_time", type=float, default=1.0, help="Sleep time (in seconds) between processing records.")
    args = parser.parse_args()
    main(args.input_file, args.output_file, args.model_name, args.sleep_time)

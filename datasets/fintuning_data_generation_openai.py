import os
import json
import time
import openai
import argparse

# Set up your Azure OpenAI Service credentials:
# Ensure you have set these environment variables:
# AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT (e.g., "gpt-35-turbo-16k")
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")  # e.g., "https://YOUR_RESOURCE_NAME.openai.azure.com/"
openai.api_type = "azure"
# Use an API version that supports chat completions with your chosen model
openai.api_version = "2023-05-15"
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT")  # e.g., "gpt-35-turbo-16k"

def generate_qa_pairs(chunk_text, ner_entities, relation_triplets, max_new_tokens=300, temperature=0.7):
    """
    Generate question-answer pairs from the given text using an Azure OpenAI model.
    
    Args:
        chunk_text (str): The source text to base QA generation on.
        ner_entities (list): List of named entity objects (can be used to provide additional context).
        relation_triplets (list): List of relation triplets (can be included in the prompt if desired).
        max_new_tokens (int): Maximum tokens for generated output.
        temperature (float): Sampling temperature.
    
    Returns:
        A list of QA pair dictionaries, e.g. [{"question": "…", "answer": "…"}, ...]
    """
    prompt = (
        "You are an expert in creating question answering datasets for fine-tuning large language models. "
        "Given the following text chunk, generate a list of clear, concise question and answer pairs that test comprehension of the text. "
        "Ensure that the answers are directly supported by the text. "
        "If the text does not contain enough information, generate only one QA pair that captures the main idea. \n\n"
        "Text:\n"
        f"{chunk_text}\n\n"
        "Named Entities (if any): " + ", ".join([e['entity'] for e in ner_entities]) + "\n\n"
        "Relation Triplets (if any): " + "; ".join([f"{t['source']} {t['relation']} {t['target']}" for t in relation_triplets]) + "\n\n"
        "Generate the output as a JSON list of objects, where each object has 'question' and 'answer' keys. "
        "Output only the JSON."
    )

    try:
        response = openai.ChatCompletion.create(
            engine=DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=temperature,
            n=1,
            stop=None,
        )
        output = response.choices[0].message.content.strip()
        qa_pairs = json.loads(output)
        # If the result is a dict, wrap it in a list.
        if isinstance(qa_pairs, dict):
            qa_pairs = [qa_pairs]
        return qa_pairs
    except Exception as e:
        print("Error during QA generation:", e)
        return []

def main(input_file, output_file, sleep_time=1):
    # Load enriched data (list of records)
    with open(input_file, "r", encoding="utf-8") as f:
        records = json.load(f)
    
    qa_dataset = []
    for idx, record in enumerate(records):
        print(f"Processing record {idx+1}/{len(records)}...")
        chunk_text = record.get("chunk_text", "")
        ner_entities = record.get("ner_entities", [])
        relation_triplets = record.get("relation_triplets", [])
        
        qa_pairs = generate_qa_pairs(chunk_text, ner_entities, relation_triplets)
        
        qa_record = {
            "chunk_text": chunk_text,
            "qa_pairs": qa_pairs,
            "ner_entities": ner_entities,
            "relation_triplets": relation_triplets
        }
        qa_dataset.append(qa_record)
        time.sleep(sleep_time)  # to avoid rate limits
    
    # Save the QA dataset to output_file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(qa_dataset, f, indent=2)
    print(f"QA dataset generated and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a question answering dataset from enriched input JSON using Azure OpenAI Service."
    )
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSON file containing enriched data.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output JSON file to save the generated QA dataset.")
    parser.add_argument("--sleep_time", type=float, default=1.0, help="Sleep time between API calls (in seconds).")
    args = parser.parse_args()
    main(args.input_file, args.output_file, args.sleep_time)

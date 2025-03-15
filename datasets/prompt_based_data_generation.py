import json
import argparse

def generate_qa_from_record(record):
    qa_pairs = []
    chunk_text = record.get("chunk_text", "").strip()
    ner_entities = record.get("ner_entities", [])
    relation_triplets = record.get("relation_triplets", [])
    
    if not chunk_text:
        return qa_pairs

    # 1. Generate a QA pair based on the top named entity (if available)
    if ner_entities:
        # Sort entities by score descending (if score is provided)
        sorted_entities = sorted(ner_entities, key=lambda x: x.get("score", 0), reverse=True)
        key_entity = sorted_entities[0].get("entity", "")
        if key_entity:
            question = f"What does the text say about {key_entity}?"
            qa_pairs.append({"question": question, "answer": chunk_text})
    
    # 2. Generate a QA pair based on the first relation triplet (if available)
    if relation_triplets:
        first_relation = relation_triplets[0]
        source = first_relation.get("source", "")
        relation = first_relation.get("relation", "")
        target = first_relation.get("target", "")
        if source and relation and target:
            question = f"What does the text mention about how {source} {relation} {target}?"
            qa_pairs.append({"question": question, "answer": chunk_text})
    
    # 3. If no ner_entities and no relation_triplets, generate a generic QA pair
    if not ner_entities and not relation_triplets:
        question = "What is the main information provided in the text?"
        qa_pairs.append({"question": question, "answer": chunk_text})
    
    return qa_pairs

def main(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    qa_dataset = []
    for record in data:
        qa_pairs = generate_qa_from_record(record)
        qa_dataset.append({
            "chunk_text": record.get("chunk_text", ""),
            "qa_pairs": qa_pairs,
            "ner_entities": record.get("ner_entities", []),
            "relation_triplets": record.get("relation_triplets", [])
        })
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(qa_dataset, f, indent=2)
    print(f"Generated QA dataset saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a QA dataset using a static prompt template from enriched JSON data."
    )
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input enriched JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output QA JSON file.")
    args = parser.parse_args()
    main(args.input_file, args.output_file)

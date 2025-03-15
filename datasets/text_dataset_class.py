import json
from torch.utils.data import Dataset

class QADatasetWithKG(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        """
        Dataset for fine-tuning the LLM on QA pairs.
        In addition to prompt–completion pairs, each example returns the original chunk_text.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        
        self.examples = []
        for record in records:
            chunk_text = record.get("chunk_text", "").strip()
            qa_pairs = record.get("qa_pairs", [])
            for qa in qa_pairs:
                question = qa.get("question", "").strip()
                answer = qa.get("answer", "").strip()
                # Construct the prompt with the question and context.
                prompt = f"Q: {question}\nContext: {chunk_text}\nA:"
                completion = f" {answer}"
                self.examples.append({
                    "prompt": prompt,
                    "completion": completion,
                    "chunk_text": chunk_text  # Include for KG lookup
                })
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        full_text = example["prompt"] + example["completion"]
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": tokenized["input_ids"].squeeze(0),
            "chunk_text": example["chunk_text"]  # Return for KG lookup in training
        }

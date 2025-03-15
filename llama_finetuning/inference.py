import torch
from transformers import AutoTokenizer
from model.model import KGFusedLM

def generate_response(prompt, 
                      chunk_text,  # New argument to provide text for KG lookup
                      model_path="checkpoints/epoch_3", 
                      kg_embedder=None, 
                      max_new_tokens=50, 
                      device="cuda"):
    """
    Generate a response from the trained KGFusedLM model.
    Use the chunk_text to retrieve the KG embedding for grounding.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = KGFusedLM(llama_model_name=model_path)
    model.eval()
    model.to(device)

    # Retrieve KG embedding using chunk_text if kg_embedder is provided.
    if kg_embedder is not None and chunk_text:
        kg_emb = kg_embedder.get_chunk_embedding(chunk_text).unsqueeze(0).to(device)
    else:
        kg_emb = torch.zeros(1, model.kg_transform.in_features).to(device)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        generation = model.llama.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7
        )
    
    output_text = tokenizer.decode(generation[0], skip_special_tokens=True)
    return output_text

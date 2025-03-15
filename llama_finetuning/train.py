import os
import yaml
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from accelerate import Accelerator

from .dataset import TextDataset
from model.kg_module import KnowledgeGraphEmbeddings
from model.model import KGFusedLM

def train_model(config_file="configs/config.yaml"):
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    kg_cfg = cfg["kg"]
    output_cfg = cfg["output"]

    accelerator = Accelerator()
    
    # Build / train KG embeddings (this populates chroma DB, etc.)
    kg_embedder = KnowledgeGraphEmbeddings(
        kg_file=data_cfg["kg_file"], 
        embedding_dim=kg_cfg["embedding_dim"],
        model_name=kg_cfg["model_name"],
        num_epochs=kg_cfg["num_epochs"]
    )
    kg_embedder.train()  # Assume this writes embeddings into your Chroma DB

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name_or_path"], trust_remote_code=model_cfg["trust_remote_code"])
    model = KGFusedLM(llama_model_name=model_cfg["name_or_path"],
                      trust_remote_code=model_cfg["trust_remote_code"],
                      kg_embedding_dim=kg_cfg["embedding_dim"])

    # Use the updated dataset that also returns chunk_text
    train_dataset = QADatasetWithKG(data_cfg["train_file"], tokenizer, max_length=train_cfg["max_seq_length"])
    val_dataset   = QADatasetWithKG(data_cfg["val_file"], tokenizer, max_length=train_cfg["max_seq_length"])

    train_loader = DataLoader(train_dataset, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=train_cfg["batch_size"], shuffle=False)

    # ---------------------------
    # Optimizer
    # ---------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"])

    # Prepare everything with accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    # ---------------------------
    # Training Loop
    # ---------------------------
    global_step = 0
    for epoch in range(train_cfg["epochs"]):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            # Instead of using zeros, retrieve KG embeddings for each example based on chunk_text.
            kg_emb_list = []
            for ct in batch["chunk_text"]:
                # For each text, get its KG embedding from Chroma DB via the KG embedder.
                # This assumes that kg_embedder.get_chunk_embedding returns a tensor of shape (kg_embedding_dim,)
                kg_emb = kg_embedder.get_chunk_embedding(ct)
                kg_emb_list.append(kg_emb)
            # Stack them into a batch tensor.
            kg_emb = torch.stack(kg_emb_list).to(batch["input_ids"].device)
            
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                kg_embedding=kg_emb
            )
            loss = outputs["loss"]
            accelerator.backward(loss)
            optimizer.step()
            global_step += 1

        # ---------------------------
        # Validation (update similarly)
        # ---------------------------
        model.eval()
        val_loss_sum = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                kg_emb_list = []
                for ct in batch["chunk_text"]:
                    kg_emb = kg_embedder.get_chunk_embedding(ct)
                    kg_emb_list.append(kg_emb)
                kg_emb = torch.stack(kg_emb_list).to(batch["input_ids"].device)

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    kg_embedding=kg_emb
                )
                val_loss_sum += outputs["loss"].item()
                val_steps += 1
        avg_val_loss = val_loss_sum / val_steps
        print(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss}")

        # Save checkpoint
        if accelerator.is_main_process:
            save_path = os.path.join(output_cfg["save_dir"], f"epoch_{epoch+1}")
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.llama.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

    print("Training complete!")

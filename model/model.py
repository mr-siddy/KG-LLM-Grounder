import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig

class KGFusedLM(nn.Module):
    """
    A model that fuses textual input with a KG embedding.
    The KG embedding is projected to the LLM’s hidden size and fused via a learnable gating mechanism.
    """
    def __init__(self, 
                 llama_model_name="your-organization/llama-3.3-8b",
                 trust_remote_code=True,
                 kg_embedding_dim=128):
        super().__init__()
        config = AutoConfig.from_pretrained(llama_model_name, trust_remote_code=trust_remote_code)
        self.llama = AutoModelForCausalLM.from_pretrained(llama_model_name, config=config)
        
        hidden_size = config.hidden_size
        self.kg_transform = nn.Linear(kg_embedding_dim, hidden_size)
        self.activation = nn.Tanh()
        # Gate: learn a scalar (per sample) that weighs the contribution of the KG embedding.
        self.gate = nn.Linear(kg_embedding_dim, 1)

    def forward(self, input_ids, attention_mask=None, labels=None, kg_embedding=None):
        """
        Args:
            input_ids: (batch, seq_len)
            kg_embedding: (batch, kg_embedding_dim)
        """
        if kg_embedding is not None:
            # Project KG embedding and compute a gating scalar (per sample)
            kg_proj = self.activation(self.kg_transform(kg_embedding))  # (batch, hidden_size)
            gate_value = torch.sigmoid(self.gate(kg_embedding))  # (batch, 1)
        else:
            kg_proj = torch.zeros(input_ids.size(0), self.llama.config.hidden_size, device=input_ids.device)
            gate_value = torch.zeros(input_ids.size(0), 1, device=input_ids.device)

        # Get LM hidden states.
        outputs = self.llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )
        last_hidden_state = outputs.hidden_states[-1]  # (batch, seq_len, hidden_size)
        
        # Expand kg_proj to add to every token: (batch, 1, hidden_size)
        kg_proj_expanded = kg_proj.unsqueeze(1)
        # Fuse by adding a weighted version of the KG embedding to each token's hidden state.
        fused_state = last_hidden_state + gate_value.unsqueeze(2) * kg_proj_expanded

        # Pass the fused hidden state through the LM head.
        logits = self.llama.lm_head(fused_state)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        return {"loss": loss, "logits": logits}

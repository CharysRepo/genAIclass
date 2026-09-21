import torch
import torch.nn as nn

class NanoGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings and input processing layers
        self.tok_embedding = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_embedding = nn.Embedding(config["context_length"], config["emb_dim"])
        self.drop_1 = nn.Dropout(p=config["drop_rate"])
        
        # Multi-head self-attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=config["emb_dim"], 
            num_heads=config["n_heads"], 
            batch_first=True
        )

        # Dropout layer for attention output
        self.drop_attn = nn.Dropout(p=config["drop_rate"])
        
        # feed-forward network (FFN) with GELU activation and dropout
        self.ffn = nn.Sequential(
            nn.Linear(config["emb_dim"], config["hidden_dim"]),
            nn.GELU(),
            nn.Linear(config["hidden_dim"], config["emb_dim"]),
            nn.Dropout(p=config["drop_rate"]) 
        )
        
        # Pre-layer normalization layers 
        self.ln_attn = nn.LayerNorm(config["emb_dim"])
        self.ln_ffn = nn.LayerNorm(config["emb_dim"])
        
        # Final output layers
        self.ln_f = nn.LayerNorm(config["emb_dim"])
        self.output_layer = nn.Linear(config["emb_dim"], config["vocab_size"])

    def forward(self, x):
        # Embedding and positional encoding
        tok_embeds = self.tok_embedding(x)
        positions = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        pos_embeds = self.pos_embedding(positions)
        embeds = tok_embeds + pos_embeds
        
        x = self.drop_1(embeds) 
        
        # Generate causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=x.device)
        
        # Attention Block (norm_first residual connection)
        norm_x = self.ln_attn(x)
        attn_output, _ = self.attention(
            query=norm_x, 
            key=norm_x, 
            value=norm_x, 
            attn_mask=mask
        )
        x = x + self.drop_attn(attn_output)
        
        # Feed-Forward Block 
        norm_x2 = self.ln_ffn(x)
        ffn_output = self.ffn(norm_x2)
        x = x + ffn_output 
            
        x = self.ln_f(x)    
        last_token_output = x[:, -1, :]
        return self.output_layer(last_token_output)

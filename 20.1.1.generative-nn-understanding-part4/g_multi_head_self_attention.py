import torch
import torch.nn as nn

seq_len = 5  
embed_dim = 10 
num_heads = 2 

emb = torch.randn(seq_len, embed_dim)
print(emb, emb.shape)

mha_layer = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
attn_output, attn_weights = mha_layer(emb, emb, emb, attn_mask=mask)
print(attn_weights, attn_weights.shape)
print(attn_output, attn_output.shape) 

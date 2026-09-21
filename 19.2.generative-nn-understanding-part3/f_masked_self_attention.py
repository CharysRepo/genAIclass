# The river bank overflowed.  
# The money is in the bank.
# The cat chased the mouse because it was hungry.
# The chef cooked the soup

import torch
import torch.nn as nn
import math

class SingleheadSelfAttention(nn.Module):
    def __init__(self, emb_dim, attn_dim):
        super().__init__()
        self.W_q = nn.Linear(emb_dim, attn_dim, bias=False)
        self.W_k = nn.Linear(emb_dim, attn_dim, bias=False)
        self.W_v = nn.Linear(emb_dim, attn_dim, bias=False)

    def forward(self, x, mask=False):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.shape[-1])
        if mask:
            upper_triangular  = torch.triu(attn_scores, diagonal=1).bool()
            attn_scores[upper_triangular] = float("-inf")
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        return attn_output
    
torch.manual_seed(123)
embed = nn.Embedding(num_embeddings=10, embedding_dim=5)
inp_ids = torch.tensor([3, 2, 8])
embedded_sentence = embed(inp_ids)
print(embedded_sentence)

attention = SingleheadSelfAttention(emb_dim=5, attn_dim=5)
output = attention(embedded_sentence)
print(output)



import torch.nn as nn
from model.attention import CausalSelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size):
        super().__init__()

        self.attn = CausalSelfAttention(embed_dim, num_heads, block_size)
        self.ln1 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Attention + residual
        x = x + self.attn(self.ln1(x))

        # Feedforward + residual
        x = x + self.ff(self.ln2(x))

        return x

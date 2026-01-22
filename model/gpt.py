import torch
import torch.nn as nn
from model.block import TransformerBlock

class GPT(nn.Module):
    def __init__(self, vocab_size, block_size,
                 embed_dim=128, num_layers=4, num_heads=4):
        super().__init__()

        self.block_size = block_size

        # Token + position embeddings
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, block_size)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        B, T = idx.shape

        # Positions [0, 1, 2, ...]
        pos = torch.arange(T, device=idx.device)

        # Embedding sum
        x = self.token_emb(idx) + self.pos_emb(pos)

        # Transformer
        x = self.blocks(x)

        # Final normalization + logits
        x = self.ln_f(x)
        return self.head(x)

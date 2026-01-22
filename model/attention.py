import torch
import torch.nn as nn
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.block_size = block_size

        # Linear projections for Q, K, V
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)

        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)

        # Causal mask (upper triangle blocked)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, embed_dim)
        """
        B, T, C = x.shape

        # Create Q, K, V
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask
        scores = scores.masked_fill(
            self.mask[:T, :T] == 0,
            float("-inf")
        )

        # Softmax
        attn = torch.softmax(scores, dim=-1)

        # Weighted sum
        out = attn @ v

        # Recombine heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(out)

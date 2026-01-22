import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # convert token IDs → vectors
        self.embedding = nn.Embedding(vocab_size, 32)

        # self-attention (the magic)
        self.attention = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=1,
            batch_first=True
        )

        # feedforward network
        self.ff = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # normalization
        self.ln = nn.LayerNorm(32)

        # output layer: vectors → vocabulary probabilities
        self.output = nn.Linear(32, vocab_size)

    def forward(self, x):
        """
        x shape: (batch, sequence_length)
        """

        # 1. tokens → embeddings
        x = self.embedding(x)

        # 2. self-attention
        attn_out, _ = self.attention(x, x, x)

        # 3. residual + normalization
        x = self.ln(x + attn_out)

        # 4. feedforward processing
        x = self.ff(x)

        # 5. predict next token
        logits = self.output(x)

        return logits

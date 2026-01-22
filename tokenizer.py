import torch

# read training text
text = open("data.txt", "r").read()

# find all unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

# character ↔ number mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# encode entire text as numbers
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

# expose variables for other files
__all__ = ["data", "stoi", "itos", "vocab_size"]

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from model.gpt import GPT

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

def decode(ids):
    return tokenizer.decode(ids)

def encode(text):
    return tokenizer.encode(text).ids

vocab_size = tokenizer.get_vocab_size()

block_size = 96

model = GPT(
    vocab_size=vocab_size,
    block_size=block_size,
    embed_dim=384,
    num_layers=6,
    num_heads=6
).to(device)

import sys
model_path = sys.argv[1] if len(sys.argv) > 1 else "gpt.pth"

checkpoint = torch.load(model_path, map_location=device)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    print(f"Loading checkpoint from {model_path} (step {checkpoint.get('step', '?')})")
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    print(f"Loading direct state dict from {model_path}")
    model.load_state_dict(checkpoint)

model.eval()

@torch.no_grad()
def generate(start="h", length=300):
    idx = torch.tensor([encode(start)], device=device)

    for _ in range(length):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_idx = torch.multinomial(probs, 1)
        idx = torch.cat([idx, next_idx], dim=1)

    return decode(idx[0].tolist())

def answer_question(question, resume_text):
    prompt = f"""
Resume:
{resume_text}

Question:
{question}

Answer:
"""
    print("--- PROMPT START ---")
    print(prompt)
    print("--- PROMPT END ---")
    tokens = tokenizer.encode(prompt).ids
    idx = torch.tensor([tokens], device=device)

    for _ in range(150):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_idx = torch.multinomial(probs, 1)
        idx = torch.cat([idx, next_idx], dim=1)

    # Decode only the generated part (excluding the prompt)
    # The prompt length in tokens is len(tokens)
    # The generated tokens are from len(tokens) onwards
    return decode(idx[0, len(tokens):].tolist())

if __name__ == "__main__":
    try:
        with open("data/resume.txt", "r") as f:
            resume_text = f.read()
        print(answer_question("Where is rishu from?", resume_text))
    except Exception as e:
        print(f"Error: {e}")

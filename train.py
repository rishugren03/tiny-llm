import torch
import torch.nn.functional as F
from model.gpt import GPT
from tokenizers import Tokenizer
import argparse
import os
import time
import logging
import sys

# ------------------
# CONFIG & ARGS
# ------------------
parser = argparse.ArgumentParser(description="Train GPT on custom corpus")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--block_size", type=int, default=256, help="Context length (block size)")
parser.add_argument("--embed_dim", type=int, default=384, help="Embedding dimension")
parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
parser.add_argument("--num_heads", type=int, default=6, help="Number of attention heads")
parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--max_steps", type=int, default=10000, help="Maximum training steps")
parser.add_argument("--eval_interval", type=int, default=500, help="Steps between validation")
parser.add_argument("--eval_iters", type=int, default=200, help="Iterations for loss estimation")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
parser.add_argument("--data_path", type=str, default="data/corpus.txt", help="Path to training data")
parser.add_argument("--tokenizer_path", type=str, default="tokenizer/tokenizer.json", help="Path to tokenizer file")
parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint path")

args = parser.parse_args()

# ------------------
# SETUP LOGGING
# ------------------
os.makedirs(args.checkpoint_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(args.checkpoint_dir, "train.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# ------------------
# LOAD RESOURCES
# ------------------
if not os.path.exists(args.data_path):
    logger.error(f"Data file not found: {args.data_path}")
    sys.exit(1)

if not os.path.exists(args.tokenizer_path):
    logger.error(f"Tokenizer file not found: {args.tokenizer_path}")
    sys.exit(1)

logger.info(f"Loading tokenizer from {args.tokenizer_path}...")
try:
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    logger.info(f"Vocab size: {vocab_size}")
except Exception as e:
    logger.error(f"Failed to load tokenizer: {e}")
    sys.exit(1)

logger.info(f"Loading data from {args.data_path}...")
try:
    with open(args.data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    if not text:
        logger.error("Data file is empty!")
        sys.exit(1)
    
    encoded = tokenizer.encode(text)
    data_tensor = torch.tensor(encoded.ids, dtype=torch.long)
    logger.info(f"Total tokens: {len(data_tensor)}")
    
    if len(data_tensor) < args.block_size + 1:
        logger.error(f"Data is too short (len={len(data_tensor)}) for block_size={args.block_size}")
        sys.exit(1)

except Exception as e:
    logger.error(f"Error checking data: {e}")
    sys.exit(1)

# Train/Val Split
n = int(0.9 * len(data_tensor))
train_data = data_tensor[:n]
val_data = data_tensor[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    # Safety Check for small validation sets
    if len(data) <= args.block_size:
         ix = torch.randint(len(data)-1, (args.batch_size,))
         x = torch.stack([data[i:i+1] for i in ix])
         y = torch.stack([data[i+1:i+2] for i in ix])
         return x.to(device), y.to(device)

    ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
    x = torch.stack([data[i:i+args.block_size] for i in ix])
    y = torch.stack([data[i+1:i+args.block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            X, Y = get_batch(split)
            logits = model(X)
            loss = F.cross_entropy(logits.view(-1, vocab_size), Y.view(-1))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ------------------
# MODEL INITIALIZATION
# ------------------
model = GPT(
    vocab_size=vocab_size,
    block_size=args.block_size,
    embed_dim=args.embed_dim,
    num_layers=args.num_layers,
    num_heads=args.num_heads
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

start_step = 0
best_val_loss = float('inf')

if args.resume_from:
    if os.path.exists(args.resume_from):
        logger.info(f"Resuming from {args.resume_from}...")
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step']
        if 'loss' in checkpoint:
            best_val_loss = checkpoint['loss']
            logger.info(f"Resumed best val loss: {best_val_loss:.4f}")
    else:
        logger.warning(f"Checkpoint {args.resume_from} not found. Starting from scratch.")

logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# ------------------
# TRAINING LOOP
# ------------------
logger.info("Starting training...")
start_time = time.time()

try:
    for step in range(start_step, args.max_steps):
        
        # Evaluation & Checkpointing
        if step % args.eval_interval == 0 or step == args.max_steps - 1:
            losses = estimate_loss(model)
            dt = time.time() - start_time
            logger.info(f"step {step} | time {dt:.2f}s | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
            start_time = time.time() # Reset timer
            
            # Save periodic checkpoint
            ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_{step}.pt")
            checkpoint_data = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': losses['val'],
                'config': vars(args)
            }
            torch.save(checkpoint_data, ckpt_path)
            
            # Save recent model
            torch.save(model.state_dict(), "gpt.pth")
            
            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                best_path = os.path.join(args.checkpoint_dir, "best_model.pt")
                torch.save(checkpoint_data, best_path)
                logger.info(f"New best model saved! Val loss: {best_val_loss:.4f}")

        # Training step
        xb, yb = get_batch('train')
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

except KeyboardInterrupt:
    logger.info("Training interrupted by user. Saving checkpoint...")
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': 0.0, # Unknown
    }, os.path.join(args.checkpoint_dir, "interrupt_ckpt.pt"))
    logger.info("Saved interrupt_ckpt.pt")
    sys.exit(0)

logger.info("Training complete.")
torch.save(model.state_dict(), "gpt.pth")

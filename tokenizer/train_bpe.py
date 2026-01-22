from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# ------------------
# Configuration
# ------------------
VOCAB_SIZE = 4096      # better for 10MB corpus
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

# ------------------
# Create tokenizer
# ------------------
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=SPECIAL_TOKENS
)

# ------------------
# Train
# ------------------
tokenizer.train(
    files=["data/corpus.txt"],
    trainer=trainer
)

# ------------------
# Save tokenizer
# ------------------
tokenizer.save("tokenizer/tokenizer.json")

print("Tokenizer trained and saved.")

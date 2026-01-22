# Simple LLM

A simple GPT-style language model implementation on a custom corpus.

## Requirements

You need Python 3.8+ and the following packages:

- `torch` (PyTorch)
- `tokenizers` (Hugging Face Tokenizers)
- `requests` (For scraping data)

Install them via pip:

```bash
pip install torch tokenizers requests
```

## Setup & Training Workflow

Follow these steps to prepare your data and train the model.

### 1. Generate Training Data
Run the scraper to fetch data from Simple Wikipedia (or replace `data/corpus.txt` with your own text file).
```bash
python scrape_wiki.py
```
This will create `data/corpus.txt`.

### 2. Clean Data (Optional)
If you want to normalize the text (lowercase, remove garbage):
```bash
python clean_corpus.py
```
*Note: If you run this, make sure to point `train.py` to the cleaned file or rename `data/corpus_clean.txt` to `data/corpus.txt`.*

### 3. Train Tokenizer
The project uses a BPE tokenizer. You can verify or retrain it on your new corpus:
```bash
python tokenizer/train_bpe.py
```
This saves the tokenizer model to `tokenizer/tokenizer.json`.

### 4. Train the Model
Now you can start training. The script uses GPU if available.
```bash
python train.py
```

Arguments:
- `--batch_size`: Batch size (default: 64)
- `--block_size`: Context length (default: 256)
- `--max_steps`: Total training steps (default: 10000)
- `--resume_from`: Path to a checkpoint to resume training (e.g., `checkpoints/ckpt_500.pt`)

Example:
```bash
python train.py --batch_size 32 --max_steps 5000
```

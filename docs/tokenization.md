# Tokenization Engine in nano-gpt-prod

The tokenization engine provides clean, modular, and high-performance tokenizers for training and evaluating autoregressive Transformer models.

## Available Tokenizers

1. **Character-Level Tokenizer (`CharTokenizer`)**:
   - Ideal for educational experiments, character-level generative modeling (e.g. Tiny Shakespeare), and small memory footprints.
   - Preserves exact characters, supports special tokens (`<|endoftext|>`, `<|pad|>`, `<|unk|>`).
   - Zero out-of-vocabulary surprises with `<|unk|>` fallback.

2. **Byte-Pair Encoding (`BPETokenizer`)**:
   - Full implementation of the BPE algorithm trained directly from raw bytes (0–255).
   - Iteratively merges the most frequent byte pairs up to a target vocabulary size.
   - Fast merge lookup and UTF-8 byte fallback (never fails on unseen characters or emojis).

3. **Tiktoken Wrapper (`TiktokenTokenizer`)**:
   - Production integration with OpenAI's official `tiktoken` library.
   - Supports `gpt2`, `cl100k_base` (GPT-4 / ChatGPT), and custom encodings.

## Python Usage Examples

### 1. Training a Custom BPE Tokenizer

```python
from gpt.tokenizer import BPETokenizer

# Train on arbitrary raw text
corpus = open("input.txt", "r", encoding="utf-8").read()
tokenizer = BPETokenizer.train_from_text(corpus, vocab_size=1000)

# Encode text
token_ids = tokenizer.encode("To be or not to be, that is the question:")
print("Tokens:", token_ids)

# Decode back to string
decoded_text = tokenizer.decode(token_ids)
print("Decoded:", decoded_text)

# Save and load
tokenizer.save("data/my_tokenizer.json")
loaded_tokenizer = BPETokenizer.load("data/my_tokenizer.json")
```

### 2. Using the Unified Tokenizer Factory

```python
from gpt.tokenizer import get_tokenizer

# Automatically creates tokenizer based on name or saved file
tok_char = get_tokenizer("char", text="Sample training text...")
tok_bpe = get_tokenizer("data/my_tokenizer.json")
```

### 3. Training via CLI

```bash
python scripts/train_tokenizer.py --input input.txt --output data/bpe_1000.json --type bpe --vocab_size 1000
```

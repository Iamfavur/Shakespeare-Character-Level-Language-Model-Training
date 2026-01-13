# Shakespeare Character-Level Language Model Training (PyTorch)

This codebase contains a set of character-level language models trained on Shakespeare-like text, progressing from a simple Bigram model to a Transformer with multi-head self-attention. It also includes educational scripts that illustrate attention mechanics and small utilities for device introspection.

Contents:

- bigram.py — Minimal character bigram model.
- main.py — GPT-like Transformer (token + position embeddings, multi-head attention, feed-forward, residuals, layer norms).
- test.py — Step-by-step educational notebook-style script, including toy attention demos.
- input.txt — Training corpus (character-level).

All models use raw character tokens, building an index mapping (stoi/itos) from the dataset found in input.txt.

---

## 1) Quick start

Prerequisites:

- Python 3.9+ (3.10 recommended)
- PyTorch (CPU is fine; CUDA if you have an NVIDIA GPU)

Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# .\venv\Scripts\activate  # Windows PowerShell
pip install --upgrade pip
pip install torch  # see https://pytorch.org/get-started/locally/ for GPU-specific wheels
```

The dataset is already present as input.txt. If you want to replace it, ensure the file remains at the project root with UTF-8 encoding.

Run the models:

```bash
# Bigram model
python bigram.py

# Transformer model
python main.py

# Educational script (toy attention)
python test.py
```

---

## 2) Data and vocabulary

All scripts read the raw text and build a character vocabulary:

- chars = sorted(list(set(text))) — all unique characters
- stoi/itos — character↔index maps
- encode(s) / decode(l) — convenience helpers

The entire dataset is encoded to a 1D LongTensor:

```python
data = torch.tensor(encode(text), dtype=torch.long)
```

Dataset splits:

- First 90% → train_data
- Last 10% → val_data

---

## 3) Batching and sequence windows

Each training step samples random windows of fixed length block_size from the dataset.

Shapes:

- B = batch_size
- T = block_size

get_batch(split):

- x: (B, T) — indices for current tokens
- y: (B, T) — indices for next-token targets (x shifted by 1)

Example (conceptually):

- x = data[i : i+T]
- y = data[i+1 : i+T+1]

This is used across bigram.py, main.py, and test.py (with minor variations).

---

## 4) Models

### 4.1 Bigram model (bigram.py, also in test.py)

A minimal baseline: one embedding table directly outputs logits for the next character.

Key points:

- token_embedding_table: nn.Embedding(vocab_size, vocab_size)
- forward(idx, targets):
  - logits: (B, T, C=vocab_size)
  - loss: cross-entropy over flattened (B*T, C) vs (B*T)
- generate(idx, max_new_tokens):
  - Iteratively sample next token from softmax(logits[:, -1, :])

Why it works:

- Learns a next-character conditional distribution p(next | current).
- No context beyond 1 token (T only used for batched scoring).

### 4.2 Transformer model (main.py)

A GPT-style decoder-only stack with causal self-attention.

High-level flow:

- Token embeddings: nn.Embedding(vocab_size, n_embd)
- Positional embeddings: nn.Embedding(block_size, n_embd)
- Blocks: n_layer repeated
  - LayerNorm → Multi-head Attention → residual
  - LayerNorm → FeedForward → residual
- Final LayerNorm
- Linear head to vocab_size for logits

Important hyperparameters (top of main.py):

- batch_size, block_size, max_iters, eval_interval
- learning_rate, n_embd, n_head, n_layer, dropout

Attention math in each Head:

- k = W_k x, q = W_q x, v = W_v x where shapes ∈ (B, T, head_size)
- scores = (q @ k^T) / sqrt(head_size) → (B, T, T)
- Causal mask with lower-triangular matrix (tril) prevents attending to future tokens
- Softmax over last dimension (keys axis), then weighted sum: out = softmax(scores) @ v

MultiHeadAttention:

- Concatenate head outputs along the channel dimension: (B, T, num_heads \* head_size)
- Project back to n_embd with a Linear layer: nn.Linear(num_heads \* head_size, n_embd)
  - This allows n_embd not necessarily divisible by n_head
  - If you want even splits per head, commonly choose head_size = n_embd // n_head

FeedForward (per token):

- MLP: Linear(n_embd → 4*n_embd) → ReLU → Linear(4*n_embd → n_embd) → Dropout

Loss:

- Cross-entropy over (B*T, vocab_size) vs (B*T) targets

Generation:

- Crop context to last block_size tokens
- Feed forward, take logits at last step, softmax → sample → append

---

## 5) Training loop and evaluation

Training (bigram.py and main.py):

- Optimizer: AdamW
- Periodic evaluation via estimate_loss() using @torch.no_grad()
- Losses are averaged across multiple batches for both train and val

Reproducibility:

- torch.manual_seed(1337) is set in each script
- For strict determinism across platforms/backends, additional flags may be needed (e.g., torch.use_deterministic_algorithms), but that isn’t strictly required here.

---

## 6) Devices (CPU/CUDA/MPS)

- main.py and bigram.py choose device = 'cuda' if available, else CPU.


## 7) Educational script (test.py)

This file is a self-contained walkthrough:

- Data encoding and batching (as above)
- Bigram language model training
- Toy attention demos that illustrate:
  - Causal masking via tril
  - Scaled dot-product attention
  - Why softmax(dim=-1) matters (normalize across keys dimension)
  - Attending to v (values), not to raw x
  - Importance of scaling by 1/sqrt(head_size)

Shapes to remember in attention:

- x: (B, T, C)
- q, k, v: (B, T, head_size)
- scores = q @ k^T: (B, T, T)
- weights = softmax(scores, dim=-1)
- out = weights @ v: (B, T, head_size)

Common pitfalls demonstrated:

- Using softmax over wrong dimension (e.g., dim=0 or dim=1) can produce NaNs due to masked -inf columns.
- Forgetting scaling leads to overly peaky softmax.

---

## 8) Running and tuning

Bigram:

```bash
python bigram.py
# Tune: batch_size, block_size, max_iters, learning_rate
```

Transformer:

```bash
python main.py
# Tune: n_embd, n_head, n_layer, dropout, block_size, max_iters, learning_rate
```

Generation is automatically printed at the end of scripts:

- Seed context: a single token of 0 (often the first vocab index)
- Increase max_new_tokens for longer samples

---

## 10) Extending the project

- Add checkpointing (save/load model/optimizer state_dict).
- Add learning rate schedulers and gradient clipping.
- Increase context length (block_size), embedding size (n_embd), and layers (n_layer) for richer models.
- Swap ReLU for GELU in the feed-forward.
- Add attention/MLP dropout and embedding dropout.
- Port device selection to support MPS directly in main.py/bigram.py if desired.

---

## 11) File-by-file summary

- input.txt
  - Training corpus. UTF-8 plain text.
- bigram.py
  - Character bigram language model, training loop, text generation.
- main.py
  - Transformer language model with multi-head attention, feed-forward blocks, residuals and norms; training + generation.
- test.py
  - Educational script showing end-to-end bigram training and detailed attention math with toy tensors.
---

## 12) Expected outputs

- Loss decreases across training (both bigram and transformer).
- Generated text starts as noise, gradually picks up Shakespeare-like structure at character level.
- In attention demos (test.py), weights form a lower-triangular pattern due to causal masking.


---

Happy experimenting!

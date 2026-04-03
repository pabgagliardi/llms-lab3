# Stage I — Character-level GPT Decoder

## File structure

```
gpt_model.py   ← model architecture (all components)
config.py      ← hyperparameter dataclass + design justifications
dataset.py     ← tokeniser + sliding-window Dataset
train.py       ← training loop, benchmarking, generation
```

---

## Architecture overview

```
Input chars  ──►  token_emb (nn.Embedding)
                     │
                  + pos_emb  ─── learned / sinusoidal / rope
                     │
           ┌─── × N Transformer blocks ───┐
           │  LayerNorm (pre-norm)         │
           │  MultiHead CausalSelfAttn     │ ← flash or standard backend
           │  residual add                 │
           │  LayerNorm (pre-norm)         │
           │  FeedForward (GELU, 4×)       │
           │  residual add                 │
           └──────────────────────────────┘
                     │
              LayerNorm_final
                     │
               lm_head (Linear)  ←── weight-tied with token_emb
                     │
              logits (B, T, V) ──► cross-entropy loss
```

### Key design choices

| Component | Choice | Reason |
|---|---|---|
| Norm placement | Pre-norm | More stable gradients; used in GPT-3, PaLM |
| Activation | GELU | Standard in GPT-family; smoother than ReLU |
| Attention QKV | Fused single projection | One matmul instead of three |
| Weight tying | emb ↔ lm_head | Reduces params, regularises token representations |
| Pos embedding | **Learned** (default) | Best for fixed-length, short poetry lines; see config.py |

---

## Attention backends

### Flash (`--backend flash`)
```python
F.scaled_dot_product_attention(q, k, v, is_causal=True)
```
Automatically dispatches to FlashAttention (if CUDA + compatible GPU), or
falls back to a memory-efficient kernel. No explicit mask tensor is created.
**Expected speedup**: 2–4× on A100/H100; ~1.5× on consumer GPUs.

### Standard (`--backend standard`)
```python
scores = (q @ k.T) / sqrt(d)
scores.masked_fill(upper_triangle, -inf)
softmax(scores) @ v
```
Classical implementation with a materialised triangular mask.
Useful as a reference baseline and for pedagogical inspection.

---

## Positional embeddings

| Type | Params | Extrapolates? | Use when |
|---|---|---|---|
| `learned` | yes | no | fixed max_len, best empirical perf |
| `sinusoidal` | no | limited | length generalisation needed |
| `rope` | no | yes | long seqs, production |

---

## Quick start

```bash
# Install dependencies
pip install torch

# Copy corpus
cp corpus_small_clean.txt .

# Train with defaults (flash attention, learned pos emb)
python train.py --corpus corpus_small_clean.txt

# Train with standard attention
python train.py --backend standard

# Benchmark flash vs standard (speed + memory)
python train.py --benchmark

# Sinusoidal positional encoding
python train.py --pos sinusoidal

# Larger model
python train.py --d_model 512 --n_layers 8 --n_heads 8
```

---

## Benchmark comparison

Run `python train.py --benchmark` to produce a table like:

```
┌──────────────┬─────────────────┬───────────────────┐
│ Backend      │ Avg step (ms)   │ Peak GPU mem (MB) │
├──────────────┼─────────────────┼───────────────────┤
│ flash        │          12.34  │             142.0 │
│ standard     │          18.71  │             287.5 │
└──────────────┴─────────────────┴───────────────────┘
```

Flash attention is faster and uses significantly less memory because it avoids
materialising the full (T × T) attention matrix by computing attention in tiles
that fit in SRAM.

---

## Monitoring (Stage III preview)

Training logs are written to `checkpoints/training_log.csv`:

```
step, train_loss, val_loss, lr, elapsed_s
0,    3.8421,     3.8563,   0.000006, 1.2
200,  2.1045,     2.1892,   0.000300, 47.8
...
```

Text samples are generated every 400 steps to qualitatively track learning.

---

## Checkpoints

- `checkpoints/best.pt`       — best validation loss
- `checkpoints/step_N.pt`     — periodic snapshots (every 1000 steps)

Each checkpoint contains model weights, optimizer state, config, and the
tokeniser vocabulary — everything needed to resume training or run probing
experiments in Stage IV.

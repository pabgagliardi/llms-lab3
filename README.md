# 🧠 GPT Decoder — Character-Level Language Model

A from-scratch GPT-style decoder-only transformer trained on a text corpus (originally designed for poetry). Built to run end-to-end in a single Google Colab notebook with GPU acceleration.

---

## ✨ Features

- **Character-level tokenizer** — no external tokenization libraries needed
- **Configurable positional embeddings** — choose between Learned, Sinusoidal, or Rotary (RoPE)
- **Attention backends** — Flash Attention (`scaled_dot_product_attention`) or standard softmax attention
- **Weight tying** — token embedding and LM head share weights
- **Cosine LR schedule** with linear warmup
- **Gradient accumulation** for larger effective batch sizes
- **Periodic checkpointing** — saves raw (uncompiled) state dicts
- **Periodic sampling** — generates text samples during training to track qualitative progress
- **BPC metric** (bits per character) for evaluation alongside cross-entropy loss
- **`torch.compile` support** — ~10–30% free speedup on PyTorch 2.0+

---

## 🚀 Quickstart (Google Colab)

1. Open `TRAINING_Model_one_cell.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Set the runtime to **GPU** (T4 or better recommended): `Runtime → Change runtime type → GPU`
3. Run the cells in order:
   - **Cell 1** — checks PyTorch and CUDA availability
   - **Cell 2** — uploads your corpus (a plain `.txt` file)
   - **Cell 3** — defines all model classes and utilities
   - **Cell 4** — builds the dataset and instantiates the model
   - **Cell 5** — runs the training loop

---

## ⚙️ Configuration

All hyperparameters live in the `GPTConfig` dataclass. Key defaults:

| Parameter | Default | Description |
|---|---|---|
| `d_model` | 256 | Model embedding dimension |
| `n_heads` | 8 | Number of attention heads |
| `n_layers` | 6 | Number of transformer blocks |
| `max_len` | 256 | Context window (sequence length) |
| `dropout` | 0.1 | Dropout rate |
| `batch_size` | 64 | Batch size per step |
| `lr` | 3e-4 | Peak learning rate |
| `weight_decay` | 0.1 | AdamW weight decay |
| `grad_clip` | 1.0 | Gradient norm clipping |
| `max_iters` | 20,000 | Total training steps |
| `warmup_iters` | 500 | Linear LR warmup steps |
| `grad_accum_steps` | 1 | Steps for gradient accumulation |
| `eval_interval` | 200 | Steps between evaluations |
| `sample_interval` | 2,000 | Steps between text samples |
| `checkpoint_interval` | 2,000 | Steps between checkpoint saves |
| `attn_backend` | `"flash"` | `"flash"` or `"standard"` |
| `pos_emb_type` | `"learned"` | `"learned"`, `"sinusoidal"`, or `"rope"` |
| `train_frac` | 0.9 | Fraction of corpus used for training |

---

## 🏗️ Architecture

```
Input tokens
    │
Token Embedding + Positional Embedding (Learned / Sinusoidal / RoPE)
    │
┌───┴─────────────────────────────────┐
│  TransformerBlock × n_layers        │
│  ┌─────────────────────────────┐    │
│  │  LayerNorm                  │    │
│  │  MultiHeadCausalSelfAttn    │    │
│  │  + residual                 │    │
│  ├─────────────────────────────┤    │
│  │  LayerNorm                  │    │
│  │  FeedForward (GELU, ×4)     │    │
│  │  + residual                 │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
    │
LayerNorm → LM Head (weight-tied to token embedding)
    │
Logits / Cross-Entropy Loss
```

---

## 📁 Corpus Format

Upload any plain UTF-8 `.txt` file. The tokenizer builds its vocabulary automatically from the unique characters in the file — no preprocessing required.

A clean, de-duplicated corpus will yield better results. Recommended minimum size: **~500 KB**.

---

## 💾 Checkpoints

Checkpoints are saved every `checkpoint_interval` steps to the `checkpoints/` directory. Each file contains:

```python
{
    "step": int,
    "model_state": state_dict,   # always raw, never compiled
    "config": GPTConfig,
    "tokenizer_vocab": list,
}
```

To resume or do inference, load with:

```python
ckpt = torch.load("checkpoints/step_XXXXX.pt")
config = ckpt["config"]
vocab  = ckpt["tokenizer_vocab"]
# rebuild tokenizer and model from config, then load state dict
```

---

## 📊 Metrics

| Metric | Description |
|---|---|
| `train_loss` / `val_loss` | Average cross-entropy loss |
| `train_bpc` / `val_bpc` | Bits per character (lower = better) |
| Random baseline BPC | `log2(vocab_size)` — the score of a random model |

---

## 🛠️ Requirements

| Package | Notes |
|---|---|
| `torch >= 2.0` | For Flash Attention and `torch.compile` |
| `google-colab` | For the file upload widget (Cell 2) |

No other external dependencies.

---

## 📝 License

MIT — feel free to use, modify, and share.

---

## 🎨 Inference

Use `INFERENCE.ipynb` to generate text from a saved checkpoint — no retraining needed.

### Quickstart

1. Open `INFERENCE.ipynb` in Google Colab (GPU recommended but not required)
2. **Cell 1** — loads all class definitions and the checkpoint you upload
3. **Cell 2** — runs generation across a set of prompts and saves the output to `generated_poems.txt`

### Generation parameters

| Parameter | Default | Description |
|---|---|---|
| `temperature` | 0.8 | Higher = more creative/random, lower = more conservative |
| `top_k` | 40 | Limits sampling to the top-k most likely next characters |
| `n_chars` | 100,000 | Number of characters to generate per prompt |

### Default prompts

The notebook ships with 8 starter prompts:

```
"The night is", "When silence falls", "I have seen the",
"In the dark of", "The wind that blows", "The", "I", "A"
```

You can swap these out freely — just edit the `prompts` list before running Cell 2.

### Output

All generated texts are saved to `generated_poems.txt` in the Colab working directory. You can download it via `Files → generated_poems.txt → Download`.

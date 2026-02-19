# RangeFlow-GPT — Training a GPT-2 Scale LLM from Scratch

> ⚠️ **Experimental Project** — This is a personal research and learning experiment. The model was trained under non-ideal conditions with suboptimal dataset choices, and the results reflect that. It is not a production-ready language model. Read the [Honest Assessment](#honest-assessment-what-went-wrong-and-why) section before using the weights.

---

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Training Infrastructure](#training-infrastructure)
- [Dataset Strategy (and the mistakes)](#dataset-strategy-and-the-mistakes)
- [Training Timeline](#training-timeline)
- [Training Metrics](#training-metrics)
- [What the Model Actually Learned](#what-the-model-actually-learned)
- [Honest Assessment: What Went Wrong and Why](#honest-assessment-what-went-wrong-and-why)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [Reproducing Training](#reproducing-training)
- [Log Analysis Tools](#log-analysis-tools)
- [Lessons Learned](#lessons-learned)

---

## Overview

This project trains a GPT-2-scale (~124M parameter) autoregressive language model entirely from scratch — no pretrained weights, no transfer learning. The model uses the GPT-2 tokenizer (`tiktoken`, `gpt2` encoding, vocab size 50,257) and is trained on Google Colab  using mixed-precision training on a single GPU.

The training run covered roughly **1,680+ logged steps** (continuing from a checkpoint at step 600, so the actual total is higher), and suffered from a significant midcourse dataset strategy change that impacted learning continuity.

**Bottom line:** The model learned to produce fluent-looking English text. It does a passable job of story continuation and free-form rambling. It fails at instruction following, factual recall, reasoning, and coding — largely due to dataset choices, not architecture.

---

## Model Architecture

A clean GPT-2-style decoder-only transformer, implemented from scratch in PyTorch.

| Component | Value |
|-----------|-------|
| Architecture | Decoder-only Transformer (GPT-2 style) |
| Parameters | ~124M |
| Vocabulary | 50,257 (GPT-2 / tiktoken `gpt2`) |
| Embedding dimension (`d_model`) | 768 |
| Transformer layers | 12 |
| Attention heads | 12 |
| Head dimension (`d_k`) | 64 |
| Feed-forward dimension (`d_ff`) | 3,072 (4× d_model) |
| Max sequence length | 1,024 tokens |
| Dropout | 0.1 |
| Attention type | Causal multi-head (manual `tril` mask) |
| Positional encoding | Learned absolute positional embeddings |
| Normalization | Pre-norm LayerNorm (applied before attention and FFN) |
| Activation | GELU |
| Weight tying | Token embedding and output projection share weights |
| Weight initialization | Normal(0, 0.02) — same as GPT-2 paper |

### Architecture Notes

- **Pre-norm layout:** LayerNorm is applied *before* the attention and FFN sublayers (not after, as in the original "Attention Is All You Need" paper). This is the GPT-2 convention and improves training stability.
- **Weight tying:** The input embedding matrix and the output linear projection share the same weights. This reduces the parameter count by ~38.6M and is standard practice for language models.
- **No Flash Attention / SDPA:** The attention is computed manually (QKV projection → softmax → weighted sum), without using `torch.nn.functional.scaled_dot_product_attention`. This is fine for research but slower than optimized kernels.
- **No RoPE / ALiBi:** Positional information comes from learned absolute embeddings capped at 1,024. The model cannot generalize to sequences longer than this.

---

## Training Infrastructure

Training ran on **Google Colab Pro** with a single NVIDIA A100 GPU (the log consistently shows ~14.8 GB GPU memory usage).

| Setting | Value |
|---------|-------|
| Platform | Google Colab (T4 GPU) |
| GPU memory used | ~14.8 GB |
| Mixed precision | Yes — `torch.cuda.amp` (FP16) |
| Optimizer | `bitsandbytes` AdamW 8-bit (`bnb.optim.AdamW8bit`) |
| Learning rate | 1e-4 (peak) |
| LR schedule | Linear warmup (1,750 steps) → cosine decay to max step |
| Warmup steps | 1,750 |
| Max planned steps | 25,000 |
| Batch size | 6 sequences per forward pass |
| Gradient accumulation | 21 steps |
| Effective batch size | 6 × 21 × 1,024 = **129,024 tokens per update** |
| Gradient clipping | Max norm 1.0 |
| Weight decay | 0.1 |
| Adam betas | (0.9, 0.95) |
| Checkpoint frequency | Every 100 steps |
| Checkpoint retention | Last 3 only |
| Sample generation | Every 100 steps |
| Data storage | Google Drive (tokenized `.npy` files) |
| Local cache | Colab `/content/` SSD (~30 GB chunks) |
| Throughput | ~8,360–8,400 tokens/second (very stable) |

The 8-bit AdamW optimizer (`bitsandbytes`) was chosen to reduce optimizer memory overhead, allowing a larger effective batch with the 14.8 GB GPU budget.

---

## Dataset Strategy (and the mistakes)

This is where things went wrong, and it's worth being honest about.

### Phase 1: C4 Only (~15,000 steps, not logged here)

The first ~15,000 training steps used only **C4 (Common Crawl, cleaned English)** — a large web-crawled corpus of general English text. C4 is a reasonable pre-training corpus for a model this size, but training on it *alone for that long* created a model biased toward:

- Web-style prose (news articles, blog posts, product descriptions)
- Completing sentences in a journalistic/descriptive tone
- No instruction following whatsoever (it's raw completion data)

The C4 phase was not logged in the available log files (training resumed from step 600 in the captured log), but based on the loss values at step 610 (~6.27) and the model's behavior in samples, a significant amount of pre-training had already happened before this log begins.

### Phase 2: Dataset Switch (the key mistake)

Partway through training, the dataset was switched to a **mixed corpus** without the C4 data:

| Dataset | Mix Ratio | Purpose |
|---------|-----------|---------|
| Cosmopedia (`web_samples_v1`) | 50% | Educational/synthetic web content |
| Stanford Alpaca | 30% | Instruction-following pairs |
| Python (CodeAlpaca-20k / The Stack) | 20% | Code reasoning |

The mixing was handled by `HybridMixedDatasetLoader` with a pre-generated random schedule (seeded at 42) and a `DatasetStateManager` that persisted file-level bookmarks to Google Drive to survive Colab restarts.

**Why this was a problem:**

1. **C4 only training.** A model as small as 124M trained on web text was bounded to do this, web text is utterly random, it gave the model english understanding and grammer, but failed to give overall semantic understanding we expect from it.

2. **Abrupt distribution shift.** The model had spent ~15,000 steps learning C4's statistical patterns. Suddenly switching to a completely different data distribution (educational text + instructions + code) caused the model to partially "unlearn" C4 patterns without fully learning the new ones. This is visible in the loss curves — the loss at step 610 is 6.27 and only reaches under 2.0 at the best point.

3. **Alpaca formatting mismatch.** The Alpaca data was formatted with `<|user|>` and `<|assistant|>` special tokens, but the GPT-2 tokenizer doesn't have these as special tokens — they get split into subword pieces. The model never learned to treat them as control signals, making instruction following structurally impossible.

4. **Cosmopedia at 50% dominance.** Cosmopedia's educational/synthetic content has a different style from both C4 and Alpaca. Rather than helping, it created a three-way tug-of-war that prevented clean convergence on any style.

---

## Training Timeline

```
Steps 1–~15,000   →  C4 only (not in this log)
                       Large web text corpus, pure completion pretraining

Step 10           →  First logged step. Loss: 10.0179

Steps >15,000     →  Mixed dataset (Cosmopedia 50% / Alpaca 30% / Python 20%)
                       Loss slowly improving with high variance

Step 1000         →  First sample generation checkpoint
```

The training was designed to run to 25,000 steps. It never got there.

---

## Training Metrics

All metrics are extracted from the training log via `llm_postprocess.py`, you ca see [here](./metadata/)


### Loss Curve Observations

Loss is **noisy and high-variance** throughout — characteristic of a model in distribution conflict between its prior training (C4) and the new mixed dataset.

The training curves PNG (`llm_training_curves.png`) shows loss, learning rate schedule, tokens/sec throughput, and GPU memory across all logged steps.

---

## What the Model Actually Learned

Sample generations were recorded every 100 steps using two fixed prompts. Here's an honest look at progression:

### Prompt: `"The future of AI is"`

**Step 700** (early):
> *"...also not only be certain old looking to be affected by experts that I's most of my image. I' I usually a lot of my error in that I am a child're making me to go. It'm"*

Very incoherent. Token-level fluency exists but grammatical structure is broken. The model is essentially sampling from a confused prior.

**Step 1,000** (mid-run):
> *"...able to sell a company. The tool is a little bit of strong, and the source of the company has worked on social media and can be a successful Google on a smartphone..."*

Noticeably better. Complete sentences, real nouns, some semantic coherence. Still drifts into unrelated territory (company → Google → smartphone).

**Step 1,600** (late):
> *"...a Man of the Star Wars Cup – which is a good news. Totally, but he will be able to push for another 2018 Championship, therefore it is little shame to have been a decisive choice for the winning season..."*

Grammatically fluent but topically nonsensical — it shifted from AI to sports. The model is generating plausible-looking English without any real concept of the prompt's topic.

### Prompt: `"Once upon a time,"`

**Step 700:**
> *"...which was not dropped it really long-preometric hurdles. The pre-3-shaped standard of the printer details of the screen/X_AIDS-commercial Spring..."*

Completely broken.

**Step 1,000:**
> *"...consistently in a few days. Nitthan, who set up a minimum. Are you feel teachers there. You need to make a business, just "moving". Our goal is also a lot of where we start the"*

Structure is better (short sentences, discourse markers), still hallucinating and drifting.

**Step 1,600:**
> *"...I believe that it's not only thought to be the same thing about me and I'd do a good job. I've both been working on a three-year weekend away. I must go on the other side so"*

First-person narrative rambling. This is actually the model's strongest mode — it can produce something that reads like a personal blog post or inner monologue. Not coherent, but fluent.

### Summary of Capabilities

| Task | Performance |
|------|-------------|
| Story rambling / continuation | ⚠️ Mediocre but functional — most consistent capability |
| Free-form prose generation | ⚠️ Can produce fluent sentences, drifts off-topic |
| Factual statements | ❌ Mostly wrong or confabulated |
| Instruction following | ❌ Does not follow instructions at all |
| Question answering | ❌ Treats questions as prompts to continue generating prose |
| Code generation | ❌ Never reliably produces valid code |
| Logical reasoning | ❌ No structured reasoning observed |

---

## Honest Assessment: What Went Wrong and Why

### 1. Dataset switching without curriculum design
Switching from C4 to a mixed corpus halfway through training without any curriculum strategy (like gradually blending in new data) created a distribution conflict. The model was forced to simultaneously "remember" C4 patterns and "learn" new ones, doing neither well.

**What to do instead:** Either commit to one corpus from step 0, or use a curriculum where the new data is gradually introduced — e.g., 90% C4 + 10% new at the switch point, then linearly increase the new data's share over thousands of steps.

### 2. Alpaca formatting is incompatible with GPT-2 tokenizer
Using `<|user|>` and `<|assistant|>` as delimiter tokens when the GPT-2 tokenizer has no concept of them means the model sees `<`, `|`, `user`, `|`, `>` as separate tokens. There is no signal to learn that this boundary means "switch speaker." The GPT-2 special token for end-of-text (`<|endoftext|>`) *is* a real special token (ID 50256) — but the chat delimiters are not.

**What to do instead:** Use a tokenizer that natively supports chat special tokens (e.g., `<|im_start|>` and `<|im_end|>` in the ChatML format used by GPT-4/Qwen, or the Llama-style `[INST]` tokens), or add custom special tokens to the tokenizer and resize the embedding matrix.

### 3. Too few steps on the new data
~1,000 effective steps (logged portion after the switch) with 50/30/20 mixing gives each domain very little exposure. The Chinchilla scaling laws suggest a 124M parameter model needs roughly 2.5 billion tokens for optimal training. This run covered approximately:

```
1,688 steps × 129,024 tokens/step ≈ 217.8 million tokens (logged portion only)
```

Even including the ~15,000 C4 steps:
```
15,000 steps × ~130,000 tokens/step ≈ 1.95 billion tokens (estimated)
```

The total is in the right ballpark for Chinchilla-optimal, but the dataset quality and distribution problems dilute the effective learning signal significantly.

### 4. No evaluation set / no perplexity tracking
Training loss was tracked but no held-out validation set was used. Without a validation loss, it's impossible to know whether the model is overfitting to the training data or actually generalizing. The loss spike at step 760 might be overfitting beginning, or just a bad batch — there's no way to tell.

### 5. Colab session drops caused gaps
The training was interrupted multiple times by Colab disconnections. While the checkpoint system worked correctly (the `DatasetStateManager` saves progress to Drive), each restart loses the optimizer's momentum state (since it's tied to the current session), which can temporarily destabilize training.

---

## Project Structure

```
.
├── llm_model_from_scratch.ipynb      # Main training notebook (Colab)
│   ├── Cell 1   — Package installation (bitsandbytes)
│   ├── Cell 2   — Library imports, Drive mount
│   ├── Cell 3   — Config class (all hyperparameters)
│   ├── Cell 4   — SmartResumeTokenDataset (single-file dataset)
│   ├── Cell 5B  — HybridMixedDatasetLoader (multi-dataset with state tracking)
│   ├── Cell 6   — Model architecture (MultiHeadAttention, FeedForward, GPTModel)
│   ├── Cell 7   — Training utilities (logger, LR schedule, checkpoint save/load, sampler)
│   ├── Cell 8   — train_complete() — main training loop
│   ├── Cell 9   — Entry point (kicks off training)
│   ├── Cell 9b  — C4 state removal utility (for dataset switching)
│   ├── Cell 10  — Interactive text generation (streaming token-by-token)
│   ├── Cell 11  — Training visualization (matplotlib curves)
│   ├── Cell 12  — C4 downloader/tokenizer (one-time, ~2–6 hours)
│   ├── Cell 13  — Cosmopedia + Alpaca + Python downloader (one-time, ~3–5 hours)
│   └── Cell 14  — Python-only downloader (CodeAlpaca / The Stack)
│
├── llm_postprocess.py                # Log parser, JSON exporter, plotter
    ├── Parses training_*.log files
    ├── Outputs: llm_training_metrics.json, llm_training_samples.json
    └── Outputs: llm_training_curves.png (4-panel plot)

```

---

## Setup & Usage

### Requirements

```bash
cd backend
pip install -r requirements.txt
```

Python 3.10+ recommended. GPU with at least 16 GB VRAM for the same config. The notebook is designed for **Google Colab** with Google Drive mounted.

### Loading a Checkpoint

```python
import torch
import tiktoken
from pathlib import Path

# Define the same Config class as in the notebook
# (copy Cell 3 from the notebook)

model = GPTModel(Config).cuda()
checkpoint = torch.load("checkpoint_step_1688.pt", map_location="cuda", weights_only=False)

# Handle checkpoints from torch.compile() if needed
state_dict = checkpoint["model_state_dict"]
if any(k.startswith("_orig_mod.") for k in state_dict):
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()

print(f"Loaded from step {checkpoint['step']}, loss {checkpoint['loss']:.4f}")
```

### Generating Text

```python
tokenizer = tiktoken.get_encoding("gpt2")

def generate(model, prompt, max_tokens=150, temperature=0.8):
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device="cuda").unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_tokens):
            if tokens.size(1) >= 1024:
                tokens = tokens[:, -1024:]

            logits = model(tokens)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

            if next_token.item() == tokenizer.eot_token:
                break

    return tokenizer.decode(tokens[0].cpu().numpy())

# Best use case: story/prose continuation
print(generate(model, "Once upon a time, in a forest far from the city,"))
print(generate(model, "The old scientist walked into the lab and noticed"))
```

**Recommended prompts:** Open-ended narrative starters work best. Avoid questions, instructions, or anything requiring factual accuracy.

**Temperature guide:**
- `0.7` — More focused, less creative. Tends to loop or repeat phrases.
- `0.8` — Sweet spot for this model.
- `1.0` — More varied but more incoherent.
- `1.2+` — Mostly nonsense at this model's capability level.

---

## Reproducing Training

### Step 1: Download and Tokenize Data

Run **Cell 12** (C4 preprocessor) for the initial C4 corpus:
- Target: ~50–100 GB of C4 English text
- Output: `tokens_0000.npy`, `tokens_0001.npy`, ... (~10M tokens each, uint16)
- Expected time: 2–6 hours on Colab

Run **Cell 13** for the mixed corpus:
- Cosmopedia: ~20 GB (`cosmopedia_tokens_*.npy`)
- Alpaca: ~100 MB (`alpaca_tokens_*.npy`) — Stanford Alpaca 52K
- Python: ~5 GB (`python_tokens_*.npy`) — CodeAlpaca or The Stack (Python)

All token files are stored on Google Drive at:
```
/content/drive/MyDrive/llm_training/data/
├── tokens_0000.npy ... tokens_XXXX.npy    (C4)
├── cosmopedia/cosmopedia_tokens_*.npy
├── alpaca/alpaca_tokens_*.npy
└── python/python_tokens_*.npy
```

### Step 2: Configure

Edit `Config` in Cell 3 to change hyperparameters. Key fields:

```python
class Config:
    DRIVE_DATA_DIR       = "/content/drive/MyDrive/llm_training/data"
    DRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/llm_training/checkpoints"
    D_MODEL    = 768     # Model width
    N_LAYERS   = 12      # Depth
    N_HEADS    = 12      # Attention heads
    MAX_SEQ_LEN = 1024   # Context window
    BATCH_SIZE = 6       # Per-GPU batch
    GRADIENT_ACCUM_STEPS = 21  # Effective batch = 6×21×1024 = 129,024 tokens
    LEARNING_RATE = 5e-5
    WARMUP_STEPS = 1750
    MAX_STEPS = 25000
```

### Step 3: Train

Run cells in order: 2 → 3 → 5B → 6 → 7 → 8 → 9.

Cell 9 is the entry point:
```python
mixed_dataloader = setup_mixed_dataloader(current_step=0)
trained_model = train_complete()
```

The training loop automatically:
- Detects and resumes from the latest checkpoint
- Persists dataset file position to Drive (survives Colab restarts)
- Generates text samples every 100 steps
- Saves checkpoints every 100 steps, retaining only the last 3

### Step 4: Switch Datasets Mid-Run (what was done here)

To remove C4 from the dataloader state without resetting everything, run **Cell 9b**:
```python
remove_c4_from_state()
```

This removes the `"c4"` key from `dataset_state.json` on Drive, so the next restart will not try to resume C4 file position. Then update the `probs` dict in `setup_mixed_dataloader()` to set `c4: 0.0`.

---

## Log Analysis Tools

### `llm_postprocess.py`

Parses one or more training log files and produces structured outputs.

```bash
# Basic usage (current directory)
python llm_postprocess.py

# Custom paths
python llm_postprocess.py --log_dir ./logs --out_dir ./results

# Override glob pattern
python llm_postprocess.py --pattern "training_*.log"
```

**Outputs:**

| File | Contents |
|------|----------|
| `llm_training_metrics.json` | One JSON object per step: timestamp, step, loss, lr, tokens_per_s, gpu_gb, elapsed_s, checkpoint_saved, checkpoint_removed, interrupted |
| `llm_training_samples.json` | One block per sample event: step, timestamp, list of {prompt, response, timestamp} pairs |
| `llm_training_curves.png` | 4-panel matplotlib figure: loss (raw + smoothed + checkpoint markers), LR schedule, throughput, GPU memory |

**Multi-file support:** If you have multiple log files from different sessions (common with Colab), place them all in the same directory. The script merges them and resolves step-level duplicates by keeping the entry with the most recent timestamp.

---

## Lessons Learned

Distilled from this experiment for anyone attempting something similar:

1. **Plan your dataset before you start.** Switching datasets mid-training without a curriculum is the single most damaging thing you can do. Decide upfront: pretrain on clean web text *or* go straight to the curated mix, not both in sequence without a transition plan.

2. **Chinchilla scaling matters.** For a 124M parameter model, you need ~2.5B tokens of high-quality data for optimal compute efficiency. Splitting those tokens poorly across incompatible domains doesn't count as 2.5B useful tokens.

3. **Tokenizer and data format must be aligned.** If your instruction data uses special tokens, your tokenizer needs to natively support them. Adding ad-hoc `<|user|>` strings to a base GPT-2 tokenizer does nothing.

4. **Validation loss is non-optional.** Without a held-out split, you're flying blind. Even a small 1% split would have shown whether the loss improvements were real generalization or just fitting to training data.

5. **8-bit AdamW works well.** `bitsandbytes.AdamW8bit` saved significant memory with no visible degradation in training stability. The throughput stayed rock-steady at ~8,374 tokens/sec throughout.

6. **The checkpoint/resume system was solid.** `DatasetStateManager` (Drive-backed JSON with file-level bookmarks) survived multiple Colab disconnections without losing significant progress. This was the most reliable part of the whole project.

7. **Story rambling ≠ understanding.** The model learns to mimic the *style* of text far before it learns any semantic content. Fluent-looking output at step 1,600 is the model pattern-matching surface statistics, not reasoning.

---

## Acknowledgements

- Architecture based on the GPT-2 paper: *Language Models are Unsupervised Multitask Learners* (Radford et al., 2019)
- Tokenizer: `tiktoken` by OpenAI (GPT-2 BPE encoding)
- Datasets: AllenAI C4, HuggingFaceTB Cosmopedia, Stanford Alpaca, BigCode The Stack / CodeAlpaca
- Optimizer: `bitsandbytes` 8-bit AdamW by Tim Dettmers

---

*This README was written after the training run completed. All metrics, sample outputs, and observations are based on actual training logs and generated samples from the run.*

### Made by Dheeren
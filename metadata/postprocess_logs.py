"""
llm_postprocess.py — LLM Training Log Parser, JSON Exporter & Plotter

Parses one or more training log files that follow this structure:
  • Step metrics lines    →  llm_training_metrics.json + training curves PNG
  • Sample blocks         →  llm_training_samples.json
  • Checkpoint events     →  included in metrics JSON
  • Multi-file support    →  overlapping steps resolved by latest timestamp

Usage:
    python llm_postprocess.py                        # uses defaults in Config
    python llm_postprocess.py --log_dir ./logs       # override log directory
    python llm_postprocess.py --log_dir ./logs --out_dir ./results
"""

import re
import json
import glob
import argparse
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Configuration — edit these or pass CLI args
# ---------------------------------------------------------------------------

class Config:
    LOG_DIR      = r"."               # directory containing *.log files
    LOG_PATTERN  = "training_*.log"   # glob pattern to match log files
    OUT_DIR      = r"."               # where to write JSON + PNG outputs

    OUT_METRICS_JSON  = "llm_training_metrics.json"
    OUT_SAMPLES_JSON  = "llm_training_samples.json"
    OUT_PLOT_PNG      = "llm_training_curves.png"


# ---------------------------------------------------------------------------
# Regex patterns — compiled once
# ---------------------------------------------------------------------------

# [2025-11-18 07:15:06] Step    610 | Loss: 6.2691 | LR: 2.09e-04 | Tokens/s: 8,362 | GPU: 14.8GB | Time: 158s
RE_STEP = re.compile(
    r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+'
    r'Step\s+(\d+)\s*\|\s*'
    r'Loss:\s*([\d.]+)\s*\|\s*'
    r'LR:\s*([\d.eE+\-]+)\s*\|\s*'
    r'Tokens/s:\s*([\d,]+)\s*\|\s*'
    r'GPU:\s*([\d.]+)GB\s*\|\s*'
    r'Time:\s*(\d+)s'
)

# [2025-11-18 07:15:06]    📊 File #-1 | Seen: 0/237
RE_FILE_SEEN = re.compile(
    r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+'
    r'.*?File\s+#(-?\d+)\s*\|\s*Seen:\s*(\d+)/(\d+)'
)

# [2025-11-18 07:38:15] 📝 SAMPLES  ← start of a sample block
RE_SAMPLE_HEADER = re.compile(
    r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+.*?SAMPLES'
)

# [2025-11-18 07:38:16] The future of AI is → The future of AI is also not only...
RE_SAMPLE_LINE = re.compile(
    r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+(.*?)\s+→\s+(.*)'
)

# ============================================================  (divider)
RE_DIVIDER = re.compile(r'^=+\s*$')

# [2025-11-18 07:38:20] 💾 Checkpoint saved: /path/to/checkpoint_step_700.pt
RE_CKPT_SAVED = re.compile(
    r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+.*?Checkpoint saved:\s*(.+)'
)

# [2025-11-18 07:38:20] 🗑️  Removed old checkpoint: checkpoint_step_428.pt
RE_CKPT_REMOVED = re.compile(
    r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+.*?Removed old checkpoint:\s*(.+)'
)

# ⚠️  Interrupted!
RE_INTERRUPTED = re.compile(r'Interrupted')


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_log_file(file_path: str) -> tuple[list[dict], list[dict]]:
    """
    Parse a single log file.

    Returns
    -------
    metrics : list[dict]
        One entry per Step line, enriched with checkpoint/interruption events.
    samples : list[dict]
        One entry per sample block (contains all prompts + responses).
    """
    metrics: list[dict] = []
    samples: list[dict] = []

    path = Path(file_path)
    print(f"   Parsing: {path.name}")

    with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    # ---- State machine ----
    i = 0
    last_step_entry: dict | None = None   # most recent metrics entry (for attaching events)
    in_sample_block  = False
    sample_block_step: int | None = None
    sample_block_ts:   str | None = None
    current_samples: list[dict] = []
    current_prompt:  str | None = None
    current_ts:      str | None = None
    continuation_lines: list[str] = []   # multi-line response accumulator

    def flush_current_sample():
        """Finalise the in-progress prompt→response pair."""
        nonlocal current_prompt, current_ts, continuation_lines
        if current_prompt is not None:
            response = " ".join(continuation_lines).strip()
            current_samples.append({
                "prompt":    current_prompt,
                "response":  response,
                "timestamp": current_ts,
            })
        current_prompt      = None
        current_ts          = None
        continuation_lines  = []

    def flush_sample_block():
        """Commit the accumulated sample block to the samples list."""
        nonlocal in_sample_block, current_samples, sample_block_step, sample_block_ts
        flush_current_sample()
        if current_samples:
            samples.append({
                "step":      sample_block_step,
                "timestamp": sample_block_ts,
                "samples":   list(current_samples),
            })
        in_sample_block    = False
        current_samples    = []
        sample_block_step  = None
        sample_block_ts    = None

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        # --- Divider: close any open sample block ---
        if RE_DIVIDER.match(stripped):
            if in_sample_block:
                flush_sample_block()
            continue

        # --- Interrupted ---
        if RE_INTERRUPTED.search(stripped):
            if last_step_entry:
                last_step_entry["interrupted"] = True
            continue

        # --- Step metric line ---
        m = RE_STEP.match(line)
        if m:
            ts, step, loss, lr, tokens_s, gpu_gb, elapsed = m.groups()
            entry = {
                "timestamp":    ts,
                "step":         int(step),
                "loss":         float(loss),
                "lr":           float(lr),
                "tokens_per_s": int(tokens_s.replace(",", "")),
                "gpu_gb":       float(gpu_gb),
                "elapsed_s":    int(elapsed),
                "checkpoint_saved":   None,
                "checkpoint_removed": None,
                "interrupted":        False,
            }
            metrics.append(entry)
            last_step_entry = entry
            continue

        # --- Checkpoint saved ---
        m = RE_CKPT_SAVED.match(line)
        if m:
            ts, ckpt_path = m.groups()
            if last_step_entry:
                last_step_entry["checkpoint_saved"] = ckpt_path.strip()
            continue

        # --- Checkpoint removed ---
        m = RE_CKPT_REMOVED.match(line)
        if m:
            ts, ckpt_name = m.groups()
            if last_step_entry:
                last_step_entry["checkpoint_removed"] = ckpt_name.strip()
            continue

        # --- Sample block header ---
        m = RE_SAMPLE_HEADER.match(line)
        if m:
            in_sample_block   = True
            sample_block_ts   = m.group(1)
            sample_block_step = last_step_entry["step"] if last_step_entry else None
            current_samples   = []
            current_prompt    = None
            continuation_lines = []
            continue

        # --- Inside a sample block ---
        if in_sample_block:
            # Try to match a new "prompt → start_of_response" line
            m = RE_SAMPLE_LINE.match(line)
            if m:
                flush_current_sample()          # save previous pair first
                ts, prompt, response_start = m.groups()
                current_prompt     = prompt.strip()
                current_ts         = ts
                continuation_lines = [response_start.strip()]
            else:
                # Continuation line for the current response (multi-line responses)
                # Strip the timestamp prefix, then skip noise lines
                bare = re.sub(r'^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]\s*', '', line).strip()
                skip = (
                    not bare
                    or RE_DIVIDER.match(bare)
                    or RE_FILE_SEEN.search(line)   # 📊 File #N | Seen: N/N
                    or RE_STEP.match(line)          # step line leaked in
                    or bare.startswith('💾')        # checkpoint save line
                    or bare.startswith('🗑')        # checkpoint remove line
                    or bare.startswith('📊')        # file tracking emoji
                )
                if not skip:
                    continuation_lines.append(bare)

    # Finalise anything still open at EOF
    if in_sample_block:
        flush_sample_block()

    print(f"      → {len(metrics)} step entries, {len(samples)} sample blocks")
    return metrics, samples


# ---------------------------------------------------------------------------
# Merging (same overlap strategy as clip_postprocess.py)
# ---------------------------------------------------------------------------

def merge_metrics(all_metrics: list[dict]) -> list[dict]:
    """
    Deduplicate by step; for duplicate steps keep the entry with the
    latest timestamp (i.e. from the most recent log file / run resumption).
    """
    step_dict: dict[int, dict] = {}
    for entry in all_metrics:
        step = entry["step"]
        if step not in step_dict:
            step_dict[step] = entry
        else:
            existing_ts = datetime.fromisoformat(step_dict[step]["timestamp"])
            new_ts      = datetime.fromisoformat(entry["timestamp"])
            if new_ts > existing_ts:
                step_dict[step] = entry

    return sorted(step_dict.values(), key=lambda x: x["step"])


def merge_samples(all_samples: list[dict]) -> list[dict]:
    """
    Deduplicate sample blocks by step; keep latest timestamp.
    """
    step_dict: dict[int | None, dict] = {}
    for block in all_samples:
        step = block["step"]
        if step not in step_dict:
            step_dict[step] = block
        else:
            if block["timestamp"] and step_dict[step]["timestamp"]:
                if block["timestamp"] > step_dict[step]["timestamp"]:
                    step_dict[step] = block

    return sorted(
        step_dict.values(),
        key=lambda x: (x["step"] is None, x["step"] or 0)
    )


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def save_json(data, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Saved {len(data)} entries → {output_path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_metrics(metrics: list[dict], output_path: str) -> None:
    steps     = [m["step"]         for m in metrics]
    losses    = [m["loss"]         for m in metrics]
    lrs       = [m["lr"]           for m in metrics]
    tokens_ps = [m["tokens_per_s"] for m in metrics]
    gpu_gbs   = [m["gpu_gb"]       for m in metrics]

    # Mark checkpoint steps
    ckpt_steps  = [m["step"] for m in metrics if m.get("checkpoint_saved")]
    ckpt_losses = [m["loss"] for m in metrics if m.get("checkpoint_saved")]

    # Mark sample steps (from metrics that align with sample blocks)
    sample_step_set = set()
    for m in metrics:
        if m.get("checkpoint_saved"):
            sample_step_set.add(m["step"])  # samples are emitted just before checkpoints

    # Smoothed loss (rolling average)
    window = max(1, len(losses) // 20)
    smooth_losses = np.convolve(losses, np.ones(window) / window, mode="valid")
    smooth_steps  = steps[window - 1:]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("LLM Training Curves", fontsize=15, fontweight="bold", y=1.01)

    # ---- Panel 1: Loss ----
    ax = axes[0, 0]
    ax.plot(steps, losses, color="#aec6e8", linewidth=0.8, alpha=0.6, label="Raw loss")
    ax.plot(smooth_steps, smooth_losses, color="#1f77b4", linewidth=2.0, label=f"Smoothed (w={window})")
    if ckpt_steps:
        ax.scatter(ckpt_steps, ckpt_losses, color="#d62728", s=40, zorder=5,
                   label="Checkpoint", marker="v")
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Training Loss", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)

    # ---- Panel 2: Learning Rate ----
    ax = axes[0, 1]
    ax.plot(steps, lrs, color="#ff7f0e", linewidth=1.5)
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Learning Rate", fontsize=11)
    ax.set_title("Learning Rate Schedule", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2e"))
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)

    # ---- Panel 3: Tokens/s throughput ----
    ax = axes[1, 0]
    ax.plot(steps, tokens_ps, color="#2ca02c", linewidth=1.0, alpha=0.7)
    ax.axhline(np.mean(tokens_ps), color="#2ca02c", linewidth=1.5,
               linestyle="--", label=f"Mean: {np.mean(tokens_ps):,.0f}")
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Tokens / second", fontsize=11)
    ax.set_title("Training Throughput", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)

    # ---- Panel 4: GPU memory ----
    ax = axes[1, 1]
    ax.plot(steps, gpu_gbs, color="#9467bd", linewidth=1.2)
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("GPU Memory (GB)", fontsize=11)
    ax.set_title("GPU Memory Usage", fontsize=12)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved plot → {output_path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(metrics: list[dict], samples: list[dict]) -> None:
    if not metrics:
        print("   ❌ No metrics to summarise.")
        return

    losses = [m["loss"] for m in metrics]
    tps    = [m["tokens_per_s"] for m in metrics]

    best_loss_entry = min(metrics, key=lambda m: m["loss"])
    ckpts_saved     = [m["checkpoint_saved"]   for m in metrics if m.get("checkpoint_saved")]
    interrupted     = any(m.get("interrupted") for m in metrics)

    print("\n📉 Training Summary")
    print(f"   Step range       : {metrics[0]['step']} → {metrics[-1]['step']}")
    print(f"   Total steps      : {len(metrics)}")
    print(f"   Initial loss     : {losses[0]:.4f}")
    print(f"   Final loss       : {losses[-1]:.4f}")
    print(f"   Best loss        : {best_loss_entry['loss']:.4f}  (step {best_loss_entry['step']})")
    print(f"   Loss improvement : {losses[0] - losses[-1]:+.4f}")
    print(f"   Avg tokens/s     : {np.mean(tps):,.0f}")
    print(f"   Checkpoints saved: {len(ckpts_saved)}")
    if ckpts_saved:
        print(f"   Last checkpoint  : {Path(ckpts_saved[-1]).name}")
    print(f"   Run interrupted  : {'Yes ⚠️' if interrupted else 'No'}")
    print(f"\n📝 Sample Blocks")
    print(f"   Total blocks     : {len(samples)}")
    if samples:
        prompts = list({s['prompt'] for block in samples for s in block['samples']})
        print(f"   Unique prompts   : {len(prompts)}")
        for p in prompts:
            print(f"      • \"{p}\"")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM training log postprocessor")
    parser.add_argument("--log_dir",  default=Config.LOG_DIR,  help="Directory containing log files")
    parser.add_argument("--pattern",  default=Config.LOG_PATTERN, help="Glob pattern for log files")
    parser.add_argument("--out_dir",  default=Config.OUT_DIR,  help="Output directory for JSON + PNG")
    args = parser.parse_args()

    print("=" * 70)
    print("  LLM TRAINING LOG PARSER & PLOTTER")
    print("=" * 70)

    log_dir  = Path(args.log_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover log files
    log_files = sorted(glob.glob(str(log_dir / args.pattern)))
    if not log_files:
        print(f"\n❌ No log files found matching '{args.pattern}' in {log_dir.absolute()}")
        return

    print(f"\n🔍 Found {len(log_files)} log file(s):")
    for f in log_files:
        print(f"   - {Path(f).name}")

    # Parse all files
    print("\n📊 Parsing...")
    all_metrics: list[dict] = []
    all_samples: list[dict] = []

    for log_file in log_files:
        m, s = parse_log_file(log_file)
        all_metrics.extend(m)
        all_samples.extend(s)

    print(f"\n   Raw entries  : {len(all_metrics)} metrics, {len(all_samples)} sample blocks")

    # Deduplicate & merge
    print("\n🔄 Merging & deduplicating...")
    merged_metrics = merge_metrics(all_metrics)
    merged_samples = merge_samples(all_samples)
    print(f"   After dedup  : {len(merged_metrics)} metrics, {len(merged_samples)} sample blocks")

    if not merged_metrics:
        print("\n❌ No metrics found — check your log format or glob pattern.")
        return

    # Save JSON
    print("\n💾 Saving JSON...")
    metrics_path = out_dir / Config.OUT_METRICS_JSON
    samples_path = out_dir / Config.OUT_SAMPLES_JSON
    save_json(merged_metrics, str(metrics_path))
    save_json(merged_samples, str(samples_path))

    # Plot
    print("\n📈 Generating plots...")
    plot_path = out_dir / Config.OUT_PLOT_PNG
    plot_metrics(merged_metrics, str(plot_path))

    # Summary
    print_summary(merged_metrics, merged_samples)

    print("\n" + "=" * 70)
    print("  ✅ ALL DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
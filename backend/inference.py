"""
inference.py — Generation engine for the Noir Whisper backend.

Two generation methods:
  • generate()        — blocking, returns the full text at once.
  • generate_stream() — pushes each decoded token into an asyncio.Queue
                        as it is produced, enabling SSE streaming.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass

import tiktoken
import torch

from config import CHECKPOINT_PATH, ModelConfig
from model import GPTModel

logger = logging.getLogger("noir_whisper.inference")


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

@dataclass
class GenerationRequest:
    prompt:             str
    max_tokens:         int   = 512
    temperature:        float = 0.7
    top_p:              float = 0.9
    top_k:              int   = 50
    repetition_penalty: float = 1.1
    range_epsilon:      float = 0.1


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class InferenceEngine:
    """
    Singleton-style inference engine.  Create once, call generate() or
    generate_stream() many times.

    A threading.Lock prevents concurrent generation requests from corrupting
    the model's RangeFlow anchor state.
    """

    def __init__(self) -> None:
        self._lock      = threading.Lock()
        self._device    = self._pick_device()
        self._model: GPTModel | None = None
        self._tokenizer = tiktoken.get_encoding("gpt2")
        logger.info("InferenceEngine created — device: %s", self._device.upper())

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_model(self) -> None:
        """Build the model and load checkpoint weights.  Called once at startup."""
        logger.info("=" * 60)
        logger.info("🚀  RangeFlow Inference Engine — loading model")
        logger.info("    Device      : %s", self._device.upper())
        logger.info("    Checkpoint  : %s", CHECKPOINT_PATH)
        logger.info("    Architecture: vocab=%d  d_model=%d  layers=%d  heads=%d",
                    ModelConfig.VOCAB_SIZE, ModelConfig.D_MODEL,
                    ModelConfig.N_LAYERS,   ModelConfig.N_HEADS)
        logger.info("=" * 60)

        import os
        if not os.path.exists(CHECKPOINT_PATH):
            raise RuntimeError(
                f"Checkpoint not found at '{CHECKPOINT_PATH}'. "
                "Set the CHECKPOINT_PATH environment variable or place the file "
                "at the default location."
            )

        # Build model graph
        t0    = time.perf_counter()
        model = GPTModel(ModelConfig).to(self._device)
        logger.info("Model graph built in %.2f s — %.1f M parameters",
                    time.perf_counter() - t0, model.parameter_count() / 1e6)

        # Load weights
        t1 = time.perf_counter()
        try:
            checkpoint = torch.load(
                CHECKPOINT_PATH,
                map_location=self._device,
                weights_only=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to read checkpoint: {exc}") from exc

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if missing:
            logger.warning("Missing keys in checkpoint (%d): %s …", len(missing), missing[:5])
        if unexpected:
            logger.warning("Unexpected keys in checkpoint (%d): %s …", len(unexpected), unexpected[:5])

        model.eval()
        self._model = model

        logger.info("✅  Checkpoint loaded in %.2f s", time.perf_counter() - t1)
        for key in ("step", "loss", "epoch"):
            if key in checkpoint:
                logger.info("    Checkpoint %s: %s", key, checkpoint[key])
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, req: GenerationRequest) -> dict:
        """
        Blocking generation — runs the full RangeFlow pipeline and returns
        a dict with keys: response, tokens_generated, elapsed_ms, device.
        """
        if self._model is None:
            raise RuntimeError("Model is not loaded.  Call load_model() first.")

        with self._lock:
            return self._run_generation(req, stream_queue=None, loop=None)

    def generate_stream(
        self,
        req: GenerationRequest,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """
        Streaming generation — same pipeline as generate() but pushes each
        decoded token into *queue* as it is produced so the FastAPI SSE
        endpoint can relay it to the browser immediately.

        Queue message format:  ("token", str) | ("done", int) | ("error", str)

        This method is called from a ThreadPoolExecutor (it blocks).
        """
        if self._model is None:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", "Model not loaded"))
            return

        with self._lock:
            self._run_generation(req, stream_queue=queue, loop=loop)

    # ------------------------------------------------------------------
    # Core generation loop (shared by both public methods)
    # ------------------------------------------------------------------

    def _run_generation(
        self,
        req: GenerationRequest,
        stream_queue: asyncio.Queue | None,
        loop: asyncio.AbstractEventLoop | None,
    ) -> dict:
        """
        Two-phase RangeFlow generation.

        If stream_queue is provided, each token is pushed into the queue
        via loop.call_soon_threadsafe() as it is produced.
        If stream_queue is None, tokens are accumulated and returned as a dict.
        """
        streaming = stream_queue is not None

        logger.info("-" * 50)
        logger.info("📥  Generation request  [%s]", "stream" if streaming else "full")
        logger.info("    Prompt     : %r  (%d chars)", req.prompt[:80], len(req.prompt))
        logger.info("    max_tokens : %d", req.max_tokens)
        logger.info("    temperature: %.3f", req.temperature)
        logger.info("    top_k      : %d", req.top_k)
        logger.info("    top_p      : %.3f", req.top_p)
        logger.info("    rep_penalty: %.3f", req.repetition_penalty)
        logger.info("    range_ε    : %.3f", req.range_epsilon)
        logger.info("-" * 50)

        t_start = time.perf_counter()
        model   = self._model

        # Encode prompt
        prompt_ids = self._tokenizer.encode(req.prompt)
        if not prompt_ids:
            logger.warning("Empty token sequence — returning empty response")
            if streaming:
                loop.call_soon_threadsafe(stream_queue.put_nowait, ("done", 0))
            return {"response": "", "tokens_generated": 0, "elapsed_ms": 0.0, "device": self._device}

        tokens = torch.tensor(prompt_ids, dtype=torch.long, device=self._device).unsqueeze(0)
        logger.info("    Prompt tokens: %d", tokens.size(1))

        generated_ids: list[int] = []

        with torch.no_grad():
            # ----------------------------------------------------------
            # Phase A — CAPTURE
            # Run one forward pass over the full prompt to record the K/V
            # bounding boxes (the "anchor") in every attention layer.
            # ----------------------------------------------------------
            logger.debug("Phase A: capture — establishing RangeFlow anchor")
            model.clear_anchors()
            model.set_epsilon(req.range_epsilon)
            model.set_range_mode("capture")
            _ = model(tokens)
            logger.debug("Phase A complete — anchor captured across %d layers", ModelConfig.N_LAYERS)

            # ----------------------------------------------------------
            # Phase B — GUARD
            # Autoregressive generation with K/V constrained to the anchor.
            # ----------------------------------------------------------
            logger.debug("Phase B: guard — autoregressive generation begins")
            model.set_range_mode("guard")

            eot_token = getattr(self._tokenizer, "eot_token", None)

            for step in range(req.max_tokens):
                if tokens.size(1) >= ModelConfig.MAX_SEQ_LEN:
                    logger.info("    [step %d] MAX_SEQ_LEN reached — stopping", step)
                    break

                logits = model(tokens)[:, -1, :]

                # Sampling stack
                logits = self._apply_repetition_penalty(logits, tokens, req.repetition_penalty)
                if req.temperature != 1.0:
                    logits = logits / max(req.temperature, 1e-8)
                if req.top_k > 0:
                    logits = self._top_k_filter(logits, req.top_k)
                if req.top_p < 1.0:
                    logits = self._top_p_filter(logits, req.top_p)

                probs      = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                token_id   = next_token.item()

                if eot_token is not None and token_id == eot_token:
                    logger.debug("    [step %d] EOT token — stopping", step)
                    break

                generated_ids.append(token_id)
                tokens = torch.cat([tokens, next_token], dim=1)

                # Decode and emit the token immediately for streaming
                if streaming:
                    word = self._tokenizer.decode([token_id])
                    loop.call_soon_threadsafe(stream_queue.put_nowait, ("token", word))

                if (step + 1) % 50 == 0:
                    logger.debug("    [step %d] %d tokens so far", step + 1, len(generated_ids))

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.info("✅  Generation complete — %d tokens in %.1f ms  (%.1f tok/s)",
                    len(generated_ids), elapsed_ms,
                    len(generated_ids) / max(elapsed_ms / 1000, 1e-6))

        model.set_range_mode("standard")

        if streaming:
            loop.call_soon_threadsafe(stream_queue.put_nowait, ("done", len(generated_ids)))
            # Return value is ignored in streaming mode
            return {}

        response_text = self._tokenizer.decode(generated_ids)
        logger.info("    Response preview: %r …", response_text[:80])
        return {
            "response":         response_text,
            "tokens_generated": len(generated_ids),
            "elapsed_ms":       round(elapsed_ms, 2),
            "device":           self._device,
        }

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        if penalty == 1.0:
            return logits
        scores = torch.gather(logits, dim=1, index=token_ids)
        scores = torch.where(scores < 0, scores * penalty, scores / penalty)
        return logits.scatter_(1, token_ids, scores)

    @staticmethod
    def _top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        k         = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, k).values[:, -1, None]
        return logits.masked_fill(logits < threshold, float("-inf"))

    @staticmethod
    def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove_mask          = cum_probs > top_p
        remove_mask[..., 1:] = remove_mask[..., :-1].clone()
        remove_mask[..., 0]  = False
        sorted_logits[remove_mask] = float("-inf")
        return sorted_logits.scatter(1, sorted_idx, sorted_logits)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return self._device


# Module-level singleton
engine = InferenceEngine()
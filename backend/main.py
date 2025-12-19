import torch
import torch.nn as nn
import tiktoken
import time
import uvicorn
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

torch.set_float32_matmul_precision('high')
# --- 1. CONFIGURATION ---
class Config:
    VOCAB_SIZE = 50257
    D_MODEL = 768
    N_LAYERS = 12
    N_HEADS = 12
    D_FF = 3072
    MAX_SEQ_LEN = 1024
    DROPOUT = 0.0 # No dropout for inference

# --- 2. MODEL ARCHITECTURE ---
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        qkv = self.qkv_proj(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().reshape(batch_size, seq_len, d_model)
        return self.out_proj(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask):
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embed = nn.Embedding(config.VOCAB_SIZE, config.D_MODEL)
        self.pos_embed = nn.Embedding(config.MAX_SEQ_LEN, config.D_MODEL)
        self.blocks = nn.ModuleList([TransformerBlock(config.D_MODEL, config.N_HEADS, config.D_FF, config.DROPOUT) for _ in range(config.N_LAYERS)])
        self.ln_final = nn.LayerNorm(config.D_MODEL)
        self.head = nn.Linear(config.D_MODEL, config.VOCAB_SIZE, bias=False)
        self.token_embed.weight = self.head.weight

    def forward(self, x):
        batch_size, seq_len = x.shape
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        x = self.token_embed(x) + self.pos_embed(positions)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        for block in self.blocks: x = block(x, mask)
        return self.head(self.ln_final(x))

# --- 3. SERVER SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("AI_Server")

app = FastAPI()

# Allow React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8081"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model (Global)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "model/checkpoint_step_7100.pt"

print(f"🔌 Loading model on {DEVICE}...")
model = GPTModel(Config).to(DEVICE)
try:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

tokenizer = tiktoken.get_encoding("gpt2")

# Stop sequences to prevent repetition and over-generation
STOP_SEQUENCES = ["\nUser:", "\n\nUser:", "User:", "\nAI:", "\n\nAI:"]
# Encode stop sequences for efficient checking
STOP_TOKEN_SEQUENCES = []
for stop_seq in STOP_SEQUENCES:
    try:
        encoded = tokenizer.encode(stop_seq)
        if encoded:
            STOP_TOKEN_SEQUENCES.append(encoded)
    except:
        pass

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

def stream_generator(prompt, max_new_tokens, temperature):
    """Generates tokens one by one for streaming"""
    
    # Metrics Tracking
    start_time = time.time()
    first_token_time = 0
    token_count = 0
    
    tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=DEVICE).unsqueeze(0)
    generated_text = ""
    
    with torch.no_grad():
        for i in range(max_new_tokens):
            if tokens.size(1) >= Config.MAX_SEQ_LEN: break
            
            # Forward pass
            logits = model(tokens)[:, -1, :] / temperature
            
            # Sampling (Nucleus + TopK for speed/quality)
            v, _ = torch.topk(logits, min(50, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Decode token
            word = tokenizer.decode([next_token.item()])
            generated_text += word
            
            # Check for stop sequences and repetition patterns
            should_stop = False
            
            if next_token.item() == tokenizer.eot_token:
                should_stop = True
            else:
                # Check if any stop sequence appears at the end of generated text
                for stop_seq in STOP_SEQUENCES:
                    if generated_text.endswith(stop_seq):
                        should_stop = True
                        # Remove the stop sequence from the word we yield
                        if word.endswith(stop_seq):
                            word = word[:-len(stop_seq)]
                        break
                    # Also check if stop sequence is in the recent text (last 50 chars)
                    elif len(generated_text) > 50 and stop_seq in generated_text[-50:]:
                        # Check if it's near the end (within last 20 chars)
                        idx = generated_text.rfind(stop_seq)
                        if idx >= len(generated_text) - 20:
                            should_stop = True
                            break
                
                # Check for repetition patterns (if same phrase repeats 2+ times consecutively)
                if not should_stop and len(generated_text) > 50:
                    words = generated_text.split()
                    if len(words) > 15:
                        # Check for repeating phrases in the last portion
                        for window_size in [3, 5, 7]:
                            if len(words) >= window_size * 2:
                                recent_words = words[-window_size * 2:]
                                first_phrase = " ".join(recent_words[:window_size])
                                second_phrase = " ".join(recent_words[window_size:])
                                # Only stop if exact repetition (case-insensitive)
                                if first_phrase.lower() == second_phrase.lower() and len(first_phrase) > 10:
                                    should_stop = True
                                    break
                            if should_stop:
                                break
            
            # Update metrics
            if token_count == 0:
                first_token_time = time.time() - start_time
            token_count += 1
            
            # Yield word to client (stop sequence removed if found)
            yield word
            
            if should_stop:
                break
            
            tokens = torch.cat([tokens, next_token], dim=1)
                
    # Final Metrics Log
    total_time = time.time() - start_time
    tps = token_count / total_time
    logger.info(f"⚡ Request Complete | TTFT: {first_token_time*1000:.2f}ms | Speed: {tps:.2f} T/s | Total: {token_count} tokens")

@app.post("/generate")
async def generate_stream(request: PromptRequest):
    return StreamingResponse(
        stream_generator(request.prompt, request.max_tokens, request.temperature), 
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
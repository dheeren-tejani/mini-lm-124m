import torch
import torch.nn as nn
import tiktoken
import time
import sys
import os

# --- 1. CONFIGURATION ---
class Config:
    VOCAB_SIZE = 50257
    D_MODEL = 768
    N_LAYERS = 12
    N_HEADS = 12
    D_FF = 3072
    MAX_SEQ_LEN = 1024
    DROPOUT = 0.0
    
    # --- RANGEFLOW HYPERPARAMETERS ---
    # 0.05 = Strict adherence to topic
    # 0.20 = Looser adherence
    RANGE_EPSILON = 0.2  

CHECKPOINT_PATH = "model/checkpoint_step_18300.pt"

# --- 2. RANGE-AWARE ARCHITECTURE ---
class RangeAwareAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # RangeFlow State
        self.mode = "standard" 
        self.epsilon = Config.RANGE_EPSILON
        
        # Anchor Buffers
        self.register_buffer("anchor_k_min", None)
        self.register_buffer("anchor_k_max", None)
        self.register_buffer("anchor_v_min", None)
        self.register_buffer("anchor_v_max", None)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        qkv = self.qkv_proj(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # --- RANGEFLOW LOGIC ---
        if self.mode == 'capture':
            # Calculate Prompt Bounding Box
            self.anchor_k_min = k.min(dim=2, keepdim=True)[0].detach()
            self.anchor_k_max = k.max(dim=2, keepdim=True)[0].detach()
            self.anchor_v_min = v.min(dim=2, keepdim=True)[0].detach()
            self.anchor_v_max = v.max(dim=2, keepdim=True)[0].detach()
            
        elif self.mode == 'guard' and self.anchor_k_min is not None:
            # Iron Dome Logic: Intersect current token interval with Anchor
            k_min_curr, k_max_curr = k - self.epsilon, k + self.epsilon
            v_min_curr, v_max_curr = v - self.epsilon, v + self.epsilon
            
            # Intersection
            valid_k_min = torch.max(k_min_curr, self.anchor_k_min)
            valid_k_max = torch.min(k_max_curr, self.anchor_k_max)
            valid_v_min = torch.max(v_min_curr, self.anchor_v_min)
            valid_v_max = torch.min(v_max_curr, self.anchor_v_max)
            
            # Fix empty intervals
            valid_k_min = torch.min(valid_k_min, valid_k_max) 
            valid_v_min = torch.min(valid_v_min, valid_v_max)

            # Clamp
            k = torch.max(valid_k_min, torch.min(k, valid_k_max))
            v = torch.max(valid_v_min, torch.min(v, valid_v_max))

        # Standard Attention
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
        self.attn = RangeAwareAttention(d_model, n_heads, dropout)
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
    
    def set_range_mode(self, mode: str):
        for block in self.blocks:
            block.attn.mode = mode

# --- 3. INITIALIZATION ---
DEVICE = "cpu"

print(f"🚀 RangeFlow Inference Engine Initializing on {DEVICE.upper()}...")
model = GPTModel(Config).to(DEVICE)

if os.path.exists(CHECKPOINT_PATH):
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        print(f"✅ Loaded checkpoint: {CHECKPOINT_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
else:
    print(f"⚠️ Checkpoint not found at {CHECKPOINT_PATH}")
    sys.exit(1)

tokenizer = tiktoken.get_encoding("gpt2")

# --- 4. TERMINAL INTERACTION LOOP ---
def generate(prompt, max_tokens=1024, temperature=1.0, top_k=50, top_p=0.9, repetition_penalty=1.8):
    tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=DEVICE).unsqueeze(0)
    
    print(f"\n🤖 Generating (Temp={temperature}, TopK={top_k}, TopP={top_p}, RepPen={repetition_penalty})...")
    print(f"\033[96m{prompt}\033[0m", end="", flush=True)  # Print prompt in cyan
    
    with torch.no_grad():
        # PHASE A: CAPTURE (Establish the RangeFlow anchor)
        model.set_range_mode("capture")
        _ = model(tokens)
        
        # PHASE B: GUARD (Generate with constraints)
        model.set_range_mode("guard")
        
        for _ in range(max_tokens):
            if tokens.size(1) >= Config.MAX_SEQ_LEN: break
            
            # Get logits for the last token
            logits = model(tokens)[:, -1, :] 

            # --- 1. Repetition Penalty ---
            # Penalizes tokens that have already been generated to reduce loops
            if repetition_penalty != 1.0:
                score = torch.gather(logits, 1, tokens)
                # If score < 0 push it further down, if > 0 push it towards 0
                score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                logits.scatter_(1, tokens, score)

            # --- 2. Temperature Scaling ---
            logits = logits / temperature

            # --- 3. Top-K Sampling ---
            # Filters out all tokens except the K most likely ones
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # --- 4. Top-P (Nucleus) Sampling ---
            # Dynamically selects the smallest set of tokens whose cumulative probability exceeds P
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')

            # --- 5. Sampling ---
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            word = tokenizer.decode([next_token.item()])
            print(word, end="", flush=True)
            
            # Stop if End of Text token is generated (Tiktoken specific check)
            try:
                if hasattr(tokenizer, 'eot_token') and next_token.item() == tokenizer.eot_token:
                    break
            except:
                pass # Fail silently if eot_token isn't defined
                
            tokens = torch.cat([tokens, next_token], dim=1)
            
    print("\n" + "="*50)
    model.set_range_mode("standard")

if __name__ == "__main__":
    print("\n--- RangeFlow 124M Terminal Interface ---")
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            generate(user_input)
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
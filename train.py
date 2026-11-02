"""
train.py — a tiny GPT trained from scratch on Armenian text.

Idea
----
A decoder-only transformer learns one task: given the previous characters,
predict the next one. Everything else (words, grammar, style) emerges from that.

Architecture (≈10.7M parameters with the defaults):
    token embedding + position embedding
    → 6 × [LayerNorm → causal self-attention → residual
           LayerNorm → MLP (4× wider, GELU) → residual]
    → LayerNorm → linear head (weights tied to the token embedding)

Besides the model, this script records everything the video needs:
    out/loss.csv              train loss every 50 steps, val loss at every eval
    out/samples/step_XXXXX.txt  generated text at fixed steps (same prompt + seed)
    out/attention.npz         attention weights for one example sentence
    out/ckpt.pt               final model weights + config
    out/run.json              timings and settings (to quote in the video)

Complexity per training step: O(B · T · (L·d² + L·T·d)) — the d² part is the
linear layers, the T·d part is attention. With T=256 and d=384 the linear
layers dominate, so cost is roughly 6 · params · tokens_per_step FLOPs.

Usage:
    python train.py                      # defaults tuned for an 8 GB RTX 4060
    python train.py --compile            # ~1.5–2× faster on Linux, slower first step
"""

import argparse                         # command-line flags
import csv                              # loss log for the loss-curve animation
import json                             # meta.json / run.json
import math                             # sqrt for attention scaling, cos for LR schedule
import os                               # paths
import time                             # wall-clock timing for "it trained in N minutes"
from dataclasses import dataclass, asdict

import numpy as np                      # memory-mapped dataset
import torch
import torch.nn as nn
from torch.nn import functional as F


# ===========================================================================
# Model
# ===========================================================================
@dataclass
class GPTConfig:
    vocab_size: int                     # number of distinct characters (from meta.json)
    block_size: int = 256               # context length in characters
    n_layer: int = 6                    # transformer blocks
    n_head: int = 6                     # attention heads per block
    n_embd: int = 384                   # width of every vector in the model
    dropout: float = 0.2                # regularisation: small data → overfitting risk


class CausalSelfAttention(nn.Module):
    """Every position looks at earlier positions and mixes their information."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must split evenly across heads"
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)  # one matmul makes Q, K and V
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)     # merge heads back together
        self.attn_dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, return_attn: bool = False):
        B, T, C = x.shape                                              # batch, time, channels
        q, k, v = self.qkv(x).split(C, dim=2)                          # each (B, T, C)
        hs = C // self.n_head                                          # head size
        # reshape to (B, heads, T, head_size) so every head attends independently
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        if return_attn:
            # Explicit version: slower, but returns the attention matrix for visualisation.
            att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)            # similarity of every pair (B,h,T,T)
            future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
            att = att.masked_fill(future, float("-inf"))               # causal: no peeking ahead
            att = F.softmax(att, dim=-1)                               # each row sums to 1
            y = att @ v                                                # weighted mix of values
        else:
            # Fused kernel (FlashAttention when available): same maths, much faster.
            att = None
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.attn_dropout if self.training else 0.0)

        y = y.transpose(1, 2).contiguous().view(B, T, C)               # concatenate heads
        return self.resid_drop(self.proj(y)), att


class MLP(nn.Module):
    """Per-position feed-forward network: where most of the 'knowledge' is stored."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)                # expand 4×
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)              # project back
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-LayerNorm transformer block: communicate (attention), then compute (MLP)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, return_attn: bool = False):
        a, att = self.attn(self.ln1(x), return_attn)                   # residual connection #1
        x = x + a
        x = x + self.mlp(self.ln2(x))                                  # residual connection #2
        return x, att


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)        # character id → vector
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)        # position → vector
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)  # vector → next-char scores
        self.head.weight = self.tok_emb.weight                         # weight tying: fewer params, better results
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)              # GPT-2 style init
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, return_attn: bool = False):
        B, T = idx.shape
        assert T <= self.cfg.block_size, "sequence longer than the context window"
        pos = torch.arange(T, device=idx.device)                       # 0, 1, …, T-1
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))           # what + where
        attns = []
        for block in self.blocks:
            x, att = block(x, return_attn)
            if return_attn:
                attns.append(att)
        logits = self.head(self.ln_f(x))                               # (B, T, vocab)
        loss = None
        if targets is not None:                                        # cross-entropy on every position
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, attns

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8,
                 top_k: int | None = 20, generator: torch.Generator | None = None):
        """Autoregressive sampling: predict one character, append it, repeat."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]                   # crop to context window
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :].float() / temperature            # next-char scores only; fp32 for safe sampling
            if top_k is not None:                                      # keep the k most likely characters
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator)     # sample (seeded → comparable snapshots)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


# ===========================================================================
# Training
# ===========================================================================
def main() -> None:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")                          # Armenian prints safely even in a non-UTF-8 terminal/log
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="out")
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)                    # small models tolerate a high LR
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=50)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--snapshots", default="0,50,100,200,300,500,1000,2000,3500,5000",
                   help="steps at which to save a text sample (the 'evolution' shots)")
    p.add_argument("--prompt", default="Հայաստանը")                   # every snapshot starts from this
    p.add_argument("--sample-len", type=int, default=400)
    p.add_argument("--attn-sentence", default="Երևանը Հայաստանի մայրաքաղաքն է։")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(os.path.join(args.out_dir, "samples"), exist_ok=True)
    device = args.device

    # --- data ---------------------------------------------------------------
    with open(os.path.join(args.data_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    itos = meta["itos"]                                               # id → char
    stoi = {ch: i for i, ch in enumerate(itos)}                       # char → id
    encode = lambda s: [stoi[c] for c in s if c in stoi]              # unknown chars are dropped
    decode = lambda ids: "".join(itos[i] for i in ids)

    # memmap: the dataset stays on disk; only the sampled windows are read
    train_data = np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")

    def get_batch(split: str):
        data = train_data if split == "train" else val_data
        ix = np.random.randint(0, len(data) - args.block_size - 1, size=args.batch_size)  # random windows
        x = torch.from_numpy(np.stack([data[i:i + args.block_size] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([data[i + 1:i + 1 + args.block_size] for i in ix]).astype(np.int64))
        if device.startswith("cuda"):                                 # async copy to the GPU
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)

    # --- model --------------------------------------------------------------
    cfg = GPTConfig(vocab_size=meta["vocab_size"], block_size=args.block_size, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
    model = GPT(cfg).to(device)
    n_params = model.num_params()
    print(f"model: {n_params/1e6:.2f}M parameters, device: {device}")

    # bf16 autocast on GPUs that support it (RTX 40xx do): ~2× faster, no loss scaling needed
    use_bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16
                else torch.autocast(device_type="cpu", enabled=False))

    # weight decay only on matrices, not on biases / LayerNorm / embeddings' 1-D params
    decay = [p_ for p_ in model.parameters() if p_.dim() >= 2]
    no_decay = [p_ for p_ in model.parameters() if p_.dim() < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                                   {"params": no_decay, "weight_decay": 0.0}],
                                  lr=args.lr, betas=(0.9, 0.99))

    train_model = torch.compile(model) if args.compile else model     # compiled view shares weights

    def lr_at(step: int) -> float:
        """Linear warmup, then cosine decay to min_lr."""
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.max_steps - args.warmup)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * min(1.0, progress)))

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(args.eval_batches)
            for i in range(args.eval_batches):
                x, y = get_batch(split)
                with autocast:
                    _, loss, _ = model(x, y)
                losses[i] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    snapshot_steps = {int(s) for s in args.snapshots.split(",") if s.strip()}
    prompt_ids = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)

    def save_sample(step: int):
        model.eval()
        g = torch.Generator(device=device).manual_seed(args.seed)     # same randomness every snapshot
        with autocast:
            out = model.generate(prompt_ids, args.sample_len, generator=g)
        text = decode(out[0].tolist())
        with open(os.path.join(args.out_dir, "samples", f"step_{step:05d}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        print(f"--- sample @ step {step} ---\n{text[:200]}\n")
        model.train()

    # --- loop ---------------------------------------------------------------
    log_f = open(os.path.join(args.out_dir, "loss.csv"), "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["step", "train_loss", "val_loss", "lr", "elapsed_s"])

    t0 = time.time()
    best_val = float("inf")
    for step in range(args.max_steps + 1):
        if step in snapshot_steps:                                     # "evolution" text for the video
            save_sample(step)

        if step % args.eval_every == 0 or step == args.max_steps:      # proper train/val estimate
            losses = estimate_loss()
            elapsed = time.time() - t0
            log.writerow([step, f"{losses['train']:.4f}", f"{losses['val']:.4f}", f"{lr_at(step):.2e}", f"{elapsed:.1f}"])
            log_f.flush()
            print(f"step {step:5d} | train {losses['train']:.3f} | val {losses['val']:.3f} | {elapsed/60:.1f} min")
            best_val = min(best_val, losses["val"])

        if step == args.max_steps:
            break

        for group in optimizer.param_groups:                           # apply the LR schedule
            group["lr"] = lr_at(step)

        x, y = get_batch("train")
        with autocast:
            _, loss, _ = train_model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)        # guard against loss spikes
        optimizer.step()

        if step % args.log_every == 0 and step % args.eval_every != 0: # cheap per-batch loss in between
            log.writerow([step, f"{loss.item():.4f}", "", f"{lr_at(step):.2e}", f"{time.time()-t0:.1f}"])

    train_seconds = time.time() - t0
    log_f.close()

    # --- attention weights for the animation ---------------------------------
    model.eval()
    sent_ids = encode(args.attn_sentence)[: args.block_size]
    with torch.no_grad():
        _, _, attns = model(torch.tensor([sent_ids], device=device), return_attn=True)
    att = torch.stack([a[0] for a in attns]).float().cpu().numpy()     # (layers, heads, T, T)
    np.savez_compressed(os.path.join(args.out_dir, "attention.npz"),
                        attention=att, chars=np.array(list(decode(sent_ids))))

    # --- checkpoint + run summary ------------------------------------------------
    torch.save({"model": model.state_dict(), "config": asdict(cfg), "itos": itos},
               os.path.join(args.out_dir, "ckpt.pt"))
    run = {
        "parameters": n_params,
        "train_minutes": round(train_seconds / 60, 1),
        "steps": args.max_steps,
        "tokens_seen": args.max_steps * args.batch_size * args.block_size,
        "best_val_loss": round(best_val, 4),
        "device": torch.cuda.get_device_name(0) if device.startswith("cuda") else device,
        "bf16": use_bf16,
        "config": asdict(cfg),
    }
    with open(os.path.join(args.out_dir, "run.json"), "w") as f:
        json.dump(run, f, indent=2)
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()

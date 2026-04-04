"""
Training loop for one K-value experiment.
Usage: python train.py --K 4 --seed 42
"""

import os
import time
import math
import json
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from model import Transformer, DecoderOnlyTransformer
from data import get_dataloaders, get_lm_dataloaders, build_masks

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────
# Scheduler: warmup + inverse sqrt decay
# ──────────────────────────────────────────
class WarmupInvSqrtScheduler:
    """
    lrate = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})
    As in 'Attention Is All You Need'.
    """
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.opt = optimizer
        self.d_model = d_model
        self.warmup = warmup_steps
        self._step = 0

    def step(self):
        self._step += 1
        lr = self._lrate()
        for pg in self.opt.param_groups:
            pg['lr'] = lr
        return lr

    def _lrate(self):
        s = self._step
        return self.d_model ** -0.5 * min(s ** -0.5, s * self.warmup ** -1.5)


# ──────────────────────────────────────────
# Label-smoothed cross entropy
# ──────────────────────────────────────────
class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, pad_idx=0, smoothing=0.1):
        super().__init__()
        self.vocab = vocab_size
        self.pad = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, target):
        # logits: [B*T, V]   target: [B*T]
        B = logits.size(0)
        with torch.no_grad():
            smooth_val = self.smoothing / (self.vocab - 2)
            dist = torch.full_like(logits, smooth_val)
            dist.scatter_(1, target.unsqueeze(1), self.confidence)
            dist[:, self.pad] = 0
            mask = (target == self.pad)
            dist[mask] = 0

        log_prob = torch.log_softmax(logits, dim=-1)
        loss = -(dist * log_prob).sum(1)
        non_pad = (~mask).sum().float().clamp(min=1)
        return loss.sum() / non_pad


# ──────────────────────────────────────────
# GPU memory helpers
# ──────────────────────────────────────────
def gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0


def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ──────────────────────────────────────────
# One epoch
# ──────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, scheduler, device,
              train=True, grad_clip=1.0):
    model.train() if train else model.eval()
    total_loss, total_tokens = 0.0, 0
    t0 = time.time()

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)

            # decoder input = tgt[:-1], target = tgt[1:]
            tgt_in  = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            src_mask, tgt_mask = build_masks(src, tgt_in, pad_idx=0)
            src_mask = src_mask.to(device)
            tgt_mask = tgt_mask.to(device)

            logits = model(src, tgt_in, src_mask, tgt_mask)
            # logits: [B, T, V] → [B*T, V]
            B, T, V = logits.shape
            loss = criterion(logits.reshape(-1, V), tgt_out.reshape(-1))

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scheduler.step()
                optimizer.step()


            n_tokens = (tgt_out != 0).sum().item()
            total_loss   += loss.item() * n_tokens
            total_tokens += n_tokens

    elapsed = time.time() - t0
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 100))
    tps = total_tokens / elapsed       # tokens per second
    return avg_loss, ppl, tps


def run_epoch_lm(model, loader, criterion, optimizer, scheduler, device,
                 train=True, grad_clip=1.0, pad_idx=0):
    model.train() if train else model.eval()
    total_loss, total_tokens = 0.0, 0
    t0 = time.time()

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)  # [B, T+1]

            x = batch[:, :-1]
            y = batch[:, 1:]

            logits = model(x)
            B, T, V = logits.shape
            loss = criterion(logits.reshape(-1, V), y.reshape(-1))

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scheduler.step()
                optimizer.step()

            n_tokens = (y != pad_idx).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    elapsed = time.time() - t0
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 100))
    tps = total_tokens / elapsed
    return avg_loss, ppl, tps

@torch.no_grad()
def greedy_decode(model, src, src_mask, tgt_tokenizer, device, max_len=64, min_len=3):
    model.eval()
    bos_id = tgt_tokenizer.token_to_id("<s>")
    eos_id = tgt_tokenizer.token_to_id("</s>")

    memory = model.encode(src, src_mask)
    B = src.size(0)

    ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)

    for step in range(max_len):
        # ── 修复：只给 tgt 掩码，src_mask 直接复用外面传进来的 ──
        T = ys.size(1)
        tgt_pad_mask = (ys != 0).unsqueeze(1).unsqueeze(2)           # [B,1,1,T]
        causal_mask  = torch.tril(torch.ones(T, T, device=device)).bool()
        tgt_mask     = tgt_pad_mask & causal_mask.unsqueeze(0).unsqueeze(0)

        # Decode returns hidden states; project to vocab before argmax.
        out = model.decode(ys, memory, src_mask, tgt_mask)
        logits = model.out_proj(out)
        # Avoid immediate EOS/PAD collapse for very short outputs by masking logits.
        if step < min_len:
            logits[:, -1, eos_id] = -1e9
            logits[:, -1, 0] = -1e9
        next_token = logits[:, -1, :].argmax(dim=-1)

        ys   = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
        done |= (next_token == eos_id)
        if done.all():
            break

    return ys[:, 1:].cpu().tolist()


def _strip_special(ids, bos_id, eos_id, pad_id=0):
    out = []
    for t in ids:
        if t == eos_id:
            break
        if t == bos_id or t == pad_id:
            continue
        out.append(t)
    return out


def _decode_stats(all_ids):
    total = len(all_ids)
    if total == 0:
        return dict(total=0, empty=0, all_same=0)
    empty = 0
    all_same = 0
    for ids in all_ids:
        if len(ids) == 0:
            empty += 1
        elif all(t == ids[0] for t in ids):
            all_same += 1
    return dict(total=total, empty=empty, all_same=all_same)


def compute_bleu(model, val_loader, src_tokenizer, tgt_tokenizer, device, max_batches=50):
    import sacrebleu
    hypotheses = []
    references  = []
    bos_id = tgt_tokenizer.token_to_id("<s>")
    eos_id = tgt_tokenizer.token_to_id("</s>")

    all_pred_clean = []
    for i, (src, tgt) in enumerate(val_loader):
        if max_batches and i >= max_batches:
            break

        src = src.to(device)

        # ── 修复：src_mask 只看 src 的 padding，和 tgt 无关 ──
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)   # [B,1,1,S]

        pred_ids = greedy_decode(model, src, src_mask, tgt_tokenizer, device)

        for j, (pred, ref) in enumerate(zip(pred_ids, tgt.tolist())):
            pred_clean = _strip_special(pred, bos_id, eos_id, pad_id=0)
            ref_clean  = _strip_special(ref,  bos_id, eos_id, pad_id=0)

            hyp     = tgt_tokenizer.decode(pred_clean)
            ref_str = tgt_tokenizer.decode(ref_clean)

            if i == 0 and j < 3:
                print(f"[debug] pred_ids : {pred[:20]}")
                print(f"[debug] ref_ids  : {ref_clean[:20]}")
                print(f"[debug] hyp      : {repr(hyp)}")
                print(f"[debug] ref_str  : {repr(ref_str)}")
                print()

            hypotheses.append(hyp)
            references.append(ref_str)
            all_pred_clean.append(pred_clean)

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    stats = _decode_stats(all_pred_clean)
    if stats["total"] > 0:
        empty_pct = 100.0 * stats["empty"] / stats["total"]
        same_pct = 100.0 * stats["all_same"] / stats["total"]
        print(f"[debug] decode_stats: total={stats['total']} empty={empty_pct:.1f}% all_same={same_pct:.1f}%")
    return round(bleu.score, 2)

# ──────────────────────────────────────────
# Inference latency benchmark
# ──────────────────────────────────────────
@torch.no_grad()
def measure_latency(model, src_vocab, device, seq_len=32, batch_sizes=(1, 32), n_runs=50):
    model.eval()
    results = {}
    for bs in batch_sizes:
        src = torch.randint(1, src_vocab, (bs, seq_len), device=device)
        tgt = torch.randint(1, 100,       (bs, seq_len), device=device)
        src_mask, tgt_mask = build_masks(src, tgt)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)

        # warm-up
        for _ in range(5):
            _ = model(src, tgt, src_mask, tgt_mask)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model(src, tgt, src_mask, tgt_mask)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) / n_runs * 1000
        results[f'latency_ms_bs{bs}'] = round(elapsed_ms, 2)
    return results


@torch.no_grad()
def measure_latency_lm(model, vocab_size, device, seq_len=32, batch_sizes=(1, 32), n_runs=50):
    model.eval()
    results = {}
    for bs in batch_sizes:
        x = torch.randint(1, vocab_size, (bs, seq_len), device=device)

        for _ in range(5):
            _ = model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) / n_runs * 1000
        results[f'latency_ms_bs{bs}'] = round(elapsed_ms, 2)
    return results


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',        type=str,   default='lm', choices=['mt', 'lm'])
    parser.add_argument('--dataset',     type=str,   default='wikitext-103')
    parser.add_argument('--K',           type=int,   default=4)
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--epochs',      type=int,   default=20)
    parser.add_argument('--batch_size',  type=int,   default=64)
    parser.add_argument('--d_model',     type=int,   default=512)
    parser.add_argument('--n_heads',     type=int,   default=8)
    parser.add_argument('--n_layers',    type=int,   default=6)
    parser.add_argument('--dropout',     type=float, default=0.1)
    parser.add_argument('--warmup',      type=int,   default=4000)
    parser.add_argument('--max_len',     type=int,   default=128)
    parser.add_argument('--lr',          type=float, default=1.0)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--train_samples', type=int, default=200000)
    parser.add_argument('--val_samples',   type=int, default=0)
    parser.add_argument('--out_dir',     type=str,   default='results')
    parser.add_argument('--attn_pattern', type=str,  default='alternating')
    parser.add_argument('--sparse_window', type=int, default=128)
    args = parser.parse_args()

    # reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device} | task={args.task} | K={args.K} | seed={args.seed}")

    # ── data ──
    val_samples = args.val_samples if args.val_samples and args.val_samples > 0 else None
    if args.task == "lm":
        train_loader, val_loader, vocab_size, tok = get_lm_dataloaders(
            args.batch_size, args.max_len,
            train_samples=args.train_samples, val_samples=val_samples,
            dataset_name=args.dataset
        )
        src_vocab = vocab_size
        tgt_vocab = vocab_size
        src_tok = tok
        tgt_tok = tok
    else:
        train_loader, val_loader, src_vocab, tgt_vocab, src_tok, tgt_tok = \
            get_dataloaders(args.batch_size, args.max_len,
                            train_samples=args.train_samples, val_samples=val_samples)

    # ── model ──
    if args.task == "lm":
        model = DecoderOnlyTransformer(
            vocab_size=tgt_vocab,
            d_model=args.d_model, n_heads=args.n_heads,
            n_layers=args.n_layers, K=args.K,
            dropout=args.dropout, max_len=args.max_len,
            pad_idx=0, attn_pattern=args.attn_pattern,
            sparse_window=args.sparse_window
        ).to(device)
    else:
        model = Transformer(
            src_vocab=src_vocab, tgt_vocab=tgt_vocab,
            d_model=args.d_model, n_heads=args.n_heads,
            n_layers=args.n_layers, K=args.K,
            dropout=args.dropout, max_len=args.max_len
        ).to(device)

    n_params     = model.count_parameters()
    n_ffn_params = model.count_ffn_parameters()
    logger.info(f"Total params: {n_params:,}  |  FFN params: {n_ffn_params:,}")

    # ── optimizer ──
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=(0.9, 0.98), eps=1e-9)
    scheduler = WarmupInvSqrtScheduler(optimizer, args.d_model, args.warmup)
    criterion = LabelSmoothingLoss(tgt_vocab, pad_idx=0, smoothing=args.label_smoothing)

    # ── measure baseline GPU memory ──
    reset_peak_memory()

    # ── training loop ──
    history = []
    best_val_ppl = float('inf')
    best_ckpt = None

    out_dir = Path(args.out_dir) / f'K{args.K}_seed{args.seed}'
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        if args.task == "lm":
            train_loss, train_ppl, train_tps = run_epoch_lm(
                model, train_loader, criterion, optimizer, scheduler, device, train=True)
            val_loss, val_ppl, val_tps = run_epoch_lm(
                model, val_loader, criterion, optimizer, scheduler, device, train=False)
        else:
            train_loss, train_ppl, train_tps = run_epoch(
                model, train_loader, criterion, optimizer, scheduler, device, train=True)
            val_loss, val_ppl, val_tps = run_epoch(
                model, val_loader, criterion, optimizer, scheduler, device, train=False)
        peak_mem = gpu_memory_mb()

        bleu = None
        if args.task == "mt" and (epoch % 1 == 0 or epoch == args.epochs):
            bleu = compute_bleu(model, val_loader, src_tok, tgt_tok, device,
                                max_batches=50)
            logger.info(f"Epoch {epoch:02d} | BLEU={bleu}")

        # row = dict(epoch=epoch, K=args.K, seed=args.seed,
        #            train_ppl=round(train_ppl, 2),
        #            val_ppl=round(val_ppl, 2),
        #            bleu=bleu,
        #            train_tps=round(train_tps, 0),
        #            peak_mem_mb=round(peak_mem, 1))

        row = dict(epoch=epoch, K=args.K, seed=args.seed,
                   train_loss=round(train_loss, 4),
                   train_ppl=round(train_ppl, 2),
                   val_loss=round(val_loss, 4),
                   val_ppl=round(val_ppl, 2),
                   bleu=bleu,
                   train_tps=round(train_tps, 0),
                   peak_mem_mb=round(peak_mem, 1))
        history.append(row)

        logger.info(
            f"Epoch {epoch:02d} | "
            f"train_ppl={train_ppl:.2f} val_ppl={val_ppl:.2f} | "
            f"tps={train_tps:.0f} | mem={peak_mem:.0f}MB"
        )

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_ckpt = out_dir / 'best.pt'
            torch.save(model.state_dict(), best_ckpt)

    # ── latency benchmark ──
    if best_ckpt:
        model.load_state_dict(torch.load(best_ckpt))
    if args.task == "lm":
        latency = measure_latency_lm(model, src_vocab, device, seq_len=32)
    else:
        latency = measure_latency(model, src_vocab, device, seq_len=32)

    # ── save results ──
    summary = dict(
        K=args.K, seed=args.seed,
        n_params=n_params, n_ffn_params=n_ffn_params,
        d_ff=args.K * args.d_model,
        best_val_ppl=round(best_val_ppl, 2),
        task=args.task,
        attn_pattern=args.attn_pattern if args.task == "lm" else "softmax",
        **latency,
        history=history
    )
    summary_path = out_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved results to {summary_path}")


if __name__ == '__main__':
    main()

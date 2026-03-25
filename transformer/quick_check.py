"""
quick_check.py  — smoke test that runs WITHOUT any dataset download.
Verifies:
  1. Model instantiation for all K values
  2. Forward pass shapes
  3. Loss backward
  4. Latency measurement
  5. Parameter count table

Run before starting the full experiment:
    python quick_check.py
"""

import time
import math
import torch
import torch.nn as nn

import sys
sys.path.insert(0, '.')
from model import Transformer, FeedForward
from data import build_masks


def param_table():
    print("\n" + "="*55)
    print(f"{'K':>4} | {'d_ff':>6} | {'总参数量':>10} | {'FFN参数量':>10} | {'FFN占比':>7}")
    print("-"*55)

    D_MODEL  = 512
    N_HEADS  = 8
    N_LAYERS = 6
    VOCAB    = 8000

    for K in [1, 2, 4, 8, 16]:
        m = Transformer(VOCAB, VOCAB, D_MODEL, N_HEADS, N_LAYERS, K, dropout=0.0)
        total = m.count_parameters()
        ffn   = m.count_ffn_parameters()
        ratio = ffn / total * 100
        print(f"{K:>4} | {K*D_MODEL:>6} | {total:>10,} | {ffn:>10,} | {ratio:>6.1f}%")
    print("="*55)


def forward_check():
    print("\n── Forward pass check ──")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, S, T, V = 2, 20, 18, 1000

    for K in [1, 2, 4, 8, 16]:
        m = Transformer(V, V, d_model=128, n_heads=4, n_layers=2, K=K, dropout=0.0)
        m.to(device).eval()

        src = torch.randint(1, V, (B, S), device=device)
        tgt = torch.randint(1, V, (B, T), device=device)
        src_mask, tgt_mask = build_masks(src, tgt)
        src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)

        with torch.no_grad():
            out = m(src, tgt, src_mask, tgt_mask)
        assert out.shape == (B, T, V), f"Shape error: {out.shape}"
        print(f"  K={K:>2} ✓  output {out.shape}")


def backward_check():
    print("\n── Backward pass check ──")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, S, T, V = 2, 20, 18, 1000

    m = Transformer(V, V, d_model=128, n_heads=4, n_layers=2, K=4, dropout=0.1)
    m.to(device).train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss(ignore_index=0)

    src = torch.randint(1, V, (B, S), device=device)
    tgt = torch.randint(1, V, (B, T), device=device)
    tgt_in  = tgt[:, :-1]
    tgt_out = tgt[:, 1:]
    src_mask, tgt_mask = build_masks(src, tgt_in)
    src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)

    logits = m(src, tgt_in, src_mask, tgt_mask)
    loss = crit(logits.reshape(-1, V), tgt_out.reshape(-1))
    loss.backward()
    opt.step()
    print(f"  loss={loss.item():.4f}  grad_norm={_grad_norm(m):.4f} ✓")


def _grad_norm(model):
    total = 0.
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total)


def latency_check():
    print("\n── Latency check (seq_len=32) ──")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    V = 1000

    print(f"  {'K':>4} | {'bs=1 (ms)':>10} | {'bs=32 (ms)':>11}")
    print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*11}")
    for K in [1, 2, 4, 8, 16]:
        m = Transformer(V, V, d_model=128, n_heads=4, n_layers=2, K=K, dropout=0.0)
        m.to(device).eval()
        res = {}
        for bs in [1, 32]:
            src = torch.randint(1, V, (bs, 32), device=device)
            tgt = torch.randint(1, V, (bs, 32), device=device)
            sm, tm = build_masks(src, tgt)
            sm, tm = sm.to(device), tm.to(device)
            with torch.no_grad():
                for _ in range(5): m(src, tgt, sm, tm)   # warmup
                if device.type == 'cuda': torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(50): m(src, tgt, sm, tm)
                if device.type == 'cuda': torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / 50 * 1000
            res[bs] = ms
        print(f"  {K:>4} | {res[1]:>10.2f} | {res[32]:>11.2f}")


if __name__ == '__main__':
    print("FFN 放大倍数消融实验 — 快速自检")
    print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")

    param_table()
    forward_check()
    backward_check()
    latency_check()

    print("\n✓ 所有检查通过，可以开始正式训练！")
    print("  下一步: python train.py --K 4 --seed 42 --epochs 1  (先用1 epoch验证)")

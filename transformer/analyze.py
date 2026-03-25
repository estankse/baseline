"""
analyze.py — aggregate all experiment results and produce:
  1. Console summary table
  2. Four matplotlib figures saved to results/plots/
  3. results/final_table.csv

Run after all K experiments finish:
    python analyze.py
"""

import json
import math
import glob
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

RESULTS_DIR = Path("results")
PLOTS_DIR   = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['axes.unicode_minus'] = False

STYLE = {
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.35,
    'grid.linestyle':     '--',
    'font.family':        'Microsoft YaHei',
    'font.size':          11,
}
plt.rcParams.update(STYLE)

COLORS = {1: '#888', 2: '#5B8BD4', 4: '#E8593C', 6: '#3BB88C', 16: '#9B5DE5'}
K_LABEL = {k: f'K={k}' for k in COLORS}


# ──────────────────────────────────────────
# Load results
# ──────────────────────────────────────────
def load_results():
    rows = []
    for path in glob.glob(str(RESULTS_DIR / '**/summary.json'), recursive=True):
        with open(path) as f:
            d = json.load(f)
        rows.append(d)
    return rows


def aggregate(rows):
    """
    Group by K, compute mean ± std over seeds for key metrics.
    Also extract per-epoch history for learning curves.
    """
    by_k = defaultdict(list)
    for r in rows:
        by_k[r['K']].append(r)

    agg = {}
    for K, runs in sorted(by_k.items()):
        ppls    = [r['best_val_ppl']                      for r in runs]
        params  = [r['n_params']       / 1e6              for r in runs]  # millions
        lat_bs1 = [r.get('latency_ms_bs1', float('nan')) for r in runs]
        lat_bs32= [r.get('latency_ms_bs32', float('nan'))for r in runs]

        final_bleus = []
        for r in runs:
            # history 是一个 list，每个元素是一个 epoch 的 dict
            bleu_vals = [
                row['bleu']
                for row in r.get('history', [])
                if row.get('bleu') is not None
            ]
            if bleu_vals:
                final_bleus.append(bleu_vals[-1])

        agg[K] = dict(
            K=K,
            d_ff=runs[0]['d_ff'],
            n_params_M=np.mean(params),
            n_ffn_params=runs[0]['n_ffn_params'] / 1e6,
            ppl_mean=np.mean(ppls),
            ppl_std=np.std(ppls),
            lat_bs1_mean=np.nanmean(lat_bs1),
            lat_bs32_mean=np.nanmean(lat_bs32),
            bleu_mean=round(np.mean(final_bleus), 2) if final_bleus else None,
            bleu_std=round(np.std(final_bleus), 2) if final_bleus else None,
        )

    return agg


# ──────────────────────────────────────────
# Plots
# ──────────────────────────────────────────
def plot_ppl_vs_K(agg):
    """Val perplexity vs K (bar + errorbar)."""
    Ks    = sorted(agg)
    means = [agg[k]['ppl_mean'] for k in Ks]
    stds  = [agg[k]['ppl_std']  for k in Ks]
    clrs  = [COLORS[k] for k in Ks]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([str(k) for k in Ks], means, color=clrs, alpha=0.85,
                  yerr=stds, capsize=5, error_kw={'elinewidth': 1.2})
    ax.set_xlabel('FFN 放大倍数 K')
    ax.set_ylabel('Validation Perplexity ↓')
    ax.set_title('Val PPL vs K  (mean ± std, 3 seeds)')

    # Highlight K=4
    idx4 = Ks.index(4)
    bars[idx4].set_edgecolor('black')
    bars[idx4].set_linewidth(2)
    ax.annotate('原论文设定', xy=(idx4, means[idx4]),
                xytext=(idx4 + 0.4, means[idx4] + stds[idx4] + 0.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
                fontsize=10)

    fig.tight_layout()
    p = PLOTS_DIR / 'ppl_vs_K.png'
    fig.savefig(p, dpi=150)
    print(f"Saved {p}")
    plt.close(fig)


def plot_params_vs_K(agg):
    """Total params and FFN params vs K."""
    Ks     = sorted(agg)
    total  = [agg[k]['n_params_M']   for k in Ks]
    ffn    = [agg[k]['n_ffn_params']  for k in Ks]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([str(k) for k in Ks], total, 'o-', color='#5B8BD4', lw=2, label='总参数量 (M)')
    ax.plot([str(k) for k in Ks], ffn,   's--',color='#E8593C', lw=2, label='FFN参数量 (M)')
    ax.set_xlabel('FFN 放大倍数 K')
    ax.set_ylabel('参数量 (百万)')
    ax.set_title('参数量 vs K')
    ax.legend()
    ax.axvline(x=Ks.index(4), color='gray', linestyle=':', alpha=0.5, label='K=4')

    fig.tight_layout()
    p = PLOTS_DIR / 'params_vs_K.png'
    fig.savefig(p, dpi=150)
    print(f"Saved {p}")
    plt.close(fig)


def plot_pareto(agg):
    """
    Pareto front: val PPL vs total parameters.
    We want low PPL (quality) and low params (efficiency).
    """
    Ks    = sorted(agg)
    ppls  = [agg[k]['ppl_mean']   for k in Ks]
    params= [agg[k]['n_params_M'] for k in Ks]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for K, ppl, param in zip(Ks, ppls, params):
        ax.scatter(param, ppl, s=120, color=COLORS[K], zorder=3)
        ax.annotate(f'K={K}', (param, ppl),
                    textcoords='offset points', xytext=(8, 4), fontsize=10)

    ax.set_xlabel('总参数量 (M)')
    ax.set_ylabel('Validation Perplexity ↓')
    ax.set_title('质量 vs 效率  (Pareto视角)\n左下角 = 更好')

    # draw dashed "ideal" arrow
    ax.annotate('', xy=(min(params)*0.97, min(ppls)*0.97),
                xytext=(max(params)*0.5, max(ppls)*0.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1, linestyle='dashed'))

    fig.tight_layout()
    p = PLOTS_DIR / 'pareto.png'
    fig.savefig(p, dpi=150)
    print(f"Saved {p}")
    plt.close(fig)

def plot_bleu_vs_K(agg):
    Ks    = [k for k in sorted(agg) if agg[k].get('bleu_mean') is not None]
    means = [agg[k]['bleu_mean'] for k in Ks]
    stds  = [agg[k]['bleu_std']  for k in Ks]
    clrs  = [COLORS[k] for k in Ks]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([str(k) for k in Ks], means, color=clrs, alpha=0.85,
                  yerr=stds, capsize=5, error_kw={'elinewidth': 1.2})
    ax.set_xlabel('FFN 放大倍数 K')
    ax.set_ylabel('BLEU score ↑')
    ax.set_title('BLEU vs K  (mean ± std, 3 seeds)')

    idx4 = Ks.index(4) if 4 in Ks else None
    if idx4 is not None:
        bars[idx4].set_edgecolor('black')
        bars[idx4].set_linewidth(2)

    fig.tight_layout()
    p = PLOTS_DIR / 'bleu_vs_K.png'
    fig.savefig(p, dpi=150)
    print(f"Saved {p}")
    plt.close(fig)

def plot_ppl_bleu_dual(agg):
    """双轴图：左轴 PPL，右轴 BLEU，同一张图对比趋势。"""
    Ks    = sorted(agg)
    ppls  = [agg[k]['ppl_mean']  for k in Ks]
    bleus = [agg[k].get('bleu_mean', None) for k in Ks]

    if any(b is None for b in bleus):
        print("BLEU 数据不完整，跳过双轴图")
        return

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    xs = [str(k) for k in Ks]
    ax1.plot(xs, ppls,  'o-', color='#5B8BD4', lw=2, label='Val PPL ↓')
    ax2.plot(xs, bleus, 's--', color='#E8593C', lw=2, label='BLEU ↑')

    ax1.set_xlabel('FFN 放大倍数 K')
    ax1.set_ylabel('Val Perplexity ↓', color='#5B8BD4')
    ax2.set_ylabel('BLEU score ↑',     color='#E8593C')
    ax1.tick_params(axis='y', labelcolor='#5B8BD4')
    ax2.tick_params(axis='y', labelcolor='#E8593C')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    ax1.set_title('PPL vs BLEU — 两个指标是否同步？')

    fig.tight_layout()
    p = PLOTS_DIR / 'ppl_bleu_dual.png'
    fig.savefig(p, dpi=150)
    print(f"Saved {p}")
    plt.close(fig)

def plot_latency(agg):
    """Inference latency vs K."""
    Ks     = sorted(agg)
    lat1   = [agg[k]['lat_bs1_mean']  for k in Ks]
    lat32  = [agg[k]['lat_bs32_mean'] for k in Ks]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([str(k) for k in Ks], lat1,  'o-', color='#5B8BD4', lw=2, label='batch=1 (ms)')
    ax.plot([str(k) for k in Ks], lat32, 's--',color='#3BB88C', lw=2, label='batch=32 (ms)')
    ax.set_xlabel('FFN 放大倍数 K')
    ax.set_ylabel('推理延迟 (ms/batch)')
    ax.set_title('推理延迟 vs K')
    ax.legend()

    fig.tight_layout()
    p = PLOTS_DIR / 'latency_vs_K.png'
    fig.savefig(p, dpi=150)
    print(f"Saved {p}")
    plt.close(fig)


def print_table(agg):
    """Pretty print summary table and save CSV."""
    rows = []
    for K in sorted(agg):
        a = agg[K]
        rows.append({
            'K':           K,
            'd_ff':        a['d_ff'],
            '参数量(M)':   f"{a['n_params_M']:.1f}",
            'FFN参数(M)':  f"{a['n_ffn_params']:.1f}",
            'Val PPL':     f"{a['ppl_mean']:.2f} ± {agg[K]['ppl_std']:.2f}",
            '延迟bs=1(ms)':f"{a['lat_bs1_mean']:.1f}",
            '延迟bs=32(ms)':f"{a['lat_bs32_mean']:.1f}",
        })
    df = pd.DataFrame(rows)
    print("\n" + "="*70)
    print("FFN放大倍数消融实验  汇总")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)

    csv_path = RESULTS_DIR / 'final_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to {csv_path}")


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────
if __name__ == '__main__':
    rows = load_results()
    if not rows:
        print("No result files found. Did you run train.py yet?")
        print("Example:  python train.py --K 4 --seed 42 --epochs 1")
        exit(0)

    agg = aggregate(rows)

    print_table(agg)
    plot_ppl_vs_K(agg)
    plot_params_vs_K(agg)
    plot_pareto(agg)
    plot_latency(agg)
    plot_bleu_vs_K(agg)
    plot_ppl_bleu_dual(agg)

    # interpretation hint
    K_star = min(agg, key=lambda k: agg[k]['ppl_mean'])
    print(f"\n最低 Val PPL 出现在 K={K_star}")
    if K_star == 4:
        print("→ 结论支持：K=4 是质量最优点，与原论文一致。")
    elif K_star > 4:
        print(f"→ 在你的数据规模/设定下 K={K_star} 更优，"
              "可能因为模型欠拟合；增大数据量或训练更久后 K=4 可能收敛到相近水平。")
    else:
        print(f"→ K={K_star} 更优，说明在这个小规模实验里更窄的FFN足够。"
              "尝试增大数据量或增加训练步数，观察拐点是否向 K=4 移动。")

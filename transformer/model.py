"""
Transformer model with configurable FFN width multiplier K.
d_ff = K * d_model
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        #multihead in one metrix
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)# batch size
        # project and reshape to [B, heads, seq, d_k]
        q = self.w_q(q).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:#filled the masked position as -infinity
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_k)#QKᵀ / √d_k
        return self.w_o(out)


class LinearAttention(nn.Module):
    """
    Kernel-based linear attention (elu + 1) with optional causal prefix-sum.
    Designed for decoder-only self-attention.
    """
    def __init__(self, d_model, n_heads, dropout=0.1, eps=1e-6):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.eps = eps

    @staticmethod
    def _phi(x):
        return F.elu(x) + 1.0

    def forward(self, x, pad_mask=None, causal=False):
        B, T, _ = x.size()
        q = self.w_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        q = self._phi(q)
        k = self._phi(k)

        if pad_mask is not None:
            mask = pad_mask[:, None, :, None].float()
            k = k * mask
            v = v * mask

        if causal:
            kv = torch.einsum('bhtd,bhte->bhtde', k, v)  # [B,H,T,D,D]
            kv = kv.cumsum(dim=2)
            k_cum = k.cumsum(dim=2)                     # [B,H,T,D]
            out = torch.einsum('bhtd,bhtde->bhte', q, kv)
            z = torch.einsum('bhtd,bhtd->bht', q, k_cum)
        else:
            kv = torch.einsum('bhtd,bhte->bhde', k, v)   # [B,H,D,D]
            out = torch.einsum('bhtd,bhde->bhte', q, kv)
            k_sum = k.sum(dim=2)                        # [B,H,D]
            z = torch.einsum('bhtd,bhd->bht', q, k_sum)

        out = out / (z.unsqueeze(-1) + self.eps)
        out = self.dropout(out)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_k)
        return self.w_o(out)


class LocalCausalAttention(nn.Module):
    """
    Sparse (local window) causal attention implemented via masked softmax.
    """
    def __init__(self, d_model, n_heads, window_size=128, dropout=0.1):
        super().__init__()
        self.window = window_size
        self.inner = MultiHeadAttention(d_model, n_heads, dropout)

    def forward(self, x, pad_mask):
        B, T, _ = x.size()
        device = x.device
        idx = torch.arange(T, device=device)
        dist = idx[None, :] - idx[:, None]
        window_mask = (dist >= 0) & (dist < self.window)  # [T, T]
        window_mask = window_mask.unsqueeze(0).unsqueeze(0)  # [1,1,T,T]

        pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)        # [B,1,1,T]
        full_mask = pad_mask & window_mask
        return self.inner(x, x, x, full_mask)


class FeedForward(nn.Module):
    """FFN with configurable multiplier K: d_ff = K * d_model."""
    def __init__(self, d_model, K=4, dropout=0.1):
        super().__init__()
        d_ff = K * d_model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, K, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, K, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.drop(self.attn(x, x, x, mask)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, K, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, K, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.drop(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.drop(self.ff(x)))
        return x


class DecoderOnlyBlock(nn.Module):
    def __init__(self, d_model, n_heads, K, attn_type="softmax",
                 dropout=0.1, sparse_window=128):
        super().__init__()
        self.attn_type = attn_type
        if attn_type == "linear":
            self.attn = LinearAttention(d_model, n_heads, dropout)
        elif attn_type == "sparse":
            self.attn = LocalCausalAttention(d_model, n_heads, sparse_window, dropout)
        else:
            self.attn = MultiHeadAttention(d_model, n_heads, dropout)

        self.ff = FeedForward(d_model, K, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask, causal_mask=None):
        if self.attn_type == "linear":
            attn_out = self.attn(x, pad_mask=pad_mask, causal=True)
        elif self.attn_type == "sparse":
            attn_out = self.attn(x, pad_mask)
        else:
            attn_out = self.attn(x, x, x, causal_mask)

        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, n_heads=8,
                 n_layers=6, K=4, dropout=0.1, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.K = K

        self.src_embed = nn.Embedding(src_vocab, d_model, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, K, dropout) for _ in range(n_layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, K, dropout) for _ in range(n_layers)]
        )
        self.out_proj = nn.Linear(d_model, tgt_vocab)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        x = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, memory, src_mask, tgt_mask):
        x = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder:
            x = layer(x, memory, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encode(src, src_mask)
        out = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.out_proj(out)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_ffn_parameters(self):
        """Parameters in FFN layers only."""
        total = 0
        for m in self.modules():
            if isinstance(m, FeedForward):
                total += sum(p.numel() for p in m.parameters())
        return total


class DecoderOnlyTransformer(nn.Module):
    """
    Decoder-only Transformer for language modeling.
    Supports softmax / linear / sparse attention per layer.
    """
    def __init__(self, vocab_size, d_model=512, n_heads=8,
                 n_layers=6, K=4, dropout=0.1, max_len=512,
                 pad_idx=0, attn_pattern="softmax", sparse_window=128):
        super().__init__()
        self.d_model = d_model
        self.K = K
        self.pad_idx = pad_idx

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        attn_types = self._parse_attn_pattern(attn_pattern, n_layers)
        self.layers = nn.ModuleList([
            DecoderOnlyBlock(
                d_model, n_heads, K, attn_type=attn_types[i],
                dropout=dropout, sparse_window=sparse_window
            )
            for i in range(n_layers)
        ])
        self.out_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    @staticmethod
    def _parse_attn_pattern(pattern, n_layers):
        pattern = pattern.lower().strip()
        if pattern in ("softmax", "standard"):
            return ["softmax"] * n_layers
        if pattern in ("linear", "all_linear"):
            return ["linear"] * n_layers
        if pattern in ("sparse", "all_sparse"):
            return ["sparse"] * n_layers
        if pattern in ("alternating", "alt"):
            return ["linear" if i % 2 == 0 else "softmax" for i in range(n_layers)]
        if pattern in ("alt_sparse", "alternating_sparse"):
            return ["linear" if i % 2 == 0 else "sparse" for i in range(n_layers)]

        parts = [p.strip() for p in pattern.split(",") if p.strip()]
        if len(parts) == 1:
            return parts * n_layers
        if len(parts) < n_layers:
            parts = (parts * (n_layers // len(parts) + 1))[:n_layers]
        return parts[:n_layers]

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        B, T = x.size()
        pad_mask = (x != self.pad_idx)  # [B, T]
        causal = torch.tril(torch.ones(T, T, device=x.device)).bool()
        causal_mask = pad_mask.unsqueeze(1).unsqueeze(2) & causal.unsqueeze(0).unsqueeze(0)

        h = self.pos_enc(self.embed(x) * math.sqrt(self.d_model))
        for layer in self.layers:
            h = layer(h, pad_mask, causal_mask)
        return self.out_proj(h)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_ffn_parameters(self):
        total = 0
        for m in self.modules():
            if isinstance(m, FeedForward):
                total += sum(p.numel() for p in m.parameters())
        return total

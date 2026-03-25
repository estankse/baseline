"""
Data utilities for WMT14 En-De translation.
Uses HuggingFace datasets + tokenizers for reproducibility.

If you don't have the full WMT14, falls back to a smaller
Multi30k dataset so you can test on a laptop.
"""
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import logging

logger = logging.getLogger(__name__)


def build_masks(src, tgt, pad_idx=0):
    """Build source padding mask and target causal mask."""
    # src_mask: [B, 1, 1, S]  (1 = keep, 0 = ignore)
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)

    # tgt_mask: causal + padding  [B, 1, T, T]
    T = tgt.size(1)
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)         # [B,1,1,T]
    causal = torch.tril(torch.ones(T, T, device=tgt.device)).bool()  # [T,T]
    tgt_mask = pad_mask & causal.unsqueeze(0).unsqueeze(0)

    return src_mask, tgt_mask


class TranslationDataset(Dataset):
    """
    Wraps a HuggingFace dataset split into a PyTorch Dataset.
    Each item returns (src_ids, tgt_ids) as LongTensors.
    """
    def __init__(self, hf_dataset, src_tokenizer, tgt_tokenizer,
                 max_len=128, src_lang='en', tgt_lang='de'):
        self.data = hf_dataset
        self.src_tok = src_tokenizer
        self.tgt_tok = tgt_tokenizer
        self.max_len = max_len
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 检查是否有 'translation' 这个嵌套键
        if 'translation' in item:
            src = item['translation'][self.src_lang]
            tgt = item['translation'][self.tgt_lang]
        else:
            # 如果没有，直接通过语言代码获取
            src = item[self.src_lang]
            tgt = item[self.tgt_lang]

        # Add special tokens so training matches inference (BOS/EOS).
        src_eos = self.src_tok.token_to_id("</s>")
        tgt_bos = self.tgt_tok.token_to_id("<s>")
        tgt_eos = self.tgt_tok.token_to_id("</s>")

        src_ids = self.src_tok.encode(src).ids[:max(self.max_len - 1, 1)]
        src_ids = src_ids + [src_eos]

        tgt_ids = self.tgt_tok.encode(tgt).ids[:max(self.max_len - 2, 1)]
        tgt_ids = [tgt_bos] + tgt_ids + [tgt_eos]

        return torch.tensor(src_ids, dtype=torch.long), \
               torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch, pad_idx=0):
    src_batch, tgt_batch = zip(*batch)
    src_pad = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_pad = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_pad, tgt_pad


def get_dataloaders(batch_size=64, max_len=128, num_workers=2):
    """
    Download and prepare WMT14 En-De.
    Falls back to multi30k if WMT14 isn't available.
    Returns (train_loader, val_loader, src_vocab_size, tgt_vocab_size,
             src_tokenizer, tgt_tokenizer)
    """
    from datasets import load_dataset
    from tokenizers import ByteLevelBPETokenizer
    import os, tempfile

    logger.info("Loading dataset...")
    # try:
    #     ds = load_dataset("wmt14", "de-en", split={'train': 'train', 'validation': 'validation'})
    #     logger.info("Using WMT14 en-de")
    #     src_lang, tgt_lang = 'en', 'de'
    # except Exception:
    #     logger.warning("WMT14 not available, falling back to multi30k")
    #     ds = load_dataset("bentrevett/multi30k", split={'train': 'train', 'validation': 'validation'})
    ds = load_dataset("opus100", "de-en", split={'train': 'train', 'validation': 'validation'})
    src_lang, tgt_lang = 'en', 'de'

    # Train BPE tokenizers (shared vocab for En+De is common; here separate for clarity)
    tmp = tempfile.mkdtemp()

    def _write_corpus(split_data, lang, path):
        with open(path, 'w', encoding='utf-8') as f:
            for item in split_data:
                line = item['translation'][lang] if 'translation' in item else item[lang]
                f.write(line.strip() + '\n')

    src_corpus = os.path.join(tmp, 'src.txt')
    tgt_corpus = os.path.join(tmp, 'tgt.txt')
    _write_corpus(ds['train'], src_lang, src_corpus)
    _write_corpus(ds['train'], tgt_lang, tgt_corpus)

    logger.info("Training BPE tokenizers...")
    src_tok = ByteLevelBPETokenizer()
    src_tok.train([src_corpus], vocab_size=8000, min_frequency=2,
                  special_tokens=["<pad>", "<s>", "</s>", "<unk>"])
    tgt_tok = ByteLevelBPETokenizer()
    tgt_tok.train([tgt_corpus], vocab_size=8000, min_frequency=2,
                  special_tokens=["<pad>", "<s>", "</s>", "<unk>"])

    src_vocab = src_tok.get_vocab_size()
    tgt_vocab = tgt_tok.get_vocab_size()
    logger.info(f"src_vocab={src_vocab}, tgt_vocab={tgt_vocab}")

    # Enable padding/truncation
    src_tok.enable_padding(pad_id=0, pad_token="<pad>")
    tgt_tok.enable_padding(pad_id=0, pad_token="<pad>")

    train_ds = TranslationDataset(ds['train'], src_tok, tgt_tok, max_len, src_lang, tgt_lang)
    val_ds   = TranslationDataset(ds['validation'], src_tok, tgt_tok, max_len, src_lang, tgt_lang)

    _collate = partial(collate_fn, pad_idx=0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=_collate,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=_collate,
                              pin_memory=True)

    return train_loader, val_loader, src_vocab, tgt_vocab, src_tok, tgt_tok

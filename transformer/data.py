"""
Data utilities for WMT14 En-De translation.
Uses HuggingFace datasets + tokenizers for reproducibility.

If you don't have the full WMT14, falls back to a smaller
Multi30k dataset so you can test on a laptop.
"""
from functools import partial
import math

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


def build_lm_masks(tgt, pad_idx=0):
    """Build padding mask and causal mask for LM."""
    pad_mask = (tgt != pad_idx)  # [B, T]
    T = tgt.size(1)
    causal = torch.tril(torch.ones(T, T, device=tgt.device)).bool()
    return pad_mask, causal


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


class LanguageModelingDataset(Dataset):
    """
    Blocks of token ids for decoder-only LM.
    Each item is a fixed-length tensor [block_size].
    """
    def __init__(self, tokens, block_size, pad_idx=0):
        self.tokens = tokens
        self.block = block_size
        self.pad_idx = pad_idx

    def __len__(self):
        return math.ceil(len(self.tokens) / self.block)

    def __getitem__(self, idx):
        start = idx * self.block
        end = start + self.block
        block = self.tokens[start:end]
        if len(block) < self.block:
            block = block + [self.pad_idx] * (self.block - len(block))
        return torch.tensor(block, dtype=torch.long)


def collate_fn(batch, pad_idx=0):
    src_batch, tgt_batch = zip(*batch)
    src_pad = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_pad = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_pad, tgt_pad


def get_dataloaders(batch_size=64, max_len=128, num_workers=2, train_samples=200000, val_samples=None):
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

    train_split = f"train[:{train_samples}]" if train_samples else "train"
    val_split = f"validation[:{val_samples}]" if val_samples else "validation"
    ds = load_dataset("opus100", "de-en", split={'train': train_split, 'validation': val_split})


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


def get_lm_dataloaders(batch_size=64, max_len=128, num_workers=2,
                       dataset_name="wikitext-103", train_samples=None, val_samples=None):
    """
    Load WikiText-103 or TinyShakespeare for decoder-only LM.
    Returns (train_loader, val_loader, vocab_size, tokenizer).
    """
    from datasets import load_dataset
    from tokenizers import ByteLevelBPETokenizer
    import os, tempfile

    logger.info(f"Loading LM dataset: {dataset_name}")

    if dataset_name.lower() in ("wikitext-103", "wikitext103", "wikitext"):
        ds = load_dataset("wikitext", "wikitext-103-raw-v1")
        text_field = "text"
    elif dataset_name.lower() in ("tinyshakespeare", "tiny_shakespeare", "shakespeare"):
        ds = load_dataset("tiny_shakespeare")
        text_field = "text"
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    if "validation" not in ds:
        ds = ds["train"].train_test_split(test_size=0.05, seed=42)
        ds = {"train": ds["train"], "validation": ds["test"]}

    if train_samples:
        ds["train"] = ds["train"].select(range(min(train_samples, len(ds["train"]))))
    if val_samples:
        ds["validation"] = ds["validation"].select(range(min(val_samples, len(ds["validation"]))))

    tmp = tempfile.mkdtemp()
    corpus_path = os.path.join(tmp, "lm.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for item in ds["train"]:
            line = item[text_field]
            f.write(line.strip() + "\n")

    logger.info("Training LM BPE tokenizer...")
    tok = ByteLevelBPETokenizer()
    tok.train([corpus_path], vocab_size=8000, min_frequency=2,
              special_tokens=["<pad>", "<s>", "</s>", "<unk>"])

    tok.enable_padding(pad_id=0, pad_token="<pad>")
    pad_id = tok.token_to_id("<pad>")
    eos_id = tok.token_to_id("</s>")

    def _encode_split(split_data):
        tokens = []
        for item in split_data:
            text = item[text_field]
            if not text:
                continue
            ids = tok.encode(text).ids
            tokens.extend(ids + [eos_id])
        return tokens

    train_tokens = _encode_split(ds["train"])
    val_tokens = _encode_split(ds["validation"])

    block_size = max_len + 1  # input + next token
    train_ds = LanguageModelingDataset(train_tokens, block_size, pad_idx=pad_id)
    val_ds = LanguageModelingDataset(val_tokens, block_size, pad_idx=pad_id)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, tok.get_vocab_size(), tok

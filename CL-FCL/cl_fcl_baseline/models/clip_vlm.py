"""Small, dependency-light OpenAI CLIP wrapper used by the VLM FCL methods.

The original repository previously converted only CLIP's visual transformer and
discarded both projection matrices and the complete text tower.  That is enough
for a linear image classifier, but it cannot reproduce MoAFCL or Fed-Duet: both
methods learn context tokens and classify with CLIP image/text similarities.

This module deliberately reuses :class:`PromptedVisionTransformer` as the image
tower so adapters and multi-scale hooks stay shared across the FCL algorithms.
It implements the original CLIP text transformer, tokenizer, projections and
temperature and loads them from an OpenAI ``.pt``/state-dict checkpoint.
"""

from __future__ import annotations

import gzip
import html
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .fcl_models import PromptedVisionTransformer, QuickGELU


@lru_cache()
def _bytes_to_unicode() -> dict[int, str]:
    values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    characters = values[:]
    offset = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            characters.append(256 + offset)
            offset += 1
    return dict(zip(values, (chr(value) for value in characters)))


def _pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return set(zip(word, word[1:]))


def default_bpe_path() -> Path:
    """Locate OpenAI's public BPE vocabulary.

    A packaged vocabulary is preferred.  During in-repository reproduction we
    also accept the copies shipped with the authors' reference implementations.
    ``CLIP_BPE_PATH`` is useful when the package is installed independently.
    """

    candidates: list[Path] = []
    configured = os.environ.get("CLIP_BPE_PATH", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(Path(__file__).with_name("bpe_simple_vocab_16e6.txt.gz"))
    root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            root / "otherswork" / "FCL复现" / "MoAFCL-main" / "clip"
            / "bpe_simple_vocab_16e6.txt.gz",
            root / "otherswork" / "FCL复现" / "Fed-Duet-main" / "cil" / "clip"
            / "bpe_simple_vocab_16e6.txt.gz",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "OpenAI CLIP BPE vocabulary was not found. Set CLIP_BPE_PATH to "
        "bpe_simple_vocab_16e6.txt.gz."
    )


class SimpleCLIPTokenizer:
    """OpenAI CLIP byte-pair tokenizer without a mandatory ``ftfy`` import."""

    def __init__(self, bpe_path: str | Path | None = None) -> None:
        try:
            import regex as regex_module
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise RuntimeError("CLIP tokenization requires the `regex` package.") from exc

        self._regex = regex_module
        self.byte_encoder = _bytes_to_unicode()
        with gzip.open(Path(bpe_path) if bpe_path else default_bpe_path(), "rt", encoding="utf-8") as handle:
            merges = handle.read().splitlines()[1 : 49152 - 256 - 2 + 1]
        merge_pairs = [tuple(value.split()) for value in merges]
        vocab = list(self.byte_encoder.values())
        vocab.extend(value + "</w>" for value in self.byte_encoder.values())
        vocab.extend("".join(pair) for pair in merge_pairs)
        vocab.extend(("<|startoftext|>", "<|endoftext|>"))
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.bpe_ranks = dict(zip(merge_pairs, range(len(merge_pairs))))
        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }
        self.pattern = regex_module.compile(
            r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|"
            r"[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
            regex_module.IGNORECASE,
        )

    def _bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, math.inf))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            merged: list[str] = []
            index = 0
            while index < len(word):
                try:
                    match = word.index(first, index)
                    merged.extend(word[index:match])
                    index = match
                except ValueError:
                    merged.extend(word[index:])
                    break
                if index < len(word) - 1 and word[index + 1] == second:
                    merged.append(first + second)
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = tuple(merged)
            if len(word) == 1:
                break
            pairs = _pairs(word)
        result = " ".join(word)
        self.cache[token] = result
        return result

    def encode(self, text: str) -> list[int]:
        try:
            import ftfy

            text = ftfy.fix_text(text)
        except ImportError:
            pass
        text = " ".join(html.unescape(html.unescape(text)).strip().split()).lower()
        output: list[int] = []
        for token in self._regex.findall(self.pattern, text):
            encoded = "".join(self.byte_encoder[value] for value in token.encode("utf-8"))
            output.extend(self.encoder[piece] for piece in self._bpe(encoded).split(" "))
        return output

    def tokenize(
        self,
        texts: str | Sequence[str],
        context_length: int = 77,
        truncate: bool = False,
    ) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        start = self.encoder["<|startoftext|>"]
        end = self.encoder["<|endoftext|>"]
        result = torch.zeros(len(texts), int(context_length), dtype=torch.long)
        for row, text in enumerate(texts):
            tokens = [start, *self.encode(str(text)), end]
            if len(tokens) > context_length:
                if not truncate:
                    raise RuntimeError(f"Input {text!r} is too long for CLIP context length {context_length}.")
                tokens = tokens[:context_length]
                tokens[-1] = end
            result[row, : len(tokens)] = torch.tensor(tokens)
        return result


class _CLIPTextBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 4),
            QuickGELU(),
            nn.Linear(width * 4, width),
        )

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normalized = self.ln_1(inputs)
        attended = self.attn(normalized, normalized, normalized, attn_mask=mask, need_weights=False)[0]
        inputs = inputs + attended
        return inputs + self.mlp(self.ln_2(inputs))


class CLIPTextTower(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.context_length = int(context_length)
        self.width = int(width)
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        self.blocks = nn.ModuleList([_CLIPTextBlock(width, heads) for _ in range(layers)])
        self.ln_final = nn.LayerNorm(width)
        self.text_projection = nn.Parameter(torch.empty(width, output_dim))
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.text_projection, std=width**-0.5)

    def forward(
        self,
        tokens: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = tokens.to(self.token_embedding.weight.device)
        embeddings = self.token_embedding(tokens)
        if context is not None:
            context = context.to(device=embeddings.device, dtype=embeddings.dtype)
            if context.ndim == 2:
                context = context.unsqueeze(0).expand(tokens.shape[0], -1, -1)
            if context.ndim != 3 or context.shape[0] != tokens.shape[0]:
                raise ValueError("CLIP context must have shape [n_ctx, width] or [classes, n_ctx, width].")
            if context.shape[2] != self.width or context.shape[1] >= self.context_length - 1:
                raise ValueError("CLIP context shape is incompatible with the text tower.")
            embeddings = embeddings.clone()
            embeddings[:, 1 : 1 + context.shape[1]] = context
        hidden = embeddings + self.positional_embedding.to(embeddings)
        mask = torch.full(
            (self.context_length, self.context_length),
            float("-inf"),
            device=hidden.device,
            dtype=hidden.dtype,
        ).triu_(1)
        for block in self.blocks:
            hidden = block(hidden, mask)
        hidden = self.ln_final(hidden)
        eot_positions = tokens.argmax(dim=-1)
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), eot_positions] @ self.text_projection

    def load_openai_state(self, state: Mapping[str, torch.Tensor]) -> None:
        converted: dict[str, torch.Tensor] = {
            "token_embedding.weight": state["token_embedding.weight"],
            "positional_embedding": state["positional_embedding"],
            "ln_final.weight": state["ln_final.weight"],
            "ln_final.bias": state["ln_final.bias"],
            "text_projection": state["text_projection"],
        }
        for index in range(len(self.blocks)):
            source = f"transformer.resblocks.{index}"
            destination = f"blocks.{index}"
            for name in (
                "ln_1.weight", "ln_1.bias", "attn.in_proj_weight", "attn.in_proj_bias",
                "attn.out_proj.weight", "attn.out_proj.bias", "ln_2.weight", "ln_2.bias",
            ):
                converted[f"{destination}.{name}"] = state[f"{source}.{name}"]
            converted[f"{destination}.mlp.0.weight"] = state[f"{source}.mlp.c_fc.weight"]
            converted[f"{destination}.mlp.0.bias"] = state[f"{source}.mlp.c_fc.bias"]
            converted[f"{destination}.mlp.2.weight"] = state[f"{source}.mlp.c_proj.weight"]
            converted[f"{destination}.mlp.2.bias"] = state[f"{source}.mlp.c_proj.bias"]
        self.load_state_dict(converted, strict=True)


class CLIPVisionLanguageModel(nn.Module):
    """Frozen OpenAI CLIP dual encoder with trainable contexts/adapters outside it."""

    def __init__(
        self,
        visual: PromptedVisionTransformer,
        text: CLIPTextTower,
        visual_projection: torch.Tensor,
        logit_scale: torch.Tensor,
        tokenizer: SimpleCLIPTokenizer,
        class_names: Sequence[str],
    ) -> None:
        super().__init__()
        self.visual = visual
        self.text = text
        self.visual_projection = nn.Parameter(visual_projection.detach().clone(), requires_grad=False)
        self.logit_scale = nn.Parameter(logit_scale.detach().clone(), requires_grad=False)
        self.tokenizer = tokenizer
        self.class_names = [str(name).replace("_", " ") for name in class_names]
        self.freeze_backbone()

    @classmethod
    def from_checkpoint(
        cls,
        visual: PromptedVisionTransformer,
        checkpoint: str | Path,
        class_names: Sequence[str],
        bpe_path: str | Path | None = None,
    ) -> "CLIPVisionLanguageModel":
        path = Path(checkpoint).expanduser()
        raw = PromptedVisionTransformer._load_torch_checkpoint(path)
        state = PromptedVisionTransformer._unwrap_pytorch_state_dict(raw)
        required = {
            "visual.proj", "token_embedding.weight", "positional_embedding",
            "ln_final.weight", "ln_final.bias", "text_projection", "logit_scale",
        }
        missing = sorted(required - set(state))
        if missing:
            raise ValueError(
                f"{path} is not a complete OpenAI CLIP checkpoint; missing: " + ", ".join(missing)
            )
        if visual.backbone_source != "clip":
            raise ValueError("CLIPVisionLanguageModel requires a visual backbone_source='clip'.")
        if tuple(state["visual.proj"].shape)[0] != visual.embed_dim:
            raise ValueError(
                "CLIP visual projection width does not match the selected visual backbone."
            )
        if state["text_projection"].shape[1] != state["visual.proj"].shape[1]:
            raise ValueError("CLIP image and text projection dimensions do not match.")
        visual.load_pretrained_checkpoint(path)
        text_layers = len(
            {
                int(key.split(".")[2])
                for key in state
                if key.startswith("transformer.resblocks.") and key.split(".")[2].isdigit()
            }
        )
        if text_layers <= 0:
            raise ValueError("CLIP checkpoint contains no text transformer blocks.")
        width = int(state["ln_final.weight"].numel())
        tower = CLIPTextTower(
            vocab_size=int(state["token_embedding.weight"].shape[0]),
            context_length=int(state["positional_embedding"].shape[0]),
            width=width,
            layers=text_layers,
            heads=max(width // 64, 1),
            output_dim=int(state["text_projection"].shape[1]),
        )
        tower.load_openai_state(state)
        return cls(
            visual=visual,
            text=tower,
            visual_projection=state["visual.proj"],
            logit_scale=state["logit_scale"],
            tokenizer=SimpleCLIPTokenizer(bpe_path),
            class_names=class_names,
        )

    @property
    def image_width(self) -> int:
        return self.visual.embed_dim

    @property
    def text_width(self) -> int:
        return self.text.width

    @property
    def output_dim(self) -> int:
        return int(self.visual_projection.shape[1])

    def freeze_backbone(self) -> None:
        for parameter in self.visual.parameters():
            parameter.requires_grad = False
        for parameter in self.text.parameters():
            parameter.requires_grad = False

    def encode_image(
        self,
        images: torch.Tensor,
        adapters: Sequence[nn.Module | None] | None = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        features = self.visual.encode(images, adapters=adapters)
        assert isinstance(features, torch.Tensor)
        features = features @ self.visual_projection.to(features)
        return F.normalize(features, dim=-1) if normalize else features

    def prompt_tokens(self, n_ctx: int, class_names: Sequence[str] | None = None) -> torch.Tensor:
        names = self.class_names if class_names is None else [str(name).replace("_", " ") for name in class_names]
        prefix = " ".join(["X"] * int(n_ctx))
        prompts = [f"{prefix} {name}." for name in names]
        return self.tokenizer.tokenize(prompts, context_length=self.text.context_length)

    def encode_text(
        self,
        context: torch.Tensor | None = None,
        class_names: Sequence[str] | None = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        if context is None:
            names = self.class_names if class_names is None else class_names
            prompts = [f"a photo of a {str(name).replace('_', ' ')}." for name in names]
            tokens = self.tokenizer.tokenize(prompts, context_length=self.text.context_length)
        else:
            tokens = self.prompt_tokens(int(context.shape[-2]), class_names)
        features = self.text(tokens.to(self.logit_scale.device), context=context)
        return F.normalize(features, dim=-1) if normalize else features

    def zero_shot_logits(
        self,
        images: torch.Tensor,
        context: torch.Tensor | None = None,
        class_names: Sequence[str] | None = None,
        adapters: Sequence[nn.Module | None] | None = None,
    ) -> torch.Tensor:
        image_features = self.encode_image(images, adapters=adapters)
        text_features = self.encode_text(context=context, class_names=class_names)
        return self.logit_scale.exp().clamp(max=100.0) * image_features @ text_features.t()


class FeaturePromptAdapter(nn.Module):
    """MoAFCL's two-layer intermediate-feature to text-context adapter."""

    def __init__(
        self,
        feature_dim: int,
        text_width: int,
        prompt_length: int = 8,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.text_width = int(text_width)
        self.prompt_length = int(prompt_length)
        self.input = nn.Linear(int(feature_dim), int(hidden_dim))
        self.dropout = nn.Dropout(float(dropout))
        self.output = nn.Linear(int(hidden_dim), self.text_width * self.prompt_length)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        contexts = self.output(self.dropout(F.relu(self.input(features))))
        return contexts.reshape(*features.shape[:-1], self.prompt_length, self.text_width)


def infer_class_names(dataset: object, num_classes: int) -> list[str]:
    """Recover class names through the repository's dataset/subset wrappers."""
    current = dataset
    visited: set[int] = set()
    while hasattr(current, "dataset") and id(current) not in visited:
        visited.add(id(current))
        current = getattr(current, "dataset")
    classes = getattr(current, "classes", None)
    if classes is not None and len(classes) >= int(num_classes):
        return [str(value) for value in list(classes)[: int(num_classes)]]
    samples = getattr(current, "samples", None)
    if samples:
        names: dict[int, str] = {}
        for path, label, *_rest in samples:
            names.setdefault(int(label), Path(str(path)).parent.name)
        if all(label in names for label in range(int(num_classes))):
            return [names[label] for label in range(int(num_classes))]
    return [f"class {index}" for index in range(int(num_classes))]


@torch.no_grad()
def semantic_prompt_repository(
    model: CLIPVisionLanguageModel,
    concepts: Sequence[str],
    size: int,
    prompt_length: int,
    kmeans_steps: int = 25,
) -> torch.Tensor:
    """Fed-Duet semantic repository initialization from concept centroids."""
    if not concepts:
        raise ValueError("At least one concept is required to initialize the prompt repository.")
    tokens = model.tokenizer.tokenize(concepts, context_length=model.text.context_length)
    embeddings = model.text.token_embedding(tokens.to(model.logit_scale.device))
    vectors = embeddings[:, 1:].mean(dim=1).float()
    count = int(size)
    if vectors.shape[0] >= count:
        centers = vectors[torch.linspace(0, vectors.shape[0] - 1, count).long()].clone()
        labels = torch.full((vectors.shape[0],), -1, device=vectors.device, dtype=torch.long)
        for _ in range(max(int(kmeans_steps), 1)):
            new_labels = torch.cdist(vectors, centers).argmin(1)
            if torch.equal(labels, new_labels):
                break
            labels = new_labels
            for index in range(count):
                selected = vectors[labels == index]
                if selected.numel():
                    centers[index] = selected.mean(0)
    else:
        repeats = (count + vectors.shape[0] - 1) // vectors.shape[0]
        centers = vectors.repeat(repeats, 1)[:count]
        centers = centers + 0.01 * torch.randn_like(centers)
    template_tokens = model.tokenizer.tokenize(
        "a photo of a", context_length=model.text.context_length
    ).to(model.logit_scale.device)
    template = model.text.token_embedding(template_tokens)[0, 1 : 1 + int(prompt_length)]
    if template.shape[0] < int(prompt_length):
        padding = template.new_zeros(int(prompt_length) - template.shape[0], template.shape[1])
        template = torch.cat((template, padding))
    prompts = template.unsqueeze(0).repeat(count, 1, 1)
    prompts[:, -1] = centers.to(prompts)
    return prompts

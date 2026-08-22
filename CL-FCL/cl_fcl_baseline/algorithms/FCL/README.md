# FCL implementations

This package contains the paper-oriented FCL implementations reconstructed from
`otherswork/FCL复现`.  Shared tensor operations live in `_common.py`; every method
keeps its client/server protocol in a separate module.

| Method | Main reproduced mechanism |
| --- | --- |
| FedProTIP | Activation subspaces, orthogonal gradient projection, server-side subspace union, task inference |
| FedViT | Private task heads, low-loss class-balanced memories, gradient restoration and integration |
| FedMGP | Frozen ViT, global input prompts, class-wise prefix prompts, proxy-data selective fusion |
| MultiFCL | First-task CLIP adapters, multimodal prototype initialization, multi-scale heads, inverse-KL self-distillation |
| MoAFCL | Intermediate CLIP features, domain-context adapters, image/text logits, clustering, sparse private routing |
| Powder | CODA-style prompts, inter-task correlation, two-stage prompt selection, class-relation distillation |
| Fed-Duet | Semantic prompt repository, loss-aware dispatch, cross-attention fusion, per-layer adapter experts |

FedWeIT, FedKNOW, and Loci are re-exported here.  All FCL runners live directly
under `experiments/FCL/`.

## Paper datasets and default backbones

| Method | Datasets exposed by the original project | Original/common backbone |
| --- | --- | --- |
| FedProTIP | CIFAR-100, DomainNet, ImageNet-R | ViT-B/16 for the ViT experiment; ResNet-18 is also used |
| FedMGP | CIFAR-100, Five-Datasets, Office-Home | `vit_base_patch16_224` |
| MultiFCL | CIFAR-100, Tiny-ImageNet, ImageNet-R, CUB-200 | CLIP vision transformer |
| MoAFCL | Office-Home, DomainNet/DomainNet-Sub, Adaptiope, CIFAR-100, MiniImageNet | CLIP ViT-B/16 |
| Powder | ImageNet-R, DomainNet | ImageNet-21K pretrained ViT-B/16 |
| Fed-Duet | CIFAR-100, Tiny-ImageNet; its README also lists DomainNet and five fine-grained datasets | CLIP ViT-B/16 |
| FedViT | VOC/BSDS500, Rain datasets and GoPro paired restoration data | split CNN + 8-layer, 512-wide transformer |

Powder defaults follow the supplied paper/source configuration: an orthogonally
initialized prompt pool with `M=10`, `L=8`, prompt-tuning in ViT blocks 4--6,
top-3 task transfer, correlation exponent `p=30`, NTD temperature 3, and three
rounds per task switch.  Its default partition is IID because the paper samples
a fixed proportion from every class instead of using a Dirichlet split;
`--partition noniid --noniid-method dirichlet` remains available as an ablation.

The classification FCL runners accept these names through `--dataset`.  The six
classification methods above default to `--model ViTBasePatch16 --image-size 224
--vit-patch-size 16`.  `ViT-B/16`, `vitb16`, and `vit_base_patch16_224` are accepted
aliases. MoAFCL, MultiFCL, and Fed-Duet default to a complete OpenAI CLIP
ViT-B/16 checkpoint at `experiments/checkpoint/ViT-B-16.pt`. The loader restores
the visual tower, `visual.proj`, text transformer, `text_projection`, tokenizer,
and learned temperature. MultiFCL uses text semantics for prototype
initialization; MoAFCL and Fed-Duet retain the text tower in their training
objectives. Set `CLIP_BPE_PATH` or `--clip-bpe-path` when the BPE vocabulary is
not available in the reference-source tree.

Faithful schedules are algorithm-specific: MultiFCL defaults to 10 clients, 5
local epochs and 5 rounds per task; Fed-Duet uses 5 clients, 1 local epoch and
10 rounds; MoAFCL uses 10 clients, one aggregation per task and 500 server-gate
epochs. MoAFCL and Fed-Duet also expose `--scenario domain` for folder datasets
laid out as `domain/class/image`.

Example launches (all three expect the full CLIP checkpoint):

```bash
python -m cl_fcl_baseline.experiments.FCL.run_MultiFCL \
  --dataset cub200 --backbone-checkpoint checkpoints/ViT-B-16.pt

python -m cl_fcl_baseline.experiments.FCL.run_MoAFCL \
  --dataset officehome --scenario domain --data-dir /path/to/data \
  --backbone-checkpoint checkpoints/ViT-B-16.pt

python -m cl_fcl_baseline.experiments.FCL.run_FedDuet \
  --dataset cifar100 --scenario class --classes-per-task 10 \
  --backbone-checkpoint checkpoints/ViT-B-16.pt
```

## FedViT reference note

The supplied `FedViT.pdf` and source tree both describe **Federated Split Vision
Transformer for Task-Agnostic Privacy-Preserving Vision Tasks**, an image-restoration
method rather than classification FCL.  The current `run_FedViT.py` is retained as
a classification-compatible experimental entry, but selecting VOC/Rain/GoPro in
that runner would be misleading: faithful support requires a separate paired-image,
image-to-image training and evaluation pipeline.

## Backbones

Other prompt-based methods retain the repository-native ViT path. The three VLM
methods fail early when the complete CLIP checkpoint is absent or contains only
a visual tower; silently using a random text tower is not a valid reproduction.

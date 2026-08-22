from __future__ import annotations

import argparse
from pathlib import Path

from cl_fcl_baseline.datasets import DATASET_NAMES, normalize_dataset_name
from cl_fcl_baseline.models import (
    DEFAULT_VIT_ATTENTION_DROPOUT,
    DEFAULT_VIT_DROPOUT,
    DEFAULT_VIT_MLP_RATIO,
    DEFAULT_VIT_PATCH_SIZE,
    MODEL_NAMES,
    normalize_model_name,
)

CLASSIFICATION_DATASETS = ("random_classification", *DATASET_NAMES)


def _normalize_classification_dataset(name: str) -> str:
    if str(name).lower() == "random_classification":
        return "random_classification"
    return normalize_dataset_name(name)






def _add_common_fl_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--client-sample-ratio", type=float, default=0.6, help="fraction of clients sampled each round (0, 1].")
    parser.add_argument("--device", type=str, default="auto", help="cpu | cuda | cuda:0 | cuda:1 | auto")
    parser.add_argument("--partition", type=str, default="noniid", choices=["iid", "noniid"])
    parser.add_argument("--noniid-method", type=str, default="dirichlet", choices=["shards", "dirichlet"], help="noniid partitioning strategy when --partition noniid is selected.",)
    parser.add_argument("--noniid-shards", type=int, default=2)
    parser.add_argument("--dirichlet-beta", type=float, default=0.5)
    parser.add_argument("--num-rounds", type=int, default=200)
    parser.add_argument(
        "--local-epochs",
        "--local_epochs",
        dest="local_epochs",
        type=int,
        default=10,
    )
    parser.add_argument("--batch-size", type=int, default=128, help="<=0 for full-batch")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam"])
    parser.add_argument("--dataset", type=normalize_dataset_name, default="cifar100", choices=DATASET_NAMES)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument(
        "--image-size",
        type=int,
        default=32,
        help="input image size; ViT-B/16 paper configurations use 224",
    )
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="download torchvision datasets when missing; folder datasets are always local",
    )
    parser.add_argument("--num-samples", type=int, default=0, help="<=0 for full-dataset")
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument(
        "--model",
        type=normalize_model_name,
        default="ResNet32",
        choices=MODEL_NAMES,
    )
    parser.add_argument("--hidden-dim", type=int, default=200)
    parser.add_argument("--vit-patch-size", type=int, default=DEFAULT_VIT_PATCH_SIZE, help="patch size used by ViT backbones; must divide the input height and width")
    parser.add_argument("--vit-dropout", type=float, default=DEFAULT_VIT_DROPOUT, help="dropout applied to ViT token embeddings and MLP blocks")
    parser.add_argument("--vit-attention-dropout", type=float, default=DEFAULT_VIT_ATTENTION_DROPOUT, help="dropout applied inside ViT multi-head self-attention")
    parser.add_argument("--vit-mlp-ratio", type=float, default=DEFAULT_VIT_MLP_RATIO, help="hidden expansion ratio used by ViT MLP blocks")
    parser.add_argument("--show-progress", action="store_true",default=False)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--log-file", type=str, default="")


def build_fedavg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a FedAvg baseline.")
    _add_common_fl_args(parser)
    parser.add_argument("--algorithm", type=str, default="fedavg", choices=["fedavg"])
    return parser


# def build_fl_parser() -> argparse.ArgumentParser:
#     return build_fedavg_parser()


def build_fedprox_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a FedProx baseline.")
    _add_common_fl_args(parser)
    parser.add_argument("--algorithm", type=str, default="fedprox", choices=["fedprox"])
    parser.add_argument("--prox-mu", type=float, default=0.01, help="FedProx proximal term coefficient (mu).")
    return parser


def build_fedkemf_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a FedKEMF baseline.")
    _add_common_fl_args(parser)
    parser.add_argument("--algorithm", type=str, default="fedkemf", choices=["fedkemf"])
    parser.add_argument("--distill", action="store_true", default=True, help="enable client-side distillation")
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--distill-alpha", type=float, default=0.5)
    parser.add_argument("--mutual-learning", action="store_true", default=True, help="enable deep mutual learning")
    parser.add_argument("--server-distill-epochs", type=int, default=10)
    parser.add_argument("--server-distill-lr", type=float, default=0.001)
    parser.add_argument("--server-distill-temperature", type=float, default=1.0)
    parser.add_argument("--server-ensemble", type=str, default="max", choices=["max", "mean"])
    parser.add_argument("--server-data-ratio", type=float, default=0.6, help="fraction of the training dataset used as the server public set (0, 1].",)
    return parser


def build_scaffold_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a SCAFFOLD baseline.")
    _add_common_fl_args(parser)
    parser.add_argument("--algorithm", type=str, default="scaffold", choices=["scaffold"])
    parser.add_argument("--global-lr", type=float, default=1.0, help="global step size (eta_g) for SCAFFOLD")
    return parser


def build_moon_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a MOON baseline.")
    _add_common_fl_args(parser)
    parser.add_argument("--algorithm", type=str, default="moon", choices=["moon"])
    parser.add_argument("--moon-temperature", type=float, default=0.5, help="temperature tau for contrastive loss")
    parser.add_argument("--moon-mu", type=float, default=1.0, help="weight mu for the contrastive loss term")
    return parser


def build_fat_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Federated Adversarial Training baseline.")
    _add_common_fl_args(parser)
    parser.add_argument("--algorithm", type=str, default="fat", choices=["fat"])
    _add_pgd_args(parser)
    parser.add_argument("--fat-adversarial-ratio", type=float, default=0.5, help="proportion of each local minibatch replaced by PGD adversarial examples after warmup")
    parser.add_argument("--fat-warmup-rounds", type=int, default=0, help="number of initial rounds trained with --fat-warmup-adversarial-ratio")
    parser.add_argument("--fat-warmup-adversarial-ratio", type=float, default=0.1, help="adversarial minibatch proportion used during FAT warmup rounds")
    return parser


def build_sfat_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Slack Federated Adversarial Training baseline.")
    _add_common_fl_args(parser)
    parser.add_argument("--algorithm", type=str, default="sfat", choices=["sfat"])
    _add_pgd_args(parser)
    parser.add_argument("--sfat-adversarial-ratio", type=float, default=0.5, help="proportion of each local minibatch replaced by PGD adversarial examples after warmup")
    parser.add_argument("--sfat-warmup-rounds", type=int, default=0, help="number of initial rounds trained with --sfat-warmup-adversarial-ratio")
    parser.add_argument("--sfat-warmup-adversarial-ratio", type=float, default=0.1, help="adversarial minibatch proportion used during SFAT warmup rounds")
    parser.add_argument("--sfat-alpha", type=float, default=1.0 / 11.0, help="SFAT alpha-slack value in [0, 1); top clients use (1 + alpha) / (1 - alpha)")
    parser.add_argument("--sfat-enhanced-clients", type=int, default=1, help="number of low adversarial-loss clients upweighted in each aggregation")
    parser.add_argument("--sfat-loss-metric", type=str, default="adv_ce_loss", choices=["adv_ce_loss", "ce_loss", "loss"], help="client metric used for SFAT ascending loss ranking")
    return parser


def build_calfat_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Calibrated Federated Adversarial Training baseline.")
    _add_common_fl_args(parser)
    parser.add_argument("--algorithm", type=str, default="calfat", choices=["calfat"])
    _add_pgd_args(parser)
    parser.add_argument("--calfat-prior-smoothing", type=float, default=1e-6, help="small positive constant delta added to each client class prior for calibrated logits")
    return parser


def build_rbn_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a FedRBN-style FL_robust federated learning baseline.")
    _add_common_fl_args(parser)
    parser.add_argument("--algorithm", type=str, default="rbn", choices=["rbn"])
    _add_pgd_args(parser)
    parser.add_argument("--rbn-at-ratio", type=float, default=0.2, help="fraction of clients treated as AT users; FedRBN is designed for mixed AT/ST clients, and the rest receive propagated BNa")
    parser.add_argument("--rbn-adv-lambda", type=float, default=0.5, help="weight on the adversarial loss term for AT users")
    parser.add_argument("--rbn-src-weight-mode", type=str, default="cos", choices=["eq", "cos"], help="source-client weighting used to propagate BNa from AT users to ST users")
    parser.add_argument("--rbn-pnc", type=float, default=0.5, help="coefficient of the pseudo-noise calibration loss for ST users; <0 disables PNC")
    parser.add_argument("--rbn-pnc-warmup", type=int, default=10, help="number of initial rounds using zero PNC coefficient")
    parser.add_argument("--rbn-attack-noised-bn", action=argparse.BooleanOptionalAction, default=True, help="use the noised BN path while generating PGD adversarial examples for AT users")
    return parser


def build_sylva_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Sylva-inspired personalized adversarial fine-tuning baseline.")
    _add_common_fl_args(parser)
    parser.add_argument("--algorithm", type=str, default="sylva", choices=["sylva"])
    _add_pgd_args(parser)
    parser.add_argument("--sylva-class-balance-power", type=float, default=0.6, help="inverse-frequency exponent for Sylva's class-balanced local loss")
    parser.add_argument("--sylva-class-balance-smoothing", type=float, default=1e-3, help="small positive constant added to local class counts before weighting")
    parser.add_argument("--sylva-dynamic-rounds", type=int, default=3, help="number of rounds used to ramp class weights from uniform to local imbalance-aware")
    parser.add_argument("--sylva-clean-weight", type=float, default=0.8, help="weight applied to Sylva's clean cross-entropy term")
    parser.add_argument("--sylva-adv-weight", type=float, default=1.25, help="weight applied to Sylva's adversarial cross-entropy term")
    parser.add_argument("--sylva-kl-weight", type=float, default=8.0, help="weight applied to Sylva's TRADES-style KL consistency term")
    parser.add_argument("--sylva-global-reg", type=float, default=1e-4, help="coefficient for Sylva's local-to-global alignment penalty")
    parser.add_argument("--sylva-agg-temperature", type=float, default=0.7, help="temperature scaling used by Sylva's similarity-aware aggregation")
    parser.add_argument("--sylva-agg-neighbors", type=int, default=2, help="number of nearest clients used for Sylva similarity scoring; <=0 uses all peers")
    parser.add_argument("--sylva-phase2-epochs", type=int, default=10, help="local benign refinement epochs for Sylva phase 2 after each round")
    parser.add_argument("--sylva-phase2-topk-layers", type=int, default=1, help="number of layer groups selected for Sylva phase 2 benign refinement")
    parser.add_argument("--sylva-phase2-tradeoff", type=float, default=0.7, help="penalty weight for adversarial sensitivity when scoring Sylva phase 2 layer groups")
    parser.add_argument("--sylva-phase2-lr-scale", type=float, default=0.0015, help="multiplier applied to the base learning rate during Sylva phase 2 benign refinement")
    parser.add_argument("--sylva-phase2-max-batches", type=int, default=10, help="maximum batches used for Sylva phase 2 layer scoring and benign refinement; <=0 uses all batches")
    return parser






def _add_common_cl_args(
    *,
    description: str,
    default_model: str,
    hidden_dim: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    scenario: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="cpu | cuda | cuda:0 | auto")
    parser.add_argument("--scenario", type=str, default=scenario, choices=["class", "task", "domain"])
    parser.add_argument(
        "--dataset",
        type=_normalize_classification_dataset,
        default="cifar100",
        choices=CLASSIFICATION_DATASETS,
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--num-samples", type=int, default=0, help="<=0 for full-dataset")
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--input-shape", type=int, nargs=3, default=[1, 28, 28])
    parser.add_argument(
        "--image-size",
        type=int,
        default=32,
        help="input image size; ViT-B/16 paper configurations use 224",
    )
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="download torchvision datasets when missing; folder datasets are always local",
    )
    parser.add_argument("--num-tasks", type=int, default=0)
    parser.add_argument("--classes-per-task", type=int, default=10)
    parser.add_argument(
        "--task-order-shuffle",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--batch-size", type=int, default=batch_size, help="<=0 for full-batch")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument(
        "--model",
        type=normalize_model_name,
        default=default_model,
        choices=MODEL_NAMES,
    )
    parser.add_argument("--hidden-dim", type=int, default=hidden_dim)
    parser.add_argument("--vit-patch-size", type=int, default=DEFAULT_VIT_PATCH_SIZE, help="patch size used by ViT backbones; must divide the input height and width")
    parser.add_argument("--vit-dropout", type=float, default=DEFAULT_VIT_DROPOUT, help="dropout applied to ViT token embeddings and MLP blocks")
    parser.add_argument("--vit-attention-dropout", type=float, default=DEFAULT_VIT_ATTENTION_DROPOUT, help="dropout applied inside ViT multi-head self-attention")
    parser.add_argument("--vit-mlp-ratio", type=float, default=DEFAULT_VIT_MLP_RATIO, help="hidden expansion ratio used by ViT MLP blocks")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=weight_decay)
    parser.add_argument("--log-file", type=str, default="")
    parser.add_argument("--log-each-epoch", action="store_true", default=False)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="evaluate every N local epochs; <=0 evaluates only at task end",
    )
    return parser


def build_ewc_parser() -> argparse.ArgumentParser:
    parser = _add_common_cl_args(
        description="Run an EWC continual-learning baseline.",
        default_model="mlp",
        hidden_dim=400,
        batch_size=64,
        epochs=20,
        lr=0.001,
        weight_decay=0.0,
        scenario="task",
    )
    # Kirkpatrick et al. evaluate supervised EWC on Permuted-MNIST: every
    # task has the same ten labels and a different fixed pixel permutation.
    # Class-incremental CIFAR-100 without replay introduces output-layer bias
    # that parameter regularisation alone is not designed to solve.
    parser.set_defaults(dataset="mnist")
    parser.add_argument("--algorithm", type=str, default="ewc", choices=["ewc"])
    parser.add_argument("--ewc-lambda", type=float, default=400.0)
    parser.add_argument("--fisher-samples", type=int, default=200)
    parser.add_argument("--fisher-mode", type=str, default="sampled", choices=["sampled", "empirical"])
    return parser


def build_gem_parser() -> argparse.ArgumentParser:
    parser = _add_common_cl_args(
        description="Run a GEM continual-learning baseline.",
        default_model="ResNet32",
        hidden_dim=100,
        batch_size=64,
        epochs=20,
        lr=0.001,
        weight_decay=0.0,
        scenario="task",
    )
    parser.add_argument("--algorithm", type=str, default="gem", choices=["gem"])
    parser.add_argument("--memory-size", type=int, default=256, help="episodic examples per task")
    parser.add_argument("--memory-strength", type=float, default=0.5, help="GEM gamma")
    parser.add_argument("--qp-eps", type=float, default=1e-3)
    return parser


def build_lwf_parser() -> argparse.ArgumentParser:
    parser = _add_common_cl_args(
        description="Run a Learning without Forgetting baseline.",
        default_model="ResNet32",
        hidden_dim=200,
        batch_size=128,
        epochs=30,
        lr=0.001,
        weight_decay=5e-4,
        # The original LwF classifier has one softmax head per task and is
        # evaluated with the task identity.  Class-incremental LwF.MC is a
        # different loss/evaluation protocol (Rebuffi et al., 2017).
        scenario="task",
    )
    parser.add_argument("--algorithm", type=str, default="lwf", choices=["lwf"])
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--distillation-weight", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    return parser


def build_icarl_parser() -> argparse.ArgumentParser:
    parser = _add_common_cl_args(
        description="Run an iCaRL continual-learning baseline.",
        default_model="ResNet32",
        hidden_dim=200,
        batch_size=128,
        epochs=30,
        lr=2.0,
        weight_decay=1e-5,
        scenario="class",
    )
    parser.add_argument("--algorithm", type=str, default="icarl", choices=["icarl"])
    parser.add_argument("--memory-budget", type=int, default=2000, help="fixed total exemplar budget K")
    parser.add_argument("--lr-decay", type=float, default=5.0)
    parser.add_argument(
        "--prototype-flip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="average original/flipped exemplar features for CIFAR NME prototypes",
    )
    # The paper evaluates after each class batch, not after every epoch.
    parser.set_defaults(task_order_shuffle=True)
    return parser





def _add_cl_robust_attack_args(
    parser: argparse.ArgumentParser,
    *,
    train_steps: int,
    step_size: float = 2.0 / 255.0,
) -> None:
    _add_pgd_args(parser)
    parser.set_defaults(
        pgd_epsilon=8.0 / 255.0,
        pgd_step_size=step_size,
        pgd_steps=train_steps,
        pgd_random_start=True,
        pgd_max_batches=0,
    )
    parser.add_argument(
        "--eval-pgd-steps",
        type=int,
        default=10,
        help="number of PGD steps used for robust continual-learning evaluation",
    )
    parser.add_argument(
        "--eval-pgd-step-size",
        type=float,
        default=2.0 / 255.0,
        help=(
            "PGD step size used only for evaluation; keeping this independent "
            "from the method-specific training attack makes robust results comparable"
        ),
    )


def build_taba_parser() -> argparse.ArgumentParser:
    parser = _add_common_cl_args(
        description="Run Task-Aware Boundary Augmentation (TABA).",
        default_model="ResNet18",
        hidden_dim=200,
        batch_size=128,
        epochs=30,
        lr=0.01,
        weight_decay=1e-5,
        scenario="class",
    )
    # Setting I in the paper always uses five equal class increments.  A zero
    # classes-per-task value is resolved from the selected dataset at runtime,
    # so both CIFAR-10 (2 classes/task) and CIFAR-100 (20 classes/task) work.
    # parser.set_defaults(num_tasks=5, classes_per_task=0, eval_every=0)
    parser.add_argument("--algorithm", type=str, default="taba", choices=["taba"])
    parser.add_argument("--memory-budget", type=int, default=2000)
    parser.add_argument(
        "--replay-batch-size",
        type=int,
        default=0,
        help="deprecated for TABA; replay is sampled from X_new union memory_old as in Algorithm 1",
    )
    parser.add_argument("--taba-mix-batch-size", type=int, default=0, help="m'; 0 matches stream batch size")
    parser.add_argument("--taba-lambda-min", type=float, default=0.45)
    parser.add_argument("--taba-lambda-max", type=float, default=0.55)
    _add_cl_robust_attack_args(parser, train_steps=10)
    parser.set_defaults(eval_pgd_steps=10, eval_pgd_step_size=2.0 / 255.0)
    return parser


def build_daml_parser() -> argparse.ArgumentParser:
    parser = _add_common_cl_args(
        description="Run Distillation and Additional Memory-data Loss (DAML).",
        default_model="ResNet18",
        hidden_dim=200,
        batch_size=128,
        epochs=30,
        lr=5e-4,
        weight_decay=0.0,
        scenario="class",
    )
    # Mukai et al. optimize DAML with Adam.  Falling back to the common CL
    # SGD default is especially harmful here because unified BCE averages
    # over every active output (10 already in CIFAR-100 task 0), shrinking
    # each classifier-row gradient before any continual-learning component
    # is involved.
    parser.set_defaults(optimizer="adam")
    parser.add_argument("--algorithm", type=str, default="daml", choices=["daml"])
    parser.add_argument("--memory-budget", type=int, default=2000)
    parser.add_argument("--replay-batch-size", type=int, default=0, help="0 matches stream batch size")
    parser.add_argument("--daml-alpha", type=float, default=0.2, help="additional memory CE weight")
    _add_cl_robust_attack_args(parser, train_steps=10)
    parser.set_defaults(eval_pgd_steps=10, eval_pgd_step_size=2.0 / 255.0)
    return parser


def build_flair_parser() -> argparse.ArgumentParser:
    parser = _add_common_cl_args(
        description="Run FLatness-preserving Adversarial Incremental Learning (FLAIR).",
        default_model="ResNet18",
        hidden_dim=200,
        batch_size=32,
        epochs=30,
        lr=0.1,
        weight_decay=0.0,
        scenario="class",
    )
    # parser.set_defaults(dataset="cifar10", classes_per_task=2, num_tasks=5, eval_every=0)
    parser.add_argument("--algorithm", type=str, default="flair", choices=["flair"])
    parser.add_argument("--memory-budget", type=int, default=2000)
    parser.add_argument("--replay-batch-size", type=int, default=0, help="0 matches stream batch size")
    parser.add_argument("--flair-alpha", type=float, default=1.0, help="ADSL distillation weight")
    parser.add_argument("--flair-beta", type=float, default=1.0, help="FPD weight")
    _add_cl_robust_attack_args(parser, train_steps=10)
    return parser


def build_aflc_raer_parser() -> argparse.ArgumentParser:
    parser = _add_common_cl_args(
        description="Run AFLC with Robustness-Aware Experience Replay (RAER).",
        default_model="ResNet18",
        hidden_dim=200,
        batch_size=32,
        # AFLC's training-only margin needs the ER+AT horizon used by Mi et
        # al.; their CIFAR protocol trains every task for 50 epochs.
        epochs=30,
        lr=0.1,
        weight_decay=0.0,
        scenario="class",
    )
    # parser.set_defaults(dataset="cifar10", classes_per_task=2, num_tasks=5, eval_every=0)
    parser.add_argument("--algorithm", type=str, default="aflc_raer", choices=["aflc_raer"])
    parser.add_argument("--memory-budget", type=int, default=2000)
    parser.add_argument("--replay-batch-size", type=int, default=0, help="0 matches stream batch size")
    parser.add_argument("--aflc-alpha", type=float, default=3.5)
    parser.add_argument("--raer-threshold", type=int, default=5)
    parser.add_argument(
        "--future-prior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply AFLC's future-head prior from Eq. (10)",
    )
    parser.add_argument(
        "--adaptive-eval-attack",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply AFLC inside PGD during adaptive robustness evaluation",
    )
    _add_cl_robust_attack_args(parser, train_steps=10)
    return parser







def _add_common_fcl_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a minimal FCL baseline.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--client-sample-ratio", type=float, default=1.0,
                        help="fraction of clients sampled each round (0, 1].")
    parser.add_argument("--device", type=str, default="auto", help="cpu | cuda | cuda:0 | auto")
    parser.add_argument("--partition", type=str, default="noniid", choices=["iid", "noniid"])
    parser.add_argument("--noniid-method", type=str, default="dirichlet", choices=["shards", "dirichlet"],
                        help="noniid partitioning strategy when --partition noniid is selected.", )
    parser.add_argument("--noniid-shards", type=int, default=2)
    parser.add_argument("--dirichlet-beta", type=float, default=0.5)
    parser.add_argument("--num-rounds", type=int, default=100)
    parser.add_argument("--num-tasks", type=int, default=0)
    parser.add_argument("--task-order-shuffle", action="store_true", default=False)
    parser.add_argument(
        "--heterogeneous-task-order",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "give every client an independently shuffled task stream; "
            "disabled by default so all clients follow the shared task order"
        ),
    )
    parser.add_argument(
        "--heterogeneous-eval-mode",
        type=str,
        default="position",
        choices=["position", "task"],
        help=(
            "evaluation grouping used only with --heterogeneous-task-order: "
            "position averages each client's task at the same arrival position "
            "(the 0718 protocol), while task groups by the actual task ID"
        ),
    )
    parser.add_argument("--classes-per-task", type=int, default=20)
    parser.add_argument("--rounds-per-task", type=int, default=10)
    parser.add_argument(
        "--local-epochs",
        "--local_epochs",
        dest="local_epochs",
        type=int,
        default=10,
    )
    parser.add_argument("--batch-size", type=int, default=64, help="<=0 for full-batch")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam"])
    parser.add_argument(
        "--dataset",
        type=_normalize_classification_dataset,
        default="imagenetr",
        choices=CLASSIFICATION_DATASETS,
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--num-samples", type=int, default=0, help="<=0 for full-dataset")
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--input-shape", type=int, nargs=3, default=[1, 28, 28])
    parser.add_argument(
        "--image-size",
        type=int,
        default=32,
        help="input image size; ViT-B/16 paper configurations use 224",
    )
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="download torchvision datasets when missing; folder datasets are always local",
    )
    parser.add_argument(
        "--model",
        type=normalize_model_name,
        default="ResNet32",
        choices=MODEL_NAMES,
    )
    parser.add_argument("--hidden-dim", type=int, default=200)
    parser.add_argument("--vit-patch-size", type=int, default=DEFAULT_VIT_PATCH_SIZE, help="patch size used by ViT backbones; must divide the input height and width")
    parser.add_argument("--vit-dropout", type=float, default=DEFAULT_VIT_DROPOUT, help="dropout applied to ViT token embeddings and MLP blocks")
    parser.add_argument("--vit-attention-dropout", type=float, default=DEFAULT_VIT_ATTENTION_DROPOUT, help="dropout applied inside ViT multi-head self-attention")
    parser.add_argument("--vit-mlp-ratio", type=float, default=DEFAULT_VIT_MLP_RATIO, help="hidden expansion ratio used by ViT MLP blocks")
    parser.add_argument("--show-progress", action="store_true", default=False)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--log-file", type=str, default="")
    return parser

def build_fedweit_parser() -> argparse.ArgumentParser:
    parser = _add_common_fcl_args()
    parser.description = "Run a FedWeIT baseline."
    parser.add_argument("--algorithm", type=str, default="fedweit", choices=["fedweit"])
    parser.add_argument("--lambda1", type=float, default=0.0005, help="FedWeIT lambda_l1 for adaptive parameters.")
    parser.add_argument("--lambda2", type=float, default=30.0, help="FedWeIT lambda_l2 retroactive coefficient.")
    parser.add_argument("--lambda-mask", type=float, default=0.0, help="FedWeIT lambda_mask for raw task masks.")
    parser.add_argument("--kb-sample-size", type=int, default=0, help="knowledge-base samples per task; <=0 means all.")
    parser.add_argument("--mask-init", type=float, default=-1.0, help="initial sigmoid mask value; <0 uses random raw-mask init like the reference implementation.")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help=argparse.SUPPRESS)
    parser.add_argument("--client-sparsity", type=float, default=0.3, help="fraction of smallest raw-mask entries pruned from communicable B*m.")
    parser.add_argument("--adaptive-threshold", type=float, default=-1.0, help="hard threshold for communicable A; <0 uses lambda1.")
    return parser



def build_fedknow_parser() -> argparse.ArgumentParser:
    parser = _add_common_fcl_args()
    parser.description = "Run a FedKNOW baseline."
    parser.add_argument("--algorithm", type=str, default="fedknow", choices=["fedknow"])
    parser.add_argument("--knowledge-ratio", type=float, default=0.1, help="FedKNOW rho: ratio of top-magnitude weights retained as signature task knowledge.")
    parser.add_argument("--signature-k", type=int, default=10, help="number of most dissimilar signature task gradients integrated for each batch.")
    parser.add_argument("--integrator-steps", type=int, default=100, help="projected-gradient steps used to solve FedKNOW's dual QP gradient integrator.")
    parser.add_argument("--knowledge-finetune-epochs", type=int, default=2, help="PackNet fine-tuning epochs after pruning; the reference FedKNOW code uses 2.")
    parser.add_argument("--post-aggregation-epochs", type=int, default=1, help="one local fine-tuning epoch after global aggregation, as specified in FedKNOW Section III-A.")
    parser.add_argument("--distillation-warmup-epochs", type=int, default=2, help="dense old-model distillation epochs before FedKNOW representation integration; the reference code uses 2.")
    parser.add_argument("--restorer-loss", type=str, default="soft", choices=["hard", "soft"], help="loss used by the gradient restorer: hard pseudo labels or soft KL targets from retained knowledge.")
    parser.add_argument("--restorer-temperature", type=float, default=2.0, help="temperature used for FedKNOW soft distillation in the gradient restorer.")
    return parser


def build_loci_parser() -> argparse.ArgumentParser:
    parser = _add_common_fcl_args()
    parser.description = "Run the Loci task-grained FCL algorithm."
    # GEM projects the SGD update direction. Adam or momentum would transform
    # that direction after projection and no longer implement GEM's constraint.
    parser.set_defaults(
        heterogeneous_task_order=True,
        classes_per_task=0,
        optimizer="sgd",
    )
    parser.add_argument("--algorithm", type=str, default="loci", choices=["loci"])
    parser.add_argument("--loci-kd-model", type=normalize_model_name, default="SixCNN", choices=MODEL_NAMES, help="common compact model exchanged by heterogeneous Loci clients")
    parser.add_argument("--loci-client-models", type=normalize_model_name, nargs="+", default=[], choices=MODEL_NAMES, help="optional heterogeneous main-model list, cycled across clients")
    parser.add_argument("--loci-client-lrs", type=float, nargs="+", default=[], help="optional per-client learning rates, cycled across clients")
    parser.add_argument("--loci-client-local-epochs", type=int, nargs="+", default=[], help="optional per-client local epoch counts, cycled across clients")
    parser.add_argument("--loci-kd-hidden-dim", type=int, default=128)
    parser.add_argument("--loci-kd-epochs", type=int, default=1)
    parser.add_argument("--loci-kd-lr", type=float, default=0.001)
    parser.add_argument("--loci-temperature", type=float, default=2.0)
    parser.add_argument("--loci-kd-alpha", type=float, default=0.5, help="weight of teacher KL in local KD-model extraction")
    parser.add_argument("--loci-integrator-weight", type=float, default=1.0, help="coefficient multiplying the accuracy-weighted KD gradient in Eq. (7)")
    parser.add_argument("--loci-continual-method", type=str, default="gem", choices=["gem", "ewc"], help="client-side continual learner; the Loci paper evaluates GEM")
    parser.add_argument("--loci-ewc-lambda", type=float, default=100.0)
    parser.add_argument("--loci-weight-decay", type=float, default=0.0)
    parser.add_argument("--loci-fisher-batches", type=int, default=0, help="batches used for EWC Fisher estimation; <=0 uses the complete task partition")
    parser.add_argument("--loci-gem-memory-size", type=int, default=256, help="GEM episodic examples retained per task")
    parser.add_argument("--loci-gem-memory-strength", type=float, default=0.5, help="GEM projection margin")
    parser.add_argument("--loci-gem-qp-eps", type=float, default=1e-3, help="diagonal stabilizer for GEM's dual QP")
    parser.add_argument("--loci-knowledge-ratio", type=float, default=0.05, help="fraction of largest-magnitude KD weights retained as task knowledge")
    parser.add_argument("--loci-knowledge-finetune-epochs", type=int, default=1)
    parser.add_argument("--loci-similar-tasks", type=int, default=4)
    parser.add_argument("--loci-similarity", type=str, default="activation", choices=["activation", "weight"])
    parser.add_argument("--loci-selector-candidates", type=int, default=20, help="weight-index shortlist size before activation scoring; <=0 scans all knowledge")
    parser.add_argument("--loci-selector-batches", type=int, default=5, help="server public batches used by activation similarity; <=0 uses all")
    parser.add_argument("--loci-public-samples", type=int, default=64, help="public samples retained per task for activation similarity; <=0 uses all")
    parser.add_argument("--loci-ot-regularization", type=float, default=0.05)
    parser.add_argument("--loci-ot-iterations", type=int, default=20)
    parser.add_argument("--loci-image-size", type=int, default=32)
    return parser


def _set_vit_b16_defaults(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(
        model="ViTBasePatch16",
        optimizer="adam",
        image_size=224,
        vit_patch_size=16,
    )


def _add_prompt_backbone_args(parser: argparse.ArgumentParser) -> None:
    _set_vit_b16_defaults(parser)
    parser.add_argument(
        "--backbone-source",
        type=str,
        choices=["vit", "clip"],
        default="vit",
        help=(
            "checkpoint architecture: vit loads repository/Google ViT weights; "
            "clip loads an original OpenAI CLIP ViT visual tower"
        ),
    )
    parser.add_argument("--fcl-embed-dim", type=int, default=768)
    parser.add_argument("--fcl-depth", type=int, default=12)
    parser.add_argument("--fcl-num-heads", type=int, default=12)
    parser.add_argument("--fcl-adapter-dim", type=int, default=64)
    parser.add_argument("--fcl-prompt-pool-size", type=int, default=10)
    parser.add_argument("--fcl-prompt-length", type=int, default=5)
    parser.add_argument("--fcl-prompt-top-k", type=int, default=1)
    parser.add_argument(
        "--backbone-checkpoint",
        type=str,
        default=str(Path(__file__).resolve().parent / "checkpoint" / "ViT-B_16.npz"),
        help=(
            "pretrained PromptedVisionTransformer state dict, official Google ViT "
            ".npz, or OpenAI CLIP .pt selected by --backbone-source; VLM runners "
            "also restore CLIP's text tower, projections and temperature"
        ),
    )


def _set_clip_vlm_defaults(parser: argparse.ArgumentParser) -> None:
    """Defaults for methods whose published objective requires both CLIP towers."""
    parser.set_defaults(
        backbone_source="clip",
        backbone_checkpoint=str(
            Path(__file__).resolve().parent / "checkpoint" / "ViT-B-16.pt"
        ),
    )
    parser.add_argument(
        "--clip-bpe-path",
        type=str,
        default="",
        help="optional path to OpenAI bpe_simple_vocab_16e6.txt.gz",
    )


def build_fedprotip_parser() -> argparse.ArgumentParser:
    parser = _add_common_fcl_args()
    parser.description = "Run FedProTIP with local replay-free gradient projection."
    _add_prompt_backbone_args(parser)
    parser.set_defaults(lr=1e-3)
    parser.add_argument("--algorithm", default="fedprotip", choices=["fedprotip"])
    parser.add_argument("--fedprotip-threshold", type=float, default=0.7)
    parser.add_argument("--fedprotip-activation-batches", type=int, default=5)
    parser.add_argument("--fedprotip-activation-columns", type=int, default=512)
    return parser


def build_fedvit_parser() -> argparse.ArgumentParser:
    parser = _add_common_fcl_args()
    parser.description = (
        "Run the classification-compatible FedViT entry. The original FedViT paper "
        "is an image-restoration split-transformer and requires a paired-image pipeline."
    )
    _add_prompt_backbone_args(parser)
    parser.set_defaults(lr=5e-4, optimizer="adam")
    parser.add_argument("--algorithm", default="fedvit", choices=["fedvit"])
    parser.add_argument("--fedvit-head-epochs", type=int, default=1)
    parser.add_argument("--fedvit-post-aggregation-epochs", type=int, default=1)
    parser.add_argument("--fedvit-knowledge-ratio", type=float, default=0.1)
    parser.add_argument("--fedvit-signature-k", type=int, default=10)
    parser.add_argument("--fedvit-integrator-steps", type=int, default=100)
    parser.add_argument("--fedvit-weight-decay", type=float, default=0.05)
    return parser


def build_fedmgp_parser() -> argparse.ArgumentParser:
    parser = _add_common_fcl_args()
    parser.description = "Run FedMGP multi-granularity prompting."
    _add_prompt_backbone_args(parser)
    parser.set_defaults(lr=1e-3)
    parser.add_argument("--algorithm", default="fedmgp", choices=["fedmgp"])
    parser.add_argument("--fedmgp-pull-constraint", type=float, default=0.1)
    parser.add_argument("--fedmgp-local-prompt-length", type=int, default=5)
    parser.add_argument("--fedmgp-local-prompt-layers", type=int, default=3)
    parser.add_argument("--fedmgp-fusion-epochs", type=int, default=1)
    parser.add_argument("--fedmgp-fusion-lr", type=float, default=1e-3)
    parser.add_argument("--server-public-samples", type=int, default=64)
    return parser


def build_moafcl_parser() -> argparse.ArgumentParser:
    parser = _add_common_fcl_args()
    parser.description = "Run MoAFCL feature-aware adapter federation."
    _add_prompt_backbone_args(parser)
    _set_clip_vlm_defaults(parser)
    parser.set_defaults(
        lr=1e-4,
        dataset="officehome",
        num_clients=10,
        client_sample_ratio=1.0,
        num_tasks=10,
        rounds_per_task=1,
        local_epochs=1,
        batch_size=32,
    )
    parser.add_argument("--algorithm", default="moafcl", choices=["moafcl"])
    parser.add_argument("--moafcl-num-adapters", type=int, default=5)
    parser.add_argument("--moafcl-gate-top-k", type=int, default=1)
    parser.add_argument("--moafcl-gate-temperature", type=float, default=5.0)
    parser.add_argument("--moafcl-gate-lr", type=float, default=1.5)
    parser.add_argument("--moafcl-gate-epochs", type=int, default=500)
    parser.add_argument("--moafcl-dp-epsilon", type=float, default=100.0)
    parser.add_argument("--moafcl-summary-batches", type=int, default=10)
    parser.add_argument("--moafcl-extract-layer", type=int, default=5)
    parser.add_argument("--moafcl-prompt-length", type=int, default=8)
    parser.add_argument("--moafcl-adapter-hidden-dim", type=int, default=1024)
    parser.add_argument("--scenario", choices=["class", "domain"], default="domain")
    return parser


def build_multifcl_parser() -> argparse.ArgumentParser:
    parser = _add_common_fcl_args()
    parser.description = "Run MultiFCL multi-scale expertise orchestration."
    _add_prompt_backbone_args(parser)
    _set_clip_vlm_defaults(parser)
    parser.set_defaults(
        seed=42,
        # dataset="cub200",
        # partition="iid",
        num_clients=10,
        client_sample_ratio=1.0,
        # num_tasks=10,
        rounds_per_task=5,
        local_epochs=5,
        batch_size=32,
    )
    parser.add_argument("--algorithm", default="multifcl", choices=["multifcl"])
    parser.add_argument("--multifcl-num-experts", type=int, default=4)
    parser.add_argument("--multifcl-adapter-lr", type=float, default=1e-4)
    parser.add_argument("--multifcl-head-lr", type=float, default=3e-3)
    parser.add_argument("--multifcl-weight-decay", type=float, default=5e-2)
    parser.add_argument("--multifcl-adapter-dropout", type=float, default=0.1)
    return parser


def build_powder_parser() -> argparse.ArgumentParser:
    parser = _add_common_fcl_args()
    parser.description = "Run Powder prompt-based dual knowledge transfer."
    _add_prompt_backbone_args(parser)
    parser.set_defaults(
        lr=5e-3,
        # partition="iid",
        # heterogeneous_task_order=True,
        # rounds_per_task=3,
        fcl_prompt_length=8,
        # num_rounds = 30
    )
    parser.add_argument("--algorithm", default="powder", choices=["powder"])
    parser.add_argument("--powder-top-k-tasks", type=int, default=3)
    parser.add_argument("--powder-correlation-power", type=float, default=30.0)
    parser.add_argument("--powder-dual-weight", type=float, default=1.0)
    parser.add_argument("--powder-temperature", type=float, default=3.0)
    parser.add_argument(
        "--powder-prompt-layers",
        type=int,
        nargs="+",
        default=[3, 4, 5],
        help="zero-based ViT blocks receiving prompt-tuning tokens; paper default is blocks 4,5,6",
    )
    return parser


def build_fedduet_parser() -> argparse.ArgumentParser:
    parser = _add_common_fcl_args()
    parser.description = "Run Fed-Duet dual semantic/parametric experts."
    _add_prompt_backbone_args(parser)
    _set_clip_vlm_defaults(parser)
    parser.set_defaults(
        lr=3e-4,
        dataset="cifar100",
        partition="iid",
        num_clients=5,
        client_sample_ratio=1.0,
        num_tasks=10,
        classes_per_task=10,
        rounds_per_task=10,
        local_epochs=1,
        batch_size=16,
    )
    parser.add_argument("--algorithm", default="fedduet", choices=["fedduet"])
    parser.add_argument("--fedduet-repository-size", type=int, default=64)
    parser.add_argument("--fedduet-num-experts", type=int, default=8)
    parser.add_argument("--fedduet-dispatch-count", type=int, default=4)
    parser.add_argument("--fedduet-phase-switch-round", type=int, default=5)
    parser.add_argument("--fedduet-shared-logit-weight", type=float, default=0.5)
    parser.add_argument("--fedduet-moe-weight", type=float, default=1.0)
    parser.add_argument("--fedduet-cross-modal-weight", type=float, default=1.0)
    parser.add_argument("--fedduet-stability-weight", type=float, default=1.0)
    parser.add_argument("--fedduet-gate-lr", type=float, default=1e-3)
    parser.add_argument("--fedduet-fusion-heads", type=int, default=8)
    parser.add_argument("--fedduet-fusion-dropout", type=float, default=0.2)
    parser.add_argument("--fedduet-top-k", type=int, default=2)
    parser.add_argument("--fedduet-adapter-dropout", type=float, default=0.4)
    parser.add_argument("--fedduet-adapter-scale", type=float, default=0.3)
    parser.add_argument("--scenario", choices=["class", "domain"], default="class")
    parser.add_argument("--fedduet-dp-clip", type=float, default=0.0)
    parser.add_argument("--fedduet-dp-noise-multiplier", type=float, default=0.0)
    return parser





def _add_pgd_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pgd-epsilon", type=float, default=8.0 / 255.0, help="PGD L-inf radius. For normalized torchvision datasets this is interpreted in raw pixel space unless --pgd-normalized-space is set.",)
    parser.add_argument("--pgd-step-size", type=float, default=2.0 / 255.0, help="PGD step size. For normalized torchvision datasets this is interpreted in raw pixel space unless --pgd-normalized-space is set.",)
    parser.add_argument("--pgd-steps", type=int, default=10, help="number of PGD ascent steps")
    parser.add_argument("--pgd-random-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pgd-normalized-space", action="store_true", default=False, help="treat --pgd-epsilon and --pgd-step-size as model input-space values instead of raw pixel-space values",)
    parser.add_argument("--pgd-max-batches", type=int, default=0, help="maximum eval batches for PGD FL_robust testing; <=0 evaluates the full test loader",)


def _set_algorithm_choice(parser: argparse.ArgumentParser, name: str) -> None:
    for action in parser._actions:
        if action.dest == "algorithm":
            action.default = name
            action.choices = [name]
            return


def _parse_fedweit_pgd_args() -> argparse.Namespace:
    parser = build_fedweit_parser()
    parser.description = "Run a FedWeIT baseline with PGD FL_robust evaluation."
    _add_pgd_args(parser)
    return parser.parse_args()


def _parse_fedweit_fat_args() -> argparse.Namespace:
    parser = build_fedweit_parser()
    parser.description = "Run FedWeIT with Federated Adversarial Training and PGD FL_robust evaluation."
    _set_algorithm_choice(parser, "fedweit_fat")
    _add_pgd_args(parser)
    parser.add_argument("--fat-adversarial-ratio", type=float, default=0.5, help="proportion of each local minibatch replaced by PGD adversarial examples after warmup")
    parser.add_argument("--fat-warmup-rounds", type=int, default=0, help="number of initial rounds per task trained with --fat-warmup-adversarial-ratio")
    parser.add_argument("--fat-warmup-adversarial-ratio", type=float, default=0.1, help="adversarial minibatch proportion used during FAT warmup rounds")
    return parser.parse_args()


def build_own_parser() -> argparse.ArgumentParser:
    """Build the RAMP-LOCI runner and its three legacy robust-loss ablations."""
    parser = build_loci_parser()
    parser.description = (
        "Run RobustLoci: task-aware adversarial training and robust knowledge "
        "distillation inside LOCI."
    )
    parser.set_defaults(
        local_epochs=5,
        rounds_per_task=20,
        num_rounds=200,
        loci_kd_epochs=3,
        loci_ewc_lambda=10.0,
        loci_knowledge_ratio=0.2,
        loci_similar_tasks=2,
        loci_public_samples=128,
    )
    _set_algorithm_choice(parser, "own")
    _add_pgd_args(parser)
    parser.add_argument(
        "--own-variant",
        type=str,
        default="ramp",
        choices=["ramp", "radt", "trades", "ard"],
        help=(
            "ramp uses snapshot-free robust GEM constraints and boundary-aware "
            "LOCI integration; radt, trades and ard are legacy ablations"
        ),
    )
    parser.add_argument(
        "--own-epsilon",
        type=float,
        default=8.0 / 255.0,
        help="training L-inf radius in raw pixel space unless --own-normalized-space is set",
    )
    parser.add_argument(
        "--own-step-size",
        type=float,
        default=2.0 / 255.0,
        help="training PGD step size in raw pixel space unless --own-normalized-space is set",
    )
    parser.add_argument("--own-steps", type=int, default=7)
    parser.add_argument(
        "--own-random-start", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--own-normalized-space", action="store_true", default=False)
    parser.add_argument("--own-clean-weight", type=float, default=1.0)
    parser.add_argument("--own-adversarial-weight", type=float, default=1.25)
    parser.add_argument("--own-trades-weight", type=float, default=2.0)
    parser.add_argument("--own-robust-kd-weight", type=float, default=1.0)
    parser.add_argument(
        "--own-boundary-weight",
        type=float,
        default=1.0,
        help="weight of clean-to-adversarial boundary displacement transfer",
    )
    parser.add_argument(
        "--own-robust-gradient-ratio",
        type=float,
        default=1.0,
        help="maximum robust-gradient norm relative to the primary clean gradient",
    )
    parser.add_argument(
        "--own-teacher-clean-weight",
        type=float,
        default=0.5,
        help="clean-accuracy share in LOCI's incoming KD-teacher selection score",
    )
    parser.add_argument(
        "--own-teacher-eval-batches",
        type=int,
        default=5,
        help="local batches used to score incoming teachers; <=0 uses all",
    )
    parser.add_argument(
        "--own-teacher-weight-floor",
        type=float,
        default=0.0,
        help=(
            "minimum reliability weight for the incoming KD teacher; zero avoids "
            "forcing below-chance teachers observed in early RobustLoci rounds"
        ),
    )
    parser.add_argument(
        "--own-warmup-rounds",
        type=int,
        default=3,
        help="linearly ramp robust losses over this many rounds within each task",
    )
    parser.add_argument("--own-fisher-adversarial-weight", type=float, default=1.0)
    parser.add_argument(
        "--own-importance-weight",
        type=float,
        default=1.0,
        help="robust Fisher multiplier in memory-palace sparse-weight ranking",
    )
    parser.add_argument("--own-knowledge-robust-weight", type=float, default=1.0)
    parser.add_argument("--own-class-balance-power", type=float, default=0.5)
    parser.add_argument("--own-class-balance-smoothing", type=float, default=1e-3)
    parser.add_argument("--own-class-weight-max", type=float, default=3.0)
    parser.add_argument(
        "--own-robust-memory-batch-size",
        type=int,
        default=32,
        help="GEM examples per old task used to form each robust constraint; <=0 uses all",
    )
    parser.add_argument("--own-replay-budget", type=int, default=0)
    parser.add_argument("--own-replay-batch-size", type=int, default=8)
    parser.add_argument("--own-replay-selection-batches", type=int, default=5)
    parser.add_argument("--own-replay-weight", type=float, default=1.0)
    parser.add_argument("--own-public-refine-epochs", type=int, default=0)
    parser.add_argument("--own-public-refine-lr-scale", type=float, default=0.5)
    parser.add_argument(
        "--own-robust-similarity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use clean/adversarial dual-view activation signatures for task selection",
    )
    parser.add_argument(
        "--own-selector-steps",
        type=int,
        default=1,
        help="PGD steps used for server dual-view activation signatures",
    )
    parser.add_argument(
        "--own-fusion-clean-tolerance",
        type=float,
        default=0.0,
        help="maximum public clean-accuracy drop allowed by guarded OT fusion",
    )
    parser.add_argument(
        "--own-fusion-clean-loss-tolerance",
        type=float,
        default=0.02,
        help="relative public clean-loss increase allowed by guarded OT fusion",
    )
    parser.add_argument(
        "--own-fusion-min-robust-gain",
        type=float,
        default=0.0,
        help="minimum decrease in public adversarial loss required to accept fusion",
    )
    return parser


def parse_own_args() -> argparse.Namespace:
    return build_own_parser().parse_args()


def build_loci_robust_parser() -> argparse.ArgumentParser:
    parser = build_loci_parser()
    parser.description = (
        "Run LOCI-AT: the original LOCI protocol with only local PGD "
        "adversarial training and PGD robust evaluation added."
    )
    _set_algorithm_choice(parser, "loci_at")
    _add_pgd_args(parser)
    parser.set_defaults(
        local_epochs=5,
        rounds_per_task=20,
        num_rounds=200,
        loci_kd_epochs=3,
        loci_ewc_lambda=10.0,
        loci_knowledge_ratio=0.2,
        loci_similar_tasks=2,
        loci_public_samples=128,
    )
    parser.add_argument(
        "--loci-adversarial-ratio",
        type=float,
        default=0.5,
        help="proportion of each local minibatch replaced by PGD adversarial examples after warmup",
    )
    parser.add_argument(
        "--loci-warmup-rounds",
        type=int,
        default=0,
        help="number of initial rounds per task trained with --loci-warmup-adversarial-ratio",
    )
    parser.add_argument(
        "--loci-warmup-adversarial-ratio",
        type=float,
        default=0.1,
        help="adversarial minibatch proportion used during LOCI-AT warmup rounds",
    )
    return parser


def parse_loci_robust_args() -> argparse.Namespace:
    return build_loci_robust_parser().parse_args()


def _parse_fedweit_sfat_args() -> argparse.Namespace:
    parser = build_fedweit_parser()
    parser.description = "Run FedWeIT with Slack Federated Adversarial Training and PGD FL_robust evaluation."
    _set_algorithm_choice(parser, "fedweit_sfat")
    _add_pgd_args(parser)
    parser.add_argument("--sfat-adversarial-ratio", type=float, default=0.5, help="proportion of each local minibatch replaced by PGD adversarial examples after warmup")
    parser.add_argument("--sfat-warmup-rounds", type=int, default=0, help="number of initial rounds per task trained with --sfat-warmup-adversarial-ratio")
    parser.add_argument("--sfat-warmup-adversarial-ratio", type=float, default=0.1, help="adversarial minibatch proportion used during SFAT warmup rounds")
    parser.add_argument("--sfat-alpha", type=float, default=1.0 / 11.0, help="SFAT alpha-slack value in [0, 1); top clients use (1 + alpha) / (1 - alpha)")
    parser.add_argument("--sfat-enhanced-clients", type=int, default=1, help="number of low adversarial-loss clients upweighted in each aggregation")
    parser.add_argument("--sfat-loss-metric", type=str, default="adv_ce_loss", choices=["adv_ce_loss", "ce_loss", "loss"], help="client metric used for SFAT ascending loss ranking")
    return parser.parse_args()


def _parse_fedweit_calfat_args() -> argparse.Namespace:
    parser = build_fedweit_parser()
    parser.description = "Run FedWeIT with Calibrated Federated Adversarial Training and PGD FL_robust evaluation."
    _set_algorithm_choice(parser, "fedweit_calfat")
    _add_pgd_args(parser)
    parser.add_argument("--calfat-prior-smoothing", type=float, default=1e-6,
                        help="small positive constant delta added to each client-task class prior for calibrated logits",)
    return parser.parse_args()


def _parse_fedweit_rbn_args() -> argparse.Namespace:
    parser = build_fedweit_parser()
    parser.description = "Run FedWeIT with FedRBN-style local BatchNorm personalization and PGD FL_robust evaluation."
    _set_algorithm_choice(parser, "fedweit_rbn")
    _add_pgd_args(parser)
    parser.add_argument("--rbn-at-ratio", type=float, default=0.2, help="fraction of clients treated as AT users; FedRBN is designed for mixed AT/ST clients, and the rest receive propagated BNa")
    parser.add_argument("--rbn-adv-lambda", type=float, default=0.5, help="weight on the adversarial loss term for AT users")
    parser.add_argument("--rbn-src-weight-mode", type=str, default="cos", choices=["eq", "cos"], help="source-client weighting used to propagate BNa from AT users to ST users")
    parser.add_argument("--rbn-pnc", type=float, default=0.5, help="coefficient of the pseudo-noise calibration loss for ST users; <0 disables PNC")
    parser.add_argument("--rbn-pnc-warmup", type=int, default=10, help="number of initial task rounds using zero PNC coefficient")
    parser.add_argument("--rbn-attack-noised-bn", action=argparse.BooleanOptionalAction, default=True, help="use the noised BN path while generating PGD adversarial examples for AT users")
    return parser.parse_args()


def _parse_fedweit_sylva_args() -> argparse.Namespace:
    parser = build_fedweit_parser()
    parser.description = "Run FedWeIT with Sylva-inspired personalized adversarial fine-tuning and PGD FL_robust evaluation."
    _set_algorithm_choice(parser, "fedweit_sylva")
    _add_pgd_args(parser)
    parser.add_argument("--sylva-class-balance-power", type=float, default=0.6, help="inverse-frequency exponent for Sylva's class-balanced local loss")
    parser.add_argument("--sylva-class-balance-smoothing", type=float, default=1e-3, help="small positive constant added to local class counts before weighting")
    parser.add_argument("--sylva-dynamic-rounds", type=int, default=3, help="number of task rounds used to ramp class weights from uniform to local imbalance-aware")
    parser.add_argument("--sylva-clean-weight", type=float, default=0.8, help="weight applied to Sylva's clean cross-entropy term")
    parser.add_argument("--sylva-adv-weight", type=float, default=1.25, help="weight applied to Sylva's adversarial cross-entropy term")
    parser.add_argument("--sylva-kl-weight", type=float, default=8.0, help="weight applied to Sylva's TRADES-style KL consistency term")
    parser.add_argument("--sylva-global-reg", type=float, default=1e-4, help="coefficient for Sylva's local-to-global shared-parameter alignment penalty")
    parser.add_argument("--sylva-agg-temperature", type=float, default=0.7, help="temperature scaling used by Sylva's similarity-aware aggregation")
    parser.add_argument("--sylva-agg-neighbors", type=int, default=2, help="number of nearest clients used for Sylva similarity scoring; <=0 uses all peers")
    parser.add_argument("--sylva-phase2-epochs", type=int, default=10, help="local benign refinement epochs for Sylva phase 2 after each task")
    parser.add_argument("--sylva-phase2-topk-layers", type=int, default=1, help="number of layer groups selected for Sylva phase 2 benign refinement")
    parser.add_argument("--sylva-phase2-tradeoff", type=float, default=0.7, help="penalty weight for adversarial sensitivity when scoring Sylva phase 2 layer groups")
    parser.add_argument("--sylva-phase2-lr-scale", type=float, default=0.0015, help="multiplier applied to the base learning rate during Sylva phase 2 benign refinement")
    parser.add_argument("--sylva-phase2-max-batches", type=int, default=10, help="maximum batches used for Sylva phase 2 layer scoring and benign refinement; <=0 uses all batches")
    return parser.parse_args()


def parse_fedavg_args() -> argparse.Namespace:
    return build_fedavg_parser().parse_args()


def parse_fedprox_args() -> argparse.Namespace:
    return build_fedprox_parser().parse_args()


def parse_fedkemf_args() -> argparse.Namespace:
    return build_fedkemf_parser().parse_args()


def parse_fedweit_args() -> argparse.Namespace:
    return build_fedweit_parser().parse_args()


def parse_fedknow_args() -> argparse.Namespace:
    return build_fedknow_parser().parse_args()


def parse_loci_args() -> argparse.Namespace:
    return build_loci_parser().parse_args()


def parse_fedprotip_args() -> argparse.Namespace:
    return build_fedprotip_parser().parse_args()


def parse_fedvit_args() -> argparse.Namespace:
    return build_fedvit_parser().parse_args()


def parse_fedmgp_args() -> argparse.Namespace:
    return build_fedmgp_parser().parse_args()


def parse_moafcl_args() -> argparse.Namespace:
    return build_moafcl_parser().parse_args()


def parse_multifcl_args() -> argparse.Namespace:
    return build_multifcl_parser().parse_args()


def parse_powder_args() -> argparse.Namespace:
    return build_powder_parser().parse_args()


def parse_fedduet_args() -> argparse.Namespace:
    return build_fedduet_parser().parse_args()


def parse_scaffold_args() -> argparse.Namespace:
    return build_scaffold_parser().parse_args()


def parse_moon_args() -> argparse.Namespace:
    return build_moon_parser().parse_args()


def parse_fat_args() -> argparse.Namespace:
    return build_fat_parser().parse_args()


def parse_sfat_args() -> argparse.Namespace:
    return build_sfat_parser().parse_args()


def parse_calfat_args() -> argparse.Namespace:
    return build_calfat_parser().parse_args()


def parse_rbn_args() -> argparse.Namespace:
    return build_rbn_parser().parse_args()


def parse_sylva_args() -> argparse.Namespace:
    return build_sylva_parser().parse_args()


def parse_ewc_args() -> argparse.Namespace:
    return build_ewc_parser().parse_args()


def parse_gem_args() -> argparse.Namespace:
    return build_gem_parser().parse_args()


def parse_lwf_args() -> argparse.Namespace:
    return build_lwf_parser().parse_args()


def parse_icarl_args() -> argparse.Namespace:
    return build_icarl_parser().parse_args()


def parse_taba_args() -> argparse.Namespace:
    return build_taba_parser().parse_args()


def parse_daml_args() -> argparse.Namespace:
    return build_daml_parser().parse_args()


def parse_flair_args() -> argparse.Namespace:
    return build_flair_parser().parse_args()


def parse_aflc_raer_args() -> argparse.Namespace:
    return build_aflc_raer_parser().parse_args()

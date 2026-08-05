from __future__ import annotations

import argparse

from cl_fcl_baseline.models import (
    DEFAULT_VIT_ATTENTION_DROPOUT,
    DEFAULT_VIT_DROPOUT,
    DEFAULT_VIT_MLP_RATIO,
    DEFAULT_VIT_PATCH_SIZE,
)






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
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128, help="<=0 for full-batch")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam"])
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["mnist", "cifar10", "cifar100"])
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--num-samples", type=int, default=0, help="<=0 for full-dataset")
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument(
        "--model",
        type=str,
        default="ResNet32",
        choices=["mlp", "simplecnn", "VGG11", "ResNet18", "ResNet20", "ResNet32", "ViTTiny", "ViTSmall", "ViTBase"],
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
        type=str,
        default="cifar100",
        choices=["random_classification", "mnist", "cifar10", "cifar100"],
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--num-samples", type=int, default=0, help="<=0 for full-dataset")
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--input-shape", type=int, nargs=3, default=[1, 28, 28])
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
        type=str,
        default=default_model,
        choices=["mlp", "simplecnn", "VGG11", "ResNet18", "ResNet20", "ResNet32", "ViTTiny", "ViTSmall", "ViTBase"],
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
    parser.add_argument("--classes-per-task", type=int, default=10)
    parser.add_argument("--rounds-per-task", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64, help="<=0 for full-batch")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam"])
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["mnist", "cifar10", "cifar100"])
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--num-samples", type=int, default=0, help="<=0 for full-dataset")
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument(
        "--model",
        type=str,
        default="ResNet32",
        choices=["mlp", "simplecnn", "VGG11", "ResNet18", "ResNet20", "ResNet32", "ViTTiny", "ViTSmall", "ViTBase"],
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
    for action in parser._actions:
        if action.dest == "dataset":
            action.choices = ["random_classification", "mnist", "cifar10", "cifar100"]
            action.default = "cifar100"
            break
    parser.add_argument("--algorithm", type=str, default="fedweit", choices=["fedweit"])
    parser.add_argument("--lambda1", type=float, default=0.0005, help="FedWeIT lambda_l1 for adaptive parameters.")
    parser.add_argument("--lambda2", type=float, default=30.0, help="FedWeIT lambda_l2 retroactive coefficient.")
    parser.add_argument("--lambda-mask", type=float, default=0.0, help="FedWeIT lambda_mask for raw task masks.")
    parser.add_argument("--kb-sample-size", type=int, default=0, help="knowledge-base samples per task; <=0 means all.")
    parser.add_argument("--mask-init", type=float, default=-1.0, help="initial sigmoid mask value; <0 uses random raw-mask init like the reference implementation.")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help=argparse.SUPPRESS)
    parser.add_argument("--client-sparsity", type=float, default=0.3, help="fraction of smallest raw-mask entries pruned from communicable B*m.")
    parser.add_argument("--adaptive-threshold", type=float, default=-1.0, help="hard threshold for communicable A; <0 uses lambda1.")
    parser.add_argument("--input-shape", type=int, nargs=3, default=[1, 28, 28])
    return parser



def build_fedknow_parser() -> argparse.ArgumentParser:
    parser = _add_common_fcl_args()
    parser.description = "Run a FedKNOW baseline."
    for action in parser._actions:
        if action.dest == "dataset":
            action.choices = ["random_classification", "mnist", "cifar10", "cifar100"]
            action.default = "cifar100"
            break
    _set_algorithm_choice(parser, "fedknow")
    parser.add_argument("--knowledge-ratio", type=float, default=0.1, help="FedKNOW rho: ratio of top-magnitude weights retained as signature task knowledge.")
    parser.add_argument("--signature-k", type=int, default=10, help="number of most dissimilar signature task gradients integrated for each batch.")
    parser.add_argument("--integrator-steps", type=int, default=100, help="projected-gradient steps used to solve FedKNOW's dual QP gradient integrator.")
    parser.add_argument("--knowledge-finetune-epochs", type=int, default=2, help="PackNet fine-tuning epochs after pruning; the reference FedKNOW code uses 2.")
    parser.add_argument("--post-aggregation-epochs", type=int, default=1, help="one local fine-tuning epoch after global aggregation, as specified in FedKNOW Section III-A.")
    parser.add_argument("--distillation-warmup-epochs", type=int, default=2, help="dense old-model distillation epochs before FedKNOW representation integration; the reference code uses 2.")
    parser.add_argument("--restorer-loss", type=str, default="soft", choices=["hard", "soft"], help="loss used by the gradient restorer: hard pseudo labels or soft KL targets from retained knowledge.")
    parser.add_argument("--restorer-temperature", type=float, default=2.0, help="temperature used for FedKNOW soft distillation in the gradient restorer.")
    parser.add_argument("--input-shape", type=int, nargs=3, default=[1, 28, 28])
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


def _parse_fedweit_own_args() -> argparse.Namespace:
    parser = build_fedweit_parser()
    parser.description = "Run FedWeIT with the risk-aware personalized UAP defense variant and PGD FL_robust evaluation."
    _set_algorithm_choice(parser, "fedweit_own")
    _add_pgd_args(parser)
    parser.add_argument("--own-uap-epsilon", type=float, default=10.0 / 255.0, help="L-inf radius for the local universal perturbation. For normalized torchvision datasets this is interpreted in raw pixel space unless --own-uap-normalized-space is set.")
    parser.add_argument("--own-uap-lr", type=float, default=0.05, help="optimizer learning rate for local UAP generation")
    parser.add_argument("--own-uap-gen-epochs", type=int, default=2, help="number of local epochs used to generate each client's UAP")
    parser.add_argument("--own-uap-data-ratio", type=float, default=0.1, help="fraction of the local task dataset used for UAP generation")
    parser.add_argument("--own-uap-normalized-space", action="store_true", default=False, help="treat --own-uap-epsilon as model input-space values instead of raw pixel-space values")
    parser.add_argument("--own-adv-epochs", type=int, default=4, help="number of local defense epochs after each aggregation")
    parser.add_argument("--own-stage1-epochs", type=int, default=2, help="number of early defense epochs that perturb only high-risk samples with the full personalized UAP pool")
    parser.add_argument("--own-adv-mix-ratio", type=float, default=0.1, help="probability of perturbing low-risk samples during stage 2 of local defense")
    parser.add_argument("--own-conf-threshold", type=float, default=0.1, help="samples below this max-softmax confidence are treated as high risk")
    parser.add_argument("--own-defense-lr-scale", type=float, default=0.5, help="multiplier applied to the base learning rate during the local defense phase")
    parser.add_argument("--own-k-uap", type=int, default=2, help="number of most similar peer UAPs added to each client's personalized defense pool")
    return parser.parse_args()


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

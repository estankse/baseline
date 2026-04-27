from __future__ import annotations

import argparse


def _add_common_fl_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-clients", type=int, default=30)
    parser.add_argument("--client-sample-ratio", type=float, default=0.4, help="fraction of clients sampled each round (0, 1].")
    parser.add_argument("--device", type=str, default="auto", help="cpu | cuda | cuda:0 | auto")
    parser.add_argument("--partition", type=str, default="noniid", choices=["iid", "noniid"])
    parser.add_argument("--noniid-method", type=str, default="dirichlet", choices=["shards", "dirichlet"], help="noniid partitioning strategy when --partition noniid is selected.",)
    parser.add_argument("--noniid-shards", type=int, default=2)
    parser.add_argument("--dirichlet-beta", type=float, default=0.1)
    parser.add_argument("--num-rounds", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64, help="<=0 for full-batch")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["mnist", "cifar10", "cifar100"])
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--num-samples", type=int, default=0, help="<=0 for full-dataset")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--model", type=str, default="ResNet32", choices=["mlp", "simplecnn","VGG11","ResNet18","ResNet20","ResNet32"])
    parser.add_argument("--hidden-dim", type=int, default=200)
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
    parser.add_argument("--num-rounds", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64, help="<=0 for full-batch")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam"])
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["mnist", "cifar10", "cifar100"])
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--num-samples", type=int, default=0, help="<=0 for full-dataset")
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--model", type=str, default="ResNet32",
                        choices=["mlp", "simplecnn", "VGG11", "ResNet18", "ResNet20", "ResNet32"])
    parser.add_argument("--hidden-dim", type=int, default=200)
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
    parser.add_argument("--num-tasks", type=int, default=0)
    parser.add_argument("--classes-per-task", type=int, default=10)
    parser.add_argument("--task-order-shuffle", action="store_true", default=False)
    parser.add_argument("--rounds-per-task", type=int, default=20)
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

def _add_pgd_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pgd-epsilon", type=float, default=8.0 / 255.0, help="PGD L-inf radius. For normalized torchvision datasets this is interpreted in raw pixel space unless --pgd-normalized-space is set.",)
    parser.add_argument("--pgd-step-size", type=float, default=2.0 / 255.0, help="PGD step size. For normalized torchvision datasets this is interpreted in raw pixel space unless --pgd-normalized-space is set.",)
    parser.add_argument("--pgd-steps", type=int, default=10, help="number of PGD ascent steps")
    parser.add_argument("--pgd-random-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pgd-normalized-space", action="store_true", default=False, help="treat --pgd-epsilon and --pgd-step-size as model input-space values instead of raw pixel-space values",)
    parser.add_argument("--pgd-max-batches", type=int, default=0, help="maximum eval batches for PGD robust testing; <=0 evaluates the full test loader",)


def _set_algorithm_choice(parser: argparse.ArgumentParser, name: str) -> None:
    for action in parser._actions:
        if action.dest == "algorithm":
            action.default = name
            action.choices = [name]
            return


def _parse_fedweit_pgd_args() -> argparse.Namespace:
    parser = build_fedweit_parser()
    parser.description = "Run a FedWeIT baseline with PGD robust evaluation."
    _add_pgd_args(parser)
    return parser.parse_args()


def _parse_fedweit_fat_args() -> argparse.Namespace:
    parser = build_fedweit_parser()
    parser.description = "Run FedWeIT with Federated Adversarial Training and PGD robust evaluation."
    _set_algorithm_choice(parser, "fedweit_fat")
    _add_pgd_args(parser)
    parser.add_argument("--fat-adversarial-ratio", type=float, default=0.5, help="proportion of each local minibatch replaced by PGD adversarial examples after warmup")
    parser.add_argument("--fat-warmup-rounds", type=int, default=0, help="number of initial rounds per task trained with --fat-warmup-adversarial-ratio")
    parser.add_argument("--fat-warmup-adversarial-ratio", type=float, default=0.1, help="adversarial minibatch proportion used during FAT warmup rounds")
    return parser.parse_args()


def _parse_fedweit_sfat_args() -> argparse.Namespace:
    parser = build_fedweit_parser()
    parser.description = "Run FedWeIT with Slack Federated Adversarial Training and PGD robust evaluation."
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
    parser.description = "Run FedWeIT with Calibrated Federated Adversarial Training and PGD robust evaluation."
    _set_algorithm_choice(parser, "fedweit_calfat")
    _add_pgd_args(parser)
    parser.add_argument("--calfat-prior-smoothing", type=float, default=1e-6,
                        help="small positive constant delta added to each client-task class prior for calibrated logits",)
    return parser.parse_args()


def _parse_fedweit_rbn_args() -> argparse.Namespace:
    parser = build_fedweit_parser()
    parser.description = "Run FedWeIT with FedRBN-style local BatchNorm personalization and PGD robust evaluation."
    _set_algorithm_choice(parser, "fedweit_rbn")
    _add_pgd_args(parser)
    parser.add_argument("--rbn-at-ratio", type=float, default=1.0, help="fraction of clients treated as AT users; the rest are ST users receiving propagated BNa")
    parser.add_argument("--rbn-adv-lambda", type=float, default=0.5, help="weight on the adversarial loss term for AT users")
    parser.add_argument("--rbn-src-weight-mode", type=str, default="cos", choices=["eq", "cos"], help="source-client weighting used to propagate BNa from AT users to ST users")
    parser.add_argument("--rbn-pnc", type=float, default=0.5, help="coefficient of the pseudo-noise calibration loss for ST users; <0 disables PNC")
    parser.add_argument("--rbn-pnc-warmup", type=int, default=10, help="number of initial task rounds using zero PNC coefficient")
    parser.add_argument("--rbn-attack-noised-bn", action=argparse.BooleanOptionalAction, default=True, help="use the noised BN path while generating PGD adversarial examples for AT users")
    return parser.parse_args()


def _parse_fedweit_sylva_args() -> argparse.Namespace:
    parser = build_fedweit_parser()
    parser.description = "Run FedWeIT with Sylva-inspired personalized adversarial fine-tuning and PGD robust evaluation."
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


def parse_scaffold_args() -> argparse.Namespace:
    return build_scaffold_parser().parse_args()


def parse_moon_args() -> argparse.Namespace:
    return build_moon_parser().parse_args()

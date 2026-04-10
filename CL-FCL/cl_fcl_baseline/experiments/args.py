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
    parser.add_argument("--dirichlet-beta", type=float, default=0.5)
    parser.add_argument("--num-rounds", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64, help="<=0 for full-batch")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["mnist", "cifar10"])
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--num-samples", type=int, default=0, help="<=0 for full-dataset")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=200)
    parser.add_argument("--show-progress", action="store_true",default=True)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--log-file", type=str, default="")


def build_fedavg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a FedAvg baseline.")
    _add_common_fl_args(parser)
    parser.add_argument("--algorithm", type=str, default="fedavg", choices=["fedavg"])
    parser.add_argument("--model", type=str, default="simplecnn", choices=["mlp", "simplecnn","VGG11","ResNet18","ResNet20","ResNet32"])
    return parser


# def build_fl_parser() -> argparse.ArgumentParser:
#     return build_fedavg_parser()


def build_fedkemf_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a FedKEMF baseline.")
    _add_common_fl_args(parser)
    parser.add_argument("--algorithm", type=str, default="fedkemf", choices=["fedkemf"])
    parser.add_argument("--model", type=str, default="VGG11", choices=["mlp", "simplecnn","VGG11","ResNet18","ResNet20","ResNet32"])
    parser.add_argument("--distill", action="store_true", default=True, help="enable client-side distillation")
    parser.add_argument("--distill-epochs", type=int, default=10)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--distill-alpha", type=float, default=0.5)
    parser.add_argument("--mutual-learning", action="store_true", default=True, help="enable deep mutual learning")
    parser.add_argument("--server-distill-epochs", type=int, default=1)
    parser.add_argument("--server-distill-lr", type=float, default=0.01)
    parser.add_argument("--server-distill-temperature", type=float, default=2.0)
    parser.add_argument("--server-ensemble", type=str, default="max", choices=["max", "mean"])
    parser.add_argument("--server-data-ratio", type=float, default=0.6, help="fraction of the training dataset used as the server public set (0, 1].",)
    return parser


def build_fcl_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a minimal FCL baseline.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument(
        "--client-sample-ratio",
        type=float,
        default=1.0,
        help="fraction of clients sampled each round (0, 1].",
    )
    parser.add_argument("--device", type=str, default="auto", help="cpu | cuda | cuda:0 | auto")
    parser.add_argument("--partition", type=str, default="iid", choices=["iid", "noniid"])
    parser.add_argument(
        "--noniid-method",
        type=str,
        default="shards",
        choices=["shards", "dirichlet"],
        help="noniid partitioning strategy when --partition noniid is selected.",
    )
    parser.add_argument("--noniid-shards", type=int, default=2)
    parser.add_argument("--dirichlet-beta", type=float, default=0.1)
    parser.add_argument("--rounds-per-task", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16, help="<=0 for full-batch")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--dataset", type=str, default="mnist", choices=["random_classification", "mnist", "cifar10"])
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--input-shape", type=int, nargs=3, default=[1, 28, 28])
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--tasks", type=str, nargs="+", default=["task_a", "task_b"])
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--log-file", type=str, default="")
    return parser


def parse_fedavg_args() -> argparse.Namespace:
    return build_fedavg_parser().parse_args()


def parse_fedkemf_args() -> argparse.Namespace:
    return build_fedkemf_parser().parse_args()


def parse_fcl_args() -> argparse.Namespace:
    return build_fcl_parser().parse_args()

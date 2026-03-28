from __future__ import annotations

import argparse


def build_fl_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a FL baseline.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--partition", type=str, default="iid", choices=["iid", "noniid"])
    parser.add_argument("--noniid-shards", type=int, default=2)
    parser.add_argument("--num-rounds", type=int, default=100)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=10, help="<=0 for full-batch")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--dataset", type=str, default="mnist", choices=["random_classification", "mnist", "cifar10"])
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--model", type=str, default="simplecnn", choices=["mlp","simplecnn"])
    parser.add_argument("--algorithm", type=str, default="fedavg", choices=["fedavg"])
    parser.add_argument("--num-samples", type=int, default=0, help="<=0 for full-dataset")
    parser.add_argument("--input-shape", type=int, nargs=3, default=[1, 28, 28])
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=200)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--eval-every", type=int, default=1)
    return parser


def build_fcl_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a minimal FCL baseline.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--partition", type=str, default="iid", choices=["iid", "noniid"])
    parser.add_argument("--noniid-shards", type=int, default=2)
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
    return parser


def parse_fedavg_args() -> argparse.Namespace:
    return build_fl_parser().parse_args()


def parse_fcl_args() -> argparse.Namespace:
    return build_fcl_parser().parse_args()

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from cl_fcl_baseline.algorithms.robust.PGD import evaluate_pgd_robustness
from cl_fcl_baseline.algorithms.robust.SFAT import SFATAggregator, SFATClient
from cl_fcl_baseline.datasets import build_torchvision_dataset, dataset_info
from cl_fcl_baseline.datasets.build import (
    build_dataloader,
    partition_dataset_dirichlet,
    partition_dataset_iid,
    partition_dataset_noniid,
)
from cl_fcl_baseline.experiments.args import parse_sfat_args
from cl_fcl_baseline.experiments.FCL_robust.fcl_robust import build_model, build_optimizer, build_pgd_config
from cl_fcl_baseline.trainers.server import FederatedExperiment, FederatedServer
from cl_fcl_baseline.trainers.trainer import BaseTrainer
from cl_fcl_baseline.trainers.utils import set_seed


def main() -> None:
    args = parse_sfat_args()
    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    input_shape, num_classes = dataset_info(args.dataset)
    dataset = build_torchvision_dataset(
        name=args.dataset,
        train=True,
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    test_dataset = build_torchvision_dataset(
        name=args.dataset,
        train=False,
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        seed=args.seed + 1,
    )
    if args.partition == "iid":
        partitions = partition_dataset_iid(dataset, num_clients=args.num_clients, seed=args.seed)
    elif args.noniid_method == "dirichlet":
        partitions = partition_dataset_dirichlet(
            dataset,
            num_clients=args.num_clients,
            beta=args.dirichlet_beta,
            num_classes=num_classes,
            seed=args.seed,
        )
    else:
        partitions = partition_dataset_noniid(
            dataset,
            num_clients=args.num_clients,
            num_shards=args.noniid_shards,
            seed=args.seed,
        )
    loaders = [build_dataloader(part, batch_size=args.batch_size, shuffle=True) for part in partitions]
    test_loader = build_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)

    pgd_config = build_pgd_config(args)
    clients = []
    for idx, loader in enumerate(loaders):
        model = build_model(args, input_shape=input_shape, num_classes=num_classes)
        optimizer = build_optimizer(args, model)
        trainer = BaseTrainer(model=model, optimizer=optimizer, device=device)
        clients.append(
            SFATClient(
                client_id=f"client_{idx}",
                trainer=trainer,
                train_loader=loader,
                epochs=args.local_epochs,
                pgd_config=pgd_config,
                adversarial_ratio=args.sfat_adversarial_ratio,
                warmup_rounds=args.sfat_warmup_rounds,
                warmup_adversarial_ratio=args.sfat_warmup_adversarial_ratio,
            )
        )

    server_model = build_model(args, input_shape=input_shape, num_classes=num_classes)
    server = FederatedServer(
        model=server_model,
        clients=clients,
        aggregator=SFATAggregator(
            alpha=args.sfat_alpha,
            enhanced_clients=args.sfat_enhanced_clients,
            slack_loss_metric=args.sfat_loss_metric,
        ),
        client_sample_ratio=args.client_sample_ratio,
    )

    eval_optimizer = build_optimizer(args, server.model)
    eval_trainer = BaseTrainer(model=server.model, optimizer=eval_optimizer, device=device)
    log_path = args.log_file.strip()
    if not log_path:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = str(log_dir / f"sfat_{timestamp}.jsonl")
    else:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    pgd_max_batches = None if int(args.pgd_max_batches) <= 0 else int(args.pgd_max_batches)

    def _eval_round(round_idx: int) -> None:
        test_metrics = eval_trainer.evaluate(test_loader)
        robust_metrics = evaluate_pgd_robustness(
            server.model,
            test_loader,
            pgd_config,
            device=device,
            max_batches=pgd_max_batches,
        )
        print(
            f"[eval] round {round_idx}: "
            f"acc={test_metrics.get('accuracy', 0.0):.4f} robust_acc={robust_metrics.get('accuracy', 0.0):.4f}"
        )
        record = {
            "type": "eval",
            "round": round_idx,
            "metrics": {
                "accuracy": float(test_metrics.get("accuracy", 0.0)),
                "loss": float(test_metrics.get("loss", 0.0)),
                "robust_accuracy": float(robust_metrics.get("accuracy", 0.0)),
                "robust_loss": float(robust_metrics.get("loss", 0.0)),
                "num_pgd_batches": float(robust_metrics.get("num_batches", 0.0)),
                "num_pgd_samples": float(robust_metrics.get("num_samples", 0.0)),
            },
            "pgd": {
                "epsilon": args.pgd_epsilon,
                "step_size": args.pgd_step_size,
                "steps": args.pgd_steps,
                "random_start": args.pgd_random_start,
                "normalized_space": args.pgd_normalized_space,
                "max_batches": args.pgd_max_batches,
            },
            "sfat": {
                "adversarial_ratio": args.sfat_adversarial_ratio,
                "warmup_rounds": args.sfat_warmup_rounds,
                "warmup_adversarial_ratio": args.sfat_warmup_adversarial_ratio,
                "alpha": args.sfat_alpha,
                "enhanced_clients": args.sfat_enhanced_clients,
                "loss_metric": args.sfat_loss_metric,
            },
        }
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    experiment = FederatedExperiment(
        server=server,
        num_rounds=args.num_rounds,
        show_progress=args.show_progress,
        log_each_round=True,
        eval_every=args.eval_every,
        eval_fn=_eval_round if args.eval_every and args.eval_every > 0 else None,
        log_path=log_path,
    )

    experiment.run()
    print("SFAT finished.")


if __name__ == "__main__":
    main()

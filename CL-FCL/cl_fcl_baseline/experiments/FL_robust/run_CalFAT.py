from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from cl_fcl_baseline.algorithms.robust.CalFAT import (
    CalFATClient,
    evaluate_calfat_model,
    evaluate_calfat_pgd_robustness,
)
from cl_fcl_baseline.algorithms.fl import FedAvgAggregator
from cl_fcl_baseline.datasets import build_torchvision_dataset, dataset_info
from cl_fcl_baseline.datasets.build import (
    build_dataloader,
    partition_dataset_dirichlet,
    partition_dataset_iid,
    partition_dataset_noniid,
)
from cl_fcl_baseline.experiments.args import parse_calfat_args
from cl_fcl_baseline.experiments.FCL_robust.fcl_robust import build_model, build_optimizer, build_pgd_config
from cl_fcl_baseline.trainers.server import FederatedExperiment, FederatedServer
from cl_fcl_baseline.trainers.trainer import BaseTrainer
from cl_fcl_baseline.trainers.utils import set_seed


def main() -> None:
    args = parse_calfat_args()
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
            CalFATClient(
                client_id=f"client_{idx}",
                trainer=trainer,
                train_loader=loader,
                epochs=args.local_epochs,
                pgd_config=pgd_config,
                num_classes=num_classes,
                prior_smoothing=args.calfat_prior_smoothing,
            )
        )

    server_model = build_model(args, input_shape=input_shape, num_classes=num_classes)
    server = FederatedServer(
        model=server_model,
        clients=clients,
        aggregator=FedAvgAggregator(),
        client_sample_ratio=args.client_sample_ratio,
    )

    eval_model = build_model(args, input_shape=input_shape, num_classes=num_classes)
    log_path = args.log_file.strip()
    if not log_path:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = str(log_dir / f"calfat_{timestamp}.jsonl")
    else:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    pgd_max_batches = None if int(args.pgd_max_batches) <= 0 else int(args.pgd_max_batches)

    def _eval_round(round_idx: int) -> None:
        evaluated_clients = 0
        total_accuracy = 0.0
        total_loss = 0.0
        total_robust_accuracy = 0.0
        total_robust_loss = 0.0
        total_robust_batches = 0.0
        total_robust_samples = 0.0

        global_state = server.get_global_state()
        for client in clients:
            eval_model.load_state_dict(global_state, strict=True)
            log_prior = client.class_log_prior(device)
            local_metrics = evaluate_calfat_model(eval_model, test_loader, log_prior, device=device)
            robust_metrics = evaluate_calfat_pgd_robustness(
                eval_model,
                test_loader,
                log_prior,
                pgd_config,
                device=device,
                max_batches=pgd_max_batches,
            )
            evaluated_clients += 1
            total_accuracy += float(local_metrics.get("accuracy", 0.0))
            total_loss += float(local_metrics.get("loss", 0.0))
            total_robust_accuracy += float(robust_metrics.get("accuracy", 0.0))
            total_robust_loss += float(robust_metrics.get("loss", 0.0))
            total_robust_batches += float(robust_metrics.get("num_batches", 0.0))
            total_robust_samples += float(robust_metrics.get("num_samples", 0.0))

        avg_metrics = {
            "accuracy": total_accuracy / max(1, evaluated_clients),
            "loss": total_loss / max(1, evaluated_clients),
            "robust_accuracy": total_robust_accuracy / max(1, evaluated_clients),
            "robust_loss": total_robust_loss / max(1, evaluated_clients),
            "num_eval_clients": float(evaluated_clients),
            "num_eval_samples": float(len(test_loader.dataset)),
            "num_pgd_batches": total_robust_batches / max(1, evaluated_clients),
            "num_pgd_samples": total_robust_samples / max(1, evaluated_clients),
        }
        print(
            f"[eval] round {round_idx}: "
            f"acc={avg_metrics['accuracy']:.4f} robust_acc={avg_metrics['robust_accuracy']:.4f}"
        )
        record = {
            "type": "eval",
            "round": round_idx,
            "metrics": avg_metrics,
            "pgd": {
                "epsilon": args.pgd_epsilon,
                "step_size": args.pgd_step_size,
                "steps": args.pgd_steps,
                "random_start": args.pgd_random_start,
                "normalized_space": args.pgd_normalized_space,
                "max_batches": args.pgd_max_batches,
            },
            "calfat": {
                "prior_smoothing": args.calfat_prior_smoothing,
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
    print("CalFAT finished.")


if __name__ == "__main__":
    main()

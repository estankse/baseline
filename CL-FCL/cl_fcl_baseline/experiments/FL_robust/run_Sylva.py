from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from cl_fcl_baseline.algorithms.robust.PGD import evaluate_pgd_robustness
from cl_fcl_baseline.algorithms.robust.Sylva import SylvaAggregator, SylvaClient
from cl_fcl_baseline.datasets import build_torchvision_dataset, dataset_info
from cl_fcl_baseline.datasets.build import (
    build_dataloader,
    partition_dataset_dirichlet,
    partition_dataset_iid,
    partition_dataset_noniid,
)
from cl_fcl_baseline.experiments.args import parse_sylva_args
from cl_fcl_baseline.experiments.FCL_robust.fcl_robust import build_model, build_optimizer, build_pgd_config
from cl_fcl_baseline.trainers.server import FederatedExperiment, FederatedServer
from cl_fcl_baseline.trainers.trainer import BaseTrainer
from cl_fcl_baseline.trainers.utils import detach_state_dict, set_seed


def _partition_eval_dataset(args, dataset, num_classes: int):
    if args.partition == "iid":
        return partition_dataset_iid(dataset, num_clients=args.num_clients, seed=args.seed)
    if args.noniid_method == "dirichlet":
        return partition_dataset_dirichlet(
            dataset,
            num_clients=args.num_clients,
            beta=args.dirichlet_beta,
            num_classes=num_classes,
            seed=args.seed,
        )
    return partition_dataset_noniid(
        dataset,
        num_clients=args.num_clients,
        num_shards=args.noniid_shards,
        seed=args.seed,
    )


def main() -> None:
    args = parse_sylva_args()
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
    eval_partitions = _partition_eval_dataset(args, test_dataset, num_classes=num_classes)
    eval_loaders = [build_dataloader(part, batch_size=args.batch_size, shuffle=False) for part in eval_partitions]

    pgd_config = build_pgd_config(args)
    clients = []
    for idx, loader in enumerate(loaders):
        model = build_model(args, input_shape=input_shape, num_classes=num_classes)
        optimizer = build_optimizer(args, model)
        trainer = BaseTrainer(model=model, optimizer=optimizer, device=device)
        clients.append(
            SylvaClient(
                client_id=f"client_{idx}",
                trainer=trainer,
                train_loader=loader,
                epochs=args.local_epochs,
                pgd_config=pgd_config,
                num_classes=num_classes,
                class_balance_power=args.sylva_class_balance_power,
                class_balance_smoothing=args.sylva_class_balance_smoothing,
                dynamic_weight_rounds=args.sylva_dynamic_rounds,
                clean_loss_weight=args.sylva_clean_weight,
                adv_loss_weight=args.sylva_adv_weight,
                kl_weight=args.sylva_kl_weight,
                global_reg=args.sylva_global_reg,
                phase2_epochs=args.sylva_phase2_epochs,
                phase2_topk_layers=args.sylva_phase2_topk_layers,
                phase2_tradeoff=args.sylva_phase2_tradeoff,
                phase2_lr_scale=args.sylva_phase2_lr_scale,
                phase2_max_batches=args.sylva_phase2_max_batches,
            )
        )

    server_model = build_model(args, input_shape=input_shape, num_classes=num_classes)
    server = FederatedServer(
        model=server_model,
        clients=clients,
        aggregator=SylvaAggregator(
            temperature=args.sylva_agg_temperature,
            neighbors=args.sylva_agg_neighbors,
        ),
        client_sample_ratio=args.client_sample_ratio,
    )

    eval_model = build_model(args, input_shape=input_shape, num_classes=num_classes)
    eval_optimizer = build_optimizer(args, eval_model)
    eval_trainer = BaseTrainer(model=eval_model, optimizer=eval_optimizer, device=device)
    log_path = args.log_file.strip()
    if not log_path:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = str(log_dir / f"sylva_{timestamp}.jsonl")
    else:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    pgd_max_batches = None if int(args.pgd_max_batches) <= 0 else int(args.pgd_max_batches)

    def _evaluate_state(model_state: dict[str, torch.Tensor]) -> tuple[dict[str, float], dict[str, float]]:
        eval_model.load_state_dict(model_state, strict=True)
        test_metrics = eval_trainer.evaluate(test_loader)
        robust_metrics = evaluate_pgd_robustness(
            eval_model,
            test_loader,
            pgd_config,
            device=device,
            max_batches=pgd_max_batches,
        )
        return test_metrics, robust_metrics

    def _eval_round(round_idx: int) -> None:
        global_test_metrics, global_robust_metrics = _evaluate_state(server.get_global_state())
        personalized_clients = 0
        total_accuracy = 0.0
        total_loss = 0.0
        total_robust_accuracy = 0.0
        total_robust_loss = 0.0
        total_robust_batches = 0.0
        total_robust_samples = 0.0
        for client, client_test_loader in zip(clients, eval_loaders):
            personalized_state = detach_state_dict(client.trainer.model.state_dict())
            eval_model.load_state_dict(personalized_state, strict=True)
            test_metrics = eval_trainer.evaluate(client_test_loader)
            robust_metrics = evaluate_pgd_robustness(
                eval_model,
                client_test_loader,
                pgd_config,
                device=device,
                max_batches=pgd_max_batches,
            )
            personalized_clients += 1
            total_accuracy += float(test_metrics.get("accuracy", 0.0))
            total_loss += float(test_metrics.get("loss", 0.0))
            total_robust_accuracy += float(robust_metrics.get("accuracy", 0.0))
            total_robust_loss += float(robust_metrics.get("loss", 0.0))
            total_robust_batches += float(robust_metrics.get("num_batches", 0.0))
            total_robust_samples += float(robust_metrics.get("num_samples", 0.0))

        metrics = {
            "accuracy": total_accuracy / max(1, personalized_clients),
            "loss": total_loss / max(1, personalized_clients),
            "robust_accuracy": total_robust_accuracy / max(1, personalized_clients),
            "robust_loss": total_robust_loss / max(1, personalized_clients),
            "global_accuracy": float(global_test_metrics.get("accuracy", 0.0)),
            "global_loss": float(global_test_metrics.get("loss", 0.0)),
            "global_robust_accuracy": float(global_robust_metrics.get("accuracy", 0.0)),
            "global_robust_loss": float(global_robust_metrics.get("loss", 0.0)),
            "num_eval_clients": float(personalized_clients),
            "num_eval_samples": float(sum(len(loader.dataset) for loader in eval_loaders)),
            "num_pgd_batches": total_robust_batches / max(1, personalized_clients),
            "num_pgd_samples": total_robust_samples / max(1, personalized_clients),
        }
        print(
            f"[eval] round {round_idx}: "
            f"personalized_acc={metrics['accuracy']:.4f} "
            f"personalized_robust_acc={metrics['robust_accuracy']:.4f} "
            f"global_acc={metrics['global_accuracy']:.4f} "
            f"global_robust_acc={metrics['global_robust_accuracy']:.4f}"
        )
        record = {
            "type": "eval",
            "round": round_idx,
            "metrics": metrics,
            "pgd": {
                "epsilon": args.pgd_epsilon,
                "step_size": args.pgd_step_size,
                "steps": args.pgd_steps,
                "random_start": args.pgd_random_start,
                "normalized_space": args.pgd_normalized_space,
                "max_batches": args.pgd_max_batches,
            },
            "sylva": {
                "class_balance_power": args.sylva_class_balance_power,
                "class_balance_smoothing": args.sylva_class_balance_smoothing,
                "dynamic_rounds": args.sylva_dynamic_rounds,
                "clean_weight": args.sylva_clean_weight,
                "adv_weight": args.sylva_adv_weight,
                "kl_weight": args.sylva_kl_weight,
                "global_reg": args.sylva_global_reg,
                "agg_temperature": args.sylva_agg_temperature,
                "agg_neighbors": args.sylva_agg_neighbors,
                "phase2_epochs": args.sylva_phase2_epochs,
                "phase2_topk_layers": args.sylva_phase2_topk_layers,
                "phase2_tradeoff": args.sylva_phase2_tradeoff,
                "phase2_lr_scale": args.sylva_phase2_lr_scale,
                "phase2_max_batches": args.sylva_phase2_max_batches,
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
    print("Sylva finished.")


if __name__ == "__main__":
    main()

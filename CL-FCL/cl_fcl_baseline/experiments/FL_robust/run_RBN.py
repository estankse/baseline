from __future__ import annotations

import json
import random
import sys
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from cl_fcl_baseline.algorithms.robust.PGD import evaluate_pgd_robustness
from cl_fcl_baseline.algorithms.robust.RBN import (
    RBNAggregator,
    RBNClient,
    RBNServer,
    enable_dual_batch_norm,
    set_dual_bn_mode,
)
from cl_fcl_baseline.datasets import build_torchvision_dataset, dataset_info
from cl_fcl_baseline.datasets.build import (
    build_dataloader,
    partition_dataset_dirichlet,
    partition_dataset_iid,
    partition_dataset_noniid,
)
from cl_fcl_baseline.experiments.args import parse_rbn_args
from cl_fcl_baseline.experiments.FCL_robust.fcl_robust import build_model as _build_model
from cl_fcl_baseline.experiments.FCL_robust.fcl_robust import build_optimizer, build_pgd_config
from cl_fcl_baseline.trainers.server import FederatedExperiment
from cl_fcl_baseline.trainers.trainer import BaseTrainer
from cl_fcl_baseline.trainers.utils import set_seed


def _build_rbn_model(args, input_shape: tuple[int, int, int], num_classes: int) -> torch.nn.Module:
    model = _build_model(args, input_shape=input_shape, num_classes=num_classes)
    num_replaced = enable_dual_batch_norm(model)
    if num_replaced <= 0:
        raise ValueError(
            f"RBN requires a BatchNorm-based backbone, but model '{args.model}' exposes no BatchNorm layers."
        )
    return model


def _select_at_client_ids(num_clients: int, at_ratio: float, seed: int) -> set[str]:
    clamped_ratio = min(max(float(at_ratio), 0.0), 1.0)
    num_at_clients = max(1, int(round(num_clients * clamped_ratio))) if num_clients > 0 and clamped_ratio > 0.0 else 0
    indices = list(range(num_clients))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return {f"client_{idx}" for idx in sorted(indices[:num_at_clients])}


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
    args = parse_rbn_args()
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
    at_client_ids = _select_at_client_ids(args.num_clients, args.rbn_at_ratio, args.seed)
    if len(at_client_ids) == args.num_clients:
        print(
            "[warn] rbn_at_ratio selects all clients as AT users. "
            "FedRBN's robustness propagation is intended for mixed AT/ST settings."
        )
    clients = []
    exclude_names: set[str] = set()
    for idx, loader in enumerate(loaders):
        client_id = f"client_{idx}"
        model = _build_rbn_model(args, input_shape=input_shape, num_classes=num_classes)
        optimizer = build_optimizer(args, model)
        trainer = BaseTrainer(model=model, optimizer=optimizer, device=device)
        client = RBNClient(
            client_id=client_id,
            trainer=trainer,
            train_loader=loader,
            epochs=args.local_epochs,
            pgd_config=pgd_config,
            is_at_client=client_id in at_client_ids,
            adv_lambda=args.rbn_adv_lambda,
            pnc_coef=args.rbn_pnc,
            pnc_warmup=args.rbn_pnc_warmup,
            src_weight_mode=args.rbn_src_weight_mode,
            attack_noised_bn=args.rbn_attack_noised_bn,
        )
        exclude_names = set(client.local_bn_param_names) | set(client.local_bn_buffer_names)
        clients.append(client)

    server_model = _build_rbn_model(args, input_shape=input_shape, num_classes=num_classes)
    server = RBNServer(
        model=server_model,
        clients=clients,
        aggregator=RBNAggregator(exclude_names=exclude_names),
        client_sample_ratio=args.client_sample_ratio,
        src_weight_mode=args.rbn_src_weight_mode,
        propagate_before_training=args.rbn_pnc >= 0.0,
    )

    eval_model = _build_rbn_model(args, input_shape=input_shape, num_classes=num_classes)
    eval_optimizer = build_optimizer(args, eval_model)
    eval_trainer = BaseTrainer(model=eval_model, optimizer=eval_optimizer, device=device)
    log_path = args.log_file.strip()
    if not log_path:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = str(log_dir / f"rbn_{timestamp}.jsonl")
    else:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    pgd_max_batches = None if int(args.pgd_max_batches) <= 0 else int(args.pgd_max_batches)

    def _eval_round(round_idx: int) -> None:
        global_state = server.get_global_state()
        eval_model.load_state_dict(global_state, strict=False)
        set_dual_bn_mode(eval_model, False)
        global_metrics = eval_trainer.evaluate(test_loader)
        set_dual_bn_mode(eval_model, True)
        global_robust_metrics = evaluate_pgd_robustness(
            eval_model,
            test_loader,
            pgd_config,
            device=device,
            max_batches=pgd_max_batches,
        )
        set_dual_bn_mode(eval_model, False)

        evaluated_clients = 0
        total_accuracy = 0.0
        total_loss = 0.0
        total_robust_accuracy = 0.0
        total_robust_loss = 0.0
        total_robust_batches = 0.0
        total_robust_samples = 0.0
        for client, client_test_loader in zip(clients, eval_loaders):
            eval_model.load_state_dict(client.build_eval_state(global_state), strict=True)
            set_dual_bn_mode(eval_model, False)
            local_metrics = eval_trainer.evaluate(client_test_loader)
            set_dual_bn_mode(eval_model, True)
            robust_metrics = evaluate_pgd_robustness(
                eval_model,
                client_test_loader,
                pgd_config,
                device=device,
                max_batches=pgd_max_batches,
            )
            set_dual_bn_mode(eval_model, False)
            evaluated_clients += 1
            total_accuracy += float(local_metrics.get("accuracy", 0.0))
            total_loss += float(local_metrics.get("loss", 0.0))
            total_robust_accuracy += float(robust_metrics.get("accuracy", 0.0))
            total_robust_loss += float(robust_metrics.get("loss", 0.0))
            total_robust_batches += float(robust_metrics.get("num_batches", 0.0))
            total_robust_samples += float(robust_metrics.get("num_samples", 0.0))

        metrics = {
            "accuracy": total_accuracy / max(1, evaluated_clients),
            "loss": total_loss / max(1, evaluated_clients),
            "robust_accuracy": total_robust_accuracy / max(1, evaluated_clients),
            "robust_loss": total_robust_loss / max(1, evaluated_clients),
            "global_accuracy": float(global_metrics.get("accuracy", 0.0)),
            "global_loss": float(global_metrics.get("loss", 0.0)),
            "global_robust_accuracy": float(global_robust_metrics.get("accuracy", 0.0)),
            "global_robust_loss": float(global_robust_metrics.get("loss", 0.0)),
            "num_eval_clients": float(evaluated_clients),
            "num_eval_samples": float(sum(len(loader.dataset) for loader in eval_loaders)),
            "num_pgd_batches": total_robust_batches / max(1, evaluated_clients),
            "num_pgd_samples": total_robust_samples / max(1, evaluated_clients),
        }
        print(
            f"[eval] round {round_idx}: "
            f"local_acc={metrics['accuracy']:.4f} "
            f"local_robust_acc={metrics['robust_accuracy']:.4f} "
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
            "rbn": {
                "at_ratio": args.rbn_at_ratio,
                "at_clients": sorted(at_client_ids),
                "adv_lambda": args.rbn_adv_lambda,
                "src_weight_mode": args.rbn_src_weight_mode,
                "pnc": args.rbn_pnc,
                "pnc_warmup": args.rbn_pnc_warmup,
                "attack_noised_bn": args.rbn_attack_noised_bn,
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
    print("RBN finished.")


if __name__ == "__main__":
    main()

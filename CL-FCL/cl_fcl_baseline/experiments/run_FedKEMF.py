from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import Subset

from cl_fcl_baseline.algorithms.fl import FedAvgAggregator
from cl_fcl_baseline.algorithms.fedkem import (
    DistillMLP,
    DistillationConfig,
    FedKEMClient,
    FedKEMServerAggregator,
)
from cl_fcl_baseline.datasets import build_torchvision_dataset, dataset_info
from cl_fcl_baseline.datasets.build import (
    RandomClassificationDataset,
    build_dataloader,
    partition_dataset_dirichlet,
    partition_dataset_iid,
    partition_dataset_noniid,
)
from cl_fcl_baseline.models import ResNet18, ResNet20, ResNet32, VGG11
from cl_fcl_baseline.models.simple_model import MLPClassifier, SimpleCNN
from cl_fcl_baseline.trainers.client import FederatedClient
from cl_fcl_baseline.trainers.server import FederatedExperiment, FederatedServer
from cl_fcl_baseline.trainers.trainer import BaseTrainer
from cl_fcl_baseline.trainers.utils import set_seed

try:
    from .args import parse_fedkemf_args
except ImportError:  # pragma: no cover
    from cl_fcl_baseline.experiments.args import parse_fedkemf_args


def main() -> None:
    global netmodel
    args = parse_fedkemf_args()
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

    # Use a subset of the training data as the server's public set.
    server_ratio = float(args.server_data_ratio)
    if not (0.0 < server_ratio <= 1.0):
        raise ValueError("server_data_ratio must be in (0, 1].")
    total_train = len(dataset)
    server_size = max(1, int(total_train * server_ratio))
    server_indices = torch.randperm(
        total_train, generator=torch.Generator().manual_seed(int(args.seed) + 2)
    )[:server_size].tolist()
    server_dataset = Subset(dataset, server_indices)
    server_loader = build_dataloader(server_dataset, batch_size=args.batch_size, shuffle=True)



    clients = []
    for idx, loader in enumerate(loaders):
        if args.model == "mlp":
            model = MLPClassifier(
                input_shape=input_shape,
                hidden_dim=args.hidden_dim,
                num_classes=num_classes,
            )
        elif args.model == "simplecnn":
            model = SimpleCNN(
                input_shape=input_shape,
                num_classes=num_classes,
            )
        elif args.model == "VGG11":
            model = VGG11(
                input_channels=3,
                num_classes=num_classes,
            )
        elif args.model == "ResNet18":
            model = ResNet18(
                input_channels=3,
                num_classes=num_classes,
            )
        elif args.model == "ResNet20":
            model = ResNet20(
                input_channels=3,
                num_classes=num_classes,
            )
        elif args.model == "ResNet32":
            model = ResNet32(
                input_channels=3,
                num_classes=num_classes,
            )
        if args.optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        elif args.optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)
        trainer = BaseTrainer(model=model, optimizer=optimizer, device=device)
        # Always maintain a student in FedKEMF. Distillation config controls
        # whether client-side distillation is enabled.
        student = ResNet18(
            input_channels=3,
            num_classes=num_classes,
        )

        distill_cfg = None
        if args.distill:
            distill_cfg = DistillationConfig(
                epochs=args.distill_epochs,
                temperature=args.distill_temperature,
                alpha=args.distill_alpha,
            )
        clients.append(
            FedKEMClient(
                client_id=f"client_{idx}",
                trainer=trainer,
                train_loader=loader,
                distill_student=student,
                distill_config=distill_cfg,
                epochs=args.local_epochs,
                mutual_learning=args.mutual_learning,
            )
        )

    # Server exchanges student parameters in FedKEMF.
    server_model = ResNet18(
        input_channels=3,
        num_classes=num_classes,
    )

    aggregator = FedKEMServerAggregator(
        model=server_model,
        public_loader=server_loader,
        lr=args.server_distill_lr,
        temperature=args.server_distill_temperature,
        epochs=args.server_distill_epochs,
        device=device,
        ensemble=args.server_ensemble,
    )



    server = FederatedServer(
        model=server_model,
        clients=clients,
        aggregator=aggregator,
        client_sample_ratio=args.client_sample_ratio,
    )

    if args.optimizer == "adam":
        eval_optimizer = torch.optim.Adam(server.model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        eval_optimizer = torch.optim.SGD(server.model.parameters(), lr=args.lr,momentum=0.9)
    eval_trainer = BaseTrainer(
        model=server.model,
        optimizer=eval_optimizer,
        device=device,
    )
    log_path = args.log_file.strip()
    if not log_path:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = str(log_dir / f"fedkemf_{timestamp}.jsonl")

    def _eval_round(round_idx: int) -> None:
        test_metrics = eval_trainer.evaluate(test_loader)
        print(f"[eval] round {round_idx}: acc={test_metrics.get('accuracy', 0.0):.4f}")
        record = {"type": "eval", "round": round_idx, "metrics": test_metrics}
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

    history = experiment.run()
    print("FedKEMF finished.")


if __name__ == "__main__":
    main()

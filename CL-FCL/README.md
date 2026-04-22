# CL-FCL Baseline

A compact, explicit PyTorch baseline for **Federated Learning (FL)** and **Federated Continual Learning (FCL)**, plus implementations of **FedKEMF**, **FedAvg**, **FedProx**, **Scaffold**, **MOON**, and **FedWeIT**.

The code intentionally avoids:
- registries
- YAML configs
- implicit component construction

Everything is created directly in Python so the flow is easy to read and edit.

**What's included**

Federated Learning (FL)
- `FedAvgAggregator`
- `FederatedServer`, `FederatedClient`, `FederatedExperiment`

Federated Continual Learning (FCL)
- `NaiveContinualStrategy`
- `ContinualClient`, `FCLServer`, `FCLExperiment`

FedKEMF (knowledge distillation + multi-model fusion)
- `FedKEMClient`
- `FedKEMServerAggregator`
- `run_FedKEMF.py`

FedWeIT Federated Continual Learning
- `FedWeITClient`
- `FedWeITServer`
- `run_FedWeIT.py`

FedProx (proximal regularization)
- `FedProxClient`, `FedProxTrainer`
- `run_FedProx.py`

MOON (model-contrastive federated learning)
- `MoonClient`, `MoonTrainer`
- `run_MOON.py`

Data + Models
- `cifar10`, `cifar100`, `MNIST`
- `VGG11`, `ResNet18`, `ResNet20`, `ResNet32`


**Quickstart**

Run FedAvg:
```bash
python -m cl_fcl_baseline.experiments.run_FedAvg
```

Run FedKEMF:
```bash
python -m cl_fcl_baseline.experiments.run_FedKEMF
```

Run FedProx:
```bash
python -m cl_fcl_baseline.experiments.run_FedProx
```

Run MOON:
```bash
python -m cl_fcl_baseline.experiments.run_MOON
```

Run FCL:
```bash
python -m cl_fcl_baseline.experiments.run_fcl
```

Run FedWeIT:
```bash
python -m cl_fcl_baseline.experiments.run_FedWeIT
```

**FedWeIT usage notes**

`run_FedWeIT.py` targets the Algorithm 1 FedWeIT flow. Each client maintains `B_c`, task masks `m_c^(t)`, task-adaptive parameters `A_c^(1:t)`, and attention weights `alpha_c^(t)`. The server samples a task knowledge base, distributes it once per client/task, aggregates `B_c^(t,r) * m_c^(t,r)` by client mean, and appends client `A_c^(t)` states into `kb` at task end.

Common options:
- `--dataset` `random_classification` | `mnist` | `cifar10` | `cifar100`
- `--num-tasks` number of continual tasks
- `--classes-per-task` class split size for class-incremental datasets
- `--rounds-per-task` communication rounds per task
- `--local_epochs` local epochs per round
- `--lambda1` FedWeIT `lambda_l1` coefficient for adaptive parameters `A`
- `--lambda2` Eq.2 retroactive coefficient for old-task compensation
- `--lambda-mask` FedWeIT `lambda_mask` coefficient for raw mask variables; the reference default is `0`
- `--kb-sample-size` number of knowledge-base entries sampled per task (`<=0` means all)
- `--client-sparsity` fraction of smallest raw-mask entries pruned from communicable `B*m`
- `--mask-init` task-mask initialization; `<0` uses random raw-mask initialization like the reference implementation

Example:
configuration:
```bash
python -m cl_fcl_baseline.experiments.run_FedWeIT \
  --dataset cifar100 \
  --model ResNet32 \
  --client-sample-ratio 1.0 \
  --num-clients 5 \
  --client-sample-ratio 0.4 \
  --partition noniid \
  --noniid-method dirichlet \
  --dirichlet-beta 0.5 \
  --num-tasks 10 \
  --classes-per-task 10 \
  --rounds-per-task 40 \
  --local_epochs 10 \
  --batch-size 64 \
  --lr 0.005 \
  --optimizer adam \
  --lambda1 0.0005 \
  --lambda2 30 \
  --lambda-mask 0 \
  --kb-sample-size 0 \
  --client-sparsity 0.3 \
  --mask-init -1.0 \
  --adaptive-threshold -1.0 \
```

result:
![FedWeIT](cl_fcl_baseline/analyse/plot-FedWeIT/eval_avg_global/global_avg_accuracy.png)
![FedWeIT](cl_fcl_baseline/analyse/plot-FedWeIT/eval_avg_global/global_avg_loss.png)
![FedWeIT](cl_fcl_baseline/analyse/plot-FedWeIT/eval_compare/all_tasks_accuracy.png)


with robust accuracy:
```bash
python -m cl_fcl_baseline.experiments.run_FedWeIT \
  --dataset cifar100 \
  --model ResNet32 \
  --client-sample-ratio 1.0 \
  --num-clients 5 \
  --client-sample-ratio 0.4 \
  --partition noniid \
  --noniid-method dirichlet \
  --dirichlet-beta 0.5 \
  --num-tasks 10 \
  --classes-per-task 10 \
  --rounds-per-task 20 \
  --local_epochs 10 \
  --batch-size 64 \
  --lr 0.005 \
  --optimizer adam \
  --lambda1 0.0005 \
  --lambda2 30 \
  --lambda-mask 0 \
  --kb-sample-size 0 \
  --client-sparsity 0.3 \
  --mask-init -1.0 \
  --adaptive-threshold -1.0 \
  --pgd-epsilon 8.0/255.0 \
  --pgd-step-size 2.0/255.0 \
  --pgd-steps 10 \
  --pgd-random-start True \
  --pgd-normalized-space False \
  --pgd-max-batches 0 \
```
result:
![FedWeIT](cl_fcl_baseline/analyse/plot-FedWeIT-robust/eval_avg_global/global_avg_accuracy.png)
![FedWeIT](cl_fcl_baseline/analyse/plot-FedWeIT-robust/eval_avg_global/global_avg_robust_accuracy.png)
![FedWeIT](cl_fcl_baseline/analyse/plot-FedWeIT-robust/eval_compare/all_tasks_accuracy.png)
![FedWeIT](cl_fcl_baseline/analyse/plot-FedWeIT-robust/eval_compare/all_tasks_robust_accuracy.png)



**FedAvg usage notes**

FedAvg is the standard federated averaging baseline: each client trains locally and the server aggregates client parameters with sample-size weighting. Important settings live in `cl_fcl_baseline/experiments/args.py` and are parsed by `run_FedAvg.py`.

Common options:
- `--num-clients` total clients
- `--client-sample-ratio` fraction of clients per round
- `--partition` `iid` or `noniid`
- `--noniid-method` `dirichlet` or `shards`
- `--dirichlet-beta` heterogeneity (smaller = more non-IID)
- `--num-rounds` communication rounds
- `--local_epochs` local epochs per round
- `--batch-size` batch size
- `--lr` learning rate
- `--optimizer` `sgd` or `adam`
- `--device` `cpu` | `cuda` | `cuda:0` | `auto`
- `--model` `mlp` | `simplecnn` | `VGG11` | `ResNet18` | `ResNet20` | `ResNet32`
- `--eval-every` evaluation frequency (rounds)
- `--log-file` JSONL log path (empty = auto)

Example (reproducing experiments):
configuration:
```bash
python -m cl_fcl_baseline.experiments.run_FedAvg \
  --seed 0 \
  --dataset cifar10 \
  --model ResNet32 \
  --num-clients 30 \
  --client-sample-ratio 0.4 \
  --partition noniid \
  --noniid-method dirichlet \
  --dirichlet-beta 0.5 \
  --num-rounds 200 \
  --local_epochs 10 \
  --batch-size 64 \
  --lr 0.01 \
  --optimizer sgd \
  --algorithm fedavg \
```
result:
![FedAvg Accuracy](cl_fcl_baseline/analyse/BETA=0.5/plot-FedAvg/eval_accuracy.png)
![FedAvg Accuracy](cl_fcl_baseline/analyse/BETA=0.5/plot-FedAvg/eval_loss.png)


**FedKEMF usage notes**

FedKEMF uses a small knowledge network on the client side and an optional server-side distillation step. Important settings live in `cl_fcl_baseline/experiments/args.py` and are parsed by `run_FedKEMF.py`.

Common options:
- `--num-clients` total clients
- `--client-sample-ratio` fraction of clients per round
- `--partition` `iid` or `noniid`
- `--noniid-method` `dirichlet` or `shards`
- `--dirichlet-beta` heterogeneity (smaller = more non-IID)
- `--num-rounds` communication rounds
- `--local_epochs` local epochs per round
- `--batch-size` batch size
- `--lr` learning rate
- `--model` `mlp` | `simplecnn` | `VGG11` | `ResNet18` | `ResNet20` | `ResNet32`
- `--distill` enable client-side distillation
- `--mutual-learning` enable deep mutual learning
- `--server-distill-epochs` server distillation epochs
- `--server-distill-lr` server distillation learning rate
- `--server-distill-temperature` server distillation temperature
- `--server-ensemble` `max` or `mean`
- `--server-data-ratio` fraction of *training* data used as server public set

Example (reproducing experiments):

configuration:
```bash
python -m cl_fcl_baseline.experiments.run_FedKEMF \
  --seed 0 \
  --dataset cifar10 \
  --model VGG11 \
  --num-clients 30 \
  --client-sample-ratio 0.4 \
  --partition noniid \
  --noniid-method dirichlet \
  --dirichlet-beta 0.5 \
  --num-rounds 200 \
  --distill-epochs 10 \
  --batch-size 64 \
  --lr 0.01 \
  --server-data-ratio 0.6 \
  --optimizer sgd \
  --algorithm fedkemf \
  --distill-temperature 2.0 \
  --server-ensemble max \
  --server-distill-epochs 1 \
  --server-distill-temperature 2.0 \
  --server-distill-lr 0.01 \
```
result:
![FedKEMF Accuracy](cl_fcl_baseline/analyse/BETA=0.5/plots-kemf/eval_accuracy.png)
![FedKEMF Accuracy](cl_fcl_baseline/analyse/BETA=0.5/plots-kemf/eval_loss.png)

configuration:
```bash
python -m cl_fcl_baseline.experiments.run_FedKEMF \
  --seed 0 \
  --dataset cifar10 \
  --model Resnet32 \
  --num-clients 30 \
  --client-sample-ratio 0.4 \
  --partition noniid \
  --noniid-method dirichlet \
  --dirichlet-beta 0.5 \
  --num-rounds 200 \
  --distill-epochs 10 \
  --batch-size 64 \
  --lr 0.01 \
  --server-data-ratio 0.6 \
  --optimizer sgd \
  --algorithm fedkemf \
  --distill-temperature 2.0 \
  --server-ensemble max \
  --server-distill-epochs 1 \
  --server-distill-temperature 2.0 \
  --server-distill-lr 0.01 \
```
result:
![FedKEMF Accuracy](cl_fcl_baseline/analyse/BETA=0.5/plots1-kemf/eval_accuracy.png)
![FedKEMF Accuracy](cl_fcl_baseline/analyse/BETA=0.5/plots1-kemf/eval_loss.png)

**FedProx usage notes**

FedProx adds a proximal regularization term to the client objective to keep local updates close to the current global model. Important settings live in `cl_fcl_baseline/experiments/args.py` and are parsed by `run_FedProx.py`.

Common options:
- `--num-clients` total clients
- `--client-sample-ratio` fraction of clients per round
- `--partition` `iid` or `noniid`
- `--noniid-method` `dirichlet` or `shards`
- `--dirichlet-beta` heterogeneity (smaller = more non-IID)
- `--num-rounds` communication rounds
- `--local_epochs` local epochs per round
- `--batch-size` batch size
- `--lr` learning rate
- `--optimizer` `sgd` or `adam`
- `--device` `cpu` | `cuda` | `cuda:0` | `auto`
- `--model` `mlp` | `simplecnn` | `VGG11` | `ResNet18` | `ResNet20` | `ResNet32`
- `--prox-mu` proximal term coefficient (mu)
- `--eval-every` evaluation frequency (rounds)
- `--log-file` JSONL log path (empty = auto)

Example (reproducing experiments):
configuration:
```bash
python -m cl_fcl_baseline.experiments.run_FedProx   
--seed 0 \   
--dataset cifar10 \   
--model ResNet32 \  
--num-clients 30 \   
--client-sample-ratio 0.4 \  
--partition noniid \   
--noniid-method dirichlet \   
--dirichlet-beta 0.5 \   
--num-rounds 200 \   
--local_epochs 10 \   
--batch-size 64 \   
--lr 0.01 \   
--optimizer sgd \   
--algorithm fedprox \   
--prox-mu 0.01 \
```
result:
![FedProx Accuracy](cl_fcl_baseline/analyse/BETA=0.5/plots-prox/eval_accuracy.png)
![FedProx Accuracy](cl_fcl_baseline/analyse/BETA=0.5/plots-prox/eval_loss.png)

configuration:
```bash
python -m cl_fcl_baseline.experiments.run_FedProx   
--seed 0 \   
--dataset cifar10 \   
--model ResNet32 \  
--num-clients 30 \   
--client-sample-ratio 0.4 \  
--partition noniid \   
--noniid-method dirichlet \   
--dirichlet-beta 0.1 \   
--num-rounds 200 \   
--local_epochs 10 \   
--batch-size 64 \   
--lr 0.01 \   
--optimizer sgd \   
--algorithm fedprox \   
--prox-mu 0.01 \
```
result:
![FedProx Accuracy](cl_fcl_baseline/analyse/BETA=0.1/plot-fedprox/eval_accuracy.png)
![FedProx Accuracy](cl_fcl_baseline/analyse/BETA=0.1/plot-fedprox/eval_loss.png)


**SCAFFOLD usage notes**

SCAFFOLD introduces control variates on server and clients to reduce client-drift under non-IID data. Important settings live in `cl_fcl_baseline/experiments/args.py` and are parsed by `run_scaffold.py`.

Common options:
- `--num-clients` total clients
- `--client-sample-ratio` fraction of clients per round
- `--partition` `iid` or `noniid`
- `--noniid-method` `dirichlet` or `shards`
- `--dirichlet-beta` heterogeneity (smaller = more non-IID)
- `--num-rounds` communication rounds
- `--local_epochs` local epochs per round
- `--batch-size` batch size
- `--lr` local learning rate
- `--global-lr` global update step size for SCAFFOLD
- `--optimizer` `sgd` or `adam`
- `--device` `cpu` | `cuda` | `cuda:0` | `auto`
- `--model` `mlp` | `simplecnn` | `VGG11` | `ResNet18` | `ResNet20` | `ResNet32`
- `--eval-every` evaluation frequency (rounds)
- `--log-file` JSONL log path (empty = auto)

Example (reproducing experiments):
configuration:
```bash
python -m cl_fcl_baseline.experiments.run_scaffold \
  --seed 0 \
  --dataset cifar10 \
  --model ResNet32 \
  --num-clients 30 \
  --client-sample-ratio 0.4 \
  --partition noniid \
  --noniid-method dirichlet \
  --dirichlet-beta 0.5 \
  --num-rounds 200 \
  --local_epochs 10 \
  --batch-size 64 \
  --lr 0.01 \
  --global-lr 1.0 \
  --optimizer sgd \
  --algorithm scaffold \
```
result:
![SCAFFOLD Accuracy](cl_fcl_baseline/analyse/BETA=0.5/plot-SCAFFOLD/eval_accuracy.png)
![SCAFFOLD Accuracy](cl_fcl_baseline/analyse/BETA=0.5/plot-SCAFFOLD/eval_loss.png)

configuration:
```bash
python -m cl_fcl_baseline.experiments.run_scaffold \
  --seed 0 \
  --dataset cifar10 \
  --model ResNet32 \
  --num-clients 30 \
  --client-sample-ratio 0.4 \
  --partition noniid \
  --noniid-method dirichlet \
  --dirichlet-beta 0.1 \
  --num-rounds 200 \
  --local_epochs 10 \
  --batch-size 64 \
  --lr 0.01 \
  --global-lr 1.0 \
  --optimizer sgd \
  --algorithm scaffold \
```
result:
![SCAFFOLD Accuracy](cl_fcl_baseline/analyse/BETA=0.1/plot-SCAFFOLD/eval_accuracy.png)
![SCAFFOLD Accuracy](cl_fcl_baseline/analyse/BETA=0.1/plot-SCAFFOLD/eval_loss.png)


**MOON usage notes**

MOON adds a model-contrastive objective on each client: supervised classification loss plus a contrastive term that aligns local representations with the current global model and separates them from the previous local model. Important settings live in `cl_fcl_baseline/experiments/args.py` and are parsed by `run_MOON.py`.

Common options:
- `--num-clients` total clients
- `--client-sample-ratio` fraction of clients per round
- `--partition` `iid` or `noniid`
- `--noniid-method` `dirichlet` or `shards`
- `--dirichlet-beta` heterogeneity (smaller = more non-IID)
- `--num-rounds` communication rounds
- `--local_epochs` local epochs per round
- `--batch-size` batch size
- `--lr` learning rate
- `--optimizer` `sgd` or `adam`
- `--device` `cpu` | `cuda` | `cuda:0` | `auto`
- `--model` `mlp` | `simplecnn` | `VGG11` | `ResNet18` | `ResNet20` | `ResNet32`
- `--moon-temperature` contrastive temperature (tau)
- `--moon-mu` weight of the contrastive term (mu)
- `--eval-every` evaluation frequency (rounds)
- `--log-file` JSONL log path (empty = auto)

Example (reproducing experiments):
configuration:
```bash
python -m cl_fcl_baseline.experiments.run_MOON \
  --seed 0 \
  --dataset cifar10 \
  --model ResNet32 \
  --num-clients 30 \
  --client-sample-ratio 0.4 \
  --partition noniid \
  --noniid-method dirichlet \
  --dirichlet-beta 0.5 \
  --num-rounds 200 \
  --local_epochs 10 \
  --batch-size 64 \
  --lr 0.01 \
  --optimizer sgd \
  --algorithm moon \
  --moon-temperature 0.5 \
  --moon-mu 1.0 \
```
result:
![MOON Accuracy](cl_fcl_baseline/analyse/BETA=0.5/plot-MOON/eval_accuracy.png)
![MOON Accuracy](cl_fcl_baseline/analyse/BETA=0.5/plot-MOON/eval_loss.png)

**Logs**

Training and evaluation logs are written to `cl_fcl_baseline/experiments/logs/` as JSONL files when `--log-file` is not provided. Each line is a JSON record with `type=train|eval`, `round`, and `metrics`.

**Philosophy**

This project is a baseline for quick experiments:
- explicit construction over configuration files
- minimal abstractions
- readable core logic over features

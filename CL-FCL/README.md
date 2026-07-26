# CL-FCL Baseline

A compact, explicit PyTorch baseline for **Continual Learning (CL)**, **Adversarial Continual Learning (CL+AT)**, **Federated Learning (FL)**, and **Federated Continual Learning (FCL)**, plus implementations of **FedKEMF**, **FedAvg**, **FedProx**, **Scaffold**, **MOON**, and **FedWeIT**.

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

Standalone Continual Learning (CL)
- EWC, GEM, Learning without Forgetting (LwF), and iCaRL
- runners under `cl_fcl_baseline/experiments/CL/`
- algorithms under `cl_fcl_baseline/algorithms/CL/`

Adversarial Continual Learning (CL+AT)
- AFLC-RAER, DAML, FLAIR, and TABA
- clean and PGD-robust continual evaluation
- runners under `cl_fcl_baseline/experiments/CL_robust/`
- algorithms under `cl_fcl_baseline/algorithms/CL_robust/`

FedKEMF (knowledge distillation + multi-model fusion)
- `FedKEMClient`
- `FedKEMServerAggregator`
- `run_FedKEMF.py`

FedWeIT Federated Continual Learning
- `FedWeITClient`
- `FedWeITServer`
- `run_FedWeIT.py`
- `FCL_robust/run_FedWeIT_PGD.py` adds PGD robust evaluation to FedWeIT
- `FCL_robust/run_FedWeIT_FAT.py` combines FedWeIT with local Federated Adversarial Training
- `FCL_robust/run_FedWeIT_SFAT.py` combines FedWeIT with Slack Federated Adversarial Training
- `FCL_robust/run_FedWeIT_CalFAT.py` combines FedWeIT with Calibrated Federated Adversarial Training
- `FCL_robust/run_FedWeIT_RBN.py` combines FedWeIT with FedRBN-style local BatchNorm personalization
- `FCL_robust/run_FedWeIT_Sylva.py` combines FedWeIT with Sylva-inspired personalized adversarial fine-tuning

Standalone robust FL baselines
- `FL_robust/run_FAT.py`
- `FL_robust/run_SFAT.py`
- `FL_robust/run_CalFAT.py`
- `FL_robust/run_RBN.py`
- `FL_robust/run_Sylva.py`

FedProx (proximal regularization)
- `FedProxClient`, `FedProxTrainer`
- `run_FedProx.py`

MOON (model-contrastive federated learning)
- `MoonClient`, `MoonTrainer`
- `run_MOON.py`

Data + Models
- `cifar10`, `cifar100`, `MNIST`
- `VGG11`, `ResNet18`, `ResNet20`, `ResNet32`, `ViTTiny`, `ViTSmall`, `ViTBase`


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

Run standalone CL:
```bash
python -m cl_fcl_baseline.experiments.CL.run_EWC
python -m cl_fcl_baseline.experiments.CL.run_GEM
python -m cl_fcl_baseline.experiments.CL.run_LwF
python -m cl_fcl_baseline.experiments.CL.run_iCaRL
```

Run CL+AT:
```bash
python -m cl_fcl_baseline.experiments.CL_robust.run_AFLC_RAER
python -m cl_fcl_baseline.experiments.CL_robust.run_DAML
python -m cl_fcl_baseline.experiments.CL_robust.run_FLAIR
python -m cl_fcl_baseline.experiments.CL_robust.run_TABA
```

Run FedWeIT:
```bash
python -m cl_fcl_baseline.experiments.run_FedWeIT
```

Robust FedWeIT runners live under `cl_fcl_baseline/experiments/FCL_robust/` and the standalone robust FL baselines live under `cl_fcl_baseline/experiments/FL_robust/`.

Run FedWeIT-PGD:
```bash
python cl_fcl_baseline/experiments/FCL_robust/run_FedWeIT_PGD.py
```

Run FedWeIT-FAT:
```bash
python cl_fcl_baseline/experiments/FCL_robust/run_FedWeIT_FAT.py
```

Run FedWeIT-SFAT:
```bash
python cl_fcl_baseline/experiments/FCL_robust/run_FedWeIT_SFAT.py
```

Run FedWeIT-CalFAT:
```bash
python cl_fcl_baseline/experiments/FCL_robust/run_FedWeIT_CalFAT.py
```

Run FedWeIT-RBN:
```bash
python cl_fcl_baseline/experiments/FCL_robust/run_FedWeIT_RBN.py
```

Run FedWeIT-Sylva:
```bash
python cl_fcl_baseline/experiments/FCL_robust/run_FedWeIT_Sylva.py
```

Run standalone FAT:
```bash
python cl_fcl_baseline/experiments/FL_robust/run_FAT.py
```

Run standalone SFAT:
```bash
python cl_fcl_baseline/experiments/FL_robust/run_SFAT.py
```

Run standalone CalFAT:
```bash
python cl_fcl_baseline/experiments/FL_robust/run_CalFAT.py
```

Run standalone RBN:
```bash
python cl_fcl_baseline/experiments/FL_robust/run_RBN.py
```

Run standalone Sylva:
```bash
python cl_fcl_baseline/experiments/FL_robust/run_Sylva.py
```

Analyse standalone robust logs in `cl_fcl_baseline/experiments/FL_robust/logs`:
```bash
python cl_fcl_baseline/analyse/analyse-fl-robust.py
```

The script writes per-run plots plus latest-per-method comparisons to `cl_fcl_baseline/analyse/plot-FL-robust/`.

**Standalone robust FL usage notes**

The standalone robust runners under `cl_fcl_baseline/experiments/FL_robust/` follow the standard FL data partitioning flow and add robust local training and/or robust evaluation on top of it. They all use the shared argument definitions in `cl_fcl_baseline/experiments/args.py`, emit JSONL logs into `cl_fcl_baseline/experiments/FL_robust/logs/`, and can be visualized with `cl_fcl_baseline/analyse/analyse-fl-robust.py`.

Common options:
- `--num-clients` total clients
- `--client-sample-ratio` fraction of clients sampled each round
- `--partition` `iid` or `noniid`
- `--noniid-method` `dirichlet` or `shards`
- `--dirichlet-beta` heterogeneity for Dirichlet splits
- `--num-rounds` communication rounds
- `--local_epochs` local epochs per round
- `--batch-size` local minibatch size
- `--lr` learning rate
- `--optimizer` `sgd` or `adam`
- `--device` `cpu` | `cuda` | `cuda:0` | `auto`
- `--model` `mlp` | `simplecnn` | `VGG11` | `ResNet18` | `ResNet20` | `ResNet32` | `ViTTiny` | `ViTSmall` | `ViTBase`
- `--eval-every` evaluation frequency in rounds
- `--log-file` JSONL log path (empty = auto)
- `--pgd-epsilon`, `--pgd-step-size`, `--pgd-steps`, `--pgd-random-start` control PGD adversarial example generation and robust evaluation
- `--pgd-normalized-space` treats PGD magnitudes in normalized model-input space instead of raw pixel space
- `--pgd-max-batches` limits robust evaluation batches; `<=0` uses the full test set

### **FAT usage notes**

`run_FAT.py` keeps the server side as plain FedAvg and replaces part of each local minibatch with PGD adversarial examples during client training.

Method-specific options:
- `--fat-adversarial-ratio` target adversarial proportion in each local minibatch after warmup
- `--fat-warmup-rounds` number of initial rounds using the warmup adversarial ratio
- `--fat-warmup-adversarial-ratio` adversarial proportion used during the warmup stage

Example:
```bash
python cl_fcl_baseline/experiments/FL_robust/run_FAT.py \
  --seed 0 \
  --dataset cifar10 \
  --model ResNet18 \
  --num-clients 10 \
  --client-sample-ratio 0.6 \
  --partition noniid \
  --noniid-method dirichlet \
  --dirichlet-beta 0.5 \
  --num-rounds 200 \
  --local_epochs 10 \
  --batch-size 128 \
  --lr 0.0001 \
  --optimizer adam \
  --fat-adversarial-ratio 0.5 \
  --fat-warmup-rounds 0 \
  --fat-warmup-adversarial-ratio 0.1 \
  --pgd-epsilon 0.031372549 \
  --pgd-step-size 0.007843137 \
  --pgd-steps 10
```

result:
![FAT accuracy](cl_fcl_baseline/analyse/plot-FL-robust/runs/fat/eval/accuracy.png)
![FAT robust accuracy](cl_fcl_baseline/analyse/plot-FL-robust/runs/fat/eval/robust_accuracy.png)


### **SFAT usage notes**

`run_SFAT.py` uses the same local adversarial training step as FAT, then applies SFAT's slack-based client reweighting during aggregation.

Method-specific options:
- `--sfat-adversarial-ratio` target adversarial proportion in each local minibatch after warmup
- `--sfat-warmup-rounds` number of initial rounds using the warmup adversarial ratio
- `--sfat-warmup-adversarial-ratio` adversarial proportion used during warmup
- `--sfat-alpha` SFAT slack factor for enhancing selected low-loss clients
- `--sfat-enhanced-clients` number of clients upweighted in each round
- `--sfat-loss-metric` ranking metric used by SFAT, such as `adv_ce_loss`, `ce_loss`, or `loss`

Example:
```bash
python cl_fcl_baseline/experiments/FL_robust/run_SFAT.py \
  --seed 0 \
  --dataset cifar10 \
  --model ResNet18 \
  --num-clients 10 \
  --client-sample-ratio 0.6 \
  --partition noniid \
  --noniid-method dirichlet \
  --dirichlet-beta 0.5 \
  --num-rounds 200 \
  --local_epochs 10 \
  --batch-size 128 \
  --lr 0.0001 \
  --optimizer adam \
  --sfat-adversarial-ratio 0.5 \
  --sfat-warmup-rounds 0 \
  --sfat-warmup-adversarial-ratio 0.1 \
  --sfat-alpha 0.09090909 \
  --sfat-enhanced-clients 1 \
  --sfat-loss-metric adv_ce_loss \
  --pgd-epsilon 0.031372549 \
  --pgd-step-size 0.007843137 \
  --pgd-steps 10
```

result:
![SFAT accuracy](cl_fcl_baseline/analyse/plot-FL-robust/runs/sfat/eval/accuracy.png)
![SFAT robust accuracy](cl_fcl_baseline/analyse/plot-FL-robust/runs/sfat/eval/robust_accuracy.png)

### **CalFAT usage notes**

`run_CalFAT.py` keeps FedAvg aggregation and uses calibrated adversarial supervision based on smoothed local class priors when training each client.

Method-specific options:
- `--calfat-prior-smoothing` smoothing constant added to each client class prior before calibration

Example:
```bash
python cl_fcl_baseline/experiments/FL_robust/run_CalFAT.py \
  --seed 0 \
  --dataset cifar10 \
  --model ResNet18 \
  --num-clients 10 \
  --client-sample-ratio 0.6 \
  --partition noniid \
  --noniid-method dirichlet \
  --dirichlet-beta 0.5 \
  --num-rounds 200 \
  --local_epochs 10 \
  --batch-size 128 \
  --lr 0.0001 \
  --optimizer adam \
  --calfat-prior-smoothing 1e-6 \
  --pgd-epsilon 0.031372549 \
  --pgd-step-size 0.007843137 \
  --pgd-steps 10
```

result:
![CalFAT accuracy](cl_fcl_baseline/analyse/plot-FL-robust/runs/calfat/eval/accuracy.png)
![CalFAT robust accuracy](cl_fcl_baseline/analyse/plot-FL-robust/runs/calfat/eval/robust_accuracy.png)


### **RBN usage notes**

`run_RBN.py` implements a FedRBN-style mixed AT/ST setting: AT clients optimize a robust objective, ST clients receive propagated adversarial BatchNorm statistics, and optional pseudo-noise calibration can be enabled.

Method-specific options:
- `--rbn-at-ratio` fraction of clients treated as adversarial-training users
- `--rbn-adv-lambda` weight on the adversarial loss term for AT users
- `--rbn-src-weight-mode` propagation weighting for adversarial BatchNorm statistics, `eq` or `cos`
- `--rbn-pnc` pseudo-noise calibration coefficient for ST users; `<0` disables PNC
- `--rbn-pnc-warmup` number of initial rounds using zero PNC coefficient
- `--rbn-attack-noised-bn` whether PGD for AT users uses the noised BatchNorm path

`run_RBN.py` requires a BatchNorm-based backbone such as `VGG11`, `ResNet18`, `ResNet20`, or `ResNet32`; pure ViT backbones are not suitable here.

Example:
```bash
python cl_fcl_baseline/experiments/FL_robust/run_RBN.py \
  --seed 0 \
  --dataset cifar10 \
  --model ResNet18 \
  --num-clients 10 \
  --client-sample-ratio 0.6 \
  --partition noniid \
  --noniid-method dirichlet \
  --dirichlet-beta 0.5 \
  --num-rounds 200 \
  --local_epochs 10 \
  --batch-size 128 \
  --lr 0.0001 \
  --optimizer adam \
  --rbn-at-ratio 0.2 \
  --rbn-adv-lambda 0.5 \
  --rbn-src-weight-mode cos \
  --rbn-pnc 0.5 \
  --rbn-pnc-warmup 10 \
  --pgd-epsilon 0.031372549 \
  --pgd-step-size 0.007843137 \
  --pgd-steps 10
```

result:
![RBN accuracy](cl_fcl_baseline/analyse/plot-FL-robust/runs/rbn/eval/accuracy.png)
![RBN robust accuracy](cl_fcl_baseline/analyse/plot-FL-robust/runs/rbn/eval/robust_accuracy.png)

### **Sylva usage notes**

`run_Sylva.py` combines class-balanced adversarial local training, similarity-aware aggregation, and a second benign refinement phase on selected layer groups after each round.

Method-specific options:
- `--sylva-class-balance-power` inverse-frequency exponent used for class-balanced weighting
- `--sylva-class-balance-smoothing` smoothing constant added to local class counts
- `--sylva-dynamic-rounds` number of rounds used to ramp from uniform to imbalance-aware class weights
- `--sylva-clean-weight` weight on the clean cross-entropy term
- `--sylva-adv-weight` weight on the adversarial cross-entropy term
- `--sylva-kl-weight` weight on the TRADES-style KL consistency term
- `--sylva-global-reg` strength of the local-to-global alignment penalty
- `--sylva-agg-temperature` temperature scaling used by similarity-aware aggregation
- `--sylva-agg-neighbors` number of nearest clients used for aggregation; `<=0` uses all peers
- `--sylva-phase2-epochs` benign refinement epochs in the second phase
- `--sylva-phase2-topk-layers` number of layer groups refined in the second phase
- `--sylva-phase2-tradeoff` penalty weight for adversarial sensitivity during layer scoring
- `--sylva-phase2-lr-scale` learning-rate multiplier for the second phase
- `--sylva-phase2-max-batches` maximum batches used during layer scoring and benign refinement; `<=0` uses all batches

Example:
```bash
python cl_fcl_baseline/experiments/FL_robust/run_Sylva.py \
  --seed 0 \
  --dataset cifar10 \
  --model ResNet18 \
  --num-clients 10 \
  --client-sample-ratio 0.6 \
  --partition noniid \
  --noniid-method dirichlet \
  --dirichlet-beta 0.5 \
  --num-rounds 200 \
  --local_epochs 10 \
  --batch-size 128 \
  --lr 0.0001 \
  --optimizer adam \
  --sylva-class-balance-power 0.6 \
  --sylva-class-balance-smoothing 1e-3 \
  --sylva-dynamic-rounds 3 \
  --sylva-clean-weight 0.8 \
  --sylva-adv-weight 1.25 \
  --sylva-kl-weight 8.0 \
  --sylva-global-reg 1e-4 \
  --sylva-agg-temperature 0.7 \
  --sylva-agg-neighbors 2 \
  --sylva-phase2-epochs 10 \
  --sylva-phase2-topk-layers 1 \
  --sylva-phase2-tradeoff 0.7 \
  --sylva-phase2-lr-scale 0.0015 \
  --sylva-phase2-max-batches 10 \
  --pgd-epsilon 0.031372549 \
  --pgd-step-size 0.007843137 \
  --pgd-steps 10
```

result:
![Sylva accuracy](cl_fcl_baseline/analyse/plot-FL-robust/runs/sylva/eval/accuracy.png)
![Sylva robust accuracy](cl_fcl_baseline/analyse/plot-FL-robust/runs/sylva/eval/robust_accuracy.png)


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
  --num-clients 5 \
  --client-sample-ratio 1.0 \
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
![FedWeIT](cl_fcl_baseline/analyse/plot-FCL/FedWeIT/eval_avg_global/global_avg_accuracy.png)
![FedWeIT](cl_fcl_baseline/analyse/plot-FCL/FedWeIT/eval_avg_global/global_avg_loss.png)
![FedWeIT](cl_fcl_baseline/analyse/plot-FCL/FedWeIT/eval_compare/all_tasks_accuracy.png)


### **FedWeIT-PGD**

FedWeIT-PGD keeps the FedWeIT optimization flow unchanged and adds PGD robust evaluation after each task/round.

with robust accuracy:
```bash
python cl_fcl_baseline/experiments/FCL_robust/run_FedWeIT_PGD.py \
  --dataset cifar100 \
  --model ResNet32 \
  --num-clients 5 \
  --client-sample-ratio 1.0 \
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
  --pgd-max-batches 0 \
```
result:
![FedWeIT](cl_fcl_baseline/analyse/plot-FCL-robust/FedWeIT-PGD/eval_avg_global/global_avg_accuracy.png)
![FedWeIT](cl_fcl_baseline/analyse/plot-FCL-robust/FedWeIT-PGD/eval_avg_global/global_avg_robust_accuracy.png)
![FedWeIT](cl_fcl_baseline/analyse/plot-FCL-robust/FedWeIT-PGD/eval_compare/all_tasks_accuracy.png)
![FedWeIT](cl_fcl_baseline/analyse/plot-FCL-robust/FedWeIT-PGD/eval_compare/all_tasks_robust_accuracy.png)

### **FedWeIT-FAT**

FedWeIT-FAT uses the same FedWeIT task/mask/knowledge-base flow, but each local minibatch is mixed with PGD adversarial examples before the FedWeIT update. FAT-specific options:
- `--fat-adversarial-ratio` target adversarial proportion in each local minibatch after warmup
- `--fat-warmup-rounds` number of initial rounds per task using the warmup ratio
- `--fat-warmup-adversarial-ratio` adversarial proportion during warmup
- `--pgd-epsilon`, `--pgd-step-size`, `--pgd-steps`, `--pgd-random-start` control both FAT example generation and robust evaluation

Example:
```bash
python cl_fcl_baseline/experiments/FCL_robust/run_FedWeIT_FAT.py \
  --dataset cifar100 \
  --model ResNet32 \
  --num-clients 5 \
  --client-sample-ratio 1.0 \
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
  --kb-sample-size 0 \
  --fat-adversarial-ratio 0.5 \
  --fat-warmup-rounds 0 \
  --pgd-epsilon 0.031372549 \
  --pgd-step-size 0.007843137 \
  --pgd-steps 10
```
result:
![FedWeIT-FAT](cl_fcl_baseline/analyse/plot-FCL-robust/FedWeIT-FAT/eval_avg_global/global_avg_accuracy.png)
![FedWeIT-FAT](cl_fcl_baseline/analyse/plot-FCL-robust/FedWeIT-FAT/eval_avg_global/global_avg_robust_accuracy.png)
![FedWeIT-FAT](cl_fcl_baseline/analyse/plot-FCL-robust/FedWeIT-FAT/eval_compare/all_tasks_accuracy.png)
![FedWeIT-FAT](cl_fcl_baseline/analyse/plot-FCL-robust/FedWeIT-FAT/eval_compare/all_tasks_robust_accuracy.png)



### **FedWeIT-SFAT**

FedWeIT-SFAT keeps the FedWeIT + adversarial-training local step, then applies SFAT's alpha-slack client reweighting during server aggregation. SFAT-specific options:
- `--sfat-adversarial-ratio` target adversarial proportion in each local minibatch after warmup
- `--sfat-warmup-rounds` number of initial rounds per task using the warmup ratio
- `--sfat-warmup-adversarial-ratio` adversarial proportion during warmup
- `--sfat-alpha` SFAT slack factor used to upweight selected low-loss clients
- `--sfat-enhanced-clients` number of clients upweighted in each round
- `--sfat-loss-metric` client metric used for SFAT ranking
- `--pgd-epsilon`, `--pgd-step-size`, `--pgd-steps`, `--pgd-random-start` control both adversarial example generation and robust evaluation

Example:
```bash
python cl_fcl_baseline/experiments/FCL_robust/run_FedWeIT_SFAT.py \
  --dataset cifar100 \
  --model ResNet32 \
  --num-clients 5 \
  --client-sample-ratio 1.0 \
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
  --kb-sample-size 0 \
  --sfat-adversarial-ratio 0.5 \
  --sfat-alpha 0.09090909 \
  --sfat-enhanced-clients 1 \
  --sfat-loss-metric adv_ce_loss \
  --pgd-epsilon 0.031372549 \
  --pgd-step-size 0.007843137 \
  --pgd-steps 10
```
result:
![FedWeIT-SFAT](cl_fcl_baseline/analyse/plot-FCL-robust/FedWeIT-SFAT/eval_avg_global/global_avg_accuracy.png)
![FedWeIT-SFAT](cl_fcl_baseline/analyse/plot-FCL-robust/FedWeIT-SFAT/eval_avg_global/global_avg_robust_accuracy.png)
![FedWeIT-SFAT](cl_fcl_baseline/analyse/plot-FCL-robust/FedWeIT-SFAT/eval_compare/all_tasks_accuracy.png)
![FedWeIT-SFAT](cl_fcl_baseline/analyse/plot-FCL-robust/FedWeIT-SFAT/eval_compare/all_tasks_robust_accuracy.png)

### **FedWeIT-CalFAT**

FedWeIT-CalFAT keeps the FedWeIT task/mask/knowledge-base flow and uses calibrated adversarial supervision based on smoothed local class priors. CalFAT-specific options:
- `--calfat-prior-smoothing` smoothing constant added to class priors before calibration
- `--pgd-epsilon`, `--pgd-step-size`, `--pgd-steps`, `--pgd-random-start` control both adversarial example generation and robust evaluation

Example:
```bash
python cl_fcl_baseline/experiments/FCL_robust/run_FedWeIT_CalFAT.py \
  --dataset cifar100 \
  --model ResNet32 \
  --num-clients 5 \
  --client-sample-ratio 1.0 \
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
  --kb-sample-size 0 \
  --calfat-prior-smoothing 1e-6 \
  --pgd-epsilon 0.031372549 \
  --pgd-step-size 0.007843137 \
  --pgd-steps 10
```
result:
![FedWeIT-CalFAT](cl_fcl_baseline/analyse/plot-FCL-robust/FedWeIT-CalFAT/eval_avg_global/global_avg_accuracy.png)
![FedWeIT-CalFAT](cl_fcl_baseline/analyse/plot-FCL-robust/FedWeIT-CalFAT/eval_avg_global/global_avg_robust_accuracy.png)
![FedWeIT-CalFAT](cl_fcl_baseline/analyse/plot-FCL-robust/FedWeIT-CalFAT/eval_compare/all_tasks_accuracy.png)
![FedWeIT-CalFAT](cl_fcl_baseline/analyse/plot-FCL-robust/FedWeIT-CalFAT/eval_compare/all_tasks_robust_accuracy.png)


### **FedWeIT-RBN**

FedWeIT-RBN combines FedWeIT with FedRBN-style local BatchNorm personalization: a subset of AT users trains with adversarial loss, ST users receive propagated adversarial BatchNorm statistics, and optional PNC regularization can be enabled. RBN-specific options:
- `--rbn-at-ratio` fraction of clients treated as AT users
- `--rbn-adv-lambda` weight on the adversarial loss term for AT users
- `--rbn-src-weight-mode` propagation weighting for BatchNorm statistics (`eq` or `cos`)
- `--rbn-pnc` pseudo-noise calibration coefficient for ST users; `<0` disables PNC
- `--rbn-pnc-warmup` number of initial task rounds with zero PNC coefficient
- `--rbn-attack-noised-bn` whether PGD for AT users uses the noised BatchNorm path
- `--pgd-epsilon`, `--pgd-step-size`, `--pgd-steps`, `--pgd-random-start` control both adversarial example generation and robust evaluation

FedWeIT-RBN requires a BatchNorm-based backbone such as `VGG11`, `ResNet18`, `ResNet20`, or `ResNet32`.


Example:
```bash
python cl_fcl_baseline/experiments/FCL_robust/run_FedWeIT_RBN.py \
  --dataset cifar100 \
  --model ResNet32 \
  --num-clients 5 \
  --client-sample-ratio 1.0 \
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
  --kb-sample-size 0 \
  --rbn-at-ratio 1.0 \
  --rbn-adv-lambda 0.5 \
  --rbn-src-weight-mode cos \
  --rbn-pnc 0.5 \
  --rbn-pnc-warmup 10 \
  --pgd-epsilon 0.031372549 \
  --pgd-step-size 0.007843137 \
  --pgd-steps 10
```

result:
![FedWeIT-RBN](cl_fcl_baseline/analyse/plot-FCL-robust/FedWeIT-RBN/eval_avg_global/global_avg_accuracy.png)
![FedWeIT-RBN](cl_fcl_baseline/analyse/plot-FCL-robust/FedWeIT-RBN/eval_avg_global/global_avg_robust_accuracy.png)
![FedWeIT-RBN](cl_fcl_baseline/analyse/plot-FCL-robust/FedWeIT-RBN/eval_compare/all_tasks_accuracy.png)
![FedWeIT-RBN](cl_fcl_baseline/analyse/plot-FCL-robust/FedWeIT-RBN/eval_compare/all_tasks_robust_accuracy.png)


### **FedWeIT-Sylva**

FedWeIT-Sylva combines FedWeIT with Sylva-inspired personalized adversarial fine-tuning: class-balanced adversarial local training, similarity-aware aggregation, and a second benign refinement phase on selected layer groups after each task. Sylva-specific options:
- `--sylva-class-balance-power` inverse-frequency exponent for class-balanced local loss
- `--sylva-dynamic-rounds` number of rounds used to ramp from uniform to imbalance-aware weights
- `--sylva-clean-weight`, `--sylva-adv-weight`, `--sylva-kl-weight` weights for clean CE, adversarial CE, and TRADES-style KL consistency
- `--sylva-global-reg` local-to-global shared-parameter alignment penalty
- `--sylva-agg-temperature`, `--sylva-agg-neighbors` similarity-aware aggregation controls
- `--sylva-phase2-epochs`, `--sylva-phase2-topk-layers`, `--sylva-phase2-tradeoff`, `--sylva-phase2-lr-scale`, `--sylva-phase2-max-batches` control the post-task benign refinement phase
- `--pgd-epsilon`, `--pgd-step-size`, `--pgd-steps`, `--pgd-random-start` control both adversarial example generation and robust evaluation

Example:
```bash
python cl_fcl_baseline/experiments/FCL_robust/run_FedWeIT_Sylva.py \
  --dataset cifar100 \
  --model ResNet32 \
  --num-clients 5 \
  --client-sample-ratio 1.0 \
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
  --kb-sample-size 0 \
  --sylva-class-balance-power 0.6 \
  --sylva-dynamic-rounds 3 \
  --sylva-clean-weight 0.8 \
  --sylva-adv-weight 1.25 \
  --sylva-kl-weight 8.0 \
  --sylva-global-reg 1e-4 \
  --sylva-agg-temperature 0.7 \
  --sylva-agg-neighbors 2 \
  --sylva-phase2-epochs 10 \
  --sylva-phase2-topk-layers 1 \
  --sylva-phase2-tradeoff 0.7 \
  --sylva-phase2-lr-scale 0.0015 \
  --sylva-phase2-max-batches 10 \
  --pgd-epsilon 0.031372549 \
  --pgd-step-size 0.007843137 \
  --pgd-steps 10
```
result:
![FedWeIT-Sylva](cl_fcl_baseline/analyse/plot-FCL-robust/FedWeIT-Sylva/eval_avg_global/global_avg_accuracy.png)
![FedWeIT-Sylva](cl_fcl_baseline/analyse/plot-FCL-robust/FedWeIT-Sylva/eval_avg_global/global_avg_robust_accuracy.png)
![FedWeIT-Sylva](cl_fcl_baseline/analyse/plot-FCL-robust/FedWeIT-Sylva/eval_compare/all_tasks_accuracy.png)
![FedWeIT-Sylva](cl_fcl_baseline/analyse/plot-FCL-robust/FedWeIT-Sylva/eval_compare/all_tasks_robust_accuracy.png)


**Standalone CL usage notes**

The runners under `cl_fcl_baseline/experiments/CL/` train one model over a sequence of tasks and write JSONL logs to `cl_fcl_baseline/experiments/CL/logs/`. Evaluation records contain the current average accuracy, average forgetting, backward transfer, and per-task accuracy.

Common options:
- `--scenario` `class`, `task`, or `domain`
- `--dataset` `random_classification`, `mnist`, `cifar10`, or `cifar100`
- `--num-tasks` number of tasks; `0` derives it from the dataset and `--classes-per-task`
- `--classes-per-task` number of classes introduced by each class split
- `--task-order-shuffle` / `--no-task-order-shuffle` control class-order randomization
- `--epochs` local training epochs per task
- `--batch-size`, `--lr`, `--optimizer`, `--momentum`, and `--weight-decay` control optimization
- `--eval-every` evaluation interval in epochs; `<=0` evaluates only at task end
- `--log-file` explicit JSONL path; an empty value uses the runner's automatic name

### **EWC**

EWC estimates parameter importance with the Fisher information after each task and penalizes changes to parameters important to earlier tasks.

Method-specific options:
- `--ewc-lambda` weight of the EWC quadratic penalty
- `--fisher-samples` maximum samples used to estimate the Fisher information
- `--fisher-mode` `sampled` for sampled-label Fisher or `empirical` for ground-truth-label Fisher

Example:
```bash
python -m cl_fcl_baseline.experiments.CL.run_EWC \
  --scenario task \
  --dataset cifar100 \
  --model ResNet32 \
  --classes-per-task 10 \
  --epochs 20 \
  --batch-size 64 \
  --optimizer sgd \
  --lr 0.001 \
  --ewc-lambda 400 \
  --fisher-samples 200 \
  --fisher-mode sampled
```

result:
![EWC average accuracy](cl_fcl_baseline/analyse/plot-CL/runs/cl_ewc_20260717_125354/eval_average/average_accuracy.png)
![EWC task accuracy](cl_fcl_baseline/analyse/plot-CL/runs/cl_ewc_20260717_125354/eval_compare/all_tasks_accuracy.png)

### **GEM**

GEM stores episodic samples from previous tasks and projects a new-task gradient when it would increase a remembered task's loss.

Method-specific options:
- `--memory-size` episodic examples retained per task
- `--memory-strength` GEM margin parameter
- `--qp-eps` diagonal stabilization used by the gradient-projection QP

Example:
```bash
python -m cl_fcl_baseline.experiments.CL.run_GEM \
  --scenario task \
  --dataset cifar100 \
  --model ResNet32 \
  --classes-per-task 10 \
  --epochs 20 \
  --batch-size 64 \
  --optimizer sgd \
  --lr 0.001 \
  --memory-size 256 \
  --memory-strength 0.5 \
  --qp-eps 0.001
```

result:
![GEM average accuracy](cl_fcl_baseline/analyse/plot-CL/runs/cl_gem_20260717_124443/eval_average/average_accuracy.png)
![GEM task accuracy](cl_fcl_baseline/analyse/plot-CL/runs/cl_gem_20260717_124443/eval_compare/all_tasks_accuracy.png)

### **LwF**

Learning without Forgetting snapshots the previous model and distills its old-task predictions while learning the current task.

Method-specific options:
- `--temperature` distillation temperature
- `--distillation-weight` weight of the old-task distillation loss
- `--warmup-epochs` new-task-only warmup epochs before distillation

Example:
```bash
python -m cl_fcl_baseline.experiments.CL.run_LwF \
  --scenario task \
  --dataset cifar100 \
  --model ResNet32 \
  --classes-per-task 10 \
  --epochs 30 \
  --batch-size 128 \
  --optimizer sgd \
  --lr 0.001 \
  --weight-decay 0.0005 \
  --temperature 2 \
  --distillation-weight 1 \
  --warmup-epochs 1
```

result:
![LwF average accuracy](cl_fcl_baseline/analyse/plot-CL/runs/cl_lwf_20260717_200855/eval_average/average_accuracy.png)
![LwF task accuracy](cl_fcl_baseline/analyse/plot-CL/runs/cl_lwf_20260717_200855/eval_compare/all_tasks_accuracy.png)

### **iCaRL**

iCaRL combines exemplar rehearsal, distillation, herding-based exemplar selection, and nearest-mean-of-exemplars prediction for class-incremental learning.

Method-specific options:
- `--memory-budget` fixed total exemplar budget
- `--lr-decay` learning-rate decay factor used by the task schedule
- `--prototype-flip` / `--no-prototype-flip` control horizontal-flip averaging for CIFAR prototypes

Example:
```bash
python -m cl_fcl_baseline.experiments.CL.run_iCaRL \
  --scenario class \
  --dataset cifar100 \
  --model ResNet32 \
  --classes-per-task 10 \
  --task-order-shuffle \
  --epochs 70 \
  --batch-size 128 \
  --optimizer sgd \
  --lr 2.0 \
  --weight-decay 0.00001 \
  --memory-budget 2000 \
  --lr-decay 5 \
  --prototype-flip
```

result:
![iCaRL average accuracy](cl_fcl_baseline/analyse/plot-CL/runs/cl_icarl_20260718_131432/eval_average/average_accuracy.png)
![iCaRL task accuracy](cl_fcl_baseline/analyse/plot-CL/runs/cl_icarl_20260718_131432/eval_compare/all_tasks_accuracy.png)

**Adversarial continual-learning (CL+AT) usage notes**

AFLC-RAER, DAML, FLAIR, and TABA live under `cl_fcl_baseline/experiments/CL_robust/`. They use class-incremental replay plus adversarial training and log both clean and PGD-robust continual metrics to `cl_fcl_baseline/experiments/CL_robust/logs/`.

Common CL+AT options include the standalone CL options above plus:
- `--memory-budget` fixed total replay-memory budget
- `--replay-batch-size` replay minibatch size; `0` matches the stream batch size
- `--pgd-epsilon`, `--pgd-step-size`, and `--pgd-steps` control the training attack
- `--pgd-random-start` / `--no-pgd-random-start` toggle random PGD initialization
- `--pgd-normalized-space` interprets attack magnitudes in normalized input space
- `--eval-pgd-steps` and `--eval-pgd-step-size` independently control robust evaluation
- `--pgd-max-batches` limits robust evaluation; `<=0` uses the full evaluation set

### **AFLC-RAER**

AFLC applies anti-forgetting logit calibration during adversarial replay, while RAER selects replay samples according to their PGD robustness difficulty.

Method-specific options:
- `--aflc-alpha` logit-calibration strength
- `--raer-threshold` maximum PGD success count accepted by RAER
- `--future-prior` / `--no-future-prior` toggle the future-head prior
- `--adaptive-eval-attack` attacks through AFLC calibration during evaluation

Example:
```bash
python -m cl_fcl_baseline.experiments.CL_robust.run_AFLC_RAER \
  --dataset cifar100 \
  --model ResNet18 \
  --classes-per-task 10 \
  --epochs 30 \
  --batch-size 32 \
  --optimizer sgd \
  --lr 0.1 \
  --memory-budget 2000 \
  --aflc-alpha 3.5 \
  --raer-threshold 5 \
  --future-prior \
  --pgd-epsilon 0.031372549 \
  --pgd-step-size 0.007843137 \
  --pgd-steps 10 \
  --eval-pgd-steps 10
```

result:
![AFLC-RAER clean average accuracy](cl_fcl_baseline/analyse/plot-CL-robust/runs/cl_robust_aflc_raer_20260725_225622/eval_average/clean_average_accuracy.png)
![AFLC-RAER robust average accuracy](cl_fcl_baseline/analyse/plot-CL-robust/runs/cl_robust_aflc_raer_20260725_225622/eval_average/robust_average_accuracy.png)
![AFLC-RAER clean task accuracy](cl_fcl_baseline/analyse/plot-CL-robust/runs/cl_robust_aflc_raer_20260725_225622/eval_compare/all_tasks_accuracy.png)
![AFLC-RAER robust task accuracy](cl_fcl_baseline/analyse/plot-CL-robust/runs/cl_robust_aflc_raer_20260725_225622/eval_compare/all_tasks_robust_accuracy.png)

### **DAML**

DAML combines a unified BCE distillation target on adversarial stream/replay samples with an additional loss on a disjoint memory minibatch.

Method-specific option:
- `--daml-alpha` weight of the additional memory classification loss

Example:
```bash
python -m cl_fcl_baseline.experiments.CL_robust.run_DAML \
  --dataset cifar100 \
  --model ResNet18 \
  --classes-per-task 10 \
  --epochs 30 \
  --batch-size 128 \
  --optimizer adam \
  --lr 0.0005 \
  --memory-budget 2000 \
  --daml-alpha 0.2 \
  --pgd-epsilon 0.031372549 \
  --pgd-step-size 0.007843137 \
  --pgd-steps 10 \
  --eval-pgd-steps 10
```

result:
![DAML clean average accuracy](cl_fcl_baseline/analyse/plot-CL-robust/runs/cl_robust_daml_20260725_225618/eval_average/clean_average_accuracy.png)
![DAML robust average accuracy](cl_fcl_baseline/analyse/plot-CL-robust/runs/cl_robust_daml_20260725_225618/eval_average/robust_average_accuracy.png)
![DAML clean task accuracy](cl_fcl_baseline/analyse/plot-CL-robust/runs/cl_robust_daml_20260725_225618/eval_compare/all_tasks_accuracy.png)
![DAML robust task accuracy](cl_fcl_baseline/analyse/plot-CL-robust/runs/cl_robust_daml_20260725_225618/eval_compare/all_tasks_robust_accuracy.png)

### **FLAIR**

FLAIR combines adversarial distillation (ADSL) with a flatness-preserving distillation penalty (FPD).

Method-specific options:
- `--flair-alpha` ADSL distillation weight
- `--flair-beta` FPD weight

Example:
```bash
python -m cl_fcl_baseline.experiments.CL_robust.run_FLAIR \
  --dataset cifar100 \
  --model ResNet18 \
  --classes-per-task 10 \
  --epochs 30 \
  --batch-size 32 \
  --optimizer sgd \
  --lr 0.1 \
  --memory-budget 2000 \
  --flair-alpha 1 \
  --flair-beta 1 \
  --pgd-epsilon 0.031372549 \
  --pgd-step-size 0.007843137 \
  --pgd-steps 10 \
  --eval-pgd-steps 10
```

result:
![FLAIR clean average accuracy](cl_fcl_baseline/analyse/plot-CL-robust/runs/cl_robust_flair_20260725_225621/eval_average/clean_average_accuracy.png)
![FLAIR robust average accuracy](cl_fcl_baseline/analyse/plot-CL-robust/runs/cl_robust_flair_20260725_225621/eval_average/robust_average_accuracy.png)
![FLAIR clean task accuracy](cl_fcl_baseline/analyse/plot-CL-robust/runs/cl_robust_flair_20260725_225621/eval_compare/all_tasks_accuracy.png)
![FLAIR robust task accuracy](cl_fcl_baseline/analyse/plot-CL-robust/runs/cl_robust_flair_20260725_225621/eval_compare/all_tasks_robust_accuracy.png)

### **TABA**

TABA augments adversarial continual learning with task-aware interpolation between old-task boundary samples and new-task samples.

Method-specific options:
- `--taba-mix-batch-size` boundary-mix minibatch size; `0` matches the stream batch size
- `--taba-lambda-min` and `--taba-lambda-max` bound the interpolation coefficient

Example:
```bash
python -m cl_fcl_baseline.experiments.CL_robust.run_TABA \
  --dataset cifar100 \
  --model ResNet18 \
  --num-tasks 10 \
  --classes-per-task 10 \
  --epochs 30 \
  --batch-size 128 \
  --optimizer sgd \
  --lr 0.01 \
  --weight-decay 0.00001 \
  --memory-budget 2000 \
  --taba-lambda-min 0.45 \
  --taba-lambda-max 0.55 \
  --pgd-epsilon 0.031372549 \
  --pgd-step-size 0.007843137 \
  --pgd-steps 7 \
  --eval-pgd-steps 7
```

result:
![TABA clean average accuracy](cl_fcl_baseline/analyse/plot-CL-robust/runs/cl_robust_taba_20260723_234553/eval_average/clean_average_accuracy.png)
![TABA robust average accuracy](cl_fcl_baseline/analyse/plot-CL-robust/runs/cl_robust_taba_20260723_234553/eval_average/robust_average_accuracy.png)
![TABA clean task accuracy](cl_fcl_baseline/analyse/plot-CL-robust/runs/cl_robust_taba_20260723_234553/eval_compare/all_tasks_accuracy.png)
![TABA robust task accuracy](cl_fcl_baseline/analyse/plot-CL-robust/runs/cl_robust_taba_20260723_234553/eval_compare/all_tasks_robust_accuracy.png)

**CL and CL+AT log analysis**

Run both analysers from the repository root:
```bash
python cl_fcl_baseline/analyse/analyse-cl.py
python cl_fcl_baseline/analyse/analyse-cl-robust.py
```

Their default inputs and outputs are:
- `cl_fcl_baseline/experiments/CL/logs/*.jsonl` -> `cl_fcl_baseline/analyse/plot-CL/`
- `cl_fcl_baseline/experiments/CL_robust/logs/*.jsonl` -> `cl_fcl_baseline/analyse/plot-CL-robust/`

The remaining result families follow the same directory convention:
- FCL results: `cl_fcl_baseline/analyse/plot-FCL/<method>/`
- robust FCL results: `cl_fcl_baseline/analyse/plot-FCL-robust/<method>/`
- robust FL results: `cl_fcl_baseline/analyse/plot-FL-robust/`

Each run receives task-wise training curves, global continual-evaluation curves, historical-task comparison curves, and a `summary.json`. CL+AT additionally plots clean/robust averages, per-task robust accuracy/loss, and AFLC-RAER replay-memory records. `compare_latest/` compares the newest log for each algorithm on a normalized task-progress axis, so runs with different epoch counts remain aligned.

Custom directories can be selected without editing the scripts:
```bash
python cl_fcl_baseline/analyse/analyse-cl.py \
  --log-dir path/to/cl/logs \
  --output-dir path/to/cl/plots

python cl_fcl_baseline/analyse/analyse-cl-robust.py \
  --log-dir path/to/cl_at/logs \
  --output-dir path/to/cl_at/plots
```

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
- `--model` `mlp` | `simplecnn` | `VGG11` | `ResNet18` | `ResNet20` | `ResNet32` | `ViTTiny` | `ViTSmall` | `ViTBase`
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
- `--model` `mlp` | `simplecnn` | `VGG11` | `ResNet18` | `ResNet20` | `ResNet32` | `ViTTiny` | `ViTSmall` | `ViTBase`
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
- `--model` `mlp` | `simplecnn` | `VGG11` | `ResNet18` | `ResNet20` | `ResNet32` | `ViTTiny` | `ViTSmall` | `ViTBase`
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
- `--model` `mlp` | `simplecnn` | `VGG11` | `ResNet18` | `ResNet20` | `ResNet32` | `ViTTiny` | `ViTSmall` | `ViTBase`
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
- `--model` `mlp` | `simplecnn` | `VGG11` | `ResNet18` | `ResNet20` | `ResNet32` | `ViTTiny` | `ViTSmall` | `ViTBase`
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

Training and evaluation logs are JSONL files. FL/FCL runners use `cl_fcl_baseline/experiments/logs/`, standalone CL runners use `cl_fcl_baseline/experiments/CL/logs/`, CL+AT runners use `cl_fcl_baseline/experiments/CL_robust/logs/`, and standalone robust FL runners use `cl_fcl_baseline/experiments/FL_robust/logs/`. The record schema follows the experiment family: FL/FCL records use rounds, while CL and CL+AT records use task indices and epochs.

**Philosophy**

This project is a baseline for quick experiments:
- explicit construction over configuration files
- minimal abstractions
- readable core logic over features

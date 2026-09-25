<div align="center">

# SCOT: Multi-Source Cross-City Transfer with Optimal-Transport Soft-Correspondence Objectives

**Yuyao Wang, Min Yang, Meng Chen, Weiming Huang, Yilong Yin, Yongshun Gong**

Boston University · Shandong University · University of Leeds

*Advances in Neural Information Processing Systems (NeurIPS), 2026*

[[Paper]](https://arxiv.org/abs/2604.07383) &nbsp; [[Slides]](https://github.com/olivia3395/olivia3395.github.io/blob/main/_pages/SCOT.pdf)

</div>

---

This repository contains the official PyTorch implementation of **SCOT**, a framework for single- and multi-source cross-city transfer of urban region representations.

## Abstract

Cross-city transfer aims to improve region-level prediction (e.g., GDP, population, and CO₂ emissions) in label-scarce target cities by exploiting supervision available in other cities. A central difficulty is that cities rarely share compatible region partitions, and no ground-truth correspondence between regions is available. SCOT addresses this by learning explicit soft correspondences through Sinkhorn-based entropic optimal transport, which yields a many-to-many coupling between region sets of unequal size without requiring one-to-one node matching. The coupling drives an OT alignment objective, an OT-weighted contrastive objective that mitigates over-mixing under cross-city heterogeneity, and a cycle-style reconstruction term that stabilizes training. For the multi-source setting, SCOT replaces independent pairwise alignments with a shared hub of learnable prototypes, together with a target-induced prototype marginal that allocates hub capacity to target-relevant semantics. Experiments on multiple Chinese cities show consistent improvements over alignment-based baselines in both single- and multi-source settings.

<div align="center">
<img src="pipeline.png" width="80%" alt="Overview of SCOT"/>
<br/>
<sub><b>Figure 1.</b> Overview of SCOT. GAT encoders produce region embeddings for each city; Sinkhorn OT computes a soft correspondence <b>P</b> between source and target regions, which is used by the alignment, contrastive, and reconstruction objectives.</sub>
</div>

## Method

**Single-source transfer.** Let $\mathbf{Z}_s \in \mathbb{R}^{n_s \times d}$ and $\mathbf{Z}_t \in \mathbb{R}^{n_t \times d}$ denote the normalized region embeddings of the source and target cities, where in general $n_s \neq n_t$. Given a cross-city cost matrix $\mathbf{C}$ computed from these embeddings, SCOT obtains a soft correspondence by solving the entropic OT problem

$$
\mathbf{P}^\star = \arg\min_{\mathbf{P} \in \Pi(\mathbf{a}, \mathbf{b})} \; \langle \mathbf{P}, \mathbf{C} \rangle - \varepsilon H(\mathbf{P}),
$$

where $\Pi(\mathbf{a}, \mathbf{b})$ is the set of couplings with marginals $\mathbf{a}$ and $\mathbf{b}$, and $\varepsilon$ controls the entropic regularization. The problem is solved with $T$ Sinkhorn iterations. The resulting coupling $\mathbf{P} \in \mathbb{R}^{n_s \times n_t}$ is shared by three objectives: an OT alignment loss, an OT-weighted contrastive (InfoNCE-style) loss that uses $\mathbf{P}$ to weight positive pairs, and a cycle-style reconstruction loss.

**Multi-source transfer.** Aligning each source to the target independently can lead to conflicting gradients when sources are heterogeneous. SCOT instead aligns every city to a shared hub of $K$ learnable prototypes via balanced entropic OT. The prototype marginal $\mathbf{b}$ is induced from the target city, so that hub capacity concentrates on semantics relevant to the target.

<div align="center">
<img src="hub.png" width="70%" alt="Multi-source hub alignment"/>
<br/>
<sub><b>Figure 2.</b> Multi-source hub alignment. Each city is aligned to <i>K</i> shared prototypes through balanced entropic OT, guided by a target-induced prototype prior.</sub>
</div>

## Results

We report MAE and MAPE (%) on GDP, population, and CO₂ prediction; lower is better. The last row gives the relative improvement of SCOT over the strongest baseline for each metric. City abbreviations: BJ (Beijing), XA (Xi'an), CD (Chengdu).

**Table 1.** Single-source transfer (XA → BJ).

| Method | GDP MAE | GDP MAPE | Pop. MAE | Pop. MAPE | CO₂ MAE | CO₂ MAPE |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Non-Align | 264.30 | 12.08 | 981.07 | 8.56 | 288.41 | 6.40 |
| MMD | 183.32 | 5.71 | 588.34 | 3.17 | 165.70 | 2.54 |
| CoRE | 157.83 | 5.46 | 611.18 | 4.05 | 166.28 | 2.95 |
| **SCOT** | **115.33** | **3.17** | **528.50** | **2.13** | **149.42** | **1.79** |
| Rel. improvement | 26.9% | 41.9% | 10.2% | 32.8% | 9.8% | 29.5% |

**Table 2.** Multi-source transfer (target: BJ).

| Method | GDP MAE | GDP MAPE | Pop. MAE | Pop. MAPE | CO₂ MAE | CO₂ MAPE |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| MMD | 127.45 | 4.93 | 605.81 | 4.11 | 160.52 | 1.29 |
| CoRE | 152.88 | 5.86 | 620.34 | 4.30 | 152.24 | 1.99 |
| **SCOT** | **104.16** | **2.57** | **525.10** | **1.87** | **143.53** | **1.16** |
| Rel. improvement | 18.3% | 47.9% | 13.3% | 54.5% | 5.7% | 10.1% |

## Installation

The code requires Python ≥ 3.9 and PyTorch ≥ 2.4. Install PyTorch Geometric for your CUDA/CPU configuration, then the remaining dependencies:

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
pip install torch-geometric
pip install -r requirements.txt
```

## Usage

Single-source transfer (e.g., BJ → CD):

```bash
python train.py --source BJ --target CD \
    --eps_ot 0.15 --lambda_align 1.0 --eta 0.5 --tau 0.1
```

Multi-source transfer with the prototype hub (e.g., BJ + XA → CD):

```bash
python train_hub.py --sources BJ XA --target CD \
    --K 32 --use_target_prior 1 --tau_b 0.5
```

The main hyperparameters are listed below. In our experiments, a single global configuration transferred well across city pairs, and per-target tuning was rarely necessary.

| Script | Argument | Description | Default |
|:---|:---|:---|:---:|
| `train.py` | `--eps_ot` | Entropic regularization $\varepsilon$ | 0.15 |
| | `--sinkhorn_iters` | Number of Sinkhorn iterations $T$ | 50 |
| | `--lambda_align` | Weight of the OT alignment loss | 1.0 |
| | `--lambda_rec` | Weight of the reconstruction loss | 0.5 |
| | `--eta` | Weight of the contrastive loss | 0.5 |
| | `--tau` | Contrastive temperature | 0.1 |
| | `--beta` | Weight of the entropy penalty | 0.05 |
| `train_hub.py` | `--K` | Number of hub prototypes | 32 |
| | `--use_target_prior` | Use the target-induced prototype marginal | 1 |
| | `--tau_b` | Temperature of the target prior | 0.5 |



## Contact

For questions, please contact Yuyao Wang (yuyaow@bu.edu) or the corresponding author, Yongshun Gong (ysgong@sdu.edu.cn).

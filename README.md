
# **SCOT**: *Multi-Source Cross-City Transfer with Optimal-Transport Soft-Correspondence Objectives*.

SCOT learns **explicit soft correspondences** between cities/regions using **Sinkhorn-based entropic OT**, enhanced by **OT-guided contrastive alignment** and **cycle-style reconstruction**.  
For **multi-source transfer**, SCOT aligns each source and the target to a shared **prototype hub** via **balanced entropic transport** (optionally with a **target-induced prototype prior**).


## Features
- Entropic OT (Sinkhorn) soft correspondence \(P\) between unequal region sets
- OT + OT-weighted contrastive (InfoNCE-style) alignment
- Cycle-style reconstruction regularization
- Multi-source hub prototypes (K) for scalable multi-city transfer
- Alignment diagnostics (OT heatmap / hub usage / PCA-tSNE)



## Setup

### Single-source transfer (e.g., BJ → CD)

```bash
python train.py 
```

### Multi-source transfer with hub (e.g., BJ + XA → CD)

```bash
python train_hub.py
```


## Key Args

| Arg                  | Meaning                                  | Typical     |
| -------------------- | ---------------------------------------- | ----------- |
| `--eps_ot`           | OT entropic regularization (\varepsilon) | `0.10~0.20` |
| `--sinkhorn_iters`   | Sinkhorn iterations                      | `30~100`    |
| `--lambda_align`     | alignment weight                         | `1.0`       |
| `--lambda_rec`       | reconstruction weight                    | `1.0`       |
| `--eta`              | contrastive weight                       | `0.1~1.0`   |
| `--tau`              | contrastive temperature                  | `0.1~0.5`   |
| `--K`                | # hub prototypes (multi-source)          | `16/32/64`  |
| `--use_target_prior` | target-induced prototype marginal        | `0/1`       |






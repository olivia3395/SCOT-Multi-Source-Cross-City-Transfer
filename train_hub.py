import numpy as np
import tqdm
from evaluator.evaluator_cross_city_2source import Evaluator
from torch.utils.data import DataLoader
import pickle
import torch
from utils import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from optim_schedule import ScheduledOptim
import pandas as pd
import random
from data import *
import argparse
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


from Model.CoREPP_OT_hub import CoREPP_2Source_Hub

import torch.nn.functional as F


import numpy as np
import torch
import torch.nn.functional as F

def _entropy(p, dim=-1, eps=1e-8):
    return -(p * torch.log(p + eps)).sum(dim=dim)
@torch.no_grad()
def hub_city_diag(z_city, anchors, Pi, name, tau=0.1, topk=8):
    """
    z_city: (n,d) tensor
    anchors: (K,d) tensor
    Pi: (n,K) balanced coupling from sinkhorn
    Prints:
      - row_sum stats (should be ~ 1/n if balanced)
      - Q row-entropy (assignment sharpness)
      - softmax entropy over anchors
      - mass_pos = E[ sum_k Q_ik * softmax(S_ik) ]  in (0,1)
      - top used anchors
    """
    device = z_city.device
    z = F.normalize(z_city, dim=-1)
    a = F.normalize(anchors, dim=-1)

    # Balanced Pi row sum ~ 1/n; make Q row-stochastic
    row_sum = Pi.sum(dim=1, keepdim=True) + 1e-8
    Q = Pi / row_sum  # (n,K), each row sums to 1

    # Similarity -> softmax probs over anchors
    sim = (z @ a.t()) / tau
    sim = sim - sim.max(dim=1, keepdim=True)[0]
    p_soft = F.softmax(sim, dim=1)  # (n,K)

    # Diagnostics
    q_ent = _entropy(Q, dim=1).mean()
    p_ent = _entropy(p_soft, dim=1).mean()
    mass_pos = torch.sum(Q * p_soft, dim=1).mean()  # interpretable, in (0,1)

    # Anchor usage (prob mass per anchor)
    usage = Q.mean(dim=0)  # (K,), sums to 1
    vals, idx = torch.topk(usage, k=min(topk, usage.numel()))
    idx_np = idx.detach().cpu().numpy()
    vals_np = vals.detach().cpu().numpy()

    # Top-1 assignment histogram (how many nodes pick each anchor)
    top1 = torch.argmax(Q, dim=1)
    hist = torch.bincount(top1, minlength=Q.size(1)).float()
    hist = hist / hist.sum()

    print(f"\n[HubDiag:{name}] n={z_city.size(0)}, K={anchors.size(0)}")
    print(f"  row_sum: mean={row_sum.mean().item():.6e}, std={row_sum.std().item():.3e} "
          f"(balanced expects ~1/n={1.0/z_city.size(0):.3e})")
    print(f"  Entropy:  H(Q)={q_ent.item():.3f} (<=logK={np.log(anchors.size(0)):.3f}), "
          f"H(softmax)={p_ent.item():.3f}")
    print(f"  mass_pos (E[Q·softmax]) = {mass_pos.item():.6f}  (should be not ~1e-4 after Q-normalization)")

    print("  Top used anchors by usage Q.mean(0):")
    for k, v in zip(idx_np, vals_np):
        print(f"    anchor[{k:02d}] usage={v:.4f} | top1_freq={hist[k].item():.4f}")

    return usage.detach(), hist.detach(), mass_pos.detach()




def hub_cross_city_diag(hist_s1, hist_s2, hist_t, name_s1="s1", name_s2="s2", name_t="t", eps=1e-8):
    def normalize(p):
        p = torch.clamp(p, min=eps)
        return p / p.sum()

    def js(p, q):
        p, q = normalize(p), normalize(q)
        m = 0.5 * (p + q)
        return 0.5 * (torch.sum(p * (torch.log(p) - torch.log(m))) +
                      torch.sum(q * (torch.log(q) - torch.log(m))))

    h1, h2, ht = normalize(hist_s1), normalize(hist_s2), normalize(hist_t)

    cos12 = F.cosine_similarity(h1, h2, dim=0).item()
    cos1t = F.cosine_similarity(h1, ht, dim=0).item()
    cos2t = F.cosine_similarity(h2, ht, dim=0).item()

    js12 = js(h1, h2).item()
    js1t = js(h1, ht).item()
    js2t = js(h2, ht).item()

    print(f"\n[HubCrossDiag-Hard] top1 histogram")
    print(f"  cosine({name_s1},{name_s2})={cos12:.3f} | cosine({name_s1},{name_t})={cos1t:.3f} | cosine({name_s2},{name_t})={cos2t:.3f}")
    print(f"  JS({name_s1},{name_s2})={js12:.4f} | JS({name_s1},{name_t})={js1t:.4f} | JS({name_s2},{name_t})={js2t:.4f}")


# =========================
# Visualization helpers
# =========================
def plot_pca(s_emb, t_emb, path, s_name="Source", t_name="Target"):
    s_np = s_emb.detach().cpu().numpy()
    t_np = t_emb.detach().cpu().numpy()

    X = np.concatenate([s_np, t_np], axis=0)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(7, 6))
    plt.scatter(X_pca[:len(s_np), 0], X_pca[:len(s_np), 1],
                s=5, alpha=0.6, label=s_name, c='#1f77b4')
    plt.scatter(X_pca[len(s_np):, 0], X_pca[len(s_np):, 1],
                s=5, alpha=0.6, label=t_name, c='#ff7f0e')

    plt.legend()
    plt.title(f"PCA Embeddings ({s_name} vs {t_name})")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[Saved] PCA -> {path}")


def visualize_tsne(Z_s, Z_t, gdp_s=None, gdp_t=None, save_path="tsne.png",
                   s_name="Source", t_name="Target"):
    Zs = Z_s.detach().cpu().numpy()
    Zt = Z_t.detach().cpu().numpy()

    Z = np.vstack([Zs, Zt])
    labels = np.array([0] * len(Zs) + [1] * len(Zt))

    if gdp_s is not None and gdp_t is not None:
        gdp = np.concatenate([gdp_s, gdp_t])
        use_color = True
    else:
        gdp = np.ones(len(Z))
        use_color = False

    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    Z_2d = tsne.fit_transform(Z)

    plt.figure(figsize=(7, 6))
    idx_s = labels == 0
    idx_t = labels == 1

    if use_color:
        colors = []
        q1, q2 = np.percentile(gdp, [33, 66])
        for v in gdp:
            if v <= q1:
                colors.append("yellow")
            elif v <= q2:
                colors.append("green")
            else:
                colors.append("red")
    else:
        colors = ["blue" if l == 0 else "orange" for l in labels]

    plt.scatter(Z_2d[idx_s, 0], Z_2d[idx_s, 1],
                marker="o", c=np.array(colors)[idx_s], label=s_name, alpha=0.7)
    plt.scatter(Z_2d[idx_t, 0], Z_2d[idx_t, 1],
                marker="^", c=np.array(colors)[idx_t], label=t_name, alpha=0.7)

    plt.title(f"t-SNE Visualization ({s_name} vs {t_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Saved] t-SNE -> {save_path}")


@torch.no_grad()
def _row_stochastic(P: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # P: (n, K)
    return P / (P.sum(dim=1, keepdim=True) + eps)


def _tsne_2d(X: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    X: (N, D) numpy
    returns: (N, 2)
    """
    N = X.shape[0]
    # Make perplexity valid for small/large N
    # Rule of thumb: 5~50, and must be < N
    perp = int(np.clip(N / 50, 10, 50))
    perp = min(perp, max(5, (N - 1) // 3))  # hard constraint for sklearn TSNE

    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        learning_rate="auto",
        init="pca",
        random_state=seed,
        max_iter=1000,
        verbose=0,
    )
    return tsne.fit_transform(X)


def save_tsne_triplet(
    out_dir: str,
    epoch: int,
    s1_emb: torch.Tensor, s2_emb: torch.Tensor, t_emb: torch.Tensor,
    Pi1: torch.Tensor, Pi2: torch.Tensor, Pit: torch.Tensor,
    anchors: torch.Tensor,
    seed: int = 0,
    point_size: int = 5,
    alpha: float = 0.7,
):
    """
    Saves three figures:
      1) raw-tSNE on concat([s1_emb, s2_emb, t_emb])
      2) Q-tSNE on concat([Q1, Q2, Qt]) where Q is row-stochastic coupling
      3) (Q@A)-tSNE on concat([Q1@A, Q2@A, Qt@A]) (hub-projected embeddings)

    Call this inside your training loop once every k epochs.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---- prepare labels ----
    n1, n2, nt = s1_emb.size(0), s2_emb.size(0), t_emb.size(0)
    labels = np.array([0] * n1 + [1] * n2 + [2] * nt)

    # ---- 1) raw embeddings ----
    Z_raw = torch.cat([s1_emb, s2_emb, t_emb], dim=0).detach().cpu().numpy()
    Z_raw_2d = _tsne_2d(Z_raw, seed=seed)

    plt.figure(figsize=(7, 6))
    for c, name in [(0, "source1"), (1, "source2"), (2, "target")]:
        idx = labels == c
        plt.scatter(Z_raw_2d[idx, 0], Z_raw_2d[idx, 1], s=point_size, alpha=alpha, label=name)
    plt.title(f"raw-tSNE (epoch {epoch})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"tsne_raw_epoch_{epoch}.png"), dpi=200)
    plt.close()

    # ---- 2) Q space (row-stochastic) ----
    Q1 = _row_stochastic(Pi1).detach().cpu().numpy()
    Q2 = _row_stochastic(Pi2).detach().cpu().numpy()
    Qt = _row_stochastic(Pit).detach().cpu().numpy()
    Z_Q = np.concatenate([Q1, Q2, Qt], axis=0)
    Z_Q_2d = _tsne_2d(Z_Q, seed=seed)

    plt.figure(figsize=(7, 6))
    for c, name in [(0, "source1"), (1, "source2"), (2, "target")]:
        idx = labels == c
        plt.scatter(Z_Q_2d[idx, 0], Z_Q_2d[idx, 1], s=point_size, alpha=alpha, label=name)
    plt.title(f"Q-tSNE (epoch {epoch})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"tsne_Q_epoch_{epoch}.png"), dpi=200)
    plt.close()

    # ---- 3) hub-projected: Q @ A ----
    A = anchors.detach().cpu().numpy()  # (K, d)
    Z_proj = np.concatenate([Q1 @ A, Q2 @ A, Qt @ A], axis=0)  # (N, d)
    Z_proj_2d = _tsne_2d(Z_proj, seed=seed)

    plt.figure(figsize=(7, 6))
    for c, name in [(0, "source1"), (1, "source2"), (2, "target")]:
        idx = labels == c
        plt.scatter(Z_proj_2d[idx, 0], Z_proj_2d[idx, 1], s=point_size, alpha=alpha, label=name)
    plt.title(f"(Q@A)-tSNE (epoch {epoch})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"tsne_QA_epoch_{epoch}.png"), dpi=200)
    plt.close()



def plot_ot_heatmap(pi, path, max_size=300, log_scale=True, top_k=None, title="Transport Plan",
                    xlab="Target", ylab="Source"):
    """
    Works for:
      - city->city OT: (Ns, Nt)
      - city->hub OT:  (Ns, K)
    """
    pi_np = pi.detach().cpu().numpy()
    Ns, Nt = pi_np.shape

    idx_s = np.random.choice(Ns, min(Ns, max_size), replace=False)
    idx_t = np.random.choice(Nt, min(Nt, max_size), replace=False)
    pi_sub = pi_np[np.ix_(idx_s, idx_t)]

    if top_k is not None:
        flat = pi_sub.flatten()
        thresh = np.partition(flat, -top_k)[-top_k]
        pi_sub = np.where(pi_sub >= thresh, pi_sub, 0.0)

    if log_scale:
        pi_plot = np.log(pi_sub + 1e-8)
    else:
        pi_plot = pi_sub

    vmin = np.percentile(pi_plot, 2)
    vmax = np.percentile(pi_plot, 98)

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        pi_plot,
        cmap="magma",
        vmin=vmin, vmax=vmax,
        cbar=True,
        square=True
    )
    plt.title(title + (" (log-scaled)" if log_scale else ""))
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[Saved] Heatmap -> {path}")


# =========================
# Collapse check helper
# =========================
def check_embedding_stats(emb, name):
    with torch.no_grad():
        norms = torch.norm(emb, dim=1)
        avg_norm = norms.mean().item()
        std_val = torch.std(emb).item()

        sample_size = min(500, emb.size(0))
        idx = torch.randperm(emb.size(0))[:sample_size]
        sample = emb[idx]

        sample_norm = F.normalize(sample, dim=1)
        sim_matrix = torch.matmul(sample_norm, sample_norm.T)
        mask = ~torch.eye(sample_size, dtype=torch.bool, device=emb.device)
        avg_sim = sim_matrix[mask].mean().item()

    return avg_norm, std_val, avg_sim


# =========================
# Tee logger
# =========================
class Tee:
    def __init__(self, filepath):
        self.file = open(filepath, "w", buffering=1)  # line-buffered
        self.stdout = sys.stdout

    def write(self, msg):
        self.stdout.write(msg)
        self.file.write(msg)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


if __name__ == '__main__':
    epochs = 300
    d_feature = 128

    exp_id = int(random.SystemRandom().random() * 1000000)
    exp_cache_file = './output/{}'.format(exp_id)
    ensure_cache_dir(exp_cache_file)

    save_path = os.path.join(exp_cache_file, "train_cache")
    ensure_cache_dir(save_path)

    # =========================
    # Multi-source setting: 2 sources -> 1 target
    # =========================

    # s1_dataset = 'cd'
    # s2_dataset = 'xa'
    # t_dataset  = 'bj'

    s1_dataset = 'cd'
    s2_dataset = 'bj'
    t_dataset  = 'xa'

    # s1_dataset  = 'bj'
    # s2_dataset = 'xa'
    # t_dataset = 'cd'

    print(f'learning {s1_dataset},{s2_dataset} --- {t_dataset}')

    # load data
    s1_region_num, s1_graph_info = load_data(s1_dataset)
    s2_region_num, s2_graph_info = load_data(s2_dataset)
    t_region_num,  t_graph_info  = load_data(t_dataset)

    # =========================
    #  Model: Shared HUB (B1) instead of scalar-gating (A)
    # =========================
    model = CoREPP_2Source_Hub(
        s1_region_num=s1_region_num,
        s2_region_num=s2_region_num,
        t_region_num=t_region_num,
        s1_graph_info=s1_graph_info,
        s2_graph_info=s2_graph_info,
        t_graph_info=t_graph_info,
        hidden_dim=128,
        gat_layers=2,
        num_heads=8,
        lambda_align=1.0,
        lambda_rec=0.5,
        tau=0.1,
        beta=0.05,
        hub_K=64,  # shared semantic hub size
        ot_reg=0.15,
        ot_iter=50,
        contrast_w=0.5,
        hub_bal_w=0.0,    # 0.1  # avoid dead anchors
        debug=True
    )

    evaluator = Evaluator(exp_id=exp_id)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    # =========================
    # Logging
    # =========================
    log_file = os.path.join(save_path, f"train_log_exp{exp_id}.txt")
    sys.stdout = Tee(log_file)
    print(f"[Logging] stdout is tee'd to: {log_file}")

    VIS_EVERY = 20

    for epoch_id in range(epochs):
        model.train()

        # forward (HUB)
        # returns: (L_s1, L_s2, L_t, L_rec, L_align, total, (Pi1,Pi2,Pit))
        L_s1, L_s2, L_t, L_rec, L_align, total_loss, (Pi1, Pi2, Pit) = model.forward()

        optim.zero_grad()
        total_loss.backward()
        optim.step()

        print(f"Epoch {epoch_id+1}/{epochs} | total={total_loss.item():.4f} | "
              f"Align={L_align.item():.4f} | Rec={L_rec.item():.4f} | "
              f"Intra_s1={L_s1.item():.4f} | Intra_s2={L_s2.item():.4f} | Intra_t={L_t.item():.4f}")

        # ==========================
        # Collapse check
        # ==========================
        if (epoch_id + 1) % 5 == 0:
            s1_emb_check, s2_emb_check, t_emb_check = model.get_region_emb()

            s1_norm, s1_std, s1_sim = check_embedding_stats(s1_emb_check, "Source1")
            s2_norm, s2_std, s2_sim = check_embedding_stats(s2_emb_check, "Source2")
            t_norm,  t_std,  t_sim  = check_embedding_stats(t_emb_check,  "Target")

            print(f"\n[Collapse Check Epoch {epoch_id+1}]")
            print(f"  Source1 > Norm: {s1_norm:.4f} | Std: {s1_std:.4f} | AvgSim: {s1_sim:.4f}")
            print(f"  Source2 > Norm: {s2_norm:.4f} | Std: {s2_std:.4f} | AvgSim: {s2_sim:.4f}")
            print(f"  Target  > Norm: {t_norm:.4f}  | Std: {t_std:.4f}  | AvgSim: {t_sim:.4f}")
            print(f"  Raw Sample (S1[0]): {s1_emb_check[0, :5].detach().cpu().numpy()}")
            print(f"  Raw Sample (S2[0]): {s2_emb_check[0, :5].detach().cpu().numpy()}")
            print("-" * 50)
            # === Hub diagnostics ===
            # with torch.no_grad():
            #     anchors = model.anchors.detach()

            #     u1, h1, m1 = hub_city_diag(s1_emb_vis, anchors, Pi1, name=f"{s1_dataset}", tau=model.tau, topk=8)
            #     u2, h2, m2 = hub_city_diag(s2_emb_vis, anchors, Pi2, name=f"{s2_dataset}", tau=model.tau, topk=8)
            #     ut, ht, mt = hub_city_diag(t_emb_vis,  anchors, Pit, name=f"{t_dataset}",  tau=model.tau, topk=8)

            #     hub_cross_city_diag(u1, u2, ut, name_s1=s1_dataset, name_s2=s2_dataset, name_t=t_dataset)
         
        # ==========================
        # Visualization
        # ==========================
        if (epoch_id + 1) % VIS_EVERY == 0 or epoch_id == 0:
            s1_emb_vis, s2_emb_vis, t_emb_vis = model.get_region_emb()

            # PCA (source vs target)
            plot_pca(s1_emb_vis, t_emb_vis,
                     f"{save_path}/pca_{s1_dataset}_{t_dataset}_epoch_{epoch_id+1}.png",
                     s_name=f"Source1({s1_dataset})", t_name=f"Target({t_dataset})")
            plot_pca(s2_emb_vis, t_emb_vis,
                     f"{save_path}/pca_{s2_dataset}_{t_dataset}_epoch_{epoch_id+1}.png",
                     s_name=f"Source2({s2_dataset})", t_name=f"Target({t_dataset})")

            # PCA (target vs hub anchors) for interpretability
            try:
                anchors = model.anchors.detach()
                plot_pca(t_emb_vis, anchors,
                         f"{save_path}/pca_{t_dataset}_hub_epoch_{epoch_id+1}.png",
                         s_name=f"Target({t_dataset})", t_name="Hub(anchors)")
            except Exception as e:
                print(f"[Warning] Cannot visualize hub PCA: {e}")
                # ---- NEW: raw-tSNE, Q-tSNE, (Q@A)-tSNE (3-domain) ----
            try:

                save_tsne_triplet(
                    out_dir=save_path,
                    epoch=epoch_id + 1,
                    s1_emb=s1_emb_vis,
                    s2_emb=s2_emb_vis,
                    t_emb=t_emb_vis,
                    Pi1=Pi1,
                    Pi2=Pi2,
                    Pit=Pit,
                    anchors=model.anchors,
                    seed=0,
                )
            except Exception as e:
                print(f"[Warning] Cannot save 3-way t-SNE triplet: {e}")


            # # t-SNE (source vs target)
            # try:
            #     visualize_tsne(s1_emb_vis, t_emb_vis,
            #                    save_path=f"{save_path}/tsne_{s1_dataset}_{t_dataset}_epoch_{epoch_id+1}.png",
            #                    s_name=f"Source1({s1_dataset})", t_name=f"Target({t_dataset})")
            #     visualize_tsne(s2_emb_vis, t_emb_vis,
            #                    save_path=f"{save_path}/tsne_{s2_dataset}_{t_dataset}_epoch_{epoch_id+1}.png",
            #                    s_name=f"Source2({s2_dataset})", t_name=f"Target({t_dataset})")
            # except Exception as e:
            #     print(f"[Warning] Cannot visualize t-SNE: {e}")

            # Hub transport heatmaps: (city -> hub)
            try:
                plot_ot_heatmap(
                    Pi1,
                    f"{save_path}/hub_plan_{s1_dataset}_epoch_{epoch_id+1}.png",
                    title=f"Hub Plan: {s1_dataset} -> Hub",
                    xlab="Hub prototypes",
                    ylab=f"{s1_dataset} regions"
                )
                plot_ot_heatmap(
                    Pi2,
                    f"{save_path}/hub_plan_{s2_dataset}_epoch_{epoch_id+1}.png",
                    title=f"Hub Plan: {s2_dataset} -> Hub",
                    xlab="Hub prototypes",
                    ylab=f"{s2_dataset} regions"
                )
                plot_ot_heatmap(
                    Pit,
                    f"{save_path}/hub_plan_{t_dataset}_epoch_{epoch_id+1}.png",
                    title=f"Hub Plan: {t_dataset} -> Hub",
                    xlab="Hub prototypes",
                    ylab=f"{t_dataset} regions"
                )
            except Exception as e:
                print(f"[Warning] Cannot visualize hub plan: {e}")

        # ==========================
        # Save embeddings + Evaluate (MULTI-SOURCE)
        # ==========================
        if (epoch_id + 1) % 5 == 0 or epoch_id == 0:
            s1_emb, s2_emb, t_emb = model.get_region_emb()
            s1_np = s1_emb.detach().cpu().numpy()
            s2_np = s2_emb.detach().cpu().numpy()
            t_np  = t_emb.detach().cpu().numpy()

            np.save(f"{save_path}/{s1_dataset}_region_emb_{epoch_id+1}.npy", s1_np)
            np.save(f"{save_path}/{s2_dataset}_region_emb_{epoch_id+1}.npy", s2_np)
            np.save(f"{save_path}/{t_dataset}_region_emb_{epoch_id+1}.npy",  t_np)

            # (Optional) save hub anchors
            anchors_np = model.anchors.detach().cpu().numpy()
            np.save(f"{save_path}/hub_anchors_{epoch_id+1}.npy", anchors_np)

            #  True multi-source evaluation: concatenate sources to train ONE regressor -> test on target.
            # Since HUB removes scalar gating, we typically use NO weights (None) or uniform weights.
            evaluator.evaluate_multisource(
                s_datasets=[s1_dataset, s2_dataset],
                t_dataset=t_dataset,
                s_region_emb_list=[s1_np, s2_np],
                t_region_emb=t_np,
                w_list=None   # or [0.5, 0.5] if you want explicit uniform weighting
            )
            print("")


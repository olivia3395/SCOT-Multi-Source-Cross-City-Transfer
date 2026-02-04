import numpy as np
import tqdm
# from Model.CoRE import *
from evaluator.evaluator_cross_city import Evaluator
from torch.utils.data import DataLoader
import pickle
import torch
from utils import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from optim_schedule import ScheduledOptim
import pandas as pd
import random
# import faiss
from data import *
import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
# # import umap.umap_ as umap


# def plot_umap(s_emb, t_emb, save_path, n_neighbors=30, min_dist=0.1):
#     """
#     #     """
#     import numpy as np
#     import matplotlib.pyplot as plt

#     X = np.vstack([s_emb, t_emb])
#     labels = np.array([0] * len(s_emb) + [1] * len(t_emb))

#     reducer = umap.UMAP(
#         n_neighbors=n_neighbors,
#         min_dist=min_dist,
#         metric='cosine',
#         random_state=42
#     )
#     X_umap = reducer.fit_transform(X)

#     plt.figure(figsize=(8, 7))
#     plt.scatter(X_umap[labels==0, 0], X_umap[labels==0, 1],
#                 s=6, alpha=0.7, label="Source", color="#1f77b4")
#     plt.scatter(X_umap[labels==1, 0], X_umap[labels==1, 1],
#                 s=6, alpha=0.7, label="Target", color="#ff7f0e")

#     plt.title("UMAP Embeddings (Source vs Target)")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.close()

#     print(f"[Saved] UMAP -> {save_path}")


def plot_pca(s_emb, t_emb, path):
    s_np = s_emb.detach().cpu().numpy()
    t_np = t_emb.detach().cpu().numpy()

    X = np.concatenate([s_np, t_np], axis=0)
    labels = np.array([0]*len(s_np) + [1]*len(t_np))


    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(7, 6))
    plt.scatter(X_pca[:len(s_np), 0], X_pca[:len(s_np), 1],
                s=5, alpha=0.6, label='Source', c='#1f77b4')
    plt.scatter(X_pca[len(s_np):, 0], X_pca[len(s_np):, 1],
                s=5, alpha=0.6, label='Target', c='#ff7f0e')

    plt.legend()
    plt.title("PCA Embeddings (Source vs Target)")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"[Saved] PCA -> {path}")



def visualize_tsne(Z_s, Z_t, gdp_s=None, gdp_t=None, save_path="tsne.png"):
    """
    Z_s: (Ns, d) source city embeddings (torch.tensor or np.array)
    Z_t: (Nt, d) target city embeddings
    gdp_s, gdp_t: optional arrays for coloring (same length as Z_s/Z_t)
    """

    # convert to numpy
    Zs = Z_s.detach().cpu().numpy()
    Zt = Z_t.detach().cpu().numpy()
    
    # stack for TSNE
    Z = np.vstack([Zs, Zt])
    labels = np.array([0] * len(Zs) + [1] * len(Zt))  # 0=source, 1=target

    # GDP for coloring (optional)
    if gdp_s is not None and gdp_t is not None:
        gdp = np.concatenate([gdp_s, gdp_t])
        use_color = True
    else:
        # fallback: assign all ones
        gdp = np.ones(len(Z))
        use_color = False

    # TSNE projection
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    Z_2d = tsne.fit_transform(Z)

    # plotting
    plt.figure(figsize=(7, 6))


    idx_s = labels == 0
    idx_t = labels == 1

    if use_color:
        # GDP coloring: low→yellow, mid→green, high→red
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

    # scatter
    plt.scatter(Z_2d[idx_s, 0], Z_2d[idx_s, 1], 
                marker="o", c=np.array(colors)[idx_s], label="Source City", alpha=0.7)
    plt.scatter(Z_2d[idx_t, 0], Z_2d[idx_t, 1], 
                marker="^", c=np.array(colors)[idx_t], label="Target City", alpha=0.7)

    plt.title("t-SNE Visualization of Region Embeddings (Figure 2 style)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()






# def plot_ot_heatmap(pi, path, max_size=300):
#     pi_np = pi.detach().cpu().numpy()

#     Ns, Nt = pi_np.shape

#     idx_s = np.random.choice(Ns, min(Ns, max_size), replace=False)
#     idx_t = np.random.choice(Nt, min(Nt, max_size), replace=False)

#     pi_sub = pi_np[np.ix_(idx_s, idx_t)]

#     plt.figure(figsize=(7, 6))
#     sns.heatmap(pi_sub, cmap="viridis")
#     plt.title("OT Transport Plan Heatmap")
#     plt.tight_layout()
#     plt.savefig(path, dpi=300)
#     plt.close()
#     print(f"[Saved] OT Heatmap -> {path}")


def plot_ot_heatmap(pi, path, max_size=300, log_scale=True, top_k=None):
    """
    Improved OT heatmap visualization.
    - log scaling to enhance contrast
    - dynamic color range
    - optional top-k mass visualization
    """

    pi_np = pi.detach().cpu().numpy()
    Ns, Nt = pi_np.shape

    # 1. Subsample for visualization
    idx_s = np.random.choice(Ns, min(Ns, max_size), replace=False)
    idx_t = np.random.choice(Nt, min(Nt, max_size), replace=False)
    pi_sub = pi_np[np.ix_(idx_s, idx_t)]

    # 2. Optional: keep only top-k mass entries
    if top_k is not None:
        flat = pi_sub.flatten()
        thresh = np.partition(flat, -top_k)[-top_k]
        pi_sub = np.where(pi_sub >= thresh, pi_sub, 0.0)

    # 3. Log-scaling (VERY helpful for readability)
    if log_scale:
        pi_plot = np.log(pi_sub + 1e-8)   # log transform
    else:
        pi_plot = pi_sub

    # 4. Normalize for better contrast
    vmin = np.percentile(pi_plot, 2)     # lower clipping to remove background noise
    vmax = np.percentile(pi_plot, 98)    # upper clipping to enhance bright areas

    # 5. Plot
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        pi_plot,
        cmap="magma",               # more contrast than viridis
        vmin=vmin, vmax=vmax,
        cbar=True,
        square=True
    )
    plt.title("OT Transport Plan (log-scaled)")
    plt.xlabel("Target regions")
    plt.ylabel("Source regions")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[Saved] OT Heatmap -> {path}")





from Model.CoREPP_OT import *




import torch.nn.functional as F

def check_embedding_stats(emb, name):
    """
    """
    with torch.no_grad():
        # 1.
        norms = torch.norm(emb, dim=1)
        avg_norm = norms.mean().item()

        # 2. 
        std_val = torch.std(emb).item()

        # 3.
        sample_size = min(500, emb.size(0))
        idx = torch.randperm(emb.size(0))[:sample_size]
        sample = emb[idx]
        
        # 
        sample_norm = F.normalize(sample, dim=1)
        # 
        sim_matrix = torch.matmul(sample_norm, sample_norm.T)
        # 
        mask = ~torch.eye(sample_size, dtype=torch.bool, device=emb.device)
        avg_sim = sim_matrix[mask].mean().item()

    return avg_norm, std_val, avg_sim


if __name__ == '__main__':
    # set some parameters
    ## epochs = 600
    epochs = 300
    d_feature = 128

    exp_id = int(random.SystemRandom().random() * 1000000)
    exp_cache_file = './output/{}'.format(exp_id)
    ensure_cache_dir(exp_cache_file)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load and preprocess datas

    # s_dataset = 'bj'
    # t_dataset = 'xa'

    s_dataset = 'xa'
    t_dataset = 'bj'

    # s_dataset = 'xa'
    # t_dataset = 'cd'

    # s_dataset = 'cd'
    # t_dataset = 'xa'

    # s_dataset = 'bj'
    # t_dataset = 'cd'

    # s_dataset = 'cd'
    # t_dataset = 'bj'
    

    # s_dataset = 'bj'
    # t_dataset = 'bj'
    

    # s_dataset = 'cd'
    # t_dataset = 'cd'
    

    # s_dataset = 'xa'
    # t_dataset = 'xa'

    print('learning {} --- {}'.format(s_dataset, t_dataset))
    s_region_num, s_graph_info = load_data(s_dataset)
    t_region_num, t_graph_info = load_data(t_dataset)

  

    model = CoREPP(s_region_num, t_region_num, s_graph_info, t_graph_info, hidden_dim=128, gat_layers=2, num_heads=8,
                 lambda_align=1.0, lambda_rec=0.5, tau=0.1, beta=0.05,
                 skeleton_K=64)
       
    # model = CoREPP(s_region_num, t_region_num, s_graph_info, t_graph_info, hidden_dim=128)


    evaluator = Evaluator(exp_id=exp_id)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    results = []
    min_loss = float('inf')
    best_epoch = 0
    best_s_emb, best_t_emb = None, None
    save_path = exp_cache_file + '/train_cache'

    import os, sys

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

    log_file = os.path.join(save_path, f"train_log_exp{exp_id}.txt")
    sys.stdout = Tee(log_file)
    print(f"[Logging] stdout is tee'd to: {log_file}")





    for epoch_id in range(epochs):
        model.train()

        # s_intraCity_loss, t_intraCity_loss, crossRec_loss, crossAlign_loss = model.forward(sample_num=1000)

        # all_loss = (s_intraCity_loss + t_intraCity_loss) + crossRec_loss + crossAlign_loss

        # print('Epoch {}/{}, all_loss={:.5f}'.format(
        #         epoch_id + 1, epochs, all_loss.item(), ))

        # optim.zero_grad()
        # all_loss.backward()
        # optim.step()


        # optim.zero_grad()
        # total_loss, s_intraCity_loss, t_intraCity_loss, L_gw, L_refine = model.forward()
        # total_loss.backward()
        # optim.step()

        # # print(f"Loss: {total_loss.item():.4f}, GW: {L_gw.item():.4f}, Refine: {L_refine.item():.4f}")
        # print('Epoch {}/{} | total_loss={:.5f} | GW={:.5f} | Refine={:.5f}'.format(epoch_id + 1, epochs,
        #     total_loss.item(),
        #     L_gw.item(),
        #     L_refine.item())
        # )

        

        L_s, L_t, L_rec, L_align, total_loss = model.forward(sample_num=1000)
        optim.zero_grad()
        total_loss.backward()
        optim.step()

       
        print(f"Epoch {epoch_id+1}/{epochs} | total={total_loss.item():.4f} | Align={L_align.item():.4f} | Rec={L_rec.item():.4f}")

       # ==========================

        if (epoch_id + 1) % 5 == 0: 
            s_emb_check, t_emb_check = model.get_region_emb()
            
            s_norm, s_std, s_sim = check_embedding_stats(s_emb_check, "Source")
            t_norm, t_std, t_sim = check_embedding_stats(t_emb_check, "Target")
            
            print(f"\n[Collapse Check Epoch {epoch_id+1}]")
            print(f"  Source > Norm: {s_norm:.4f} | Std: {s_std:.4f} | AvgSim: {s_sim:.4f}")
            print(f"  Target > Norm: {t_norm:.4f} | Std: {t_std:.4f} | AvgSim: {t_sim:.4f}")
            
            

            print(f"  Raw Sample (Source[0]): {s_emb_check[0, :5].detach().cpu().numpy()}")
            print("-" * 50)

       

        VIS_EVERY = 20

        if (epoch_id + 1) % VIS_EVERY == 0 or epoch_id == 0:


            s_emb_vis, t_emb_vis = model.get_region_emb()

  
            pca_path = f"{save_path}/pca_epoch_{epoch_id+1}.png"
            plot_pca(s_emb_vis, t_emb_vis, pca_path)


            tsne_path = f"{save_path}/tsne_epoch_{epoch_id+1}.png"
            try:
                visualize_tsne(s_emb_vis, t_emb_vis, save_path=tsne_path)
            except Exception as e:
                print(f"[Warning] Cannot visualize t-SNE: {e}")
            else:
                print(f"[Saved] t-SNE -> {tsne_path}")

     
            try:
                L_align_out = model.cross_align(s_emb_vis, t_emb_vis)

                # cross_align 可能返回 (loss, pi) 或仅返回 loss
                if isinstance(L_align_out, tuple) and len(L_align_out) == 2:
                    _, pi_vis = L_align_out
                    heatmap_path = f"{save_path}/ot_heatmap_epoch_{epoch_id+1}.png"
                    plot_ot_heatmap(pi_vis, heatmap_path)
                else:
                    print("[Skip] No OT π returned, heatmap not plotted")

            except Exception as e:
                print(f"[Warning] Cannot visualize OT matrix: {e}")



        
        if (epoch_id + 1) % 5 == 0 or epoch_id == 0:
            best_epoch = epoch_id + 1

            s_emb, t_emb = model.get_region_emb()
            s_emb, t_emb = s_emb.detach().cpu().numpy(), t_emb.detach().cpu().numpy()
            print(save_path)
            np.save(save_path + '/{}_region_emb_{}.npy'.format(
                s_dataset, best_epoch), s_emb)
            np.save(save_path + '/{}_region_emb_{}.npy'.format(
                t_dataset, best_epoch), t_emb)

            evaluator.evaluate(s_dataset=s_dataset, t_dataset=t_dataset, s_region_emb=s_emb,
                               t_region_emb=t_emb)




import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import math

import torch
import torch.nn.functional as F



def sinkhorn_balanced(C, reg=0.05, a=None, b=None, max_iter=50):
    ns, nt = C.shape
    device = C.device

    if a is None:
        a = torch.full((ns,), 1.0 / ns, device=device)  
    if b is None:
        b = torch.full((nt,), 1.0 / nt, device=device)  

    K = torch.exp(-C / reg)
    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for _ in range(max_iter):
        u = a / (K @ v)
        v = b / (K.T @ u)

    P = torch.diag(u) @ K @ torch.diag(v)
    return P



def sinkhorn(C, reg=0.05, max_iter=50):
    """
    C: cost matrix (ns x nt)
    returns P: doubly stochastic soft matching
    """
    ns, nt = C.shape
    K = torch.exp(-C / reg)  # Gibbs kernel
    u = torch.ones(ns, 1, device=C.device)
    v = torch.ones(nt, 1, device=C.device)

    # Sinkhorn iterations
    for _ in range(max_iter):
        u = 1.0 / (K @ v)
        v = 1.0 / (K.T @ u)

    P = torch.diagflat(u) @ K @ torch.diagflat(v)
    return P

def partial_sinkhorn(cost, reg=0.1, frac=0.7, max_iter=50):
    ns, nt = cost.shape
    mass = frac * min(ns, nt)

    # uniform partial masses (total < 1)
    a = torch.full((ns,), mass/ns, device=cost.device)
    b = torch.full((nt,), mass/nt, device=cost.device)

    K = torch.exp(-cost / reg)

    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for _ in range(max_iter):
        u = a / (K @ v)
        v = b / (K.T @ u)

    P = torch.diag(u) @ K @ torch.diag(v)
    return P



def build_laplacian(edge_index, num_nodes, device):
    row, col = edge_index
    indices = torch.stack([row, col], dim=0)

    # 1) Construct adjacency
    A = torch.sparse_coo_tensor(
        indices, torch.ones(len(row)), 
        (num_nodes, num_nodes),
        device=device
    ).coalesce()

    # 2) Degree
    deg = torch.sparse.sum(A, dim=1).to_dense()
    deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)

    # 3) D^{-1/2} A D^{-1/2}
    #    normalized adjacency
    DAD = A.coalesce()
    r, c = DAD.indices()
    v = DAD.values() * deg_inv_sqrt[r] * deg_inv_sqrt[c]
    normA = torch.sparse_coo_tensor(
        torch.stack([r, c]), v, (num_nodes, num_nodes), device=device
    )

    # 4) Laplacian L = I - D^{-1/2} A D^{-1/2}
    I = torch.eye(num_nodes, device=device)
    L = I - normA.to_dense()   # dense is fine for <= 2000 nodes

    return L



class CoREPP(nn.Module):
    def __init__(self, s_region_num, t_region_num, s_graph_info, t_graph_info,
                 hidden_dim, gat_layers=2, num_heads=8,
                 lambda_align=1.0, lambda_rec=0.5, tau=0.1, beta=0.05,
                 skeleton_K=64):
        super().__init__()
        self.skeleton_K = skeleton_K

        self.lambda_align = lambda_align
        self.lambda_rec = lambda_rec
        self.tau = tau
        self.beta = beta
        # self.bt_w = 0.02  

        # region embeddings
        self.s_region_emb = nn.Embedding(s_region_num, hidden_dim)
        self.t_region_emb = nn.Embedding(t_region_num, hidden_dim)

        # graph info
        self.s_edge_index, self.s_edge_weight, self.s_mob = \
            s_graph_info['edge_index'], s_graph_info['edge_weight'], s_graph_info['mob']
        self.t_edge_index, self.t_edge_weight, self.t_mob = \
            t_graph_info['edge_index'], t_graph_info['edge_weight'], t_graph_info['mob']

        # GAT encoder
        self.gat_layers = gat_layers
        self.gat_s = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False, edge_dim=1)
            for _ in range(gat_layers)
        ])
        self.gat_t = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False, edge_dim=1)
            for _ in range(gat_layers)
        ])
        self.activation = nn.PReLU()
        # Build Laplacians
        self.Ls = build_laplacian(self.s_edge_index, s_region_num, device=self.s_region_emb.weight.device)
        self.Lt = build_laplacian(self.t_edge_index, t_region_num, device=self.t_region_emb.weight.device)

        # self.s_skel_idx = self.compute_skeleton_indices(self.s_mob, skeleton_K)
        # self.t_skel_idx = self.compute_skeleton_indices(self.t_mob, skeleton_K)

        # projection heads for contrastive & reconstruction
        self.proj_s = nn.Linear(hidden_dim, hidden_dim)
        self.proj_t = nn.Linear(hidden_dim, hidden_dim)
        self.wq = nn.Linear(hidden_dim, hidden_dim)
        self.wk = nn.Linear(hidden_dim, hidden_dim)

    # ---------------------------------------------------------
    #  GAT propagation
    # ---------------------------------------------------------
    def get_region_emb(self):
        s_emb, t_emb = self.s_region_emb.weight, self.t_region_emb.weight
        for i in range(self.gat_layers - 1):
            s_emb = self.activation(self.gat_s[i](s_emb, self.s_edge_index, self.s_edge_weight))
            t_emb = self.activation(self.gat_t[i](t_emb, self.t_edge_index, self.t_edge_weight))
        s_emb = self.gat_s[-1](s_emb, self.s_edge_index, self.s_edge_weight)
        t_emb = self.gat_t[-1](t_emb, self.t_edge_index, self.t_edge_weight)
        return s_emb, t_emb

    
    # ---------------------------------------------------------
    # Cross-domain Contrastive Coupling (strong gradient)
    # ---------------------------------------------------------
    # def cross_align(self, s_emb, t_emb):
    #     s_proj = F.normalize(self.proj_s(s_emb), dim=-1)
    #     t_proj = F.normalize(self.proj_t(t_emb), dim=-1)
    #     sim_matrix = torch.matmul(s_proj, t_proj.T) / self.tau
    #     logits = F.log_softmax(sim_matrix, dim=1)
    #     return -torch.mean(torch.diag(logits))  # InfoNCE loss

    # ---------------------------------------------------------
    #  Cross-domain Contrastive Coupling (strong gradient)
    # ---------------------------------------------------------

    
    # ---------------------------------------------------------
        # def variance_regularization_loss(self, s_emb, t_emb, gamma=5e-4):
        #     Z = torch.cat([s_emb, t_emb], dim=0)
        #     Z_centered = Z - Z.mean(dim=0)
            

        #     std_z = torch.sqrt(Z_centered.pow(2).mean(dim=0) + 1e-6)
        #     L_var = torch.mean(F.relu(1.0 - std_z))

    #     return gamma * L_var
   


    def cross_align(self, s_emb, t_emb, return_pi = False):
        # 1) Normalize embeddings
        s_norm = F.normalize(s_emb, dim=-1)
        t_norm = F.normalize(t_emb, dim=-1)

        # 2) cosine cost matrix (ns x nt)
        ######## cost = -torch.matmul(s_norm, t_norm.T)
        cost = torch.cdist(s_norm, t_norm, p=2)
       


        # 3) Soft OT matching matrix P
        # P = sinkhorn(cost, reg=0.2, max_iter=30)   # <--- powerful
        # P = sinkhorn(cost, reg=0.15, max_iter=30)
        # P = partial_sinkhorn(cost, reg=0.15, frac=0.7)
        ########## P = sinkhorn(cost, reg=0.15, max_iter=50)
        P = sinkhorn(cost, reg=0.15, max_iter=50)
        # P = sinkhorn_balanced(cost, reg=0.1, max_iter=50)


        # 4) OT-based alignment loss
        L_ot = torch.sum(P * cost)
        L_ot = L_ot / min(s_emb.size(0), t_emb.size(0))

        # 5) Contrastive term with OT weights
        sim = torch.matmul(s_norm, t_norm.T) / self.tau
        exp_sim = torch.exp(sim)
        weighted_pos = torch.sum(P * exp_sim, dim=1)
        all_pos = torch.sum(exp_sim, dim=1)

        L_contrast = -torch.mean(torch.log(weighted_pos / all_pos + 1e-8))

        # 6) Graph geometric consistency
        # (build Laplacian only once outside in __init__)
        Ls_S = self.Ls @ s_emb
        Lt_T = self.Lt @ t_emb
        L_geo = F.mse_loss(Ls_S, P @ Lt_T)
        # L_norm = (s_emb.norm(dim=1).mean() + t_emb.norm(dim=1).mean())
        # L_norm = 1e-3 * L_norm
        # L_var = self.variance_regularization_loss(s_emb, t_emb)
        if self.training:
            print(f"[Align Debug]  L_ot={L_ot.item():.6f} | "
                f"L_contrast={L_contrast.item():.6f} | "
                f"L_geo={L_geo.item():.6f}")

       

        # ---- total loss ----
        # loss = L_ot + 0.5 * L_contrast + 0.1 * L_geo 
        ## loss = L_ot + 0.5 * L_contrast + 0.5 * L_geo 
        # loss = L_ot + 0.5 * L_contrast
        # ---- return P when needed ----

        loss = L_ot + 0.5 * L_contrast

    
    
        # Final ICML loss
        return loss, P
    
   

    
    # # ---------------------------------------------------------
    # #  Entropy-Regularized Reconstruction (dimension-safe)
    # # ---------------------------------------------------------
    def cross_rec(self, s_emb, t_emb):
        s_q, s_k = self.wq(s_emb), self.wk(s_emb)
        t_q, t_k = self.wq(t_emb), self.wk(t_emb)

   
        # => [1311 x 1056] & [1056 x 1311]
        s_to_t = F.softmax(torch.matmul(s_q, t_k.T) / math.sqrt(s_k.shape[1]), dim=-1)
        t_to_s = F.softmax(torch.matmul(t_q, s_k.T) / math.sqrt(t_k.shape[1]), dim=-1)


        recon_loss = F.mse_loss(torch.matmul(s_to_t, t_to_s), torch.eye(s_to_t.shape[0], device=s_emb.device)[:s_to_t.shape[0]])


        entropy_reg = -torch.mean(torch.sum(s_to_t * torch.log(s_to_t + 1e-8), dim=-1))
        return recon_loss + self.beta * entropy_reg
        # return recon_loss + self.beta *  entropy_reg
        

    # ---------------------------------------------------------
    # Mobility prediction (intra-city)
    # ---------------------------------------------------------
    def mobility_prediction_loss(self, s_embeds, d_embeds, mob):
        mask = torch.tensor(mob != 0, dtype=torch.bool)
        mob = torch.tensor(mob)
        inner_prod = torch.matmul(s_embeds, d_embeds.transpose(-2, -1))
        ps_hat = F.log_softmax(inner_prod, dim=-1)
        ps_hat = torch.masked_select(ps_hat, mask)
        mob_s = torch.masked_select(mob, mask)
        return torch.sum(-torch.mul(mob_s, ps_hat))

    # ---------------------------------------------------------
    # Full forward
    # ---------------------------------------------------------
    def forward(self, sample_num=1000):
        s_emb, t_emb = self.get_region_emb()

        L_intra_s = self.mobility_prediction_loss(s_emb, s_emb, self.s_mob)
        L_intra_t = self.mobility_prediction_loss(t_emb, t_emb, self.t_mob)
        L_align, P = self.cross_align(s_emb, t_emb)
        # L_align, P = self.cross_align(s_emb, t_emb)
        # L_rec = self.cross_rec(s_emb, t_emb, P=P)
        L_rec = self.cross_rec(s_emb, t_emb)

        total_loss = (L_intra_s + L_intra_t) + \
                     self.lambda_align * L_align + \
                     self.lambda_rec * L_rec

        return L_intra_s, L_intra_t, L_rec, L_align, total_loss




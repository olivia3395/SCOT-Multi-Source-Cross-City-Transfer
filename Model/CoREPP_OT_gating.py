# -*- coding: utf-8 -*-
"""
CoREPP_OT_2source_gating.py

Ablation-1 baseline: Pairwise OT + scalar gating (2 sources -> 1 target).

We compute two pairwise OT couplings:
  s1 -> t,  s2 -> t
and combine their alignment/reconstruction objectives by a learnable scalar gate:
  alpha = softmax([w1, w2])

This baseline is known to collapse to one source (alpha ~ [1,0] or [0,1])
under strong heterogeneity, which is exactly what the HUB ablation should show.

forward() returns:
  (L_intra_s1, L_intra_s2, L_intra_t, L_rec, L_align, total_loss, (Pi1t, Pi2t), alpha)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


# =========================================================
# OT solver: balanced Sinkhorn for rectangular matrices
# =========================================================
def sinkhorn_balanced(C, reg=0.15, a=None, b=None, max_iter=50, eps=1e-8):
    """
    Balanced entropic OT for rectangular matrices.

    Args:
      C: (ns, nt) cost matrix
      reg: entropic reg (epsilon)
      a: (ns,) source marginal, sums to 1
      b: (nt,) target marginal, sums to 1
    Returns:
      P: (ns, nt) coupling, sum(P)=1 for balanced a,b.
    """
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
        u = a / (K @ v + eps)
        v = b / (K.t() @ u + eps)

    P = (u.unsqueeze(1) * K) * v.unsqueeze(0)
    return P


# =========================================================
# Graph Laplacian (dense; ok for <= ~2000 nodes)
# =========================================================
def build_laplacian(edge_index, num_nodes, device):
    row, col = edge_index
    indices = torch.stack([row, col], dim=0)

    A = torch.sparse_coo_tensor(
        indices,
        torch.ones(len(row), device=device),
        (num_nodes, num_nodes),
        device=device
    ).coalesce()

    deg = torch.sparse.sum(A, dim=1).to_dense()
    deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)

    r, c = A.indices()
    v = A.values() * deg_inv_sqrt[r] * deg_inv_sqrt[c]
    normA = torch.sparse_coo_tensor(
        torch.stack([r, c]), v, (num_nodes, num_nodes), device=device
    )

    I = torch.eye(num_nodes, device=device)
    L = I - normA.to_dense()
    return L


# =========================================================
# Pairwise+Gating model
# =========================================================
class CoREPP_2Source_Gating(nn.Module):
    def __init__(
        self,
        s1_region_num, s2_region_num, t_region_num,
        s1_graph_info, s2_graph_info, t_graph_info,
        hidden_dim=128, gat_layers=2, num_heads=8,
        lambda_align=1.0, lambda_rec=0.5,
        tau=0.1, beta=0.05,
        # OT settings
        ot_reg=0.15,
        ot_iter=50,
        # loss weights inside align
        contrast_w=0.5,
        # optional geometry regularizer (off by default)
        use_geo=False,
        geo_w=0.0,
        debug=True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gat_layers = gat_layers

        self.lambda_align = lambda_align
        self.lambda_rec = lambda_rec

        self.tau = tau
        self.beta = beta

        self.ot_reg = ot_reg
        self.ot_iter = ot_iter
        self.contrast_w = contrast_w

        self.use_geo = use_geo
        self.geo_w = geo_w
        self.debug = debug

        # ---------- region embeddings ----------
        self.s1_region_emb = nn.Embedding(s1_region_num, hidden_dim)
        self.s2_region_emb = nn.Embedding(s2_region_num, hidden_dim)
        self.t_region_emb  = nn.Embedding(t_region_num, hidden_dim)

        # ---------- graph info ----------
        self.s1_edge_index, self.s1_edge_weight, self.s1_mob = \
            s1_graph_info['edge_index'], s1_graph_info['edge_weight'], s1_graph_info['mob']
        self.s2_edge_index, self.s2_edge_weight, self.s2_mob = \
            s2_graph_info['edge_index'], s2_graph_info['edge_weight'], s2_graph_info['mob']
        self.t_edge_index, self.t_edge_weight, self.t_mob = \
            t_graph_info['edge_index'], t_graph_info['edge_weight'], t_graph_info['mob']

        # ---------- GAT encoders ----------
        self.activation = nn.PReLU()
        self.gat_s1 = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False, edge_dim=1)
            for _ in range(gat_layers)
        ])
        self.gat_s2 = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False, edge_dim=1)
            for _ in range(gat_layers)
        ])
        self.gat_t = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False, edge_dim=1)
            for _ in range(gat_layers)
        ])

        # ---------- Laplacians as buffers (optional geo) ----------
        cpu = torch.device("cpu")
        self.register_buffer("Ls1", build_laplacian(self.s1_edge_index, s1_region_num, device=cpu))
        self.register_buffer("Ls2", build_laplacian(self.s2_edge_index, s2_region_num, device=cpu))
        self.register_buffer("Lt",  build_laplacian(self.t_edge_index,  t_region_num,  device=cpu))

        # ---------- gating (learnable scalar) ----------
        # alpha = softmax(gate_logits), 2 scalars
        self.gate_logits = nn.Parameter(torch.zeros(2))

        # ---------- reconstruction attention heads ----------
        self.wq = nn.Linear(hidden_dim, hidden_dim)
        self.wk = nn.Linear(hidden_dim, hidden_dim)

    # ---------------------------------------------------------
    # 1) GAT propagation
    # ---------------------------------------------------------
    def get_region_emb(self):
        s1_emb = self.s1_region_emb.weight
        s2_emb = self.s2_region_emb.weight
        t_emb  = self.t_region_emb.weight

        for i in range(self.gat_layers - 1):
            s1_emb = self.activation(self.gat_s1[i](s1_emb, self.s1_edge_index, self.s1_edge_weight))
            s2_emb = self.activation(self.gat_s2[i](s2_emb, self.s2_edge_index, self.s2_edge_weight))
            t_emb  = self.activation(self.gat_t[i](t_emb,  self.t_edge_index,  self.t_edge_weight))

        s1_emb = self.gat_s1[-1](s1_emb, self.s1_edge_index, self.s1_edge_weight)
        s2_emb = self.gat_s2[-1](s2_emb, self.s2_edge_index, self.s2_edge_weight)
        t_emb  = self.gat_t[-1](t_emb,  self.t_edge_index,  self.t_edge_weight)
        return s1_emb, s2_emb, t_emb

    # ---------------------------------------------------------
    # 2) Intra-city mobility prediction loss
    # ---------------------------------------------------------
    def mobility_prediction_loss(self, s_embeds, d_embeds, mob):
        device = s_embeds.device
        mob_t = torch.tensor(mob, device=device)
        mask = (mob_t != 0)

        inner_prod = torch.matmul(s_embeds, d_embeds.transpose(-2, -1))
        ps_hat = F.log_softmax(inner_prod, dim=-1)

        ps_hat = ps_hat[mask]
        mob_s  = mob_t[mask]
        return torch.sum(-mob_s * ps_hat)

    # ---------------------------------------------------------
    # 3) Pairwise OT alignment + OT-guided InfoNCE (s -> t)
    # ---------------------------------------------------------
    def align_pair(self, z_s, z_t, return_pi=False):
        """
        z_s: (ns,d), z_t: (nt,d)
        Returns:
          loss, Pi, L_ot, L_contrast, mass
        """
        device = z_s.device

        zs = F.normalize(z_s, dim=-1)
        zt = F.normalize(z_t, dim=-1)

        cost = torch.cdist(zs, zt, p=2)  # (ns, nt)

        ns, nt = cost.shape
        a = torch.full((ns,), 1.0 / ns, device=device)
        b = torch.full((nt,), 1.0 / nt, device=device)

        Pi = sinkhorn_balanced(cost, reg=self.ot_reg, a=a, b=b, max_iter=self.ot_iter)

        L_ot = torch.sum(Pi * cost)

        # Row-stochastic Q for OT-guided soft positives over target nodes
        row_sum = Pi.sum(dim=1, keepdim=True) + 1e-8
        Q = Pi / row_sum  # (ns, nt)

        sim = (zs @ zt.t()) / self.tau
        sim = sim - sim.max(dim=1, keepdim=True)[0]
        exp_sim = torch.exp(sim)
        den = exp_sim.sum(dim=1) + 1e-8
        p_soft = exp_sim / den.unsqueeze(1)  # (ns, nt)

        ratio = torch.sum(Q * p_soft, dim=1)  # (ns,)
        ratio = torch.clamp(ratio, 1e-8, 1.0)
        L_contrast = -torch.mean(torch.log(ratio))

        mass = torch.mean(ratio)

        L_geo = torch.tensor(0.0, device=device)
        if self.use_geo and self.geo_w > 0:
            # choose Laplacian by size
            if z_s.size(0) == self.Ls1.size(0):
                Ls = self.Ls1.to(device)
            elif z_s.size(0) == self.Ls2.size(0):
                Ls = self.Ls2.to(device)
            else:
                Ls = self.Lt.to(device)
            Lt = self.Lt.to(device)  # target Laplacian
            # a simple geometry consistency (optional)
            L_geo = F.mse_loss(Ls @ z_s, Pi @ (Lt @ z_t))

        loss = L_ot + self.contrast_w * L_contrast + self.geo_w * L_geo

        if self.training and self.debug:
            print(f"[Pair Align] ns={ns},nt={nt} | L_ot={L_ot.item():.6f} | "
                  f"L_con={L_contrast.item():.6f} | mass={mass.item():.6f} | L_geo={L_geo.item():.6f}")

        if return_pi:
            return loss, Pi, L_ot, L_contrast, mass
        return loss, L_ot, L_contrast, mass

    # ---------------------------------------------------------
    # 4) Pairwise cycle reconstruction (s -> t -> s) + entropy reg
    # ---------------------------------------------------------
    def rec_pair(self, z_s, z_t):
        """
        Cross-attention cycle:
          s->t: (ns,nt), t->s: (nt,ns), cycle: (ns,ns) approx I
        """
        device = z_s.device
        zs_q, zs_k = self.wq(z_s), self.wk(z_s)
        zt_q, zt_k = self.wq(z_t), self.wk(z_t)

        # s -> t
        A_st = F.softmax(torch.matmul(zs_q, zt_k.t()) / math.sqrt(zs_k.size(1)), dim=-1)  # (ns,nt)
        # t -> s
        A_ts = F.softmax(torch.matmul(zt_q, zs_k.t()) / math.sqrt(zt_k.size(1)), dim=-1)  # (nt,ns)

        ns = z_s.size(0)
        I = torch.eye(ns, device=device)

        recon = A_st @ A_ts  # (ns,ns)
        recon_loss = F.mse_loss(recon, I)

        # entropy reg on A_st (avoid too peaky collapse)
        ent = -torch.mean(torch.sum(A_st * torch.log(A_st + 1e-8), dim=-1))
        return recon_loss + self.beta * ent

    # ---------------------------------------------------------
    # 5) forward
    # ---------------------------------------------------------
    def forward(self):
        s1_emb, s2_emb, t_emb = self.get_region_emb()

        L_intra_s1 = self.mobility_prediction_loss(s1_emb, s1_emb, self.s1_mob)
        L_intra_s2 = self.mobility_prediction_loss(s2_emb, s2_emb, self.s2_mob)
        L_intra_t  = self.mobility_prediction_loss(t_emb,  t_emb,  self.t_mob)

        # gating weights
        alpha = torch.softmax(self.gate_logits, dim=0)  # (2,)
        a1, a2 = alpha[0], alpha[1]

        # pairwise alignments
        L_a1, Pi1t, L_ot1, L_c1, m1 = self.align_pair(s1_emb, t_emb, return_pi=True)
        L_a2, Pi2t, L_ot2, L_c2, m2 = self.align_pair(s2_emb, t_emb, return_pi=True)

        L_align = a1 * L_a1 + a2 * L_a2

        # pairwise recon
        L_r1 = self.rec_pair(s1_emb, t_emb)
        L_r2 = self.rec_pair(s2_emb, t_emb)
        L_rec = a1 * L_r1 + a2 * L_r2

        total_loss = (L_intra_s1 + L_intra_s2 + L_intra_t) + \
                     self.lambda_align * L_align + \
                     self.lambda_rec * L_rec

        if self.training and self.debug:
            print(f"[Gating] alpha=({a1.item():.3f},{a2.item():.3f}) | "
                  f"OT=({L_ot1.item():.4f},{L_ot2.item():.4f}) | mass=({m1.item():.4f},{m2.item():.4f})")

        return (L_intra_s1, L_intra_s2, L_intra_t,
                L_rec, L_align, total_loss, (Pi1t, Pi2t), alpha.detach())

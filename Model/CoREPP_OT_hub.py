# -*- coding: utf-8 -*-
"""
CoREPP_OT_2source_hub.py

Two-source -> one-target with a SHARED semantic HUB (prototypes / anchors).

Key idea:
  Instead of pairwise OT + scalar gating (which collapses to one source),
  we align EACH city to a shared set of K prototypes (hub):
      Z^{s1} -> A,  Z^{s2} -> A,  Z^{t} -> A
  using (i) balanced entropic OT (rectangular Sinkhorn) and
      (ii) OT-guided InfoNCE on (region, prototype) pairs,
  plus (iii) cycle-style reconstruction via hub-attention for stability.

This is the "B1 Prototype anchors" implementation, and is also a practical
approximation to entropic Wasserstein barycenter (B2).

Outputs:
  forward() returns:
    (L_intra_s1, L_intra_s2, L_intra_t, L_rec, L_align, total_loss, (Pi1, Pi2, Pit))

Author: adapted for your SCOT / CoREPP experiments
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
    Balanced entropic OT (Sinkhorn) for rectangular C (ns x K).

    Solves:
        min_P <P, C> + reg * KL(P || a b^T)
        s.t. P 1 = a,  P^T 1 = b

    Returns:
        P: coupling (ns, K). For balanced case, sum(P)=sum(a)=sum(b).
    """
    ns, nk = C.shape
    device = C.device

    if a is None:
        a = torch.full((ns,), 1.0 / ns, device=device)
    if b is None:
        b = torch.full((nk,), 1.0 / nk, device=device)

    K = torch.exp(-C / reg)  # Gibbs kernel

    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for _ in range(max_iter):
        u = a / (K @ v + eps)
        v = b / (K.t() @ u + eps)

    P = (u.unsqueeze(1) * K) * v.unsqueeze(0)
    return P


# =========================================================
# Graph Laplacian (dense, ok for <= ~2000 nodes)
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
# Two-source HUB model
# =========================================================
class CoREPP_2Source_Hub(nn.Module):
    def __init__(
        self,
        s1_region_num, s2_region_num, t_region_num,
        s1_graph_info, s2_graph_info, t_graph_info,
        hidden_dim=128, gat_layers=2, num_heads=8,
        lambda_align=1.0, lambda_rec=0.5,
        tau=0.1, beta=0.05,
        # hub / prototypes
        hub_K=64,
        hub_init_scale=0.02,
        # OT settings
        ot_reg=0.15,
        ot_iter=50,
        # loss weights inside align
        contrast_w=0.5,
        # hub balancing regularizer
        hub_bal_w=0.1,      # encourage all prototypes being used (avoid dead anchors)
        # optional geometry regularizer (often not needed for hub version)
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

        self.hub_K = hub_K
        self.ot_reg = ot_reg
        self.ot_iter = ot_iter
        self.contrast_w = contrast_w
        self.hub_bal_w = hub_bal_w

        self.use_geo = use_geo
        self.geo_w = geo_w
        self.debug = debug

        # ---------- region embeddings ----------
        self.s1_region_emb = nn.Embedding(s1_region_num, hidden_dim)
        self.s2_region_emb = nn.Embedding(s2_region_num, hidden_dim)
        self.t_region_emb  = nn.Embedding(t_region_num, hidden_dim)

        # ---------- shared HUB prototypes (anchors) ----------
        # Learnable prototypes: (K, d)
        A = torch.randn(hub_K, hidden_dim) * hub_init_scale
        self.anchors = nn.Parameter(A)

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
        # for anchors (hub) we can use identity Laplacian if needed
        self.register_buffer("Lh",  torch.eye(hub_K, device=cpu))

        # ---------- reconstruction heads (hub attention) ----------
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
    # 3) Align a city to HUB using balanced OT + OT-guided InfoNCE
    # ---------------------------------------------------------
    def align_to_hub(self, z_city, return_pi=False,  b_override=None):
        """
        z_city: (n, d)
        anchors: (K, d)

        Returns:
          loss_city, Pi, L_ot, L_contrast, mass
        """
        device = z_city.device
        A = self.anchors

        # normalize
        z = F.normalize(z_city, dim=-1)
        a = F.normalize(A, dim=-1)

        # OT cost: (n, K)
        cost = torch.cdist(z, a, p=2)

        # balanced marginals
        n, K = cost.shape
        marg_z = torch.full((n,), 1.0 / n, device=device)
        marg_a = torch.full((K,), 1.0 / K, device=device) if b_override is None else b_override
        marg_a = marg_a / (marg_a.sum() + 1e-8)  # ensure sums to 1
        Pi = sinkhorn_balanced(cost, reg=self.ot_reg, a=marg_z, b=marg_a, max_iter=self.ot_iter)

        # marg_z = torch.full((n,), 1.0 / n, device=device)
        # marg_a = torch.full((K,), 1.0 / K, device=device)

        # Pi = sinkhorn_balanced(cost, reg=self.ot_reg, a=marg_z, b=marg_a, max_iter=self.ot_iter)

        # OT loss (balanced => sum(Pi)=1)
        L_ot = torch.sum(Pi * cost)

        # OT-guided InfoNCE over prototypes
        # S_ik = <z_i, a_k> / tau
        # sim = torch.matmul(z, a.t()) / self.tau
        # sim = sim - sim.max(dim=1, keepdim=True)[0]  # stabilize exp
        # exp_sim = torch.exp(sim)

        # weighted_pos = torch.sum(Pi * exp_sim, dim=1)  # (n,)
        # all_pos = torch.sum(exp_sim, dim=1)            # (n,)

        # ratio = weighted_pos / (all_pos + 1e-8)
        # ratio = torch.clamp(ratio, min=1e-8, max=1.0)
        # L_contrast = -torch.mean(torch.log(ratio))

        # # "mass" proxy: average ratio
        # mass = torch.mean(ratio)
        # Pi: (n,K), balanced => row_sum = 1/n
        row_sum = Pi.sum(dim=1, keepdim=True) + 1e-8
        Q = Pi / row_sum              # (n,K) row-stochastic, each row sums to 1

        sim = torch.matmul(z, a.t()) / self.tau
        sim = sim - sim.max(dim=1, keepdim=True)[0]
        exp_sim = torch.exp(sim)

        den = exp_sim.sum(dim=1) + 1e-8                         # (n,)
        num = (Q * exp_sim).sum(dim=1)                          # (n,)

        ratio = num / den                                       # in (0,1)
        ratio = torch.clamp(ratio, 1e-8, 1.0)
        L_contrast = -torch.mean(torch.log(ratio))

        # a meaningful "mass": expected probability under Q of the softmax distribution
        p_soft = exp_sim / den.unsqueeze(1)                      # softmax over K
        mass = torch.mean(torch.sum(Q * p_soft, dim=1))          # in (0,1), interpretable


        # optional geometry (usually not needed for hub)
        L_geo = torch.tensor(0.0, device=device)
        if self.use_geo and self.geo_w > 0:
            # city Laplacian depends on n
            if z_city.size(0) == self.Ls1.size(0):
                Lc = self.Ls1.to(device)
            elif z_city.size(0) == self.Ls2.size(0):
                Lc = self.Ls2.to(device)
            else:
                Lc = self.Lt.to(device)

            Lh = self.Lh.to(device)
            L_geo = F.mse_loss(Lc @ z_city, Pi @ (Lh @ A))

        loss = L_ot + self.contrast_w * L_contrast + self.geo_w * L_geo
        # ---- DIAG: assignment sharpness + anchor usage ----
        with torch.no_grad():
            # how sharp each node's hub assignment is
            q_max = Q.max(dim=1).values.mean()           # higher => sharper
            q_ent = -(Q * torch.log(Q + 1e-8)).sum(dim=1).mean()  # higher => more diffuse

            # hub usage distribution (column sums)
            col = Pi.sum(dim=0)                          # sums to 1 (balanced)
            col_ent = -(col * torch.log(col + 1e-8)).sum()         # max = log(K)
            col_max = col.max()

            # sinkhorn marginal check
            rs = Pi.sum(dim=1).mean()
            cs = Pi.sum(dim=0).mean()

            print(f"[DIAG Hub] q_max={q_max.item():.3f} q_ent={q_ent.item():.3f} | "
                f"col_max={col_max.item():.3f} col_ent={col_ent.item():.3f} | "
                f"row_mean={rs.item():.6f} col_mean={cs.item():.6f}")
        
        if self.training and self.debug:
            print(f"[Hub Align] n={n},K={K} | L_ot={L_ot.item():.6f} | "
                  f"L_con={L_contrast.item():.6f} | mass={mass.item():.6f} | L_geo={L_geo.item():.6f}")

        if return_pi:
            return loss, Pi, L_ot, L_contrast, mass
        return loss, L_ot, L_contrast, mass

    # ---------------------------------------------------------
    # 4) Hub-cycle reconstruction (city -> hub -> city) + entropy reg
    # ---------------------------------------------------------
    def rec_via_hub(self, z_city):
        """
        Cross-attention cycle between city nodes and hub prototypes:
          city->hub: (n,K), hub->city: (K,n), cycle: (n,n) approx I

        This is analogous to your cross_rec, but with hub as the bridge.
        """
        device = z_city.device
        A = self.anchors

        z_q, z_k = self.wq(z_city), self.wk(z_city)
        a_q, a_k = self.wq(A),      self.wk(A)

        # city -> hub (n,K)
        city_to_hub = F.softmax(torch.matmul(z_q, a_k.t()) / math.sqrt(z_k.shape[1]), dim=-1)
        # hub -> city (K,n)
        hub_to_city = F.softmax(torch.matmul(a_q, z_k.t()) / math.sqrt(a_k.shape[1]), dim=-1)

        n = city_to_hub.shape[0]
        I = torch.eye(n, device=device)

        recon = city_to_hub @ hub_to_city  # (n,n)
        recon_loss = F.mse_loss(recon, I)

        # entropy regularization on city->hub assignment (avoid collapse but keep smoothness)
        entropy_reg = -torch.mean(torch.sum(city_to_hub * torch.log(city_to_hub + 1e-8), dim=-1))
        return recon_loss + self.beta * entropy_reg

    # ---------------------------------------------------------
    # 5) Hub balancing regularizer (avoid dead anchors)
    # ---------------------------------------------------------
    def hub_balance_reg(self, Pi_list):
        """
        Encourage the column-marginal over hub prototypes to be close to uniform,
        across all cities (s1,s2,t). This helps prevent "dead anchors".

        For each Pi (n,K), column marginal p = Pi^T 1  (sum to 1 for balanced OT),
        penalize KL(p || uniform).
        """
        device = Pi_list[0].device
        K = Pi_list[0].size(1)
        u = torch.full((K,), 1.0 / K, device=device)

        reg = torch.tensor(0.0, device=device)
        for Pi in Pi_list:
            p = Pi.sum(dim=0)  # (K,) sums to 1
            p = torch.clamp(p, min=1e-8)
            reg = reg + torch.sum(p * (torch.log(p) - torch.log(u)))
        return reg / len(Pi_list)

    # ---------------------------------------------------------
    # 6) Full forward
    # ---------------------------------------------------------
    def forward(self):
        s1_emb, s2_emb, t_emb = self.get_region_emb()

        # intra-city
        L_intra_s1 = self.mobility_prediction_loss(s1_emb, s1_emb, self.s1_mob)
        L_intra_s2 = self.mobility_prediction_loss(s2_emb, s2_emb, self.s2_mob)
        L_intra_t  = self.mobility_prediction_loss(t_emb,  t_emb,  self.t_mob)
        # (A) get target Pit first with uniform b
        
        # --- build non-uniform b_t from target->anchor similarities ---
        with torch.no_grad():
            zt = F.normalize(t_emb, dim=-1)
            a  = F.normalize(self.anchors, dim=-1)
            sim_ta = (zt @ a.t())  # (nt, K)
            # b_t = torch.softmax(sim_ta.mean(dim=0) / 0.2, dim=0)  # temp_b=0.2 可调
            # b_t = b_t.detach()
            # alpha = 0.2  # 0.1~0.3 之间试
            # b_t = (1 - alpha) * b_t + alpha * (torch.ones_like(b_t) / b_t.numel())
            # b_t = (b_t / b_t.sum()).detach()
            temp_b = 0.5
            b_t = torch.softmax(sim_ta.mean(dim=0) / temp_b, dim=0)

            eps = 1e-3  # floor，防止某些 anchor 完全死掉
            b_t = torch.clamp(b_t, min=eps)
            b_t = (b_t / b_t.sum()).detach()
            print("[DIAG b_t] max", b_t.max().item(), "ent", (-(b_t*torch.log(b_t+1e-8)).sum()).item())


        # (B) use target-induced b to align sources
        
        L_a1, Pi1, L_ot1, L_c1, m1 = self.align_to_hub(s1_emb, return_pi=True, b_override=b_t)
        L_a2, Pi2, L_ot2, L_c2, m2  = self.align_to_hub(s2_emb, return_pi=True, b_override=b_t)

        # (optional) recompute target with same b_t for consistency
        L_at, Pit, L_ott, L_ct, mt = self.align_to_hub(t_emb, return_pi=True, b_override=b_t)
        # hub align for each city
        # L_a1, Pi1, L_ot1, L_c1, m1 = self.align_to_hub(s1_emb, return_pi=True)
        # L_a2, Pi2, L_ot2, L_c2, m2 = self.align_to_hub(s2_emb, return_pi=True)
        # L_at, Pit, L_ott, L_ct, mt = self.align_to_hub(t_emb,  return_pi=True)

        # total align (no scalar gating, all cities contribute)
        L_align = (L_a1 + L_a2 + L_at) / 3.0

        # hub balancing (optional, but recommended)
        L_bal = self.hub_balance_reg([Pi1, Pi2, Pit])

        # reconstruction via hub (stability)
        L_rec1 = self.rec_via_hub(s1_emb)
        L_rec2 = self.rec_via_hub(s2_emb)
        L_rect = self.rec_via_hub(t_emb)
        L_rec  = (L_rec1 + L_rec2 + L_rect) / 3.0

        if self.training and self.debug:
            print(f"[Hub Summary] "
                  f"OT(s1,s2,t)=({L_ot1.item():.4f},{L_ot2.item():.4f},{L_ott.item():.4f}) | "
                  f"mass=({m1.item():.4f},{m2.item():.4f},{mt.item():.4f}) | "
                  f"L_bal={L_bal.item():.6f}")

        total_loss = (L_intra_s1 + L_intra_s2 + L_intra_t) + \
                     self.lambda_align * (L_align + self.hub_bal_w * L_bal) + \
                     self.lambda_rec * L_rec

        return (L_intra_s1, L_intra_s2, L_intra_t,
                L_rec, L_align, total_loss, (Pi1, Pi2, Pit))

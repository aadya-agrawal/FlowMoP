import math
import os
import time
import json
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import get_dataloaders as build_dataloaders
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════════════════

INPUT_NOISE_STD = 0.003
CFG_DROP_PROB   = 0.0
CFG_SCALE       = 1.0

# ── Agent-neighbor conditioning ────────────────────────────────────────────
USE_AGENT_NEIGHBORS = True
K_NEIGHBORS         = 10           # more neighbors so grouping sees both teams

# ── QK normalization ──────────────────────────────────────────────────────────
USE_QK_NORM         = False         # disable QKV RMSNorm so attention can be non-uniform

# ── Unsupervised neighbor grouping ────────────────────────────────────────────
USE_LEARNED_GROUPING = False         # learn to assign neighbors to groups via attention
N_GROUPS             = 2            # number of discoverable groups (e.g., teammate vs opponent)

# ── Dataset ────────────────────────────────────────────────────────────────────
COORD_DIM    = 3
NUM_JOINTS   = 24           # SMPL skeleton
POSE_DIM     = NUM_JOINTS * COORD_DIM  # 72
INPUT_TIME   = 25
OUTPUT_TIME  = 75
BATCH_SIZE   = 16
MAX_PEOPLE   = 22
NORMALIZE    = False
DATA_PATH    = None

EVAL_HORIZONS = {"1s": 24, "2s": 49, "3s": 74}

# SMPL left↔right mirror map for 24 joints
# L_Hip↔R_Hip, L_Knee↔R_Knee, L_Ankle↔R_Ankle, L_Foot↔R_Foot,
# L_Collar↔R_Collar, L_Shoulder↔R_Shoulder, L_Elbow↔R_Elbow, L_Wrist↔R_Wrist, L_Hand↔R_Hand
MIRROR_MAP   = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 13, 14, 16, 15, 18, 17, 20, 19, 22, 21, 23]
MIRROR_PROB  = 0.5
MIRROR_AXIS  = 0               # negate x-axis when mirroring

# ── Model ──────────────────────────────────────────────────────────────────────
DEPTH         = 8
EMBED_DIM     = 256
NUM_HEADS     = 4
MLP_RATIO     = 2
DROPOUT       = 0.15
K_PREDICTIONS = 10

# ── Training budget ───────────────────────────────────────────────────────────
TRAIN_STEPS    = 2000
EVAL_EVERY     = 500
LOG_EVERY      = 100
EVAL_K         = 10
EARLY_STOP_PATIENCE = 5           # stop after N evals without improvement
LIMIT_SAMPLES = None

# ── Optimizer ─────────────────────────────────────────────────────────────────
LR           = 2e-4
WEIGHT_DECAY = 1e-2
BETAS        = (0.9, 0.999)

# ── Flow matching ─────────────────────────────────────────────────────────────
TIMESTEP_MODE   = "logit_normal"
NOISE_SCALE     = 1.0
NOISE_SCALE_POSE = 1.0
INFERENCE_STEPS = 10

# ── Loss weighting ────────────────────────────────────────────────────────────
TRAJ_LOSS_WEIGHT = 5.0
POSE_LOSS_WEIGHT = 3.0
POSE_FDE_EXP     = 2.0
FRAME_DROP_PROB  = 0.15
VEL_REG_WEIGHT   = 0.1
TRAJ_FDE_WEIGHT  = 3.0
JPE_PROXY_WEIGHT = 5.0
USE_OT_COUPLING  = False
TRAJ_MULTI_HORIZON_FDE = True     # FDE at 25%, 50%, 75%, 100% of output

# ── Full-horizon JPE proxy ────────────────────────────────────────────────────
JPE_PROXY_FULL_HORIZON = True      # compute JPE proxy over ALL output frames (not just last)
JPE_PROXY_T_WEIGHTING  = "clamp"   # "t_sq" = t*t, "clamp" = t.clamp(0.2), "none" = 1.0

# ── Bypass kmeans in evaluation ───────────────────────────────────────────────
USE_KMEANS_SELECT = False          # if False, raw best-of-K without kmeans reduction

# ── Joint-space output head ───────────────────────────────────────────────────
USE_JOINT_HEAD     = False
JOINT_LOSS_WEIGHT  = 0.0
N_CTRL_JOINT       = 35

# SMPL kinematic tree (24 joints, indices 0–23)
# 0=Pelvis 1=L_Hip 2=R_Hip 3=Spine1 4=L_Knee 5=R_Knee 6=Spine2
# 7=L_Ankle 8=R_Ankle 9=Spine3 10=L_Foot 11=R_Foot 12=Neck 13=Thorax
# 14=Head 15=L_Collar 16=R_Collar 17=L_Shoulder 18=R_Shoulder
# 19=L_Elbow 20=R_Elbow 21=L_Wrist 22=R_Wrist 23=L_Hand
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 12, 13, 13, 15, 16, 17, 18, 19, 20, 21]

# ── B-spline control-point reparameterization ─────────────────────────────────
USE_SPLINE      = True
USE_SPLINE_TRAJ = False
TRAJ_DISPLACEMENT = True
N_CTRL_TRAJ     = 40             # ignored when USE_SPLINE_TRAJ=False
N_CTRL_POSE     = 50             # spline ctrl pts for pose (T_out=75)
SPLINE_TYPE     = "cubic"

RESULTS_DIR = "outputs/worldpose"


# ══════════════════════════════════════════════════════════════════════════════
#  B-spline / piecewise-linear basis
# ══════════════════════════════════════════════════════════════════════════════

def _make_clamped_knots(n_ctrl, degree=3):
    """Clamped knot vector for B-spline of given degree."""
    n_interior = n_ctrl - degree - 1
    if n_interior < 0:
        raise ValueError(f"n_ctrl={n_ctrl} too small for degree={degree}")
    interior = [j / (n_interior + 1) for j in range(1, n_interior + 1)]
    knots = [0.0] * (degree + 1) + interior + [1.0] * (degree + 1)
    return torch.tensor(knots, dtype=torch.float32)


def _eval_bspline_basis(n_ctrl, degree, t_values, knots):
    """Cox-de Boor recursion.  t_values: [M], knots: [n_ctrl+degree+1].  Returns [M, n_ctrl]."""
    M = t_values.shape[0]
    basis_prev = torch.zeros(M, n_ctrl, device=t_values.device)
    for i in range(n_ctrl):
        if i == n_ctrl - 1:
            basis_prev[:, i] = ((knots[i] <= t_values) & (t_values <= knots[i + 1])).float()
        else:
            basis_prev[:, i] = ((knots[i] <= t_values) & (t_values < knots[i + 1])).float()
    for k in range(1, degree + 1):
        basis_curr = torch.zeros(M, n_ctrl, device=t_values.device)
        for i in range(n_ctrl):
            denom1 = knots[i + k] - knots[i]
            denom2 = knots[i + k + 1] - knots[i + 1] if (i + 1) < len(knots) - k else 0.0
            if denom1 > 0:
                basis_curr[:, i] += ((t_values - knots[i]) / denom1) * basis_prev[:, i]
            if denom2 > 0 and (i + 1) < n_ctrl:
                basis_curr[:, i] += ((knots[i + k + 1] - t_values) / denom2) * basis_prev[:, i + 1]
        basis_prev = basis_curr
    return basis_prev


def make_spline_basis(n_ctrl, n_eval, spline_type="cubic"):
    """
    Build spline basis matrix and its pseudoinverse.

    Returns (B, B_pinv, anchor_idx):
        B          : [n_eval, n_ctrl]  — maps ctrl pts → full trajectory
        B_pinv     : [n_ctrl, n_eval]  — maps full trajectory → ctrl pts
        anchor_idx : [n_ctrl] LongTensor — frame indices for PE assignment
    """
    anchor_idx = torch.linspace(0, n_eval - 1, n_ctrl).round().long()
    if spline_type == "cubic":
        knots = _make_clamped_knots(n_ctrl, degree=3)
        t_values = torch.linspace(0, 1, n_eval)
        B = _eval_bspline_basis(n_ctrl, 3, t_values, knots)
        B_pinv = torch.linalg.pinv(B)
    else:
        # piecewise-linear
        B = torch.zeros(n_eval, n_ctrl)
        for t in range(n_eval):
            k = int((t >= anchor_idx).sum().item()) - 1
            k = max(0, min(k, n_ctrl - 2))
            t0, t1 = int(anchor_idx[k].item()), int(anchor_idx[k + 1].item())
            alpha = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            B[t, k] = 1.0 - alpha
            B[t, k + 1] = alpha
        B_pinv = torch.linalg.pinv(B)
    return B, B_pinv, anchor_idx


# ══════════════════════════════════════════════════════════════════════════════
#  Data helpers — WorldPose batch → traj/pose decomposition
# ══════════════════════════════════════════════════════════════════════════════

def get_dataloaders():
    train_loader, val_loader = build_dataloaders(
        DATA_PATH, batch_size=BATCH_SIZE, normalize=NORMALIZE,
        limit_samples=LIMIT_SAMPLES,
    )
    return train_loader, val_loader


def prepare_batch(batch, device):
    """
    Convert WorldPose batch into traj/pose decomposition with neighbor info.

    Input batch:
      past_joints_3d   (B, N, T_in, J, 3)
      future_joints_3d (B, N, T_out, J, 3)
      valid_mask        (B, N)

    Folds agents into batch → effective batch BN = B*N.
    Extracts root trajectory (joint 0) and root-relative pose.
    Also extracts K nearest neighbor past trajectories per agent (focal-centric).
    """
    past   = batch["past_joints_3d"].to(device)      # (B, N, T_in, J, 3)
    future = batch["future_joints_3d"].to(device)     # (B, N, T_out, J, 3)
    mask   = batch["valid_mask"].to(device)            # (B, N)

    B, N, T_p, J, C = past.shape
    T_f = future.shape[2]

    # root trajectory = joint 0 position (before folding, for neighbor extraction)
    past_root_all = past[:, :, :, 0, :]                # (B, N, T_p, 3)

    # agent-centric offset: last observed root per agent
    offset_all = past_root_all[:, :, -1:, :]           # (B, N, 1, 3)
    past_traj_all = past_root_all - offset_all         # (B, N, T_p, 3)  agent-centric

    # ── neighbor extraction (before folding) ─────────────────────────────
    # Find K nearest neighbors per focal agent within each scene
    K = K_NEIGHBORS
    # last observed absolute root position for distance computation
    last_pos = past_root_all[:, :, -1, :]              # (B, N, 3)
    # pairwise distances within each scene
    dists = torch.cdist(last_pos, last_pos)            # (B, N, N)
    # mask out invalid agents (set their distance to inf)
    valid_f = mask.float()                             # (B, N)
    invalid_mask = (1.0 - valid_f).bool()
    dists.masked_fill_(invalid_mask.unsqueeze(1), float('inf'))  # columns: neighbors
    # mask out self (diagonal)
    dists.masked_fill_(torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0), float('inf'))

    # find K nearest neighbors
    K_actual = min(K, N - 1)
    _, nn_idx = dists.topk(K_actual, dim=-1, largest=False)    # (B, N, K)

    # gather neighbor past trajectories (agent-centric coords of FOCAL agent)
    # neighbor absolute past root = past_root_all gathered by nn_idx
    nn_idx_exp = nn_idx.unsqueeze(-1).unsqueeze(-1).expand(B, N, K_actual, T_p, C)
    neighbor_past_abs = torch.gather(
        past_root_all.unsqueeze(2).expand(-1, -1, K_actual, -1, -1),
        dim=1,
        index=nn_idx_exp.permute(0, 2, 1, 3, 4).reshape(B, K_actual, N, T_p, C).permute(0, 2, 1, 3, 4),
    )
    # simpler gather: for each focal agent (b,i), get neighbor j's past root
    past_root_exp = past_root_all.unsqueeze(2).expand(B, N, N, T_p, C)  # (B, N, N, T_p, C)
    nn_idx_g = nn_idx.unsqueeze(-1).unsqueeze(-1).expand(B, N, K_actual, T_p, C)
    neighbor_past_abs = torch.gather(past_root_exp, 2, nn_idx_g)  # (B, N, K, T_p, C)

    # convert to focal-agent-centric coords
    neighbor_past_traj = neighbor_past_abs - offset_all.unsqueeze(2)  # (B, N, K, T_p, C)

    # neighbor validity mask
    nn_valid = torch.gather(valid_f, 1, nn_idx.view(B, N * K_actual).long()).view(B, N, K_actual)  # (B, N, K)
    # also mark as invalid if focal agent itself is invalid
    nn_valid = nn_valid * valid_f.unsqueeze(-1)                       # (B, N, K)

    # fold agents into batch dimension
    past_flat   = past.view(B * N, T_p, J, C)
    future_flat = future.view(B * N, T_f, J, C)
    mask_flat   = mask.view(B * N).float()

    # root trajectory (already computed, just fold)
    past_traj   = past_traj_all.view(B * N, T_p, C)
    offset      = offset_all.view(B * N, 1, C)
    future_root = future_flat[:, :, 0, :]              # (BN, T_f, 3)
    future_traj = future_root - offset                 # (BN, T_f, 3)

    # root-relative pose
    past_pose_rel   = (past_flat - past_flat[:, :, 0:1, :]).reshape(B * N, T_p, POSE_DIM)
    future_pose_rel = (future_flat - future_flat[:, :, 0:1, :]).reshape(B * N, T_f, POSE_DIM)

    # fold neighbor data
    neighbor_traj = neighbor_past_traj.view(B * N, K_actual, T_p, C)  # (BN, K, T_p, 3)
    neighbor_mask = nn_valid.view(B * N, K_actual)                    # (BN, K)

    return {
        "input_traj":      past_traj,         # (BN, T_p, 3)
        "output_traj":     future_traj,       # (BN, T_f, 3)
        "input_pose_rel":  past_pose_rel,     # (BN, T_p, J*3)
        "output_pose_rel": future_pose_rel,   # (BN, T_f, J*3)
        "offset_traj":     offset,            # (BN, 1, 3)
        "valid_mask":      mask_flat,          # (BN,)
        "neighbor_traj":   neighbor_traj,     # (BN, K, T_p, 3)
        "neighbor_mask":   neighbor_mask,     # (BN, K)
        "neighbor_idx":    nn_idx.view(B * N, K_actual),  # (BN, K) scene-local agent indices
    }


def denormalize_joints_for_jpe(x, mean, std):
    """Invert dataset z-score on trailing (J, 3). mean/std (J*3,) on device; None = no-op."""
    if mean is None:
        return x
    x = x.float()
    J, C = x.shape[-2], x.shape[-1]
    m = mean.view(*((1,) * (x.ndim - 2)), J, C)
    s = std.view(*((1,) * (x.ndim - 2)), J, C)
    return x * s + m


# ══════════════════════════════════════════════════════════════════════════════
#  Model components
# ══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim, heads=1):
        super().__init__()
        self.scale = dim ** -0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.gamma * self.scale


def timestep_embedding(t, dim, max_period=10000):
    half  = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb  = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class PoseRefineBlock(nn.Module):
    """Self-attention block for pose-only future refinement."""
    def __init__(self, dim, num_heads=4, mlp_ratio=2, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.qkv   = nn.Linear(dim, dim * 3)
        self.out   = nn.Linear(dim, dim)
        hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        N, T, _ = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(N, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn_out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(N, T, -1)
        x = x + self.drop(self.out(attn_out))
        x = x + self.mlp(self.norm2(x))
        return x


class Block(nn.Module):
    """
    Dual-stream transformer block (past / future).
    Future tokens attend to full past+future context;
    past tokens only attend to past.
    adaLN conditioning on time+context vector.
    """
    def __init__(self, dim, num_heads=4, mlp_ratio=2, dropout=0.1, causal_future=False, use_qk_norm=True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads     = num_heads
        self.head_dim      = dim // num_heads
        self.causal_future = causal_future
        self.has_dropout   = dropout > 0
        self.use_qk_norm   = use_qk_norm

        self.norm_p1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm_p2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm_f1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm_f2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        self.qkv_p = nn.Linear(dim, dim * 3)
        self.qkv_f = nn.Linear(dim, dim * 3)
        # QK normalization — can be disabled to allow non-uniform attention
        if use_qk_norm:
            self.norm_q_p = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.norm_k_p = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.norm_q_f = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.norm_k_f = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.out_p = nn.Linear(dim, dim)
        self.out_f = nn.Linear(dim, dim)

        hidden = dim * mlp_ratio
        self.mlp_p = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.mlp_f = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.drop  = nn.Dropout(dropout)

    def forward(self, past, future, cond, return_attn=False):
        sp_a, sh_a, sf_a, shf_a, sp_m, sh_m, sf_m, shf_m = cond.chunk(8, dim=-1)

        N, T_p, _ = past.shape
        T_f       = future.shape[1]

        # ── past stream ──
        p = self.norm_p1(past) * (1 + sp_a) + sh_a
        qkv = self.qkv_p(p).reshape(N, T_p, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        qp, kp, vp = qkv.unbind(0)
        if self.use_qk_norm:
            qp, kp = self.norm_q_p(qp), self.norm_k_p(kp)

        # ── future stream ──
        f = self.norm_f1(future) * (1 + sf_a) + shf_a
        qkv = self.qkv_f(f).reshape(N, T_f, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        qf, kf, vf = qkv.unbind(0)
        if self.use_qk_norm:
            qf, kf = self.norm_q_f(qf), self.norm_k_f(kf)

        # future attends to all past + future tokens
        k_all = torch.cat([kp, kf], dim=-2)
        v_all = torch.cat([vp, vf], dim=-2)

        attn_dict = None
        if return_attn:
            # manual attention to extract weights (probing only, not used in training)
            scale = self.head_dim ** -0.5
            # past self-attention
            past_scores = torch.matmul(qp, kp.transpose(-2, -1)) * scale
            past_attn_w = F.softmax(past_scores, dim=-1)           # (N, H, T_p, T_p)
            x_p = torch.matmul(past_attn_w, vp)
            # future cross-attention to past+future
            fut_scores = torch.matmul(qf, k_all.transpose(-2, -1)) * scale
            if self.causal_future:
                mask_pa = torch.ones(T_f, T_p, device=past.device, dtype=torch.bool)
                mask_ff = torch.tril(torch.ones(T_f, T_f, device=past.device, dtype=torch.bool))
                causal = torch.cat([mask_pa, mask_ff], dim=-1)
                fut_scores = fut_scores.masked_fill(~causal, float('-inf'))
            fut_attn_w = F.softmax(fut_scores, dim=-1)             # (N, H, T_f, T_p+T_f)
            x_f = torch.matmul(fut_attn_w, v_all)
            attn_dict = {"past": past_attn_w, "future": fut_attn_w}
        else:
            # fast SDPA path (training)
            if self.causal_future:
                mask_pa = torch.ones(T_f, T_p, device=past.device, dtype=torch.bool)
                mask_ff = torch.tril(torch.ones(T_f, T_f, device=past.device, dtype=torch.bool))
                attn_mask = torch.cat([mask_pa, mask_ff], dim=-1)
                x_f = F.scaled_dot_product_attention(qf, k_all, v_all, attn_mask=attn_mask)
            else:
                x_f = F.scaled_dot_product_attention(qf, k_all, v_all)
            x_p = F.scaled_dot_product_attention(qp, kp, vp)

        x_f = x_f.transpose(1, 2).reshape(N, T_f, -1)
        if self.has_dropout:
            future = future + self.drop(self.out_f(x_f))
        else:
            future = future + self.out_f(x_f)

        x_p = x_p.transpose(1, 2).reshape(N, T_p, -1)
        if self.has_dropout:
            past = past + self.drop(self.out_p(x_p))
        else:
            past = past + self.out_p(x_p)

        # MLP
        past   = past   + self.mlp_p(self.norm_p2(past)   * (1 + sp_m) + sh_m)
        future = future + self.mlp_f(self.norm_f2(future) * (1 + sf_m) + shf_m)

        if return_attn:
            return past, future, attn_dict
        return past, future


# ══════════════════════════════════════════════════════════════════════════════
#  PoseTrajFM — dual-stream flow-matching model (traj + root-relative pose)
# ══════════════════════════════════════════════════════════════════════════════

class PoseTrajFM(nn.Module):
    """
    Dual-input flow-matching model for joint trajectory + pose prediction.
    Adapted from MoCap UMPM to WorldPose 24-joint football data.

    Separate projections for root trajectory (3D) and root-relative pose (J*3),
    summed per timestep. Dual-stream (past/future) adaLN transformer backbone.
    Two output heads decode trajectory and pose velocities.
    """

    def __init__(
        self,
        depth=6,
        embed_dim=192,
        coord_dim=3,
        num_joints=24,
        input_time=25,
        output_time=25,
        num_heads=4,
        mlp_ratio=2,
        dropout=0.1,
        causal_future=False,
        n_ctrl_traj=None,
        n_ctrl_pose=None,
        spline_type="cubic",
        use_agent_neighbors=False,
        use_joint_head=False,
        n_ctrl_joint=None,
        use_qk_norm=True,
        use_learned_grouping=False,
        n_groups=2,
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        self.coord_dim   = coord_dim
        self.num_joints  = num_joints
        self.pose_dim    = num_joints * coord_dim
        self.input_time  = input_time
        self.output_time = output_time
        self.n_ctrl_traj = n_ctrl_traj
        self.n_ctrl_pose = n_ctrl_pose
        self.use_spline_traj = n_ctrl_traj is not None
        self.use_spline_pose = n_ctrl_pose is not None
        self.use_spline  = self.use_spline_traj or self.use_spline_pose
        self.use_agent_neighbors = use_agent_neighbors
        self.use_joint_head = use_joint_head
        self.n_ctrl_joint = n_ctrl_joint
        self.use_qk_norm = use_qk_norm

        # ── B-spline basis buffers ───────────────────────────────────────────
        if self.use_spline:
            # modality type embedding: traj vs pose vs joint ctrl-pt tokens
            self.modality_emb = nn.Embedding(3, embed_dim)

            if self.use_spline_traj:
                B_t, B_t_pinv, anc_t = make_spline_basis(n_ctrl_traj, output_time, spline_type)
                self.register_buffer("B_traj",      B_t)
                self.register_buffer("B_traj_pinv", B_t_pinv)
                self.register_buffer("anchor_traj", anc_t)
            else:
                # frame-level traj: create uniform anchors for PE
                self.register_buffer("anchor_traj",
                    torch.linspace(0, output_time - 1, output_time).long())

            if self.use_spline_pose:
                B_p, B_p_pinv, anc_p = make_spline_basis(n_ctrl_pose, output_time, spline_type)
                self.register_buffer("B_pose",      B_p)
                self.register_buffer("B_pose_pinv", B_p_pinv)
                self.register_buffer("anchor_pose", anc_p)

            # joint-space spline basis (higher ctrl pts for full J*3 output)
            if use_joint_head and n_ctrl_joint is not None:
                B_j, B_j_pinv, anc_j = make_spline_basis(n_ctrl_joint, output_time, spline_type)
                self.register_buffer("B_joint", B_j)
                self.register_buffer("B_joint_pinv", B_j_pinv)
                self.register_buffer("anchor_joint", anc_j)

        # ── Past input encoding (traj + structured per-joint pose, summed) ───
        self.scale_past_traj = nn.Linear(coord_dim, embed_dim, bias=False)
        self.joint_embed = nn.Linear(coord_dim, embed_dim // 4)
        self.joint_type_emb = nn.Embedding(num_joints, embed_dim // 4)
        self.joint_agg = nn.Sequential(
            nn.Linear(num_joints * (embed_dim // 4), embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.pe_past = PositionalEncoding(embed_dim, max_len=input_time)

        # ── Future (noisy) encoding ──────────────────────────────────────────
        self.scale_fut_traj = nn.Linear(coord_dim, embed_dim, bias=False)
        self.joint_embed_fut = nn.Linear(coord_dim, embed_dim // 4)
        self.joint_agg_fut = nn.Sequential(
            nn.Linear(num_joints * (embed_dim // 4), embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.pe_fut = PositionalEncoding(embed_dim, max_len=output_time)

        # ── Timestep + context conditioning ──────────────────────────────────
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim),
        )
        # 8 modulation vectors per block
        self.cond_mlp = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, embed_dim * 8))

        # ── Transformer blocks ───────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  dropout=dropout, causal_future=causal_future, use_qk_norm=use_qk_norm)
            for _ in range(depth)
        ])

        # ── Pose-only refinement blocks ──────────────────────────────────────
        self.pose_refine_blocks = nn.ModuleList([
            PoseRefineBlock(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(3)
        ])

        # ── Output heads ─────────────────────────────────────────────────────
        self.final_norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.final_mod  = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 2 * embed_dim))
        self.head_traj  = nn.Linear(embed_dim, coord_dim)
        # pose head with residual
        self.pose_final_norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.pose_final_mod  = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 2 * embed_dim))
        self.head_pose_pre  = nn.Linear(embed_dim, embed_dim)
        self.head_pose_res  = nn.Sequential(
            nn.GELU(), nn.Linear(embed_dim, embed_dim), nn.GELU(),
        )
        self.head_pose_out  = nn.Linear(embed_dim, self.pose_dim)

        self.frame_drop_prob = FRAME_DROP_PROB

        # ── Joint-space output head: predicts absolute J*3 directly ──────────
        if use_joint_head:
            joint_dim = num_joints * coord_dim
            self.joint_final_norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
            self.joint_final_mod  = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 2 * embed_dim))
            self.head_joint_pre = nn.Linear(embed_dim, embed_dim)
            self.head_joint_res = nn.Sequential(
                nn.GELU(), nn.Linear(embed_dim, embed_dim), nn.GELU(),
            )
            self.head_joint_out = nn.Linear(embed_dim, joint_dim)

        # ── neighbor-agent conditioning ──────────────────────────────────────
        # ego type = 0, neighbor type = 1
        if use_agent_neighbors:
            self.neighbor_type_emb = nn.Embedding(3, embed_dim)  # 3 rows kept so existing checkpoints load
            # project neighbor root trajectories (3D) into embed space
            self.scale_neighbor_traj = nn.Linear(coord_dim, embed_dim, bias=False)

        # ── unsupervised neighbor grouping ────────────────────────────────────
        # learns to assign each neighbor to one of G groups via attention to
        # learnable prototypes; replaces uniform type=1 with per-group embeddings
        self.use_learned_grouping = use_learned_grouping
        self.n_groups = n_groups
        if use_learned_grouping and use_agent_neighbors:
            self.group_prototypes = nn.Parameter(torch.randn(n_groups, embed_dim) * 0.02)
            self.group_type_emb = nn.Embedding(n_groups, embed_dim)
            self.group_query_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, past_traj, past_pose, x_t_traj, x_t_pose, t,
                cfg_uncond=False, neighbors=None, neighbor_mask=None,
                return_attn=False):
        """
        past_traj      : [N, T_in,  3]
        past_pose      : [N, T_in,  J*3]   root-relative
        x_t_traj       : [N, D_t|T_out, 3]     noisy future
        x_t_pose       : [N, D_p|T_out, J*3]   noisy future
        t              : [N]               flow-matching timestep
        neighbors      : [N, K, T_in, 3]  neighbor past root traj (focal-centric), optional
        neighbor_mask  : [N, K]            1.0=valid, 0.0=padded, optional
        return_attn    : bool              if True, return per-block attention weights

        Returns (vel_traj, vel_pose, vel_joint) or
                (vel_traj, vel_pose, vel_joint, attn_list) when return_attn=True
        """
        N    = past_traj.shape[0]
        T_in = past_traj.shape[1]

        # ── structured per-joint encoding ────────────────────────────────────
        joint_ids = torch.arange(self.num_joints, device=past_traj.device)
        jtype = self.joint_type_emb(joint_ids)  # [J, D//4]

        def encode_pose_structured(pose_flat, joint_proj, joint_agg_layer):
            T = pose_flat.shape[1]
            p = pose_flat.reshape(N, T, self.num_joints, self.coord_dim)
            j_feat = joint_proj(p) + jtype.unsqueeze(0).unsqueeze(0)
            j_flat = j_feat.reshape(N, T, -1)
            return joint_agg_layer(j_flat)

        # ── encode past ─────────────────────────────────────────────────────
        past_pose_enc = encode_pose_structured(past_pose, self.joint_embed, self.joint_agg)
        past_tok = self.pe_past(
            self.scale_past_traj(past_traj) + past_pose_enc
        )

        # ── encode noisy future ──────────────────────────────────────────────
        if self.use_spline:
            # traj tokens: frame-level or spline ctrl pts
            D_t = self.n_ctrl_traj if self.use_spline_traj else self.output_time
            D_p = self.n_ctrl_pose if self.use_spline_pose else self.output_time
            traj_emb = self.scale_fut_traj(x_t_traj)
            pose_emb = encode_pose_structured(x_t_pose, self.joint_embed_fut, self.joint_agg_fut)
            pe_traj = self.pe_fut.pe[:, self.anchor_traj, :]
            pe_pose = self.pe_fut.pe[:, self.anchor_pose, :]
            mod_traj = self.modality_emb(torch.zeros(D_t, dtype=torch.long, device=past_traj.device))
            mod_pose = self.modality_emb(torch.ones(D_p, dtype=torch.long, device=past_traj.device))
            traj_emb = traj_emb + pe_traj + mod_traj
            pose_emb = pose_emb + pe_pose + mod_pose
            future = torch.cat([traj_emb, pose_emb], dim=1)
        else:
            fut_pose_enc = encode_pose_structured(x_t_pose, self.joint_embed_fut, self.joint_agg_fut)
            future = self.pe_fut(
                self.scale_fut_traj(x_t_traj) + fut_pose_enc
            )

        # ── neighbor conditioning: concat neighbor tokens to past ────────────
        if neighbors is not None and self.use_agent_neighbors:
            _, K, T_n, C_n = neighbors.shape
            neigh_flat = neighbors.reshape(N * K, T_n, C_n)
            neigh_tok  = self.scale_neighbor_traj(neigh_flat)     # [N*K, T_n, D]
            neigh_tok  = self.pe_past(neigh_tok)                  # [N*K, T_n, D]
            neigh_tok  = neigh_tok.reshape(N, K * T_n, -1)        # [N, K*T_in, D]

            # type embeddings: ego=0, neighbor=1
            ego_type   = self.neighbor_type_emb(
                torch.zeros(N, T_in,    dtype=torch.long, device=past_traj.device)
            )

            # neighbor type embedding: learned grouping or uniform
            if self.use_learned_grouping:
                # compute per-neighbor summary by mean-pooling over time
                neigh_summary = neigh_tok.reshape(N, K, T_n, -1).mean(dim=2)  # [N, K, D]
                # attend to group prototypes to get soft assignment
                q = self.group_query_proj(neigh_summary)                      # [N, K, D]
                attn_logits = torch.matmul(q, self.group_prototypes.T)        # [N, K, G]
                attn_logits = attn_logits / (self.embed_dim ** 0.5)
                group_weights = F.softmax(attn_logits, dim=-1)                # [N, K, G]
                # mix per-group type embeddings
                group_embs = self.group_type_emb.weight                       # [G, D]
                neigh_group_emb = torch.matmul(group_weights, group_embs)     # [N, K, D]
                # broadcast to all time steps: [N, K, D] -> [N, K*T_n, D]
                neigh_type = neigh_group_emb.unsqueeze(2).expand(N, K, T_n, -1).reshape(N, K * T_n, -1)
                # store assignment weights for analysis
                self._last_group_weights = group_weights.detach()
            else:
                neigh_type = self.neighbor_type_emb(
                    torch.ones(N, K * T_n, dtype=torch.long, device=past_traj.device)
                )

            # mask invalid neighbors
            if neighbor_mask is not None:
                mask_exp = neighbor_mask.repeat_interleave(T_n, dim=1).unsqueeze(-1).float()
                neigh_tok = neigh_tok * mask_exp

            past = torch.cat([past_tok + ego_type, neigh_tok + neigh_type], dim=1)
        else:
            past = past_tok

        # frame dropout during training
        if self.training and self.frame_drop_prob > 0:
            drop = int(torch.rand(1).item() * (T_in + 1) * self.frame_drop_prob)
            if drop > 0:
                past = past.clone()
                past[:, :drop] = 0.0

        # ── conditioning ─────────────────────────────────────────────────────
        t_emb = timestep_embedding(t, self.embed_dim)
        t_emb = self.time_mlp(t_emb).unsqueeze(1)
        past_ctx = past[:, :T_in, :].mean(dim=1, keepdim=True)
        if cfg_uncond:
            past_ctx = torch.zeros_like(past_ctx)
        ctx   = t_emb + past_ctx
        cond  = self.cond_mlp(ctx)

        # ── transformer blocks ───────────────────────────────────────────────
        attn_list = [] if return_attn else None
        for block in self.blocks:
            if return_attn:
                past_fix, future, attn_dict = block(past, future, cond, return_attn=True)
                attn_list.append(attn_dict)
            else:
                past_fix, future = block(past, future, cond)

        # ── split back into traj / pose streams ──────────────────────────────
        if self.use_spline:
            n_traj_tokens = self.n_ctrl_traj if self.use_spline_traj else self.output_time
            future_traj_raw = future[:, :n_traj_tokens]
            future_pose_raw = future[:, n_traj_tokens:]
        else:
            future_traj_raw = future
            future_pose_raw = future

        # ── final norm + modulation (traj) ───────────────────────────────────
        shift, scale = self.final_mod(ctx).chunk(2, dim=-1)
        future_traj = self.final_norm(future_traj_raw) * (1 + scale) + shift

        # ── pose refinement stream ───────────────────────────────────────────
        future_pose = future_pose_raw
        for prb in self.pose_refine_blocks:
            future_pose = prb(future_pose)
        shift_p, scale_p = self.pose_final_mod(ctx).chunk(2, dim=-1)
        future_pose = self.pose_final_norm(future_pose) * (1 + scale_p) + shift_p

        # ── decode velocity ──────────────────────────────────────────────────
        vel_traj = self.head_traj(future_traj)
        h = self.head_pose_pre(future_pose)
        h = h + self.head_pose_res(h)
        vel_pose = self.head_pose_out(h)

        # ── joint-space head: decode absolute joints from combined future ────
        vel_joint = None
        if self.use_joint_head:
            # use the full future representation (all traj+pose tokens concatenated)
            shift_j, scale_j = self.joint_final_mod(ctx).chunk(2, dim=-1)
            future_joint = self.joint_final_norm(future) * (1 + scale_j) + shift_j
            hj = self.head_joint_pre(future_joint)
            hj = hj + self.head_joint_res(hj)
            vel_joint = self.head_joint_out(hj)

        if return_attn:
            return vel_traj, vel_pose, vel_joint, attn_list
        return vel_traj, vel_pose, vel_joint

    # ── spline helpers ───────────────────────────────────────────────────────

    def fit_ctrl_traj(self, traj):
        if not self.use_spline_traj:
            return traj
        return torch.einsum('dt,...tc->...dc', self.B_traj_pinv, traj)

    def fit_ctrl_pose(self, pose):
        if not self.use_spline_pose:
            return pose
        return torch.einsum('dt,...tc->...dc', self.B_pose_pinv, pose)

    def eval_spline_traj(self, ctrl):
        if not self.use_spline_traj:
            return ctrl
        return torch.einsum('td,...dc->...tc', self.B_traj, ctrl)

    def eval_spline_pose(self, ctrl):
        if not self.use_spline_pose:
            return ctrl
        return torch.einsum('td,...dc->...tc', self.B_pose, ctrl)

    def fit_ctrl_joint(self, joints):
        """Project full-resolution absolute joints to joint-space ctrl pts."""
        if not self.use_spline or not self.use_joint_head:
            return joints
        return torch.einsum('dt,...tc->...dc', self.B_joint_pinv, joints)

    def eval_spline_joint(self, ctrl):
        """Evaluate joint-space spline from ctrl pts to full resolution."""
        if not self.use_spline or not self.use_joint_head:
            return ctrl
        return torch.einsum('td,...dc->...tc', self.B_joint, ctrl)


# ══════════════════════════════════════════════════════════════════════════════
#  Training utilities
# ══════════════════════════════════════════════════════════════════════════════

def sample_t(N, device):
    if TIMESTEP_MODE == "uniform" or TIMESTEP_MODE == "continuous":
        return torch.rand(N, device=device)
    if TIMESTEP_MODE == "logit_normal":
        return torch.sigmoid(torch.randn(N, device=device) * 1.5 - 0.5)
    raise ValueError(f"Unknown TIMESTEP_MODE: {TIMESTEP_MODE!r}")


def mirror_skeleton(traj, pose, N):
    """Swap left/right joints and negate the lateral axis."""
    mm = torch.tensor(MIRROR_MAP, device=traj.device)
    T = pose.shape[1]
    p = pose.reshape(N, T, NUM_JOINTS, COORD_DIM)
    p = p[:, :, mm, :].clone()
    p[:, :, :, MIRROR_AXIS] *= -1
    traj = traj.clone()
    traj[:, :, MIRROR_AXIS] *= -1
    return traj, p.reshape(N, T, POSE_DIM)


def train_step(model, batch, optimizer, device):
    """Single flow-matching training step with masked multi-person batches."""
    model.train()
    optimizer.zero_grad()

    b = prepare_batch(batch, device)
    past_traj     = b["input_traj"]
    past_pose     = b["input_pose_rel"]
    gt_traj       = b["output_traj"]
    gt_pose       = b["output_pose_rel"]
    mask          = b["valid_mask"]           # (BN,)
    neighbors     = b["neighbor_traj"]        # (BN, K, T_in, 3)
    neighbor_mask = b["neighbor_mask"]        # (BN, K)
    N = past_traj.shape[0]

    # skeleton mirroring augmentation
    if MIRROR_PROB > 0:
        do_mirror = torch.rand(N, device=device) < MIRROR_PROB
        if do_mirror.any():
            idx = do_mirror.nonzero(as_tuple=True)[0]
            past_traj[idx], past_pose[idx] = mirror_skeleton(past_traj[idx], past_pose[idx], len(idx))
            gt_traj[idx], gt_pose[idx] = mirror_skeleton(gt_traj[idx], gt_pose[idx], len(idx))
            # mirror neighbor root trajectories (negate lateral axis)
            neighbors[idx, :, :, MIRROR_AXIS] *= -1

    # input noise augmentation
    if INPUT_NOISE_STD > 0 and model.training:
        past_traj = past_traj + torch.randn_like(past_traj) * INPUT_NOISE_STD
        past_pose = past_pose + torch.randn_like(past_pose) * INPUT_NOISE_STD

    t = sample_t(N, device)
    cfg_uncond = False
    if CFG_DROP_PROB > 0 and model.training:
        cfg_uncond = torch.rand(1).item() < CFG_DROP_PROB

    # project GT to ctrl-pt space (displacement for traj if enabled)
    gt_traj_target = gt_traj
    if TRAJ_DISPLACEMENT:
        # per-frame displacement: Δx[0]=x[0], Δx[t]=x[t]-x[t-1]
        gt_traj_target = torch.cat([
            gt_traj[:, :1],
            gt_traj[:, 1:] - gt_traj[:, :-1],
        ], dim=1)
    gt_traj_lat = model.fit_ctrl_traj(gt_traj_target)
    gt_pose_lat = model.fit_ctrl_pose(gt_pose)

    # noise in latent space
    noise_traj = torch.randn_like(gt_traj_lat) * NOISE_SCALE
    noise_pose = torch.randn_like(gt_pose_lat) * NOISE_SCALE_POSE

    # interpolate
    t_view = t.view(N, 1, 1)
    x_t_traj = torch.lerp(noise_traj, gt_traj_lat, t_view)
    x_t_pose = torch.lerp(noise_pose, gt_pose_lat, t_view)

    # predict velocity
    pred_v_traj, pred_v_pose, pred_v_joint = model(
        past_traj, past_pose, x_t_traj, x_t_pose, t,
        cfg_uncond=cfg_uncond,
        neighbors=neighbors if USE_AGENT_NEIGHBORS else None,
        neighbor_mask=neighbor_mask if USE_AGENT_NEIGHBORS else None,
    )

    # target velocity (conditional flow matching)
    target_v_traj = gt_traj_lat - noise_traj
    target_v_pose = gt_pose_lat - noise_pose

    # compute valid count early (needed by joint loss and all subsequent losses)
    valid = mask.sum().clamp(min=1.0)

    # joint-space GT: absolute joint positions = root-relative pose + root traj
    # project to joint ctrl-pt space for flow-matching target
    joint_loss = torch.tensor(0.0, device=device)
    if USE_JOINT_HEAD and pred_v_joint is not None:
        gt_abs_joints_full = (gt_pose.reshape(N, OUTPUT_TIME, NUM_JOINTS, COORD_DIM)
                              + gt_traj.unsqueeze(2))  # (N, T_out, J, 3)
        gt_joints_flat = gt_abs_joints_full.reshape(N, OUTPUT_TIME, POSE_DIM)  # (N, T_out, J*3)
        gt_joints_lat = model.fit_ctrl_joint(gt_joints_flat)
        noise_joint = torch.randn_like(gt_joints_lat) * NOISE_SCALE_POSE
        target_v_joint = gt_joints_lat - noise_joint
        # the joint head receives the same future tokens, so pred_v_joint shape
        # matches the number of future tokens (n_ctrl_traj + n_ctrl_pose for spline)
        # we need to aggregate: use all future tokens → project down to n_ctrl_joint
        # For now, compute loss in full-res space by evaluating spline
        if model.use_spline and model.use_joint_head:
            T_joint_lat = model.n_ctrl_joint if model.n_ctrl_joint else OUTPUT_TIME
            # pred_v_joint has shape (N, n_ctrl_traj + n_ctrl_pose, J*3)
            # average-pool to n_ctrl_joint tokens
            n_fut_tok = pred_v_joint.shape[1]
            if n_fut_tok != T_joint_lat:
                pred_v_joint = F.adaptive_avg_pool1d(
                    pred_v_joint.transpose(1, 2), T_joint_lat
                ).transpose(1, 2)
            # evaluate both pred and target in full-res
            pred_joint_full = model.eval_spline_joint(pred_v_joint)
            target_joint_full = model.eval_spline_joint(target_v_joint)
        else:
            pred_joint_full = pred_v_joint
            target_joint_full = target_v_joint
        joint_err = (pred_joint_full - target_joint_full).reshape(N, OUTPUT_TIME, NUM_JOINTS, COORD_DIM)
        err_joint_per_frame = torch.norm(joint_err, dim=-1).mean(dim=-1)  # (N, T_out)
        w_joint = torch.exp(torch.linspace(0, 3.0, OUTPUT_TIME, device=device))
        w_joint = w_joint / w_joint.mean()
        joint_loss = ((err_joint_per_frame * w_joint.unsqueeze(0)).mean(dim=1) * mask).sum() / valid

    # compute per-sample errors, weighting by valid_mask

    if model.use_spline_traj or model.use_spline_pose:
        if model.use_spline_traj:
            res_traj_full = model.eval_spline_traj(pred_v_traj - target_v_traj)
        else:
            res_traj_full = pred_v_traj - target_v_traj
        err_traj = torch.norm(res_traj_full, dim=-1)                      # (N, T_out or D_t)
        if model.use_spline_pose:
            res_pose_full = model.eval_spline_pose(pred_v_pose - target_v_pose)
        else:
            res_pose_full = pred_v_pose - target_v_pose
        pose_err_per_joint = res_pose_full.reshape(N, OUTPUT_TIME, NUM_JOINTS, COORD_DIM)
        err_pose = torch.norm(pose_err_per_joint, dim=-1).mean(dim=-1)    # (N, T_out)
    else:
        err_traj = torch.norm(pred_v_traj - target_v_traj, dim=-1)
        pose_err_per_joint = (pred_v_pose - target_v_pose).reshape(N, OUTPUT_TIME, NUM_JOINTS, COORD_DIM)
        err_pose = torch.norm(pose_err_per_joint, dim=-1).mean(dim=-1)

    # temporal weighting
    w_traj = torch.exp(torch.linspace(0, 5.0, OUTPUT_TIME, device=device))
    w_traj = w_traj / w_traj.mean()
    w_pose = torch.exp(torch.linspace(0, POSE_FDE_EXP, OUTPUT_TIME, device=device))
    w_pose = w_pose / w_pose.mean()

    # masked mean
    loss_traj = ((err_traj * w_traj.unsqueeze(0)).mean(dim=1) * mask).sum() / valid
    loss_pose = ((err_pose * w_pose.unsqueeze(0)).mean(dim=1) * mask).sum() / valid

    # velocity regularization
    vel_reg = torch.tensor(0.0, device=device)
    if VEL_REG_WEIGHT > 0:
        full_v_pose = model.eval_spline_pose(pred_v_pose)
        pose_accel = full_v_pose[:, 2:] - 2 * full_v_pose[:, 1:-1] + full_v_pose[:, :-2]
        vel_reg = (pose_accel.pow(2).mean(dim=(1, 2)) * mask).sum() / valid

    # JPE proxy loss — full-horizon or last-frame
    jpe_proxy_loss = torch.tensor(0.0, device=device)
    if JPE_PROXY_WEIGHT > 0:
        x1_traj_lat = x_t_traj + pred_v_traj * (1 - t_view)
        x1_pose_lat = x_t_pose + pred_v_pose * (1 - t_view)
        x1_traj_full = model.eval_spline_traj(x1_traj_lat)
        x1_pose_full = model.eval_spline_pose(x1_pose_lat)
        # displacement→position conversion for proxy losses
        if TRAJ_DISPLACEMENT:
            x1_traj_full = torch.cumsum(x1_traj_full, dim=1)

        if JPE_PROXY_FULL_HORIZON:
            # absolute joint positions at ALL output frames
            x1_abs_all = (x1_pose_full.reshape(N, OUTPUT_TIME, NUM_JOINTS, COORD_DIM)
                          + x1_traj_full.unsqueeze(2))          # (N, T_out, J, 3)
            gt_abs_all = (gt_pose.reshape(N, OUTPUT_TIME, NUM_JOINTS, COORD_DIM)
                          + gt_traj.unsqueeze(2))               # (N, T_out, J, 3)
            # per-joint error, temporally weighted (later frames matter more)
            jpe_per_frame = torch.norm(x1_abs_all - gt_abs_all, dim=-1).mean(dim=-1)  # (N, T_out)
            w_jpe = torch.exp(torch.linspace(0, 3.0, OUTPUT_TIME, device=device))
            w_jpe = w_jpe / w_jpe.mean()
            jpe_err = (jpe_per_frame * w_jpe.unsqueeze(0)).mean(dim=1)  # (N,)
        else:
            # original: last-frame only
            x1_abs_joints = x1_pose_full[:, -1].reshape(N, NUM_JOINTS, COORD_DIM) + x1_traj_full[:, -1:, :]
            gt_abs_joints = gt_pose[:, -1].reshape(N, NUM_JOINTS, COORD_DIM) + gt_traj[:, -1:, :]
            jpe_err = torch.norm(x1_abs_joints - gt_abs_joints, dim=-1).mean(dim=-1)  # (N,)

        # timestep weighting: "clamp" avoids zeroing out early timesteps
        if JPE_PROXY_T_WEIGHTING == "t_sq":
            t_weight = t * t
        elif JPE_PROXY_T_WEIGHTING == "clamp":
            t_weight = t.clamp(min=0.2)
        else:
            t_weight = torch.ones_like(t)
        jpe_proxy_loss = (jpe_err * t_weight * mask).sum() / valid

    # trajectory FDE: multi-horizon or last-frame displacement error
    traj_fde_loss = torch.tensor(0.0, device=device)
    if TRAJ_FDE_WEIGHT > 0:
        # reuse x1_traj_full if already computed by JPE proxy, else compute
        if JPE_PROXY_WEIGHT <= 0:
            x1_traj_lat = x_t_traj + pred_v_traj * (1 - t_view)
            x1_traj_full = model.eval_spline_traj(x1_traj_lat)
            if TRAJ_DISPLACEMENT:
                x1_traj_full = torch.cumsum(x1_traj_full, dim=1)
        t_weight_fde = t * t
        if TRAJ_MULTI_HORIZON_FDE:
            # FDE at 25%, 50%, 75%, 100% of output
            horizon_frames = [OUTPUT_TIME // 4, OUTPUT_TIME // 2, 3 * OUTPUT_TIME // 4, OUTPUT_TIME - 1]
            fde_sum = torch.zeros(N, device=device)
            for hf in horizon_frames:
                fde_sum = fde_sum + torch.norm(x1_traj_full[:, hf] - gt_traj[:, hf], dim=-1)
            fde_err = fde_sum / len(horizon_frames)
        else:
            fde_err = torch.norm(x1_traj_full[:, -1] - gt_traj[:, -1], dim=-1)  # (N,)
        traj_fde_loss = (fde_err * t_weight_fde * mask).sum() / valid

    loss = (TRAJ_LOSS_WEIGHT * loss_traj + POSE_LOSS_WEIGHT * loss_pose +
            VEL_REG_WEIGHT * vel_reg +
            JPE_PROXY_WEIGHT * jpe_proxy_loss + TRAJ_FDE_WEIGHT * traj_fde_loss +
            JOINT_LOSS_WEIGHT * joint_loss)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item(), loss_traj.item(), loss_pose.item()


# ══════════════════════════════════════════════════════════════════════════════
#  ODE sampler (Heun corrector)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def euler_sample_posetraj(model, past_traj, past_pose, K, device,
                          inference_steps, noise_scale, chunk_k=10,
                          neighbors=None, neighbor_mask=None):
    """
    Euler ODE sampler for joint traj+pose.
    Runs in ctrl-pt space when spline enabled; returns full-res output.

    Returns (pred_traj, pred_pose, pred_joints):
        pred_traj   : [N, K, T_out, 3]
        pred_pose   : [N, K, T_out, J*3]
        pred_joints : [N, K, T_out, J*3] or None if joint head disabled
    """
    if K > chunk_k:
        traj_chunks, pose_chunks, joint_chunks = [], [], []
        for k_start in range(0, K, chunk_k):
            k_end = min(k_start + chunk_k, K)
            ct, cp, cj = euler_sample_posetraj(
                model, past_traj, past_pose, k_end - k_start,
                device, inference_steps, noise_scale, chunk_k=chunk_k,
                neighbors=neighbors, neighbor_mask=neighbor_mask,
            )
            traj_chunks.append(ct)
            pose_chunks.append(cp)
            if cj is not None:
                joint_chunks.append(cj)
        pred_joints = torch.cat(joint_chunks, dim=1) if joint_chunks else None
        return torch.cat(traj_chunks, dim=1), torch.cat(pose_chunks, dim=1), pred_joints

    N, T_in, _ = past_traj.shape

    # expand past for K samples
    pt_exp = past_traj.unsqueeze(1).expand(-1, K, -1, -1).reshape(N * K, T_in, COORD_DIM)
    pp_exp = past_pose.unsqueeze(1).expand(-1, K, -1, -1).reshape(N * K, T_in, POSE_DIM)

    # expand neighbors for K samples
    nb_exp, nm_exp = None, None
    if neighbors is not None:
        K_n = neighbors.shape[1]
        nb_exp = neighbors.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(N * K, K_n, T_in, COORD_DIM)
        nm_exp = neighbor_mask.unsqueeze(1).expand(-1, K, -1).reshape(N * K, K_n)

    # initial noise in latent space
    T_lat_traj = model.n_ctrl_traj if model.use_spline_traj else OUTPUT_TIME
    T_lat_pose = model.n_ctrl_pose if model.use_spline_pose else OUTPUT_TIME
    z_traj = torch.randn(N * K, T_lat_traj, COORD_DIM, device=device) * noise_scale
    z_pose = torch.randn(N * K, T_lat_pose, POSE_DIM, device=device) * NOISE_SCALE_POSE

    # joint-space ODE state (if joint head is enabled)
    has_joint_head = model.use_joint_head
    T_lat_joint = model.n_ctrl_joint if (model.use_spline and model.use_joint_head and model.n_ctrl_joint) else OUTPUT_TIME
    z_joint = None
    if has_joint_head:
        z_joint = torch.randn(N * K, T_lat_joint, POSE_DIM, device=device) * NOISE_SCALE_POSE

    timesteps = torch.linspace(0, 1, inference_steps + 1, device=device)

    model.eval()
    cfg_w = CFG_SCALE
    use_cfg = cfg_w != 1.0 and CFG_DROP_PROB > 0

    def guided_velocity(pt, pp, zt, zp, tv):
        v_cond_t, v_cond_p, v_cond_j = model(pt, pp, zt, zp, tv, cfg_uncond=False,
                                    neighbors=nb_exp, neighbor_mask=nm_exp)
        if not use_cfg:
            return v_cond_t, v_cond_p, v_cond_j
        v_uncond_t, v_uncond_p, v_uncond_j = model(pt, pp, zt, zp, tv, cfg_uncond=True,
                                        neighbors=nb_exp, neighbor_mask=nm_exp)
        v_t = v_uncond_t + cfg_w * (v_cond_t - v_uncond_t)
        v_p = v_uncond_p + cfg_w * (v_cond_p - v_uncond_p)
        v_j = None
        if v_cond_j is not None and v_uncond_j is not None:
            v_j = v_uncond_j + cfg_w * (v_cond_j - v_uncond_j)
        return v_t, v_p, v_j

    for i in range(inference_steps):
        t_val = timesteps[i].item()
        dt    = (timesteps[i + 1] - timesteps[i]).item()
        t_vec = torch.full((N * K,), t_val, device=device)

        # Euler step
        v_traj, v_pose, v_joint = guided_velocity(pt_exp, pp_exp, z_traj, z_pose, t_vec)
        z_traj = z_traj + v_traj * dt
        z_pose = z_pose + v_pose * dt
        # joint-space ODE step: pool v_joint to match z_joint temporal dim
        if has_joint_head and v_joint is not None:
            n_fut = v_joint.shape[1]
            if n_fut != T_lat_joint:
                v_joint_pooled = F.adaptive_avg_pool1d(
                    v_joint.transpose(1, 2), T_lat_joint
                ).transpose(1, 2)
            else:
                v_joint_pooled = v_joint
            z_joint = z_joint + v_joint_pooled * dt

    # reshape to [N, K, T_lat, C]
    z_traj = z_traj.view(N, K, T_lat_traj, COORD_DIM)
    z_pose = z_pose.view(N, K, T_lat_pose, POSE_DIM)

    # evaluate spline back to full resolution
    pred_traj = model.eval_spline_traj(z_traj)
    pred_pose = model.eval_spline_pose(z_pose)

    # displacement→position conversion
    if TRAJ_DISPLACEMENT:
        pred_traj = torch.cumsum(pred_traj, dim=2)  # dim=2 because shape is [N, K, T, 3]

    # joint-space head output
    pred_joints = None
    if has_joint_head and z_joint is not None:
        z_joint = z_joint.view(N, K, T_lat_joint, POSE_DIM)
        pred_joints = model.eval_spline_joint(z_joint)

    return pred_traj, pred_pose, pred_joints


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation — last-frame APE + JPE with kmeans selection
# ══════════════════════════════════════════════════════════════════════════════

def _weighted_kmeans_select(pred_traj_abs, pred_pose, k=6, traj_scale=5.0, n_iter=30,
                           frame_idx=None):
    """Kmeans on features at a given frame with trajectory dims scaled up.
    Returns medoid indices [N, k]."""
    if frame_idx is None:
        frame_idx = OUTPUT_TIME - 1
    N, K_total = pred_traj_abs.shape[:2]
    traj_feat = pred_traj_abs[:, :, frame_idx] * traj_scale
    pose_feat = pred_pose[:, :, frame_idx]
    feat = torch.cat([traj_feat, pose_feat], dim=-1)
    # kmeans
    centers = feat[:, :k].clone()
    for _ in range(n_iter):
        dists = torch.cdist(feat, centers)
        assignments = dists.argmin(dim=-1)
        one_hot = torch.zeros(N, K_total, k, device=feat.device).scatter_(-1, assignments.unsqueeze(-1), 1.0)
        cluster_sums = torch.bmm(one_hot.transpose(1, 2), feat)
        counts = one_hot.sum(dim=1).unsqueeze(-1).clamp(min=1)
        new_centers = cluster_sums / counts
        empty = (one_hot.sum(dim=1) == 0).unsqueeze(-1)
        centers = torch.where(empty, centers, new_centers)
    return torch.cdist(centers, feat).argmin(dim=-1)  # [N, k]


@torch.no_grad()
def evaluate(ema_model, val_loader, device, k_predictions, inference_steps,
             noise_scale, max_batches=None, kmeans_select_k=6, traj_scale=5.0,
             jpe_mean=None, jpe_std=None):
    """
    Compute APE and JPE at multiple horizons (1s/2s/3s), best-of-K, in mm.
    Includes sequence-level JPE (JPE_seq_1s) matching deterministic protocol.
    Optionally evaluates joint-space head predictions.
    Handles multi-person masking via valid_mask.
    """
    ema_model.eval()

    # accumulators per horizon
    horizon_ape = {h: [] for h in EVAL_HORIZONS}
    horizon_jpe = {h: [] for h in EVAL_HORIZONS}
    # sequence-level JPE: mean over first 25 frames (1s), best-of-K over whole seq
    seq_jpe_1s = []
    # joint-space head metrics (if available)
    horizon_jpe_joint = {h: [] for h in EVAL_HORIZONS}
    seq_jpe_1s_joint = []

    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break

        b = prepare_batch(batch, device)
        past_traj  = b["input_traj"]
        past_pose  = b["input_pose_rel"]
        gt_traj    = b["output_traj"]
        gt_pose    = b["output_pose_rel"]
        offset     = b["offset_traj"]
        mask       = b["valid_mask"]        # (BN,)
        neighbors  = b["neighbor_traj"]     # (BN, K_n, T_in, 3)
        nb_mask    = b["neighbor_mask"]     # (BN, K_n)

        # sample K predictions: [BN, K, T_out, 3], [BN, K, T_out, J*3], [BN, K, T_out, J*3] or None
        pred_traj, pred_pose, pred_joints = euler_sample_posetraj(
            ema_model, past_traj, past_pose,
            K=k_predictions, device=device,
            inference_steps=inference_steps, noise_scale=noise_scale,
            neighbors=neighbors if USE_AGENT_NEIGHBORS else None,
            neighbor_mask=nb_mask if USE_AGENT_NEIGHBORS else None,
        )

        # denormalize if needed
        if jpe_mean is not None:
            gt_pose_dn = denormalize_joints_for_jpe(
                gt_pose.reshape(-1, OUTPUT_TIME, NUM_JOINTS, COORD_DIM), jpe_mean, jpe_std
            ).reshape(-1, OUTPUT_TIME, POSE_DIM)
            pred_pose_dn = denormalize_joints_for_jpe(
                pred_pose.reshape(-1, k_predictions, OUTPUT_TIME, NUM_JOINTS, COORD_DIM), jpe_mean, jpe_std
            ).reshape(-1, k_predictions, OUTPUT_TIME, POSE_DIM)
        else:
            gt_pose_dn = gt_pose
            pred_pose_dn = pred_pose

        # absolute trajectory
        gt_traj_abs    = gt_traj + offset
        pred_traj_abs  = pred_traj + offset.unsqueeze(1)

        BN = gt_traj.shape[0]
        K = pred_traj_abs.shape[1]

        # kmeans selection: optionally bypass for raw best-of-K
        if USE_KMEANS_SELECT and K > kmeans_select_k:
            sel_idx = _weighted_kmeans_select(pred_traj_abs, pred_pose_dn,
                                              k=kmeans_select_k, traj_scale=traj_scale,
                                              frame_idx=OUTPUT_TIME - 1)
            idx_traj = sel_idx.unsqueeze(-1).unsqueeze(-1).expand(BN, kmeans_select_k, OUTPUT_TIME, COORD_DIM)
            idx_pose = sel_idx.unsqueeze(-1).unsqueeze(-1).expand(BN, kmeans_select_k, OUTPUT_TIME, POSE_DIM)
            pred_traj_abs = torch.gather(pred_traj_abs, 1, idx_traj)
            pred_pose_dn = torch.gather(pred_pose_dn, 1, idx_pose)
            K = kmeans_select_k

        valid_idx = mask.bool()

        # compute APE/JPE at each evaluation horizon (per-frame best-of-K)
        for h_name, h_frame in EVAL_HORIZONS.items():
            # APE — root-aligned pose error (best-of-K)
            pred_pose_h = pred_pose_dn[:, :, h_frame].reshape(BN, K, NUM_JOINTS, COORD_DIM)
            gt_pose_h   = gt_pose_dn[:, h_frame].reshape(BN, NUM_JOINTS, COORD_DIM)
            pred_aligned = pred_pose_h - pred_pose_h[:, :, 0:1, :]
            gt_aligned   = gt_pose_h - gt_pose_h[:, 0:1, :]
            ape_per_mode = torch.norm(pred_aligned - gt_aligned.unsqueeze(1), dim=-1).mean(dim=-1)
            min_ape = ape_per_mode.min(dim=1).values  # (BN,)

            # JPE — absolute joint error (best-of-K)
            pred_abs_j = pred_pose_h + pred_traj_abs[:, :, h_frame].unsqueeze(2)
            gt_abs_j   = gt_pose_h + gt_traj_abs[:, h_frame].unsqueeze(1)
            jpe_per_mode = torch.norm(pred_abs_j - gt_abs_j.unsqueeze(1), dim=-1).mean(dim=-1)
            min_jpe = jpe_per_mode.min(dim=1).values  # (BN,)

            horizon_ape[h_name].extend(min_ape[valid_idx].cpu().tolist())
            horizon_jpe[h_name].extend(min_jpe[valid_idx].cpu().tolist())

        # ── sequence-level JPE (1s = frames 0..24) matching deterministic protocol ──
        # best-of-K selected by mean JPE over the full 1s window (one mode per sample)
        T_1s = min(25, OUTPUT_TIME)
        pred_abs_seq = (pred_pose_dn[:, :, :T_1s].reshape(BN, K, T_1s, NUM_JOINTS, COORD_DIM)
                        + pred_traj_abs[:, :, :T_1s].unsqueeze(3))  # (BN, K, T_1s, J, 3)
        gt_abs_seq = (gt_pose_dn[:, :T_1s].reshape(BN, T_1s, NUM_JOINTS, COORD_DIM)
                      + gt_traj_abs[:, :T_1s].unsqueeze(2))         # (BN, T_1s, J, 3)
        # mean JPE per mode across T_1s frames and J joints
        jpe_seq_per_mode = torch.norm(
            pred_abs_seq - gt_abs_seq.unsqueeze(1), dim=-1
        ).mean(dim=(2, 3))  # (BN, K)
        min_jpe_seq = jpe_seq_per_mode.min(dim=1).values  # (BN,)
        seq_jpe_1s.extend(min_jpe_seq[valid_idx].cpu().tolist())

        # ── joint-space head evaluation ─────────────────────────────────────
        if pred_joints is not None:
            K_j = pred_joints.shape[1]
            # denormalize joint predictions if needed
            if jpe_mean is not None:
                pred_joints_dn = denormalize_joints_for_jpe(
                    pred_joints.reshape(-1, K_j, OUTPUT_TIME, NUM_JOINTS, COORD_DIM), jpe_mean, jpe_std
                ).reshape(-1, K_j, OUTPUT_TIME, POSE_DIM)
            else:
                pred_joints_dn = pred_joints
            # joint head predicts absolute joints directly (agent-centric), add offset
            pred_joints_abs = pred_joints_dn.reshape(BN, K_j, OUTPUT_TIME, NUM_JOINTS, COORD_DIM) + offset.unsqueeze(1).unsqueeze(2)

            # per-horizon JPE from joint head
            for h_name, h_frame in EVAL_HORIZONS.items():
                pj_h = pred_joints_abs[:, :, h_frame]           # (BN, K_j, J, 3)
                gt_abs_h = (gt_pose_dn[:, h_frame].reshape(BN, NUM_JOINTS, COORD_DIM)
                            + gt_traj_abs[:, h_frame].unsqueeze(1))    # (BN, J, 3)
                jpe_j = torch.norm(pj_h - gt_abs_h.unsqueeze(1), dim=-1).mean(dim=-1)  # (BN, K_j)
                min_jpe_j = jpe_j.min(dim=1).values
                horizon_jpe_joint[h_name].extend(min_jpe_j[valid_idx].cpu().tolist())

            # sequence-level JPE from joint head (1s)
            pj_seq = pred_joints_abs[:, :, :T_1s]  # (BN, K_j, T_1s, J, 3)
            gt_abs_seq_full = (gt_pose_dn[:, :T_1s].reshape(BN, T_1s, NUM_JOINTS, COORD_DIM)
                               + gt_traj_abs[:, :T_1s].unsqueeze(2))
            jpe_j_seq = torch.norm(
                pj_seq - gt_abs_seq_full.unsqueeze(1), dim=-1
            ).mean(dim=(2, 3))
            min_jpe_j_seq = jpe_j_seq.min(dim=1).values
            seq_jpe_1s_joint.extend(min_jpe_j_seq[valid_idx].cpu().tolist())

    results = {}
    for h_name in EVAL_HORIZONS:
        ape_vals = horizon_ape[h_name]
        jpe_vals = horizon_jpe[h_name]
        results[f"APE_{h_name}"] = float(sum(ape_vals) / max(len(ape_vals), 1) * 1000.0)
        results[f"JPE_{h_name}"] = float(sum(jpe_vals) / max(len(jpe_vals), 1) * 1000.0)
        # joint head per-horizon
        jpe_j_vals = horizon_jpe_joint[h_name]
        if jpe_j_vals:
            results[f"JPE_joint_{h_name}"] = float(sum(jpe_j_vals) / max(len(jpe_j_vals), 1) * 1000.0)

    # sequence-level metrics
    if seq_jpe_1s:
        results["JPE_seq_1s"] = float(sum(seq_jpe_1s) / max(len(seq_jpe_1s), 1) * 1000.0)
    if seq_jpe_1s_joint:
        results["JPE_joint_seq_1s"] = float(sum(seq_jpe_1s_joint) / max(len(seq_jpe_1s_joint), 1) * 1000.0)

    # backward-compatible keys: APE/JPE = last horizon (3s)
    last_h = list(EVAL_HORIZONS.keys())[-1]
    results["APE"] = results[f"APE_{last_h}"]
    results["JPE"] = results[f"JPE_{last_h}"]
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    global DATA_PATH, RESULTS_DIR, TRAIN_STEPS, BATCH_SIZE, LIMIT_SAMPLES

    parser = argparse.ArgumentParser(description="Train FlowMoP on processed WorldPose data.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/worldpose"))
    parser.add_argument("--train-steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    DATA_PATH = args.data_dir
    RESULTS_DIR = str(args.output_dir)
    TRAIN_STEPS = args.train_steps
    BATCH_SIZE = args.batch_size
    LIMIT_SAMPLES = args.limit_samples

    device = args.device
    if device == "cuda":
        torch.backends.cudnn.benchmark        = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
    torch.manual_seed(args.seed)

    model = PoseTrajFM(
        depth=DEPTH, embed_dim=EMBED_DIM, coord_dim=COORD_DIM,
        num_joints=NUM_JOINTS, input_time=INPUT_TIME, output_time=OUTPUT_TIME,
        num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO, dropout=DROPOUT,
        n_ctrl_traj=N_CTRL_TRAJ if USE_SPLINE_TRAJ else None,
        n_ctrl_pose=N_CTRL_POSE if USE_SPLINE else None,
        spline_type=SPLINE_TYPE,
        use_agent_neighbors=USE_AGENT_NEIGHBORS,
        use_joint_head=USE_JOINT_HEAD,
        n_ctrl_joint=N_CTRL_JOINT if (USE_SPLINE and USE_JOINT_HEAD) else None,
        use_qk_norm=USE_QK_NORM,
        use_learned_grouping=USE_LEARNED_GROUPING,
        n_groups=N_GROUPS,
    ).to(device)
    ema_model = copy.deepcopy(model)
    ema_decay = 0.9995

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nFlowMoP params={n_params:,}  device={device}")
    print(f"data={DATA_PATH}  J={NUM_JOINTS}  T_in={INPUT_TIME}  T_out={OUTPUT_TIME}")
    if model.use_spline:
        traj_info = f"D_t={N_CTRL_TRAJ}" if USE_SPLINE_TRAJ else f"T_t={OUTPUT_TIME}(frame-level)"
        pose_info = f"D_p={N_CTRL_POSE}" if USE_SPLINE else f"T_p={OUTPUT_TIME}"
        n_fut = (N_CTRL_TRAJ if USE_SPLINE_TRAJ else OUTPUT_TIME) + (N_CTRL_POSE if USE_SPLINE else OUTPUT_TIME)
        print(f"spline: {traj_info}, {pose_info}, type={SPLINE_TYPE}, "
              f"future_tokens={n_fut} (was {OUTPUT_TIME})")
        if TRAJ_DISPLACEMENT:
            print("trajectory: displacement parameterization")
    print()

    # Normalization tensors
    if NORMALIZE:
        train_loader, val_loader = get_dataloaders()
        stats = train_loader.dataset.normalization_stats
        jpe_mean = torch.tensor(stats["joints_3d_mean"], device=device, dtype=torch.float32)
        jpe_std  = torch.tensor(stats["joints_3d_std"],  device=device, dtype=torch.float32)
    else:
        jpe_mean = jpe_std = None
        train_loader, val_loader = get_dataloaders()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
    warmup_steps = min(200, TRAIN_STEPS // 10)
    # cosine with 6 warm restart cycles
    n_restarts = 6
    cycle_len = TRAIN_STEPS // n_restarts
    def lr_lambda_restart(s):
        if s < warmup_steps:
            return s / warmup_steps
        cycle_pos = (s - warmup_steps) % cycle_len
        cycle_total = cycle_len
        progress = cycle_pos / max(cycle_total, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_restart)

    steps_per_epoch = max(len(train_loader), 1)
    log_every_steps = EVAL_EVERY

    # logging state
    train_metrics_by_epoch = {}
    val_metrics_by_epoch   = {}
    log_dir  = os.path.join(RESULTS_DIR, "logs")
    ckpt_dir = os.path.join(RESULTS_DIR, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val_jpe = float("inf")

    train_log_path = os.path.join(log_dir, "train_metrics.json")
    val_log_path   = os.path.join(log_dir, "val_metrics.json")
    train_eval_batches = min(10, max(1, len(train_loader) // 4))

    # eval K: use more samples when bypassing kmeans
    eval_k_preds = K_PREDICTIONS if not USE_KMEANS_SELECT else 20

    print(f"logging every {log_every_steps} steps")
    print(f"eval: best-of-{eval_k_preds}, kmeans={'ON' if USE_KMEANS_SELECT else 'OFF'}, train={train_eval_batches} batches")
    if LIMIT_SAMPLES is not None:
        print(f"sample limit: {LIMIT_SAMPLES}, train={len(train_loader)} batches, val={len(val_loader)} batches")
    if USE_JOINT_HEAD:
        print(f"joint-space head: ON, n_ctrl_joint={N_CTRL_JOINT}, weight={JOINT_LOSS_WEIGHT}")

    # ── helpers ──────────────────────────────────────────────────────────────

    def infinite(loader):
        while True:
            yield from loader

    @torch.no_grad()
    def log_metrics(step, final=False):
        """Evaluate on train subset and full val (APE/JPE), log to JSON."""
        eval_t0 = time.time()

        train_m = evaluate(
            ema_model, train_loader, device,
            k_predictions=eval_k_preds, inference_steps=10,
            noise_scale=NOISE_SCALE,
            max_batches=train_eval_batches,
            kmeans_select_k=EVAL_K, traj_scale=5.0,
            jpe_mean=jpe_mean, jpe_std=jpe_std,
        )
        # cap val batches for quick iteration (TRAIN_STEPS <= 10000)
        val_max_b = 20 if TRAIN_STEPS <= 10000 else None
        val_m = evaluate(
            ema_model, val_loader, device,
            k_predictions=eval_k_preds, inference_steps=10,
            noise_scale=NOISE_SCALE,
            max_batches=val_max_b,
            kmeans_select_k=EVAL_K, traj_scale=5.0,
            jpe_mean=jpe_mean, jpe_std=jpe_std,
        )

        epoch_key = str(max(1, int(round(step / steps_per_epoch))))
        train_metrics_by_epoch[epoch_key] = train_m
        val_metrics_by_epoch[epoch_key]   = val_m

        with open(train_log_path, "w") as f:
            json.dump(train_metrics_by_epoch, f, indent=2)
        with open(val_log_path, "w") as f:
            json.dump(val_metrics_by_epoch, f, indent=2)

        eval_mins = (time.time() - eval_t0) / 60
        # multi-horizon logging
        train_parts = "  ".join(f"{h}: APE={train_m.get(f'APE_{h}',0):.1f} JPE={train_m.get(f'JPE_{h}',0):.1f}" for h in EVAL_HORIZONS)
        val_parts   = "  ".join(f"{h}: APE={val_m.get(f'APE_{h}',0):.1f} JPE={val_m.get(f'JPE_{h}',0):.1f}" for h in EVAL_HORIZONS)
        # sequence-level and joint head metrics
        seq_1s_str = f"  JPE_seq_1s={val_m.get('JPE_seq_1s',0):.1f}" if 'JPE_seq_1s' in val_m else ""
        joint_str = ""
        if 'JPE_joint_1s' in val_m:
            joint_str = f"  JPE_joint_1s={val_m.get('JPE_joint_1s',0):.1f}"
        if 'JPE_joint_seq_1s' in val_m:
            joint_str += f"  JPE_joint_seq={val_m.get('JPE_joint_seq_1s',0):.1f}"
        print(
            f"  step {step:>5}/{TRAIN_STEPS}  epoch={epoch_key}  ({eval_mins:.1f}min eval)\n"
            f"    train: {train_parts} mm\n"
            f"    val:   {val_parts}{seq_1s_str}{joint_str} mm"
        )

        nonlocal best_val_jpe
        # Prefer the joint-head metric when that optional head is enabled.
        val_jpe_key = "JPE"
        if USE_JOINT_HEAD and f"JPE_joint_{list(EVAL_HORIZONS.keys())[-1]}" in val_m:
            val_jpe_key = f"JPE_joint_{list(EVAL_HORIZONS.keys())[-1]}"
        current_jpe = val_m.get(val_jpe_key, val_m["JPE"])
        improved = current_jpe < best_val_jpe
        if improved:
            best_val_jpe = current_jpe
            ckpt_path = os.path.join(ckpt_dir, "best.pt")
            torch.save(ema_model.state_dict(), ckpt_path)
            print(f"  [ckpt] saved best EMA ({val_jpe_key}={current_jpe:.1f} mm) -> {ckpt_path}")
        return improved

    # ── train loop ───────────────────────────────────────────────────────────

    train_iter = infinite(train_loader)
    t0 = time.time()
    recent_loss, recent_traj, recent_pose = [], [], []
    patience_counter = 0

    for step in range(0, TRAIN_STEPS + 1):
        loss, lt, lp = train_step(model, next(train_iter), optimizer, device)
        scheduler.step()
        recent_loss.append(loss)
        recent_traj.append(lt)
        recent_pose.append(lp)

        # EMA update
        with torch.no_grad():
            for ep, p in zip(ema_model.parameters(), model.parameters()):
                ep.mul_(ema_decay).add_(p, alpha=1 - ema_decay)

        if step % 100 == 0:
            avg  = sum(recent_loss) / len(recent_loss)
            at   = sum(recent_traj) / len(recent_traj)
            ap   = sum(recent_pose) / len(recent_pose)
            mins = (time.time() - t0) / 60
            print(
                f"  step {step:>5}/{TRAIN_STEPS}  loss={avg:.4f} "
                f"(traj={at:.4f} pose={ap:.4f})  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  {mins:.1f}min"
            )
            recent_loss.clear(); recent_traj.clear(); recent_pose.clear()

        # periodic eval (skip step 0)
        if step > 0 and step % log_every_steps == 0:
            improved = log_metrics(step)
            if improved:
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  [early stop] no improvement ({patience_counter}/{EARLY_STOP_PATIENCE})")
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  [early stop] patience exhausted at step {step}, stopping.")
                break

    # final eval
    final_key = str(max(1, int(round(TRAIN_STEPS / steps_per_epoch))))
    if final_key not in val_metrics_by_epoch:
        log_metrics(TRAIN_STEPS, final=True)

    last_epoch = max(val_metrics_by_epoch.keys(), key=int)
    metrics = val_metrics_by_epoch[last_epoch]

    elapsed = (time.time() - t0) / 60
    print(f"\ndone in {elapsed:.1f} min")
    for h_name in EVAL_HORIZONS:
        print(f"  {h_name}:  APE={metrics.get(f'APE_{h_name}',0):.1f}  JPE={metrics.get(f'JPE_{h_name}',0):.1f} mm")
    if 'JPE_seq_1s' in metrics:
        print(f"  seq_1s:  JPE={metrics['JPE_seq_1s']:.1f} mm")
    if 'JPE_joint_seq_1s' in metrics:
        print(f"  joint_seq_1s: JPE={metrics['JPE_joint_seq_1s']:.1f} mm")

    # save final checkpoint
    final_ckpt = os.path.join(ckpt_dir, "final.pt")
    torch.save(ema_model.state_dict(), final_ckpt)
    print(f"  [ckpt] saved final EMA -> {final_ckpt}")



if __name__ == "__main__":
    main()

import math
import os
import time
import json
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader import FlowMoPDataset
from torch_geometric.loader import DataLoader as GeoDataLoader

# ══════════════════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════════════════

INPUT_NOISE_STD = 0.003
CFG_DROP_PROB   = 0.0
CFG_SCALE       = 1.0            # no guidance

# ── Dataset ────────────────────────────────────────────────────────────────────
DATASET_NAME = "3dpw" #"mocap_umpm"
DATA_ROOT    = None
COORD_DIM    = 3
NUM_JOINTS   = 13 #15          
POSE_DIM     = NUM_JOINTS * COORD_DIM  # 45
INPUT_TIME   = 10 #25
OUTPUT_TIME  = 20 #50
BATCH_SIZE   = 64
NUM_WORKERS  = 4
OFFSET_TRAJ  = "last_input"

# Skeleton mirroring: swap left↔right joints for augmentation
MIRROR_MAP   = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 13, 14, 12, 10, 11] if NUM_JOINTS == 15 else [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11] 
MIRROR_PROB  = 0.5
MIRROR_AXIS  = 0               # negate x-axis when mirroring

# ── Model ──────────────────────────────────────────────────────────────────────
DEPTH         = 8
EMBED_DIM     = 256
NUM_HEADS     = 4
MLP_RATIO     = 2
DROPOUT       = 0.15
K_PREDICTIONS = 6

# ── Training budget ───────────────────────────────────────────────────────────
TRAIN_STEPS    = 164000
EVAL_EVERY     = 5000
LOG_EVERY      = 200  # epochs — eval every 200 epochs
EVAL_K         = 6    # best-of-6 (no kmeans)

# ── Optimizer ─────────────────────────────────────────────────────────────────
LR           = 2e-4
WEIGHT_DECAY = 1e-2
BETAS        = (0.9, 0.999)

# ── Flow matching ─────────────────────────────────────────────────────────────
TIMESTEP_MODE   = "logit_normal"
NOISE_SCALE     = 1.0
NOISE_SCALE_POSE = 0.25
INFERENCE_STEPS = 25

# ── Loss weighting ────────────────────────────────────────────────────────────
TRAJ_LOSS_WEIGHT = 0.5
POSE_LOSS_WEIGHT = 3.0
POSE_FDE_EXP     = 2.0
FRAME_DROP_PROB  = 0.15
VEL_REG_WEIGHT   = 0.1
INPUT_JOINT_DROP = 0.0
TRAJ_FDE_WEIGHT  = 0.0
JPE_PROXY_WEIGHT = 0.3           # auxiliary absolute-joint-error loss at last frame
USE_OT_COUPLING  = False


# ── B-spline control-point reparameterization (optional) ──────────────────────
# When USE_SPLINE=True, the model predicts in a compressed ctrl-pt space.
# Traj and pose can have different ctrl-pt counts (separate temporal resolutions).
USE_SPLINE      = True
N_CTRL_TRAJ     = 8              # ctrl pts for trajectory (smooth → aggressive compress)
N_CTRL_POSE     = 24             # ctrl pts for pose (fast dynamics → mild compress)
SPLINE_TYPE     = "cubic"        # "cubic", "linear", or "dct" (truncated DCT-II basis)


RESULTS_DIR = "outputs/benchmarks"


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
    elif spline_type == "dct":
        # Truncated DCT-II basis: keep first n_ctrl frequency components
        n = torch.arange(n_eval, dtype=torch.float32)
        k = torch.arange(n_ctrl, dtype=torch.float32)
        B = torch.cos(math.pi * (n.unsqueeze(1) + 0.5) * k.unsqueeze(0) / n_eval)
        B *= math.sqrt(2.0 / n_eval)
        B[:, 0] *= 1.0 / math.sqrt(2)  # DC component normalization
        # orthonormal columns → pseudoinverse is just the transpose
        B_pinv = B.T.clone()
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
#  Data
# ══════════════════════════════════════════════════════════════════════════════

def get_dataloaders():

    train_ds = FlowMoPDataset(
        dataset=DATASET_NAME, mode=0, device="cpu", input_time=INPUT_TIME,
        use_augmented_train=False, on_the_fly_augmentations=True,
        normalize_data=False, offset_traj=OFFSET_TRAJ,
        data_root=DATA_ROOT,
        augmentation_config={
            'num_people': 3,
            'rotation': True,
            'horizontal_flip': False,   # disabled — skeleton mirroring replaces this
            'vertical_flip': True,
            'scaling': True,
            'scale_range': 0.08,
            'translation': False,
            'permute_p': True,
            'coordinate_noise': True,
            'noise_std': 0.005,         # ~5mm noise
            'time_warping': False,
        },
    )
    val_ds = FlowMoPDataset(
        dataset=DATASET_NAME, mode=1, device="cpu", input_time=INPUT_TIME,
        normalize_data=False, offset_traj=OFFSET_TRAJ, data_root=DATA_ROOT,
    )

    kw = dict(num_workers=NUM_WORKERS, pin_memory=True,
              persistent_workers=True, prefetch_factor=2)
    train_loader = GeoDataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, **kw)
    val_loader   = GeoDataLoader(val_ds,   batch_size=BATCH_SIZE * 2, shuffle=False, **kw)
    return train_loader, val_loader


def prepare_batch(batch, device):
    """
    Extract trajectory (root joint, agent-centric) and root-relative pose
    from a PyG-batched MoCap UMPM sample.

    The dataloader stores:
      input_seq  [N, T_in,  J*3]   — flattened absolute joint positions
      output_seq [N, T_out, J*3]
      input_traj [N, T_in,  3]     — root trajectory, agent-centric (offset by last_input)
      output_traj[N, T_out, 3]
      offset_traj[N, 1,     3]     — last observed root position (absolute)

    We compute root-relative pose on the fly:
      pose_abs = seq.reshape(N, T, J, 3)
      root     = pose_abs[:, :, 0:1, :]         # joint 0 = Hips
      pose_rel = (pose_abs - root).reshape(N, T, J*3)
    """
    input_seq   = batch["input_seq"].to(device)    # [N, T_in, J*3]
    output_seq  = batch["output_seq"].to(device)   # [N, T_out, J*3]
    input_traj  = batch["input_traj"].to(device)   # [N, T_in, 3]
    output_traj = batch["output_traj"].to(device)  # [N, T_out, 3]
    offset_traj = batch["offset_traj"].to(device)  # [N, 1, 3]

    N = input_seq.shape[0]

    # Root-relative pose: subtract root (joint 0) from all joints
    in_abs  = input_seq.reshape(N, INPUT_TIME, NUM_JOINTS, COORD_DIM)
    out_abs = output_seq.reshape(N, OUTPUT_TIME, NUM_JOINTS, COORD_DIM)

    in_pose_rel  = (in_abs - in_abs[:, :, 0:1, :]).reshape(N, INPUT_TIME, POSE_DIM)
    out_pose_rel = (out_abs - out_abs[:, :, 0:1, :]).reshape(N, OUTPUT_TIME, POSE_DIM)

    return {
        "input_traj":      input_traj,       # [N, T_in, 3]
        "output_traj":     output_traj,      # [N, T_out, 3]
        "input_pose_rel":  in_pose_rel,      # [N, T_in, J*3]
        "output_pose_rel": out_pose_rel,     # [N, T_out, J*3]
        "offset_traj":     offset_traj,      # [N, 1, 3]
    }


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
    """Self-attention block for pose-only future refinement (no past stream)."""
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
        # self-attention on future tokens
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
    Future tokens attend to the full past+future context;
    past tokens only attend to past.
    adaLN conditioning on the combined time+context vector.
    """
    def __init__(self, dim, num_heads=4, mlp_ratio=2, dropout=0.1, causal_future=False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads     = num_heads
        self.head_dim      = dim // num_heads
        self.causal_future = causal_future
        self.has_dropout   = dropout > 0

        self.norm_p1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm_p2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm_f1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm_f2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        self.qkv_p = nn.Linear(dim, dim * 3)
        self.qkv_f = nn.Linear(dim, dim * 3)
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

    def forward(self, past, future, cond):
        sp_a, sh_a, sf_a, shf_a, sp_m, sh_m, sf_m, shf_m = cond.chunk(8, dim=-1)

        N, T_p, _ = past.shape
        T_f       = future.shape[1]

        # ── past stream ──
        p = self.norm_p1(past) * (1 + sp_a) + sh_a
        qkv = self.qkv_p(p).reshape(N, T_p, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        qp, kp, vp = qkv.unbind(0)
        qp, kp = self.norm_q_p(qp), self.norm_k_p(kp)

        # ── future stream ──
        f = self.norm_f1(future) * (1 + sf_a) + shf_a
        qkv = self.qkv_f(f).reshape(N, T_f, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        qf, kf, vf = qkv.unbind(0)
        qf, kf = self.norm_q_f(qf), self.norm_k_f(kf)

        # future attends to all past + future tokens
        k_all = torch.cat([kp, kf], dim=-2)
        v_all = torch.cat([vp, vf], dim=-2)

        if self.causal_future:
            mask_pa = torch.ones(T_f, T_p, device=past.device, dtype=torch.bool)
            mask_ff = torch.tril(torch.ones(T_f, T_f, device=past.device, dtype=torch.bool))
            attn_mask = torch.cat([mask_pa, mask_ff], dim=-1)
            x_f = F.scaled_dot_product_attention(qf, k_all, v_all, attn_mask=attn_mask)
        else:
            x_f = F.scaled_dot_product_attention(qf, k_all, v_all)

        x_f = x_f.transpose(1, 2).reshape(N, T_f, -1)
        if self.has_dropout:
            future = future + self.drop(self.out_f(x_f))
        else:
            future = future + self.out_f(x_f)

        # past self-attention
        x_p = F.scaled_dot_product_attention(qp, kp, vp).transpose(1, 2).reshape(N, T_p, -1)
        if self.has_dropout:
            past = past + self.drop(self.out_p(x_p))
        else:
            past = past + self.out_p(x_p)

        # MLP
        past   = past   + self.mlp_p(self.norm_p2(past)   * (1 + sp_m) + sh_m)
        future = future + self.mlp_f(self.norm_f2(future) * (1 + sf_m) + shf_m)

        return past, future


# ══════════════════════════════════════════════════════════════════════════════
#  PoseTrajFM — 3D trajectory + root-relative pose flow-matching model
# ══════════════════════════════════════════════════════════════════════════════

class PoseTrajFM(nn.Module):
    """
    Dual-input flow-matching model for joint trajectory + pose prediction.

    Inputs are encoded via separate projections for root trajectory (3D) and
    root-relative pose (J*3), then summed per timestep to form a single token
    sequence.  The transformer backbone is the same dual-stream (past/future)
    adaLN architecture.  Two output heads decode trajectory and pose velocities
    from the shared future embeddings.

    Flow matching operates jointly over the concatenated representation — a
    single ODE with shared timestep t.
    """

    def __init__(
        self,
        depth=6,
        embed_dim=192,
        coord_dim=3,
        num_joints=15,
        input_time=25,
        output_time=50,
        num_heads=4,
        mlp_ratio=2,
        dropout=0.1,
        causal_future=False,
        # spline params (None = disabled, backward compatible)
        n_ctrl_traj=None,
        n_ctrl_pose=None,
        spline_type="cubic",
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
        self.use_spline  = n_ctrl_traj is not None and n_ctrl_pose is not None

        # ── B-spline basis buffers (spline mode only) ────────────────────────
        if self.use_spline:
            B_t, B_t_pinv, anc_t = make_spline_basis(n_ctrl_traj, output_time, spline_type)
            B_p, B_p_pinv, anc_p = make_spline_basis(n_ctrl_pose,  output_time, spline_type)
            self.register_buffer("B_traj",      B_t)
            self.register_buffer("B_traj_pinv", B_t_pinv)
            self.register_buffer("B_pose",      B_p)
            self.register_buffer("B_pose_pinv", B_p_pinv)
            self.register_buffer("anchor_traj", anc_t)
            self.register_buffer("anchor_pose", anc_p)
            # learned modality type embedding: distinguishes traj vs pose ctrl-pt tokens
            self.modality_emb = nn.Embedding(2, embed_dim)

        # ── Past input encoding (two streams summed) ─────────────────────────
        self.scale_past_traj = nn.Linear(coord_dim, embed_dim, bias=False)
        # Structured per-joint encoding: project each joint independently, then aggregate
        self.joint_embed = nn.Linear(coord_dim, embed_dim // 4)
        self.joint_type_emb = nn.Embedding(num_joints, embed_dim // 4)
        # Aggregate per-joint features into a single token per timestep
        self.joint_agg = nn.Sequential(
            nn.Linear(num_joints * (embed_dim // 4), embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.pe_past = PositionalEncoding(embed_dim, max_len=input_time)

        # ── Future (noisy) encoding (two streams summed) ─────────────────────
        self.scale_fut_traj = nn.Linear(coord_dim, embed_dim, bias=False)
        # Same structured encoding for noisy future pose
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
                  dropout=dropout, causal_future=causal_future)
            for _ in range(depth)
        ])

        # ── Pose-only refinement blocks (separate decoder stream) ────────────
        self.pose_refine_blocks = nn.ModuleList([
            PoseRefineBlock(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(3)
        ])

        # ── Output heads ─────────────────────────────────────────────────────
        self.final_norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.final_mod  = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 2 * embed_dim))
        self.head_traj  = nn.Linear(embed_dim, coord_dim)
        # pose head reads from pose-refined stream
        self.pose_final_norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.pose_final_mod  = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 2 * embed_dim))
        # deeper pose head with residual connection
        self.head_pose_pre  = nn.Linear(embed_dim, embed_dim)
        self.head_pose_res  = nn.Sequential(
            nn.GELU(), nn.Linear(embed_dim, embed_dim), nn.GELU(),
        )
        self.head_pose_out  = nn.Linear(embed_dim, self.pose_dim)

        self.frame_drop_prob = FRAME_DROP_PROB

    def forward(self, past_traj, past_pose, x_t_traj, x_t_pose, t, cfg_uncond=False):
        """
        past_traj  : [N, T_in,  3]
        past_pose  : [N, T_in,  J*3]   root-relative
        x_t_traj   : [N, T_out|D_t, 3]     noisy future (full-res or ctrl-pt)
        x_t_pose   : [N, T_out|D_p, J*3]   noisy future (full-res or ctrl-pt)
        t          : [N]               flow-matching timestep

        Returns (vel_traj, vel_pose):
            vel_traj : [N, T_out|D_t, 3]
            vel_pose : [N, T_out|D_p, J*3]
        """
        N    = past_traj.shape[0]
        T_in = past_traj.shape[1]

        # ── structured per-joint encoding helper ─────────────────────────────
        joint_ids = torch.arange(self.num_joints, device=past_traj.device)
        jtype = self.joint_type_emb(joint_ids)  # [J, D//4]

        def encode_pose_structured(pose_flat, joint_proj, joint_agg_layer):
            # pose_flat: [N, T, J*3] → [N, T, J, 3]
            T = pose_flat.shape[1]
            p = pose_flat.reshape(N, T, self.num_joints, self.coord_dim)
            # per-joint: project coords + add joint type embedding
            j_feat = joint_proj(p) + jtype.unsqueeze(0).unsqueeze(0)  # [N, T, J, D//4]
            # aggregate all joints into single token
            j_flat = j_feat.reshape(N, T, -1)  # [N, T, J*(D//4)]
            return joint_agg_layer(j_flat)      # [N, T, D]

        # ── encode past ─────────────────────────────────────────────────────
        past_pose_enc = encode_pose_structured(past_pose, self.joint_embed, self.joint_agg)
        past_tok = self.pe_past(
            self.scale_past_traj(past_traj) + past_pose_enc
        )  # [N, T_in, D]

        # ── encode noisy future ──────────────────────────────────────────────
        if self.use_spline:
            # spline mode: traj [N, D_t, 3] and pose [N, D_p, J*3] have different token counts
            D_t = self.n_ctrl_traj
            D_p = self.n_ctrl_pose
            traj_emb = self.scale_fut_traj(x_t_traj)                          # [N, D_t, D]
            pose_emb = encode_pose_structured(x_t_pose, self.joint_embed_fut, self.joint_agg_fut)  # [N, D_p, D]
            # anchor-frame positional encoding (temporal position of each ctrl pt)
            pe_traj = self.pe_fut.pe[:, self.anchor_traj, :]                   # [1, D_t, D]
            pe_pose = self.pe_fut.pe[:, self.anchor_pose, :]                   # [1, D_p, D]
            # modality type embedding (0=traj, 1=pose)
            mod_traj = self.modality_emb(torch.zeros(D_t, dtype=torch.long, device=past_traj.device))
            mod_pose = self.modality_emb(torch.ones(D_p, dtype=torch.long, device=past_traj.device))
            traj_emb = traj_emb + pe_traj + mod_traj
            pose_emb = pose_emb + pe_pose + mod_pose
            # concatenate into single future sequence: [N, D_t + D_p, D]
            future = torch.cat([traj_emb, pose_emb], dim=1)
        else:
            # legacy mode: sum traj + pose per timestep, apply sequential PE
            fut_pose_enc = encode_pose_structured(x_t_pose, self.joint_embed_fut, self.joint_agg_fut)
            future = self.pe_fut(
                self.scale_fut_traj(x_t_traj) + fut_pose_enc
            )  # [N, T_out, D]

        past = past_tok

        # frame dropout during training (zero out early frames)
        if self.training and self.frame_drop_prob > 0:
            drop = int(torch.rand(1).item() * (T_in + 1) * self.frame_drop_prob)
            if drop > 0:
                past = past.clone()
                past[:, :drop] = 0.0

        # ── conditioning ─────────────────────────────────────────────────────
        t_emb = timestep_embedding(t, self.embed_dim)
        t_emb = self.time_mlp(t_emb).unsqueeze(1)               # [N, 1, D]
        past_ctx = past[:, :T_in, :].mean(dim=1, keepdim=True)  # [N, 1, D]
        # CFG: optionally zero out past context (unconditional mode)
        if cfg_uncond:
            past_ctx = torch.zeros_like(past_ctx)
        ctx   = t_emb + past_ctx
        cond  = self.cond_mlp(ctx)                               # [N, 1, 8*D]

        # ── transformer blocks ───────────────────────────────────────────────
        for block in self.blocks:
            past_fix, future = block(past, future, cond)

        # ── split back into traj / pose streams ──────────────────────────────
        if self.use_spline:
            future_traj_raw = future[:, :self.n_ctrl_traj]   # [N, D_t, D]
            future_pose_raw = future[:, self.n_ctrl_traj:]   # [N, D_p, D]
        else:
            future_traj_raw = future
            future_pose_raw = future

        # ── final norm + modulation (traj branch) ────────────────────────────
        shift, scale = self.final_mod(ctx).chunk(2, dim=-1)
        future_traj = self.final_norm(future_traj_raw) * (1 + scale) + shift

        # ── pose-only refinement stream ──────────────────────────────────────
        future_pose = future_pose_raw
        for prb in self.pose_refine_blocks:
            future_pose = prb(future_pose)
        shift_p, scale_p = self.pose_final_mod(ctx).chunk(2, dim=-1)
        future_pose = self.pose_final_norm(future_pose) * (1 + scale_p) + shift_p

        # ── decode velocity ──────────────────────────────────────────────────
        vel_traj = self.head_traj(future_traj)   # [N, D_t|T_out, 3]
        # deeper residual pose head (from pose-refined stream)
        h = self.head_pose_pre(future_pose)
        h = h + self.head_pose_res(h)
        vel_pose = self.head_pose_out(h)    # [N, D_p|T_out, J*3]

        return vel_traj, vel_pose

    # ── spline helper methods (no-ops when use_spline=False) ─────────────────

    def fit_ctrl_traj(self, traj):
        """[..., T_out, 3] → [..., D_t, 3]"""
        if not self.use_spline:
            return traj
        return torch.einsum('dt,...tc->...dc', self.B_traj_pinv, traj)

    def fit_ctrl_pose(self, pose):
        """[..., T_out, J*3] → [..., D_p, J*3]"""
        if not self.use_spline:
            return pose
        return torch.einsum('dt,...tc->...dc', self.B_pose_pinv, pose)

    def eval_spline_traj(self, ctrl):
        """[..., D_t, 3] → [..., T_out, 3]"""
        if not self.use_spline:
            return ctrl
        return torch.einsum('td,...dc->...tc', self.B_traj, ctrl)

    def eval_spline_pose(self, ctrl):
        """[..., D_p, J*3] → [..., T_out, J*3]"""
        if not self.use_spline:
            return ctrl
        return torch.einsum('td,...dc->...tc', self.B_pose, ctrl)


# ══════════════════════════════════════════════════════════════════════════════
#  Training utilities
# ══════════════════════════════════════════════════════════════════════════════

def sample_t(N, device):
    if TIMESTEP_MODE == "continuous" or TIMESTEP_MODE == "uniform":
        return torch.rand(N, device=device)
    if TIMESTEP_MODE == "logit_normal":
        return torch.sigmoid(torch.randn(N, device=device) * 1.5 - 0.5)
    raise ValueError(f"Unknown TIMESTEP_MODE: {TIMESTEP_MODE!r}")


def mirror_skeleton(traj, pose, N):
    """Swap left/right joints and negate the lateral axis (in-place on clones)."""
    mm = torch.tensor(MIRROR_MAP, device=traj.device)
    # reshape pose [N, T, J*3] -> [N, T, J, 3], reindex, negate lateral axis
    T = pose.shape[1]
    p = pose.reshape(N, T, NUM_JOINTS, COORD_DIM)
    p = p[:, :, mm, :].clone()
    p[:, :, :, MIRROR_AXIS] *= -1
    traj = traj.clone()
    traj[:, :, MIRROR_AXIS] *= -1
    return traj, p.reshape(N, T, POSE_DIM)


def train_step(model, batch, optimizer, device):
    model.train()
    optimizer.zero_grad()

    b = prepare_batch(batch, device)
    past_traj     = b["input_traj"]       # [N, T_in, 3]
    past_pose     = b["input_pose_rel"]   # [N, T_in, J*3]
    gt_traj       = b["output_traj"]      # [N, T_out, 3]
    gt_pose       = b["output_pose_rel"]  # [N, T_out, J*3]
    N = past_traj.shape[0]

    # on-the-fly skeleton mirroring: swap left/right joints for a subset of samples
    if MIRROR_PROB > 0:
        do_mirror = torch.rand(N, device=device) < MIRROR_PROB
        if do_mirror.any():
            idx = do_mirror.nonzero(as_tuple=True)[0]
            past_traj[idx], past_pose[idx] = mirror_skeleton(past_traj[idx], past_pose[idx], len(idx))
            gt_traj[idx], gt_pose[idx] = mirror_skeleton(gt_traj[idx], gt_pose[idx], len(idx))

    # input noise augmentation: small Gaussian noise on past observations
    if INPUT_NOISE_STD > 0 and model.training:
        past_traj = past_traj + torch.randn_like(past_traj) * INPUT_NOISE_STD
        past_pose = past_pose + torch.randn_like(past_pose) * INPUT_NOISE_STD

    # input joint dropout: randomly zero out joints in past pose (regularization)
    if INPUT_JOINT_DROP > 0 and model.training:
        jdrop_mask = torch.rand(N, 1, NUM_JOINTS, device=device) < INPUT_JOINT_DROP
        # never drop root joint (index 0)
        jdrop_mask[:, :, 0] = False
        jdrop_mask = jdrop_mask.unsqueeze(-1).expand(N, 1, NUM_JOINTS, COORD_DIM)
        pp = past_pose.reshape(N, INPUT_TIME, NUM_JOINTS, COORD_DIM)
        pp = pp.clone()
        pp[jdrop_mask.expand_as(pp)] = 0.0
        past_pose = pp.reshape(N, INPUT_TIME, POSE_DIM)

    t = sample_t(N, device)

    # CFG: randomly drop past conditioning during training
    cfg_uncond = False
    if CFG_DROP_PROB > 0 and model.training:
        cfg_uncond = torch.rand(1).item() < CFG_DROP_PROB

    # project GT to ctrl-pt space (no-op when spline disabled)
    gt_traj_lat = model.fit_ctrl_traj(gt_traj)   # [N, D_t|T_out, 3]
    gt_pose_lat = model.fit_ctrl_pose(gt_pose)    # [N, D_p|T_out, J*3]

    # noise in latent (ctrl-pt or full-res) space
    noise_traj = torch.randn_like(gt_traj_lat) * NOISE_SCALE
    noise_pose = torch.randn_like(gt_pose_lat) * NOISE_SCALE_POSE

    # OT coupling: reorder noise to minimize transport cost to data
    if USE_OT_COUPLING and N > 1:
        with torch.no_grad():
            # flatten both to vectors for cost computation
            x1_flat = torch.cat([gt_traj_lat.reshape(N, -1), gt_pose_lat.reshape(N, -1)], dim=-1)
            z0_flat = torch.cat([noise_traj.reshape(N, -1), noise_pose.reshape(N, -1)], dim=-1)
            # cost matrix: pairwise L2 distance between noise[i] and data[j]
            cost = torch.cdist(z0_flat, x1_flat, p=2)  # [N, N]
            # greedy nearest-neighbor assignment
            perm = torch.zeros(N, dtype=torch.long, device=device)
            used = torch.zeros(N, dtype=torch.bool, device=device)
            for i in range(N):
                costs_i = cost[i].clone()
                costs_i[used] = float('inf')
                j = costs_i.argmin()
                perm[i] = j
                used[j] = True
            # permute NOISE to align with data (data stays fixed, preserving past-future alignment)
            # perm[i] = j means noise[i] should be paired with data[j]
            # so we want noise_new[j] = noise_old[i], i.e. invert the permutation
            inv_perm = torch.zeros_like(perm)
            inv_perm[perm] = torch.arange(N, device=device)
            noise_traj = noise_traj[inv_perm]
            noise_pose = noise_pose[inv_perm]

    # interpolate
    t_view = t.view(N, 1, 1)
    x_t_traj = torch.lerp(noise_traj, gt_traj_lat, t_view)
    x_t_pose = torch.lerp(noise_pose, gt_pose_lat, t_view)

    # predict velocity in latent space
    pred_v_traj, pred_v_pose = model(past_traj, past_pose, x_t_traj, x_t_pose, t, cfg_uncond=cfg_uncond)

    # target velocity (conditional flow matching) in latent space
    target_v_traj = gt_traj_lat - noise_traj
    target_v_pose = gt_pose_lat - noise_pose

    # loss: map velocity residual through B to full-res for temporal weighting
    if model.use_spline:
        res_traj_full = model.eval_spline_traj(pred_v_traj - target_v_traj)  # [N, T_out, 3]
        err_traj = torch.norm(res_traj_full, dim=-1)                          # [N, T_out]
        res_pose_full = model.eval_spline_pose(pred_v_pose - target_v_pose)  # [N, T_out, J*3]
        pose_err_per_joint = res_pose_full.reshape(N, OUTPUT_TIME, NUM_JOINTS, COORD_DIM)
        err_pose = torch.norm(pose_err_per_joint, dim=-1).mean(dim=-1)        # [N, T_out]
    else:
        err_traj = torch.norm(pred_v_traj - target_v_traj, dim=-1)            # [N, T_out]
        pose_err_per_joint = (pred_v_pose - target_v_pose).reshape(N, OUTPUT_TIME, NUM_JOINTS, COORD_DIM)
        err_pose = torch.norm(pose_err_per_joint, dim=-1).mean(dim=-1)        # [N, T_out]

    # Emphasize errors near the end of the prediction horizon.
    w_traj = torch.exp(torch.linspace(0, 5.0, OUTPUT_TIME, device=device))
    w_traj = w_traj / w_traj.mean()
    # pose: milder weighting — intermediate frames matter for motion quality
    w_pose = torch.exp(torch.linspace(0, POSE_FDE_EXP, OUTPUT_TIME, device=device))
    w_pose = w_pose / w_pose.mean()

    loss_traj = (err_traj * w_traj.unsqueeze(0)).mean()
    loss_pose = (err_pose * w_pose.unsqueeze(0)).mean()

    # velocity regularization: penalize temporal jerkiness in full-res pose velocity
    vel_reg = torch.tensor(0.0, device=device)
    if VEL_REG_WEIGHT > 0:
        full_v_pose = model.eval_spline_pose(pred_v_pose) if model.use_spline else pred_v_pose
        pose_accel = full_v_pose[:, 2:] - 2 * full_v_pose[:, 1:-1] + full_v_pose[:, :-2]
        vel_reg = pose_accel.pow(2).mean()

    # auxiliary JPE-proxy loss: approximate x1, compute absolute joint error at last frame
    jpe_proxy_loss = torch.tensor(0.0, device=device)
    traj_fde_loss = torch.tensor(0.0, device=device)
    if JPE_PROXY_WEIGHT > 0 or TRAJ_FDE_WEIGHT > 0:
        # reconstruct approximate clean sample from ODE state
        x1_traj_lat = x_t_traj + pred_v_traj * (1 - t_view)
        x1_pose_lat = x_t_pose + pred_v_pose * (1 - t_view)
        # convert to full-res
        x1_traj_full = model.eval_spline_traj(x1_traj_lat) if model.use_spline else x1_traj_lat
        x1_pose_full = model.eval_spline_pose(x1_pose_lat) if model.use_spline else x1_pose_lat

        if JPE_PROXY_WEIGHT > 0:
            # absolute joint positions at last frame
            x1_abs_joints = x1_pose_full[:, -1].reshape(N, NUM_JOINTS, COORD_DIM) + x1_traj_full[:, -1:, :]
            gt_abs_joints = gt_pose[:, -1].reshape(N, NUM_JOINTS, COORD_DIM) + gt_traj[:, -1:, :]
            # per-sample JPE, weighted by t² (trust more at higher t)
            jpe_err = torch.norm(x1_abs_joints - gt_abs_joints, dim=-1).mean(dim=-1)  # [N]
            t_weight = t * t  # [N]
            jpe_proxy_loss = (jpe_err * t_weight).mean()

        # dedicated trajectory FDE loss: last-frame displacement error
        if TRAJ_FDE_WEIGHT > 0:
            fde_err = torch.norm(x1_traj_full[:, -1] - gt_traj[:, -1], dim=-1)  # [N]
            t_weight_fde = t * t  # [N]
            traj_fde_loss = (fde_err * t_weight_fde).mean()

    loss = (TRAJ_LOSS_WEIGHT * loss_traj + POSE_LOSS_WEIGHT * loss_pose +
            VEL_REG_WEIGHT * vel_reg +
            JPE_PROXY_WEIGHT * jpe_proxy_loss + TRAJ_FDE_WEIGHT * traj_fde_loss)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item(), loss_traj.item(), loss_pose.item()


# ══════════════════════════════════════════════════════════════════════════════
#  ODE sampler (Heun corrector)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def euler_sample_posetraj(model, past_traj, past_pose, K, device,
                          inference_steps, noise_scale, chunk_k=20):
    """
    Heun (2nd-order) ODE sampler for joint traj+pose.
    Runs in ctrl-pt space when spline enabled; always returns full-res output.

    Returns (pred_traj, pred_pose):
        pred_traj : [N, K, T_out, 3]
        pred_pose : [N, K, T_out, J*3]
    """
    if K > chunk_k:
        traj_chunks, pose_chunks = [], []
        for k_start in range(0, K, chunk_k):
            k_end = min(k_start + chunk_k, K)
            ct, cp = euler_sample_posetraj(
                model, past_traj, past_pose, k_end - k_start,
                device, inference_steps, noise_scale, chunk_k=chunk_k,
            )
            traj_chunks.append(ct)
            pose_chunks.append(cp)
        return torch.cat(traj_chunks, dim=1), torch.cat(pose_chunks, dim=1)

    N, T_in, _ = past_traj.shape

    # expand past for K samples
    pt_exp = past_traj.unsqueeze(1).expand(-1, K, -1, -1).reshape(N * K, T_in, COORD_DIM)
    pp_exp = past_pose.unsqueeze(1).expand(-1, K, -1, -1).reshape(N * K, T_in, POSE_DIM)

    # initial noise in latent space (ctrl-pt dims when spline enabled)
    T_lat_traj = model.n_ctrl_traj if model.use_spline else OUTPUT_TIME
    T_lat_pose = model.n_ctrl_pose if model.use_spline else OUTPUT_TIME
    z_traj = torch.randn(N * K, T_lat_traj, COORD_DIM, device=device) * noise_scale
    z_pose = torch.randn(N * K, T_lat_pose, POSE_DIM, device=device) * NOISE_SCALE_POSE

    timesteps = torch.linspace(0, 1, inference_steps + 1, device=device)

    model.eval()
    cfg_w = CFG_SCALE
    use_cfg = cfg_w != 1.0 and CFG_DROP_PROB > 0

    def guided_velocity(pt, pp, zt, zp, tv):
        """Compute velocity with optional CFG."""
        v_cond_t, v_cond_p = model(pt, pp, zt, zp, tv, cfg_uncond=False)
        if not use_cfg:
            return v_cond_t, v_cond_p
        v_uncond_t, v_uncond_p = model(pt, pp, zt, zp, tv, cfg_uncond=True)
        v_t = v_uncond_t + cfg_w * (v_cond_t - v_uncond_t)
        v_p = v_uncond_p + cfg_w * (v_cond_p - v_uncond_p)
        return v_t, v_p

    for i in range(inference_steps):
        t_val = timesteps[i].item()
        dt    = (timesteps[i + 1] - timesteps[i]).item()
        t_vec = torch.full((N * K,), t_val, device=device)

        # Heun corrector (2nd-order) with CFG
        v1_traj, v1_pose = guided_velocity(pt_exp, pp_exp, z_traj, z_pose, t_vec)
        z_traj_pred = z_traj + v1_traj * dt
        z_pose_pred = z_pose + v1_pose * dt

        t_next = (t_vec + dt).clamp(0, 1 - 1e-6)
        v2_traj, v2_pose = guided_velocity(pt_exp, pp_exp, z_traj_pred, z_pose_pred, t_next)

        v_avg_traj = (v1_traj + v2_traj) * 0.5
        v_avg_pose = (v1_pose + v2_pose) * 0.5

        z_traj = z_traj + dt * v_avg_traj
        z_pose = z_pose + dt * v_avg_pose

    # reshape to [N, K, T_lat, C]
    z_traj = z_traj.view(N, K, T_lat_traj, COORD_DIM)
    z_pose = z_pose.view(N, K, T_lat_pose, POSE_DIM)

    # evaluate spline back to full resolution (no-op when spline disabled)
    pred_traj = model.eval_spline_traj(z_traj)   # [N, K, T_out, 3]
    pred_pose = model.eval_spline_pose(z_pose)    # [N, K, T_out, J*3]
    return pred_traj, pred_pose


# ══════════════════════════════════════════════════════════════════════════════
#  Forward-pass benchmark
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def benchmark_forward_pass(model, train_loader, device, n_warmup=10, n_measure=100):
    """GPU timing for a single model() forward call using CUDA events.
    Uses train loader with multi-person augmentation (num_people=3, effective batch=192) — matches real training/inference conditions."""
    model.eval()
    batch = next(iter(train_loader))
    b     = prepare_batch(batch, device)

    T_lat_traj = model.n_ctrl_traj if model.use_spline else OUTPUT_TIME
    T_lat_pose = model.n_ctrl_pose if model.use_spline else OUTPUT_TIME
    N          = b["input_traj"].shape[0]

    # plain batch — no chunk expansion, matches one training forward pass
    pt = b["input_traj"]
    pp = b["input_pose_rel"]
    zt = torch.randn(N, T_lat_traj, COORD_DIM, device=device)
    zp = torch.randn(N, T_lat_pose, POSE_DIM,  device=device)
    tv = torch.full((N,), 0.5, device=device)

    # warm-up
    for _ in range(n_warmup):
        model(pt, pp, zt, zp, tv)
    torch.cuda.synchronize()

    # timed runs
    times_ms = []
    for _ in range(n_measure):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        model(pt, pp, zt, zp, tv)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    t = torch.tensor(times_ms)
    mean_per_sample_ms = t.mean().item() / N
    print(f"\n[benchmark]  batch={N}  (no chunk expansion — true forward pass latency)")
    print(f"[benchmark]  forward pass time:  mean={t.mean():.2f}ms  std={t.std():.2f}ms  median={t.median():.2f}ms")
    print(f"[benchmark]  per sample/forward pass:  {mean_per_sample_ms*1000:.3f}us")
    print(f"[benchmark]  ({n_warmup} warmup + {n_measure} measured calls)")


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def kmeans_select(trajectories, k, n_iter=30):
    """Select k medoids from K predictions via kmeans on last-frame features."""
    B, N, T, C = trajectories.shape
    feat = trajectories[:, :, -1].float()
    centers = feat[:, :k].clone()

    for _ in range(n_iter):
        dists = torch.cdist(feat, centers)
        assignments = dists.argmin(dim=-1)
        one_hot = torch.zeros(B, N, k, device=feat.device).scatter_(-1, assignments.unsqueeze(-1), 1.0)
        cluster_sums = torch.bmm(one_hot.transpose(1, 2), feat)
        counts = one_hot.sum(dim=1).unsqueeze(-1).clamp(min=1)
        new_centers = cluster_sums / counts
        empty = (one_hot.sum(dim=1) == 0).unsqueeze(-1)
        centers = torch.where(empty, centers, new_centers)

    medoid_idx = torch.cdist(centers, feat).argmin(dim=-1)
    idx_exp = medoid_idx.unsqueeze(-1).unsqueeze(-1).expand(B, k, T, C)
    return torch.gather(trajectories, 1, idx_exp)


def _weighted_kmeans_select(pred_traj_abs, pred_pose, k=6, traj_scale=5.0, n_iter=30):
    """Kmeans on last-frame features with trajectory dims scaled up.
    Returns medoid indices [N, k]."""
    frame_last = OUTPUT_TIME - 1
    N, K_total = pred_traj_abs.shape[:2]
    traj_feat = pred_traj_abs[:, :, frame_last] * traj_scale  # [N, K, 3]
    pose_feat = pred_pose[:, :, frame_last]                    # [N, K, J*3]
    feat = torch.cat([traj_feat, pose_feat], dim=-1)           # [N, K, 48]
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


def compute_npss(pred_joints, gt_joints):
    """Normalized Power Spectrum Similarity (DLow, ECCV 2020).
    pred_joints, gt_joints: [N, T, J, 3] — absolute joint positions.
    Returns scalar NPSS in [0, 1] (higher = smoother / more realistic).
    """
    # power spectrum via rfft along temporal axis
    pred_ps = torch.fft.rfft(pred_joints, dim=1).abs().pow(2)   # [N, F, J, 3]
    gt_ps   = torch.fft.rfft(gt_joints, dim=1).abs().pow(2)
    # flatten joints+coords: [N, F, J*3]
    pred_ps = pred_ps.reshape(pred_ps.shape[0], pred_ps.shape[1], -1)
    gt_ps   = gt_ps.reshape(gt_ps.shape[0], gt_ps.shape[1], -1)
    # normalize per frequency bin across joint dimensions
    pred_norm = pred_ps / pred_ps.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    gt_norm   = gt_ps / gt_ps.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    # cosine similarity per frequency bin, averaged over frequencies and samples
    cos_sim = F.cosine_similarity(pred_norm, gt_norm, dim=-1)  # [N, F]
    return float(cos_sim.mean().item())


def compute_apd(pred_j, half=25, pair_chunk=256):
    """Average Pairwise Distance over diverse proposals.
    pred_j: [N, K, T, J, 3] absolute positions (meters).
    Pairs processed in chunks of pair_chunk to avoid OOM for large K.
    Returns dict of APD in mm: all, 1s (frames 0:half), 2s (frames half:T).
    At 25fps half=25 corresponds to the 1-second mark."""
    N, K, T, J, _ = pred_j.shape
    idx_i, idx_j = torch.triu_indices(K, K, offset=1, device=pred_j.device)
    n_pairs = idx_i.shape[0]
    acc = torch.zeros(T, device=pred_j.device)
    count = 0
    for start in range(0, n_pairs, pair_chunk):
        end = min(start + pair_chunk, n_pairs)
        ci, cj = idx_i[start:end], idx_j[start:end]
        diff = pred_j[:, ci] - pred_j[:, cj]   # [N, chunk, T, J, 3]
        dist = diff.norm(dim=-1).mean(dim=-1)   # [N, chunk, T]
        acc += dist.sum(dim=[0, 1])             # [T]
        count += N * (end - start)
    acc /= count
    def _mean_window(s, e):
        return float(acc[s:e].mean().item() * 1000.0)
    return {
        "all": _mean_window(0, T),
        "1s":  _mean_window(0, half),
        "2s":  _mean_window(half, T),
    }


@torch.no_grad()
def evaluate(ema_model, val_loader, device, k_predictions, inference_steps,
             noise_scale, max_batches=None, kmeans_select_k=6, traj_scale=5.0,
             apd_seqs_target=256):
    """
    Compute APE, JPE, and NPSS over loader.
    When k_predictions > kmeans_select_k, uses weighted kmeans to select
    kmeans_select_k diverse samples from k_predictions proposals.
    All positional metrics in mm (×1000). NPSS in [0, 1].

    APD is computed on raw unselected proposals (K=10 and K=100) over the
    first apd_seqs_target sequences. k_predictions must be >= 100 for APD@100.
    """
    ema_model.eval()
    frame_last = OUTPUT_TIME - 1

    all_ape, all_jpe = [], []
    # NPSS accumulators
    all_npss_sum, all_npss_count = 0.0, 0
    # APD accumulators
    apd_k10_acc  = {"all": 0.0, "1s": 0.0, "2s": 0.0}
    apd_k100_acc = {"all": 0.0, "1s": 0.0, "2s": 0.0}
    apd_seqs_collected = 0
    # always sample 100 raw proposals at eval time for APD; accuracy metrics use kmeans→6
    k_sample = max(k_predictions, 100)

    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break

        b = prepare_batch(batch, device)
        past_traj  = b["input_traj"]
        past_pose  = b["input_pose_rel"]
        gt_traj    = b["output_traj"]
        gt_pose    = b["output_pose_rel"]
        offset     = b["offset_traj"]

        # sample k_sample raw i.i.d. proposals: [N, k_sample, T_out, 3/J*3]
        pred_traj, pred_pose = euler_sample_posetraj(
            ema_model, past_traj, past_pose,
            K=k_sample, device=device,
            inference_steps=inference_steps, noise_scale=noise_scale,
        )

        # absolute trajectory coordinates
        gt_traj_abs    = gt_traj + offset                          # [N, T_out, 3]
        pred_traj_abs  = pred_traj + offset.unsqueeze(1)           # [N, K, T_out, 3]

        N = gt_traj.shape[0]
        K = pred_traj_abs.shape[1]

        # ── APD: raw proposals, up to apd_seqs_target sequences ──
        if apd_seqs_collected < apd_seqs_target:
            n_take = min(N, apd_seqs_target - apd_seqs_collected)
            # build absolute joint positions [n_take, K, T, J, 3]
            pp_raw = pred_pose[:n_take].reshape(n_take, K, OUTPUT_TIME, NUM_JOINTS, COORD_DIM)
            pt_raw = pred_traj_abs[:n_take].unsqueeze(-2)           # [n_take, K, T, 1, 3]
            pj_raw = pp_raw + pt_raw                                 # [n_take, K, T, J, 3]
            # APD @ K=10 (first 10 i.i.d. draws)
            d10 = compute_apd(pj_raw[:, :10])
            for key in apd_k10_acc:
                apd_k10_acc[key] += d10[key] * n_take
            # APD @ K=100 (all 100 i.i.d. draws)
            d100 = compute_apd(pj_raw[:, :100])
            for key in apd_k100_acc:
                apd_k100_acc[key] += d100[key] * n_take
            apd_seqs_collected += n_take

        # weighted kmeans selection: K proposals → kmeans_select_k selected
        if K > kmeans_select_k:
            sel_idx = _weighted_kmeans_select(pred_traj_abs, pred_pose,
                                              k=kmeans_select_k, traj_scale=traj_scale)
            idx_traj = sel_idx.unsqueeze(-1).unsqueeze(-1).expand(N, kmeans_select_k, OUTPUT_TIME, COORD_DIM)
            idx_pose = sel_idx.unsqueeze(-1).unsqueeze(-1).expand(N, kmeans_select_k, OUTPUT_TIME, POSE_DIM)
            pred_traj_abs = torch.gather(pred_traj_abs, 1, idx_traj)
            pred_pose = torch.gather(pred_pose, 1, idx_pose)
            K = kmeans_select_k

        # APE — root-aligned pose error at last frame (best-of-K)
        pred_pose_last = pred_pose[:, :, frame_last].reshape(N, K, NUM_JOINTS, COORD_DIM)
        gt_pose_last   = gt_pose[:, frame_last].reshape(N, NUM_JOINTS, COORD_DIM)
        # root-align: subtract joint 0
        pred_aligned = pred_pose_last - pred_pose_last[:, :, 0:1, :]
        gt_aligned   = gt_pose_last - gt_pose_last[:, 0:1, :]
        ape_per_mode = torch.norm(pred_aligned - gt_aligned.unsqueeze(1), dim=-1).mean(dim=-1)  # [N, K]
        min_ape = ape_per_mode.min(dim=1).values
        all_ape.extend(min_ape.cpu().tolist())

        # JPE — absolute joint error at last frame (best-of-K), also identify best mode
        gt_pose_full   = gt_pose.reshape(N, OUTPUT_TIME, NUM_JOINTS, COORD_DIM)
        pred_abs_j = pred_pose_last + pred_traj_abs[:, :, frame_last].unsqueeze(2)  # [N, K, J, 3]
        gt_abs_j   = gt_pose_last + gt_traj_abs[:, frame_last].unsqueeze(1)   # [N, J, 3]
        jpe_per_mode = torch.norm(pred_abs_j - gt_abs_j.unsqueeze(1), dim=-1).mean(dim=-1)  # [N, K]
        min_jpe, best_k = jpe_per_mode.min(dim=1)
        all_jpe.extend(min_jpe.cpu().tolist())

        # NPSS — on best-of-K selected prediction (absolute joint positions, all frames)
        best_k_exp = best_k.view(N, 1, 1, 1).expand(N, 1, OUTPUT_TIME, NUM_JOINTS * COORD_DIM)
        best_pose = torch.gather(pred_pose, 1, best_k_exp).squeeze(1)  # [N, T, J*3]
        best_k_traj = best_k.view(N, 1, 1, 1).expand(N, 1, OUTPUT_TIME, COORD_DIM)
        best_traj = torch.gather(pred_traj_abs, 1, best_k_traj).squeeze(1)  # [N, T, 3]
        # absolute joint positions: [N, T, J, 3]
        best_abs = best_pose.reshape(N, OUTPUT_TIME, NUM_JOINTS, COORD_DIM) + best_traj.unsqueeze(2)
        gt_abs_full = gt_pose_full + gt_traj_abs.unsqueeze(2)
        npss_val = compute_npss(best_abs, gt_abs_full)
        all_npss_sum += npss_val * N
        all_npss_count += N

    mean_ape = sum(all_ape) / max(len(all_ape), 1)
    mean_jpe = sum(all_jpe) / max(len(all_jpe), 1)
    mean_npss = all_npss_sum / max(all_npss_count, 1)

    n_apd = max(apd_seqs_collected, 1)
    apd_k10  = {k: v / n_apd for k, v in apd_k10_acc.items()}
    apd_k100 = {k: v / n_apd for k, v in apd_k100_acc.items()}

    return {
        "APE":    float(mean_ape * 1000.0),
        "JPE":    float(mean_jpe * 1000.0),
        "NPSS":   float(mean_npss),
        "APD_k10_all": apd_k10["all"],
        "APD_k10_1s":  apd_k10["1s"],
        "APD_k10_2s":  apd_k10["2s"],
        "APD_k100_all": apd_k100["all"],
        "APD_k100_1s":  apd_k100["1s"],
        "APD_k100_2s":  apd_k100["2s"],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    global DATASET_NAME, DATA_ROOT, RESULTS_DIR, TRAIN_STEPS, BATCH_SIZE
    global NUM_JOINTS, POSE_DIM, INPUT_TIME, OUTPUT_TIME, MIRROR_MAP

    parser = argparse.ArgumentParser(description="Train FlowMoP on a motion benchmark.")
    parser.add_argument("--dataset", choices=["3dpw", "mocap_umpm"], default=DATASET_NAME)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default=RESULTS_DIR)
    parser.add_argument("--train-steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark", action="store_true", help="Measure forward pass latency and exit")
    args = parser.parse_args()

    DATASET_NAME = args.dataset
    DATA_ROOT = args.data_dir
    RESULTS_DIR = args.output_dir
    TRAIN_STEPS = args.train_steps
    BATCH_SIZE = args.batch_size
    if DATASET_NAME == "mocap_umpm":
        NUM_JOINTS, INPUT_TIME, OUTPUT_TIME = 15, 25, 50
        MIRROR_MAP = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 13, 14, 12, 10, 11]
    else:
        NUM_JOINTS, INPUT_TIME, OUTPUT_TIME = 13, 10, 20
        MIRROR_MAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]
    POSE_DIM = NUM_JOINTS * COORD_DIM

    device = args.device
    if device == "cuda":
        torch.backends.cudnn.benchmark        = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
    torch.manual_seed(args.seed)

    model     = PoseTrajFM(
        depth=DEPTH, embed_dim=EMBED_DIM, coord_dim=COORD_DIM,
        num_joints=NUM_JOINTS, input_time=INPUT_TIME, output_time=OUTPUT_TIME,
        num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO, dropout=DROPOUT,
        n_ctrl_traj=N_CTRL_TRAJ if USE_SPLINE else None,
        n_ctrl_pose=N_CTRL_POSE if USE_SPLINE else None,
        spline_type=SPLINE_TYPE,
    ).to(device)
    ema_model = copy.deepcopy(model)
    ema_decay = 0.9995

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nFlowMoP params={n_params:,}  device={device}")
    print(f"dataset={DATASET_NAME}  data={DATA_ROOT}  J={NUM_JOINTS}  T_in={INPUT_TIME}  T_out={OUTPUT_TIME}")

    if args.benchmark:
        train_loader, _ = get_dataloaders()
        benchmark_forward_pass(ema_model, train_loader, device)
        return
    if model.use_spline:
        print(f"spline: D_t={N_CTRL_TRAJ}, D_p={N_CTRL_POSE}, type={SPLINE_TYPE}, "
              f"future_tokens={N_CTRL_TRAJ + N_CTRL_POSE} (was {OUTPUT_TIME})")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
    warmup_steps = min(200, TRAIN_STEPS // 10)
    # Cosine schedule with six warm restart cycles.
    n_restarts = 6
    cycle_len = TRAIN_STEPS // n_restarts
    def lr_lambda_restart(s):
        if s < warmup_steps:
            return s / warmup_steps
        # which cycle are we in?
        cycle_pos = (s - warmup_steps) % cycle_len
        cycle_total = cycle_len
        progress = cycle_pos / max(cycle_total, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_restart)
    train_loader, val_loader = get_dataloaders()
    steps_per_epoch = max(len(train_loader), 1)
    log_every_steps = max(1, LOG_EVERY * steps_per_epoch)

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
    # 1/4 of train batches for eval
    train_eval_batches = min(10, max(1, len(train_loader) // 4))

    print(f"logging every {LOG_EVERY} epochs (~{log_every_steps} steps)")
    print(f"eval: best-of-{EVAL_K}, train={train_eval_batches} batches")

    # ── helpers ──────────────────────────────────────────────────────────────

    def infinite(loader):
        while True:
            yield from loader

    @torch.no_grad()
    def log_metrics(step, final=False):
        """Evaluate on 1/4 train and full val (APE/JPE only), log to JSON."""
        eval_t0 = time.time()

        # 1/4 of training set — weighted kmeans 100→6
        train_m = evaluate(
            ema_model, train_loader, device,
            k_predictions=100, inference_steps=INFERENCE_STEPS,
            noise_scale=NOISE_SCALE,
            max_batches=train_eval_batches,
            kmeans_select_k=EVAL_K, traj_scale=5.0,
        )
        # full val set — weighted kmeans 100→6
        val_m = evaluate(
            ema_model, val_loader, device,
            k_predictions=100, inference_steps=INFERENCE_STEPS,
            noise_scale=NOISE_SCALE,
            kmeans_select_k=EVAL_K, traj_scale=5.0,
        )

        epoch_key = str(max(1, int(round(step / steps_per_epoch))))
        train_metrics_by_epoch[epoch_key] = train_m
        val_metrics_by_epoch[epoch_key]   = val_m

        with open(train_log_path, "w") as f:
            json.dump(train_metrics_by_epoch, f, indent=2)
        with open(val_log_path, "w") as f:
            json.dump(val_metrics_by_epoch, f, indent=2)

        eval_mins = (time.time() - eval_t0) / 60
        print(
            f"  step {step:>5}/{TRAIN_STEPS}  epoch={epoch_key}  ({eval_mins:.1f}min eval)\n"
            f"    train: APE={train_m['APE']:.1f}  JPE={train_m['JPE']:.1f} mm  NPSS={train_m['NPSS']:.4f}\n"
            f"    val:   APE={val_m['APE']:.1f}  JPE={val_m['JPE']:.1f} mm  NPSS={val_m['NPSS']:.4f}"
        )

        nonlocal best_val_jpe
        if val_m["JPE"] < best_val_jpe:
            best_val_jpe = val_m["JPE"]
            ckpt_path = os.path.join(ckpt_dir, "best.pt")
            torch.save(ema_model.state_dict(), ckpt_path)
            print(f"  [ckpt] saved best EMA (valJPE={val_m['JPE']:.1f} mm) -> {ckpt_path}")

    # ── train loop ───────────────────────────────────────────────────────────

    train_iter = infinite(train_loader)
    t0 = time.time()
    recent_loss, recent_traj, recent_pose = [], [], []
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

        # periodic detailed logging (skip step 0 — untrained model eval is wasteful)
        if step > 0 and step % log_every_steps == 0:
            log_metrics(step)

    # ensure final epoch is logged
    final_key = str(max(1, int(round(TRAIN_STEPS / steps_per_epoch))))
    if final_key not in val_metrics_by_epoch:
        log_metrics(TRAIN_STEPS, final=True)

    # Use the last logged validation metrics to avoid a redundant evaluation pass.
    last_epoch = max(val_metrics_by_epoch.keys(), key=int)
    metrics = val_metrics_by_epoch[last_epoch]

    elapsed = (time.time() - t0) / 60
    print(f"\ndone in {elapsed:.1f} min")
    print(
        f"  APE    = {metrics['APE']:.1f} mm\n"
        f"  JPE    = {metrics['JPE']:.1f} mm\n"
        f"  NPSS   = {metrics['NPSS']:.4f}"
    )

    # save final checkpoint
    final_ckpt = os.path.join(ckpt_dir, "final.pt")
    torch.save(ema_model.state_dict(), final_ckpt)
    print(f"  [ckpt] saved final EMA -> {final_ckpt}")



if __name__ == "__main__":
    main()

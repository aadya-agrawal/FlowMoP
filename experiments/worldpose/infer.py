from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train import (
    PoseTrajFM,
    euler_sample_posetraj,
    COORD_DIM,
    NOISE_SCALE,
    USE_AGENT_NEIGHBORS,
    K_NEIGHBORS,
)
from preprocess import ProcessingConfig, WorldPoseProcessor


def make_blend_weights(horizon: int, mode: str = "triangular") -> np.ndarray:
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    if mode == "uniform":
        return np.ones(horizon, dtype=np.float32)
    if mode == "hann":
        if horizon == 1:
            return np.ones(1, dtype=np.float32)
        return np.clip(np.hanning(horizon).astype(np.float32), 1e-6, None)

    center = (horizon - 1) / 2.0
    positions = np.arange(horizon, dtype=np.float32)
    weights = 1.0 - np.abs(positions - center) / max(center, 1.0)
    return np.clip(weights, 1e-6, None).astype(np.float32)


def _forward_fill_nan(sequence: np.ndarray) -> np.ndarray:
    output = sequence.copy()
    flat = output.reshape(output.shape[0], -1)
    last_valid = None
    for frame_index, row in enumerate(flat):
        missing = np.isnan(row)
        if missing.any():
            row[missing] = 0.0 if last_valid is None else last_valid[missing]
            flat[frame_index] = row
        else:
            last_valid = row.copy()
    return flat.reshape(output.shape)


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers — decompose / recompose joints ↔ root traj + root-relative pose
# ══════════════════════════════════════════════════════════════════════════════

def decompose_joints(joints: np.ndarray):
    """
    joints: (N, T, J, 3) → root_traj (N, T, 3), pose_rel (N, T, J*3)
    root_traj is joint-0 position; pose_rel is root-relative per-joint offsets.
    """
    root_traj = joints[:, :, 0, :]                           # (N, T, 3)
    pose_rel = (joints - joints[:, :, 0:1, :])               # (N, T, J, 3)
    J = joints.shape[2]
    pose_rel = pose_rel.reshape(joints.shape[0], joints.shape[1], J * 3)
    return root_traj, pose_rel


def recompose_joints(root_traj: np.ndarray, pose_rel: np.ndarray, num_joints: int):
    """
    root_traj: (..., T, 3), pose_rel: (..., T, J*3) → joints (..., T, J, 3)
    """
    pose = pose_rel.reshape(*pose_rel.shape[:-1], num_joints, 3)
    joints = pose + root_traj[..., None, :]                   # broadcast root to all joints
    return joints


def extract_neighbors(root_traj_all: np.ndarray, valid_people: np.ndarray, k_neighbors: int):
    """
    For each person, find K nearest neighbors by last-observed-frame distance.

    root_traj_all: (N, T, 3)  — agent-centric root trajectories
    valid_people:  (N,) bool

    Returns:
        neighbor_traj: (N, K, T, 3) agent-centric neighbor root trajectories
        neighbor_mask: (N, K) float32 validity mask
    """
    N, T, _ = root_traj_all.shape
    K = min(k_neighbors, N - 1)

    # last-frame positions for distance computation
    last_pos = root_traj_all[:, -1, :]                         # (N, 3)
    dists = np.linalg.norm(last_pos[:, None, :] - last_pos[None, :, :], axis=-1)  # (N, N)

    # mask out invalid people and self
    dists[~valid_people, :] = np.inf
    dists[:, ~valid_people] = np.inf
    np.fill_diagonal(dists, np.inf)

    nn_idx = np.argsort(dists, axis=-1)[:, :K]                # (N, K)

    neighbor_traj = np.zeros((N, K, T, 3), dtype=np.float32)
    neighbor_mask = np.zeros((N, K), dtype=np.float32)
    for i in range(N):
        if not valid_people[i]:
            continue
        # agent-centric: subtract focal agent's last-frame position
        offset_i = root_traj_all[i, -1:, :]                   # (1, 3)
        for ki in range(K):
            j = nn_idx[i, ki]
            if valid_people[j]:
                # focal-centric coordinates
                neighbor_traj[i, ki] = root_traj_all[j] - offset_i
                neighbor_mask[i, ki] = 1.0

    return neighbor_traj, neighbor_mask


# ══════════════════════════════════════════════════════════════════════════════
#  FM-based prediction wrapper — replaces model_predict_numpy for the FM model
# ══════════════════════════════════════════════════════════════════════════════

def fm_predict(
    model: PoseTrajFM,
    past_joints: np.ndarray,
    valid_people: np.ndarray,
    device: torch.device,
    k_samples: int,
    inference_steps: int,
    noise_scale: float,
    use_neighbors: bool,
    k_neighbors: int = K_NEIGHBORS,
):
    """
    past_joints: (N, T_past, J, 3)
    Returns: pred_joints (K, N, T_future, J, 3) in same coordinate frame as input.
    """
    N, T_p, J, C = past_joints.shape

    # decompose into root traj + root-relative pose
    root_traj, pose_rel = decompose_joints(past_joints)

    # agent-centric offset (last observed root position)
    offset = root_traj[:, -1:, :]                              # (N, 1, 3)
    root_traj_centered = root_traj - offset                    # (N, T_p, 3)

    past_traj_t = torch.from_numpy(root_traj_centered).to(device, dtype=torch.float32)
    past_pose_t = torch.from_numpy(pose_rel).to(device, dtype=torch.float32)

    # neighbor conditioning
    neighbors_t, nb_mask_t = None, None
    if use_neighbors:
        nb_traj, nb_mask = extract_neighbors(root_traj, valid_people, k_neighbors)
        # convert neighbor positions to focal-agent-centric coords
        nb_traj_centered = nb_traj - offset[:, None, :, :]     # (N, K, T, 3)
        neighbors_t = torch.from_numpy(nb_traj_centered).to(device, dtype=torch.float32)
        nb_mask_t = torch.from_numpy(nb_mask).to(device, dtype=torch.float32)

    # ODE sample — returns (N, K, T_out, 3), (N, K, T_out, J*3), optional joints
    pred_traj, pred_pose, _pred_joints = euler_sample_posetraj(
        model, past_traj_t, past_pose_t,
        K=k_samples, device=device,
        inference_steps=inference_steps,
        noise_scale=noise_scale,
        neighbors=neighbors_t,
        neighbor_mask=nb_mask_t,
    )

    # back to numpy
    pred_traj_np = pred_traj.cpu().numpy()                     # (N, K, T_out, 3)
    pred_pose_np = pred_pose.cpu().numpy()                     # (N, K, T_out, J*3)

    # add offset back to get absolute root trajectory
    pred_traj_abs = pred_traj_np + offset[:, None, :, :]       # (N, K, T_out, 3)

    # recompose to absolute joint positions
    pred_joints = recompose_joints(pred_traj_abs, pred_pose_np, J)  # (N, K, T_out, J, 3)

    # rearrange to (K, N, T_out, J, 3)
    pred_joints = np.transpose(pred_joints, (1, 0, 2, 3, 4))
    return pred_joints


# ══════════════════════════════════════════════════════════════════════════════
#  Main inference loop
# ══════════════════════════════════════════════════════════════════════════════

def run_sequence_inference_fm(
    sequence_path: Path,
    checkpoint_path: Path,
    metadata_path: Path,
    output_path: Path,
    smpl_model_path: Path,
    smpl_gender: str,
    stride: int,
    blend: str,
    k_samples: int,
    inference_steps: int,
    noise_scale: float,
    use_neighbors: bool,
    k_neighbors: int,
    device: str,
    # FM model architecture params
    depth: int,
    embed_dim: int,
    num_heads: int,
    mlp_ratio: int,
    dropout: float,
    n_ctrl_traj: int | None,
    n_ctrl_pose: int | None,
    spline_type: str,
):
    # --- Load metadata ---
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    cfg_meta = metadata["config"]
    past_frames = int(cfg_meta.get("past_frames", 25))
    future_frames = int(cfg_meta.get("future_frames", 75))
    num_joints = int(cfg_meta.get("num_joints", 24))

    if stride <= 0:
        raise ValueError("stride must be > 0")

    # --- Build processor ---
    proc_cfg = ProcessingConfig(
        data_dir=sequence_path.parent,
        output_dir=output_path.parent,
        smpl_model_path=str(smpl_model_path),
        smpl_gender=smpl_gender,
        past_frames=past_frames,
        future_frames=future_frames,
        num_joints=num_joints,
        normalize_to_field=False,
    )
    processor = WorldPoseProcessor(proc_cfg)

    smpl_data = processor.load_sequence(sequence_path)
    features = processor.extract_pose_features(smpl_data)
    joints_field_centered = features["joints_3d"].astype(np.float32)  # (N, T, J, 3)

    valid_people = ~np.isnan(smpl_data["global_orient"]).all(axis=(1, 2))  # (N,)

    # forward-fill NaNs
    for person_idx in range(joints_field_centered.shape[0]):
        joints_field_centered[person_idx] = _forward_fill_nan(joints_field_centered[person_idx])

    # --- Model (auto-detect architecture from checkpoint) ---
    dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt

    # infer spline ctrl-pt counts and output_time from checkpoint buffers
    if "B_traj" in sd:
        ckpt_output_time = sd["B_traj"].shape[0]
        ckpt_n_ctrl_traj = sd["B_traj"].shape[1]
        ckpt_n_ctrl_pose = sd["B_pose"].shape[1]
    else:
        ckpt_output_time = future_frames
        ckpt_n_ctrl_traj = None
        ckpt_n_ctrl_pose = None

    # infer embed_dim from a weight
    ckpt_embed_dim = sd["scale_past_traj.weight"].shape[0]
    # infer depth from number of transformer blocks
    ckpt_depth = max(int(k.split(".")[1]) for k in sd if k.startswith("blocks.")) + 1
    # detect QK norm from checkpoint keys
    ckpt_use_qk_norm = "blocks.0.norm_q_p.gamma" in sd
    # infer num_heads: from per-head RMSNorm gamma when available, else use CLI arg
    if ckpt_use_qk_norm:
        ckpt_num_heads = sd["blocks.0.norm_q_p.gamma"].shape[0]
    else:
        ckpt_num_heads = num_heads
    # infer whether neighbors are used
    ckpt_use_neighbors = "neighbor_type_emb.weight" in sd
    # infer whether joint head is used and its ctrl-pt count
    ckpt_use_joint_head = "B_joint" in sd
    ckpt_n_ctrl_joint = int(sd["B_joint"].shape[1]) if ckpt_use_joint_head else None

    # override future_frames to match checkpoint
    if ckpt_output_time != future_frames:
        print(f"  [auto-detect] overriding future_frames {future_frames} → {ckpt_output_time} (from checkpoint)")
        future_frames = ckpt_output_time

    print(f"  [auto-detect] depth={ckpt_depth}, embed_dim={ckpt_embed_dim}, heads={ckpt_num_heads}, "
          f"output_time={ckpt_output_time}, n_ctrl_traj={ckpt_n_ctrl_traj}, n_ctrl_pose={ckpt_n_ctrl_pose}, "
          f"n_ctrl_joint={ckpt_n_ctrl_joint}, neighbors={ckpt_use_neighbors}, "
          f"qk_norm={ckpt_use_qk_norm}, joint_head={ckpt_use_joint_head}")

    model = PoseTrajFM(
        depth=ckpt_depth,
        embed_dim=ckpt_embed_dim,
        coord_dim=COORD_DIM,
        num_joints=num_joints,
        input_time=past_frames,
        output_time=future_frames,
        num_heads=ckpt_num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        n_ctrl_traj=ckpt_n_ctrl_traj,
        n_ctrl_pose=ckpt_n_ctrl_pose,
        spline_type=spline_type,
        use_agent_neighbors=ckpt_use_neighbors,
        use_qk_norm=ckpt_use_qk_norm,
        use_joint_head=ckpt_use_joint_head,
        n_ctrl_joint=ckpt_n_ctrl_joint,
    ).to(dev)

    model.load_state_dict(sd)
    model.eval()
    print(f"Loaded FM model from {checkpoint_path}  (params={sum(p.numel() for p in model.parameters()):,})")

    # --- Sliding-window inference ---
    n_people, total_frames, _, _ = joints_field_centered.shape
    window = past_frames + future_frames
    if total_frames < window:
        raise ValueError(
            f"Sequence too short: T={total_frames}, but needs at least {window} frames (past+future)."
        )

    starts = list(range(0, total_frames - window + 1, stride))
    if starts[-1] != (total_frames - window):
        starts.append(total_frames - window)

    blend_w = make_blend_weights(future_frames, mode=blend)

    sum_pred = np.zeros((k_samples, n_people, total_frames, num_joints, 3), dtype=np.float32)
    sum_w = np.zeros((n_people, total_frames), dtype=np.float32)

    with torch.no_grad():
        for wi, start in enumerate(starts):
            past_np = joints_field_centered[:, start : start + past_frames]  # (N, T_p, J, 3)

            # predict K modes → (K, N, T_f, J, 3)
            preds = fm_predict(
                model, past_np, valid_people, dev,
                k_samples=k_samples,
                inference_steps=inference_steps,
                noise_scale=noise_scale,
                use_neighbors=ckpt_use_neighbors,
                k_neighbors=k_neighbors,
            )

            for h in range(future_frames):
                g = start + past_frames + h
                w = float(blend_w[h])
                valid = valid_people.astype(np.float32)[:, None, None]  # (N,1,1)
                sum_pred[:, :, g] += preds[:, :, h] * (w * valid)[None, :, :, :]
                sum_w[:, g] += w * valid_people.astype(np.float32)

            print(f"  window {wi+1}/{len(starts)}: frames [{start}, {start+window})")

    # finalize weighted average
    pred_modes = np.full_like(sum_pred, np.nan, dtype=np.float32)
    for n in range(n_people):
        good = sum_w[n] > 0
        if np.any(good):
            pred_modes[:, n, good] = sum_pred[:, n, good] / sum_w[n, good][None, :, None, None]

    gt_world = joints_field_centered

    # --- Save ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        sequence_name=sequence_path.stem,
        pred_modes_world=pred_modes,         # (K, N, T, J, 3)
        gt_joints_world=gt_world,            # (N, T, J, 3)
        valid_people=valid_people,            # (N,)
        starts=np.array(starts, dtype=np.int32),
        stride=np.int32(stride),
        past_frames=np.int32(past_frames),
        future_frames=np.int32(future_frames),
        inference_mode=np.array("sliding"),
        k_samples=np.int32(k_samples),
        inference_steps=np.int32(inference_steps),
        blend_mode=np.array(blend),
    )

    print(f"Saved predictions to: {output_path}")
    print(f"  sequence: {sequence_path.stem}")
    print(f"  pred shape (K,N,T,J,3): {pred_modes.shape}")
    print(f"  gt shape   (N,T,J,3):   {gt_world.shape}")
    print(f"  valid people: {int(valid_people.sum())}/{len(valid_people)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sliding-window inference for the flow-matching PoseTrajFM model."
    )
    parser.add_argument("--sequence", dest="sequence_path", type=Path, required=True)
    parser.add_argument("--checkpoint", dest="checkpoint_path", type=Path, required=True)
    parser.add_argument("--metadata", dest="metadata_path", type=Path, required=True)
    parser.add_argument("--output", dest="output_path", type=Path, required=True)
    parser.add_argument("--smpl-model-dir", dest="smpl_model_path", type=Path, required=True)
    parser.add_argument("--smpl-gender", type=str, default="neutral",
                        choices=["neutral", "male", "female"])
    parser.add_argument("--stride", type=int, default=75)
    parser.add_argument("--blend", type=str, default="triangular",
                        choices=["triangular", "hann", "uniform"])
    parser.add_argument("--k-samples", type=int, default=10,
                        help="Number of ODE samples per window (default: 10).")
    parser.add_argument("--inference-steps", type=int, default=50,
                        help="Euler ODE integration steps (default: 10).")
    parser.add_argument("--noise-scale", type=float, default=NOISE_SCALE,
                        help="Initial trajectory noise scale.")
    parser.add_argument("--use-neighbors", action="store_true", default=USE_AGENT_NEIGHBORS,
                        help="Enable agent-neighbor conditioning.")
    parser.add_argument("--no-neighbors", dest="use_neighbors", action="store_false")
    parser.add_argument("--k-neighbors", type=int, default=K_NEIGHBORS,
                        help="Number of neighbor agents to condition on.")
    parser.add_argument("--device", type=str, default="cuda")

    # FM architecture (must match checkpoint)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--mlp_ratio", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--n_ctrl_traj", type=int, default=25,
                        help="Spline ctrl points for trajectory (None to disable spline).")
    parser.add_argument("--n_ctrl_pose", type=int, default=30,
                        help="Spline ctrl points for pose (None to disable spline).")
    parser.add_argument("--spline_type", type=str, default="cubic",
                        choices=["cubic", "linear"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sequence_inference_fm(
        sequence_path=args.sequence_path,
        checkpoint_path=args.checkpoint_path,
        metadata_path=args.metadata_path,
        output_path=args.output_path,
        smpl_model_path=args.smpl_model_path,
        smpl_gender=args.smpl_gender,
        stride=args.stride,
        blend=args.blend,
        k_samples=args.k_samples,
        inference_steps=args.inference_steps,
        noise_scale=args.noise_scale,
        use_neighbors=args.use_neighbors,        k_neighbors=args.k_neighbors,        device=args.device,
        depth=args.depth,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        n_ctrl_traj=args.n_ctrl_traj,
        n_ctrl_pose=args.n_ctrl_pose,
        spline_type=args.spline_type,
    )

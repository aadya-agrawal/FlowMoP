"""Test different selection strategies from 100 proposals → 6 selected."""
import torch
import time
import argparse
from pathlib import Path
import train as training

from train import (
    PoseTrajFM, get_dataloaders, prepare_batch,
    euler_sample_posetraj, kmeans_select,
    DEPTH, EMBED_DIM, NUM_HEADS, MLP_RATIO, DROPOUT,
    NUM_JOINTS, COORD_DIM, POSE_DIM, INPUT_TIME, OUTPUT_TIME,
    NOISE_SCALE, NOISE_SCALE_POSE, INFERENCE_STEPS,
    USE_SPLINE, N_CTRL_TRAJ, N_CTRL_POSE, SPLINE_TYPE,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = None
SAVE_PATH = None
K_PROPOSALS = 100
K_SELECT = 6


def load_model():
    model = PoseTrajFM(
        depth=DEPTH, embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO, dropout=DROPOUT,
        num_joints=NUM_JOINTS, coord_dim=COORD_DIM,
        input_time=INPUT_TIME, output_time=OUTPUT_TIME,
        n_ctrl_traj=N_CTRL_TRAJ if USE_SPLINE else None,
        n_ctrl_pose=N_CTRL_POSE if USE_SPLINE else None,
        spline_type=SPLINE_TYPE,
    ).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    model.eval()
    return model


@torch.no_grad()
def sample_with_noise(model, past_traj, past_pose, K, device,
                      inference_steps, noise_scale, chunk_k=20):
    """Same as euler_sample_posetraj but also returns initial noise (latent space)."""
    if K > chunk_k:
        tc, pc, ntc, npc, ltc, lpc = [], [], [], [], [], []
        for k_s in range(0, K, chunk_k):
            k_e = min(k_s + chunk_k, K)
            ct, cp, nt, np_, lt_, lp_ = sample_with_noise(
                model, past_traj, past_pose, k_e - k_s,
                device, inference_steps, noise_scale, chunk_k=chunk_k)
            tc.append(ct); pc.append(cp); ntc.append(nt); npc.append(np_)
            ltc.append(lt_); lpc.append(lp_)
        return (torch.cat(tc, 1), torch.cat(pc, 1), torch.cat(ntc, 1),
                torch.cat(npc, 1), torch.cat(ltc, 1), torch.cat(lpc, 1))

    N, T_in, _ = past_traj.shape
    pt_exp = past_traj.unsqueeze(1).expand(-1, K, -1, -1).reshape(N * K, T_in, COORD_DIM)
    pp_exp = past_pose.unsqueeze(1).expand(-1, K, -1, -1).reshape(N * K, T_in, POSE_DIM)

    T_lat_traj = model.n_ctrl_traj if model.use_spline else OUTPUT_TIME
    T_lat_pose = model.n_ctrl_pose if model.use_spline else OUTPUT_TIME
    z0_traj = torch.randn(N * K, T_lat_traj, COORD_DIM, device=device) * noise_scale
    z0_pose = torch.randn(N * K, T_lat_pose, POSE_DIM, device=device) * NOISE_SCALE_POSE

    # save initial noise
    init_noise_traj = z0_traj.clone()
    init_noise_pose = z0_pose.clone()

    z_traj, z_pose = z0_traj, z0_pose
    timesteps = torch.linspace(0, 1, inference_steps + 1, device=device)

    for i in range(inference_steps):
        t_val = timesteps[i].item()
        dt = (timesteps[i + 1] - timesteps[i]).item()
        t_vec = torch.full((N * K,), t_val, device=device)
        v1_t, v1_p = model(pt_exp, pp_exp, z_traj, z_pose, t_vec)
        z_traj_pred = z_traj + v1_t * dt
        z_pose_pred = z_pose + v1_p * dt
        t_next = (t_vec + dt).clamp(0, 1 - 1e-6)
        v2_t, v2_p = model(pt_exp, pp_exp, z_traj_pred, z_pose_pred, t_next)
        z_traj = z_traj + dt * (v1_t + v2_t) * 0.5
        z_pose = z_pose + dt * (v1_p + v2_p) * 0.5

    pred_traj = model.eval_spline_traj(z_traj.view(N, K, T_lat_traj, COORD_DIM))
    pred_pose = model.eval_spline_pose(z_pose.view(N, K, T_lat_pose, POSE_DIM))
    noise_traj = init_noise_traj.view(N, K, T_lat_traj, COORD_DIM)
    noise_pose = init_noise_pose.view(N, K, T_lat_pose, POSE_DIM)
    # also return final latent for scoring
    lat_traj = z_traj.view(N, K, T_lat_traj, COORD_DIM)
    lat_pose = z_pose.view(N, K, T_lat_pose, POSE_DIM)
    return pred_traj, pred_pose, noise_traj, noise_pose, lat_traj, lat_pose


@torch.no_grad()
def score_likelihood(model, past_traj, past_pose, lat_x1_traj, lat_x1_pose,
                     noise_traj, noise_pose, n_t_samples=5):
    """
    Score each sample by flow matching velocity prediction error.
    For each sample, create interpolants at random t values and check
    how well the model's velocity matches the target velocity (x1 - noise).
    Lower error → higher likelihood → better sample.
    Returns scores [N, K] (lower = better).
    """
    N, K = lat_x1_traj.shape[:2]
    T_lat_traj = lat_x1_traj.shape[2]
    T_lat_pose = lat_x1_pose.shape[2]
    T_in = past_traj.shape[1]

    # expand past
    pt = past_traj.unsqueeze(1).expand(-1, K, -1, -1).reshape(N * K, T_in, COORD_DIM)
    pp = past_pose.unsqueeze(1).expand(-1, K, -1, -1).reshape(N * K, T_in, POSE_DIM)

    x1_t = lat_x1_traj.reshape(N * K, T_lat_traj, COORD_DIM)
    x1_p = lat_x1_pose.reshape(N * K, T_lat_pose, POSE_DIM)
    n_t = noise_traj.reshape(N * K, T_lat_traj, COORD_DIM)
    n_p = noise_pose.reshape(N * K, T_lat_pose, POSE_DIM)

    # target velocity = x1 - noise (constant for linear interpolation)
    target_v_traj = x1_t - n_t
    target_v_pose = x1_p - n_p

    total_err = torch.zeros(N * K, device=device)

    for _ in range(n_t_samples):
        # random timestep, biased toward high t (where accuracy matters more)
        t = 0.5 + 0.5 * torch.rand(N * K, device=device)  # t ∈ [0.5, 1.0]
        t_view = t.view(N * K, 1, 1)

        # interpolate
        x_t_traj = torch.lerp(n_t, x1_t, t_view)
        x_t_pose = torch.lerp(n_p, x1_p, t_view)

        # predict velocity
        pred_v_traj, pred_v_pose = model(pt, pp, x_t_traj, x_t_pose, t)

        # velocity error
        err_traj = (pred_v_traj - target_v_traj).pow(2).mean(dim=(1, 2))
        err_pose = (pred_v_pose - target_v_pose).pow(2).mean(dim=(1, 2))
        total_err += err_traj + err_pose

    return (total_err / n_t_samples).view(N, K)


def select_likelihood_topk(scores, k=K_SELECT):
    """Pick k samples with lowest score (highest likelihood)."""
    _, topk_idx = scores.topk(k, dim=1, largest=False)
    return topk_idx


def select_likelihood_then_kmeans(scores, pred_traj_abs, pred_pose, k=K_SELECT, prefilter=30):
    """Filter to top-30 by likelihood, then kmeans for diversity."""
    N, K_total = pred_traj_abs.shape[:2]
    _, dense_idx = scores.topk(prefilter, dim=1, largest=False)

    idx_traj = dense_idx.unsqueeze(-1).unsqueeze(-1).expand(N, prefilter, OUTPUT_TIME, COORD_DIM)
    idx_pose = dense_idx.unsqueeze(-1).unsqueeze(-1).expand(N, prefilter, OUTPUT_TIME, POSE_DIM)
    filt_traj = torch.gather(pred_traj_abs, 1, idx_traj)
    filt_pose = torch.gather(pred_pose, 1, idx_pose)

    sub_idx = select_kmeans(filt_traj, filt_pose, k=k)
    return torch.gather(dense_idx, 1, sub_idx)


def compute_metrics(pred_traj_abs, pred_pose, gt_traj_abs, gt_pose, sel_idx):
    """Compute APE and JPE for selected indices."""
    frame_last = OUTPUT_TIME - 1
    N = gt_traj_abs.shape[0]
    K = sel_idx.shape[1]

    idx_traj = sel_idx.unsqueeze(-1).unsqueeze(-1).expand(N, K, OUTPUT_TIME, COORD_DIM)
    idx_pose = sel_idx.unsqueeze(-1).unsqueeze(-1).expand(N, K, OUTPUT_TIME, POSE_DIM)
    sel_traj = torch.gather(pred_traj_abs, 1, idx_traj)
    sel_pose = torch.gather(pred_pose, 1, idx_pose)

    # APE
    pred_pose_last = sel_pose[:, :, frame_last].reshape(N, K, NUM_JOINTS, COORD_DIM)
    gt_pose_last = gt_pose[:, frame_last].reshape(N, NUM_JOINTS, COORD_DIM)
    pred_aligned = pred_pose_last - pred_pose_last[:, :, 0:1, :]
    gt_aligned = gt_pose_last - gt_pose_last[:, 0:1, :]
    ape_per_mode = torch.norm(pred_aligned - gt_aligned.unsqueeze(1), dim=-1).mean(dim=-1)
    min_ape = ape_per_mode.min(dim=1).values

    # JPE
    pred_abs_j = pred_pose_last + sel_traj[:, :, frame_last].unsqueeze(2)
    gt_abs_j = gt_pose_last + gt_traj_abs[:, frame_last].unsqueeze(1)
    jpe_per_mode = torch.norm(pred_abs_j - gt_abs_j.unsqueeze(1), dim=-1).mean(dim=-1)
    min_jpe = jpe_per_mode.min(dim=1).values

    return min_ape, min_jpe


# ── Selection strategies ──

def select_kmeans(pred_traj_abs, pred_pose, k=K_SELECT):
    """Original kmeans medoid selection (diversity-optimized)."""
    frame_last = OUTPUT_TIME - 1
    traj_feat = pred_traj_abs[:, :, frame_last]
    pose_feat = pred_pose[:, :, frame_last]
    joint_feat = torch.cat([traj_feat, pose_feat], dim=-1).unsqueeze(2)
    sel_feat = kmeans_select(joint_feat, k).squeeze(2)
    dists = torch.cdist(sel_feat, joint_feat.squeeze(2))
    return dists.argmin(dim=-1)  # [N, k]


def select_density_topk(pred_traj_abs, pred_pose, k=K_SELECT):
    """Pick k samples from densest regions (consensus-based)."""
    frame_last = OUTPUT_TIME - 1
    traj_feat = pred_traj_abs[:, :, frame_last]
    pose_feat = pred_pose[:, :, frame_last]
    feat = torch.cat([traj_feat, pose_feat], dim=-1)  # [N, K, C]

    # compute pairwise distance, use mean distance as density proxy (lower = denser)
    pw_dist = torch.cdist(feat, feat)  # [N, K, K]
    mean_dist = pw_dist.mean(dim=-1)  # [N, K] — avg dist to all others
    # pick k samples with lowest mean distance (most central/dense)
    _, topk_idx = mean_dist.topk(k, dim=1, largest=False)
    return topk_idx  # [N, k]


def select_density_then_diverse(pred_traj_abs, pred_pose, k=K_SELECT, prefilter=30):
    """First filter to top-30 densest, then kmeans for diversity within that."""
    N, K_total = pred_traj_abs.shape[:2]
    frame_last = OUTPUT_TIME - 1
    traj_feat = pred_traj_abs[:, :, frame_last]
    pose_feat = pred_pose[:, :, frame_last]
    feat = torch.cat([traj_feat, pose_feat], dim=-1)  # [N, K, C]

    # density filter
    pw_dist = torch.cdist(feat, feat)
    mean_dist = pw_dist.mean(dim=-1)
    _, dense_idx = mean_dist.topk(prefilter, dim=1, largest=False)  # [N, prefilter]

    # gather filtered predictions
    idx_traj = dense_idx.unsqueeze(-1).unsqueeze(-1).expand(N, prefilter, OUTPUT_TIME, COORD_DIM)
    idx_pose = dense_idx.unsqueeze(-1).unsqueeze(-1).expand(N, prefilter, OUTPUT_TIME, POSE_DIM)
    filt_traj = torch.gather(pred_traj_abs, 1, idx_traj)
    filt_pose = torch.gather(pred_pose, 1, idx_pose)

    # kmeans on filtered set
    sub_idx = select_kmeans(filt_traj, filt_pose, k=k)  # [N, k] indices into filtered
    # map back to original indices
    return torch.gather(dense_idx, 1, sub_idx)


def select_median_nearest(pred_traj_abs, pred_pose, k=K_SELECT):
    """Pick k samples nearest to the per-coordinate median prediction."""
    frame_last = OUTPUT_TIME - 1
    traj_feat = pred_traj_abs[:, :, frame_last]
    pose_feat = pred_pose[:, :, frame_last]
    feat = torch.cat([traj_feat, pose_feat], dim=-1)  # [N, K, C]

    # median across K dimension
    median_feat = feat.median(dim=1, keepdim=True).values  # [N, 1, C]
    dist_to_median = torch.norm(feat - median_feat, dim=-1)  # [N, K]
    _, topk_idx = dist_to_median.topk(k, dim=1, largest=False)
    return topk_idx


def select_kmeans_density_hybrid(pred_traj_abs, pred_pose, k=K_SELECT):
    """Cluster into k groups, pick densest sample from each cluster."""
    N = pred_traj_abs.shape[0]
    K_total = pred_traj_abs.shape[1]
    frame_last = OUTPUT_TIME - 1
    traj_feat = pred_traj_abs[:, :, frame_last]
    pose_feat = pred_pose[:, :, frame_last]
    feat = torch.cat([traj_feat, pose_feat], dim=-1)  # [N, K, C]

    # run kmeans to get cluster assignments
    centers = feat[:, :k].clone()
    for _ in range(30):
        dists = torch.cdist(feat, centers)
        assignments = dists.argmin(dim=-1)  # [N, K]
        one_hot = torch.zeros(N, K_total, k, device=feat.device).scatter_(-1, assignments.unsqueeze(-1), 1.0)
        cluster_sums = torch.bmm(one_hot.transpose(1, 2), feat)
        counts = one_hot.sum(dim=1).unsqueeze(-1).clamp(min=1)
        new_centers = cluster_sums / counts
        empty = (one_hot.sum(dim=1) == 0).unsqueeze(-1)
        centers = torch.where(empty, centers, new_centers)

    # for each cluster, pick sample with highest local density (lowest mean dist to cluster members)
    dists_all = torch.cdist(feat, feat)  # [N, K, K]
    sel_indices = []
    for c in range(k):
        mask = (assignments == c)  # [N, K]
        # set dists from non-members to inf
        masked_dists = dists_all.clone()
        masked_dists[~mask.unsqueeze(2).expand_as(masked_dists)] = float('inf')
        masked_dists[~mask.unsqueeze(1).expand_as(masked_dists)] = float('inf')
        # mean distance to cluster members
        member_count = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        # sum of finite distances
        finite_mask = masked_dists < float('inf')
        sum_dists = (masked_dists * finite_mask.float()).sum(dim=-1)  # [N, K]
        avg_dists = sum_dists / member_count
        avg_dists[~mask] = float('inf')
        best_in_cluster = avg_dists.argmin(dim=-1)  # [N]
        sel_indices.append(best_in_cluster)

    return torch.stack(sel_indices, dim=1)  # [N, k]


def _run_kmeans_on_feat(feat, k, n_iter=30):
    """Run kmeans on arbitrary features [N, K, C], return medoid indices [N, k]."""
    B, N_pts, C = feat.shape
    centers = feat[:, :k].clone()
    for _ in range(n_iter):
        dists = torch.cdist(feat, centers)
        assignments = dists.argmin(dim=-1)
        one_hot = torch.zeros(B, N_pts, k, device=feat.device).scatter_(-1, assignments.unsqueeze(-1), 1.0)
        cluster_sums = torch.bmm(one_hot.transpose(1, 2), feat)
        counts = one_hot.sum(dim=1).unsqueeze(-1).clamp(min=1)
        new_centers = cluster_sums / counts
        empty = (one_hot.sum(dim=1) == 0).unsqueeze(-1)
        centers = torch.where(empty, centers, new_centers)
    medoid_idx = torch.cdist(centers, feat).argmin(dim=-1)
    return medoid_idx


def select_kmeans_traj_only(pred_traj_abs, pred_pose, k=K_SELECT):
    """Kmeans on 3D trajectory at last frame only."""
    frame_last = OUTPUT_TIME - 1
    feat = pred_traj_abs[:, :, frame_last]  # [N, K, 3]
    return _run_kmeans_on_feat(feat, k)


def select_kmeans_pca(pred_traj_abs, pred_pose, k=K_SELECT, n_components=8):
    """Kmeans on PCA-reduced joint features (48D → n_components)."""
    frame_last = OUTPUT_TIME - 1
    traj_feat = pred_traj_abs[:, :, frame_last]
    pose_feat = pred_pose[:, :, frame_last]
    feat = torch.cat([traj_feat, pose_feat], dim=-1)  # [N, K, 48]
    N, K_total, D = feat.shape
    # per-sample PCA: center, SVD, project
    centered = feat - feat.mean(dim=1, keepdim=True)
    # batch SVD on [N, K, D] — compute covariance [N, D, D] then top eigenvectors
    cov = torch.bmm(centered.transpose(1, 2), centered) / max(K_total - 1, 1)
    # eigendecomposition (symmetric)
    eigvals, eigvecs = torch.linalg.eigh(cov)  # eigvecs: [N, D, D], ascending
    # take top n_components (last columns = largest eigenvalues)
    proj = eigvecs[:, :, -n_components:]  # [N, D, n_components]
    reduced = torch.bmm(centered, proj)  # [N, K, n_components]
    return _run_kmeans_on_feat(reduced, k)


def select_kmeans_weighted(pred_traj_abs, pred_pose, k=K_SELECT, traj_scale=5.0):
    """Kmeans with trajectory dims scaled up to balance with pose."""
    frame_last = OUTPUT_TIME - 1
    traj_feat = pred_traj_abs[:, :, frame_last] * traj_scale  # [N, K, 3] scaled
    pose_feat = pred_pose[:, :, frame_last]                    # [N, K, 45]
    feat = torch.cat([traj_feat, pose_feat], dim=-1)           # [N, K, 48]
    return _run_kmeans_on_feat(feat, k)


def select_kmeans_abs_joints(pred_traj_abs, pred_pose, k=K_SELECT):
    """Kmeans on absolute joint positions at last frame (traj + pose combined)."""
    frame_last = OUTPUT_TIME - 1
    N, K_total = pred_traj_abs.shape[:2]
    pose_last = pred_pose[:, :, frame_last].reshape(N, K_total, NUM_JOINTS, COORD_DIM)
    traj_last = pred_traj_abs[:, :, frame_last].unsqueeze(2)  # [N, K, 1, 3]
    abs_joints = (pose_last + traj_last).reshape(N, K_total, -1)  # [N, K, J*3]
    return _run_kmeans_on_feat(abs_joints, k)


# ── Main eval loop ──

STRATEGIES = {
    "kmeans (baseline)": select_kmeans,
    "kmeans_traj_only": select_kmeans_traj_only,
    "kmeans_pca8": select_kmeans_pca,
    "kmeans_weighted_5x": select_kmeans_weighted,
    "kmeans_weighted_10x": lambda t, p, k=K_SELECT: select_kmeans_weighted(t, p, k, traj_scale=10.0),
    "kmeans_weighted_15x": lambda t, p, k=K_SELECT: select_kmeans_weighted(t, p, k, traj_scale=15.0),
    "kmeans_abs_joints": select_kmeans_abs_joints,
    "density_topk": select_density_topk,
    "median_nearest": select_median_nearest,
    "kmeans+density_hybrid": select_kmeans_density_hybrid,
}


@torch.no_grad()
def run_eval():
    import json
    global CKPT, SAVE_PATH, device
    parser = argparse.ArgumentParser(description="Evaluate FlowMoP proposal selection strategies.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    CKPT = str(args.checkpoint)
    SAVE_PATH = args.output
    device = torch.device(args.device)
    training.DATA_ROOT = str(args.data_dir)
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = load_model()
    _, val_loader = get_dataloaders()

    print(f"Generating {K_PROPOSALS} proposals, evaluating selection strategies per batch\n")

    # accumulators
    all_results = {name: {"ape": [], "jpe": []} for name in STRATEGIES}
    all_results["oracle_100"] = {"ape": [], "jpe": []}
    t0_total = time.time()

    for bi, batch in enumerate(val_loader):
        b = prepare_batch(batch, device)
        past_traj = b["input_traj"]
        past_pose = b["input_pose_rel"]
        gt_traj = b["output_traj"]
        gt_pose = b["output_pose_rel"]
        offset = b["offset_traj"]

        t0 = time.time()
        pred_traj, pred_pose = euler_sample_posetraj(
            model, past_traj, past_pose,
            K=K_PROPOSALS, device=device,
            inference_steps=INFERENCE_STEPS, noise_scale=NOISE_SCALE,
        )
        gt_traj_abs = gt_traj + offset
        pred_traj_abs = pred_traj + offset.unsqueeze(1)
        gen_s = time.time() - t0

        # oracle
        N = gt_traj_abs.shape[0]
        all_idx = torch.arange(K_PROPOSALS, device=device).unsqueeze(0).expand(N, -1)
        a, j = compute_metrics(pred_traj_abs, pred_pose, gt_traj_abs, gt_pose, all_idx)
        all_results["oracle_100"]["ape"].extend(a.cpu().tolist())
        all_results["oracle_100"]["jpe"].extend(j.cpu().tolist())

        # each strategy
        for name, select_fn in STRATEGIES.items():
            sel_idx = select_fn(pred_traj_abs, pred_pose, k=K_SELECT)
            a, j = compute_metrics(pred_traj_abs, pred_pose, gt_traj_abs, gt_pose, sel_idx)
            all_results[name]["ape"].extend(a.cpu().tolist())
            all_results[name]["jpe"].extend(j.cpu().tolist())

        # save partial results
        partial = {}
        for name, r in all_results.items():
            if r["ape"]:
                partial[name] = {
                    "APE": round(sum(r["ape"]) / len(r["ape"]) * 1000, 1),
                    "JPE": round(sum(r["jpe"]) / len(r["jpe"]) * 1000, 1),
                    "n": len(r["ape"]),
                }
        with SAVE_PATH.open("w") as f:
            json.dump(partial, f, indent=2)

        print(f"  batch {bi+1}/{len(val_loader)}  gen={gen_s:.0f}s  N={N}", flush=True)

        del pred_traj, pred_pose, pred_traj_abs
        torch.cuda.empty_cache()

    total_min = (time.time() - t0_total) / 60
    print(f"\nTotal time: {total_min:.1f} min\n")

    for name, r in all_results.items():
        ape = sum(r["ape"]) / len(r["ape"]) * 1000
        jpe = sum(r["jpe"]) / len(r["jpe"]) * 1000
        print(f"  {name:30s}  APE={ape:.1f}  JPE={jpe:.1f}")

    print("\nDone.")


if __name__ == "__main__":
    run_eval()

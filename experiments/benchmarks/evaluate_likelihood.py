"""True log-likelihood scoring via Hutchinson trace estimator for flow matching.

log p(x1) = log p(z0) - ∫₀¹ Tr(∂v/∂x_t) dt

Uses Hutchinson: Tr(J) = E[εᵀ J ε], computed as ε · ∇_x(v·ε) in one backward pass.
"""
import math
import torch
import time
import json
import argparse
from pathlib import Path

from train import (
    PoseTrajFM, prepare_batch,
    kmeans_select,
    DEPTH, EMBED_DIM, NUM_HEADS, MLP_RATIO, DROPOUT,
    NUM_JOINTS, COORD_DIM, POSE_DIM, INPUT_TIME, OUTPUT_TIME,
    NOISE_SCALE, NOISE_SCALE_POSE, INFERENCE_STEPS,
    USE_SPLINE, N_CTRL_TRAJ, N_CTRL_POSE, SPLINE_TYPE,
)

device = torch.device("cuda")
CKPT = None
K_PROPOSALS = 100
K_SELECT = 6
N_HUTCHINSON = 1     # single random vector per step (fast, noisier)
SCORE_ODE_STEPS = 5  # fewer Euler steps (faster)
VAL_BATCH_SIZE = 16  # smaller batches to reduce peak memory


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


def compute_log_likelihood(model, past_traj, past_pose, noise_traj, noise_pose,
                           n_steps=SCORE_ODE_STEPS, n_hutch=N_HUTCHINSON, chunk_k=10):
    """
    Compute log p(x1) for each of K samples by integrating the augmented ODE:
      dx/dt = v(x, t)
      d(log_p)/dt = -Tr(∂v/∂x)

    Args:
        past_traj: [N, T_in, 3]
        past_pose: [N, T_in, J*3]
        noise_traj: [N, K, T_lat_traj, 3] — initial noise (z0)
        noise_pose: [N, K, T_lat_pose, J*3]

    Returns:
        log_p: [N, K] — log-likelihood of each generated sample
    """
    N, K = noise_traj.shape[:2]
    T_lat_traj = noise_traj.shape[2]
    T_lat_pose = noise_pose.shape[2]
    T_in = past_traj.shape[1]

    # log p(z0): standard normal prior
    D_traj = T_lat_traj * COORD_DIM
    D_pose = T_lat_pose * POSE_DIM

    # z0_traj was scaled by NOISE_SCALE, z0_pose by NOISE_SCALE_POSE
    # log p(z0) under N(0, σ²I) = -D/2 log(2π) - 1/(2σ²) ||z0||²
    log_pz0_traj = (-0.5 * D_traj * math.log(2 * math.pi)
                    - D_traj * math.log(NOISE_SCALE)
                    - 0.5 * noise_traj.reshape(N, K, -1).pow(2).sum(-1) / (NOISE_SCALE**2))
    log_pz0_pose = (-0.5 * D_pose * math.log(2 * math.pi)
                    - D_pose * math.log(NOISE_SCALE_POSE)
                    - 0.5 * noise_pose.reshape(N, K, -1).pow(2).sum(-1) / (NOISE_SCALE_POSE**2))
    log_pz0 = log_pz0_traj + log_pz0_pose  # [N, K]

    # process in chunks of chunk_k samples to manage memory (gradients are expensive)
    all_trace_integrals = []

    for k_start in range(0, K, chunk_k):
        k_end = min(k_start + chunk_k, K)
        Kc = k_end - k_start

        # expand past for this chunk
        pt = past_traj.unsqueeze(1).expand(-1, Kc, -1, -1).reshape(N * Kc, T_in, COORD_DIM)
        pp = past_pose.unsqueeze(1).expand(-1, Kc, -1, -1).reshape(N * Kc, T_in, POSE_DIM)

        # initial state from noise
        x_traj = noise_traj[:, k_start:k_end].reshape(N * Kc, T_lat_traj, COORD_DIM).clone()
        x_pose = noise_pose[:, k_start:k_end].reshape(N * Kc, T_lat_pose, POSE_DIM).clone()

        trace_acc = torch.zeros(N * Kc, device=device)
        timesteps = torch.linspace(0, 1, n_steps + 1, device=device)

        for i in range(n_steps):
            t_val = timesteps[i].item()
            dt = (timesteps[i + 1] - timesteps[i]).item()
            t_vec = torch.full((N * Kc,), t_val, device=device)

            # Hutchinson trace: run separate forward+backward per random vector
            trace_est = torch.zeros(N * Kc, device=device)
            v_traj_detached = None

            for h in range(n_hutch):
                x_traj_g = x_traj.detach().requires_grad_(True)
                x_pose_g = x_pose.detach().requires_grad_(True)

                v_traj, v_pose = model(pt, pp, x_traj_g, x_pose_g, t_vec)

                eps_traj = torch.randn_like(x_traj_g)
                eps_pose = torch.randn_like(x_pose_g)

                s = (v_traj * eps_traj).sum() + (v_pose * eps_pose).sum()

                # retain_graph=False — graph freed after each backward
                grad_traj, grad_pose = torch.autograd.grad(
                    s, (x_traj_g, x_pose_g), retain_graph=False
                )

                tr_sample = ((eps_traj * grad_traj).reshape(N * Kc, -1).sum(-1) +
                             (eps_pose * grad_pose).reshape(N * Kc, -1).sum(-1))
                trace_est += tr_sample

                # save last velocity for Euler step (detached)
                if h == n_hutch - 1:
                    v_traj_detached = v_traj.detach()
                    v_pose_detached = v_pose.detach()

            trace_est /= n_hutch
            trace_acc += trace_est * dt

            # Euler step (no grad)
            with torch.no_grad():
                x_traj = x_traj + v_traj_detached * dt
                x_pose = x_pose + v_pose_detached * dt

        all_trace_integrals.append(trace_acc.view(N, Kc))

    trace_integral = torch.cat(all_trace_integrals, dim=1)  # [N, K]
    log_px1 = log_pz0 - trace_integral
    return log_px1


def sample_with_noise(model, past_traj, past_pose, K, n_steps=INFERENCE_STEPS, chunk_k=20):
    """Heun sampler that also returns initial noise and final latent."""
    N, T_in, _ = past_traj.shape
    T_lat_traj = model.n_ctrl_traj if model.use_spline else OUTPUT_TIME
    T_lat_pose = model.n_ctrl_pose if model.use_spline else OUTPUT_TIME

    all_pred_traj, all_pred_pose = [], []
    all_noise_traj, all_noise_pose = [], []

    for k_s in range(0, K, chunk_k):
        Kc = min(chunk_k, K - k_s)
        pt = past_traj.unsqueeze(1).expand(-1, Kc, -1, -1).reshape(N * Kc, T_in, COORD_DIM)
        pp = past_pose.unsqueeze(1).expand(-1, Kc, -1, -1).reshape(N * Kc, T_in, POSE_DIM)

        z_traj = torch.randn(N * Kc, T_lat_traj, COORD_DIM, device=device) * NOISE_SCALE
        z_pose = torch.randn(N * Kc, T_lat_pose, POSE_DIM, device=device) * NOISE_SCALE_POSE
        n_t = z_traj.clone().view(N, Kc, T_lat_traj, COORD_DIM)
        n_p = z_pose.clone().view(N, Kc, T_lat_pose, POSE_DIM)

        timesteps = torch.linspace(0, 1, n_steps + 1, device=device)
        with torch.no_grad():
            for i in range(n_steps):
                t_val = timesteps[i].item()
                dt = (timesteps[i + 1] - timesteps[i]).item()
                tv = torch.full((N * Kc,), t_val, device=device)
                v1t, v1p = model(pt, pp, z_traj, z_pose, tv)
                zt_p = z_traj + v1t * dt
                zp_p = z_pose + v1p * dt
                tn = (tv + dt).clamp(0, 1 - 1e-6)
                v2t, v2p = model(pt, pp, zt_p, zp_p, tn)
                z_traj = z_traj + dt * (v1t + v2t) * 0.5
                z_pose = z_pose + dt * (v1p + v2p) * 0.5

        pred_t = model.eval_spline_traj(z_traj.view(N, Kc, T_lat_traj, COORD_DIM))
        pred_p = model.eval_spline_pose(z_pose.view(N, Kc, T_lat_pose, POSE_DIM))
        all_pred_traj.append(pred_t)
        all_pred_pose.append(pred_p)
        all_noise_traj.append(n_t)
        all_noise_pose.append(n_p)

    return (torch.cat(all_pred_traj, 1), torch.cat(all_pred_pose, 1),
            torch.cat(all_noise_traj, 1), torch.cat(all_noise_pose, 1))


def compute_metrics(pred_traj_abs, pred_pose, gt_traj_abs, gt_pose, sel_idx):
    frame_last = OUTPUT_TIME - 1
    N, K = sel_idx.shape
    idx_t = sel_idx.unsqueeze(-1).unsqueeze(-1).expand(N, K, OUTPUT_TIME, COORD_DIM)
    idx_p = sel_idx.unsqueeze(-1).unsqueeze(-1).expand(N, K, OUTPUT_TIME, POSE_DIM)
    sel_t = torch.gather(pred_traj_abs, 1, idx_t)
    sel_p = torch.gather(pred_pose, 1, idx_p)

    pp_last = sel_p[:, :, frame_last].reshape(N, K, NUM_JOINTS, COORD_DIM)
    gp_last = gt_pose[:, frame_last].reshape(N, NUM_JOINTS, COORD_DIM)
    pa = pp_last - pp_last[:, :, 0:1, :]
    ga = gp_last - gp_last[:, 0:1, :]
    ape = torch.norm(pa - ga.unsqueeze(1), dim=-1).mean(-1).min(1).values

    pj = pp_last + sel_t[:, :, frame_last].unsqueeze(2)
    gj = gp_last + gt_traj_abs[:, frame_last].unsqueeze(1)
    jpe = torch.norm(pj - gj.unsqueeze(1), dim=-1).mean(-1).min(1).values
    return ape, jpe


def select_kmeans(pred_traj_abs, pred_pose, k=K_SELECT):
    frame_last = OUTPUT_TIME - 1
    feat = torch.cat([pred_traj_abs[:, :, frame_last],
                      pred_pose[:, :, frame_last]], dim=-1).unsqueeze(2)
    sel = kmeans_select(feat, k).squeeze(2)
    dists = torch.cdist(sel, feat.squeeze(2))
    return dists.argmin(dim=-1)


def select_ll_topk(log_p, k=K_SELECT):
    """Pick k highest log-likelihood samples."""
    _, idx = log_p.topk(k, dim=1, largest=True)
    return idx


def select_ll_then_kmeans(log_p, pred_traj_abs, pred_pose, k=K_SELECT, prefilter=30):
    """Filter top-prefilter by log-likelihood, then kmeans for diversity."""
    N = pred_traj_abs.shape[0]
    _, top_idx = log_p.topk(prefilter, dim=1, largest=True)
    it = top_idx.unsqueeze(-1).unsqueeze(-1).expand(N, prefilter, OUTPUT_TIME, COORD_DIM)
    ip = top_idx.unsqueeze(-1).unsqueeze(-1).expand(N, prefilter, OUTPUT_TIME, POSE_DIM)
    ft = torch.gather(pred_traj_abs, 1, it)
    fp = torch.gather(pred_pose, 1, ip)
    sub = select_kmeans(ft, fp, k=k)
    return torch.gather(top_idx, 1, sub)


@torch.no_grad()
def main():
    global CKPT, device
    parser = argparse.ArgumentParser(description="Evaluate likelihood-based FlowMoP selection.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-batches", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    CKPT = str(args.checkpoint)
    device = torch.device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = load_model()
    # override val batch size for memory
    from torch_geometric.loader import DataLoader as GeoDataLoader
    from dataloader import FlowMoPDataset
    val_ds = FlowMoPDataset(
        dataset="mocap_umpm", mode=1, device="cpu", input_time=INPUT_TIME,
        normalize_data=False, offset_traj="last_input", data_root=args.data_dir,
    )
    val_loader = GeoDataLoader(val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False,
                               num_workers=4, pin_memory=True)

    print(f"True log-likelihood scoring: {N_HUTCHINSON} Hutchinson vectors, {SCORE_ODE_STEPS} Euler steps")
    print(f"Generating {K_PROPOSALS} proposals, scoring, and selecting per batch\n")

    # accumulators for each strategy
    results = {
        "ll_top6":         {"ape": [], "jpe": []},
        "ll→kmeans (30)":  {"ape": [], "jpe": []},
        "ll→kmeans (50)":  {"ape": [], "jpe": []},
    }

    t0_total = time.time()
    for bi, batch in enumerate(val_loader):
        if bi >= args.max_batches:
            break
        b = prepare_batch(batch, device)
        past_traj = b["input_traj"]
        past_pose = b["input_pose_rel"]
        gt_traj = b["output_traj"]
        gt_pose = b["output_pose_rel"]
        offset = b["offset_traj"]
        N_batch = past_traj.shape[0]

        # generate proposals (on GPU, no grad)
        t0 = time.time()
        pred_traj, pred_pose, noise_t, noise_p = sample_with_noise(
            model, past_traj, past_pose, K=K_PROPOSALS)
        gt_traj_abs = gt_traj + offset
        pred_traj_abs = pred_traj + offset.unsqueeze(1)
        gen_s = time.time() - t0

        # score each sample individually via Hutchinson log-likelihood
        t0 = time.time()
        batch_logp = []
        for s in range(N_batch):
            with torch.enable_grad():
                logp_s = compute_log_likelihood(
                    model,
                    past_traj[s:s+1], past_pose[s:s+1],
                    noise_t[s:s+1], noise_p[s:s+1],
                    n_steps=SCORE_ODE_STEPS, n_hutch=N_HUTCHINSON, chunk_k=2)
            batch_logp.append(logp_s.detach())
            torch.cuda.empty_cache()
        logp = torch.cat(batch_logp, dim=0)  # [N_batch, K]
        score_s = time.time() - t0

        # evaluate each selection strategy
        for name, r in results.items():
            if name == "ll_top6":
                sel_idx = select_ll_topk(logp, K_SELECT)
            elif name == "ll→kmeans (30)":
                sel_idx = select_ll_then_kmeans(logp, pred_traj_abs, pred_pose, K_SELECT, 30)
            elif name == "ll→kmeans (50)":
                sel_idx = select_ll_then_kmeans(logp, pred_traj_abs, pred_pose, K_SELECT, 50)
            a, j = compute_metrics(pred_traj_abs, pred_pose, gt_traj_abs, gt_pose, sel_idx)
            r["ape"].extend(a.cpu().tolist())
            r["jpe"].extend(j.cpu().tolist())

        print(f"  batch {bi+1}/{min(args.max_batches, len(val_loader))}  gen={gen_s:.0f}s  score={score_s:.0f}s  "
              f"logp=[{logp.min().item():.0f}, {logp.max().item():.0f}]", flush=True)

        # save partial results after each batch
        partial = {}
        for name, r in results.items():
            if r["ape"]:
                partial[name] = {
                    "APE": sum(r["ape"]) / len(r["ape"]) * 1000,
                    "JPE": sum(r["jpe"]) / len(r["jpe"]) * 1000,
                    "n_samples": len(r["ape"]),
                }
        with args.output.open("w") as f:
            json.dump(partial, f, indent=2)

        # free GPU memory before next batch
        del pred_traj, pred_pose, noise_t, noise_p, pred_traj_abs, logp, batch_logp
        torch.cuda.empty_cache()

    total_min = (time.time() - t0_total) / 60
    print(f"\nTotal time: {total_min:.1f} min")

    for name, r in results.items():
        ape = sum(r["ape"]) / len(r["ape"]) * 1000
        jpe = sum(r["jpe"]) / len(r["jpe"]) * 1000
        print(f"\n{name}:  APE={ape:.1f}  JPE={jpe:.1f}")

    print("\nDone.")


if __name__ == "__main__":
    main()

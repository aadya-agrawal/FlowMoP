"""
Temporal Representation Ablation: Raw / DCT / Linear Spline / Cubic Spline.
Trains seven configurations from scratch, varying only
the temporal basis (USE_SPLINE, SPLINE_TYPE, N_CTRL_TRAJ, N_CTRL_POSE).

Launches one process per GPU, running up to N_GPUS jobs concurrently.
"""
import os
import sys
import argparse
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


ABLATION_CONFIGS = [
    # (name, use_spline, spline_type, n_ctrl_traj, n_ctrl_pose)
    ("raw", False, None, None, None),

    ("dct_50_50", True, "dct", 50, 50),

    ("linear_4_12", True, "linear", 4, 12),

    ("linear_8_24", True, "linear", 8, 24),

    ("cubic_4_12", True, "cubic", 4, 12),

    ("cubic_8_24", True, "cubic", 8, 24),

    ("cubic_12_36", True, "cubic", 12, 36),
]


def launch_single(gpu_id, name, use_spline, spline_type, n_ctrl_traj, n_ctrl_pose,
                  data_dir, output_dir, train_steps):
    """Launch a single training run on the specified GPU as a subprocess."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # pass config as env vars to the worker script
    env["ABLATION_USE_SPLINE"]  = str(int(use_spline))
    env["ABLATION_SPLINE_TYPE"] = spline_type or ""
    env["ABLATION_N_CTRL_TRAJ"] = str(n_ctrl_traj or 0)
    env["ABLATION_N_CTRL_POSE"] = str(n_ctrl_pose or 0)

    run_dir = os.path.join(output_dir, name)
    log_path = os.path.join(run_dir, "train.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    print(f"  [GPU {gpu_id}] Starting {name}  ->  {log_path}")
    with open(log_path, "w") as log_f:
        proc = subprocess.run(
            [sys.executable, "-u", "representation_ablation_worker.py",
             "--dataset", "mocap_umpm", "--data-dir", data_dir,
             "--output-dir", run_dir, "--train-steps", str(train_steps)],
            env=env, stdout=log_f, stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
    return name, gpu_id, proc.returncode


def run_all():
    parser = argparse.ArgumentParser(description="Run the FlowMoP temporal representation ablation.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/representation_ablation")
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--train-steps", type=int, default=82000)
    args = parser.parse_args()
    n_gpus = len(args.gpus)
    print(f"Running {len(ABLATION_CONFIGS)} ablation experiments on {n_gpus} GPUs: {args.gpus}")
    print(f"Max {n_gpus} concurrent processes\n")

    t0 = time.time()
    # map of future -> (exp_id, gpu_id)
    gpu_queue = list(args.gpus)
    pending = list(ABLATION_CONFIGS)
    active = {}  # future -> gpu_id
    results = []

    with ProcessPoolExecutor(max_workers=n_gpus) as pool:
        # seed initial batch
        while gpu_queue and pending:
            gpu_id = gpu_queue.pop(0)
            cfg = pending.pop(0)
            fut = pool.submit(launch_single, gpu_id, *cfg, args.data_dir, args.output_dir, args.train_steps)
            active[fut] = gpu_id

        # as jobs finish, launch remaining ones on freed GPUs
        while active:
            for fut in as_completed(active):
                gpu_id = active.pop(fut)
                exp_id, _, rc = fut.result()
                status = "OK" if rc == 0 else f"FAILED (rc={rc})"
                print(f"  [GPU {gpu_id}] Finished {exp_id}: {status}")
                results.append((exp_id, rc))

                # launch next pending job on this GPU
                if pending:
                    cfg = pending.pop(0)
                    new_fut = pool.submit(launch_single, gpu_id, *cfg, args.data_dir, args.output_dir, args.train_steps)
                    active[new_fut] = gpu_id
                break  # re-enter as_completed with updated dict

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*70}")
    print(f"  All {len(results)} experiments completed in {elapsed:.1f} min")
    for exp_id, rc in results:
        print(f"    {exp_id:>30}  {'OK' if rc == 0 else 'FAILED'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_all()

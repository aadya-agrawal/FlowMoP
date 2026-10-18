"""
Worker for the temporal representation ablation.
"""
import os
import train as T

# read config from environment
T.USE_SPLINE  = bool(int(os.environ["ABLATION_USE_SPLINE"]))
T.LOG_EVERY   = 200

if T.USE_SPLINE:
    T.SPLINE_TYPE  = os.environ["ABLATION_SPLINE_TYPE"]
    T.N_CTRL_TRAJ  = int(os.environ["ABLATION_N_CTRL_TRAJ"])
    T.N_CTRL_POSE  = int(os.environ["ABLATION_N_CTRL_POSE"])

print(f"[worker] spline={T.USE_SPLINE}  "
      f"type={T.SPLINE_TYPE if T.USE_SPLINE else 'N/A'}  "
      f"ctrl=({T.N_CTRL_TRAJ if T.USE_SPLINE else 50}/"
      f"{T.N_CTRL_POSE if T.USE_SPLINE else 50})")

T.main()

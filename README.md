# FlowMoP: Stochastic Multi-Person Motion Prediction

Official code for FlowMoP, a conditional flow matching model that generates diverse future 3D motion for multiple people. It predicts global root trajectories and root-relative poses together, instead of returning a fixed set of deterministic outputs per person.

- **Flow matching over spline control points.** Future motion is generated in a compact B-spline control-point space, which keeps predictions smooth and shortens the sequence the transformer processes.
- **Dual-stream transformer.** Trajectory and pose tokens are kept separate, and future tokens attend to the observed past at every layer.
- **Interaction-aware.** On WorldPose (soccer), each player can be conditioned on the observed motion of nearby players.
- **Many proposals, few outputs.** Many samples are drawn per person and reduced to a small set of representative predictions.

Evaluated on CMU-Mocap (UMPM), 3DPW, and WorldPose. See the paper for results.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install a PyTorch build that matches your CUDA version first if the default wheel does not.

## Repository layout

```text
experiments/
├── benchmarks/   # UMPM and 3DPW: training, evaluation, ablations
└── worldpose/    # WorldPose: preprocessing, training, inference
```

Run scripts from inside their own directory so sibling imports resolve. Every dataset, checkpoint, and output location is a command-line argument, and `--help` lists the options for each script.

## Benchmarks (UMPM, 3DPW)

Data is expected as preprocessed scenes:

```text
<data-dir>/
├── mocap_UMPM_input25_v2/{train,val}/*.pt
└── 3dpw_input10_v2/{train,val}/*.pt
```

Each `.pt` file is a PyTorch Geometric `Data` object with a `body_xyz` tensor of shape `[people, frames, joints, 3]`.

```bash
cd experiments/benchmarks

# train
python train.py --dataset mocap_umpm --data-dir /path/to/data --output-dir outputs/umpm
python train.py --dataset 3dpw       --data-dir /path/to/data --output-dir outputs/3dpw

# evaluate a checkpoint
python evaluate_selection.py  --checkpoint outputs/umpm/checkpoints/best.pt --data-dir /path/to/data --output selection.json
python evaluate_likelihood.py --checkpoint outputs/umpm/checkpoints/best.pt --data-dir /path/to/data --output likelihood.json

# temporal representation ablation (raw, DCT, linear and cubic splines)
python run_representation_ablation.py --data-dir /path/to/data --gpus 0 1 2
```

| Script | Purpose |
| --- | --- |
| `train.py` | Train FlowMoP; also reports forward-pass latency with `--benchmark` |
| `evaluate_selection.py` | Compare strategies for choosing final predictions from the proposal pool |
| `evaluate_likelihood.py` | Likelihood-based selection using probability-flow ODE scoring |
| `run_representation_ablation.py` | Train and compare the temporal representations |
| `dataloader.py`, `augmentations.py` | Dataset loading and on-the-fly augmentation |

## WorldPose

```bash
cd experiments/worldpose

# raw WorldPose SMPL sequences -> processed windows
python preprocess.py --data-dir /path/to/poses --output-dir /path/to/processed \
    --smpl-model-dir /path/to/body_models

# train
python train.py --data-dir /path/to/processed --output-dir outputs/worldpose

# sliding-window inference over a full sequence
python infer.py --sequence /path/to/sequence.npz \
    --checkpoint outputs/worldpose/checkpoints/best.pt \
    --metadata /path/to/processed/metadata.json \
    --smpl-model-dir /path/to/body_models --output predictions.npz
```

`preprocess.py` writes `train_data.npz`, `test_data.npz`, and `metadata.json` to the output directory. The WorldPose dataset and the SMPL body models are obtained separately and are subject to their own licenses.

SMPL body models: download the SMPL model files from the [SMPL website](https://smpl.is.tue.mpg.de/) and place them as `<smpl-model-dir>/smpl/SMPL_NEUTRAL.pkl` (and `SMPL_MALE.pkl` / `SMPL_FEMALE.pkl` if you use `--smpl-gender male` or `female`). They are needed by `preprocess.py` and `infer.py`, not by training.

Training writes `checkpoints/best.pt` (EMA weights) and metric logs under `--output-dir`.


## Citation

Citation details will be added once the paper is publicly available.

## License

See [LICENSE](LICENSE).

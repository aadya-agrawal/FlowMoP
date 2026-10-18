import os
from pathlib import Path

import torch
from torch_geometric.data import Dataset

from augmentations import apply_combined_augmentations


_DATASET_DIRS = {
    "mocap_umpm": "mocap_UMPM_input25_v2",
    "3dpw": "3dpw_input10_v2",
}


class FlowMoPDataset(Dataset):
    def __init__(
        self,
        dataset,
        mode=0,
        input_time=25,
        on_the_fly_augmentations=False,
        augmentation_config=None,
        offset_traj="last_input",
        data_root=None,
        **_,
    ):
        if dataset not in _DATASET_DIRS:
            raise ValueError(f"Unsupported FlowMoP benchmark dataset: {dataset}")

        root = Path(
            data_root
            or os.environ.get("FLOWMOP_PREPROCESSED_DIR", "data/preprocessed")
        )
        split = "train" if mode == 0 else "val"
        self.files = sorted((root / _DATASET_DIRS[dataset] / split).glob("*.pt"))
        if not self.files:
            raise ValueError(
                f"No {dataset} {split} data found under {root}. "
                "Set FLOWMOP_PREPROCESSED_DIR to the preprocessed dataset root."
            )

        self.input_time = input_time
        self.on_the_fly_augmentations = on_the_fly_augmentations
        self.augmentation_config = augmentation_config or {}
        self.offset_traj = offset_traj
        super().__init__(root=None)

    def len(self):
        return len(self.files)

    def get(self, index):
        sample = torch.load(self.files[index], map_location="cpu", weights_only=False)
        joints = torch.as_tensor(sample.body_xyz, dtype=torch.float32)

        if self.on_the_fly_augmentations:
            joints = apply_combined_augmentations(joints, self.augmentation_config)

        num_people, total_time, num_joints, coord_dim = joints.shape
        input_joints = joints[:, :self.input_time]
        output_joints = joints[:, self.input_time:]

        sample.input_seq = input_joints.reshape(
            num_people, self.input_time, num_joints * coord_dim
        )
        sample.output_seq = output_joints.reshape(
            num_people, total_time - self.input_time, num_joints * coord_dim
        )

        input_root = input_joints[:, :, 0]
        output_root = output_joints[:, :, 0]
        if self.offset_traj == "prev_frame":
            sample.input_traj = torch.zeros_like(input_root)
            sample.input_traj[:, 1:] = input_root[:, 1:] - input_root[:, :-1]
            sample.output_traj = torch.zeros_like(output_root)
            sample.output_traj[:, 0] = output_root[:, 0] - input_root[:, -1]
            sample.output_traj[:, 1:] = output_root[:, 1:] - output_root[:, :-1]
            sample.offset_traj = input_root[:, :1]
        elif self.offset_traj == "none":
            sample.input_traj = input_root
            sample.output_traj = output_root
            sample.offset_traj = torch.zeros_like(input_root[:, :1])
        else:
            sample.offset_traj = input_root[:, -1:]
            sample.input_traj = input_root - sample.offset_traj
            sample.output_traj = output_root - sample.offset_traj

        if hasattr(sample, "bos_mask"):
            sample.bos_mask = sample.bos_mask.cpu()
        return sample

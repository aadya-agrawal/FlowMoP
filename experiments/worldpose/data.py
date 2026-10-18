import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class WorldPoseDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        metadata_path: Path,
        normalize: bool = False,
        max_people: int = 22,
        limit_samples: Optional[int] = None,
    ):
        with metadata_path.open() as metadata_file:
            self.metadata = json.load(metadata_file)

        self.normalization_stats = self.metadata["normalization_stats"]
        self.normalize = normalize
        self.max_people = max_people
        self.windows = np.load(data_path, allow_pickle=True)["windows"]
        if limit_samples is not None:
            self.windows = self.windows[:limit_samples]

    def __len__(self) -> int:
        return len(self.windows)

    def _normalize(self, data: np.ndarray, feature: str) -> np.ndarray:
        mean = self.normalization_stats.get(f"{feature}_mean")
        std = self.normalization_stats.get(f"{feature}_std")
        if mean is None or std is None:
            return data
        return (data - np.asarray(mean)) / np.asarray(std)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        window = self.windows[index]
        past = np.nan_to_num(window["past_joints_3d"].copy(), nan=0.0)
        future = np.nan_to_num(window["future_joints_3d"].copy(), nan=0.0)
        valid = window["valid_mask"].copy()

        for person_index in range(past.shape[0]):
            if not valid[person_index]:
                continue
            root_height = np.concatenate(
                [past[person_index, :, 0, 2], future[person_index, :, 0, 2]]
            )
            if root_height.min() < 0.0 or root_height.max() > 3.0:
                valid[person_index] = False

        num_people, past_frames, num_joints, coord_dim = past.shape
        if num_people > self.max_people:
            past = past[:self.max_people]
            future = future[:self.max_people]
            valid = valid[:self.max_people]
        elif num_people < self.max_people:
            missing = self.max_people - num_people
            past = np.concatenate(
                [past, np.zeros((missing, past_frames, num_joints, coord_dim))], axis=0
            )
            future = np.concatenate(
                [future, np.zeros((missing, future.shape[1], num_joints, coord_dim))],
                axis=0,
            )
            valid = np.concatenate([valid, np.zeros(missing, dtype=bool)])

        if self.normalize:
            past = self._normalize(
                past.reshape(self.max_people, past_frames, -1), "joints_3d"
            ).reshape(self.max_people, past_frames, num_joints, coord_dim)
            future_frames = future.shape[1]
            future = self._normalize(
                future.reshape(self.max_people, future_frames, -1), "joints_3d"
            ).reshape(self.max_people, future_frames, num_joints, coord_dim)

        return {
            "past_joints_3d": torch.from_numpy(past).float(),
            "future_joints_3d": torch.from_numpy(future).float(),
            "valid_mask": torch.from_numpy(valid).bool(),
        }


def get_dataloaders(
    processed_dir: Path,
    batch_size: int = 32,
    limit_samples: Optional[int] = None,
    normalize: bool = False,
    num_workers: int = 4,
):
    metadata_path = processed_dir / "metadata.json"
    common = {
        "metadata_path": metadata_path,
        "normalize": normalize,
        "limit_samples": limit_samples,
    }
    train_dataset = WorldPoseDataset(processed_dir / "train_data.npz", **common)
    val_dataset = WorldPoseDataset(processed_dir / "test_data.npz", **common)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_dataset, shuffle=True, **loader_options),
        DataLoader(val_dataset, shuffle=False, **loader_options),
    )

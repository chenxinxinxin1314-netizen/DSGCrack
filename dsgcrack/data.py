"""Image/mask preprocessing and fixed split ordering used by the full method."""
import json
import random
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".ppm", ".tif", ".tiff"}


def image_paths(directory):
    directory = Path(directory)
    if directory.is_file():
        paths = [directory]
    else:
        paths = sorted(p for p in directory.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError(f"No images found: {directory}")
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError("Image stems must be unique to prevent output files from colliding.")
    return paths


def read_rgb(path):
    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None:
        raise ValueError(f"Cannot read image: {path}")
    if value.ndim == 2:
        value = cv2.cvtColor(value, cv2.COLOR_GRAY2RGB)
    elif value.shape[2] == 4:
        value = cv2.cvtColor(value, cv2.COLOR_BGRA2RGB)
    else:
        value = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
    if value.dtype != np.uint8:
        raise ValueError("The released preprocessing expects 8-bit input images.")
    return cv2.resize(value, (512, 512), interpolation=cv2.INTER_CUBIC)


def to_tensor(image):
    value = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255.)
    return value.sub_(.5).div_(.5)


class CrackDataset(Dataset):
    def __init__(self, root, dataset, split, split_file=None, augmentation=False):
        self.root = Path(root)
        self.dataset = dataset
        self.split = split
        self.augmentation = augmentation and split == "train"
        folder = self.root/f"{split}_img"
        actual = image_paths(folder)
        if split_file:
            names = json.loads(Path(split_file).read_text())[split]
            if len(names) != len(set(names)) or any(Path(n).name != n for n in names):
                raise ValueError("Split manifests must contain unique image basenames.")
            if set(names) != {p.name for p in actual}:
                raise ValueError(f"{split} images differ from the supplied split manifest.")
            self.paths = [folder/name for name in names]
        else:
            self.paths = actual
        self.labels = []
        for path in self.paths:
            stem = path.stem
            name = f"target-{stem.split('-')[-1]}.png" if dataset == "CamCrack789" else stem+".png"
            label = self.root/f"{split}_lab"/name
            if not label.is_file():
                raise FileNotFoundError(f"Missing mask: {label}")
            self.labels.append(label)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = read_rgb(self.paths[index])
        label = cv2.imread(str(self.labels[index]), cv2.IMREAD_UNCHANGED)
        if label is None:
            raise ValueError(f"Cannot read mask: {self.labels[index]}")
        if label.ndim == 3:
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        # Preserve the original interpolation + binarization for reproducibility.
        label = cv2.resize(label, (512, 512), interpolation=cv2.INTER_CUBIC)
        label = (label > 127).astype(np.uint8)
        if self.augmentation:
            horizontal, vertical, turns = random.random() < .5, random.random() < .5, random.randrange(4)
            if horizontal:
                image, label = np.flip(image, 1), np.flip(label, 1)
            if vertical:
                image, label = np.flip(image, 0), np.flip(label, 0)
            image, label = np.rot90(image, turns), np.rot90(label, turns)
        return {"image": to_tensor(image), "label": torch.from_numpy(label.copy()).float()[None],
                "name": self.paths[index].stem}


class ImageDataset(Dataset):
    def __init__(self, source):
        self.paths = image_paths(source)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return {"image": to_tensor(read_rgb(self.paths[index])), "name": self.paths[index].stem}

"""Shared CLI utilities; no workstation paths or device assignments are embedded."""
import json
import random
from pathlib import Path
import cv2
import numpy as np
import torch

DATASETS = ("CamCrack789", "CrackMap", "DeepCrack", "Crack500")
ROOT = Path(__file__).resolve().parents[1]


def configure(device, threads=1, seed=42):
    if threads < 1:
        raise ValueError("threads must be positive")
    cv2.setNumThreads(0)
    torch.set_num_threads(threads)
    if threads == 1:
        torch.set_num_interop_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Training and full-model inference require an NVIDIA CUDA GPU.")
    return target


def split_file_for(dataset, value):
    if value == "none":
        return None
    return ROOT/"splits"/f"{dataset}.json" if value is None else Path(value)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    tmp.replace(path)


def save_checkpoint(path, state):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Cannot save image: {path}")

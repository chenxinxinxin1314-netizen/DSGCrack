"""Evaluate all images in a held-out split; legacy and sigmoid metrics stay separate."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from dsgcrack import build_model
from dsgcrack.checkpoint import load_weights
from dsgcrack.data import CrackDataset
from dsgcrack.metrics import LegacyMetrics, ProbabilityMetrics, legacy_image
from dsgcrack.runtime import DATASETS, configure, split_file_for, write_json, save_image


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--split-file")
    parser.add_argument("--protocol", choices=["legacy", "sigmoid", "both"], default="both")
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--calibrate-on-val", action="store_true",
                        help="Choose a pooled-pixel-F1 sigmoid threshold on validation only")
    parser.add_argument("--legacy-checkpoint", action="store_true",
                        help="Allow the trusted original experiment checkpoint format")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must lie in [0,1]")
    if args.calibrate_on_val and args.split != "test":
        parser.error("--calibrate-on-val requires --split test")
    device = configure(args.device, args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    model = build_model().to(device)
    load_weights(model, args.checkpoint, legacy=args.legacy_checkpoint)
    model.eval()
    manifest = split_file_for(args.dataset, args.split_file)
    threshold = args.threshold
    if args.calibrate_on_val:
        val = CrackDataset(args.data_root, args.dataset, "val", manifest)
        accumulator = ProbabilityMetrics()
        with torch.no_grad():
            for batch in DataLoader(val, batch_size=1, shuffle=False):
                probability = model(batch["image"].to(device)).sigmoid()[0, 0].cpu().numpy()
                accumulator.update(probability, batch["label"][0, 0].numpy() > 0)
        threshold = accumulator.best_threshold()
        write_json(args.output/"threshold.json", {"threshold": threshold,
                   "selection_split": "val", "images": len(val), "objective": "pooled pixel F1",
                   "threshold_grid": "0.00:0.01:1.00"})
    data = CrackDataset(args.data_root, args.dataset, args.split, manifest)
    old, probability_metrics = LegacyMetrics(), ProbabilityMetrics(threshold)
    names = []
    with torch.no_grad():
        for batch in DataLoader(data, batch_size=1, shuffle=False):
            logits = model(batch["image"].to(device))[0, 0]
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite prediction")
            probability = logits.sigmoid().cpu().numpy()
            truth = batch["label"][0, 0].numpy() > 0
            name = batch["name"][0]
            names.append(name)
            if args.protocol in ("sigmoid", "both"):
                probability_metrics.update(probability, truth)
                for suffix, value in [
                    ("probability", np.rint(probability*65535).astype(np.uint16)),
                    ("mask", (probability > threshold).astype(np.uint8)*255),
                    ("label", truth.astype(np.uint8)*255)]:
                    save_image(args.output/"predictions"/f"{name}_{suffix}.png", value)
            if args.protocol in ("legacy", "both"):
                score = legacy_image(logits.cpu().numpy())
                old.update(score, truth)
                save_image(args.output/"legacy"/f"{name}_pre.png", score)
                save_image(args.output/"legacy"/f"{name}_lab.png", truth.astype(np.uint8)*255)
    if len(names) != len(data) or len(set(names)) != len(data):
        raise RuntimeError("Incomplete or repeated evaluation")
    result = {"dataset": args.dataset, "split": args.split, "images": len(data),
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "sigmoid_threshold_selection": "val" if args.calibrate_on_val else "user-specified",
              "sigmoid_threshold": threshold, "test_time_augmentation": False}
    if args.protocol in ("legacy", "both"):
        result["legacy"] = old.result()
    if args.protocol in ("sigmoid", "both"):
        result["sigmoid"] = probability_metrics.result()
    write_json(args.output/"metrics.json", result)
    write_json(args.output/"evaluation_files.json", names)
    print(result, flush=True)


if __name__ == "__main__":
    main()

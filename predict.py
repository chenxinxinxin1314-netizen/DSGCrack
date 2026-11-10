"""Predict each image independently; labels are not required."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from dsgcrack import build_model
from dsgcrack.checkpoint import load_weights
from dsgcrack.data import ImageDataset
from dsgcrack.runtime import configure, save_image, write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", type=Path, required=True, help="One image or an image folder")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--legacy-checkpoint", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must lie in [0,1]")
    device = configure(args.device, args.threads)
    model = build_model().to(device)
    load_weights(model, args.checkpoint, legacy=args.legacy_checkpoint)
    model.eval()
    data = ImageDataset(args.input)
    with torch.no_grad():
        for batch in DataLoader(data, batch_size=1, shuffle=False):
            probability = model(batch["image"].to(device)).sigmoid()[0, 0].cpu().numpy()
            if not np.isfinite(probability).all():
                raise FloatingPointError("Nonfinite probability map")
            name = batch["name"][0]
            save_image(args.output/f"{name}_probability.png", np.rint(probability*65535).astype(np.uint16))
            save_image(args.output/f"{name}_mask.png", (probability > args.threshold).astype(np.uint8)*255)
    write_json(args.output/"prediction_config.json", {"images": len(data), "threshold": args.threshold,
               "probability_encoding": "uint16 PNG; probability = pixel/65535", "output_size": [512, 512]})
    print(f"Saved independent predictions for {len(data)} images.", flush=True)


if __name__ == "__main__":
    main()

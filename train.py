"""Train the paper-default DSGCrack model; checkpoint selection uses validation only."""
import argparse
import fcntl
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from mmengine.optim.scheduler.lr_scheduler import PolyLR
from dsgcrack import build_model
from dsgcrack.checkpoint import load_weights, canonical_config
from dsgcrack.data import CrackDataset
from dsgcrack.loss import SegmentationLoss
from dsgcrack.metrics import LegacyMetrics, legacy_image
from dsgcrack.runtime import DATASETS, ROOT, configure, split_file_for, write_json, save_checkpoint


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--split-file", help="Override fixed manifest; 'none' uses sorted image filenames")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--resume", type=Path, help="A public-format checkpoint from this training run")
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()
    config = canonical_config(json.loads((args.config or ROOT/"configs"/f"{args.dataset}.json").read_text()))
    for name, value in [("epochs", args.epochs), ("batch_size", args.batch_size), ("seed", args.seed)]:
        if value is not None:
            config[name] = value
    if config["dataset"] != args.dataset or config["architecture"] != "dsgcrack_v1":
        raise ValueError("Config does not match the requested dataset/model")
    if (config["load_width"], config["load_height"]) != (512, 512):
        raise ValueError("The released full model uses 512x512 inputs")
    if (config["BCELoss_ratio"], config["DiceLoss_ratio"]) != (.87, .13):
        raise ValueError("The released full-model loss uses BCE=0.87 and Dice=0.13")
    if config["epochs"] < 1 or config["batch_size"] < 1 or args.workers < 0 or args.log_every < 1:
        raise ValueError("Invalid epoch, batch, worker or logging setting")
    device = configure(args.device, args.threads, config["seed"])
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/"training.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config_file = args.output/"config.json"
    if config_file.exists() and canonical_config(json.loads(config_file.read_text())) != config:
        raise ValueError("Output contains another training configuration; use a new directory")
    write_json(config_file, config)
    model = build_model().to(device)
    criterion = SegmentationLoss().to(device)
    manifest = split_file_for(args.dataset, args.split_file)
    train_data = CrackDataset(args.data_root, args.dataset, "train", manifest,
                              augmentation=config["train_augmentation"])
    val_data = CrackDataset(args.data_root, args.dataset, "val", manifest)
    data_names = {split: [p.name for p in data.paths]
                  for split, data in [("train", train_data), ("val", val_data)]}
    write_json(args.output/"file_order.json", data_names)
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True,
                              num_workers=args.workers)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=args.workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                 weight_decay=config["weight_decay"])
    scheduler = PolyLR(optimizer, eta_min=config["min_lr"], begin=0, end=config["epochs"])
    history, best, start = [], None, 0
    if args.resume:
        state = load_weights(model, args.resume)
        if state["config"] != config or state["file_order"] != data_names:
            raise ValueError("Resume configuration or data ordering differs")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, best, start = state["history"], state["best"], state["epoch"]
        torch.set_rng_state(state["rng"]["torch"])
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        n = state["rng"]["numpy"]
        np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
        random.setstate(state["rng"]["python"])
    elif (args.output/"last.pt").exists():
        raise ValueError("A checkpoint already exists; use --resume or a new output directory")
    print(json.dumps({"dataset": args.dataset, "train": len(train_data), "val": len(val_data),
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "start_epoch": start, "epochs": config["epochs"]}), flush=True)
    try:
        for epoch in range(start, config["epochs"]):
            begun = time.monotonic()
            model.train()
            total_loss, seen = 0., 0
            lr_used = optimizer.param_groups[0]["lr"]
            for step, batch in enumerate(train_loader, 1):
                logits = model(batch["image"].to(device))
                loss = criterion(logits, batch["label"].to(device))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                seen += len(batch["image"])
                total_loss += loss.item()*len(batch["image"])
                if step == 1 or step % args.log_every == 0 or step == len(train_loader):
                    progress = {"status": "training", "epoch": epoch+1,
                                "total_epochs": config["epochs"], "step": step,
                                "total_steps": len(train_loader), "loss": total_loss/seen}
                    write_json(args.output/"status.json", progress)
                    print(json.dumps(progress), flush=True)
            if seen != len(train_data):
                raise RuntimeError("Incomplete training epoch")
            scheduler.step()
            model.eval()
            metrics = LegacyMetrics()
            with torch.no_grad():
                for batch in val_loader:
                    logits = model(batch["image"].to(device))[0, 0].cpu().numpy()
                    metrics.update(legacy_image(logits), batch["label"][0, 0].numpy() > 0)
            values = metrics.result()
            record = {"epoch": epoch+1, "train_loss": total_loss/seen, "lr": lr_used,
                      "seconds": time.monotonic()-begun, "validation": values}
            improved = best is None or values["mIoU"] > best["validation"]["mIoU"]
            if improved:
                best = record.copy()
            history.append(record)
            for name, value in model.state_dict().items():
                if value.is_floating_point() and not torch.isfinite(value).all():
                    raise FloatingPointError(f"Nonfinite model state: {name}")
            numpy_state = np.random.get_state()
            state = {"format_version": 1, "model": model.state_dict(), "config": config,
                     "epoch": epoch+1, "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "history": history, "best": best,
                     "file_order": data_names,
                     "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                             "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
                             "python": random.getstate()}}
            save_checkpoint(args.output/"last.pt", state)
            if improved:
                save_checkpoint(args.output/"best.pt", state)
            write_json(args.output/"history.json", history)
            write_json(args.output/"best_validation.json", best)
            write_json(args.output/"status.json", {"status": "epoch_complete", "epoch": epoch+1,
                       "total_epochs": config["epochs"], "validation": values})
            print(json.dumps(record), flush=True)
        write_json(args.output/"status.json", {"status": "completed", "epoch": config["epochs"],
                   "total_epochs": config["epochs"], "best_validation_epoch": best["epoch"]})
    except BaseException as error:
        write_json(args.output/"status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt)
                   else "failed", "completed_epochs": len(history), "error": str(error)})
        raise


if __name__ == "__main__":
    main()

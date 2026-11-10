"""Export a model-only public checkpoint without optimizer, RNG or workstation paths."""
import argparse
from pathlib import Path
from dsgcrack import build_model
from dsgcrack.checkpoint import read_checkpoint, validate_state_dict
from dsgcrack.model import ARCHITECTURE
from dsgcrack.runtime import save_checkpoint


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy", action="store_true", help="Input is a trusted original experiment checkpoint")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Export must not overwrite the source checkpoint")
    state, _ = read_checkpoint(args.input, legacy=args.legacy)
    model = build_model()
    validate_state_dict(model, state)
    model.load_state_dict(state, strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.output, {"format_version": 1, "config": {"architecture": ARCHITECTURE,
                    "image_size": 512}, "model": {k: v.detach().cpu() for k, v in model.state_dict().items()}})
    print(f"Saved model-only checkpoint: {args.output}", flush=True)


if __name__ == "__main__":
    main()

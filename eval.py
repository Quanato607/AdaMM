#!/usr/bin/env python3
"""Evaluate an AdaMM student checkpoint without initializing training."""

import argparse
import csv
from collections.abc import Mapping
from pathlib import Path


# Channel order follows config['modalities'], as in train_adamm_divide.py.
MASKS = (
    "0001", "0100", "0010", "1000", "0101", "0110", "1010", "0011",
    "1001", "1100", "1110", "1011", "1101", "0111", "1111",
)
REGIONS = ("WT", "TC", "ET")
METRICS = ("Dice", "HD95", "Sensitivity", "IoU")


def modality_mask(value):
    if value not in MASKS:
        raise argparse.ArgumentTypeError("mask must be four binary digits, excluding 0000")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parent / "configs/brats.yml")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="training checkpoint containing model_missing, or a student state_dict")
    parser.add_argument("--data-root", type=Path,
                        help="override path_to_data; patients are read only from its test/ folder")
    parser.add_argument("--output", type=Path, default=Path("eval_results.csv"))
    parser.add_argument("--device", default="auto", help="auto (CUDA if available, else CPU), cpu, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--masks", nargs="+", type=modality_mask, default=list(MASKS),
                        help="modality bit masks in config channel order; default: all 15")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("--batch-size must be positive and --num-workers must be nonnegative")
    return args


def load_config(path, data_root=None):
    import yaml

    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML mapping")
    if config.get("dataset") not in ("brats", "fets"):
        raise ValueError("dataset must be brats or fets")
    modalities = config.get("modalities")
    expected = ["t2f", "t1n", "t1c", "t2w"] if config["dataset"] == "brats" else ["flair", "t1", "t1ce", "t2"]
    if modalities != expected:
        raise ValueError(f"modalities must match the training channel order: {expected}")
    shape = config.get("inputshape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 3 or any(
        not isinstance(size, int) or size < 8 or size % 8 for size in shape
    ):
        raise ValueError("inputshape must contain three positive multiples of 8")
    if data_root is not None:
        config["path_to_data"] = str(data_root)
    if not config.get("path_to_data"):
        raise ValueError("set path_to_data in the config or pass --data-root")
    return config


def make_eval_loader(config, batch_size=1, num_workers=0, pin_memory=False):
    from torch.utils.data import DataLoader
    from data import Brats2018_divide

    test_dir = Path(config["path_to_data"]).expanduser() / "test"
    prefix = "BraTS" if config["dataset"] == "brats" else "FeTS"
    patients = sorted(str(path) for path in test_dir.glob(prefix + "*") if path.is_dir())
    if not patients:
        raise ValueError(f"no {prefix} patient directories found in {test_dir}")
    dataset = Brats2018_divide(
        patients, crop_size=tuple(config["inputshape"]), modes=config["modalities"],
        train=False, dataset=config["dataset"],
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=pin_memory)


def load_student(checkpoint_path, input_shape, device):
    import torch
    from models.AdaMMKD import AdaMMKD

    # Training checkpoints also contain optimizer state and NumPy metric values.
    # Load only trusted checkpoints, using the same serialization as training.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must contain a student state_dict")
    state = checkpoint.get("model_missing", checkpoint)
    if not isinstance(state, Mapping) or not state or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise ValueError("checkpoint must contain model_missing or be a student state_dict")
    if all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    model = AdaMMKD(tuple(input_shape), in_channels=4, out_channels=4, init_channels=16, p=0.2)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def region_metrics(prediction, target):
    """Per-patient metrics for [NET, SNFH, ET] probabilities, ordered WT/TC/ET."""
    import numpy as np
    from binary import hd95

    pred = np.asarray(prediction[:3]) > 0.5
    truth = np.asarray(target[:3]) > 0.5
    pred_regions = (pred.any(axis=0), pred[0] | pred[2], pred[2])
    true_regions = (truth.any(axis=0), truth[0] | truth[2], truth[2])
    scores = {}
    for region, pred_mask, true_mask in zip(REGIONS, pred_regions, true_regions):
        intersection = int(np.count_nonzero(pred_mask & true_mask))
        pred_count, true_count = int(pred_mask.sum()), int(true_mask.sum())
        # Match the repository's smooth=1.0 overlap metrics and HD95 empty-mask policy.
        scores[f"Dice {region}"] = (2 * intersection + 1) / (pred_count + true_count + 1)
        scores[f"HD95 {region}"] = float(hd95(pred_mask, true_mask))
        scores[f"Sensitivity {region}"] = (intersection + 1) / (true_count + 1)
        scores[f"IoU {region}"] = (intersection + 1) / (pred_count + true_count - intersection + 1)
    return scores


def evaluate(model, loader, masks, device, modalities):
    import numpy as np
    import torch

    model.eval()
    rows = []
    with torch.inference_mode():
        for code in masks:
            mask = [bit == "1" for bit in code]
            history = []
            for images, targets, _ in loader:
                images = images.to(device, non_blocking=True)
                mask_tensor = images.new_tensor(mask).view(1, 4, 1, 1, 1)
                segmentation, _, _, _, _, presence = model(images * mask_tensor, mask)
                gate = (presence > 0.5).view(-1, 3, 1, 1, 1)
                predictions = (segmentation[:, :3] * gate).cpu().numpy()
                targets = targets.cpu().numpy()
                # Compute case metrics before aggregating, including a partial final batch.
                for prediction, target in zip(predictions, targets):
                    history.append(region_metrics(prediction, target))
            if not history:
                raise ValueError("evaluation loader is empty")
            row = {"mask": code, "modalities": "+".join(
                name for name, present in zip(modalities, mask) if present
            ), "cases": len(history)}
            for metric in METRICS:
                for region in REGIONS:
                    key = f"{metric} {region}"
                    values = [case[key] for case in history]
                    row[key + " mean"] = float(np.mean(values))
                    row[key + " std"] = float(np.std(values))
            rows.append(row)
            dice = "/".join(f"{row[f'Dice {region} mean']:.4f}" for region in REGIONS)
            print(f"{code} ({row['modalities']}): {len(history)} cases, Dice WT/TC/ET = {dice}", flush=True)
    return rows


def write_results(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = parse_args(argv)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    config = load_config(args.config, args.data_root)

    import random
    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu")
    loader = make_eval_loader(config, args.batch_size, args.num_workers, device.type == "cuda")
    model = load_student(args.checkpoint, config["inputshape"], device)
    print(f"Checkpoint: {args.checkpoint.resolve()} | Device: {device} | Dataset: {config['dataset']}")
    rows = evaluate(model, loader, args.masks, device, config["modalities"])
    write_results(args.output, rows)
    print(f"Results saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()

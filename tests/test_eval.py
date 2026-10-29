"""Regression tests for the standalone AdaMM evaluation entry point."""

import csv
import math
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import eval as eval_entry  # noqa: E402


def test_help_does_not_import_training_dependencies():
    result = subprocess.run(
        [sys.executable, "-S", str(ROOT / "eval.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--checkpoint" in result.stdout
    assert "train_adamm_divide" not in result.stdout


def test_missing_checkpoint_fails_before_data_or_model_initialization(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        eval_entry.main(
            [
                "--checkpoint",
                str(tmp_path / "missing.pth"),
                "--config",
                str(ROOT / "configs/brats.yml"),
            ]
        )


def test_region_metrics_use_wt_tc_et_and_ground_truth_sensitivity_denominator():
    np = pytest.importorskip("numpy")

    # Channels are [NET, SNFH, ET, background].  The prediction deliberately
    # misses two target voxels and adds one false positive.
    prediction = np.zeros((4, 2, 2, 2), dtype=np.float32)
    target = np.zeros_like(prediction)
    prediction[0].flat[[0, 4]] = 1.0
    prediction[1].flat[2] = 1.0
    prediction[2].flat[4] = 1.0
    target[0].flat[[0, 1]] = 1.0
    target[1].flat[2] = 1.0
    target[2].flat[3] = 1.0

    scores = eval_entry.region_metrics(prediction, target)
    # WT: prediction {0,2,4}, target {0,1,2,3}; intersection=2.
    assert scores["Dice WT"] == pytest.approx(5 / 8)
    assert scores["Sensitivity WT"] == pytest.approx(3 / 5)
    # TC is NET|ET: prediction {0,4}, target {0,1,3}; intersection=1.
    assert scores["Dice TC"] == pytest.approx(3 / 6)
    assert scores["Sensitivity TC"] == pytest.approx(2 / 4)
    # ET has no overlap, but smoothing remains the repository's smooth=1 value.
    assert scores["Dice ET"] == pytest.approx(1 / 3)
    assert scores["Sensitivity ET"] == pytest.approx(1 / 2)
    assert scores["IoU ET"] == pytest.approx(1 / 3)
    assert scores["HD95 WT"] > 0


def test_evaluate_is_case_wise_for_partial_batches_and_keeps_model_frozen():
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")

    class ProbeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm3d(4)
            self.weight = torch.nn.Parameter(torch.tensor(2.0))
            self.calls = []

        def forward(self, images, mask):
            self.calls.append((tuple(mask), torch.is_grad_enabled(), images.detach().clone()))
            # If evaluate() forgets eval(), BN would update these statistics.
            self.bn(images)
            batch, _, depth, height, width = images.shape
            segmentation = torch.full(
                (batch, 4, depth, height, width), 0.9, device=images.device
            )
            # Keep only SNFH.  WT should be present, while TC and ET should be
            # empty; this makes an incorrect classification gate observable.
            presence = torch.tensor(
                [[0.0, 1.0, 0.0]] * batch, device=images.device
            )
            return segmentation, None, None, None, None, presence

    model = ProbeModel()
    images = torch.ones(2, 4, 2, 2, 2)
    targets = torch.zeros(2, 4, 2, 2, 2)
    targets[0, 1] = 1.0
    # The second case has no SNFH, so its WT score differs from the first case.
    # The final one-case batch has four of eight target voxels.
    images_last = torch.full((1, 4, 2, 2, 2), 2.0)
    targets_last = torch.zeros_like(targets[:1])
    targets_last[0, 1].flatten()[:4] = 1.0
    loader = [
        (images, targets, torch.ones(2, 3)),
        (images_last, targets_last, torch.ones(1, 3)),
    ]
    before_params = [parameter.detach().clone() for parameter in model.parameters()]
    before_mean = model.bn.running_mean.detach().clone()

    rows = eval_entry.evaluate(
        model,
        loader,
        ["1010"],
        torch.device("cpu"),
        ["m0", "m1", "m2", "m3"],
    )

    assert len(rows) == 1
    assert rows[0]["cases"] == 3  # batch of 2 plus a final batch of 1
    assert rows[0]["mask"] == "1010"
    assert rows[0]["modalities"] == "m0+m2"
    expected_wt_dice = np.asarray([1.0, 1.0 / 9.0, 9.0 / 13.0])
    assert rows[0]["Dice WT mean"] == pytest.approx(expected_wt_dice.mean())
    assert rows[0]["Dice WT std"] == pytest.approx(expected_wt_dice.std())
    assert rows[0]["Dice TC mean"] == pytest.approx(1.0)
    assert rows[0]["Dice TC std"] == pytest.approx(0.0)
    assert rows[0]["Dice ET mean"] == pytest.approx(1.0)
    assert rows[0]["Dice ET std"] == pytest.approx(0.0)
    # Averaging the two batches would give a different result from averaging
    # the three cases, which catches accidental batch-level aggregation.
    batch_level_mean = ((expected_wt_dice[0] + expected_wt_dice[1]) / 2 + expected_wt_dice[2]) / 2
    assert rows[0]["Dice WT mean"] != pytest.approx(batch_level_mean)
    assert all(not grad_enabled for _, grad_enabled, _ in model.calls)
    assert torch.equal(model.bn.running_mean, before_mean)
    assert all(torch.equal(before, after) for before, after in zip(
        before_params, model.parameters()
    ))
    assert model.training is False
    # Input channel masking is applied before the model call.
    assert torch.all(model.calls[0][2][:, 0] == 1)
    assert torch.all(model.calls[0][2][:, 1] == 0)
    assert torch.all(model.calls[0][2][:, 2] == 1)
    assert torch.all(model.calls[0][2][:, 3] == 0)


@pytest.fixture(scope="module")
def tiny_student_checkpoint(tmp_path_factory):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    from models.AdaMMKD import AdaMMKD

    torch.manual_seed(7)
    source = AdaMMKD((8, 8, 8), in_channels=4, out_channels=4, init_channels=16, p=0.2)
    state = {"module." + key: value.detach().clone() for key, value in source.state_dict().items()}
    checkpoint_path = tmp_path_factory.mktemp("tiny-student") / "student.pth"
    torch.save({"model_missing": state, "dice": np.float64(0.25)}, checkpoint_path)
    loaded = eval_entry.load_student(checkpoint_path, (8, 8, 8), torch.device("cpu"))
    return checkpoint_path, source, loaded


def test_real_student_checkpoint_loads_and_runs_full_and_missing_masks(tiny_student_checkpoint):
    torch = pytest.importorskip("torch")
    checkpoint_path, source, loaded = tiny_student_checkpoint

    assert loaded.training is False
    for key, value in source.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])

    volume = torch.randn(1, 4, 8, 8, 8)
    with torch.inference_mode():
        full = loaded(volume, [1, 1, 1, 1])
        missing = loaded(volume * torch.tensor([1, 0, 1, 0]).view(1, 4, 1, 1, 1), [1, 0, 1, 0])
    assert full[0].shape == missing[0].shape == (1, 4, 8, 8, 8)
    assert full[-1].shape == missing[-1].shape == (1, 3)


def _write_brats_case(root, shape=(10, 10, 10)):
    nib = pytest.importorskip("nibabel")
    np = pytest.importorskip("numpy")
    patient = root / "test" / "BraTS_001"
    patient.mkdir(parents=True)
    name = patient.name
    modes = ("t2f", "t1n", "t1c", "t2w")
    for offset, mode in enumerate(modes):
        volume = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + offset
        nib.save(nib.Nifti1Image(volume, np.eye(4)), patient / f"{name}-{mode}.nii.gz")
    label = np.zeros(shape, dtype=np.int16)
    label[2:4, 2:4, 2:4] = 1
    label[4:6, 4:6, 4:6] = 2
    label[6:8, 6:8, 6:8] = 3
    nib.save(nib.Nifti1Image(label, np.eye(4)), patient / f"{name}-seg.nii.gz")
    return patient


def test_eval_loader_reads_test_only_data_without_train_directory(tmp_path):
    torch = pytest.importorskip("torch")
    _write_brats_case(tmp_path)
    config = {
        "dataset": "brats",
        "modalities": ["t2f", "t1n", "t1c", "t2w"],
        "inputshape": [8, 8, 8],
        "path_to_data": str(tmp_path),
    }
    loader = eval_entry.make_eval_loader(config, batch_size=1, num_workers=0)
    assert len(loader.dataset) == 1
    images, target, cls = next(iter(loader))
    assert images.shape == target.shape == (1, 4, 8, 8, 8)
    assert cls.shape == (1, 3)
    assert not (tmp_path / "train").exists()


def test_eval_loader_rejects_empty_test_directory(tmp_path):
    config = {
        "dataset": "brats",
        "modalities": ["t2f", "t1n", "t1c", "t2w"],
        "inputshape": [8, 8, 8],
        "path_to_data": str(tmp_path),
    }
    with pytest.raises(ValueError, match="no BraTS patient directories"):
        eval_entry.make_eval_loader(config)


def test_main_end_to_end_writes_finite_rows_for_test_only_data(
    tmp_path, tiny_student_checkpoint
):
    pytest.importorskip("torch")
    yaml = pytest.importorskip("yaml")
    checkpoint_path, _, _ = tiny_student_checkpoint
    _write_brats_case(tmp_path)
    config_path = tmp_path / "eval.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": "brats",
                "modalities": ["t2f", "t1n", "t1c", "t2w"],
                "inputshape": [8, 8, 8],
                "path_to_data": str(tmp_path),
            }
        )
    )
    output_path = tmp_path / "results.csv"

    eval_entry.main(
        [
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(output_path),
            "--device",
            "cpu",
            "--masks",
            "0001",
            "1111",
        ]
    )

    with output_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert [row["mask"] for row in rows] == ["0001", "1111"]
    assert all(row["cases"] == "1" for row in rows)
    assert not (tmp_path / "train").exists()
    for row in rows:
        for name, value in row.items():
            if name in {"mask", "modalities", "cases"}:
                continue
            assert math.isfinite(float(value))

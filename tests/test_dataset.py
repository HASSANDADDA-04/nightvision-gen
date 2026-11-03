"""Unit tests for dataset loading, partitioning, and patch sampling."""

from pathlib import Path

import torch
from PIL import Image

from nightvision.data.dataset import SyntheticPairDataset
from nightvision.data.patch_extractor import PatchExtractor
from nightvision.data.split_manager import LOLv2SplitManager


def test_synthetic_pair_dataset():
    """Verify synthetic paired dataset generates valid normalized tensors."""
    ds = SyntheticPairDataset(length=4, image_size=128, compute_retinex=True)
    assert len(ds) == 4

    item = ds[0]
    assert "low" in item and "normal" in item and "illumination" in item
    assert item["low"].shape == (3, 128, 128)
    assert item["normal"].shape == (3, 128, 128)
    assert item["illumination"].shape == (1, 128, 128)

    # Values in [-1, 1]
    assert item["low"].min() >= -1.0 - 1e-4
    assert item["low"].max() <= 1.0 + 1e-4
    assert item["normal"].min() >= -1.0 - 1e-4
    assert item["normal"].max() <= 1.0 + 1e-4


def test_patch_extractor_shapes():
    """Verify patch extraction and perfect canvas reconstruction."""
    extractor = PatchExtractor(patch_size=128, stride_train=64, stride_eval=64)

    # Create dummy tensor 256x256
    img = torch.rand((3, 256, 256))
    patches, coords, orig_shape = extractor.decompose_image_for_eval(img)

    assert patches.shape[1:] == (3, 128, 128)
    assert len(coords) == patches.shape[0]

    # Reconstruct
    reconstructed = extractor.reconstruct_image(patches, coords, orig_shape)
    assert reconstructed.shape == (3, 256, 256)
    # Check reconstruction fidelity on flat or smooth input
    assert torch.allclose(img, reconstructed, atol=1e-3)


def test_split_manager_pairing(tmp_path):
    """Verify discover_pairs correctly matches corresponding files."""
    low_dir = tmp_path / "Low"
    norm_dir = tmp_path / "Normal"
    low_dir.mkdir()
    norm_dir.mkdir()

    # Create 5 dummy image files
    for i in range(5):
        Image.new("RGB", (64, 64), color="black").save(low_dir / f"{i}.png")
        Image.new("RGB", (64, 64), color="white").save(norm_dir / f"{i}.png")

    # One unmatched file
    Image.new("RGB", (64, 64), color="black").save(low_dir / "unmatched.png")

    mgr = LOLv2SplitManager(root_dir=tmp_path, val_ratio=0.2, seed=42)
    pairs = mgr.discover_pairs(low_dir, norm_dir)

    assert len(pairs) == 5
    pair_ids = {p.pair_id for p in pairs}
    assert pair_ids == {"0", "1", "2", "3", "4"}


def test_split_manager_pairs_prefixed_names(tmp_path):
    """Match low00001.png with normal00001.png, including same names in both folders."""
    low_dir = tmp_path / "Low"
    norm_dir = tmp_path / "Normal"
    low_dir.mkdir()
    norm_dir.mkdir()

    Image.new("RGB", (8, 8)).save(low_dir / "low00567.png")
    Image.new("RGB", (8, 8)).save(norm_dir / "normal00567.png")
    Image.new("RGB", (8, 8)).save(low_dir / "low00002.png")
    Image.new("RGB", (8, 8)).save(norm_dir / "low00002.png")
    Image.new("RGB", (8, 8)).save(low_dir / "low00003.png")

    mgr = LOLv2SplitManager(root_dir=tmp_path, seed=42)
    pairs = mgr.discover_pairs(low_dir, norm_dir)
    by_id = {p.pair_id: Path(p.normal_path).name for p in pairs}

    assert by_id["low00567"] == "normal00567.png"
    assert by_id["low00002"] == "low00002.png"
    assert "low00003" not in by_id

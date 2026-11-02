"""Dataset discovery and partition splitting module.

Guarantees strict separation between Train, Validation, and Test partitions.
Pre-splits LOL-v2 pairs before patch extraction to avoid any data leakage.
Provides stratified sampling for ExDark across 12 object classes.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar

logger = logging.getLogger(__name__)


@dataclass
class ImagePair:
    """Represents a paired low-light and reference image."""

    pair_id: str
    low_path: str
    normal_path: str
    illumination_path: str | None = None


@dataclass
class ExDarkSample:
    """Represents an annotated ExDark image sample."""

    image_id: str
    image_path: str
    class_name: str
    coco_class_name: str | None
    annotation_path: str | None = None


class LOLv2SplitManager:
    """Manages partition splitting and pairing for the LOL-v2 dataset."""

    def __init__(
        self,
        root_dir: str | Path,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> None:
        """Initialize LOLv2SplitManager.

        Args:
            root_dir: Root directory of LOL-v2 (e.g. data/raw/LOLv2).
            val_ratio: Fraction of training pairs reserved for validation.
            seed: Random seed for deterministic partition splitting.
        """
        self.root_dir = Path(root_dir)
        self.val_ratio = val_ratio
        self.seed = seed

    def discover_pairs(
        self,
        low_dir: str | Path,
        normal_dir: str | Path,
    ) -> list[ImagePair]:
        """Find and match low-light images with their normal-light references.

        Matches are formed by comparing base filenames (e.g. '1.png' <-> '1.png' or
        'low001.png' <-> 'normal001.png').

        Args:
            low_dir: Path to directory containing dark images.
            normal_dir: Path to directory containing normal/reference images.

        Returns:
            List of matched ImagePair instances sorted by pair_id.
        """
        low_dir = Path(low_dir)
        normal_dir = Path(normal_dir)

        if not low_dir.exists():
            logger.warning("Low-light directory does not exist: %s", low_dir)
            return []
        if not normal_dir.exists():
            logger.warning("Normal-light directory does not exist: %s", normal_dir)
            return []

        # Supported image extensions
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

        low_files = {
            f.stem: f for f in low_dir.iterdir() if f.is_file() and f.suffix.lower() in extensions
        }
        normal_files = {
            f.stem: f
            for f in normal_dir.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        }

        matched_pairs: list[ImagePair] = []

        # Attempt exact stem match first
        for stem, low_path in low_files.items():
            if stem in normal_files:
                matched_pairs.append(
                    ImagePair(
                        pair_id=stem,
                        low_path=str(low_path.resolve()),
                        normal_path=str(normal_files[stem].resolve()),
                    )
                )
            else:
                # Secondary heuristic: strip leading 'low' / 'normal' prefixes
                clean_stem = stem.lower().replace("low", "").replace("dark", "").strip("_ -")
                candidate = None
                for n_stem, n_path in normal_files.items():
                    n_clean = n_stem.lower().replace("normal", "").replace("high", "").strip("_ -")
                    if clean_stem == n_clean:
                        candidate = n_path
                        break
                if candidate is not None:
                    matched_pairs.append(
                        ImagePair(
                            pair_id=stem,
                            low_path=str(low_path.resolve()),
                            normal_path=str(candidate.resolve()),
                        )
                    )
                else:
                    logger.debug("Unmatched low-light image: %s", low_path.name)

        matched_pairs.sort(key=lambda p: p.pair_id)
        return matched_pairs

    def create_splits(
        self,
        train_dir: str | Path | None = None,
        test_dir: str | Path | None = None,
    ) -> dict[str, list[ImagePair]]:
        """Create Train, Validation, and Test splits.

        Strictly reserves the 100 test pairs from the Real_captured Test folder.
        Validation is split randomly from Train using the configured seed.

        Args:
            train_dir: Directory containing Real_captured/Train (Low & Normal subdirs).
            test_dir: Directory containing Real_captured/Test (Low & Normal subdirs).

        Returns:
            Dictionary with keys 'train', 'val', 'test', each containing a list of ImagePair.
        """
        if train_dir is None:
            train_dir = self.root_dir / "Real_captured" / "Train"
        else:
            train_dir = Path(train_dir)

        if test_dir is None:
            test_dir = self.root_dir / "Real_captured" / "Test"
        else:
            test_dir = Path(test_dir)

        train_pairs = self.discover_pairs(train_dir / "Low", train_dir / "Normal")
        test_pairs = self.discover_pairs(test_dir / "Low", test_dir / "Normal")

        # Fallback if Real_captured hierarchy is flat in root_dir
        if not train_pairs and not test_pairs:
            flat_pairs = self.discover_pairs(self.root_dir / "Low", self.root_dir / "Normal")
            if flat_pairs:
                logger.info("Found %d flat pairs. Reserving last 100 for test.", len(flat_pairs))
                rng = random.Random(self.seed)
                shuffled = list(flat_pairs)
                rng.shuffle(shuffled)
                test_count = min(100, max(1, int(len(shuffled) * 0.2)))
                test_pairs = shuffled[:test_count]
                train_pairs = shuffled[test_count:]

        # Split train_pairs into train and validation
        rng = random.Random(self.seed)
        shuffled_train = list(train_pairs)
        rng.shuffle(shuffled_train)

        n_val = int(len(shuffled_train) * self.val_ratio)
        val_pairs = shuffled_train[:n_val]
        final_train_pairs = shuffled_train[n_val:]

        logger.info(
            "Splits created: %d train pairs, %d val pairs, %d test pairs",
            len(final_train_pairs),
            len(val_pairs),
            len(test_pairs),
        )

        return {
            "train": final_train_pairs,
            "val": val_pairs,
            "test": test_pairs,
        }

    def save_split_manifest(
        self,
        splits: dict[str, list[ImagePair]],
        manifest_path: str | Path,
    ) -> None:
        """Export split assignments to a JSON manifest for reproducibility."""
        manifest_path = Path(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            split_name: [asdict(pair) for pair in pairs] for split_name, pairs in splits.items()
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Saved split manifest to %s", manifest_path)

    @staticmethod
    def load_split_manifest(manifest_path: str | Path) -> dict[str, list[ImagePair]]:
        """Load split assignments from a previously saved JSON manifest."""
        manifest_path = Path(manifest_path)
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {
            split_name: [ImagePair(**item) for item in items] for split_name, items in data.items()
        }


class ExDarkManager:
    """Manages ExDark dataset images, annotations, and COCO class mapping."""

    CLASS_MAPPING_TO_COCO: ClassVar[dict[str, str | None]] = {
        "Bicycle": "bicycle",
        "Boat": "boat",
        "Bottle": "bottle",
        "Bus": "bus",
        "Car": "car",
        "Cat": "cat",
        "Chair": "chair",
        "Cup": "cup",
        "Dog": "dog",
        "Motorbike": "motorcycle",
        "People": "person",
        "Table": "dining table",
    }

    def __init__(
        self,
        root_dir: str | Path,
        seed: int = 42,
    ) -> None:
        """Initialize ExDarkManager.

        Args:
            root_dir: Root directory containing ExDark class folders or images.
            seed: Random seed for reproducible stratified sampling.
        """
        self.root_dir = Path(root_dir)
        self.seed = seed

    def discover_samples(self) -> list[ExDarkSample]:
        """Discover all images categorized by class folders in ExDark."""
        samples: list[ExDarkSample] = []
        extensions = {".png", ".jpg", ".jpeg", ".bmp"}

        for class_name, coco_class in self.CLASS_MAPPING_TO_COCO.items():
            class_folder = self.root_dir / class_name
            if not class_folder.is_dir():
                # Check case-insensitive
                matches = [
                    p
                    for p in self.root_dir.glob("*")
                    if p.is_dir() and p.name.lower() == class_name.lower()
                ]
                class_folder = matches[0] if matches else None

            if class_folder is not None and class_folder.is_dir():
                for img_file in class_folder.iterdir():
                    if img_file.is_file() and img_file.suffix.lower() in extensions:
                        annot_file = self.root_dir / "Annotations" / f"{img_file.stem}.txt"
                        if not annot_file.exists():
                            # Annotation might be in same folder or subfolder
                            annot_file = class_folder / f"{img_file.stem}.txt"

                        samples.append(
                            ExDarkSample(
                                image_id=img_file.stem,
                                image_path=str(img_file.resolve()),
                                class_name=class_name,
                                coco_class_name=coco_class,
                                annotation_path=str(annot_file.resolve())
                                if annot_file.exists()
                                else None,
                            )
                        )

        return samples

    def get_stratified_subset(
        self,
        total_samples: int = 1000,
    ) -> list[ExDarkSample]:
        """Generate a class-balanced stratified subset of approximately total_samples.

        Args:
            total_samples: Target total number of samples (e.g. 1000).

        Returns:
            List of selected ExDarkSample instances.
        """
        all_samples = self.discover_samples()
        if not all_samples:
            logger.warning("No ExDark samples discovered in %s", self.root_dir)
            return []

        # Group by class
        by_class: dict[str, list[ExDarkSample]] = {}
        for s in all_samples:
            by_class.setdefault(s.class_name, []).append(s)

        num_classes = len(by_class)
        target_per_class = max(1, total_samples // num_classes)

        rng = random.Random(self.seed)
        selected: list[ExDarkSample] = []

        for class_name, items in sorted(by_class.items()):
            shuffled = list(items)
            rng.shuffle(shuffled)
            k = min(len(shuffled), target_per_class)
            selected.extend(shuffled[:k])

        logger.info(
            "Selected %d stratified ExDark samples across %d classes.",
            len(selected),
            num_classes,
        )
        return selected

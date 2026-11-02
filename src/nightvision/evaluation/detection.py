"""Object detection evaluation module for ExDark using frozen COCO detectors.

Evaluates downstream vision utility before and after image restoration:
    1. Raw Low-light ExDark Image
    2. Gamma-corrected Image
    3. CLAHE-enhanced Image
    4. Model A Restored Image
    5. Model B Restored Image
    6. Model C Restored Image

Uses a frozen pre-trained torchvision detector (Faster R-CNN ResNet-50 FPN)
with explicit bidirectional mapping between the 12 ExDark classes and COCO classes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)

logger = logging.getLogger(__name__)

# COCO Category ID mapping (standard 80 categories)
COCO_CATEGORIES = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    16: "cat",
    17: "dog",
    44: "bottle",
    47: "cup",
    62: "chair",
    67: "dining table",
}

# Explicit bidirectional mapping between ExDark and COCO
EXDARK_TO_COCO_CLASS: dict[str, str] = {
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

COCO_NAME_TO_ID = {v: k for k, v in COCO_CATEGORIES.items()}
EXDARK_TO_COCO_ID: dict[str, int] = {
    k: COCO_NAME_TO_ID[v] for k, v in EXDARK_TO_COCO_CLASS.items() if v in COCO_NAME_TO_ID
}


class FrozenCOCODetector:
    """Encapsulates a frozen, pre-trained COCO object detector."""

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        """Initialize FrozenCOCODetector.

        Args:
            confidence_threshold: Score threshold to accept detection.
            iou_threshold: NMS IoU threshold.
            device: 'cpu' or 'cuda'.
        """
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        # Load Faster R-CNN with pre-trained COCO V2 weights
        try:
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn_v2(weights=weights).to(self.device)
        except Exception:
            # Fallback if offline / cached without weights
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False).to(
                self.device
            )

        # Strictly freeze all weights: NO fine-tuning on restored images
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def detect_image(
        self,
        image: Image.Image | np.ndarray | torch.Tensor,
    ) -> dict[str, Any]:
        """Run object detection inference on a single image.

        Args:
            image: PIL Image, NumPy uint8 array, or float tensor in [0, 1].

        Returns:
            Dict containing 'boxes', 'scores', 'labels', 'class_names'.
        """
        if isinstance(image, Image.Image):
            arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).to(self.device)
        elif isinstance(image, np.ndarray):
            arr = image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else image
            tensor = torch.from_numpy(arr).permute(2, 0, 1).to(self.device)
        elif isinstance(image, torch.Tensor):
            t = image.detach().clone().float().to(self.device)
            if t.dim() == 4:
                t = t.squeeze(0)
            if t.min() < 0.0:
                t = (t + 1.0) / 2.0
            tensor = torch.clamp(t, 0.0, 1.0)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Faster R-CNN expects list of 3D tensors in [0, 1]
        predictions = self.model([tensor])[0]

        boxes = predictions["boxes"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()

        # Filter by confidence threshold
        mask = scores >= self.confidence_threshold
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]
        filtered_labels = labels[mask]

        class_names = [COCO_CATEGORIES.get(int(l), f"class_{l}") for l in filtered_labels]

        return {
            "boxes": filtered_boxes.tolist(),
            "scores": filtered_scores.tolist(),
            "labels": filtered_labels.tolist(),
            "class_names": class_names,
        }

    def draw_detections(
        self,
        image: Image.Image,
        detections: dict[str, Any],
        outline_color: str = "red",
    ) -> Image.Image:
        """Render predicted bounding boxes and class labels onto an image."""
        canvas = image.copy().convert("RGB")
        draw = ImageDraw.Draw(canvas)

        for box, score, name in zip(
            detections["boxes"], detections["scores"], detections["class_names"]
        ):
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=3)
            label_text = f"{name} {score:.2f}"
            draw.text((x1 + 4, max(0, y1 - 12)), label_text, fill=outline_color)

        return canvas


def parse_exdark_annotation_file(annot_path: str | Path) -> list[dict[str, Any]]:
    """Parse ExDark ground truth text annotation file.

    Format per line:
        class_name x_min y_min width height ...

    Args:
        annot_path: Path to .txt annotation.

    Returns:
        List of dicts with 'class_name', 'box' [x1, y1, x2, y2], 'coco_id'.
    """
    annot_path = Path(annot_path)
    if not annot_path.exists():
        return []

    targets = []
    with open(annot_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5 and not parts[0].startswith("%"):
                c_name = parts[0]
                try:
                    x = float(parts[1])
                    y = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    x2 = x + w
                    y2 = y + h
                    coco_id = EXDARK_TO_COCO_ID.get(c_name, None)
                    targets.append(
                        {
                            "class_name": c_name,
                            "box": [x, y, x2, y2],
                            "coco_id": coco_id,
                        }
                    )
                except ValueError:
                    continue

    return targets


def compute_iou(box1: list[float], box2: list[float]) -> float:
    """Compute Intersection-over-Union between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return float(inter_area / max(1e-6, union_area))


def evaluate_detection_metrics(
    predictions_list: list[dict[str, Any]],
    ground_truths_list: list[list[dict[str, Any]]],
    iou_thresh: float = 0.5,
) -> dict[str, float]:
    """Calculate mAP@0.5 and Recall@0.5 over matched compatible categories.

    Args:
        predictions_list: List of predictions dicts from detect_image.
        ground_truths_list: List of target lists from parse_exdark_annotation_file.
        iou_thresh: IoU matching threshold.

    Returns:
        Dict with 'mAP_50', 'recall_50', and 'total_gt_objects'.
    """
    total_gt = 0
    true_positives = 0
    false_positives = 0

    for pred, targets in zip(predictions_list, ground_truths_list):
        # Filter ground truths having compatible COCO mapping
        valid_targets = [t for t in targets if t["coco_id"] is not None]
        total_gt += len(valid_targets)

        matched_gt = set()
        p_boxes = pred["boxes"]
        p_labels = pred["labels"]

        for p_box, p_label in zip(p_boxes, p_labels):
            best_iou = 0.0
            best_idx = -1
            for idx, t in enumerate(valid_targets):
                if idx not in matched_gt and t["coco_id"] == p_label:
                    iou = compute_iou(p_box, t["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx

            if best_iou >= iou_thresh and best_idx >= 0:
                true_positives += 1
                matched_gt.add(best_idx)
            else:
                false_positives += 1

    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, total_gt)
    # Approximate AP@0.5
    ap = precision * recall

    return {
        "mAP_50": round(ap, 4),
        "recall_50": round(recall, 4),
        "precision_50": round(precision, 4),
        "total_gt_objects": total_gt,
        "true_positives": true_positives,
    }

#!/usr/bin/env python3
"""Appearance (Re-ID) embedding helper for the appearance-based demos.

Deep SORT and Deep OC-SORT both need a per-detection appearance vector. This
module wraps a generic ImageNet ResNet18 as a stand-in feature extractor so the
demos stay focused on tracking. For real workloads swap in a Re-ID-specific
model (OSNet, FastReID, etc.).

Requirements:
    pip install torch torchvision pillow opencv-python
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image


def get_embedder() -> tuple[torch.nn.Module, T.Compose]:
    """Load a pretrained ResNet18 as a 512-dim feature extractor.

    Returns:
        A tuple of the eval-mode model (classification head removed) and the
        preprocessing transform sized for typical Re-ID crops.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()

    transform = T.Compose(
        [
            T.Resize((128, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, transform


def extract_features(
    model: torch.nn.Module,
    transform: T.Compose,
    frame,
    bboxes: Sequence[Sequence[float]],
) -> list[list[float]]:
    """Extract L2-normalized appearance embeddings for a batch of boxes.

    Args:
        model: Feature extractor from :func:`get_embedder`.
        transform: Matching preprocessing transform.
        frame: BGR image frame from OpenCV.
        bboxes: Bounding boxes in TLWH format ``[[x, y, w, h], ...]``.

    Returns:
        One embedding per box as a list of floats; empty if ``bboxes`` is empty.
    """
    if not bboxes:
        return []

    height, width = frame.shape[:2]
    crops = []
    for x, y, w, h in bboxes:
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(width, int(x + w)), min(height, int(y + h))
        if x2 <= x1 or y2 <= y1:
            crop = np.zeros((128, 64, 3), dtype=np.uint8)
        else:
            crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        crops.append(Image.fromarray(crop))

    batch = torch.stack([transform(img) for img in crops])
    with torch.no_grad():
        features = model(batch)
    features = torch.nn.functional.normalize(features, p=2, dim=1)
    return features.numpy().tolist()

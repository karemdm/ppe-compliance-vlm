"""Appearance embeddings of person crops for re-identification.

Two embedders are available (see :class:`ppe_compliance.tracking.reid.ReIDConfig`):

* :class:`HistogramEmbedder` (``"histogram"``, default): HSV color
  histograms of horizontal body stripes. No extra model, fast on CPU.
* :class:`YOLOEmbedder` (any other value): pooled features of an Ultralytics
  classification model such as ``yolo11n-cls.pt`` (the same kind of model
  Ultralytics uses for BoT-SORT re-ID). Runs on the GPU when available.

Both return one embedding (list of floats) per crop, or ``None`` for an
empty crop.
"""

import logging
from typing import List, Optional, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: Value of ``ReIDConfig.embedder`` that selects :class:`HistogramEmbedder`.
HISTOGRAM = "histogram"


class HistogramEmbedder:
    """HSV color histogram per horizontal stripe of the person crop.

    The crop is resized to ``size``, split into ``stripes`` horizontal bands
    (head, torso, legs...) and, per band, a joint hue-saturation histogram and
    a value histogram are computed. Each histogram is L1-normalized and
    square-rooted (Hellinger kernel), so the cosine of two embeddings is the
    Bhattacharyya coefficient averaged over the bands.
    """

    def __init__(self, stripes: int = 6, h_bins: int = 16, s_bins: int = 4, v_bins: int = 8,
                 size: Sequence[int] = (64, 128)) -> None:
        """Create the embedder.

        Args:
            stripes: Number of horizontal bands.
            h_bins: Hue bins.
            s_bins: Saturation bins.
            v_bins: Value bins.
            size: ``(width, height)`` the crop is resized to.
        """
        self.stripes = stripes
        self.h_bins, self.s_bins, self.v_bins = h_bins, s_bins, v_bins
        self.size = (int(size[0]), int(size[1]))

    def embed(self, crops: Sequence[np.ndarray]) -> List[Optional[List[float]]]:
        """Embed BGR crops.

        Args:
            crops: BGR person crops.

        Returns:
            One embedding per crop (``None`` for empty crops).
        """
        return [self._embed_one(c) for c in crops]

    def _embed_one(self, crop: np.ndarray) -> Optional[List[float]]:
        if crop is None or crop.size == 0:
            return None
        img = cv2.resize(crop, self.size, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        parts: List[np.ndarray] = []
        bounds = np.linspace(0, self.size[1], self.stripes + 1).astype(int)
        for y1, y2 in zip(bounds[:-1], bounds[1:]):
            band = hsv[y1:y2]
            hs = cv2.calcHist([band], [0, 1], None, [self.h_bins, self.s_bins], [0, 180, 0, 256]).ravel()
            v = cv2.calcHist([band], [2], None, [self.v_bins], [0, 256]).ravel()
            for hist in (hs, v):
                total = float(hist.sum())
                parts.append(np.sqrt(hist / total) if total > 0 else hist)
        vector = np.concatenate(parts).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        return (vector / norm).tolist() if norm > 0 else None


class YOLOEmbedder:
    """Pooled features of an Ultralytics model (``YOLO.embed``)."""

    def __init__(self, weights: str, imgsz: int = 224) -> None:
        """Load the model (auto-downloaded by Ultralytics when given by name).

        Args:
            weights: Path or name of the weights, e.g. ``"yolo11n-cls.pt"``.
            imgsz: Inference size of the crops.
        """
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.imgsz = imgsz
        logger.info("Re-ID embedder: %s (imgsz %d)", weights, imgsz)

    def embed(self, crops: Sequence[np.ndarray]) -> List[Optional[List[float]]]:
        """Embed BGR crops in one batch.

        Args:
            crops: BGR person crops.

        Returns:
            One embedding per crop (``None`` for empty crops).
        """
        valid = [i for i, c in enumerate(crops) if c is not None and c.size > 0]
        out: List[Optional[List[float]]] = [None] * len(crops)
        if not valid:
            return out
        features = self.model.embed([crops[i] for i in valid], imgsz=self.imgsz, verbose=False)
        for i, feat in zip(valid, features):
            vector = feat.detach().float().flatten().cpu().numpy()
            norm = float(np.linalg.norm(vector))
            out[i] = (vector / norm).tolist() if norm > 0 else None
        return out


def build_embedder(name: str):
    """Create the embedder selected by ``ReIDConfig.embedder``.

    Args:
        name: ``"histogram"`` or Ultralytics weights.

    Returns:
        An object with an ``embed(crops)`` method.
    """
    if name.lower() == HISTOGRAM:
        return HistogramEmbedder()
    return YOLOEmbedder(name)

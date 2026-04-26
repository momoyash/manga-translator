"""
ensemble.py  --  part of simple-manga-translator

Combines multiple detection strategies for maximum speech bubble coverage:
  - DBNet text detector (finds text lines)
  - Deep ONNX model (finds bubble regions)
  - CV contour analysis (catches anything both miss)
"""

import sys
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional

from .common import CommonDetector
from .default import DefaultDetector
from ..utils import Quadrilateral

_CORE_DIR = Path(__file__).parent.parent.parent.parent / "simple-manga-translator" / "core"
if str(_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_DIR))

try:
    from detector import find as find_regions, subtract, _box, _iou
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class _Wrap:
    """Thin wrapper so subtract() can work with Quadrilateral pts."""
    def __init__(self, pts): self.pts = pts


class EnsembleDetector(CommonDetector):

    def __init__(self):
        super().__init__()
        self._base = DefaultDetector()

    async def load_sub_detectors(self, device: str):
        await self._base.load(device)

    async def _detect(self, image: np.ndarray, detect_size: int,
                      text_threshold: float, box_threshold: float,
                      unclip_ratio: float, verbose: bool = False
                      ) -> Tuple[List[Quadrilateral], np.ndarray, Optional[np.ndarray]]:

        h, w = image.shape[:2]

        # Step 1 — base text detector
        lines, mask, refined = await self._base._detect(
            image, detect_size, text_threshold, box_threshold, unclip_ratio, verbose
        )

        if not _AVAILABLE:
            self.logger.warning("[smt] detector module not found — using base only")
            return lines, mask, refined

        # Step 2 — deep + CV bubble finder
        bgr      = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        regions  = find_regions(bgr, debug=verbose)

        if not regions:
            return lines, mask, refined

        # Step 3 — keep only regions not already covered
        existing = [_Wrap(q.pts) for q in lines]
        new      = subtract(regions, existing, thresh=0.25)

        if not new:
            return lines, mask, refined

        by_model = sum(1 for r in new if r.method == "model")
        by_cv    = sum(1 for r in new if r.method == "cv")
        self.logger.info(f"[smt] +{len(new)} regions  (model={by_model} cv={by_cv})")

        # Step 4 — add to results
        for r in new:
            pts = r.pts.astype(np.int64)
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            lines.append(Quadrilateral(pts, '', r.score))
            cv2.fillPoly(mask, [pts.astype(np.int32)], 255)

        return lines, mask, refined

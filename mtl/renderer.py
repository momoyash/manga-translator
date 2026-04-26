"""
core/renderer.py  --  simple-manga-translator

Post-processing step that clips translated text to speech bubble boundaries.

The translation engine caps region expansion at 1.1x the original text box,
but English is often 2-3x longer than Japanese — causing text to overflow.

Fix: after rendering, detect bubble boundaries and restore any pixel
outside a bubble that was changed (overflow text) back to the original.

Usage (standalone):
  python core/renderer.py -o original.png -t translated.png -s output.png
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from .detector import find, Region


# ── Bubble mask ────────────────────────────────────────────────────────────────

def _detect_bubble_contours(image: np.ndarray) -> np.ndarray:
    """
    Detect actual speech bubble shapes using brightness and shape analysis.
    Returns a binary mask of detected bubble interiors.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    # Use adaptive threshold to handle varying brightness
    # Also try fixed threshold at a lower value
    _, bright1 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    bright2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 51, -10)
    bright = cv2.bitwise_or(bright1, bright2)

    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel, iterations=2)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros((h, w), dtype=np.uint8)
    page_area = h * w

    for c in contours:
        area = cv2.contourArea(c)
        # Filter by size - bubbles are typically 0.2% to 20% of page
        if not (page_area * 0.002 < area < page_area * 0.20):
            continue

        # Get bounding rect for aspect ratio check
        x, y, bw, bh = cv2.boundingRect(c)
        if bh == 0:
            continue
        aspect = bw / bh
        # Bubbles have reasonable aspect ratios
        if not (0.2 < aspect < 5.0):
            continue

        # Check convexity - bubbles tend to be convex
        hull_area = cv2.contourArea(cv2.convexHull(c))
        if hull_area > 0 and area / hull_area > 0.5:
            cv2.drawContours(mask, [c], -1, 255, -1)

    return mask


def _bubble_mask(image: np.ndarray, regions: list,
                 expand_px: int = 10) -> np.ndarray:
    """
    Build a binary mask (255 = inside a bubble, 0 = outside).
    Combines region-based detection with direct bubble contour detection.
    """
    h, w = image.shape[:2]

    # Method 1: Detect bright bubble shapes directly
    mask = _detect_bubble_contours(image)

    # Method 2: Add regions that have actual contours
    for r in regions:
        if r.contour is not None and len(r.contour) > 4:
            cv2.drawContours(mask, [r.contour], -1, 255, -1)
        elif r.kind == 0:  # Only add pts for "bubble" type, not text boxes
            pts = r.pts.astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)

    # Expand the mask for anti-aliased edges
    if expand_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (expand_px * 2 + 1, expand_px * 2 + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


# ── Overflow detection ─────────────────────────────────────────────────────────

def _changed_pixels(original: np.ndarray,
                    translated: np.ndarray,
                    threshold: int = 12) -> np.ndarray:
    """
    Return a binary mask of pixels that changed between original and translated.
    Uses a threshold to ignore minor JPEG artifacts.
    """
    diff = np.abs(original.astype(np.int32) - translated.astype(np.int32))
    return (diff.max(axis=2) > threshold).astype(np.uint8) * 255


# ── Main fix ───────────────────────────────────────────────────────────────────

def clip_overflow(original: np.ndarray, translated: np.ndarray,
                  min_score: float = 0.25,
                  expand_px: int = 8,
                  debug: bool = False) -> np.ndarray:
    """
    Remove translated text that leaked outside speech bubble boundaries.

    Steps:
      1. Detect all speech bubbles in the original image
      2. Build a mask of bubble regions (with slight expansion for clean edges)
      3. Find pixels that changed between original and translated
      4. Restore any changed pixel that's outside all bubble boundaries

    Args:
        original:   The source manga page (before translation)
        translated: The rendered output (after translation)
        min_score:  Minimum bubble detection confidence
        expand_px:  How many pixels to expand bubble boundaries inward (anti-alias buffer)
        debug:      If True, save a debug image showing what was clipped

    Returns:
        Fixed image with overflow text removed
    """
    regions = find(original, min_score=min_score)

    if not regions:
        if debug:
            print("[renderer] No bubbles detected — skipping clip")
        return translated

    bubble_mask = _bubble_mask(original, regions, expand_px=expand_px)
    changed     = _changed_pixels(original, translated)

    # Overflow = changed AND outside all bubbles
    outside     = cv2.bitwise_not(bubble_mask)
    overflow    = cv2.bitwise_and(changed, outside)

    # Slightly dilate to catch partially-clipped characters
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    overflow = cv2.dilate(overflow, kernel, iterations=2)

    clipped_count = int(np.sum(overflow > 0))

    if debug:
        print(f"[renderer] Bubbles: {len(regions)}  |  Overflow pixels: {clipped_count}")

    if clipped_count == 0:
        return translated

    # Replace overflow areas with original pixels
    result       = translated.copy()
    flow_mask    = overflow[:, :, np.newaxis] / 255.0
    result       = (original * flow_mask + translated * (1 - flow_mask)).astype(np.uint8)

    return result


# ── Batch processing ───────────────────────────────────────────────────────────

def fix_folder(original_dir: str, translated_dir: str,
               output_dir: Optional[str] = None,
               min_score: float = 0.25,
               debug: bool = False) -> int:
    """
    Apply clip_overflow to every image in a folder.

    Args:
        original_dir:   Folder with original manga pages
        translated_dir: Folder with translated images (same filenames, .png)
        output_dir:     Where to save fixed images (defaults to translated_dir)
        min_score:      Bubble detection confidence threshold

    Returns:
        Number of images processed
    """
    orig_dir  = Path(original_dir)
    trans_dir = Path(translated_dir)
    out_dir   = Path(output_dir) if output_dir else trans_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    exts    = {".png", ".jpg", ".jpeg", ".webp"}
    originals = sorted(p for p in orig_dir.iterdir() if p.suffix.lower() in exts)

    count = 0
    for orig_path in originals:
        # Match translated file (may have different extension — always .png after translation)
        trans_path = trans_dir / (orig_path.stem + ".png")
        if not trans_path.exists():
            trans_path = trans_dir / orig_path.name
        if not trans_path.exists():
            continue

        orig_img  = cv2.imread(str(orig_path))
        trans_img = cv2.imread(str(trans_path))

        if orig_img is None or trans_img is None:
            continue

        # Resize original to match translated if dimensions differ
        if orig_img.shape[:2] != trans_img.shape[:2]:
            orig_img = cv2.resize(orig_img, (trans_img.shape[1], trans_img.shape[0]))

        fixed = clip_overflow(orig_img, trans_img, min_score=min_score, debug=debug)

        out_path = out_dir / (orig_path.stem + ".png")
        cv2.imwrite(str(out_path), fixed)

        if debug:
            print(f"[renderer] Fixed: {orig_path.name}")
        count += 1

    print(f"[renderer] Done — {count} images processed")
    return count


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Clip overflow text to bubble boundaries")
    sub = p.add_subparsers(dest="cmd")

    # Single image
    single = sub.add_parser("image", help="Fix a single image")
    single.add_argument("-o", "--original",   required=True, help="Original manga page")
    single.add_argument("-t", "--translated", required=True, help="Translated image")
    single.add_argument("-s", "--save",       required=True, help="Output path")
    single.add_argument("--score", type=float, default=0.25)

    # Folder
    folder = sub.add_parser("folder", help="Fix all images in a folder")
    folder.add_argument("-o", "--original",   required=True, help="Original folder")
    folder.add_argument("-t", "--translated", required=True, help="Translated folder")
    folder.add_argument("-s", "--save",       default=None,  help="Output folder (default: overwrites translated)")
    folder.add_argument("--score", type=float, default=0.25)

    args = p.parse_args()

    if args.cmd == "image":
        orig  = cv2.imread(args.original)
        trans = cv2.imread(args.translated)
        if orig is None:  print(f"Cannot load: {args.original}"); exit(1)
        if trans is None: print(f"Cannot load: {args.translated}"); exit(1)
        if orig.shape[:2] != trans.shape[:2]:
            orig = cv2.resize(orig, (trans.shape[1], trans.shape[0]))
        result = clip_overflow(orig, trans, min_score=args.score, debug=True)
        cv2.imwrite(args.save, result)
        print(f"Saved: {args.save}")

    elif args.cmd == "folder":
        fix_folder(args.original, args.translated, args.save,
                   min_score=args.score, debug=True)

    else:
        p.print_help()

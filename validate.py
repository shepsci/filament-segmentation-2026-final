#!/usr/bin/env python3
"""Pre-submit gate: a submission's predicted masks must be pixel-disjoint within each image.

**This gate is not optional.** The competition scorer rejects any image whose predicted segments
share a pixel — a panoptic prediction assigns each pixel to at most one segment — and the
rejection arrives as `SubmissionStatus.ERROR` *after* the upload, having already consumed one of
the day's five slots. The host confirmed the rule directly in competition discussion/735377.

Raw Mask R-CNN output violates it by default. Its NMS is box-level, not mask-level, so two
proposals with sufficiently different boxes both survive and their mask heads independently paint
overlapping pixels. For long thin filaments clustered in one solar region that is the normal
outcome, not an edge case.

The fix is `paint_panoptic` below: sort by score descending and let each segment claim only the
pixels no higher-scoring segment has taken. Score-ordered painting keeps more instances than
"drop the whole lower-scoring mask", which matters because PQ's numerator counts true positives.
Re-check `min_area` AFTER subtraction — an eroded remnant that survives as a sliver is a false
positive, and each false positive costs 0.5 in PQ's denominator.

Credit: the failure mode and the score-ordered-painting remedy were first written up publicly by
Siva Kumar in competition discussion/735377 (2026-08-15). Implementation here is our own;
attribution is retained in NOTICE.

Usage:
    python3 validate.py submission.csv        # exits non-zero on any overlap
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

H = W = 2048


def paint_panoptic(candidates, min_area: int):
    """(score, HxW uint8 mask) -> the same list made pixel-disjoint, highest score wins.

    Returns (score, mask, area) triples. A segment whose remaining area falls below `min_area`
    after subtraction is dropped entirely rather than submitted as a sliver.
    """
    kept = []
    claimed = np.zeros((H, W), dtype=np.uint8)
    for score, binary in sorted(candidates, key=lambda x: -x[0]):
        remaining = (binary & (1 - claimed)).astype(np.uint8)
        area = int(remaining.sum())
        if area < min_area:
            continue
        claimed |= remaining
        kept.append((score, remaining, area))
    return kept


def encode(mask: np.ndarray) -> str:
    return mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("utf-8")


def check(csv_path: Path, verbose: bool = True) -> int:
    """Returns the number of images with overlapping predictions. 0 means safe to submit."""
    import csv as _csv
    rows = list(_csv.DictReader(Path(csv_path).open(newline="")))
    if not rows:
        print("[FAIL] submission is empty")
        return 1
    need = {"filament_id", "segmentation_rle"}
    if not need.issubset(rows[0]):
        print(f"[FAIL] columns are {list(rows[0])}, expected {sorted(need)}")
        return 1

    by_image = collections.defaultdict(list)
    for r in rows:
        by_image[r["filament_id"].rsplit("_", 1)[0]].append(r)

    ids = [r["filament_id"] for r in rows]
    dupes = [k for k, n in collections.Counter(ids).items() if n > 1]
    bad_images, empty, decode_fail = [], 0, []
    for stem, group in by_image.items():
        acc = np.zeros((H, W), dtype=np.int16)
        for r in group:
            try:
                m = mask_util.decode({"size": [H, W],
                                      "counts": r["segmentation_rle"].encode("utf-8")})
            except Exception as exc:
                decode_fail.append((r["filament_id"], str(exc)[:60]))
                continue
            if m.shape != (H, W):
                decode_fail.append((r["filament_id"], f"shape {m.shape}"))
                continue
            if m.sum() == 0:
                empty += 1
            acc += m.astype(np.int16)
        if (acc > 1).any():
            bad_images.append((stem, int((acc > 1).sum())))

    if verbose:
        print(f"rows {len(rows)} · images {len(by_image)} · "
              f"mean {len(rows)/len(by_image):.1f} filaments/image")
        if dupes:
            print(f"[FAIL] {len(dupes)} duplicate filament_id, e.g. {dupes[:3]}")
        if decode_fail:
            print(f"[FAIL] {len(decode_fail)} RLEs did not decode to {H}x{W}, "
                  f"e.g. {decode_fail[:2]}")
        if empty:
            print(f"[warn] {empty} empty masks — each is a pure false positive, costing 0.5 "
                  f"in PQ's denominator for nothing")
        for stem, px in bad_images[:5]:
            print(f"[FAIL] {stem}: {px:,} pixels covered by more than one mask")
        if len(bad_images) > 5:
            print(f"[FAIL] ... and {len(bad_images)-5} more images with overlap")
        if not (bad_images or dupes or decode_fail):
            print("[PASS] no overlaps, no duplicate ids, all RLEs decode — safe to submit")
    return len(bad_images) + len(dupes) + len(decode_fail)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    a = ap.parse_args()
    sys.exit(1 if check(Path(a.submission)) else 0)

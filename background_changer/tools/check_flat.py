"""Compare a flat-path run against the pinned baseline (plan: Verification 1).

    .venv\\Scripts\\python tools\\check_flat.py output\\_baseline_flat output\\_flat_check

Every composite and debug image must decode to the same pixels; cut-outs must match on alpha, and on
colour where alpha > 0 (lossless WEBP drops colour under alpha 0); results.csv must match on the baseline's
columns. Exit code 1 on any difference.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

EXTS = {".webp", ".jpg", ".png"}


def read(path: Path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)


def compare(base: Path, new: Path) -> list[str]:
    problems = []
    n = 0
    for b in sorted(base.rglob("*")):
        if not b.is_file():
            continue
        rel = b.relative_to(base)
        o = new / rel
        if not o.is_file():
            problems.append(f"missing: {rel}")
            continue
        if b.suffix.lower() in EXTS:
            n += 1
            x, y = read(b), read(o)
            if x is None or y is None or x.shape != y.shape:
                problems.append(f"shape differs: {rel}")
                continue
            if "cutouts" in rel.parts and x.ndim == 3 and x.shape[2] == 4:
                if not np.array_equal(x[..., 3], y[..., 3]):
                    problems.append(f"cut-out alpha differs: {rel}")
                elif not np.array_equal(x[..., :3][x[..., 3] > 0], y[..., :3][y[..., 3] > 0]):
                    problems.append(f"cut-out colour differs: {rel}")
            elif not np.array_equal(x, y):
                d = np.abs(x.astype(int) - y.astype(int))
                problems.append(f"pixels differ: {rel} (max {d.max()}, {int((d.max(-1) if d.ndim == 3 else d).astype(bool).sum())} px)")
        elif b.name == "results.csv":
            with open(b, newline="", encoding="utf-8") as fb, open(o, newline="", encoding="utf-8") as fo:
                rb, ro = list(csv.DictReader(fb)), list(csv.DictReader(fo))
            if len(rb) != len(ro):
                problems.append(f"{rel}: {len(rb)} rows vs {len(ro)}")
                continue
            skip = {"segment_s"}  # timing
            for i, (x, y) in enumerate(zip(rb, ro)):
                diff = [k for k in x if k not in skip and x[k] != y.get(k)]
                if diff:
                    problems.append(f"{rel} row {i + 1}: {', '.join(diff)}")
    print(f"compared {n} images under {base}")
    return problems


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    problems = compare(Path(sys.argv[1]), Path(sys.argv[2]))
    if problems:
        print("DIFFERENCES:\n  " + "\n  ".join(problems[:200]))
        sys.exit(1)
    print("identical to the baseline")


if __name__ == "__main__":
    main()

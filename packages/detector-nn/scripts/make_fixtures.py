"""Generates the parity fixtures of @scanoficus/detector-nn with the NumPy reference interpreter.

The test frames are synthetic scenes (a page with text lines on a noisy background) produced by
an integer-only generator that `test/scene.ts` reproduces bit-exactly, so the frames themselves
are not stored. For each frame this writes:

- an entry in `cases.json`: the SHA-256 of the frame and of the network input after the Python
  preprocessing (`scanoficus_ml.data.preprocess`: area downscale to 192×192, RGB, uint8 CHW),
  which the TypeScript preprocessing must reproduce exactly, the 9 raw heads of `scanoficus_ml.rawgraph.run_graph` (corners
  then presence logit), the presence probability and the decoded quad in frame pixels.

Usage (from the repository root, with the scanoficus-ml checkout and its virtualenv):

    PYTHONDONTWRITEBYTECODE=1 ~/src/scanoficus-ml/.venv/bin/python \
        packages/detector-nn/scripts/make_fixtures.py [--ml ~/src/scanoficus-ml]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "test" / "fixtures"

# name, width, height, page corners (clockwise), text block corners, seed
CASES = [
    ("page-640x480", 640, 480, [(120, 70), (520, 95), (500, 420), (100, 400)], [(160, 120), (470, 138), (458, 360), (148, 346)], 1),
    ("partial-500x375", 500, 375, [(-60, 40), (380, 20), (420, 330), (-30, 360)], [(10, 80), (340, 66), (370, 290), (20, 312)], 2),
    ("tilted-1280x720", 1280, 720, [(400, 60), (900, 140), (820, 680), (310, 600)], [(440, 130), (840, 196), (780, 600), (380, 540)], 3),
    ("small-333x251", 333, 251, [(40, 30), (290, 50), (280, 220), (30, 210)], [(60, 60), (260, 74), (254, 190), (56, 182)], 4),
    ("wide-384x173", 384, 173, [(70, 20), (310, 30), (300, 160), (60, 150)], [(95, 45), (285, 52), (278, 135), (88, 128)], 7),
    ("tall-173x384", 173, 384, [(20, 60), (150, 50), (160, 330), (15, 340)], [(35, 90), (135, 82), (142, 300), (30, 308)], 8),
    ("tiny-160x120", 160, 120, [(25, 15), (135, 22), (130, 105), (20, 98)], [(38, 30), (122, 35), (118, 90), (34, 85)], 9),
    ("strip-768x192", 768, 192, [(150, 20), (620, 30), (610, 175), (140, 165)], [(190, 45), (580, 52), (572, 150), (182, 142)], 10),
    ("empty-192x192", 192, 192, None, None, 5),
    ("noise-192x192", 192, 192, "noise", None, 6),
]

M32 = 0xFFFFFFFF


def hash32(x: np.ndarray, y: np.ndarray, c: int, seed: int) -> np.ndarray:
    x = x.astype(np.uint64)
    y = y.astype(np.uint64)
    h = ((x * 0x9E3779B1) & M32) ^ ((y * 0x85EBCA77) & M32) ^ ((c * 0xC2B2AE3D) & M32) ^ seed
    h ^= h >> 15
    h = (h * 0x2C1B3C6D) & M32
    h ^= h >> 12
    h = (h * 0x297A2D39) & M32
    h ^= h >> 15
    return h


def inside(px: np.ndarray, py: np.ndarray, quad) -> np.ndarray:
    """Pixel centers (2x+1, 2y+1) against a clockwise quad with doubled corners, integers only."""
    ok = np.ones(px.shape, bool)
    for i in range(4):
        ax, ay = 2 * quad[i][0], 2 * quad[i][1]
        bx, by = 2 * quad[(i + 1) % 4][0], 2 * quad[(i + 1) % 4][1]
        ok &= (bx - ax) * (py - ay) - (by - ay) * (px - ax) >= 0
    return ok


def scene(width: int, height: int, page, text, seed: int, style: str = "default", blend: int | None = None) -> np.ndarray:
    """RGB uint8 (height, width, 3); `test/scene.ts` must produce the same bytes.

    `style="white"`: a light table (white-on-white, low contrast). `blend=a` (0–4096): the page
    scene blended into the same scene without a page, `(a·page + (4096 − a)·empty + 2048) >> 12`
    (integer arithmetic), for scores near the presence threshold."""
    if blend is not None:
        with_page = scene(width, height, page, text, seed, style).astype(np.int64)
        empty = scene(width, height, None, None, seed, style).astype(np.int64)
        return ((blend * with_page + (4096 - blend) * empty + 2048) >> 12).astype(np.uint8)
    y, x = np.mgrid[0:height, 0:width].astype(np.int64)
    img = np.zeros((height, width, 3), np.int64)
    px, py = 2 * x + 1, 2 * y + 1
    for c in range(3):
        h = hash32(x, y, c, seed).astype(np.int64)
        if page == "noise":
            img[..., c] = h & 255
            continue
        if style == "white":
            v = 200 - 5 * c + ((x + 2 * y) * 20) // (width + 2 * height) + (h & 15)
        else:
            v = 50 + 15 * c + ((x + 2 * y) * 60) // (width + 2 * height) + (h & 15)
        if page is not None:
            on_page = inside(px, py, page)
            v = np.where(on_page, 225 - 5 * c - (h & 15), v)
            ink = inside(px, py, text) & ((y // 6) % 3 == 0) & ((x // 23) % 5 != 4)
            v = np.where(ink, 60 + (h & 15), v)
        img[..., c] = v
    return img.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml", default=str(Path.home() / "src" / "scanoficus-ml"))
    parser.add_argument("--inputs", action="store_true", help="also write <name>.input.gz (debugging, not committed)")
    args = parser.parse_args()
    ml = Path(args.ml)
    sys.path.insert(0, str(ml))
    from scanoficus_ml.data import preprocess
    from scanoficus_ml.detectors import order_quad
    from scanoficus_ml.rawgraph import run_graph
    from scanoficus_ml.refine import refine_quad
    import cv2

    # The packaged model (scripts/package_model.py); its weights must equal the ML export.
    export = ml / "export" / "docquadnet"
    models = HERE.parent / "models" / "docquadnet"
    graph_path = models / "docquadnet.json"
    graph = json.loads(graph_path.read_text())
    blob = (models / graph["weights"]["file"]).read_bytes()
    assert blob == (export / "docquadnet.bin").read_bytes(), "models/ is stale: run package_model.py"
    weights = np.frombuffer(blob, "<f2").astype(np.float32)
    threshold = float(json.loads((export / "threshold.json").read_text())["threshold"])

    OUT.mkdir(parents=True, exist_ok=True)
    cases = []
    for name, width, height, page, text, seed in CASES:
        rgb = scene(width, height, page, text, seed)
        x = preprocess(np.ascontiguousarray(rgb[..., ::-1]), 192)  # preprocess takes BGR
        u8 = np.rint(x[0] * 255).astype(np.uint8)
        assert np.array_equal(u8.astype(np.float32) / 255, x[0])
        if args.inputs:
            (OUT / f"{name}.input.gz").write_bytes(gzip.compress(u8.tobytes(), 9, mtime=0))
        heads = run_graph(graph, weights, x).astype(np.float64)
        presence = float(1 / (1 + np.exp(-heads[8])))
        quad = order_quad(heads[:8].reshape(4, 2) * np.array([width, height]))
        # refine.py samples at pixel-index coordinates (centers at integers); the TypeScript port
        # uses the core convention (centers at +0.5), so shift by half a pixel around the call.
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        refined = refine_quad(gray, quad - 0.5, long_side=1024, search_frac=0.008) + 0.5
        cases.append({
            "name": name,
            "width": width,
            "height": height,
            "page": page,
            "text": text,
            "seed": seed,
            "frameSha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            "inputSha256": hashlib.sha256(u8.tobytes()).hexdigest(),
            "heads": [float(v) for v in heads],
            "presence": presence,
            "detected": presence > threshold,
            "quad": [[float(a), float(b)] for a, b in quad],
            "refined": [[float(a), float(b)] for a, b in refined],
        })
        print(f"{name}: presence {presence:.4f} heads {np.round(heads, 4).tolist()}")
    meta = {
        "generator": "packages/detector-nn/scripts/make_fixtures.py",
        "reference": "scanoficus_ml.rawgraph.run_graph (NumPy, float32) on scanoficus_ml.data.preprocess",
        "graphSha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
        "weightsSha256": hashlib.sha256(blob).hexdigest(),
        "threshold": threshold,
        "cases": cases,
    }
    (OUT / "cases.json").write_text(json.dumps(meta, indent=1) + "\n")


if __name__ == "__main__":
    main()

"""Parity fixtures of the round-2 pipeline (DocQuadNet 2 + corner refiner) for the test scenes.

For each scene of make_fixtures.py (plus two large frames that exercise the 1600-px work image:
2000×1500, a fractional area resize, and 3200×1800, an exact 2× one) this records:

- `main`: the 9 heads of DocQuadNet 2 (`run_graph`, fp16 weights) on the Python preprocessing,
  the presence and the raw quad (frame pixels, `order_quad`);
- `work`: the work image of `scanoficus_ml.detectors.OwnPipeline` (single `cv2.resize`
  INTER_AREA to long side 1600, if larger) — size and SHA-256 of its RGB bytes;
- `passes`: per refiner pass, the input quad (work-image pixels), the SHA-256 of the four
  uint8 RGB crops (N×64×64×3, `scanoficus_ml.refiner.Pyramid.crop`) and the offsets of the
  refiner network (`run_graph`, fp16 weights, one crop at a time);
- `refined`: `scanoficus_ml.pipeline.apply_variant('net:docquadnet2-refiner', …)` with that
  NumPy refiner (frame pixels) — the reference of the TypeScript runtime;
- `refinedOnnx`: the same with the exported fp32 ONNX refiner, i.e. pipeline.py as it is used
  in the evaluation (tolerance reference).

Writes `test/fixtures/refiner.json`. Usage: see make_fixtures.py (same virtualenv).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_fixtures import CASES, scene  # noqa: E402

LARGE = [
    ("large-2000x1500", 2000, 1500, [(420, 210), (1600, 300), (1540, 1330), (330, 1250)], [(520, 330), (1500, 400), (1450, 1200), (450, 1140)], 11),
    ("large-3200x1800", 3200, 1800, [(900, 150), (2400, 260), (2330, 1700), (820, 1620)], [(1000, 300), (2300, 380), (2240, 1560), (940, 1500)], 12),
]
OUT = HERE.parent / "test" / "fixtures" / "refiner.json"
MODELS = HERE.parent / "models" / "docquadnet2"


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml", default=str(Path.home() / "src" / "scanoficus-ml"))
    args = parser.parse_args()
    sys.path.insert(0, args.ml)
    import cv2

    from scanoficus_ml import refiner
    from scanoficus_ml.data import preprocess
    from scanoficus_ml.detectors import order_quad
    from scanoficus_ml.pipeline import apply_variant
    from scanoficus_ml.rawgraph import run_graph

    def load(name):
        graph = json.loads((MODELS / f"{name}.json").read_text())
        weights = np.frombuffer((MODELS / graph["weights"]["file"]).read_bytes(), "<f2").astype(np.float32)
        return graph, weights

    main_graph, main_w = load("docquadnet2")
    ref_graph, ref_w = load("docquadnet2-refiner")
    variant = "net:docquadnet2-refiner"
    real_session, meta = refiner._load("docquadnet2-refiner")
    log: list[dict] = []

    class NumpyRefiner:
        """Stands in for the ONNX session: run_graph on each crop, recording crops and offsets."""

        def run(self, _names, feeds):
            x = feeds["input"]
            crops = np.rint(x * 255).astype(np.uint8).transpose(0, 2, 3, 1)
            assert np.array_equal(crops.transpose(0, 3, 1, 2).astype(np.float32) / 255, x)
            off = np.stack([run_graph(ref_graph, ref_w, x[i : i + 1]) for i in range(len(x))])
            log.append({"cropSha256": sha(crops), "offsets": off.astype(float).tolist()})
            return [off.astype(np.float32)]

    out = []
    for name, width, height, page, text, seed in [*CASES, *LARGE]:
        rgb = scene(width, height, page, text, seed)
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        heads = run_graph(main_graph, main_w, preprocess(bgr, 192)).astype(np.float64)
        presence = float(1 / (1 + np.exp(-heads[8])))
        raw = order_quad(heads[:8].reshape(4, 2) * np.array([width, height]))
        s = min(1.0, 1600 / max(height, width))
        work = cv2.resize(bgr, (round(width * s), round(height * s)), interpolation=cv2.INTER_AREA) if s < 1 else bgr
        case = {
            "name": name, "width": width, "height": height, "page": page, "text": text, "seed": seed,
            "frameSha256": sha(rgb),
            "main": {"heads": heads.tolist(), "presence": presence, "rawQuad": raw.tolist()},
            "work": {"width": work.shape[1], "height": work.shape[0], "sha256": sha(work[..., ::-1])},
        }
        # Pass inputs: the quad in work pixels before each pass (refine_with_net's loop).
        quads = []
        orig_corner_frame = refiner.corner_frame

        def corner_frame(q, i, scale, _orig=orig_corner_frame):
            if i == 0:
                quads.append(np.asarray(q, np.float64).reshape(4, 2).tolist())
            return _orig(q, i, scale)

        refiner.corner_frame = corner_frame
        refiner._load = lambda _name, threads=1: (NumpyRefiner(), meta)  # noqa: E731
        log.clear()
        refined = order_quad(apply_variant(variant, raw, work, (width, height)))
        case["passes"] = [{"quad": q, "scale": sc, **entry} for q, sc, entry in zip(quads, meta["scales"], log)]
        refiner._load = lambda _name, threads=1: (real_session, meta)  # noqa: E731
        refiner.corner_frame = orig_corner_frame
        case["refined"] = refined.tolist()
        case["refinedOnnx"] = order_quad(apply_variant(variant, raw, work, (width, height))).tolist()
        moved = np.abs(refined - raw).max()
        print(f"{name}: presence {presence:.4f}, refined moved ≤ {moved:.2f} px, "
              f"NumPy vs ONNX {np.abs(refined - np.array(case['refinedOnnx'])).max():.4f} px")
        out.append(case)
    meta_out = {
        "generator": "packages/detector-nn/scripts/make_refiner_fixtures.py",
        "variant": variant,
        "scales": meta["scales"],
        "workSide": 1600,
        "cases": out,
    }
    OUT.write_text(json.dumps(meta_out, indent=1) + "\n")


if __name__ == "__main__":
    main()

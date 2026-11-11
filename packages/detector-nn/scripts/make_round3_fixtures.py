"""Parity fixtures of round 3 (DocQuadNet 2 main network + the 3-output refiner + the presence
decision of export/docquadnet3/decision.json) for the test scenes of make_refiner_fixtures.py.

Two references per scene:

- `numpy`: `scanoficus_ml.verify.run_pipeline` + `features` + the decision of
  `scanoficus_ml.detectors.OwnPipeline3`, with the main network and the refiner replaced by the
  NumPy interpreter `run_graph` on the packaged fp16 weights (the runtime's own reference; the
  main presence is the float32 sigmoid of the logit, as the ONNX graph computes it). Records the
  pass inputs (work-image quads), crop hashes, the 3 outputs per crop, the features, the score,
  the decision and the final quad (frame pixels);
- `onnx`: `OwnPipeline3` itself (fp32 ONNX exports), as evaluated in scanoficus-ml: score,
  decision and quad (tolerance reference).

Writes `test/fixtures/round3.json`. Usage: see make_fixtures.py (same virtualenv).
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
from make_refiner_fixtures import LARGE  # noqa: E402

MODELS = HERE.parent / "models"
OUT = HERE.parent / "test" / "fixtures" / "round3.json"


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml", default=str(Path.home() / "src" / "scanoficus-ml"))
    args = parser.parse_args()
    sys.path.insert(0, args.ml)
    import cv2

    from scanoficus_ml import verify
    from scanoficus_ml.data import preprocess
    from scanoficus_ml.detectors import OwnPipeline3, order_quad
    from scanoficus_ml.rawgraph import run_graph

    def load(bundle: str, name: str):
        graph = json.loads((MODELS / bundle / f"{name}.json").read_text())
        weights = np.frombuffer((MODELS / bundle / graph["weights"]["file"]).read_bytes(), "<f2").astype(np.float32)
        return graph, weights

    cfg = json.loads((MODELS / "docquadnet3" / "decision.json").read_text())
    main_graph, main_w = load("docquadnet2", "docquadnet2")
    ref_graph, ref_w = load("docquadnet3", "docquadnet3-refiner")
    log: list[dict] = []

    class NumpyMain:
        def run(self, _names, feeds):
            h = run_graph(main_graph, main_w, feeds["input"]).astype(np.float32)
            p = np.float32(1) / (np.float32(1) + np.exp(-h[8:9]))
            return [h[None, :8], p[None]]

    class NumpyRefiner:
        def run(self, _names, feeds):
            x = feeds["input"]
            crops = np.rint(x * 255).astype(np.uint8).transpose(0, 2, 3, 1)
            out = np.stack([run_graph(ref_graph, ref_w, x[i : i + 1]) for i in range(len(x))]).astype(np.float32)
            log.append({"cropSha256": sha(crops), "heads": out.astype(float).tolist()})
            return [out]

    real_session, real_load = verify._session, verify._load
    onnx = OwnPipeline3("docquadnet3")
    cases = []
    for name, width, height, page, text, seed in [*CASES, *LARGE]:
        rgb = scene(width, height, page, text, seed)
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        x = preprocess(bgr, 192)
        k = min(1.0, verify.WORK_SIDE / max(height, width))
        work = cv2.resize(bgr, (round(width * k), round(height * k)), interpolation=cv2.INTER_AREA) if k < 1 else bgr
        # NumPy reference through the same code path as OwnPipeline3.
        verify._session = lambda *_a, **_k: NumpyMain()  # noqa: E731
        verify._load = lambda *_a, **_k: (NumpyRefiner(), {"scales": cfg["scales"]})  # noqa: E731
        log.clear()
        res = verify.run_pipeline(None, cfg["main"], cfg["refiner"], tuple(cfg["scales"]), net_input=x, work=work)
        verify._session, verify._load = real_session, real_load
        f = verify.features(res, set(cfg["features"]))
        score = cfg["bias"] + sum(
            wk * (min(max(f[n], -12.0), 12.0) if n == "z" else f[n]) for n, wk in zip(cfg["features"], cfg["weights"])
        )
        wh, ww = work.shape[:2]
        quad = order_quad(res["quads"][-1] * [width / ww, height / wh])
        # The deployed Python pipeline (ONNX fp32).
        p = onnx.detect(bgr)
        cases.append({
            "name": name, "width": width, "height": height, "page": page, "text": text, "seed": seed,
            "work": [ww, wh],
            "numpy": {
                "presence": float(res["presence"]),
                "passes": [{"quad": q.tolist(), **entry} for q, entry in zip(res["quads"][:-1], log)],
                "features": {n: float(f[n]) for n in cfg["features"]},
                "score": float(score),
                "accept": bool(score > cfg["threshold"]),
                "quad": quad.tolist(),
            },
            "onnx": {
                "score": float(onnx.last_score),
                "accept": p.quad is not None,
                "quad": np.asarray(p.raw_quad).tolist(),
            },
        })
        print(f"{name}: score {score:+.4f} ({'accept' if score > cfg['threshold'] else 'reject'}), "
              f"ONNX {onnx.last_score:+.4f}, quad NumPy vs ONNX {np.abs(quad - p.raw_quad).max():.4f} px")
    OUT.write_text(json.dumps({
        "generator": "packages/detector-nn/scripts/make_round3_fixtures.py",
        "decision": "models/docquadnet3/decision.json",
        "threshold": cfg["threshold"],
        "cases": cases,
    }, indent=1) + "\n")


if __name__ == "__main__":
    main()

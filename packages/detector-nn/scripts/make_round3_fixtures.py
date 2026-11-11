"""Parity fixtures of round 3 (DocQuadNet 2 main network + the 3-output refiner + the presence
decision of export/docquadnet3/decision.json) for the test scenes of make_refiner_fixtures.py.

Scenes: the round-2 fixture scenes, 8 blends of the page scene into its empty background whose
NumPy scores lie near the threshold (margins about ±0.05, ±0.15, ±0.3, ±0.5, found by scanning
the blend), and a white page on a light table. Two references per scene:

- `numpy`: `scanoficus_ml.verify.run_pipeline` + `features` + the decision of
  `scanoficus_ml.detectors.OwnPipeline3`, with the main network and the refiner replaced by the
  NumPy interpreter `run_graph` on the packaged fp16 weights (the runtime's own reference; the
  main presence is the float32 sigmoid of the logit, as the ONNX graph computes it). Records the
  pass inputs (work-image quads), crop hashes, the 3 outputs per crop, the features, the score,
  the decision and the final quad (frame pixels);
- `onnx`: `OwnPipeline3` itself (fp32 ONNX exports), as evaluated in scanoficus-ml: score,
  decision and quad (tolerance reference).

Also writes `test/fixtures/decision-near.json`: feature vectors of scanoficus-ml's validation
caches whose scores lie closest to the threshold, plus some with |z| > 12, with their Python
scores (decision-only test).

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
# Target NumPy margins of the near-threshold blends (both sides of the threshold).
NEAR_MARGINS = [-0.5, -0.3, -0.15, -0.05, 0.05, 0.15, 0.3, 0.5]
# A thin white page on a light table (low contrast).
WHITE = ("white-640x480", 640, 480, [(130, 80), (510, 100), (495, 410), (110, 395)], [(170, 130), (460, 145), (450, 350), (160, 340)], 13)
OUT = HERE.parent / "test" / "fixtures" / "round3.json"
DECISION_OUT = HERE.parent / "test" / "fixtures" / "decision-near.json"


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

    def numpy_pipeline(rgb: np.ndarray):
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        height, width = rgb.shape[:2]
        x = preprocess(bgr, 192)
        k = min(1.0, verify.WORK_SIDE / max(height, width))
        work = cv2.resize(bgr, (round(width * k), round(height * k)), interpolation=cv2.INTER_AREA) if k < 1 else bgr
        # NumPy reference through the same code path as OwnPipeline3.
        verify._session = lambda *_a, **_k: NumpyMain()  # noqa: E731
        verify._load = lambda *_a, **_k: (NumpyRefiner(), {"scales": cfg["scales"]})  # noqa: E731
        log.clear()
        try:
            res = verify.run_pipeline(None, cfg["main"], cfg["refiner"], tuple(cfg["scales"]), net_input=x, work=work)
        finally:
            verify._session, verify._load = real_session, real_load
        f = verify.features(res, set(cfg["features"]))
        score = cfg["bias"] + sum(
            wk * (min(max(f[n], -12.0), 12.0) if n == "z" else f[n]) for n, wk in zip(cfg["features"], cfg["weights"])
        )
        return bgr, work, res, f, float(score)

    def case(name, width, height, page, text, seed, style="default", blend=None):
        rgb = scene(width, height, page, text, seed, style, blend)
        bgr, work, res, f, score = numpy_pipeline(rgb)
        wh, ww = work.shape[:2]
        quad = order_quad(res["quads"][-1] * [width / ww, height / wh])
        passes = [{"quad": q.tolist(), **entry} for q, entry in zip(res["quads"][:-1], log)]
        # The deployed Python pipeline (ONNX fp32).
        p = onnx.detect(bgr)
        print(f"{name}: score {score:+.4f} (margin {score - cfg['threshold']:+.4f}), ONNX {onnx.last_score:+.4f}, "
              f"quad NumPy vs ONNX {np.abs(quad - p.raw_quad).max():.4f} px")
        return {
            "name": name, "width": width, "height": height, "page": page, "text": text, "seed": seed,
            "style": style, "blend": blend, "work": [ww, wh],
            "numpy": {
                "presence": float(res["presence"]),
                "passes": passes,
                "features": {n: float(f[n]) for n in cfg["features"]},
                "score": score,
                "margin": score - cfg["threshold"],
                "accept": bool(score > cfg["threshold"]),
                "quad": quad.tolist(),
            },
            "onnx": {
                "score": float(onnx.last_score),
                "accept": p.quad is not None,
                "quad": np.asarray(p.raw_quad).tolist(),
            },
        }

    cases = [case(*c) for c in [*CASES, *LARGE]]

    # Near the threshold: the page scene blended into its empty background. Scan the blend in
    # NumPy and keep the blends closest to each target margin (score − threshold).
    # The score jumps by up to ~0.7 per 1/4096 of blend (8-bit rounding flips many pixels at
    # once), so scan a pool of noise seeds × blends and keep the candidates closest to each
    # target margin.
    name0, w0, h0, page0, text0, _ = CASES[0]
    pool = []
    for seed in range(101, 121):
        for a in range(100, 130, 2):
            *_, score = numpy_pipeline(scene(w0, h0, page0, text0, seed, blend=a))
            pool.append((seed, a, score - cfg["threshold"]))
    chosen: list[tuple[int, int]] = []
    for target in NEAR_MARGINS:
        seed, a, m = min((x for x in pool if x[:2] not in chosen), key=lambda x: abs(x[2] - target))
        chosen.append((seed, a))
        print(f"target margin {target:+.2f}: seed {seed}, blend {a}/4096, margin {m:+.4f}")
    for seed, a in sorted(chosen):
        cases.append(case(f"blend{a:04d}-seed{seed}-{name0}", w0, h0, page0, text0, seed, "default", a))
    cases.append(case(*WHITE, style="white"))

    # Decision-only vectors near the threshold from scanoficus-ml's validation feature caches
    # (cache/verify/*-rf3.json), plus vectors with |z| > 12 (the clip).
    vectors = []
    for f in sorted((Path(args.ml) / "cache" / "verify").glob("*-rf3.json")):
        for e in json.loads(f.read_text()):
            feats = {n: float(e["f"][n]) for n in cfg["features"]}
            score = cfg["bias"] + sum(
                wk * (min(max(feats[n], -12.0), 12.0) if n == "z" else feats[n])
                for n, wk in zip(cfg["features"], cfg["weights"])
            )
            vectors.append({"set": f.stem, "id": e["id"], "features": feats, "score": score,
                            "accept": bool(score > cfg["threshold"])})
    near = sorted(vectors, key=lambda v: abs(v["score"] - cfg["threshold"]))[:16]
    clipped = [v for v in vectors if v["features"]["z"] > 12][:2] + [v for v in vectors if v["features"]["z"] < -12][:2]
    DECISION_OUT.write_text(json.dumps({
        "generator": "packages/detector-nn/scripts/make_round3_fixtures.py",
        "source": "scanoficus-ml cache/verify/*-rf3.json",
        "vectors": near + clipped,
    }, indent=1) + "\n")
    print(f"decision vectors: {len(near)} near the threshold, {len(clipped)} with |z| > 12")

    OUT.write_text(json.dumps({
        "generator": "packages/detector-nn/scripts/make_round3_fixtures.py",
        "decision": "models/docquadnet3/decision.json",
        "threshold": cfg["threshold"],
        "cases": cases,
    }, indent=1) + "\n")


if __name__ == "__main__":
    main()

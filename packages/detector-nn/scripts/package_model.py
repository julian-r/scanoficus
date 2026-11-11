"""Packages the DocQuadNet exports of scanoficus-ml into model bundles of @scanoficus/detector-nn.

A bundle is a main network plus an optional corner refiner, in its own folder `models/<bundle>/`:

- `<model>-<sha256[:12]>.bin`: the fp16 weights of each network, unchanged, under a
  content-hashed name (caches can keep them forever);
- `<model>.json`: the graph JSON plus
  - `weights`: `{ file, byteLength, sha256 }`, so a graph never pairs with other weights
    (`loadModel` checks length and SHA-256 and resolves `file` relative to the graph URL);
  - `selfTest`: the outputs of `scanoficus_ml.rawgraph.run_graph` for a constant input of 0.5
    (one batch item) and, for batched graphs, `batch`: one different constant per item (0.0,
    0.3, …), which the WebGL2 runtime reproduces after initialization;
- `bundle.json`: the manifest for self-hosting (`loadBundle`): graph file names, presence
  threshold and, with a refiner, its crop scales, crop size and work-image side;
- `src/bundle-<bundle>.ts`: the same bundle as `new URL('../models/…', import.meta.url)`
  references (bundlers emit them as hashed assets; the path is also right from `dist/`) and
  inline settings.

Bundles: `docquadnet` (round 1, main network, threshold 0.85) and `docquadnet2` (round 2 main
network, threshold 0.98, and the corner refiner `docquadnet2-refiner`).
The attribution notice `models/LICENSE.md` is maintained by hand.

Usage (from the repository root, with the scanoficus-ml checkout and its virtualenv):

    PYTHONDONTWRITEBYTECODE=1 ~/src/scanoficus-ml/.venv/bin/python \\
        packages/detector-nn/scripts/package_model.py [--ml ~/src/scanoficus-ml]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent.parent
MODELS = PKG / "models"
BUNDLES = {
    "docquadnet": {"main": "docquadnet", "refiner": None, "const": "DOCQUADNET", "doc": "Round 1 (default)"},
    "docquadnet2": {
        "main": "docquadnet2",
        "refiner": "docquadnet2-refiner",
        "const": "DOCQUADNET2",
        "doc": "Round 2 with the corner refiner",
    },
}


def package_graph(ml: Path, name: str, folder: Path, run_graph) -> dict:
    export = ml / "export" / name
    graph = json.loads((export / f"{name}.json").read_text())
    blob = (export / f"{name}.bin").read_bytes()
    sha = hashlib.sha256(blob).hexdigest()
    file = f"{name}-{sha[:12]}.bin"
    weights = np.frombuffer(blob, "<f2").astype(np.float32)
    shape = [1, *graph["input"]["shape"][1:]]
    heads = run_graph(graph, weights, np.full(shape, 0.5, np.float32))
    (folder / file).write_bytes(blob)
    graph["weights"] = {"file": file, "byteLength": len(blob), "sha256": sha}
    graph["selfTest"] = {"input": 0.5, "heads": [float(v) for v in heads]}
    if graph["input"]["shape"][0] > 1:
        # Batched graphs (the refiner): a different constant per batch item, so the self-test
        # also catches items that are mixed up in the batch layout.
        items = []
        for value in (0.0, 0.3, 0.7, 1.0)[: graph["input"]["shape"][0]]:
            out = run_graph(graph, weights, np.full(shape, value, np.float32))
            items.append({"input": value, "heads": [float(v) for v in out]})
        graph["selfTest"]["batch"] = items
    (folder / f"{name}.json").write_text(json.dumps(graph, indent=1) + "\n")
    print(f"{folder.name}/{file}: {len(blob)} bytes, self-test {np.round(heads, 4).tolist()}")
    return {"graph": f"{name}.json", "weights": file}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml", default=str(Path.home() / "src" / "scanoficus-ml"))
    args = parser.parse_args()
    ml = Path(args.ml)
    sys.path.insert(0, str(ml))
    from scanoficus_ml.rawgraph import run_graph

    for bundle, spec in BUNDLES.items():
        folder = MODELS / bundle
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        main_export = ml / "export" / spec["main"]
        threshold = float(json.loads((main_export / "threshold.json").read_text())["threshold"])
        main = package_graph(ml, spec["main"], folder, run_graph)
        manifest: dict = {"format": "scanoficus-nn-bundle/1", "name": bundle, "threshold": threshold,
                          "main": {"graph": main["graph"]}}
        refiner = None
        if spec["refiner"]:
            refiner = package_graph(ml, spec["refiner"], folder, run_graph)
            meta = json.loads((ml / "export" / spec["refiner"] / "refiner.json").read_text())
            pipeline = json.loads((main_export / "pipeline.json").read_text())
            assert pipeline["refine"] == f"net:{spec['refiner']}@" + "/".join(str(s) for s in meta["scales"]), pipeline
            refiner.update({"scales": meta["scales"], "crop": meta["crop"], "workSide": int(pipeline["workSide"])})
            manifest["refiner"] = {"graph": refiner["graph"], "scales": meta["scales"], "crop": meta["crop"],
                                   "workSide": refiner["workSide"]}
        (folder / "bundle.json").write_text(json.dumps(manifest, indent=1) + "\n")

        url = lambda f: f"new URL('../models/{bundle}/{f}', import.meta.url)"  # noqa: E731
        lines = [
            "// Generated by scripts/package_model.py; do not edit.",
            "import type { BundleSource } from './bundle.ts';",
            "",
            f"/** {spec['doc']}: the model files shipped in `models/{bundle}/`. */",
            f"export const {spec['const']}: BundleSource = {{",
            f"  name: '{bundle}',",
            f"  threshold: {threshold},",
            f"  main: {{ graphUrl: {url(main['graph'])}, weightsUrl: {url(main['weights'])} }},",
        ]
        if refiner:
            lines += [
                "  refiner: {",
                f"    graphUrl: {url(refiner['graph'])},",
                f"    weightsUrl: {url(refiner['weights'])},",
                f"    scales: {json.dumps(refiner['scales'])},",
                f"    crop: {refiner['crop']},",
                f"    workSide: {refiner['workSide']},",
                "  },",
            ]
        lines += ["};", ""]
        (PKG / "src" / f"bundle-{bundle}.ts").write_text("\n".join(lines))


if __name__ == "__main__":
    main()

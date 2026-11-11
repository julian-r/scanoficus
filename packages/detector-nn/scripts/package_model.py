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

Bundles: `docquadnet` (round 1, main network, threshold 0.85), `docquadnet2` (round 2 main
network, threshold 0.98, and the corner refiner `docquadnet2-refiner`) and `docquadnet3` (the
round-2 main network, shared with `docquadnet2/`, the refiner `docquadnet3-refiner` with a
corner-confidence output, `decision.json` from export/docquadnet3/ plus `mainSha256` and
`refinerSha256` (the weights it was fitted with; `loadBundle` rejects other weights), and
`coco-credits.csv`, the COCO images behind round 3 with Flickr author and title from
`fetch_flickr_credits.py`'s cache). The bundle module carries the decision's runtime fields
inline (no file fetch); `bundle.json` points to `decision.json` for self-hosting.
The attribution notice `models/LICENSE.md` is maintained by hand.

Usage (from the repository root, with the scanoficus-ml checkout and its virtualenv):

    PYTHONDONTWRITEBYTECODE=1 ~/src/scanoficus-ml/.venv/bin/python \\
        packages/detector-nn/scripts/package_model.py [--ml ~/src/scanoficus-ml]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
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
    # Round 3: the round-2 main network (files shared with `docquadnet2/`), the refiner with a
    # corner-confidence output and the presence decision of export/docquadnet3/decision.json.
    "docquadnet3": {
        "main": "docquadnet2",
        "mainBundle": "docquadnet2",
        "refiner": "docquadnet3-refiner",
        "decision": "docquadnet3",
        "const": "DOCQUADNET3",
        "doc": "Round 3: round-2 main network, refiner with corner scores and the verified presence decision",
    },
}

COCO_ROLES = {"A": "refiner-training", "B": "decision-fit", "val": "validation"}


def coco_credits(ml: Path, folder: Path, flickr_cache: Path) -> None:
    """`coco-credits.csv`: every COCO train2017 image of data/coco-train-negatives.json with its
    role in round 3 (subset A trained the refiner, subset B fitted the decision, the val split was
    used for validation), its Flickr license, the Flickr photo page (https://flic.kr/p/<base58
    id>), and for CC BY photos the author and title from Flickr's oEmbed endpoint
    (`fetch_flickr_credits.py`)."""
    import csv
    import io

    from fetch_flickr_credits import photo_id, short_url

    m = json.loads((ml / "data" / "coco-train-negatives.json").read_text())
    licenses = {int(k): v for k, v in m["licenses"].items()}
    flickr: dict[int, dict] = {}
    for line in flickr_cache.read_text().splitlines():
        d = json.loads(line)
        if d["status"] in ("ok", "unavailable"):
            flickr[d["id"]] = d
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["coco_id", "role", "license", "flickr_page", "status", "author", "title",
                     "flickr_license_now", "flickr_url"])
    for e in sorted(m["images"], key=lambda e: e["id"]):
        if e["split"] == "train":
            a = int(hashlib.sha256(f"coco{e['id']}".encode()).hexdigest(), 16) % 10 < 7
            role = COCO_ROLES["A" if a else "B"]
        else:
            role = COCO_ROLES["val"]
        pid = photo_id(e["flickrUrl"])
        page = short_url(pid)
        if e["license"] != 4:
            row = ["no-attribution-required", "", "", ""]
        else:
            d = flickr.get(pid)
            if d is None:
                raise SystemExit(f"photo {pid} not in {flickr_cache}: run fetch_flickr_credits.py first")
            if d["status"] == "unavailable":
                row = ["unavailable", "", "", ""]
            else:
                row = ["ok", d["author"], d["title"], d["license"]]
        writer.writerow([e["id"], role, e["license"], page, *row, e["flickrUrl"]])
    header = [
        "# COCO train2017 images used for DocQuadNet round 3 (scanoficus-ml data/coco-train-negatives.json).",
        f"# Source: {m['source']}. Selection: {m['rule']}.",
        *(f"# license {k}: {v['name']} ({v['url']})" for k, v in sorted(licenses.items())),
        "# Changes: cropped (corner crops), resized, rotated and augmented for training.",
        "# Columns: COCO image id (images.cocodataset.org/train2017/<id, 12 digits>.jpg); role in round 3;",
        "# license id at COCO collection time; Flickr photo page; status (ok: author and title from",
        "# Flickr's oEmbed endpoint; unavailable: the photo is no longer on Flickr; no-attribution-required:",
        "# license 7); author (Flickr name); title (empty: untitled on Flickr); the photo's license on",
        "# Flickr today; the Flickr image URL recorded by COCO.",
    ]
    (folder / "coco-credits.csv").write_text("\n".join(header) + "\n" + buf.getvalue())
    rows = buf.getvalue().splitlines()[1:]
    counts = {r: sum(1 for x in rows if f",{r}," in x) for r in COCO_ROLES.values()}
    status = {s_: sum(1 for x in rows if f",{s_}," in x) for s_ in ("ok", "unavailable", "no-attribution-required")}
    print(f"{folder.name}/coco-credits.csv: {len(rows)} images {counts} {status}")


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
    return {"graph": f"{name}.json", "weights": file, "sha256": sha}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml", default=str(Path.home() / "src" / "scanoficus-ml"))
    parser.add_argument("--flickr-cache", default=str(Path.home() / ".cache" / "scanoficus" / "flickr-oembed.jsonl"))
    parser.add_argument("--keep-credits", action="store_true",
                        help="keep the existing coco-credits.csv (while the Flickr fetch is incomplete)")
    args = parser.parse_args()
    ml = Path(args.ml)
    sys.path.insert(0, str(ml))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from scanoficus_ml.rawgraph import run_graph

    packaged: dict[str, dict] = {}
    for bundle, spec in BUNDLES.items():
        folder = MODELS / bundle
        credits = folder / "coco-credits.csv"
        kept = credits.read_bytes() if args.keep_credits and credits.exists() else None
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        if kept is not None:
            credits.write_bytes(kept)
        main_export = ml / "export" / spec["main"]
        decision = None
        if spec.get("decision"):
            decision_file = ml / "export" / spec["decision"] / "decision.json"
            decision = json.loads(decision_file.read_text())
            assert decision["main"] == spec["main"] and decision["refiner"] == spec["refiner"], decision
            threshold = None  # from the decision
        else:
            threshold = float(json.loads((main_export / "threshold.json").read_text())["threshold"])
        if spec.get("mainBundle"):
            # Shared with another bundle: reference its files instead of copying 688 kB.
            shared = packaged[spec["mainBundle"]]["main"]
            main = {"graph": f"../{spec['mainBundle']}/{shared['graph']}",
                    "weights": f"../{spec['mainBundle']}/{shared['weights']}", "sha256": shared["sha256"]}
        else:
            main = package_graph(ml, spec["main"], folder, run_graph)
        manifest: dict = {"format": "scanoficus-nn-bundle/1", "name": bundle, "threshold": threshold,
                          "main": {"graph": main["graph"]}}
        refiner = None
        if spec["refiner"]:
            refiner = package_graph(ml, spec["refiner"], folder, run_graph)
            meta = json.loads((ml / "export" / spec["refiner"] / "refiner.json").read_text())
            pipeline = json.loads((main_export / "pipeline.json").read_text())
            if decision:
                # Round 3 (scanoficus_ml/verify.py): scales from the decision, the work side of
                # verify.WORK_SIDE, the network's normalized corners times the work-image size.
                from scanoficus_ml.verify import WORK_SIDE

                assert decision["scales"] == meta["scales"] and WORK_SIDE == int(pipeline["workSide"])
                coordinates = "normalized"
            else:
                assert pipeline["refine"] == f"net:{spec['refiner']}@" + "/".join(str(s) for s in meta["scales"]), pipeline
                coordinates = "frame"
            refiner.update({"scales": meta["scales"], "crop": meta["crop"], "workSide": int(pipeline["workSide"]),
                            "coordinates": coordinates})
            manifest["refiner"] = {"graph": refiner["graph"], "scales": meta["scales"], "crop": meta["crop"],
                                   "workSide": refiner["workSide"], "coordinates": coordinates}
        if decision:
            # Pin the decision to the weights it was fitted with (added in the package copy).
            decision["mainSha256"] = main["sha256"]
            decision["refinerSha256"] = refiner["sha256"]
            (folder / "decision.json").write_text(json.dumps(decision, indent=1) + "\n")
            manifest["decision"] = "decision.json"
            if not args.keep_credits:
                coco_credits(ml, folder, Path(args.flickr_cache))
        (folder / "bundle.json").write_text(json.dumps(manifest, indent=1) + "\n")
        packaged[bundle] = {"main": main}

        url = lambda f: f"new URL('../models/{posixpath.normpath(f'{bundle}/{f}')}', import.meta.url)"  # noqa: E731
        lines = [
            "// Generated by scripts/package_model.py; do not edit.",
            "import type { BundleSource } from './bundle.ts';",
            "",
            f"/** {spec['doc']}: the model files shipped in `models/{bundle}/`. */",
            f"export const {spec['const']}: BundleSource = {{",
            f"  name: '{bundle}',",
            *([f"  threshold: {threshold},"] if threshold is not None else []),
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
                f"    coordinates: '{refiner['coordinates']}',",
                "  },",
            ]
        if decision:
            # Only the fields the runtime reads, inline: no fetch (and no data: URL) for it.
            keys = ["main", "refiner", "scales", "features", "weights", "bias", "threshold",
                    "mainSha256", "refinerSha256"]
            lines += ["  decision: {", *(f"    {k}: {json.dumps(decision[k])}," for k in keys), "  },"]
        lines += ["};", ""]
        (PKG / "src" / f"bundle-{bundle}.ts").write_text("\n".join(lines))


if __name__ == "__main__":
    main()

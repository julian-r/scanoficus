"""Tiny synthetic graphs in the raw format, with random fp16 weights, and their NumPy reference
outputs (`scanoficus_ml.rawgraph.run_graph`). They cover what DocQuadNet does not: non-square
and odd-sized tensors with stride 2, channel counts that are not a multiple of 4, a 2-channel
input, an add that cannot be fused (ELEMENTWISE shader), hswish, a padding-free 3×3
convolution and a sigmoid node.

Writes `test/fixtures/synthetic/<name>.json` (graph + `test: { input, output }`) and
`<name>.bin`. Usage: see make_fixtures.py (same virtualenv).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "test" / "fixtures" / "synthetic"


class Builder:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.chunks: list[np.ndarray] = []
        self.offset = 0
        self.nodes: list[dict] = []

    def blob(self, a: np.ndarray) -> dict:
        a = a.astype(np.float16)
        ref = {"offset": self.offset, "length": int(a.size), "shape": list(a.shape)}
        self.chunks.append(a.reshape(-1))
        self.offset += a.size
        return ref

    def conv(self, src, out, cin, cout, k, stride, pad, groups, act):
        fan = (cin // groups) * k * k
        w = self.rng.normal(0, 1.5 / np.sqrt(fan), (cout, cin // groups, k, k))
        b = self.rng.normal(0, 0.2, cout)
        self.nodes.append({"op": "conv", "in": src, "out": out, "weight": self.blob(w), "bias": self.blob(b),
                           "stride": stride, "pad": pad, "groups": groups, "act": act})

    def linear(self, src, out, n_in, n_out, act):
        w = self.rng.normal(0, 1.5 / np.sqrt(n_in), (n_out, n_in))
        b = self.rng.normal(0, 0.2, n_out)
        self.nodes.append({"op": "linear", "in": src, "out": out, "weight": self.blob(w), "bias": self.blob(b), "act": act})

    def graph(self, shape, output) -> tuple[dict, np.ndarray]:
        graph = {"format": "scanoficus-docquadnet-raw/1", "dtype": "float16",
                 "input": {"name": "input", "shape": shape}, "nodes": self.nodes, "outputs": {"heads": output}}
        return graph, np.concatenate(self.chunks)


def odd_mix(rng):
    """3×13×9 input: dense s2 (3→6, hswish), depthwise s2, two branches and an unfused add,
    a fused residual add, a 3×3 convolution without padding, linear layers and a sigmoid."""
    g = Builder(rng)
    g.conv("input", "a", 3, 6, 3, 2, 1, 1, "hswish")         # 6×7×5
    g.conv("a", "b", 6, 6, 3, 2, 1, 6, "relu6")              # 6×4×3 depthwise
    g.conv("b", "c", 6, 5, 1, 1, 0, 1, "none")               # 5×4×3
    g.conv("b", "d", 6, 5, 1, 1, 0, 1, "relu6")              # 5×4×3, last before the add
    g.nodes.append({"op": "add", "a": "c", "b": "d", "out": "e"})        # not fusable
    g.conv("e", "f", 5, 5, 3, 1, 1, 5, "hswish")             # depthwise, stride 1
    g.conv("f", "g", 5, 5, 1, 1, 0, 1, "none")
    g.nodes.append({"op": "add", "a": "f", "b": "g", "out": "h"})        # fused residual
    g.conv("h", "i", 5, 7, 3, 1, 0, 1, "none")               # 7×2×1, no padding
    g.nodes.append({"op": "flatten", "in": "i", "out": "j"})
    g.linear("j", "k", 14, 6, "hswish")
    g.linear("k", "l", 6, 3, "none")
    g.nodes.append({"op": "sigmoid", "in": "l", "out": "m"})
    return g.graph([1, 3, 13, 9], "m")


def two_channels(rng):
    """2×11×17 input: dense s2 (2→4), depthwise s2 on odd sizes, pointwise 4→9 (hswish),
    flatten and a linear layer with 5 outputs."""
    g = Builder(rng)
    g.conv("input", "a", 2, 4, 3, 2, 1, 1, "relu6")          # 4×6×9
    g.conv("a", "b", 4, 4, 3, 2, 1, 4, "none")               # 4×3×5 depthwise
    g.conv("b", "c", 4, 9, 1, 1, 0, 1, "hswish")             # 9×3×5
    g.nodes.append({"op": "flatten", "in": "c", "out": "d"})
    g.linear("d", "e", 135, 5, "none")
    return g.graph([1, 2, 11, 17], "e")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml", default=str(Path.home() / "src" / "scanoficus-ml"))
    args = parser.parse_args()
    sys.path.insert(0, args.ml)
    from scanoficus_ml.rawgraph import run_graph

    OUT.mkdir(parents=True, exist_ok=True)
    for seed, (name, make) in enumerate([("odd-mix", odd_mix), ("two-channels", two_channels)]):
        rng = np.random.default_rng(100 + seed)
        graph, weights = make(rng)
        x = rng.random(graph["input"]["shape"]).astype(np.float32)
        y = run_graph(graph, weights.astype(np.float32), x)
        graph["test"] = {"input": [float(v) for v in x.reshape(-1)], "output": [float(v) for v in y.reshape(-1)]}
        (OUT / f"{name}.json").write_text(json.dumps(graph) + "\n")
        (OUT / f"{name}.bin").write_bytes(weights.astype("<f2").tobytes())
        print(name, np.round(y.reshape(-1), 4).tolist())


if __name__ == "__main__":
    main()

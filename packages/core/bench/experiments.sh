#!/bin/sh
# Runs tuning-split experiments one after another: each argument is "name=<json options>".
# Results: eval-out/bench/<name>/; score them with `node bench/tune.ts --runs <names>`.
set -e
cd "$(dirname "$0")/.."
for spec in "$@"; do
  name="${spec%%=*}"
  options="${spec#*=}"
  node bench/run.ts --split tune --datasets smartdoc,midv500,negatives --sizes 384 \
    --min-score 0 --run "$name" --options "$options" > "/tmp/bench-$name.log" 2>&1
  echo "$name done"
done

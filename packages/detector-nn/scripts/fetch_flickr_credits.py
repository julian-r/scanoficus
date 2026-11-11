"""Fetches author and title of the CC BY COCO photos behind round 3 from Flickr's public oEmbed
endpoint (no API key), for `models/docquadnet3/coco-credits.csv` (CC BY 2.0 asks for the
author's name if supplied, the title and the URI).

Input: scanoficus-ml data/coco-train-negatives.json (license id 4 = CC BY 2.0; id 7, "no known
copyright restrictions", needs no author and is skipped). For each photo:
`https://www.flickr.com/services/oembed/?format=json&url=https://flic.kr/p/<base58 photo id>`.
Results go to a JSONL cache (one line per photo id: author, author URL, title, page, the current
license, or `unavailable` with the HTTP status), so the run can be stopped and resumed; photos
already in the cache are skipped. Rate: 4 requests/s; 429 and 5xx are retried with backoff.

Usage (any Python 3):
    python3 packages/detector-nn/scripts/fetch_flickr_credits.py \\
        [--ml ~/src/scanoficus-ml] [--cache ~/.cache/scanoficus/flickr-oembed.jsonl] [--rate 4]
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE58 = "123456789abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ"
USER_AGENT = "scanoficus-attribution/1.0 (+https://github.com/julian-r/scanoficus)"


def base58(n: int) -> str:
    """Flickr's short-URL alphabet (flic.kr/p/<id>)."""
    s = ""
    while n:
        n, r = divmod(n, 58)
        s = BASE58[r] + s
    return s


def photo_id(flickr_url: str) -> int:
    """http://farmN.staticflickr.com/<server>/<id>_<secret>_z.jpg → id."""
    return int(flickr_url.rsplit("/", 1)[1].split("_", 1)[0])


def short_url(pid: int) -> str:
    return f"https://flic.kr/p/{base58(pid)}"


def fetch(pid: int, retries: int = 6) -> dict:
    url = "https://www.flickr.com/services/oembed/?format=json&url=" + urllib.parse.quote(short_url(pid), safe="")
    delay = 2.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.loads(r.read())
            return {
                "id": pid,
                "status": "ok",
                "author": d.get("author_name", ""),
                "authorUrl": d.get("author_url", ""),
                "title": d.get("title", ""),
                "page": d.get("web_page", ""),
                "license": d.get("license", ""),
                "licenseId": d.get("license_id", ""),
            }
        except urllib.error.HTTPError as e:
            if e.code in (404, 410, 403, 401):
                return {"id": pid, "status": "unavailable", "http": e.code}
            if attempt == retries - 1:
                return {"id": pid, "status": "error", "http": e.code}
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if attempt == retries - 1:
                return {"id": pid, "status": "error", "error": str(e)[:200]}
        time.sleep(delay)
        delay = min(delay * 2, 120)
    return {"id": pid, "status": "error"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml", default=str(Path.home() / "src" / "scanoficus-ml"))
    parser.add_argument("--cache", default=str(Path.home() / ".cache" / "scanoficus" / "flickr-oembed.jsonl"))
    parser.add_argument("--rate", type=float, default=4.0, help="requests per second")
    args = parser.parse_args()
    images = json.loads((Path(args.ml) / "data" / "coco-train-negatives.json").read_text())["images"]
    ids = sorted({photo_id(e["flickrUrl"]) for e in images if e["license"] == 4})
    cache = Path(args.cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if cache.exists():
        for line in cache.read_text().splitlines():
            d = json.loads(line)
            if d["status"] in ("ok", "unavailable"):
                done.add(d["id"])
    todo = [i for i in ids if i not in done]
    print(f"{len(ids)} CC BY photos, {len(done)} cached, {len(todo)} to fetch", flush=True)
    interval = 1.0 / args.rate
    with cache.open("a") as out:
        for k, pid in enumerate(todo):
            t = time.monotonic()
            out.write(json.dumps(fetch(pid)) + "\n")
            out.flush()
            if k % 200 == 0:
                print(f"{k}/{len(todo)}", flush=True)
            time.sleep(max(0.0, interval - (time.monotonic() - t)))
    print("done", flush=True)


if __name__ == "__main__":
    main()

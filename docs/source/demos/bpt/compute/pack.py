"""Write the demo's data, ``../assets/data.js``, with its photos inline.

A 224 x 224 photo's BPT has 50,176 leaves (its pixels) and 50,175 merges, too many to ship. The
demo starts from the partition into K regions that the first merges reach, and keeps the K - 1
merges after it: K is the smallest power of two from 4,096 up whose partition still contains every
segment the explainer touched, so the replay is exact. The data per photo:

- ``base``: the K regions, a run-length code of the 224 x 224 label map (row by row);
- ``merges``: the K - 1 merges in order, two children each (0..K-1 base regions, K.. merges);
- ``firstMerge``: the number of merges before the base partition, to count merges as in the BPT;
- for BPT and AA: ``segments`` (a tree node or a rectangle, and the segment whose pixels are
  present with it), ``coalitions`` (segment, f(S u AB), f(S), depth) and ``events`` (the split
  coalition, the first of its four children), as logged by explain.py.

Integer arrays are base64 of little-endian uint16/int32, so that the file stays small.

Usage::

    python pack.py
"""

from __future__ import annotations

import argparse
import base64
import json
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def b64(array: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


def leaf_pixels(n_pix: int, left: np.ndarray, right: np.ndarray, U: int, roots: list[int]):
    """Label every pixel (leaf) by the root of ``roots`` above it."""
    label = np.full(U + len(left), -1, dtype=np.int64)
    for k, r in enumerate(roots):
        label[r] = k
    for n in range(U + len(left) - 1, U - 1, -1):  # children have smaller indices
        if label[n] >= 0:
            label[left[n - U]] = label[n]
            label[right[n - U]] = label[n]
    return label[:n_pix]


def compact(result: dict, title: str) -> dict:
    U = result["tree"]["U"]
    left = np.array(result["tree"]["left"])
    right = np.array(result["tree"]["right"])
    N = U + len(left)
    needed = {seg[0] for seg in result["BPT"]["segments"]}
    for K in (4096, 8192, 16384, 32768):  # 4,096 at least: the replay starts from small regions
        first = U - K  # merges done before the base partition
        cut = U + first  # nodes below this index exist in the base partition
        kept = range(cut, N)
        base = sorted(
            {int(c) for n in kept for c in (left[n - U], right[n - U]) if c < cut}
        )
        if all(s >= cut or s in set(base) for s in needed):
            break
    index = {node: k for k, node in enumerate(base)}
    index.update({node: K + (node - cut) for node in kept})
    merges = np.array([[index[left[n - U]], index[right[n - U]]] for n in kept]).ravel()
    labels = leaf_pixels(U, left, right, U, base)
    runs, start = [], 0
    for i in range(1, U + 1):  # row-major run-length code: (label, length) pairs
        if i == U or labels[i] != labels[start] or i % 224 == 0:
            runs += [labels[start], i - start]
            start = i
    out = {
        "id": result["id"],
        "title": title,
        "label": result["label"],
        "full": round(result["full"], 4),
        "empty": round(result["empty"], 4),
        "K": K,
        "firstMerge": int(first),
        "totalMerges": int(len(left)),
        "base": b64(np.array(runs), "<u2"),
        "merges": b64(merges, "<u2"),
    }
    for method in ("BPT", "AA"):
        log = result[method]
        if method == "BPT":
            shapes = [index[s] for s, _ in log["segments"]]
        else:
            shapes = [v for s, _ in log["segments"] for v in s]  # y0, y1, x0, x1
        out[method] = {
            "shapes": b64(np.array(shapes), "<u2"),
            "parents": b64(np.array([p for _, p in log["segments"]]), "<i4"),
            "coalitions": b64(np.array([c[0] for c in log["coalitions"]]), "<u4"),
            "depth": b64(np.array([c[3] for c in log["coalitions"]]), "<u2"),
            "f": b64(np.array([v for c in log["coalitions"] for v in (c[1], c[2])]), "<f4"),
            "events": b64(np.array(log["events"]).ravel(), "<u4"),
        }
    print(
        f"{result['id']}: K = {K} base regions, {len(log['events'])} AA splits, "
        f"{len(result['BPT']['events'])} BPT splits"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--assets", type=Path, default=HERE.parent / "assets")
    args = parser.parse_args()
    config = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
    photos = []
    for photo in config["photos"]:
        path = args.results / f"{photo['id']}.json"
        if not path.exists():
            print(f"{photo['id']}: not explained, left out")
            continue
        packed = compact(json.loads(path.read_text(encoding="utf-8")), photo["title"])
        # the photo and its masking background, inline: the page reads their pixels, which a
        # data URI allows from any origin (and from disk)
        for key, suffix in (("photo", ""), ("blur", "-blur")):
            jpg = (args.results / f"{photo['id']}{suffix}.jpg").read_bytes()
            packed[key] = "data:image/jpeg;base64," + base64.b64encode(jpg).decode("ascii")
        photos.append(packed)
    # data.js: every photo's header (its title, prediction and images) and the first photo's tree
    # and logs; the others' are in photo-<id>.js, which the page loads when they are chosen
    args.assets.mkdir(parents=True, exist_ok=True)
    for old in args.assets.glob("photo-*.js"):
        old.unlink()
    header = ("id", "title", "label", "full", "empty", "photo", "blur")
    index = []
    for k, packed in enumerate(photos):
        if k == 0:
            index.append(packed)
            continue
        index.append({key: packed[key] for key in header} | {"file": f"photo-{packed['id']}.js"})
        rest = {key: value for key, value in packed.items() if key not in header}
        text = json.dumps(rest, separators=(",", ":"))
        (args.assets / f"photo-{packed['id']}.js").write_text(
            f"// The ShapBPT demo's data for one photo (see demo.js), written by pack.py.\n"
            f"(window.SHAPBPT_DEMO_PHOTOS ||= {{}})[{json.dumps(packed['id'])}] = {text};\n",
            encoding="utf-8",
        )
        print(f"photo-{packed['id']}.js: {len(text) / 1024:.0f} KiB")
    data = {"model": config.get("model", "ResNet-50"), "budget": config["budget"], "photos": index}
    text = json.dumps(data, separators=(",", ":"))
    (args.assets / "data.js").write_text(
        "// The ShapBPT demo's data (see demo.js), written by ../compute/pack.py.\n"
        f"window.SHAPBPT_DEMO = {text};\n",
        encoding="utf-8",
    )
    print(f"data.js: {len(photos)} photos, {len(text) / 1024:.0f} KiB")


if __name__ == "__main__":
    main()

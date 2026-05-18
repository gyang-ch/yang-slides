"""
Build a slim JSON file consumed by the reveal.js UMAP-network slide.

Computes UMAP coordinates + K-Means clusters from
`illustrations.embeddings.jsonl`, joins with `illustrations.public.jsonl`,
selects a stratified, spread-out subset for which thumbnails will be
fetched directly from the public Azure blob (botany-data container), and
writes everything as `umap_data.json` next to the slides.
"""

import json
import os
import random

import numpy as np
import umap
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "illustrations.embeddings.jsonl")
PUBLIC_PATH = os.path.join(BASE_DIR, "illustrations.public.jsonl")
OUTPUT_PATH = os.path.join(BASE_DIR, "umap_data.json")

K = 20
N_DISPLAY = 600
RANDOM_SEED = 42

BLOB_BASE = "https://phytovision.blob.core.windows.net/botany-data/"
SAS = (
    "sv=2024-11-04&ss=bfqt&srt=sco&sp=rwdlacupiytfx"
    "&se=2030-02-15T18:31:09Z&st=2026-03-29T09:16:09Z"
    "&spr=https&sig=prvNh26kGPtCZ%2BcJWtZ1HDzihcxCYDNiIrv3hfu3VVY%3D"
)

PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#dddddd",
]


def load_embeddings(path):
    out = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            out[rec["illustration_id"]] = np.array(
                rec["embedding"]["vector"], dtype=np.float32
            )
    return out


def load_public(path):
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
    return out


def spread_sample(ids, lookup, n_keep, min_frac=0.022):
    """Greedy: keep at most n_keep ids, no pair closer than min_frac in both axes."""
    pool = list(ids)
    random.shuffle(pool)
    kept = []
    kept_xy = []
    for iid in pool:
        cx, cy = lookup[iid]
        too_close = any(
            abs(cx - kx) < min_frac and abs(cy - ky) < min_frac for kx, ky in kept_xy
        )
        if too_close:
            continue
        kept.append(iid)
        kept_xy.append((cx, cy))
        if len(kept) >= n_keep:
            break
    return kept


def main():
    print("Loading embeddings …")
    embeddings = load_embeddings(EMBEDDINGS_PATH)
    print(f"  {len(embeddings):,} embeddings")

    print("Loading public metadata …")
    records = load_public(PUBLIC_PATH)
    print(f"  {len(records):,} records")

    ids = [r["illustration_id"] for r in records if r["illustration_id"] in embeddings]
    print(f"  {len(ids):,} aligned")

    X = np.stack([embeddings[i] for i in ids])
    Xn = normalize(X)

    print(f"Running K-Means (k={K}) …")
    km = KMeans(n_clusters=K, random_state=RANDOM_SEED, n_init=10, max_iter=500)
    labels = km.fit_predict(Xn)
    id_to_cluster = {i: int(l) for i, l in zip(ids, labels)}

    print("Running UMAP …")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=20,
        min_dist=0.25,
        metric="cosine",
        random_state=RANDOM_SEED,
        low_memory=False,
    )
    xy = reducer.fit_transform(Xn)

    # Normalise UMAP coords into [0, 1] for portability.
    xs, ys = xy[:, 0], xy[:, 1]
    xn = (xs - xs.min()) / (xs.max() - xs.min())
    yn = (ys - ys.min()) / (ys.max() - ys.min())
    id_to_xy = {i: (float(xn[k]), float(yn[k])) for k, i in enumerate(ids)}

    id_to_crop = {r["illustration_id"]: r["illustration"]["crop_image"] for r in records}

    cluster_to_ids = {}
    for i, lbl in id_to_cluster.items():
        cluster_to_ids.setdefault(lbl, []).append(i)

    random.seed(RANDOM_SEED)
    per_cluster = max(1, (N_DISPLAY * 3) // K)
    candidates = []
    for lbl, pool in cluster_to_ids.items():
        candidates.extend(random.sample(pool, min(per_cluster, len(pool))))

    display = set(spread_sample(candidates, id_to_xy, N_DISPLAY, min_frac=0.022))
    print(f"  {len(display)} thumbnails selected for display")

    nodes = []
    for iid in ids:
        x, y = id_to_xy[iid]
        node = {
            "x": round(x, 4),
            "y": round(y, 4),
            "c": id_to_cluster[iid],
        }
        if iid in display:
            crop = id_to_crop.get(iid, "")
            node["img"] = f"{BLOB_BASE}{crop}?{SAS}"
        nodes.append(node)

    output = {
        "k": K,
        "colors": PALETTE[:K],
        "cluster_counts": {str(c): len(cluster_to_ids[c]) for c in sorted(cluster_to_ids)},
        "n_total": len(nodes),
        "n_thumbs": len(display),
        "nodes": nodes,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, separators=(",", ":"))
    print(f"Wrote {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

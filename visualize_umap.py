"""
Compute K-Means clusters on illustration embeddings, update public metadata,
then produce a UMAP thumbnail network graph as a professional PNG.
"""

import json
import os
import random
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors
from PIL import Image, ImageOps, ImageDraw
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import umap

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "illustrations.embeddings.jsonl")
PUBLIC_PATH = os.path.join(BASE_DIR, "illustrations.public.jsonl")
OUTPUT_PATH = os.path.join(BASE_DIR, "illustrations_umap.png")

K = 20          # K-Means clusters
N_DISPLAY = 500 # illustrations to show in graph
THUMB_SIZE = 48 # px — thumbnail edge length in data space render
RANDOM_SEED = 42

# ── 1. LOAD EMBEDDINGS ───────────────────────────────────────────────────────

print("Loading embeddings …")
embeddings_by_id = {}
with open(EMBEDDINGS_PATH) as f:
    for line in f:
        rec = json.loads(line)
        vec = rec["embedding"]["vector"]
        embeddings_by_id[rec["illustration_id"]] = np.array(vec, dtype=np.float32)

print(f"  {len(embeddings_by_id)} embeddings loaded")

# ── 2. LOAD PUBLIC METADATA ──────────────────────────────────────────────────

print("Loading public metadata …")
public_records = []
with open(PUBLIC_PATH) as f:
    for line in f:
        public_records.append(json.loads(line))

print(f"  {len(public_records)} records loaded")

# ── 3. BUILD ALIGNED MATRIX ─────────────────────────────────────────────────

ids_with_emb = [r["illustration_id"] for r in public_records
                if r["illustration_id"] in embeddings_by_id]

print(f"  {len(ids_with_emb)} records have matching embeddings")

X = np.stack([embeddings_by_id[i] for i in ids_with_emb], axis=0)
X_norm = normalize(X)  # cosine ≈ euclidean on L2-normalised vectors

# ── 4. K-MEANS CLUSTERING ────────────────────────────────────────────────────

print(f"Running K-Means (k={K}) …")
km = KMeans(n_clusters=K, random_state=RANDOM_SEED, n_init=10, max_iter=500)
labels = km.fit_predict(X_norm)

# Map id → cluster label
id_to_cluster = {iid: int(lbl) for iid, lbl in zip(ids_with_emb, labels)}

cluster_counts = {}
for lbl in labels:
    cluster_counts[int(lbl)] = cluster_counts.get(int(lbl), 0) + 1
for c, n in sorted(cluster_counts.items()):
    print(f"  cluster {c:2d}: {n} illustrations")

# ── 5. UPDATE PUBLIC JSONL ───────────────────────────────────────────────────

print("Writing updated public metadata …")
tmp_path = PUBLIC_PATH + ".tmp"
with open(tmp_path, "w") as f:
    for rec in public_records:
        iid = rec["illustration_id"]
        if iid in id_to_cluster:
            rec.setdefault("kmeans_cluster", {})
            rec["kmeans_cluster"] = {
                "algorithm": "kmeans",
                "params": {"k": K, "metric": "cosine (l2-normalised euclidean)"},
                "cluster_id": id_to_cluster[iid],
            }
        f.write(json.dumps(rec) + "\n")
os.replace(tmp_path, PUBLIC_PATH)
print("  illustrations.public.jsonl updated ✓")

# ── 6. UMAP PROJECTION ───────────────────────────────────────────────────────

print("Running UMAP …")
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=20,
    min_dist=0.25,
    metric="cosine",
    random_state=RANDOM_SEED,
    low_memory=False,
)
embedding_2d = reducer.fit_transform(X_norm)

id_to_umap = {iid: embedding_2d[i] for i, iid in enumerate(ids_with_emb)}

# ── 7. STRATIFIED + SPREAD-OUT SAMPLE ───────────────────────────────────────

random.seed(RANDOM_SEED)
per_cluster = N_DISPLAY // K  # ≈ 25
candidate_ids = []
cluster_to_ids = {}
for iid, lbl in id_to_cluster.items():
    cluster_to_ids.setdefault(lbl, []).append(iid)

for lbl in range(K):
    pool = cluster_to_ids.get(lbl, [])
    chosen = random.sample(pool, min(per_cluster * 2, len(pool)))  # 2× over-sample
    candidate_ids.extend(chosen)

# Greedy minimum-distance filter to reduce thumbnail overlap
# Work in normalised [0,1] coordinates
def spread_sample(ids_pool, umap_lookup, n_keep, min_frac=0.018):
    """Keep up to n_keep points such that no two are closer than min_frac of range."""
    coords = np.array([[umap_lookup[i][0], umap_lookup[i][1]] for i in ids_pool])
    xr = float(coords[:, 0].max() - coords[:, 0].min()) or 1.0
    yr = float(coords[:, 1].max() - coords[:, 1].min()) or 1.0
    min_dx = min_frac * xr
    min_dy = min_frac * yr

    random.shuffle(ids_pool)
    kept = []
    kept_coords = []
    for i, iid in enumerate(ids_pool):
        cx, cy = umap_lookup[iid]
        too_close = any(
            abs(cx - kx) < min_dx and abs(cy - ky) < min_dy
            for kx, ky in kept_coords
        )
        if not too_close:
            kept.append(iid)
            kept_coords.append((cx, cy))
        if len(kept) >= n_keep:
            break
    return kept

selected_ids = spread_sample(candidate_ids, id_to_umap, N_DISPLAY, min_frac=0.018)

print(f"  {len(selected_ids)} illustrations selected for display")

# ── 8. BUILD COLOUR PALETTE ──────────────────────────────────────────────────

# Professional, perceptually distinct palette
BASE_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#ffffff",
]
cluster_colors = {i: BASE_COLORS[i % len(BASE_COLORS)] for i in range(K)}

# ── 9. LOAD THUMBNAILS ───────────────────────────────────────────────────────

print("Loading thumbnails …")
THUMB_PX = 52  # rendered thumbnail pixel size

def load_thumb(crop_image_rel: str, size: int, border_color: str) -> np.ndarray | None:
    """Load, resize to fit within size, add coloured border, keep aspect ratio."""
    path = os.path.join(BASE_DIR, crop_image_rel)
    if not os.path.isfile(path):
        return None
    try:
        img = Image.open(path).convert("RGBA")
        inner = size - 6  # leave room for border + shadow
        img.thumbnail((inner, inner), Image.LANCZOS)
        w, h = img.width, img.height
        # White background for the image area only (no padding)
        bg = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        # Coloured border exactly around the image
        r, g, b = int(border_color[1:3], 16), int(border_color[3:5], 16), int(border_color[5:7], 16)
        bordered = ImageOps.expand(bg.convert("RGB"), border=3,
                                   fill=(r, g, b)).convert("RGBA")
        return np.array(bordered)
    except Exception:
        return None

# Build a lookup: illustration_id → crop_image path
id_to_crop = {r["illustration_id"]: r["illustration"]["crop_image"]
              for r in public_records}

thumbs = {}
for iid in selected_ids:
    crop = id_to_crop.get(iid, "")
    cluster = id_to_cluster.get(iid, 0)
    color = cluster_colors[cluster]
    arr = load_thumb(crop, THUMB_PX, color)
    if arr is not None:
        thumbs[iid] = arr

print(f"  {len(thumbs)} thumbnails loaded")

# ── 10. DRAW FIGURE ──────────────────────────────────────────────────────────

print("Drawing figure …")

FIG_W, FIG_H = 24, 20  # inches
DPI = 150

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

# UMAP coords for selected ids
xs = np.array([id_to_umap[i][0] for i in selected_ids])
ys = np.array([id_to_umap[i][1] for i in selected_ids])

# Axis limits with padding
pad = 0.08
xmin, xmax = xs.min(), xs.max()
ymin, ymax = ys.min(), ys.max()
xr, yr = xmax - xmin, ymax - ymin
ax.set_xlim(xmin - pad * xr, xmax + pad * xr)
ax.set_ylim(ymin - pad * yr, ymax + pad * yr)

# Soft grid
ax.grid(color="#1f2937", linewidth=0.4, zorder=0)
ax.set_axisbelow(True)

# Cluster convex-hull fills (subtle)
from scipy.spatial import ConvexHull

for cluster_id in range(K):
    c_ids = [i for i in selected_ids if id_to_cluster.get(i) == cluster_id]
    if len(c_ids) < 4:
        continue
    cx = np.array([id_to_umap[i][0] for i in c_ids])
    cy = np.array([id_to_umap[i][1] for i in c_ids])
    try:
        hull = ConvexHull(np.column_stack([cx, cy]))
        verts = np.column_stack([cx[hull.vertices], cy[hull.vertices]])
        poly = plt.Polygon(verts, closed=True,
                           facecolor=cluster_colors[cluster_id],
                           alpha=0.06, edgecolor=cluster_colors[cluster_id],
                           linewidth=0.8, linestyle="--", zorder=1)
        ax.add_patch(poly)
    except Exception:
        pass

# Scatter dots as faint background for non-displayed points
all_xs = embedding_2d[:, 0]
all_ys = embedding_2d[:, 1]
all_colors = [cluster_colors[id_to_cluster[i]] for i in ids_with_emb]
ax.scatter(all_xs, all_ys, c=all_colors, s=2, alpha=0.12, linewidths=0, zorder=2)

# Place thumbnails — zoom is set per-image so each fits in ~THUMB_PX screen px
TARGET_PX = THUMB_PX  # desired display size in screen pixels at DPI
for iid in selected_ids:
    if iid not in thumbs:
        continue
    x, y = id_to_umap[iid]
    img_arr = thumbs[iid]
    h_px = img_arr.shape[0]
    zoom = TARGET_PX / h_px
    im = OffsetImage(img_arr, zoom=zoom, interpolation="lanczos")
    ab = AnnotationBbox(
        im, (x, y),
        frameon=False,
        pad=0,
        zorder=3,
    )
    ax.add_artist(ab)

# ── Legend ───────────────────────────────────────────────────────────────────

legend_handles = []
for c in range(K):
    patch = plt.Rectangle((0, 0), 1, 1,
                           facecolor=cluster_colors[c],
                           edgecolor="white", linewidth=0.5,
                           label=f"Cluster {c:02d}  (n={cluster_counts.get(c,0)})")
    legend_handles.append(patch)

leg = ax.legend(
    handles=legend_handles,
    title="K-Means Clusters  (k=20)",
    title_fontsize=9,
    fontsize=7.5,
    loc="lower left",
    ncol=2,
    framealpha=0.75,
    facecolor="#161b22",
    edgecolor="#30363d",
    labelcolor="white",
    handlelength=1.2,
    handleheight=1.2,
    borderpad=0.8,
    labelspacing=0.4,
)
leg.get_title().set_color("#c9d1d9")

# ── Axis labels & title ───────────────────────────────────────────────────────

ax.set_xlabel("UMAP dimension 1", color="#8b949e", fontsize=11, labelpad=8)
ax.set_ylabel("UMAP dimension 2", color="#8b949e", fontsize=11, labelpad=8)
ax.tick_params(colors="#8b949e", labelsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor("#30363d")

ax.set_title(
    "UMAP Embedding Space of Plant Illustrations",
    color="#e6edf3", fontsize=16, fontweight="bold", pad=18,
)

subtitle = (
    f"n = {len(ids_with_emb):,} illustrations  ·  "
    f"{len(selected_ids)} displayed  ·  "
    f"768-d CLIP embeddings → UMAP(cosine)  ·  "
    f"K-Means k={K}"
)
fig.text(0.5, 0.935, subtitle,
         ha="center", va="top", color="#8b949e", fontsize=9,
         transform=fig.transFigure)

plt.tight_layout(rect=[0, 0.01, 1, 0.935])

print(f"Saving {OUTPUT_PATH} …")
fig.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)
print("Done ✓")

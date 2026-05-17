"""Stage 05b: PCA + UMAP reduction of neural embeddings.

For each (model, layer) pair in {whisper, xlsr} x layers, fit:
  - PCA -> d=50 (for clustering [09] and LME [09], truncated to first 5 PCs)
  - UMAP -> d=2 (for visualisation [07])

Design notes:
- One PCA AND one UMAP per (model, layer): scales differ across layers/models
  (LayerNorm in Whisper, none in XLS-R), a single global fit would break
  geometry.
- PCA d=50 only; the d=2 used for visualisation is UMAP, not PCA[:, :2].
- UMAP is non-deterministic under parallelism; we force n_jobs=1 to make
  the output reproducible bit-for-bit given random_state.
- The RSMs / distance matrices used in [07] (RSA, Mantel) are NOT computed
  from these reduced embeddings: they use the 1024-dim originals.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import umap
import yaml
from sklearn.decomposition import PCA


def reduce_one(
    npz_in: Path,
    pca_out: Path,
    umap_out: Path,
    pca_n_components: int,
    umap_n_components: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    random_state: int,
) -> None:
    """Fit PCA and UMAP on embeddings in npz_in, write two reduced npz files."""
    print(f"[normalise_neural] {npz_in.name}")
    data = np.load(npz_in, allow_pickle=False)

    # float16 -> float32 before reduction (numerical safety).
    emb = data["embeddings"].astype(np.float32)
    n_tokens, d_in = emb.shape

    # ---- PCA ----
    pca = PCA(n_components=pca_n_components, random_state=random_state)
    emb_pca = pca.fit_transform(emb).astype(np.float32)
    print(f"  PCA  ({n_tokens}, {d_in}) -> {emb_pca.shape}  "
          f"| PC1 = {pca.explained_variance_ratio_[0]:.3f}  "
          f"cum@{pca_n_components} = {pca.explained_variance_ratio_.sum():.3f}")

    pca_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pca_out,
        embeddings_pca=emb_pca,
        token_ids=data["token_ids"],
        n_frames_used=data["n_frames_used"],
        layer=data["layer"],
        model=data["model"],
        pooling=data["pooling"],
        frame_rate_hz=data["frame_rate_hz"],
        pca_n_components=np.int32(pca_n_components),
        pca_explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        pca_source_file=str(npz_in.name),
    )

    # ---- UMAP ----
    # n_jobs=1 forces deterministic output given random_state (parallel TBB
    # would introduce small non-reproducible variations).
    reducer = umap.UMAP(
        n_components=umap_n_components,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric="cosine",
        random_state=random_state,
        n_jobs=1,
    )
    emb_umap = reducer.fit_transform(emb).astype(np.float32)
    print(f"  UMAP ({n_tokens}, {d_in}) -> {emb_umap.shape}  "
          f"| n_neighbors={umap_n_neighbors}  min_dist={umap_min_dist}")

    umap_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        umap_out,
        embeddings_umap=emb_umap,
        token_ids=data["token_ids"],
        n_frames_used=data["n_frames_used"],
        layer=data["layer"],
        model=data["model"],
        pooling=data["pooling"],
        frame_rate_hz=data["frame_rate_hz"],
        umap_n_components=np.int32(umap_n_components),
        umap_n_neighbors=np.int32(umap_n_neighbors),
        umap_min_dist=np.float32(umap_min_dist),
        umap_metric="cosine",
        umap_source_file=str(npz_in.name),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    cfg = config["normalise_neural"]

    in_dir = Path(cfg["input_dir"])
    out_dir = Path(cfg["output_dir"])
    pca_suffix = cfg["pca_output_suffix"]
    umap_suffix = cfg["umap_output_suffix"]

    pca_n_components = cfg["pca_n_components"]
    umap_n_components = cfg["umap_n_components"]
    umap_n_neighbors = cfg["umap_n_neighbors"]
    umap_min_dist = cfg["umap_min_dist"]
    random_state = cfg["random_state"]

    # Discover (prefix, layer) pairs from upstream extract_neural_* configs.
    jobs: list[tuple[str, int]] = []
    for upstream_key in ("extract_neural_whisper", "extract_neural_xlsr"):
        up = config[upstream_key]
        for L in up["layers"]:
            jobs.append((up["output_prefix"], L))

    for prefix, L in jobs:
        npz_in = in_dir / f"{prefix}_L{L:02d}.npz"
        pca_out = out_dir / f"{prefix}_L{L:02d}_{pca_suffix}.npz"
        umap_out = out_dir / f"{prefix}_L{L:02d}_{umap_suffix}.npz"
        if not npz_in.exists():
            raise FileNotFoundError(
                f"Expected upstream file not found: {npz_in}. "
                f"Did stage extract_neural_* run successfully?"
            )
        reduce_one(
            npz_in, pca_out, umap_out,
            pca_n_components=pca_n_components,
            umap_n_components=umap_n_components,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
            random_state=random_state,
        )

    print(f"[normalise_neural] done: {len(jobs)} files reduced (PCA d={pca_n_components}, UMAP d={umap_n_components}).")


if __name__ == "__main__":
    main()
"""Stage 05b: PCA reduction of neural embeddings.

For each (model, layer) pair in {whisper, xlsr} x layers, fit a PCA on
the 1024-dim embeddings of that single file and project to d=50.

Design notes:
- One PCA per (model, layer): mixing layers or models in the same fit would
  break the geometry (different scales, different representational roles).
- d=50 only; the d=2 used for visualisation is just emb_pca[:, :2].
- Explained variance ratio is stored as a metadata key for downstream
  discussion (how informative are the first PCs of L4 vs L12 vs L20?).
- The RSMs / distance matrices used in [07] (RSA, Mantel) are NOT computed
  from these PCA-reduced embeddings: they use the 1024-dim originals.
  PCA outputs feed [07] (2D viz), [09] (clustering), [09] (LME, truncated
  to first 5 PCs).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from sklearn.decomposition import PCA


def reduce_one(
    npz_in: Path,
    npz_out: Path,
    n_components: int,
    random_state: int,
) -> None:
    """Fit PCA on the embeddings in npz_in, write reduced npz to npz_out."""
    print(f"[normalise_neural] {npz_in.name}")
    data = np.load(npz_in, allow_pickle=False)

    # Cast to float32 before PCA (npz is stored float16 for size; computations
    # in float16 are numerically unsafe).
    emb = data["embeddings"].astype(np.float32)
    n_tokens, d_in = emb.shape

    pca = PCA(n_components=n_components, random_state=random_state)
    emb_pca = pca.fit_transform(emb).astype(np.float32)

    print(f"  in:  ({n_tokens}, {d_in})  ->  out: {emb_pca.shape}")
    print(f"  explained variance: first PC = {pca.explained_variance_ratio_[0]:.3f}, "
          f"cum @ {n_components} PCs = {pca.explained_variance_ratio_.sum():.3f}")

    npz_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_out,
        embeddings_pca=emb_pca,
        token_ids=data["token_ids"],
        n_frames_used=data["n_frames_used"],
        layer=data["layer"],
        model=data["model"],
        pooling=data["pooling"],
        frame_rate_hz=data["frame_rate_hz"],
        pca_n_components=np.int32(n_components),
        pca_explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        pca_source_file=str(npz_in.name),
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
    suffix = cfg["output_suffix"]
    n_components = cfg["pca_n_components"]
    random_state = cfg["random_state"]

    # Discover the (prefix, layer) pairs from the upstream extract_neural_* configs.
    # No duplication: if [04]/[05] change layers, this stage adapts automatically.
    jobs: list[tuple[str, int]] = []
    for upstream_key in ("extract_neural_whisper", "extract_neural_xlsr"):
        up = config[upstream_key]
        for L in up["layers"]:
            jobs.append((up["output_prefix"], L))

    for prefix, L in jobs:
        npz_in = in_dir / f"{prefix}_L{L:02d}.npz"
        npz_out = out_dir / f"{prefix}_L{L:02d}_{suffix}.npz"
        if not npz_in.exists():
            raise FileNotFoundError(
                f"Expected upstream file not found: {npz_in}. "
                f"Did stage extract_neural_* run successfully?"
            )
        reduce_one(npz_in, npz_out, n_components, random_state)

    print(f"[normalise_neural] done: {len(jobs)} files reduced to d={n_components}.")


if __name__ == "__main__":
    main()
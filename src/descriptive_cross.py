"""
descriptive_cross.py

Robust cross-representation descriptive comparison.

This script builds acoustic and neural Representational Similarity Matrices
(RSMs) over oral-vowel centroids, then compares their upper triangles with a
Mantel-style Spearman correlation.

By default it computes two acoustic RSM variants:
  - main: F1/F2, matching the core vowel-space analysis;
  - extended: optional extra acoustic descriptors if configured and available.

Tables are computed for all configured neural layers. RSM figures are generated
only for selected layers to avoid producing low-value plots.

Writes
------
results/tables/
  tab_mantel_results.csv

results/figures/descriptive/cross/
  fig_rsm_acoustic_main.png
  fig_rsm_acoustic_extended.png  [if extended features are configured]
  fig_rsm_{model}_L{NN}.png      [only selected layers]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

ORAL_VOWEL_ORDER = ["i", "e", "ɛ", "a", "ɑ", "ɔ", "o", "u", "y", "ø", "œ", "ə"]
REQUIRED_META_COLUMNS = {"token_id", "phoneme_base", "speaker_id", "L1_status", "gender"}


# ---------------------------------------------------------------------
# I/O and validation
# ---------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def require_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{context}: missing required columns: {missing}")


def load_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input acoustic/metadata table not found: {path}")

    df = pd.read_csv(path)
    require_columns(df, REQUIRED_META_COLUMNS, context=str(path))

    if df["token_id"].duplicated().any():
        dup = int(df["token_id"].duplicated().sum())
        raise ValueError(f"token_id must be unique in {path}; found {dup} duplicates")

    df["group"] = df["L1_status"].astype(str) + "/" + df["gender"].astype(str)
    return df


def model_layer_pairs(cfg: dict) -> list[tuple[str, str, int]]:
    """Return [(model_tag, output_prefix, layer_int)]."""
    pairs: list[tuple[str, str, int]] = []

    for tag, key in [
        ("whisper", "extract_neural_whisper"),
        ("xlsr", "extract_neural_xlsr"),
    ]:
        block = cfg[key]
        for layer in block["layers"]:
            pairs.append((tag, block["output_prefix"], int(layer)))

    return pairs


def load_raw_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing raw neural NPZ: {path}")

    d = np.load(path, allow_pickle=True)
    if "embeddings" not in d or "token_ids" not in d:
        raise KeyError(f"{path} must contain arrays 'embeddings' and 'token_ids'")

    emb = d["embeddings"].astype(np.float32)
    ids = d["token_ids"]

    if len(emb) != len(ids):
        raise ValueError(f"{path}: embeddings/token_ids length mismatch: {len(emb)} vs {len(ids)}")

    return emb, ids


def align_metadata(ids: np.ndarray, meta: pd.DataFrame, context: str) -> pd.DataFrame:
    meta_by_id = meta.set_index("token_id")
    missing = pd.Index(ids).difference(meta_by_id.index)
    if len(missing) > 0:
        raise KeyError(f"{context}: {len(missing)} token_ids are missing from metadata")
    return meta_by_id.loc[ids].reset_index()


# ---------------------------------------------------------------------
# Centroids and RSMs
# ---------------------------------------------------------------------

def observed_vowels_with_acoustic_data(
    df: pd.DataFrame,
    vowels: list[str],
    feature_cols: list[str],
    min_tokens: int = 5,
) -> list[str]:
    """Keep vowels with enough complete acoustic observations."""
    require_columns(df, set(feature_cols), context="acoustic features")

    kept: list[str] = []
    for ph in vowels:
        n = df.loc[df["phoneme_base"] == ph, feature_cols].dropna().shape[0]
        if n >= min_tokens:
            kept.append(ph)
    return kept


def acoustic_centroids(df: pd.DataFrame, phonemes: list[str], feature_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    """Return acoustic centroid matrix and kept phoneme order."""
    rows: list[np.ndarray] = []
    kept: list[str] = []

    for ph in phonemes:
        sub = df.loc[df["phoneme_base"] == ph, feature_cols].dropna()
        if sub.empty:
            continue
        rows.append(sub.mean(axis=0).to_numpy(dtype=np.float32))
        kept.append(ph)

    return np.array(rows, dtype=np.float32), kept


def neural_centroids(emb: np.ndarray, ids: np.ndarray, meta: pd.DataFrame, phonemes: list[str]) -> tuple[np.ndarray, list[str]]:
    """Return neural centroid matrix in the requested phoneme order."""
    aligned = align_metadata(ids, meta, context="neural_centroids")

    rows: list[np.ndarray] = []
    kept: list[str] = []

    for ph in phonemes:
        idx = np.where(aligned["phoneme_base"].to_numpy() == ph)[0]
        if len(idx) == 0:
            continue
        rows.append(emb[idx].mean(axis=0))
        kept.append(ph)

    return np.array(rows, dtype=np.float32), kept


def rsm_from_centroids(C: np.ndarray, metric: str) -> np.ndarray:
    """
    Compute a representation similarity matrix.

    metric='neg_euclidean': S_ij = -||c_i - c_j||
    metric='cosine':        S_ij = cosine(c_i, c_j)
    """
    if C.ndim != 2 or C.shape[0] < 2:
        raise ValueError(f"Centroid matrix must be 2D with at least two rows, got {C.shape}")

    if metric == "neg_euclidean":
        diffs = C[:, None, :] - C[None, :, :]
        distances = np.sqrt((diffs ** 2).sum(axis=-1))
        return -distances

    if metric == "cosine":
        norms = np.linalg.norm(C, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Cn = C / norms
        return Cn @ Cn.T

    raise ValueError(f"Unknown metric: {metric}")


# ---------------------------------------------------------------------
# Mantel-style comparison
# ---------------------------------------------------------------------

def upper_tri(M: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(M, k=1)
    return M[iu]


def mantel_test(S1: np.ndarray, S2: np.ndarray, n_perm: int, rng: np.random.Generator) -> tuple[float, float, int]:
    """Spearman rank correlation between RSM upper triangles with permutation p-value."""
    if S1.shape != S2.shape:
        raise ValueError(f"RSM shape mismatch: {S1.shape} vs {S2.shape}")

    n = S1.shape[0]
    v1 = rankdata(upper_tri(S1))
    v2 = rankdata(upper_tri(S2))
    r_obs = float(np.corrcoef(v1, v2)[0, 1])

    if not np.isfinite(r_obs):
        return float("nan"), float("nan"), int(n_perm)

    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        S2_perm = S2[perm][:, perm]
        v2_perm = rankdata(upper_tri(S2_perm))
        r_perm = float(np.corrcoef(v1, v2_perm)[0, 1])
        if np.isfinite(r_perm) and r_perm >= r_obs:
            count += 1

    p_value = float((count + 1) / (n_perm + 1))
    return r_obs, p_value, int(n_perm)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_rsm(S: np.ndarray, labels: list[str], title: str, out_path: Path) -> None:
    """Plot a representation similarity matrix."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    im = ax.imshow(S, cmap="viridis", aspect="equal")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_title(title)

    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5, alpha=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.colorbar(im, ax=ax, label="similarity", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def rsm_layers_to_plot(block: dict) -> dict[str, list[int]]:
    layers = block.get("rsm_plot_layers", {})
    return {
        "whisper": [int(x) for x in layers.get("whisper", [])],
        "xlsr": [int(x) for x in layers.get("xlsr", [])],
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    block = cfg["descriptive_cross"]

    in_acoustic = Path(block["input_acoustic"])
    interim_dir = Path(block["interim_dir"])
    fig_dir = Path(block["figures_dir"]) / "descriptive" / "cross"
    tab_dir = Path(block["tables_dir"])
    ensure_dirs(fig_dir, tab_dir)

    n_perm = int(block.get("mantel_n_permutations", 5000))
    seed = int(block.get("random_state", 42))

    acoustic_main_cols = block.get("acoustic_main_feature_cols", ["F1_norm", "F2_norm"])
    acoustic_extended_cols = block.get("acoustic_extended_feature_cols", [])

    df = load_metadata(in_acoustic)

    all_vowels = cfg["normalise_acoustic"]["vowel_inventory"]
    oral_vowels = [v for v in ORAL_VOWEL_ORDER if v in all_vowels]

    main_vowels = observed_vowels_with_acoustic_data(
        df=df,
        vowels=oral_vowels,
        feature_cols=acoustic_main_cols,
        min_tokens=5,
    )

    if len(main_vowels) < 3:
        raise RuntimeError(f"Too few vowels with complete main acoustic data: {main_vowels}")

    print(f"Cross-representation phoneme set: {main_vowels}")

    acoustic_rsms: dict[str, tuple[np.ndarray, list[str], list[str]]] = {}

    C_main, kept_main = acoustic_centroids(df, main_vowels, acoustic_main_cols)
    S_main = rsm_from_centroids(C_main, metric="neg_euclidean")
    acoustic_rsms["main"] = (S_main, kept_main, acoustic_main_cols)

    plot_rsm(
        S=S_main,
        labels=kept_main,
        title=f"Acoustic RSM main: -Euclidean distance ({', '.join(acoustic_main_cols)})",
        out_path=fig_dir / "fig_rsm_acoustic_main.png",
    )
    print(f"Wrote {fig_dir / 'fig_rsm_acoustic_main.png'}")

    # Backward-compatible alias for scripts/reports that expect fig_rsm_acoustic.png.
    plot_rsm(
        S=S_main,
        labels=kept_main,
        title=f"Acoustic RSM main: -Euclidean distance ({', '.join(acoustic_main_cols)})",
        out_path=fig_dir / "fig_rsm_acoustic.png",
    )
    print(f"Wrote {fig_dir / 'fig_rsm_acoustic.png'}")

    if acoustic_extended_cols:
        available_ext = [c for c in acoustic_extended_cols if c in df.columns]
        if len(available_ext) >= 2:
            ext_vowels = observed_vowels_with_acoustic_data(
                df=df,
                vowels=main_vowels,
                feature_cols=available_ext,
                min_tokens=5,
            )
            # Keep the same phoneme set as main when possible.
            ext_vowels = [v for v in main_vowels if v in ext_vowels]
            if len(ext_vowels) >= 3:
                C_ext, kept_ext = acoustic_centroids(df, ext_vowels, available_ext)
                S_ext = rsm_from_centroids(C_ext, metric="neg_euclidean")
                acoustic_rsms["extended"] = (S_ext, kept_ext, available_ext)
                plot_rsm(
                    S=S_ext,
                    labels=kept_ext,
                    title=f"Acoustic RSM extended: -Euclidean distance ({', '.join(available_ext)})",
                    out_path=fig_dir / "fig_rsm_acoustic_extended.png",
                )
                print(f"Wrote {fig_dir / 'fig_rsm_acoustic_extended.png'}")
            else:
                print("[warning] Too few vowels for extended acoustic RSM; skipping")
        else:
            print("[warning] Extended acoustic features not available; skipping extended RSM")

    # Neural RSMs.
    plot_layers = rsm_layers_to_plot(block)
    neural_rsms: dict[tuple[str, int], tuple[np.ndarray, list[str]]] = {}

    for tag, prefix, layer in model_layer_pairs(cfg):
        layer_str = f"L{layer:02d}"
        raw_path = interim_dir / f"{prefix}_{layer_str}.npz"
        emb, ids = load_raw_npz(raw_path)

        C_n, kept_n = neural_centroids(emb=emb, ids=ids, meta=df, phonemes=kept_main)
        if kept_n != kept_main:
            print(f"[warning] phoneme mismatch for {tag} {layer_str}: {kept_n} vs {kept_main}; skipping")
            continue

        S_n = rsm_from_centroids(C_n, metric="cosine")
        neural_rsms[(tag, int(layer))] = (S_n, kept_n)

        if int(layer) in plot_layers.get(tag, []):
            out = fig_dir / f"fig_rsm_{tag}_{layer_str}.png"
            plot_rsm(
                S=S_n,
                labels=kept_n,
                title=f"Neural RSM: cosine similarity — {tag} {layer_str}",
                out_path=out,
            )
            print(f"Wrote {out}")

    # Mantel-style correlations.
    rows: list[dict] = []

    for analysis_name, (S_ac, kept_ac, feature_cols) in acoustic_rsms.items():
        rng = np.random.default_rng(seed)

        for (tag, layer), (S_n, kept_n) in neural_rsms.items():
            if kept_n != kept_ac:
                # This will usually only affect the extended analysis if a feature
                # has more missing values and changes the vowel set.
                common = [p for p in kept_ac if p in kept_n]
                if len(common) < 3:
                    continue
                idx_ac = [kept_ac.index(p) for p in common]
                idx_n = [kept_n.index(p) for p in common]
                S1 = S_ac[np.ix_(idx_ac, idx_ac)]
                S2 = S_n[np.ix_(idx_n, idx_n)]
                used_phonemes = common
            else:
                S1 = S_ac
                S2 = S_n
                used_phonemes = kept_ac

            r, p, n = mantel_test(S1=S1, S2=S2, n_perm=n_perm, rng=rng)

            rows.append({
                "analysis": analysis_name,
                "acoustic_features": " ".join(feature_cols),
                "rep_a": f"acoustic_{analysis_name}",
                "rep_b": f"{tag}_L{layer:02d}",
                "n_phonemes": int(len(used_phonemes)),
                "phonemes": " ".join(used_phonemes),
                "mantel_r_spearman": r,
                "p_value": p,
                "n_permutations": n,
            })

            print(
                f"Mantel acoustic_{analysis_name} vs {tag}_L{layer:02d}: "
                f"r={r:.4f}, p={p:.4f}"
            )

    # Whisper vs XLS-R at matching layers, independent of acoustic feature choice.
    by_layer: dict[int, dict[str, tuple[np.ndarray, list[str]]]] = {}
    for (tag, layer), value in neural_rsms.items():
        by_layer.setdefault(int(layer), {})[tag] = value

    rng = np.random.default_rng(seed)
    for layer, d in sorted(by_layer.items()):
        if "whisper" not in d or "xlsr" not in d:
            continue

        S_w, kept_w = d["whisper"]
        S_x, kept_x = d["xlsr"]
        if kept_w != kept_x:
            common = [p for p in kept_w if p in kept_x]
            if len(common) < 3:
                continue
            idx_w = [kept_w.index(p) for p in common]
            idx_x = [kept_x.index(p) for p in common]
            S1 = S_w[np.ix_(idx_w, idx_w)]
            S2 = S_x[np.ix_(idx_x, idx_x)]
            used_phonemes = common
        else:
            S1 = S_w
            S2 = S_x
            used_phonemes = kept_w

        r, p, n = mantel_test(S1=S1, S2=S2, n_perm=n_perm, rng=rng)
        rows.append({
            "analysis": "neural_vs_neural",
            "acoustic_features": "NA",
            "rep_a": f"whisper_L{layer:02d}",
            "rep_b": f"xlsr_L{layer:02d}",
            "n_phonemes": int(len(used_phonemes)),
            "phonemes": " ".join(used_phonemes),
            "mantel_r_spearman": r,
            "p_value": p,
            "n_permutations": n,
        })
        print(f"Mantel whisper_L{layer:02d} vs xlsr_L{layer:02d}: r={r:.4f}, p={p:.4f}")

    out_tab = tab_dir / "tab_mantel_results.csv"
    pd.DataFrame(rows).to_csv(out_tab, index=False)
    print(f"Wrote {out_tab}")


if __name__ == "__main__":
    main()

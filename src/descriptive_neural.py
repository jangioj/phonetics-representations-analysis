"""
descriptive_neural.py

Robust descriptive statistics and report-ready figures for neural speech
representations.

Tables are computed for every configured model/layer. Figures are generated only
for the report-selected layers by default, to avoid producing dozens of low-value
plots. This behaviour is controlled from config.yaml.

Writes
------
results/tables/
  tab_neural_between_class_ratio.csv
  tab_neural_cosine_within_between.csv
  tab_neural_inter_speaker_variability.csv

results/figures/descriptive/neural/
  fig_{pca,umap}_{model}_L{NN}_by_phoneme.png
  fig_{pca,umap}_{model}_L{NN}_by_L1_status.png
  fig_{pca,umap}_{model}_L{NN}_by_gender.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

ORAL_VOWEL_ORDER = ["i", "e", "ɛ", "a", "ɑ", "ɔ", "o", "u", "y", "ø", "œ", "ə"]

VOWEL_COLORS = {
    "i": "#1f77b4",
    "e": "#2ca02c",
    "ɛ": "#98df8a",
    "a": "#d62728",
    "ɑ": "#ff9896",
    "ɔ": "#ff7f0e",
    "o": "#ffbb78",
    "u": "#9467bd",
    "y": "#8c564b",
    "ø": "#e377c2",
    "œ": "#f7b6d2",
    "ə": "#7f7f7f",
}

L1_COLORS = {
    "L1": "#1f77b4",
    "L2": "#d62728",
}

GENDER_COLORS = {
    "F": "#9467bd",
    "M": "#ff7f0e",
}

REQUIRED_META_COLUMNS = {
    "token_id",
    "phoneme_base",
    "speaker_id",
    "L1_status",
    "gender",
}


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
        raise FileNotFoundError(f"Metadata table not found: {path}")

    meta = pd.read_csv(path)
    require_columns(meta, REQUIRED_META_COLUMNS, context=str(path))
    meta["group"] = meta["L1_status"].astype(str) + "/" + meta["gender"].astype(str)

    if meta["token_id"].duplicated().any():
        dup = int(meta["token_id"].duplicated().sum())
        raise ValueError(f"Metadata token_id must be unique; found {dup} duplicates")

    return meta


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


def load_npz_array(path: Path, array_key: str) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing NPZ file: {path}")

    d = np.load(path, allow_pickle=True)
    if array_key not in d or "token_ids" not in d:
        raise KeyError(f"{path} must contain arrays {array_key!r} and 'token_ids'")

    X = d[array_key].astype(np.float32)
    ids = d["token_ids"]

    if len(X) != len(ids):
        raise ValueError(f"{path}: embeddings/token_ids length mismatch: {len(X)} vs {len(ids)}")

    return X, ids


def load_raw_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    return load_npz_array(path, "embeddings")


def load_pca_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    return load_npz_array(path, "embeddings_pca")


def load_umap_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    return load_npz_array(path, "embeddings_umap")


def align_metadata(ids: np.ndarray, meta: pd.DataFrame, context: str) -> pd.DataFrame:
    meta_by_id = meta.set_index("token_id")
    missing = pd.Index(ids).difference(meta_by_id.index)
    if len(missing) > 0:
        raise KeyError(f"{context}: {len(missing)} token_ids are missing from metadata")
    return meta_by_id.loc[ids].reset_index()


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def between_class_variance_ratio(emb_2d: np.ndarray, labels: np.ndarray) -> float:
    """Between-class scatter divided by total scatter in a 2D projection."""
    valid = ~pd.isna(labels)
    X = emb_2d[valid]
    y = np.asarray(labels[valid])

    if len(X) == 0:
        return float("nan")

    global_mean = X.mean(axis=0)
    total_scatter = float(np.sum((X - global_mean) ** 2))
    if total_scatter == 0:
        return float("nan")

    between_scatter = 0.0
    for label in np.unique(y):
        Xc = X[y == label]
        if len(Xc) == 0:
            continue
        class_mean = Xc.mean(axis=0)
        between_scatter += len(Xc) * float(np.sum((class_mean - global_mean) ** 2))

    return float(between_scatter / total_scatter)


def cosine_within_between(
    emb: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    n_within_per_class: int = 500,
    n_between_pairs: int = 5000,
) -> tuple[float, float, float]:
    """Estimate average cosine similarity within and between phonemes."""
    valid = ~pd.isna(labels)
    X = emb[valid]
    y = np.asarray(labels[valid])

    if len(X) < 2:
        return float("nan"), float("nan"), float("nan")

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms

    within_sims: list[np.ndarray] = []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        if len(idx) < 2:
            continue

        max_pairs = len(idx) * (len(idx) - 1) // 2
        k = min(n_within_per_class, max_pairs)
        if k <= 0:
            continue

        # Sample pairs efficiently with replacement. Reject self-pairs.
        pairs_a: list[int] = []
        pairs_b: list[int] = []
        attempts = 0
        while len(pairs_a) < k and attempts < k * 50:
            a, b = rng.integers(0, len(idx), size=2)
            attempts += 1
            if a == b:
                continue
            pairs_a.append(idx[int(a)])
            pairs_b.append(idx[int(b)])

        if pairs_a:
            sims = np.einsum("ij,ij->i", Xn[pairs_a], Xn[pairs_b])
            within_sims.append(sims)

    within_mean = float(np.mean(np.concatenate(within_sims))) if within_sims else float("nan")

    between_a: list[int] = []
    between_b: list[int] = []
    attempts = 0
    while len(between_a) < n_between_pairs and attempts < n_between_pairs * 50:
        a, b = rng.integers(0, len(X), size=2)
        attempts += 1
        if y[int(a)] == y[int(b)]:
            continue
        between_a.append(int(a))
        between_b.append(int(b))

    if between_a:
        between_sims = np.einsum("ij,ij->i", Xn[between_a], Xn[between_b])
        between_mean = float(np.mean(between_sims))
    else:
        between_mean = float("nan")

    ratio = (
        float(within_mean / between_mean)
        if np.isfinite(within_mean) and np.isfinite(between_mean) and between_mean != 0
        else float("nan")
    )

    return within_mean, between_mean, ratio


def mean_intra_speaker_cosine_distance(emb: np.ndarray, ids: np.ndarray, meta: pd.DataFrame) -> float:
    """Mean cosine distance between tokens from the same speaker and phoneme."""
    aligned = align_metadata(ids, meta, context="mean_intra_speaker_cosine_distance")

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = emb / norms

    cell_means: list[float] = []
    cell_weights: list[int] = []

    for (_, _), grp in aligned.groupby(["speaker_id", "phoneme_base"]):
        idx = grp.index.to_numpy()
        if len(idx) < 2:
            continue

        block = Xn[idx]
        S = block @ block.T
        iu = np.triu_indices_from(S, k=1)
        sims = S[iu]
        if len(sims) == 0:
            continue

        cell_means.append(float(np.mean(sims)))
        cell_weights.append(int(len(sims)))

    if not cell_means:
        return float("nan")

    mean_sim = float(np.average(cell_means, weights=cell_weights))
    return float(1.0 - mean_sim)


def inter_speaker_variability_neural(
    emb: np.ndarray,
    ids: np.ndarray,
    meta: pd.DataFrame,
    phonemes: list[str],
) -> pd.DataFrame:
    """Average cosine distance among per-speaker centroids, per phoneme."""
    aligned = align_metadata(ids, meta, context="inter_speaker_variability_neural")
    rows: list[dict] = []

    for ph in phonemes:
        sub = aligned[aligned["phoneme_base"] == ph]
        if sub.empty:
            continue

        centroids: list[np.ndarray] = []
        for _, grp in sub.groupby("speaker_id"):
            idx = grp.index.to_numpy()
            if len(idx) == 0:
                continue
            centroids.append(emb[idx].mean(axis=0))

        if len(centroids) < 2:
            continue

        C = np.stack(centroids).astype(np.float32)
        norms = np.linalg.norm(C, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Cn = C / norms
        S = Cn @ Cn.T
        iu = np.triu_indices_from(S, k=1)

        rows.append({
            "phoneme": ph,
            "n_speakers": int(len(centroids)),
            "mean_inter_speaker_cosine_distance": float(1.0 - np.mean(S[iu])),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def selected_layers_for_figures(cfg: dict, block: dict, tag: str, configured_layers: list[int]) -> list[int]:
    """Return layers to plot. Tables are still computed for all layers."""
    if bool(block.get("plot_all_layers", False)):
        return list(configured_layers)

    report_layers = block.get("report_layers", {})
    layers = report_layers.get(tag, [])
    layers = [int(x) for x in layers if int(x) in configured_layers]

    if layers:
        return layers

    # Safe fallback: plot the middle configured layer.
    return [int(configured_layers[len(configured_layers) // 2])]


def palette_for(color_by: str, labels: np.ndarray) -> dict[str, str]:
    if color_by == "phoneme_base":
        return VOWEL_COLORS
    if color_by == "L1_status":
        return L1_COLORS
    if color_by == "gender":
        return GENDER_COLORS

    unique = [str(x) for x in pd.Series(labels).dropna().unique()]
    cmap = plt.get_cmap("tab20")
    return {lab: cmap(i % 20) for i, lab in enumerate(unique)}


def pretty_color_label(color_by: str) -> str:
    return {
        "phoneme_base": "phoneme",
        "L1_status": "L1_status",
        "gender": "gender",
    }.get(color_by, color_by)


def filename_color_label(color_by: str) -> str:
    return {
        "phoneme_base": "phoneme",
        "L1_status": "L1_status",
        "gender": "gender",
    }.get(color_by, color_by)


def plot_projection(
    emb_2d: np.ndarray,
    meta_2d: pd.DataFrame,
    title: str,
    out_path: Path,
    x_label: str,
    y_label: str,
    color_by: str,
    oral_vowels: list[str],
) -> None:
    """Plot a 2D projection for oral vowels, coloured by a categorical variable."""
    require_columns(meta_2d, {"phoneme_base", color_by}, context="projection metadata")

    mask = meta_2d["phoneme_base"].isin(oral_vowels).to_numpy()
    X = emb_2d[mask]
    M = meta_2d.loc[mask].reset_index(drop=True)

    if len(X) == 0:
        raise RuntimeError(f"No oral-vowel points available for {title}")

    labels = M[color_by].astype(str).to_numpy()
    pal = palette_for(color_by, labels)

    fig, ax = plt.subplots(figsize=(8.8, 7.0))

    if color_by == "phoneme_base":
        ordered_labels = [v for v in ORAL_VOWEL_ORDER if v in set(labels)]
    else:
        ordered_labels = sorted(pd.Series(labels).dropna().unique())

    for lab in ordered_labels:
        lab = str(lab)
        idx = labels == lab
        if not np.any(idx):
            continue
        ax.scatter(
            X[idx, 0],
            X[idx, 1],
            s=18,
            alpha=0.70,
            color=pal.get(lab, "black"),
            label=lab,
            edgecolor="none",
        )

    # Phoneme-centroid labels stay useful even when colouring by L1/gender:
    # they allow direct comparison with the vowel-space question.
    for vowel in ORAL_VOWEL_ORDER:
        idx = M["phoneme_base"].to_numpy() == vowel
        if not np.any(idx):
            continue
        cx = float(np.mean(X[idx, 0]))
        cy = float(np.mean(X[idx, 1]))
        ax.text(
            cx,
            cy,
            vowel,
            fontsize=13,
            weight="bold",
            ha="center",
            va="center",
            color="black",
            bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 1.5},
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(
        title=pretty_color_label(color_by),
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    block = cfg["descriptive_neural"]

    in_meta_csv = Path(block["input_metadata"])
    interim_dir = Path(block["interim_dir"])
    fig_dir = Path(block["figures_dir"]) / "descriptive" / "neural"
    tab_dir = Path(block["tables_dir"])
    ensure_dirs(fig_dir, tab_dir)

    n_within = int(block.get("n_within_per_class", 500))
    n_between = int(block.get("n_between_pairs", 5000))
    seed = int(block.get("random_state", 42))
    color_by_fields = block.get("color_by", ["phoneme_base", "L1_status", "gender"])

    all_vowels = cfg["normalise_acoustic"]["vowel_inventory"]
    oral_vowels = [v for v in ORAL_VOWEL_ORDER if v in all_vowels]

    meta = load_metadata(in_meta_csv)

    rows_ratio: list[dict] = []
    rows_cos: list[dict] = []
    rows_inter_speaker: list[pd.DataFrame] = []

    configured_layers_by_tag = {
        "whisper": [int(x) for x in cfg["extract_neural_whisper"]["layers"]],
        "xlsr": [int(x) for x in cfg["extract_neural_xlsr"]["layers"]],
    }

    plot_layers_by_tag = {
        tag: selected_layers_for_figures(cfg, block, tag, layers)
        for tag, layers in configured_layers_by_tag.items()
    }

    print(f"Report figure layers: {plot_layers_by_tag}")

    for tag, prefix, layer in model_layer_pairs(cfg):
        layer_str = f"L{layer:02d}"
        raw_path = interim_dir / f"{prefix}_{layer_str}.npz"
        pca_path = interim_dir / f"{prefix}_{layer_str}_pca.npz"
        umap_path = interim_dir / f"{prefix}_{layer_str}_umap.npz"

        print(f"=== {tag} {layer_str} ===")
        emb_raw, ids_raw = load_raw_npz(raw_path)
        meta_raw = align_metadata(ids_raw, meta, context=str(raw_path))

        emb_pca_full, ids_pca = load_pca_npz(pca_path)
        emb_pca_2d = emb_pca_full[:, :2]
        meta_pca = align_metadata(ids_pca, meta, context=str(pca_path))

        emb_umap, ids_umap = load_umap_npz(umap_path)
        meta_umap = align_metadata(ids_umap, meta, context=str(umap_path))

        # Tables: all configured phonemes, all configured layers.
        bcr_pca = between_class_variance_ratio(emb_pca_2d, meta_pca["phoneme_base"].to_numpy())
        bcr_umap = between_class_variance_ratio(emb_umap, meta_umap["phoneme_base"].to_numpy())

        rows_ratio.append({
            "model": tag,
            "layer": int(layer),
            "n_tokens": int((~meta_raw["phoneme_base"].isna()).sum()),
            "between_class_variance_ratio_pca2d": bcr_pca,
            "between_class_variance_ratio_umap2d": bcr_umap,
        })

        rng = np.random.default_rng(seed + int(layer) + (0 if tag == "whisper" else 1000))
        within_mean, between_mean, ratio = cosine_within_between(
            emb=emb_raw,
            labels=meta_raw["phoneme_base"].to_numpy(),
            rng=rng,
            n_within_per_class=n_within,
            n_between_pairs=n_between,
        )

        intra_speaker_distance = mean_intra_speaker_cosine_distance(
            emb=emb_raw,
            ids=ids_raw,
            meta=meta,
        )

        rows_cos.append({
            "model": tag,
            "layer": int(layer),
            "cosine_within_mean": within_mean,
            "cosine_between_mean": between_mean,
            "ratio_within_over_between": ratio,
            "mean_intra_speaker_cosine_distance": intra_speaker_distance,
            "n_within_per_class": int(n_within),
            "n_between_pairs": int(n_between),
        })

        df_inter = inter_speaker_variability_neural(
            emb=emb_raw,
            ids=ids_raw,
            meta=meta,
            phonemes=oral_vowels,
        )
        if not df_inter.empty:
            df_inter.insert(0, "layer", int(layer))
            df_inter.insert(0, "model", tag)
            rows_inter_speaker.append(df_inter)

        # Figures: selected report layers only by default.
        if int(layer) in plot_layers_by_tag.get(tag, []):
            for method, emb_2d, meta_2d, x_label, y_label in [
                ("pca", emb_pca_2d, meta_pca, "PC1", "PC2"),
                ("umap", emb_umap, meta_umap, "UMAP-1", "UMAP-2"),
            ]:
                for color_by in color_by_fields:
                    out = fig_dir / f"fig_{method}_{tag}_{layer_str}_by_{filename_color_label(color_by)}.png"
                    plot_projection(
                        emb_2d=emb_2d,
                        meta_2d=meta_2d,
                        title=f"{method.upper()} vowel projection — {tag} {layer_str} by {pretty_color_label(color_by)}",
                        out_path=out,
                        x_label=x_label,
                        y_label=y_label,
                        color_by=color_by,
                        oral_vowels=oral_vowels,
                    )
                    print(f"Wrote {out}")

        print(
            f"  between-class ratio: PCA={bcr_pca:.4f}, UMAP={bcr_umap:.4f}; "
            f"cosine within={within_mean:.4f}, between={between_mean:.4f}, ratio={ratio:.4f}"
        )

    out_ratio = tab_dir / "tab_neural_between_class_ratio.csv"
    out_cos = tab_dir / "tab_neural_cosine_within_between.csv"
    out_inter = tab_dir / "tab_neural_inter_speaker_variability.csv"

    pd.DataFrame(rows_ratio).to_csv(out_ratio, index=False)
    pd.DataFrame(rows_cos).to_csv(out_cos, index=False)

    if rows_inter_speaker:
        pd.concat(rows_inter_speaker, ignore_index=True).to_csv(out_inter, index=False)
    else:
        pd.DataFrame(
            columns=["model", "layer", "phoneme", "n_speakers", "mean_inter_speaker_cosine_distance"]
        ).to_csv(out_inter, index=False)

    print(f"Wrote {out_ratio}")
    print(f"Wrote {out_cos}")
    print(f"Wrote {out_inter}")


if __name__ == "__main__":
    main()

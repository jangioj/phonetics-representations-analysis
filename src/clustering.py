"""Hierarchical clustering PDF Section 9"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score


REQUIRED_NORM_COLUMNS = {
    "token_id", "phoneme_base", "speaker_id", "L1_status", "gender", "is_vowel",
    "F1_norm", "F2_norm",
}
REQUIRED_RAW_COLUMNS = {
    "token_id", "speaker_id", "phoneme_base", "duration_ms", "scg_hz", "scg_applicable",
}


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


def load_csv(path: Path, required: set[str], label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    df = pd.read_csv(path)
    require_columns(df, required, context=str(path))
    if "token_id" in df.columns and df["token_id"].duplicated().any():
        raise ValueError(f"{path}: token_id must be unique")
    return df


def load_pca_npz(path: Path, n_components: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing PCA neural NPZ: {path}")
    d = np.load(path, allow_pickle=True)
    if "embeddings_pca" not in d or "token_ids" not in d:
        raise KeyError(f"{path} must contain 'embeddings_pca' and 'token_ids'")
    X = d["embeddings_pca"].astype(np.float32)
    ids = d["token_ids"]
    if len(X) != len(ids):
        raise ValueError(f"{path}: embeddings/token_ids mismatch")
    if n_components is not None:
        X = X[:, :min(int(n_components), X.shape[1])]
    return X, ids


def model_layer_pairs(cfg: dict, block: dict) -> list[tuple[str, str, int]]:
    pairs = []
    for tag, key in [("whisper", "extract_neural_whisper"), ("xlsr", "extract_neural_xlsr")]:
        upstream = cfg[key]
        configured = [int(x) for x in upstream["layers"]]
        requested = [int(x) for x in block.get("neural_layers", {}).get(tag, configured)]
        for layer in requested:
            if layer not in configured:
                raise ValueError(f"Requested {tag} layer {layer}; configured layers are {configured}")
            pairs.append((tag, upstream["output_prefix"], int(layer)))
    return pairs


def l2_normalise(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms > 0, norms, 1.0)


def parse_bool(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).to_numpy() != 0
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"]).to_numpy()


def align_to_metadata(X: np.ndarray, ids: np.ndarray, meta: pd.DataFrame, context: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    meta_by_id = meta.set_index("token_id")
    mask = pd.Index(ids).isin(meta_by_id.index)
    if not np.any(mask):
        raise RuntimeError(f"{context}: no token_id overlaps with metadata")
    X2 = X[mask]
    ids2 = ids[mask]
    meta2 = meta_by_id.loc[ids2].reset_index()
    return X2, ids2, meta2


def build_label_map(partition: dict[str, list[str]]) -> dict[str, str]:
    return {p: cls for cls, items in partition.items() for p in items}


def restrict_to_labelled(items: list[str], label_map: dict[str, str]) -> tuple[list[str], list[str]]:
    kept = [x for x in items if x in label_map]
    return kept, [label_map[x] for x in kept]

def filter_phonemes_by_count(
    df: pd.DataFrame,
    phonemes: list[str],
    min_tokens: int,
    feature_cols: list[str] | None = None,
) -> list[str]:
    kept = []
    for p in phonemes:
        sub = df[df["phoneme_base"] == p]
        if feature_cols is not None:
            sub = sub.dropna(subset=feature_cols)
        if len(sub) >= min_tokens:
            kept.append(p)
    return kept

def acoustic_vowel_centroids(df: pd.DataFrame, phonemes: list[str], cols: list[str]) -> tuple[np.ndarray, list[str], np.ndarray]:
    rows, kept, counts = [], [], []
    for p in phonemes:
        sub = df.loc[df["phoneme_base"] == p, cols].dropna()
        if len(sub) == 0:
            continue
        rows.append(sub.to_numpy(dtype=np.float64).mean(axis=0))
        kept.append(p)
        counts.append(len(sub))
    if not rows:
        raise RuntimeError("No acoustic vowel centroids could be computed")
    return np.vstack(rows), kept, np.asarray(counts, dtype=int)


def neural_phoneme_centroids(X: np.ndarray, meta: pd.DataFrame, phonemes: list[str]) -> tuple[np.ndarray, list[str], np.ndarray]:
    rows, kept, counts = [], [], []
    labels = meta["phoneme_base"].to_numpy()
    for p in phonemes:
        mask = labels == p
        if not np.any(mask):
            continue
        rows.append(X[mask].astype(np.float64).mean(axis=0))
        kept.append(p)
        counts.append(int(mask.sum()))
    if not rows:
        raise RuntimeError("No neural phoneme centroids could be computed")
    return np.vstack(rows).astype(np.float32), kept, np.asarray(counts, dtype=int)


def build_cv_acoustic_features(norm_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    raw_use = raw_df[["token_id", "duration_ms", "scg_hz", "scg_applicable"]].copy()
    df = norm_df[["token_id", "phoneme_base", "F1_norm", "F2_norm"]].merge(raw_use, on="token_id", how="left")

    dur = df["duration_ms"].to_numpy(dtype=np.float64)
    dur_sd = np.nanstd(dur, ddof=1)
    df["duration_z"] = (dur - np.nanmean(dur)) / (dur_sd if dur_sd > 0 else 1.0)

    scg = df["scg_hz"].to_numpy(dtype=np.float64).copy()
    scg[~parse_bool(df["scg_applicable"])] = np.nan
    scg_sd = np.nanstd(scg, ddof=1)
    df["scg_z"] = (scg - np.nanmean(scg)) / (scg_sd if scg_sd > 0 else 1.0)
    return df


def acoustic_cv_centroids(
    cv_df: pd.DataFrame,
    phonemes: list[str],
    feature_cols: list[str],
) -> tuple[np.ndarray, list[str], np.ndarray]:
    rows: list[np.ndarray] = []
    kept: list[str] = []
    n_tok: list[int] = []

    for p in phonemes:
        sub = cv_df.loc[cv_df["phoneme_base"] == p, feature_cols]
        if len(sub) == 0:
            continue

        arr = sub.to_numpy(dtype=np.float64)
        row = np.array(
            [
                np.nanmean(arr[:, j]) if np.isfinite(arr[:, j]).any() else np.nan
                for j in range(arr.shape[1])
            ],
            dtype=np.float64,
        )

        if not np.isfinite(row).any():
            continue

        rows.append(row)
        kept.append(p)
        n_tok.append(len(sub))

    if not rows:
        raise RuntimeError("No C/V acoustic centroids could be computed.")

    if len(rows) != len(kept):
        raise RuntimeError(
            f"C/V acoustic centroid mismatch: rows={len(rows)}, labels={len(kept)}"
        )

    return np.vstack(rows), kept, np.asarray(n_tok, dtype=int)

def missing_aware_euclidean(X: np.ndarray) -> np.ndarray:
    n, D = X.shape
    out = np.zeros((n, n), dtype=np.float64)
    finite = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = X[i] - X[j]
            ok = np.isfinite(diff)
            if not np.any(ok):
                out[i, j] = out[j, i] = np.nan
                continue
            d = float(np.sqrt((diff[ok] ** 2).sum() * (D / ok.sum())))
            out[i, j] = out[j, i] = d
            finite.append(d)
    if not finite:
        raise RuntimeError("All pairwise C/V distances are undefined")
    nan_mask = ~np.isfinite(out)
    np.fill_diagonal(nan_mask, False)
    out[nan_mask] = float(np.nanmax(out))
    return out


def linkage_ward(X: np.ndarray) -> np.ndarray:
    return linkage(X, method="ward", metric="euclidean")


def linkage_cosine_ward(X: np.ndarray) -> np.ndarray:
    return linkage_ward(l2_normalise(X.astype(np.float64)))


def linkage_average_precomputed(D: np.ndarray) -> np.ndarray:
    D = np.asarray(D, dtype=np.float64)
    if D.ndim == 2:
        if D.shape[0] != D.shape[1]:
            raise ValueError(f"Precomputed distance matrix must be square, got {D.shape}")
        condensed = squareform(D, checks=False)
    elif D.ndim == 1:
        condensed = D
    else:
        raise ValueError(f"Expected square or condensed distance matrix, got ndim={D.ndim}")
    return linkage(condensed, method="average")


def cut_clusters(Z: np.ndarray, k: int) -> np.ndarray:
    return fcluster(Z, t=k, criterion="maxclust")


def silhouette_curve(X_or_D: np.ndarray, Z: np.ndarray, k_range: list[int], metric: str) -> dict[int, float]:
    scores = {}
    n = X_or_D.shape[0]
    for k in k_range:
        if k < 2 or k >= n:
            scores[k] = float("nan")
            continue
        labels = cut_clusters(Z, k)
        try:
            scores[k] = float(silhouette_score(X_or_D, labels, metric=metric))
        except ValueError:
            scores[k] = float("nan")
    return scores


def k_silhouette_max(scores: dict[int, float]) -> int:
    valid = {k: v for k, v in scores.items() if np.isfinite(v)}
    return int(max(valid, key=valid.get)) if valid else 2


def k_dendrogram_elbow(Z: np.ndarray, k_range: list[int]) -> int:
    heights = Z[:, 2]
    best_k, best_gap = int(k_range[0]), -np.inf
    for k in k_range:
        idx = len(heights) - k
        if 0 <= idx < len(heights) - 1:
            gap = heights[idx + 1] - heights[idx]
            if gap > best_gap:
                best_k, best_gap = int(k), float(gap)
    return best_k


def plot_dendrogram(Z: np.ndarray, labels: list[str], title: str, out_path: Path, figsize=(8, 5)) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    dendrogram(Z, labels=labels, leaf_rotation=0, leaf_font_size=10, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Linkage distance")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_vowel_clustering(norm_df: pd.DataFrame, cfg: dict, block: dict, tab_dir: Path, fig_dir: Path) -> dict:
    min_tokens = int(block.get("min_phoneme_tokens", 5))
    vowels = filter_phonemes_by_count(
        norm_df,
        block["oral_vowels"],
        min_tokens,
        block["acoustic_vowel_feature_cols"],
    )
    k_range = block["k_range"]
    pca_n = int(block.get("neural_pca_n_components", block.get("speaker_neural_pca_n_components", 50)))
    partitions = block["vowel_partitions"]

    results = {}
    X_ac, kept_ac, n_ac = acoustic_vowel_centroids(norm_df, vowels, block["acoustic_vowel_feature_cols"])
    results["acoustic"] = {"Z": linkage_ward(X_ac), "labels": kept_ac, "X": X_ac, "metric": "euclidean", "n_tokens": n_ac}

    meta_min = norm_df[["token_id", "phoneme_base"]].copy()
    for tag, prefix, layer in model_layer_pairs(cfg, block):
        L = f"L{layer:02d}"
        X, ids = load_pca_npz(Path(block["interim_dir"]) / f"{prefix}_{L}_pca.npz", pca_n)
        X, ids, meta = align_to_metadata(X, ids, meta_min, f"{tag}_{L}")
        Xc, kept, n_tok = neural_phoneme_centroids(X, meta, vowels)
        key = f"{tag}_{L}"
        Xn = l2_normalise(Xc.astype(np.float64))
        results[key] = {"Z": linkage_ward(Xn), "labels": kept, "X": Xn, "metric": "euclidean", "n_tokens": n_tok}

    ari_rows, assign_rows = [], []
    maps = {"front_back": build_label_map(partitions["front_back"]), "height": build_label_map(partitions["height"])}
    ks = {"front_back": 2, "height": 3}

    for rep, r in results.items():
        r["silhouette_scores"] = silhouette_curve(r["X"], r["Z"], k_range, r["metric"])
        r["k_silhouette"] = k_silhouette_max(r["silhouette_scores"])
        r["k_dendrogram"] = k_dendrogram_elbow(r["Z"], k_range)

        for part, label_map in maps.items():
            items, true_labels = restrict_to_labelled(r["labels"], label_map)
            row = {"representation": rep, "partition": part, "n_phonemes": len(items),
                   "k_linguistic": ks[part], "k_silhouette": r["k_silhouette"]}
            if len(items) < 2 or len(set(true_labels)) < 2:
                row["ari_at_k_linguistic"] = float("nan")
                row["ari_at_k_silhouette"] = float("nan")
            else:
                cl_ling = dict(zip(r["labels"], cut_clusters(r["Z"], ks[part])))
                pred_ling = [int(cl_ling[p]) for p in items]
                row["ari_at_k_linguistic"] = float(adjusted_rand_score(true_labels, pred_ling))
                cl_sil = dict(zip(r["labels"], cut_clusters(r["Z"], r["k_silhouette"])))
                pred_sil = [int(cl_sil[p]) for p in items]
                row["ari_at_k_silhouette"] = float(adjusted_rand_score(true_labels, pred_sil))
                for phon, true, cl in zip(items, true_labels, pred_ling):
                    assign_rows.append({"representation": rep, "partition": part, "phoneme": phon, "true_class": true, "cluster": cl})
            ari_rows.append(row)

    pd.DataFrame(ari_rows).to_csv(tab_dir / "tab_clust_vowel_ari.csv", index=False)
    pd.DataFrame(assign_rows).to_csv(tab_dir / "tab_clust_vowel_assignments.csv", index=False)

    plot_dendrogram(results["acoustic"]["Z"], results["acoustic"]["labels"],
                    "Vowel clustering — acoustic", fig_dir / "fig_dendro_vowel_acoustic.png")
    reps = block["representative_layers"]
    for tag in ("whisper", "xlsr"):
        key = f"{tag}_L{int(reps[tag]):02d}"
        plot_dendrogram(results[key]["Z"], results[key]["labels"],
                        f"Vowel clustering — {key} PCA-{pca_n}", fig_dir / f"fig_dendro_vowel_{key}.png")

    pairs = model_layer_pairs(cfg, block)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (tag, _prefix, layer) in zip(axes.ravel(), pairs):
        key = f"{tag}_L{layer:02d}"
        dendrogram(results[key]["Z"], labels=results[key]["labels"], leaf_font_size=8, ax=ax)
        ax.set_title(f"{tag} L{layer:02d}")
        ax.set_ylabel("Linkage distance")
    fig.suptitle(f"Vowel clustering — neural PCA-{pca_n}")
    plt.tight_layout()
    fig.savefig(fig_dir / "fig_dendro_vowel_all_layers.png", dpi=150)
    plt.close(fig)
    return results


def run_cv_clustering(norm_df: pd.DataFrame, raw_df: pd.DataFrame, cfg: dict, block: dict, tab_dir: Path, fig_dir: Path) -> dict:
    min_tokens = int(block.get("min_phoneme_tokens", 5))
    vowels = filter_phonemes_by_count(
        norm_df,
        block["oral_vowels"],
        min_tokens,
        block["acoustic_vowel_feature_cols"],
    )
    consonants = block["consonants"]
    phonemes = list(vowels) + list(consonants)
    k_range = block["k_range"]
    pca_n = int(block.get("neural_pca_n_components", block.get("speaker_neural_pca_n_components", 50)))

    X_ac, kept_ac, n_ac = acoustic_cv_centroids(
        build_cv_acoustic_features(norm_df, raw_df),
        phonemes,
        block["acoustic_cv_feature_cols"],
    )
    D_ac = missing_aware_euclidean(X_ac)
    Z_ac = linkage_average_precomputed(D_ac)

    if Z_ac.shape[0] + 1 != len(kept_ac):
        raise RuntimeError(
            f"C/V acoustic dendrogram mismatch: "
            f"X={X_ac.shape}, D={D_ac.shape}, "
            f"Z implies n={Z_ac.shape[0] + 1}, labels={len(kept_ac)}"
        )

    results = {
        "acoustic": {
            "Z": Z_ac,
            "labels": kept_ac,
            "X": D_ac,
            "metric": "precomputed",
            "n_tokens": n_ac,
        }
    }

    meta_min = norm_df[["token_id", "phoneme_base"]].copy()
    for tag, prefix, layer in model_layer_pairs(cfg, block):
        L = f"L{layer:02d}"
        X, ids = load_pca_npz(Path(block["interim_dir"]) / f"{prefix}_{L}_pca.npz", pca_n)
        X, ids, meta = align_to_metadata(X, ids, meta_min, f"{tag}_{L}")
        Xc, kept, n_tok = neural_phoneme_centroids(X, meta, phonemes)
        key = f"{tag}_{L}"
        Xn = l2_normalise(Xc.astype(np.float64))
        results[key] = {"Z": linkage_ward(Xn), "labels": kept, "X": Xn, "metric": "euclidean", "n_tokens": n_tok}

    vowel_set, consonant_set = set(vowels), set(consonants)
    manner = block.get("consonant_manner", {})
    ari_rows, assign_rows = [], []

    for rep, r in results.items():
        r["silhouette_scores"] = silhouette_curve(r["X"], r["Z"], k_range, r["metric"])
        r["k_silhouette"] = k_silhouette_max(r["silhouette_scores"])
        r["k_dendrogram"] = k_dendrogram_elbow(r["Z"], k_range)

        truth = ["vowel" if p in vowel_set else "consonant" if p in consonant_set else None for p in r["labels"]]
        items = [p for p, t in zip(r["labels"], truth) if t is not None]
        true = [t for t in truth if t is not None]
        row = {"representation": rep, "partition": "consonant_vs_vowel", "n_phonemes": len(items),
               "k_linguistic": 2, "k_silhouette": r["k_silhouette"]}
        if len(set(true)) > 1:
            cl_ling = dict(zip(r["labels"], cut_clusters(r["Z"], 2)))
            pred_ling = [int(cl_ling[p]) for p in items]
            row["ari_at_k_linguistic"] = float(adjusted_rand_score(true, pred_ling))
            cl_sil = dict(zip(r["labels"], cut_clusters(r["Z"], r["k_silhouette"])))
            pred_sil = [int(cl_sil[p]) for p in items]
            row["ari_at_k_silhouette"] = float(adjusted_rand_score(true, pred_sil))
            for p, t, cl in zip(items, true, pred_ling):
                assign_rows.append({"representation": rep, "phoneme": p, "true_class": t, "manner": manner.get(p, "vowel"), "cluster": cl})
        else:
            row["ari_at_k_linguistic"] = float("nan")
            row["ari_at_k_silhouette"] = float("nan")
        ari_rows.append(row)

    pd.DataFrame(ari_rows).to_csv(tab_dir / "tab_clust_cv_ari.csv", index=False)
    pd.DataFrame(assign_rows).to_csv(tab_dir / "tab_clust_cv_assignments.csv", index=False)

    plot_dendrogram(results["acoustic"]["Z"], results["acoustic"]["labels"],
                    "C+V clustering — acoustic", fig_dir / "fig_dendro_cv_acoustic.png", figsize=(10, 5))
    reps = block["representative_layers"]
    for tag in ("whisper", "xlsr"):
        key = f"{tag}_L{int(reps[tag]):02d}"
        plot_dendrogram(results[key]["Z"], results[key]["labels"],
                        f"C+V clustering — {key} PCA-{pca_n}", fig_dir / f"fig_dendro_cv_{key}.png", figsize=(10, 5))
    return results


def speaker_vowel_concat_acoustic(norm_df: pd.DataFrame, vowels: list[str], cols: list[str], min_vowels: int):
    per = norm_df.dropna(subset=cols).groupby(["speaker_id", "phoneme_base"])[cols].mean().reset_index()
    glob = per.groupby("phoneme_base")[cols].mean()
    meta = norm_df[["speaker_id", "L1_status", "gender"]].drop_duplicates("speaker_id")
    rows, speakers, imputed = [], [], {}

    for sp in sorted(per["speaker_id"].unique()):
        sub = per[per["speaker_id"] == sp].set_index("phoneme_base")
        present = [v for v in vowels if v in sub.index]
        if len(present) < min_vowels:
            continue
        parts, n_imp = [], 0
        for v in vowels:
            if v in sub.index:
                parts.append(sub.loc[v, cols].to_numpy(dtype=np.float64))
            elif v in glob.index:
                parts.append(glob.loc[v, cols].to_numpy(dtype=np.float64))
                n_imp += 1
            else:
                parts.append(np.zeros(len(cols), dtype=np.float64))
        rows.append(np.concatenate(parts))
        speakers.append(str(sp))
        imputed[str(sp)] = n_imp

    diagnostic = {
        "n_speakers": len(speakers),
        "vector_dim": len(vowels) * len(cols),
        "imputed_slots_total": int(sum(imputed.values())),
        "speakers_with_imputation": int(sum(x > 0 for x in imputed.values())),
    }
    return np.vstack(rows), speakers, meta.set_index("speaker_id").loc[speakers].reset_index(), diagnostic


def speaker_vowel_concat_neural(X: np.ndarray, meta: pd.DataFrame, vowels: list[str], min_vowels: int):
    d = X.shape[1]
    df = meta[["speaker_id", "phoneme_base", "L1_status", "gender"]].copy()
    by_speaker, glob = {}, {}

    for ph in vowels:
        mask = df["phoneme_base"].to_numpy() == ph
        if np.any(mask):
            glob[ph] = X[mask].astype(np.float64).mean(axis=0)

    for (sp, ph), idx in df.groupby(["speaker_id", "phoneme_base"]).indices.items():
        if ph in vowels:
            by_speaker.setdefault(str(sp), {})[ph] = X[idx].astype(np.float64).mean(axis=0)

    rows, speakers, imputed = [], [], {}
    for sp in sorted(by_speaker):
        if len(by_speaker[sp]) < min_vowels:
            continue
        parts, n_imp = [], 0
        for v in vowels:
            if v in by_speaker[sp]:
                parts.append(by_speaker[sp][v])
            elif v in glob:
                parts.append(glob[v])
                n_imp += 1
            else:
                parts.append(np.zeros(d, dtype=np.float64))
        rows.append(np.concatenate(parts))
        speakers.append(sp)
        imputed[sp] = n_imp

    sp_meta = df[["speaker_id", "L1_status", "gender"]].drop_duplicates("speaker_id")
    diagnostic = {
        "n_speakers": len(speakers),
        "d_pca": d,
        "vector_dim": len(vowels) * d,
        "imputed_slots_total": int(sum(imputed.values())),
        "speakers_with_imputation": int(sum(x > 0 for x in imputed.values())),
    }
    return np.vstack(rows), speakers, sp_meta.set_index("speaker_id").loc[speakers].reset_index(), diagnostic


def run_speaker_clustering(norm_df: pd.DataFrame, cfg: dict, block: dict, tab_dir: Path, fig_dir: Path) -> dict:
    min_tokens = int(block.get("min_phoneme_tokens", 5))
    vowels = filter_phonemes_by_count(
        norm_df,
        block["oral_vowels"],
        min_tokens,
        block["acoustic_vowel_feature_cols"],
    )
    k_range = block["k_range"]
    min_vowels = int(block.get("speaker_min_vowels", 5))
    pca_n = int(block.get("speaker_neural_pca_n_components", block.get("neural_pca_n_components", 50)))

    X_ac, sp_ac, meta_ac, diag_ac = speaker_vowel_concat_acoustic(norm_df, vowels, block["acoustic_vowel_feature_cols"], min_vowels)
    results = {"acoustic": {"Z": linkage_ward(X_ac), "labels": sp_ac, "X": X_ac, "metric": "euclidean", "meta": meta_ac, "diagnostic": diag_ac}}

    meta_cols = ["token_id", "speaker_id", "phoneme_base", "L1_status", "gender"]
    for tag, prefix, layer in model_layer_pairs(cfg, block):
        L = f"L{layer:02d}"
        X, ids = load_pca_npz(Path(block["interim_dir"]) / f"{prefix}_{L}_pca.npz", pca_n)
        X, ids, meta = align_to_metadata(X, ids, norm_df[meta_cols].copy(), f"speaker_{tag}_{L}")
        X_sp, speakers, meta_sp, diag = speaker_vowel_concat_neural(X, meta, vowels, min_vowels)
        key = f"{tag}_{L}"
        Xn = l2_normalise(X_sp.astype(np.float64))
        results[key] = {"Z": linkage_ward(Xn), "labels": speakers, "X": Xn, "metric": "euclidean", "meta": meta_sp, "diagnostic": diag}

    ari_rows, assign_rows = [], []
    for rep, r in results.items():
        r["silhouette_scores"] = silhouette_curve(r["X"], r["Z"], k_range, r["metric"])
        r["k_silhouette"] = k_silhouette_max(r["silhouette_scores"])
        r["k_dendrogram"] = k_dendrogram_elbow(r["Z"], k_range)

        cluster_ling = cut_clusters(r["Z"], 2)
        cluster_sil = cut_clusters(r["Z"], r["k_silhouette"])
        meta_idx = r["meta"].set_index("speaker_id").loc[r["labels"]]
        for gt_col in ["L1_status", "gender"]:
            true = meta_idx[gt_col].astype(str).str.upper().tolist()
            ari_rows.append({
                "representation": rep, "ground_truth": gt_col, "n_speakers": len(true),
                "k_linguistic": 2, "k_silhouette": r["k_silhouette"],
                "ari_at_k_linguistic": float(adjusted_rand_score(true, cluster_ling)),
                "ari_at_k_silhouette": float(adjusted_rand_score(true, cluster_sil)),
            })
        for sp, cl in zip(r["labels"], cluster_ling):
            assign_rows.append({
                "representation": rep,
                "speaker_id": sp,
                "L1_status": meta_idx.loc[sp, "L1_status"],
                "gender": meta_idx.loc[sp, "gender"],
                "cluster": int(cl),
            })

    pd.DataFrame(ari_rows).to_csv(tab_dir / "tab_clust_speaker_ari.csv", index=False)
    pd.DataFrame(assign_rows).to_csv(tab_dir / "tab_clust_speaker_assignments.csv", index=False)

    def speaker_labels(r: dict) -> list[str]:
        m = r["meta"].set_index("speaker_id").loc[r["labels"]]
        return [f"{sp} [{m.loc[sp, 'L1_status']}/{m.loc[sp, 'gender']}]" for sp in r["labels"]]

    plot_dendrogram(results["acoustic"]["Z"], speaker_labels(results["acoustic"]),
                    "Speaker clustering — acoustic", fig_dir / "fig_dendro_speaker_acoustic.png", figsize=(10, 5))
    reps = block["representative_layers"]
    for tag in ("whisper", "xlsr"):
        key = f"{tag}_L{int(reps[tag]):02d}"
        plot_dendrogram(results[key]["Z"], speaker_labels(results[key]),
                        f"Speaker clustering — {key} PCA-{pca_n}", fig_dir / f"fig_dendro_speaker_{key}.png", figsize=(10, 5))
    return results


def build_k_selection_table(vowel_res: dict, cv_res: dict, speaker_res: dict) -> pd.DataFrame:
    rows = []
    for rep, r in vowel_res.items():
        for part, k_ling in [("front_back", 2), ("height", 3)]:
            rows.append({"scope": "vowel", "representation": rep, "partition": part, "k_linguistic": k_ling,
                         "k_silhouette": r["k_silhouette"], "k_dendrogram": r["k_dendrogram"],
                         "silhouette_at_k_linguistic": r["silhouette_scores"].get(k_ling, float("nan"))})
    for rep, r in cv_res.items():
        rows.append({"scope": "cv", "representation": rep, "partition": "consonant_vs_vowel", "k_linguistic": 2,
                     "k_silhouette": r["k_silhouette"], "k_dendrogram": r["k_dendrogram"],
                     "silhouette_at_k_linguistic": r["silhouette_scores"].get(2, float("nan"))})
    for rep, r in speaker_res.items():
        rows.append({"scope": "speaker", "representation": rep, "partition": "L1_status_or_gender", "k_linguistic": 2,
                     "k_silhouette": r["k_silhouette"], "k_dendrogram": r["k_dendrogram"],
                     "silhouette_at_k_linguistic": r["silhouette_scores"].get(2, float("nan"))})
    return pd.DataFrame(rows)


def plot_silhouette_summary(vowel_res: dict, cv_res: dict, speaker_res: dict, block: dict, out_path: Path) -> None:
    reps = block["representative_layers"]
    keys = ["acoustic", f"whisper_L{int(reps['whisper']):02d}", f"xlsr_L{int(reps['xlsr']):02d}"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (scope, store) in zip(axes, [("vowel", vowel_res), ("cv", cv_res), ("speaker", speaker_res)]):
        for key in keys:
            if key not in store:
                continue
            s = store[key]["silhouette_scores"]
            xs = sorted(s)
            ax.plot(xs, [s[x] for x in xs], marker="o", label=key)
        ax.set_title(scope)
        ax.set_xlabel("k")
        ax.set_ylabel("Mean silhouette")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Silhouette curves — representative representations")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ari_summary(tab_dir: Path, out_path: Path) -> None:
    v = pd.read_csv(tab_dir / "tab_clust_vowel_ari.csv")
    cv = pd.read_csv(tab_dir / "tab_clust_cv_ari.csv")
    sp = pd.read_csv(tab_dir / "tab_clust_speaker_ari.csv")
    v["scope_partition"] = "vowel_" + v["partition"]
    cv["scope_partition"] = "cv_" + cv["partition"]
    sp["scope_partition"] = "speaker_" + sp["ground_truth"]
    df = pd.concat([
        v[["representation", "scope_partition", "ari_at_k_linguistic"]],
        cv[["representation", "scope_partition", "ari_at_k_linguistic"]],
        sp[["representation", "scope_partition", "ari_at_k_linguistic"]],
    ], ignore_index=True)
    pivot = df.pivot(index="representation", columns="scope_partition", values="ari_at_k_linguistic")
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Adjusted Rand Index (at k_linguistic)")
    ax.set_title("ARI across scopes and representations (k = ground-truth cardinality)")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.legend(fontsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_q16_table(vowel_res: dict, block: dict, tab_dir: Path) -> pd.DataFrame:
    reps = block["representative_layers"]
    ref_keys = ["acoustic", f"whisper_L{int(reps['whisper']):02d}", f"xlsr_L{int(reps['xlsr']):02d}"]
    rows = []
    for part, k_ling in [("front_back", 2), ("height", 3)]:
        label_map = build_label_map(block["vowel_partitions"][part])
        clusters = {key: dict(zip(vowel_res[key]["labels"], cut_clusters(vowel_res[key]["Z"], k_ling))) for key in ref_keys}
        majority = {}
        for key in ref_keys:
            by_class = {}
            for phon, cl in clusters[key].items():
                if phon in label_map:
                    by_class.setdefault(label_map[phon], []).append(int(cl))
            majority[key] = {cls: int(pd.Series(vals).mode().iat[0]) for cls, vals in by_class.items()}
        phonemes = set(label_map)
        for key in ref_keys:
            phonemes &= set(clusters[key])
        for phon in sorted(phonemes):
            cls = label_map[phon]
            errors = {key: bool(clusters[key][phon] != majority[key].get(cls)) for key in ref_keys}
            row = {"partition": part, "phoneme": phon, "true_class": cls, "systematic_misclassification": all(errors.values())}
            row.update({f"misclassified_{key}": val for key, val in errors.items()})
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(tab_dir / "tab_clust_q16_systematic_errors.csv", index=False)
    return df


def write_consonant_set_table(norm_df: pd.DataFrame, block: dict, tab_dir: Path) -> None:
    manner = block.get("consonant_manner", {})
    place = block.get("consonant_place", {})
    rows = [{"phoneme": c, "manner": manner.get(c, ""), "place": place.get(c, ""),
             "n_tokens": int((norm_df["phoneme_base"] == c).sum())} for c in block["consonants"]]
    pd.DataFrame(rows).to_csv(tab_dir / "tab_clust_consonant_set.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 10: hierarchical clustering")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    block = cfg["clustering"]
    tab_dir = Path(block["tables_dir"])
    fig_dir = Path(block["figures_dir"]) / "clustering"
    ensure_dirs(tab_dir, fig_dir)

    norm_df = load_csv(Path(block["input_acoustic_norm"]), REQUIRED_NORM_COLUMNS, "normalised acoustic table")
    raw_df = load_csv(Path(block["input_acoustic_raw"]), REQUIRED_RAW_COLUMNS, "raw acoustic table")

    print(f"[clustering] n_tokens normalised: {len(norm_df)}")
    print(f"[clustering] vowels: {block['oral_vowels']}")
    print(f"[clustering] consonants: {block['consonants']}")

    write_consonant_set_table(norm_df, block, tab_dir)
    vowel_res = run_vowel_clustering(norm_df, cfg, block, tab_dir, fig_dir)
    cv_res = run_cv_clustering(norm_df, raw_df, cfg, block, tab_dir, fig_dir)
    speaker_res = run_speaker_clustering(norm_df, cfg, block, tab_dir, fig_dir)

    build_k_selection_table(vowel_res, cv_res, speaker_res).to_csv(tab_dir / "tab_clust_k_selection.csv", index=False)
    plot_silhouette_summary(vowel_res, cv_res, speaker_res, block, fig_dir / "fig_silhouette_summary.png")
    build_q16_table(vowel_res, block, tab_dir)
    plot_ari_summary(tab_dir, fig_dir / "fig_ari_summary.png")

    print("[clustering] done")


if __name__ == "__main__":
    main()

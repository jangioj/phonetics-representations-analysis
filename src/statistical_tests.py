"""
statistical_tests.py

Inferential/statistical tests for PDF Section 6.

This stage reads the normalised acoustic table and raw neural embeddings, then writes
statistical tables and diagnostic figures for:
  - L1/L2 acoustic comparisons with assumption checks and FDR correction;
  - residual gender checks after Lobanov normalisation;
  - L1/L2 neural permutation tests on cosine centroid distances;
  - inter-phoneme acoustic/neural distance matrices and Mantel comparisons;
  - speaker-level bootstrap CIs for selected phoneme-pair distances;
  - nearest-centroid phoneme identification with leave-one-speaker-out CV,
    per-class F1, confusion matrices, and McNemar comparisons.

Design notes
------------
- Acoustic group tests use speaker-level phoneme means to avoid token-level
  pseudo-replication.
- Neural group tests use speaker-level phoneme centroids before permuting labels.
- Neural distances and classifiers use raw 1024-d embeddings, not PCA/UMAP.
- Bootstrap resampling is performed at the speaker level, as required by the PDF.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

ORAL_VOWEL_ORDER = ["i", "e", "ɛ", "a", "ɑ", "ɔ", "o", "u", "y", "ø", "œ", "ə"]
REQUIRED_ACOUSTIC_COLUMNS = {
    "token_id",
    "phoneme_base",
    "speaker_id",
    "L1_status",
    "gender",
    "is_vowel",
}


# ---------------------------------------------------------------------
# Generic helpers
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


def safe_float(x: float | np.floating | None) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def bh_fdr(p_values: Iterable[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR correction, preserving NaNs."""
    p = np.asarray(list(p_values), dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if valid.sum() == 0:
        return q

    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    tmp = np.empty_like(adjusted)
    tmp[order] = adjusted
    q[valid] = tmp
    return q


def upper_tri(D: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(D, k=1)
    return D[iu]


def rank_mantel(D1: np.ndarray, D2: np.ndarray, n_perm: int, rng: np.random.Generator) -> tuple[float, float, int]:
    """Spearman Mantel-style test on distance-matrix upper triangles."""
    if D1.shape != D2.shape:
        raise ValueError(f"Distance matrix shape mismatch: {D1.shape} vs {D2.shape}")

    n = D1.shape[0]
    v1 = upper_tri(D1)
    v2 = upper_tri(D2)
    r_obs = safe_float(stats.spearmanr(v1, v2).statistic)

    if not np.isfinite(r_obs):
        return float("nan"), float("nan"), int(n_perm)

    # Positive one-sided Mantel p-value: how often the permuted structure is
    # at least as aligned as the observed structure.
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        D2_perm = D2[perm][:, perm]
        r_perm = safe_float(stats.spearmanr(v1, upper_tri(D2_perm)).statistic)
        if np.isfinite(r_perm) and r_perm >= r_obs:
            count += 1

    p_value = float((count + 1) / (n_perm + 1))
    return r_obs, p_value, int(n_perm)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(1.0 - np.dot(a, b) / (na * nb))


def pairwise_euclidean(C: np.ndarray) -> np.ndarray:
    diffs = C[:, None, :] - C[None, :, :]
    return np.sqrt(np.sum(diffs ** 2, axis=-1))


def pairwise_cosine_distance(C: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(C, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Cn = C / norms
    return 1.0 - (Cn @ Cn.T)


def regularised_inverse_covariance(X: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    cov = np.cov(X.T)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    cov = np.asarray(cov, dtype=np.float64)
    scale = float(np.trace(cov) / cov.shape[0]) if cov.shape[0] > 0 else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    cov = cov + np.eye(cov.shape[0]) * ridge * scale
    return np.linalg.pinv(cov)


def pairwise_mahalanobis(C: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    n = C.shape[0]
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diff = C[i] - C[j]
            d = float(np.sqrt(max(diff @ inv_cov @ diff, 0.0)))
            D[i, j] = d
            D[j, i] = d
    return D


# ---------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------

def load_acoustic(path: Path, acoustic_cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing acoustic table: {path}")
    df = pd.read_csv(path)
    require_columns(df, REQUIRED_ACOUSTIC_COLUMNS | set(acoustic_cols), context=str(path))

    if df["token_id"].duplicated().any():
        dup = int(df["token_id"].duplicated().sum())
        raise ValueError(f"token_id must be unique in {path}; found {dup} duplicates")

    df["group"] = df["L1_status"].astype(str) + "/" + df["gender"].astype(str)
    return df


def load_raw_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing raw neural NPZ: {path}")
    d = np.load(path, allow_pickle=True)
    if "embeddings" not in d or "token_ids" not in d:
        raise KeyError(f"{path} must contain arrays 'embeddings' and 'token_ids'")
    X = d["embeddings"].astype(np.float32)
    ids = d["token_ids"]
    if len(X) != len(ids):
        raise ValueError(f"{path}: embeddings/token_ids mismatch: {len(X)} vs {len(ids)}")
    return X, ids


def align_metadata(ids: np.ndarray, meta: pd.DataFrame, context: str) -> pd.DataFrame:
    meta_by_id = meta.set_index("token_id")
    missing = pd.Index(ids).difference(meta_by_id.index)
    if len(missing) > 0:
        raise KeyError(f"{context}: {len(missing)} token_ids are missing from metadata")
    return meta_by_id.loc[ids].reset_index()


def model_layer_pairs(cfg: dict, block: dict) -> list[tuple[str, str, int]]:
    pairs: list[tuple[str, str, int]] = []
    for tag, key in [("whisper", "extract_neural_whisper"), ("xlsr", "extract_neural_xlsr")]:
        up = cfg[key]
        configured = [int(x) for x in up["layers"]]
        requested = [int(x) for x in block.get("neural_layers", {}).get(tag, configured)]
        for layer in requested:
            if layer not in configured:
                raise ValueError(f"Requested {tag} layer {layer}, but configured layers are {configured}")
            pairs.append((tag, up["output_prefix"], int(layer)))
    return pairs


# ---------------------------------------------------------------------
# Acoustic L1/L2 and gender tests
# ---------------------------------------------------------------------

def speaker_phoneme_means(df: pd.DataFrame, phonemes: list[str], feature_cols: list[str]) -> pd.DataFrame:
    sub = df[df["phoneme_base"].isin(phonemes)].copy()
    keep = ["speaker_id", "phoneme_base", "L1_status", "gender"] + feature_cols
    sub = sub[keep]
    agg = (
        sub.groupby(["speaker_id", "phoneme_base", "L1_status", "gender"], as_index=False)
        .agg({c: "mean" for c in feature_cols})
    )
    return agg


def run_two_group_feature_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    group_a: str,
    group_b: str,
    alpha: float,
) -> dict:
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    values_a = values_a[np.isfinite(values_a)]
    values_b = values_b[np.isfinite(values_b)]

    out = {
        f"n_{group_a}": int(len(values_a)),
        f"n_{group_b}": int(len(values_b)),
        f"mean_{group_a}": float(np.mean(values_a)) if len(values_a) else np.nan,
        f"mean_{group_b}": float(np.mean(values_b)) if len(values_b) else np.nan,
        "difference_group_b_minus_group_a": (
            float(np.mean(values_b) - np.mean(values_a)) if len(values_a) and len(values_b) else np.nan
        ),
        f"shapiro_p_{group_a}": np.nan,
        f"shapiro_p_{group_b}": np.nan,
        "levene_p": np.nan,
        "selected_test": "insufficient_data",
        "statistic": np.nan,
        "p_value": np.nan,
        "assumptions_hold": False,
    }

    if len(values_a) < 2 or len(values_b) < 2:
        return out

    if 3 <= len(values_a) <= 5000:
        out[f"shapiro_p_{group_a}"] = safe_float(stats.shapiro(values_a).pvalue)
    if 3 <= len(values_b) <= 5000:
        out[f"shapiro_p_{group_b}"] = safe_float(stats.shapiro(values_b).pvalue)

    out["levene_p"] = safe_float(stats.levene(values_a, values_b, center="median").pvalue)

    normality_known = np.isfinite(out[f"shapiro_p_{group_a}"]) and np.isfinite(out[f"shapiro_p_{group_b}"])
    normality_ok = (
        normality_known
        and out[f"shapiro_p_{group_a}"] > alpha
        and out[f"shapiro_p_{group_b}"] > alpha
    )
    variance_ok = np.isfinite(out["levene_p"]) and out["levene_p"] > alpha

    if normality_ok and variance_ok:
        res = stats.ttest_ind(values_a, values_b, equal_var=True, nan_policy="omit")
        out["selected_test"] = "two_sample_t_test"
        out["statistic"] = safe_float(res.statistic)
        out["p_value"] = safe_float(res.pvalue)
        out["assumptions_hold"] = True
    else:
        res = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
        out["selected_test"] = "mann_whitney_u"
        out["statistic"] = safe_float(res.statistic)
        out["p_value"] = safe_float(res.pvalue)
        out["assumptions_hold"] = False

    return out


def acoustic_l1_l2_tests(
    spk_means: pd.DataFrame,
    phonemes: list[str],
    feature_cols: list[str],
    alpha: float,
    min_speakers_per_group: int,
) -> pd.DataFrame:
    rows: list[dict] = []

    for feat in feature_cols:
        for ph in phonemes:
            cell = spk_means[spk_means["phoneme_base"] == ph]
            l1 = cell.loc[cell["L1_status"] == "L1", feat].dropna().to_numpy()
            l2 = cell.loc[cell["L1_status"] == "L2", feat].dropna().to_numpy()

            row = {"phoneme": ph, "feature": feat, "fdr_family": f"acoustic_L1_L2_{feat}"}
            row.update(run_two_group_feature_test(l1, l2, "L1", "L2", alpha=alpha))

            if len(l1) < min_speakers_per_group or len(l2) < min_speakers_per_group:
                row["selected_test"] = "insufficient_data"
                row["p_value"] = np.nan

            rows.append(row)

    out = pd.DataFrame(rows)
    out["q_value_bh"] = np.nan
    out["significant_bh"] = False
    for family, idx in out.groupby("fdr_family").groups.items():
        q = bh_fdr(out.loc[idx, "p_value"])
        out.loc[idx, "q_value_bh"] = q
        out.loc[idx, "significant_bh"] = q < alpha
    return out


def acoustic_gender_tests(
    spk_means: pd.DataFrame,
    phonemes: list[str],
    feature_cols: list[str],
    alpha: float,
    min_speakers_per_group: int,
) -> pd.DataFrame:
    """
    Residual gender test after Lobanov normalisation.

    Gender is a between-speaker factor in the actual data, so a literal paired
    gender test is not possible without a matching variable. We therefore use
    speaker-level independent tests and make this explicit in the output.
    """
    rows: list[dict] = []

    for feat in feature_cols:
        for ph in phonemes:
            cell = spk_means[spk_means["phoneme_base"] == ph]
            f = cell.loc[cell["gender"] == "F", feat].dropna().to_numpy()
            m = cell.loc[cell["gender"] == "M", feat].dropna().to_numpy()

            row = {
                "phoneme": ph,
                "feature": feat,
                "fdr_family": f"gender_{feat}",
                "design_note": "speaker_level_independent_test; literal paired gender test is not possible because gender is between-speaker",
            }
            row.update(run_two_group_feature_test(f, m, "F", "M", alpha=alpha))

            if len(f) < min_speakers_per_group or len(m) < min_speakers_per_group:
                row["selected_test"] = "insufficient_data"
                row["p_value"] = np.nan

            rows.append(row)

    out = pd.DataFrame(rows)
    out["q_value_bh"] = np.nan
    out["significant_bh"] = False
    for family, idx in out.groupby("fdr_family").groups.items():
        q = bh_fdr(out.loc[idx, "p_value"])
        out.loc[idx, "q_value_bh"] = q
        out.loc[idx, "significant_bh"] = q < alpha
    return out


def plot_qq_by_vowel_group(
    spk_means: pd.DataFrame,
    phonemes: list[str],
    feature: str,
    out_path: Path,
) -> None:
    ncols = 4
    nrows = int(np.ceil(len(phonemes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.1 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for ax, ph in zip(axes, phonemes):
        cell = spk_means[spk_means["phoneme_base"] == ph]
        ax.set_title(f"/{ph}/")
        ax.grid(alpha=0.25)

        for status, marker in [("L1", "o"), ("L2", "x")]:
            vals = cell.loc[cell["L1_status"] == status, feature].dropna().to_numpy(dtype=float)
            if len(vals) < 3:
                continue
            osm, osr = stats.probplot(vals, dist="norm", fit=False)
            ax.scatter(osm, osr, s=22, alpha=0.75, marker=marker, label=status)

        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Ordered values")
        ax.legend(fontsize=8, frameon=True)

    for ax in axes[len(phonemes):]:
        ax.axis("off")

    fig.suptitle(f"Q-Q diagnostics by vowel and L1 status: {feature}", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------
# Neural L1/L2 permutation tests
# ---------------------------------------------------------------------

def speaker_neural_centroids(
    emb: np.ndarray,
    ids: np.ndarray,
    meta: pd.DataFrame,
    phonemes: list[str],
) -> pd.DataFrame:
    aligned = align_metadata(ids, meta, context="speaker_neural_centroids")
    rows: list[dict] = []

    for (speaker, ph, l1, gender), grp in aligned.groupby(["speaker_id", "phoneme_base", "L1_status", "gender"]):
        if ph not in phonemes:
            continue
        idx = grp.index.to_numpy()
        if len(idx) == 0:
            continue
        rows.append({
            "speaker_id": speaker,
            "phoneme": ph,
            "L1_status": l1,
            "gender": gender,
            "n_tokens": int(len(idx)),
            "centroid": emb[idx].mean(axis=0).astype(np.float32),
        })

    return pd.DataFrame(rows)


def neural_l1_l2_permutation_tests(
    centroids: pd.DataFrame,
    phonemes: list[str],
    model: str,
    layer: int,
    n_perm: int,
    min_speakers_per_group: int,
    rng: np.random.Generator,
    alpha: float,
) -> pd.DataFrame:
    rows: list[dict] = []

    for ph in phonemes:
        cell = centroids[centroids["phoneme"] == ph].copy()
        n_l1 = int((cell["L1_status"] == "L1").sum())
        n_l2 = int((cell["L1_status"] == "L2").sum())

        row = {
            "model": model,
            "layer": int(layer),
            "phoneme": ph,
            "n_speakers_L1": n_l1,
            "n_speakers_L2": n_l2,
            "observed_cosine_distance": np.nan,
            "p_value": np.nan,
            "n_permutations": int(n_perm),
            "status": "ok",
        }

        if n_l1 < min_speakers_per_group or n_l2 < min_speakers_per_group:
            row["status"] = "insufficient_data"
            rows.append(row)
            continue

        X = np.stack(cell["centroid"].to_numpy())
        labels = cell["L1_status"].to_numpy()
        obs = cosine_distance(X[labels == "L1"].mean(axis=0), X[labels == "L2"].mean(axis=0))
        row["observed_cosine_distance"] = obs

        count = 0
        n = len(labels)
        n_l1_perm = int(np.sum(labels == "L1"))
        for _ in range(n_perm):
            perm = rng.permutation(n)
            l1_idx = perm[:n_l1_perm]
            l2_idx = perm[n_l1_perm:]
            d = cosine_distance(X[l1_idx].mean(axis=0), X[l2_idx].mean(axis=0))
            if np.isfinite(d) and d >= obs:
                count += 1

        row["p_value"] = float((count + 1) / (n_perm + 1))
        rows.append(row)

    out = pd.DataFrame(rows)
    out["fdr_family"] = f"neural_L1_L2_{model}_L{layer:02d}"
    out["q_value_bh"] = bh_fdr(out["p_value"])
    out["significant_bh"] = out["q_value_bh"] < alpha
    return out


# ---------------------------------------------------------------------
# Distance matrices and bootstrap CIs
# ---------------------------------------------------------------------

def acoustic_speaker_centroids(df: pd.DataFrame, phonemes: list[str], feature_cols: list[str]) -> pd.DataFrame:
    sub = df[df["phoneme_base"].isin(phonemes)].dropna(subset=feature_cols).copy()
    rows = []
    for (speaker, ph, l1, gender), grp in sub.groupby(["speaker_id", "phoneme_base", "L1_status", "gender"]):
        rows.append({
            "speaker_id": speaker,
            "phoneme": ph,
            "L1_status": l1,
            "gender": gender,
            "n_tokens": int(len(grp)),
            "centroid": grp[feature_cols].mean(axis=0).to_numpy(dtype=np.float32),
        })
    return pd.DataFrame(rows)


def representation_centroid_matrix(spk_centroids: pd.DataFrame, phonemes: list[str]) -> tuple[np.ndarray, list[str]]:
    rows: list[np.ndarray] = []
    kept: list[str] = []
    for ph in phonemes:
        cell = spk_centroids[spk_centroids["phoneme"] == ph]
        if cell.empty:
            continue
        rows.append(np.stack(cell["centroid"].to_numpy()).mean(axis=0))
        kept.append(ph)
    return np.stack(rows).astype(np.float64), kept


def pooled_within_covariance(spk_centroids: pd.DataFrame, phonemes: list[str]) -> np.ndarray:
    centered: list[np.ndarray] = []
    for ph in phonemes:
        cell = spk_centroids[spk_centroids["phoneme"] == ph]
        if len(cell) < 2:
            continue
        X = np.stack(cell["centroid"].to_numpy()).astype(np.float64)
        centered.append(X - X.mean(axis=0, keepdims=True))
    if not centered:
        raise RuntimeError("Cannot compute pooled covariance: no phoneme has >=2 speaker centroids")
    return regularised_inverse_covariance(np.vstack(centered))


def write_distance_matrix(D: np.ndarray, labels: list[str], out_csv: Path) -> None:
    pd.DataFrame(D, index=labels, columns=labels).to_csv(out_csv)


def plot_distance_matrix(D: np.ndarray, labels: list[str], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    im = ax.imshow(D, aspect="equal")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(title)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5, alpha=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.colorbar(im, ax=ax, label="distance", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def bootstrap_pair_distances(
    acoustic_spk: pd.DataFrame,
    neural_spk_by_rep: dict[str, pd.DataFrame],
    phoneme_pairs: list[list[str]],
    feature_labels: list[str],
    n_boot: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    speakers = sorted(acoustic_spk["speaker_id"].unique())
    rows: list[dict] = []

    def sample_rows(spk_df: pd.DataFrame, sampled_speakers: np.ndarray) -> pd.DataFrame:
        parts = []
        for draw_idx, speaker in enumerate(sampled_speakers):
            part = spk_df[spk_df["speaker_id"] == speaker].copy()
            if part.empty:
                continue
            part["bootstrap_draw_id"] = draw_idx
            parts.append(part)
        if not parts:
            return pd.DataFrame(columns=list(spk_df.columns) + ["bootstrap_draw_id"])
        return pd.concat(parts, ignore_index=True)

    for pair in phoneme_pairs:
        if len(pair) != 2:
            continue
        p, q = pair[0], pair[1]
        dist_store: dict[str, list[float]] = defaultdict(list)

        for _ in range(n_boot):
            sampled = rng.choice(speakers, size=len(speakers), replace=True)

            # Acoustic distances.
            ac_b = sample_rows(acoustic_spk, sampled)
            if {p, q}.issubset(set(ac_b["phoneme"])):
                C_b, kept_b = representation_centroid_matrix(ac_b, [p, q])
                if kept_b == [p, q]:
                    dist_store["acoustic_euclidean"].append(float(np.linalg.norm(C_b[0] - C_b[1])))
                    try:
                        inv_cov_b = pooled_within_covariance(ac_b, [p, q])
                        dist_store["acoustic_mahalanobis"].append(float(pairwise_mahalanobis(C_b, inv_cov_b)[0, 1]))
                    except Exception:
                        dist_store["acoustic_mahalanobis"].append(np.nan)

            # Neural distances.
            for rep_name, spk_df in neural_spk_by_rep.items():
                nb = sample_rows(spk_df, sampled)
                if {p, q}.issubset(set(nb["phoneme"])):
                    Cn, kept_n = representation_centroid_matrix(nb, [p, q])
                    if kept_n == [p, q]:
                        dist_store[rep_name].append(cosine_distance(Cn[0], Cn[1]))

        for rep_name, vals in dist_store.items():
            arr = np.asarray(vals, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) == 0:
                rows.append({
                    "phoneme_a": p,
                    "phoneme_b": q,
                    "representation": rep_name,
                    "n_boot_valid": 0,
                    "distance_mean": np.nan,
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                })
                continue
            rows.append({
                "phoneme_a": p,
                "phoneme_b": q,
                "representation": rep_name,
                "n_boot_valid": int(len(arr)),
                "distance_mean": float(np.mean(arr)),
                "ci95_low": float(np.percentile(arr, 2.5)),
                "ci95_high": float(np.percentile(arr, 97.5)),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Nearest-centroid classification and McNemar
# ---------------------------------------------------------------------

def nearest_centroid_predict(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    metric: str,
) -> np.ndarray:
    labels = sorted(pd.Series(train_y).dropna().unique())
    centroids = []
    kept = []
    for lab in labels:
        idx = train_y == lab
        if np.any(idx):
            centroids.append(train_X[idx].mean(axis=0))
            kept.append(lab)
    C = np.stack(centroids).astype(np.float64)

    if metric == "euclidean":
        diffs = test_X[:, None, :] - C[None, :, :]
        D = np.sqrt(np.sum(diffs ** 2, axis=-1))
    elif metric == "cosine":
        test_norms = np.linalg.norm(test_X, axis=1, keepdims=True)
        test_norms[test_norms == 0] = 1.0
        C_norms = np.linalg.norm(C, axis=1, keepdims=True)
        C_norms[C_norms == 0] = 1.0
        D = 1.0 - (test_X / test_norms) @ (C / C_norms).T
    else:
        raise ValueError(f"Unknown classifier metric: {metric}")

    return np.asarray(kept, dtype=object)[np.argmin(D, axis=1)]


def loso_classifier(
    X: np.ndarray,
    meta: pd.DataFrame,
    phonemes: list[str],
    rep_name: str,
    metric: str,
) -> pd.DataFrame:
    work = meta[meta["phoneme_base"].isin(phonemes)].copy().reset_index(drop=True)
    Xw = X[work.index]

    rows: list[pd.DataFrame] = []
    for speaker in sorted(work["speaker_id"].unique()):
        train_mask = work["speaker_id"].to_numpy() != speaker
        test_mask = ~train_mask
        train_y = work.loc[train_mask, "phoneme_base"].to_numpy()
        test_y = work.loc[test_mask, "phoneme_base"].to_numpy()

        # Skip impossible folds, but keep the pipeline robust.
        if len(np.unique(train_y)) < 2 or len(test_y) == 0:
            continue

        pred = nearest_centroid_predict(Xw[train_mask], train_y, Xw[test_mask], metric=metric)
        fold = work.loc[test_mask, ["token_id", "speaker_id", "L1_status", "gender", "phoneme_base"]].copy()
        fold = fold.rename(columns={"phoneme_base": "true_phoneme"})
        fold["pred_phoneme"] = pred
        fold["correct"] = fold["true_phoneme"].to_numpy() == fold["pred_phoneme"].to_numpy()
        fold["representation"] = rep_name
        rows.append(fold)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def classifier_summary(preds: pd.DataFrame, phonemes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    acc_rows = []
    f1_rows = []

    for rep, sub in preds.groupby("representation"):
        acc_rows.append({
            "representation": rep,
            "group": "all",
            "n_tokens": int(len(sub)),
            "accuracy": float(sub["correct"].mean()) if len(sub) else np.nan,
        })
        for status, g in sub.groupby("L1_status"):
            acc_rows.append({
                "representation": rep,
                "group": str(status),
                "n_tokens": int(len(g)),
                "accuracy": float(g["correct"].mean()) if len(g) else np.nan,
            })

        y_true = sub["true_phoneme"].to_numpy()
        y_pred = sub["pred_phoneme"].to_numpy()
        scores = f1_score(y_true, y_pred, labels=phonemes, average=None, zero_division=0)
        for ph, score in zip(phonemes, scores):
            n = int(np.sum(y_true == ph))
            f1_rows.append({
                "representation": rep,
                "phoneme": ph,
                "n_true_tokens": n,
                "f1": float(score),
            })

    return pd.DataFrame(acc_rows), pd.DataFrame(f1_rows)


def plot_confusion(preds: pd.DataFrame, phonemes: list[str], rep_name: str, out_path: Path) -> None:
    sub = preds[preds["representation"] == rep_name]
    if sub.empty:
        return
    cm = confusion_matrix(sub["true_phoneme"], sub["pred_phoneme"], labels=phonemes)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(len(phonemes)))
    ax.set_xticklabels(phonemes, rotation=90)
    ax.set_yticks(range(len(phonemes)))
    ax.set_yticklabels(phonemes)
    ax.set_xlabel("Predicted phoneme")
    ax.set_ylabel("True phoneme")
    ax.set_title(f"Nearest-centroid confusion matrix: {rep_name}")
    fig.colorbar(im, ax=ax, label="row-normalised proportion", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[int, int, float]:
    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return b, c, float("nan")
    p = stats.binomtest(k=min(b, c), n=n, p=0.5, alternative="two-sided").pvalue
    return b, c, float(p)


def mcnemar_table(preds: pd.DataFrame, alpha: float) -> pd.DataFrame:
    reps = sorted(preds["representation"].unique())
    rows: list[dict] = []

    for i, rep_a in enumerate(reps):
        for rep_b in reps[i + 1:]:
            a = preds[preds["representation"] == rep_a][["token_id", "correct", "L1_status"]]
            b = preds[preds["representation"] == rep_b][["token_id", "correct", "L1_status"]]
            merged = a.merge(b, on="token_id", suffixes=("_a", "_b"))
            if merged.empty:
                continue

            for group_name, sub in [("all", merged), *list(merged.groupby("L1_status_a"))]:
                b_count, c_count, p = mcnemar_exact(sub["correct_a"].to_numpy(), sub["correct_b"].to_numpy())
                rows.append({
                    "rep_a": rep_a,
                    "rep_b": rep_b,
                    "group": str(group_name),
                    "n_matched_tokens": int(len(sub)),
                    "a_correct_b_wrong": b_count,
                    "a_wrong_b_correct": c_count,
                    "p_value": p,
                })

    out = pd.DataFrame(rows)
    out["q_value_bh"] = bh_fdr(out["p_value"]) if not out.empty else []
    out["significant_bh"] = out["q_value_bh"] < alpha if not out.empty else []
    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    block = cfg["statistical_tests"]

    acoustic_cols = block.get("acoustic_feature_cols", ["F1_norm", "F2_norm"])
    alpha = float(block.get("alpha", 0.05))
    min_speakers = int(block.get("min_speakers_per_group", 2))
    seed = int(block.get("random_state", 42))

    in_acoustic = Path(block["input_acoustic"])
    interim_dir = Path(block["interim_dir"])
    tab_dir = Path(block["tables_dir"])
    fig_base = Path(block["figures_dir"]) / "statistical_tests"
    fig_acoustic = fig_base / "acoustic"
    fig_dist = fig_base / "distances"
    fig_clf = fig_base / "classification"
    ensure_dirs(tab_dir, fig_acoustic, fig_dist, fig_clf)

    configured_vowels = block.get("oral_vowels", ORAL_VOWEL_ORDER)
    phonemes = [v for v in ORAL_VOWEL_ORDER if v in configured_vowels]

    rng = np.random.default_rng(seed)
    df = load_acoustic(in_acoustic, acoustic_cols)
    meta = df.copy()

    # Keep only vowels with complete acoustic data for the acoustic core analyses.
    observed_phonemes = []
    for ph in phonemes:
        n = df.loc[df["phoneme_base"] == ph, acoustic_cols].dropna().shape[0]
        if n >= 5:
            observed_phonemes.append(ph)
    phonemes = observed_phonemes
    if len(phonemes) < 3:
        raise RuntimeError(f"Too few oral vowels with complete acoustic data: {phonemes}")

    print(f"[statistical_tests] vowel set: {phonemes}")

    # ------------------------------------------------------------------
    # Acoustic group tests.
    # ------------------------------------------------------------------
    spk_means = speaker_phoneme_means(df, phonemes, acoustic_cols)

    acoustic_tests = acoustic_l1_l2_tests(
        spk_means=spk_means,
        phonemes=phonemes,
        feature_cols=acoustic_cols,
        alpha=alpha,
        min_speakers_per_group=min_speakers,
    )
    acoustic_tests.to_csv(tab_dir / "tab_stat_acoustic_l1_l2_tests.csv", index=False)
    print(f"Wrote {tab_dir / 'tab_stat_acoustic_l1_l2_tests.csv'}")

    gender_tests = acoustic_gender_tests(
        spk_means=spk_means,
        phonemes=phonemes,
        feature_cols=acoustic_cols,
        alpha=alpha,
        min_speakers_per_group=min_speakers,
    )
    gender_tests.to_csv(tab_dir / "tab_stat_acoustic_gender_tests.csv", index=False)
    print(f"Wrote {tab_dir / 'tab_stat_acoustic_gender_tests.csv'}")

    for feat in acoustic_cols:
        plot_qq_by_vowel_group(
            spk_means=spk_means,
            phonemes=phonemes,
            feature=feat,
            out_path=fig_acoustic / f"fig_qq_{feat}_by_vowel_group.png",
        )
        print(f"Wrote {fig_acoustic / f'fig_qq_{feat}_by_vowel_group.png'}")

    # ------------------------------------------------------------------
    # Neural permutation tests, distance matrices, classifier inputs.
    # ------------------------------------------------------------------
    neural_perm_tables: list[pd.DataFrame] = []
    neural_spk_by_rep: dict[str, pd.DataFrame] = {}
    neural_distance_by_rep: dict[str, tuple[np.ndarray, list[str]]] = {}
    all_predictions: list[pd.DataFrame] = []

    # Acoustic distance matrices and acoustic classifier.
    acoustic_spk = acoustic_speaker_centroids(df, phonemes, acoustic_cols)
    C_ac, kept_ac = representation_centroid_matrix(acoustic_spk, phonemes)
    if kept_ac != phonemes:
        phonemes = kept_ac
        print(f"[warning] acoustic kept phoneme set changed to: {phonemes}")

    inv_cov = pooled_within_covariance(acoustic_spk, phonemes)
    D_ac_e = pairwise_euclidean(C_ac)
    D_ac_m = pairwise_mahalanobis(C_ac, inv_cov)

    write_distance_matrix(D_ac_e, phonemes, tab_dir / "tab_stat_distance_matrix_acoustic_euclidean.csv")
    write_distance_matrix(D_ac_m, phonemes, tab_dir / "tab_stat_distance_matrix_acoustic_mahalanobis.csv")
    plot_distance_matrix(D_ac_e, phonemes, "Acoustic Euclidean distance: F1/F2", fig_dist / "fig_distance_matrix_acoustic_euclidean.png")
    plot_distance_matrix(D_ac_m, phonemes, "Acoustic Mahalanobis distance: F1/F2", fig_dist / "fig_distance_matrix_acoustic_mahalanobis.png")
    print(f"Wrote acoustic distance matrices and figures")

    ac_tokens = df[df["phoneme_base"].isin(phonemes)].dropna(subset=acoustic_cols).copy().reset_index(drop=True)
    X_ac = ac_tokens[acoustic_cols].to_numpy(dtype=np.float32)
    pred_ac = loso_classifier(X_ac, ac_tokens, phonemes, rep_name="acoustic_euclidean", metric="euclidean")
    all_predictions.append(pred_ac)

    n_perm_neural = int(block.get("neural_l1_l2_n_permutations", 5000))
    n_perm_mantel = int(block.get("mantel_n_permutations", 5000))

    plot_layers = block.get("classification_plot_layers", {})

    for tag, prefix, layer in model_layer_pairs(cfg, block):
        layer_str = f"L{layer:02d}"
        rep_name = f"{tag}_{layer_str}"
        raw_path = interim_dir / f"{prefix}_{layer_str}.npz"
        emb, ids = load_raw_npz(raw_path)
        aligned = align_metadata(ids, meta, context=str(raw_path))

        spk_neural = speaker_neural_centroids(emb, ids, meta, phonemes)
        neural_spk_by_rep[rep_name] = spk_neural

        perm_df = neural_l1_l2_permutation_tests(
            centroids=spk_neural,
            phonemes=phonemes,
            model=tag,
            layer=layer,
            n_perm=n_perm_neural,
            min_speakers_per_group=min_speakers,
            rng=np.random.default_rng(seed + layer + (0 if tag == "whisper" else 1000)),
            alpha=alpha,
        )
        neural_perm_tables.append(perm_df)

        C_n, kept_n = representation_centroid_matrix(spk_neural, phonemes)
        if kept_n == phonemes:
            D_n = pairwise_cosine_distance(C_n)
            neural_distance_by_rep[rep_name] = (D_n, kept_n)
            write_distance_matrix(D_n, kept_n, tab_dir / f"tab_stat_distance_matrix_{rep_name}.csv")
        else:
            print(f"[warning] {rep_name}: phoneme set mismatch in distance matrix; skipped")

        # Token-level LOSO classifier. Embedding order is aligned to ids.
        token_mask = aligned["phoneme_base"].isin(phonemes).to_numpy()
        pred_n = loso_classifier(
            X=emb[token_mask],
            meta=aligned.loc[token_mask].reset_index(drop=True),
            phonemes=phonemes,
            rep_name=rep_name,
            metric="cosine",
        )
        all_predictions.append(pred_n)

        if int(layer) in [int(x) for x in plot_layers.get(tag, [])]:
            print(f"Prepared classifier outputs for plotted representation: {rep_name}")

    neural_perm = pd.concat(neural_perm_tables, ignore_index=True) if neural_perm_tables else pd.DataFrame()
    neural_perm.to_csv(tab_dir / "tab_stat_neural_l1_l2_permutation.csv", index=False)
    print(f"Wrote {tab_dir / 'tab_stat_neural_l1_l2_permutation.csv'}")

    # ------------------------------------------------------------------
    # Mantel comparisons on distance matrices.
    # ------------------------------------------------------------------
    mantel_rows: list[dict] = []
    acoustic_distances = {
        "acoustic_euclidean": D_ac_e,
        "acoustic_mahalanobis": D_ac_m,
    }

    for ac_name, D_ac in acoustic_distances.items():
        for rep_name, (D_n, labels_n) in neural_distance_by_rep.items():
            if labels_n != phonemes:
                continue
            r, p, n = rank_mantel(D_ac, D_n, n_perm=n_perm_mantel, rng=np.random.default_rng(seed))
            mantel_rows.append({
                "rep_a": ac_name,
                "rep_b": rep_name,
                "n_phonemes": int(len(phonemes)),
                "phonemes": " ".join(phonemes),
                "mantel_r_spearman": r,
                "p_value": p,
                "n_permutations": n,
            })

    # Whisper/XLS-R at matching layers.
    for layer in sorted(set(int(x.split("_L")[-1]) for x in neural_distance_by_rep)):
        w = f"whisper_L{layer:02d}"
        x = f"xlsr_L{layer:02d}"
        if w in neural_distance_by_rep and x in neural_distance_by_rep:
            D_w, labels_w = neural_distance_by_rep[w]
            D_x, labels_x = neural_distance_by_rep[x]
            if labels_w == labels_x == phonemes:
                r, p, n = rank_mantel(D_w, D_x, n_perm=n_perm_mantel, rng=np.random.default_rng(seed + layer))
                mantel_rows.append({
                    "rep_a": w,
                    "rep_b": x,
                    "n_phonemes": int(len(phonemes)),
                    "phonemes": " ".join(phonemes),
                    "mantel_r_spearman": r,
                    "p_value": p,
                    "n_permutations": n,
                })

    mantel_df = pd.DataFrame(mantel_rows)
    mantel_df.to_csv(tab_dir / "tab_stat_distance_mantel.csv", index=False)
    print(f"Wrote {tab_dir / 'tab_stat_distance_mantel.csv'}")

    # ------------------------------------------------------------------
    # Bootstrap CIs for selected phoneme pairs.
    # ------------------------------------------------------------------
    boot_pairs = block.get("bootstrap_pairs", [["e", "ɛ"], ["o", "ɔ"], ["y", "u"]])
    boot_pairs = [pair for pair in boot_pairs if len(pair) == 2 and pair[0] in phonemes and pair[1] in phonemes]
    boot_df = bootstrap_pair_distances(
        acoustic_spk=acoustic_spk,
        neural_spk_by_rep=neural_spk_by_rep,
        phoneme_pairs=boot_pairs,
        feature_labels=acoustic_cols,
        n_boot=int(block.get("bootstrap_n_resamples", 2000)),
        rng=rng,
    )
    boot_df.to_csv(tab_dir / "tab_stat_distance_bootstrap_ci.csv", index=False)
    print(f"Wrote {tab_dir / 'tab_stat_distance_bootstrap_ci.csv'}")

    # ------------------------------------------------------------------
    # Classification summaries, confusion matrices, McNemar.
    # ------------------------------------------------------------------
    preds = pd.concat([p for p in all_predictions if p is not None and not p.empty], ignore_index=True)
    preds.to_csv(tab_dir / "tab_stat_classifier_predictions.csv", index=False)

    acc, f1 = classifier_summary(preds, phonemes)
    acc.to_csv(tab_dir / "tab_stat_classifier_accuracy.csv", index=False)
    f1.to_csv(tab_dir / "tab_stat_classifier_f1.csv", index=False)
    print(f"Wrote classifier accuracy and F1 tables")

    # Plot acoustic and selected neural confusion matrices.
    plot_confusion(preds, phonemes, "acoustic_euclidean", fig_clf / "fig_confusion_acoustic_euclidean.png")
    for tag, layers in plot_layers.items():
        for layer in layers:
            rep = f"{tag}_L{int(layer):02d}"
            plot_confusion(preds, phonemes, rep, fig_clf / f"fig_confusion_{rep}.png")
    print(f"Wrote selected confusion matrices")

    mcnemar_df = mcnemar_table(preds, alpha=alpha)
    mcnemar_df.to_csv(tab_dir / "tab_stat_mcnemar.csv", index=False)
    print(f"Wrote {tab_dir / 'tab_stat_mcnemar.csv'}")

    print("[statistical_tests] done")


if __name__ == "__main__":
    main()

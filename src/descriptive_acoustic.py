"""
descriptive_acoustic.py

Robust descriptive statistics and report-ready figures for acoustic features.

This stage is intentionally descriptive: it summarises acoustic vowel-space
structure and variability without running inferential tests.

Writes
------
results/tables/
  tab_acoustic_descriptives.csv
  tab_acoustic_missingness.csv
  tab_variance_decomposition.csv
  tab_intraspeaker_variability_sd.csv

results/figures/descriptive/acoustic/
  fig_vowel_chart_panels.png
  fig_boxplot_F1_norm_clean.png
  fig_boxplot_F2_norm_clean.png
  fig_intraspeaker_variability_sd.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Ellipse, Patch


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

GROUPS = ["L1/F", "L1/M", "L2/F", "L2/M"]

GROUP_COLORS = {
    "L1/F": "#1f77b4",
    "L1/M": "#6baed6",
    "L2/F": "#d62728",
    "L2/M": "#fb6a4a",
}

ORAL_VOWEL_ORDER = ["i", "e", "ɛ", "a", "ɑ", "ɔ", "o", "u", "y", "ø", "œ", "ə"]
DEFAULT_INTRA_SUBSET = ["i", "e", "a", "ɑ", "y", "ə"]

REQUIRED_COLUMNS = {
    "token_id",
    "phoneme_base",
    "speaker_id",
    "target_word",
    "L1_status",
    "gender",
    "is_vowel",
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


def load_acoustic_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input acoustic table not found: {path}")

    df = pd.read_csv(path)
    require_columns(df, REQUIRED_COLUMNS, context=str(path))

    # Stable group used throughout the descriptive stage.
    df["group"] = df["L1_status"].astype(str) + "/" + df["gender"].astype(str)

    unknown_groups = sorted(set(df["group"].dropna()) - set(GROUPS))
    if unknown_groups:
        print(f"[warning] Unexpected speaker groups found: {unknown_groups}")

    return df


def observed_vowels(
    df: pd.DataFrame,
    vowels: list[str],
    feature_cols: list[str],
    min_tokens: int = 5,
) -> list[str]:
    """Keep vowels with enough complete acoustic observations."""
    kept: list[str] = []

    for v in vowels:
        n = df.loc[df["phoneme_base"] == v, feature_cols].dropna().shape[0]
        if n >= min_tokens:
            kept.append(v)

    return kept


# ---------------------------------------------------------------------
# Descriptive tables
# ---------------------------------------------------------------------

def describe_per_phoneme_group(
    df: pd.DataFrame,
    feature_cols: list[str],
    group_col: str = "group",
    phoneme_col: str = "phoneme_base",
) -> pd.DataFrame:
    """Compute n, mean, median, SD, IQR, and CV per phoneme and group."""
    rows: list[dict] = []

    for (ph, grp), sub in df.groupby([phoneme_col, group_col], dropna=False):
        row = {
            "phoneme": ph,
            "group": grp,
            "n_tokens": int(len(sub)),
        }

        for col in feature_cols:
            if col not in sub.columns:
                continue

            vals = pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy()

            if len(vals) == 0:
                row.update({
                    f"{col}_n": 0,
                    f"{col}_mean": np.nan,
                    f"{col}_median": np.nan,
                    f"{col}_sd": np.nan,
                    f"{col}_iqr": np.nan,
                    f"{col}_cv": np.nan,
                })
                continue

            mean = float(np.mean(vals))
            median = float(np.median(vals))
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = float(q3 - q1)
            cv = float(sd / abs(mean)) if mean != 0 and np.isfinite(sd) else np.nan

            row.update({
                f"{col}_n": int(len(vals)),
                f"{col}_mean": mean,
                f"{col}_median": median,
                f"{col}_sd": sd,
                f"{col}_iqr": iqr,
                f"{col}_cv": cv,
            })

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(["phoneme", "group"], na_position="last")
        .reset_index(drop=True)
    )


def missingness_by_phoneme_group(
    df: pd.DataFrame,
    feature_cols: list[str],
    phoneme_col: str = "phoneme_base",
    group_col: str = "group",
) -> pd.DataFrame:
    """Report missing-value proportions per phoneme and speaker group."""
    rows: list[dict] = []

    for (ph, grp), sub in df.groupby([phoneme_col, group_col], dropna=False):
        row = {
            "phoneme": ph,
            "group": grp,
            "n_tokens": int(len(sub)),
        }
        for col in feature_cols:
            if col not in sub.columns:
                continue
            n_missing = int(sub[col].isna().sum())
            row[f"{col}_missing_n"] = n_missing
            row[f"{col}_missing_prop"] = float(n_missing / len(sub)) if len(sub) else np.nan
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(["phoneme", "group"], na_position="last")
        .reset_index(drop=True)
    )


def variance_decomposition(
    df: pd.DataFrame,
    feature: str,
    phoneme_col: str = "phoneme_base",
    speaker_col: str = "speaker_id",
    word_col: str = "target_word",
) -> pd.DataFrame:
    """
    Descriptive per-phoneme variance decomposition.

    Components:
      sigma2_total    = variance over all tokens;
      sigma2_inter    = variance of speaker means;
      sigma2_intra    = mean variance of word means within speaker;
      sigma2_residual = mean within-(speaker, word) token variance.

    With unbalanced data these components are descriptive and are not forced
    to sum exactly to total variance.
    """
    rows: list[dict] = []

    if feature not in df.columns:
        return pd.DataFrame()

    for ph, sub in df.groupby(phoneme_col):
        vals = sub[[speaker_col, word_col, feature]].dropna()
        if len(vals) < 10:
            continue

        total_var = float(np.var(vals[feature].to_numpy(), ddof=1))

        speaker_means = vals.groupby(speaker_col)[feature].mean()
        sigma2_inter = (
            float(np.var(speaker_means.to_numpy(), ddof=1))
            if len(speaker_means) > 1
            else np.nan
        )

        intra_per_speaker: list[float] = []
        for _, sub_speaker in vals.groupby(speaker_col):
            word_means = sub_speaker.groupby(word_col)[feature].mean()
            if len(word_means) > 1:
                intra_per_speaker.append(float(np.var(word_means.to_numpy(), ddof=1)))

        sigma2_intra = float(np.mean(intra_per_speaker)) if intra_per_speaker else np.nan

        residual_per_cell: list[float] = []
        for _, cell in vals.groupby([speaker_col, word_col]):
            if len(cell) > 1:
                residual_per_cell.append(float(np.var(cell[feature].to_numpy(), ddof=1)))

        sigma2_residual = float(np.mean(residual_per_cell)) if residual_per_cell else np.nan

        rows.append({
            "phoneme": ph,
            "n_tokens": int(len(vals)),
            "n_speakers": int(vals[speaker_col].nunique()),
            f"{feature}_sigma2_total": total_var,
            f"{feature}_sigma2_inter_speaker": sigma2_inter,
            f"{feature}_sigma2_intra_speaker": sigma2_intra,
            f"{feature}_sigma2_residual": sigma2_residual,
        })

    return pd.DataFrame(rows).sort_values("phoneme").reset_index(drop=True)


def compute_within_speaker_sd(
    df: pd.DataFrame,
    vowels: list[str],
    features: list[str],
) -> pd.DataFrame:
    """Compute within-speaker SD for each vowel and feature."""
    rows: list[dict] = []
    sub = df[df["phoneme_base"].isin(vowels)].copy()

    for (speaker, group, ph), cell in sub.groupby(["speaker_id", "group", "phoneme_base"]):
        for feat in features:
            if feat not in cell.columns:
                continue
            vals = pd.to_numeric(cell[feat], errors="coerce").dropna().to_numpy()
            if len(vals) < 2:
                continue
            rows.append({
                "speaker_id": speaker,
                "group": group,
                "phoneme": ph,
                "feature": feat,
                "within_speaker_sd": float(np.std(vals, ddof=1)),
                "n_tokens": int(len(vals)),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------

def add_confidence_ellipse(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    edgecolor: str,
    confidence: float = 0.95,
    min_points: int = 8,
) -> None:
    """Draw a 2D normal-theory confidence ellipse for x/y observations."""
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < min_points:
        return

    cov = np.cov(x, y)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return

    # Numerical guard for near-singular covariance matrices.
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    if np.all(eigvals == 0):
        return

    # Chi-square quantile for 2 df: qchisq(.95, 2) = 5.991464547.
    chi2_scale = 5.991464547 if confidence == 0.95 else 5.991464547
    radii = np.sqrt(eigvals * chi2_scale)

    order = eigvals.argsort()[::-1]
    eigvec = eigvecs[:, order[0]]
    angle = np.degrees(np.arctan2(eigvec[1], eigvec[0]))

    ell = Ellipse(
        xy=(float(np.mean(x)), float(np.mean(y))),
        width=float(2 * radii[order[0]]),
        height=float(2 * radii[order[1]]),
        angle=float(angle),
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=1.0,
        alpha=0.45,
        zorder=2,
    )
    ax.add_patch(ell)


def plot_vowel_chart_panels(
    df: pd.DataFrame,
    vowels: list[str],
    out_path: Path,
) -> None:
    """
    Vowel chart with per-group panels, vowel centroids, and 95% ellipses.

    The x-axis is F2_norm and the y-axis is F1_norm. Both axes are inverted
    following IPA vowel chart convention.
    """
    plot_df = (
        df[df["phoneme_base"].isin(vowels)]
        .dropna(subset=["F1_norm", "F2_norm"])
        .copy()
    )

    if plot_df.empty:
        raise RuntimeError("No complete F1_norm/F2_norm observations for vowel chart")

    centroids = (
        plot_df
        .groupby(["group", "phoneme_base"], as_index=False)
        .agg(
            F1_mean=("F1_norm", "mean"),
            F2_mean=("F2_norm", "mean"),
            n=("F1_norm", "size"),
        )
    )

    x_min, x_max = plot_df["F2_norm"].quantile([0.01, 0.99])
    y_min, y_max = plot_df["F1_norm"].quantile([0.01, 0.99])
    pad_x = 0.35
    pad_y = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, grp in zip(axes, GROUPS):
        sub_tokens = plot_df[plot_df["group"] == grp]
        sub_centroids = centroids[centroids["group"] == grp]

        ax.set_title(grp)
        ax.grid(alpha=0.25)
        ax.set_xlim(x_max + pad_x, x_min - pad_x)  # inverted F2
        ax.set_ylim(y_max + pad_y, y_min - pad_y)  # inverted F1

        for ph in vowels:
            cell = sub_tokens[sub_tokens["phoneme_base"] == ph]
            if cell.empty:
                continue

            add_confidence_ellipse(
                ax=ax,
                x=cell["F2_norm"].to_numpy(dtype=float),
                y=cell["F1_norm"].to_numpy(dtype=float),
                edgecolor=GROUP_COLORS.get(grp, "black"),
                confidence=0.95,
                min_points=8,
            )

        for _, row in sub_centroids.iterrows():
            ph = str(row["phoneme_base"])
            ax.scatter(
                row["F2_mean"],
                row["F1_mean"],
                s=80,
                color=GROUP_COLORS.get(grp, "black"),
                edgecolor="black",
                linewidth=0.6,
                alpha=0.95,
                zorder=4,
            )
            ax.text(
                row["F2_mean"],
                row["F1_mean"],
                ph,
                fontsize=12,
                ha="center",
                va="center",
                color="white",
                weight="bold",
                zorder=5,
            )

        ax.set_xlabel("F2_norm")
        ax.set_ylabel("F1_norm")

    fig.suptitle(
        "Vowel space by speaker group: Lobanov-normalised F1/F2 centroids with 95% ellipses",
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_clean_boxplot(
    df: pd.DataFrame,
    feature: str,
    vowels: list[str],
    out_path: Path,
) -> None:
    """Grouped boxplot for one formant feature."""
    sub = df[df["phoneme_base"].isin(vowels)].dropna(subset=[feature]).copy()
    if sub.empty:
        raise RuntimeError(f"No data available for {feature} boxplot")

    fig, ax = plt.subplots(figsize=(14, 6.3))
    width = 0.18

    for i, ph in enumerate(vowels):
        for j, grp in enumerate(GROUPS):
            vals = sub.loc[(sub["phoneme_base"] == ph) & (sub["group"] == grp), feature].to_numpy()
            if len(vals) == 0:
                continue

            pos = i + (j - 1.5) * width
            bp = ax.boxplot(
                [vals],
                positions=[pos],
                widths=width * 0.9,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.2},
            )

            for patch in bp["boxes"]:
                patch.set_facecolor(GROUP_COLORS[grp])
                patch.set_alpha(0.75)
                patch.set_edgecolor("black")
                patch.set_linewidth(0.7)

            for item in [*bp["whiskers"], *bp["caps"]]:
                item.set_color("black")
                item.set_linewidth(0.7)

    handles = [Patch(facecolor=GROUP_COLORS[g], alpha=0.75, label=g) for g in GROUPS]
    ax.set_xticks(range(len(vowels)))
    ax.set_xticklabels(vowels, fontsize=12)
    ax.set_xlabel("Vowel")
    ax.set_ylabel(feature)
    ax.set_title(f"Distribution of Lobanov-normalised {feature.replace('_norm', '')} by vowel and group")
    ax.legend(handles=handles, loc="best", frameon=True)
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_intraspeaker_variability_sd(
    sd_df: pd.DataFrame,
    subset_vowels: list[str],
    out_path: Path,
    rng: np.random.Generator,
) -> None:
    """Plot within-speaker SD for a selected vowel subset."""
    sub_all = sd_df[sd_df["phoneme"].isin(subset_vowels)].copy()
    if sub_all.empty:
        raise RuntimeError("No within-speaker variability data available for selected subset")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    width = 0.18

    for ax, feat in zip(axes, ["F1_norm", "F2_norm"]):
        sub = sub_all[sub_all["feature"] == feat]

        for i, ph in enumerate(subset_vowels):
            for j, grp in enumerate(GROUPS):
                vals = sub.loc[(sub["phoneme"] == ph) & (sub["group"] == grp), "within_speaker_sd"].to_numpy()
                if len(vals) == 0:
                    continue

                pos = i + (j - 1.5) * width
                bp = ax.boxplot(
                    [vals],
                    positions=[pos],
                    widths=width * 0.9,
                    patch_artist=True,
                    showfliers=False,
                    medianprops={"color": "black", "linewidth": 1.2},
                )

                for patch in bp["boxes"]:
                    patch.set_facecolor(GROUP_COLORS[grp])
                    patch.set_alpha(0.75)
                    patch.set_edgecolor("black")
                    patch.set_linewidth(0.7)

                for item in [*bp["whiskers"], *bp["caps"]]:
                    item.set_color("black")
                    item.set_linewidth(0.7)

                jitter = rng.uniform(-width * 0.22, width * 0.22, size=len(vals))
                ax.scatter(
                    np.full(len(vals), pos) + jitter,
                    vals,
                    s=18,
                    color="black",
                    alpha=0.45,
                    zorder=3,
                )

        ax.set_title(f"Within-speaker variability: {feat}")
        ax.set_xlabel("Vowel")
        ax.set_xticks(range(len(subset_vowels)))
        ax.set_xticklabels(subset_vowels, fontsize=12)
        ax.grid(alpha=0.25, axis="y")

    axes[0].set_ylabel("Within-speaker SD")
    handles = [Patch(facecolor=GROUP_COLORS[g], alpha=0.75, label=g) for g in GROUPS]
    axes[1].legend(handles=handles, loc="best", frameon=True)

    fig.suptitle("Intra-speaker variability across repetitions: selected vowel subset", y=0.98)
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
    block = cfg["descriptive_acoustic"]

    in_csv = Path(block["input_features"])
    fig_dir = Path(block["figures_dir"]) / "descriptive" / "acoustic"
    tab_dir = Path(block["tables_dir"])
    ensure_dirs(fig_dir, tab_dir)

    rng = np.random.default_rng(int(block.get("random_state", 42)))

    df = load_acoustic_table(in_csv)

    all_vowels = cfg["normalise_acoustic"]["vowel_inventory"]
    oral_vowels = [v for v in ORAL_VOWEL_ORDER if v in all_vowels]
    oral_vowels = observed_vowels(df, oral_vowels, ["F1_norm", "F2_norm"], min_tokens=5)

    if not oral_vowels:
        raise RuntimeError("No observed oral vowels with complete F1_norm/F2_norm data")

    feature_cols = [c for c in ["F1_norm", "F2_norm", "F3_norm", "f0_st", "scg_hz"] if c in df.columns]

    # Tables required by the descriptive section.
    desc = describe_per_phoneme_group(df, feature_cols)
    out_desc = tab_dir / "tab_acoustic_descriptives.csv"
    desc.to_csv(out_desc, index=False)
    print(f"Wrote {out_desc} ({len(desc)} rows)")

    missing = missingness_by_phoneme_group(df, feature_cols)
    out_missing = tab_dir / "tab_acoustic_missingness.csv"
    missing.to_csv(out_missing, index=False)
    print(f"Wrote {out_missing} ({len(missing)} rows)")

    vowel_df = df[df["is_vowel"].astype(bool)].copy()
    parts: list[pd.DataFrame] = []
    for feat in ["F1_norm", "F2_norm", "F3_norm", "f0_st"]:
        if feat not in vowel_df.columns:
            continue
        d = variance_decomposition(vowel_df, feat)
        if d.empty:
            continue
        if not parts:
            parts.append(d)
        else:
            keep = ["phoneme"] + [c for c in d.columns if c.startswith(feat)]
            parts.append(d[keep])

    if parts:
        decomp = parts[0]
        for extra in parts[1:]:
            decomp = decomp.merge(extra, on="phoneme", how="outer")
    else:
        decomp = pd.DataFrame()

    out_decomp = tab_dir / "tab_variance_decomposition.csv"
    decomp.to_csv(out_decomp, index=False)
    print(f"Wrote {out_decomp} ({len(decomp)} rows)")

    sd_df = compute_within_speaker_sd(df, oral_vowels, ["F1_norm", "F2_norm"])
    out_sd = tab_dir / "tab_intraspeaker_variability_sd.csv"
    sd_df.to_csv(out_sd, index=False)
    print(f"Wrote {out_sd} ({len(sd_df)} rows)")

    subset_cfg = block.get("intraspeaker_subset_vowels", DEFAULT_INTRA_SUBSET)
    subset_vowels = [v for v in subset_cfg if v in oral_vowels]
    if len(subset_vowels) < 3:
        subset_vowels = oral_vowels[: min(6, len(oral_vowels))]

    # Figures required by the descriptive section.
    plot_vowel_chart_panels(
        df=df,
        vowels=oral_vowels,
        out_path=fig_dir / "fig_vowel_chart_panels.png",
    )
    print(f"Wrote {fig_dir / 'fig_vowel_chart_panels.png'}")

    plot_clean_boxplot(
        df=df,
        feature="F1_norm",
        vowels=oral_vowels,
        out_path=fig_dir / "fig_boxplot_F1_norm_clean.png",
    )
    print(f"Wrote {fig_dir / 'fig_boxplot_F1_norm_clean.png'}")

    plot_clean_boxplot(
        df=df,
        feature="F2_norm",
        vowels=oral_vowels,
        out_path=fig_dir / "fig_boxplot_F2_norm_clean.png",
    )
    print(f"Wrote {fig_dir / 'fig_boxplot_F2_norm_clean.png'}")

    plot_intraspeaker_variability_sd(
        sd_df=sd_df,
        subset_vowels=subset_vowels,
        out_path=fig_dir / "fig_intraspeaker_variability_sd.png",
        rng=rng,
    )
    print(f"Wrote {fig_dir / 'fig_intraspeaker_variability_sd.png'}")


if __name__ == "__main__":
    main()

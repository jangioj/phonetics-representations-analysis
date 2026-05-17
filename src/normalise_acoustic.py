"""Stage 05a: speaker-wise Lobanov normalisation of formants + log f0.

Reads `features_acoustic.csv`, applies per-speaker z-scoring to F1/F2/F3
and their 25%/75% trajectory points using ONLY vowel tokens to fit the
per-speaker mean/std, converts f0 to semitones re a fixed reference, and
writes `features_acoustic_norm.csv`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# Actual schema produced by extract_acoustics.py
FORMANT_COLS = [
    "f1_hz",
    "f2_hz",
    "f3_hz",
    "f1_25_hz",
    "f2_25_hz",
    "f3_25_hz",
    "f1_75_hz",
    "f2_75_hz",
    "f3_75_hz",
]

# Each trajectory point is normalised using the midpoint distribution
# of the corresponding formant for that speaker.
# Example: f1_25_hz and f1_75_hz use mean/std from f1_hz.
FORMANT_REFERENCE_COLS = {
    "f1_hz": "f1_hz",
    "f2_hz": "f2_hz",
    "f3_hz": "f3_hz",
    "f1_25_hz": "f1_hz",
    "f2_25_hz": "f2_hz",
    "f3_25_hz": "f3_hz",
    "f1_75_hz": "f1_hz",
    "f2_75_hz": "f2_hz",
    "f3_75_hz": "f3_hz",
}


# Optional aliases, useful if later scripts expect F1_norm-style names.
NORM_ALIASES = {
    "f1_hz_norm": "F1_norm",
    "f2_hz_norm": "F2_norm",
    "f3_hz_norm": "F3_norm",
    "f1_25_hz_norm": "F1_25_norm",
    "f2_25_hz_norm": "F2_25_norm",
    "f3_25_hz_norm": "F3_25_norm",
    "f1_75_hz_norm": "F1_75_norm",
    "f2_75_hz_norm": "F2_75_norm",
    "f3_75_hz_norm": "F3_75_norm",
}


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns: {missing}. "
            f"Available columns are: {list(df.columns)}"
        )


def lobanov_normalise(
    df: pd.DataFrame,
    vowel_inventory: list[str],
    min_vowel_tokens: int,
) -> pd.DataFrame:
    """Add *_norm columns.

    Lobanov parameters are fitted speaker-wise using vowel tokens only.
    Consonant rows keep NaN in formant normalised columns.
    """
    out = df.copy()

    require_columns(out, ["speaker_id", "phoneme_base"])

    is_vowel = out["phoneme_base"].isin(vowel_inventory)

    present_cols = [c for c in FORMANT_COLS if c in out.columns]
    if not present_cols:
        raise KeyError(
            "None of the expected formant columns were found. "
            f"Expected one of: {FORMANT_COLS}. "
            f"Available columns are: {list(out.columns)}"
        )

    for col in present_cols:
        out[f"{col}_norm"] = np.nan

    for speaker, sub in out.loc[is_vowel].groupby("speaker_id"):
        n_tokens = len(sub)

        if n_tokens < min_vowel_tokens:
            raise ValueError(
                f"Speaker {speaker} has only {n_tokens} vowel tokens "
                f"(< min_vowel_tokens_per_speaker={min_vowel_tokens}). "
                "Aborting: Lobanov mean/std would be unreliable."
            )

        speaker_vowel_mask = (out["speaker_id"] == speaker) & is_vowel

        for col in present_cols:
            ref_col = FORMANT_REFERENCE_COLS[col]

            if ref_col not in out.columns:
                continue

            # Fit Lobanov parameters on the midpoint values of the
            # corresponding formant, using vowel tokens only.
            vals = pd.to_numeric(sub[ref_col], errors="coerce").dropna()

            if len(vals) < 2:
                continue

            mu = vals.mean()
            sigma = vals.std(ddof=1)

            if sigma == 0 or np.isnan(sigma):
                continue

            # Apply those same midpoint parameters to the target column.
            # For example, f1_25_hz is normalised with mean/std from f1_hz.
            target_values = pd.to_numeric(
                out.loc[speaker_vowel_mask, col],
                errors="coerce",
            )

            out.loc[speaker_vowel_mask, f"{col}_norm"] = (
                                                                 target_values - mu
                                                         ) / sigma

    # Backward-compatible aliases for downstream scripts that expect F1_norm.
    for src, dst in NORM_ALIASES.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]

    return out


def f0_to_semitones(df: pd.DataFrame, reference_hz: float) -> pd.DataFrame:
    """Add f0_st column from f0_mean_hz.

    Uses 12 * log2(f0 / reference). Leaves NaN for missing or non-positive f0.
    """
    out = df.copy()

    require_columns(out, ["f0_mean_hz"])

    f0 = pd.to_numeric(out["f0_mean_hz"], errors="coerce")
    out["f0_st"] = np.nan

    valid = f0.notna() & (f0 > 0)
    out.loc[valid, "f0_st"] = 12.0 * np.log2(f0.loc[valid] / reference_hz)

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cfg = config["normalise_acoustic"]

    in_path = Path(cfg["input_features"])
    out_path = Path(cfg["output_features"])

    print(f"[normalise_acoustic] reading {in_path}")
    df = pd.read_csv(in_path)
    n_in = len(df)

    df = lobanov_normalise(
        df,
        vowel_inventory=cfg["vowel_inventory"],
        min_vowel_tokens=cfg["min_vowel_tokens_per_speaker"],
    )

    df = f0_to_semitones(
        df,
        reference_hz=cfg["f0_reference_hz"],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    n_vowels = df["phoneme_base"].isin(cfg["vowel_inventory"]).sum()
    n_speakers = df["speaker_id"].nunique()
    f1_norm_missing = df["f1_hz_norm"].isna().sum()

    print(f"[normalise_acoustic] wrote {out_path}")
    print(f"  rows in/out:       {n_in} -> {len(df)}")
    print(f"  speakers:          {n_speakers}")
    print(f"  vowel tokens:      {n_vowels}")
    print(
        f"  f1_hz_norm missing: {f1_norm_missing} "
        f"(expected ~= consonant count + formant failures = {n_in - n_vowels} + failures)"
    )


if __name__ == "__main__":
    main()
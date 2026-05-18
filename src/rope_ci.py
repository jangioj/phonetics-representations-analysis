"""
rope_ci.py

Confidence Intervals and ROPE classification for PDF Section 8.

Covers procedural requirements of §8.1, §8.2, §8.3 (8.3.1 + 8.3.2), §8.4
and questions Q11, Q12, Q13.

Design notes
------------
- §8.1 Acoustic CIs. One pooled MixedLM per response (F1_norm, F2_norm) with
  formula:  y ~ 1 + L2 + Male + C(phoneme_base) + L2:C(phoneme_base) + C(vowel_height)
            + (1 | speaker_id)
  Per-vowel L1->L2 contrast = beta(L2) + beta(L2:phoneme_base==v) for non-baseline
  vowels, beta(L2) for the baseline vowel.
  Significance: Wald p-value of the per-vowel contrast computed from the
  variance/covariance matrix of the fixed effects.
  CI: parametric bootstrap, speaker-level resampling, B = 2000.
  We use bootstrap CIs in place of profile likelihood because statsmodels
  MixedLM does not expose profile likelihood; speaker-level bootstrap is the
  same protocol prescribed by §8.2 and PDF §10.2 and is therefore consistent.

- §8.1 The L1 x Gender interaction is *not* included in the Stage 9 acoustic
  model. PDF §8.1 says "L1 x Gender interaction contrast, if retained".
  Stage 8 Q9 reported no significant interaction in any representation, so the
  interaction is not retained.

- §8.2 Neural CIs. For each (model, layer) in {whisper, xlsr} x {4, 12, 20}
  and for each vowel, the L1/L2 cosine distance between speaker-level centroids
  is bootstrapped with speaker-level resampling, B = 2000. Cosine distance is
  computed in the raw embedding space (no PCA), consistent with the PDF wording
  "in Whisper and XLS-R space".

- §8.3.1 Acoustic ROPE. The PDF default is [-20, +20] Hz on F1. The acoustic
  model is fit in Lobanov units, so the ROPE is converted to Lobanov units
  using the mean per-speaker SD of F1 (and F2). The converted ROPE is reported
  alongside the original Hz value in tab_rope_acoustic_scale.csv.
  For F2 we adopt [-40, +40] Hz scaled proportionally to typical F2 (~1500 Hz),
  preserving the same ~2.5% JND ratio as the F1 ROPE.

- §8.3.2 Neural ROPE. delta_0 is the mean intra-speaker cosine distance: for
  every (speaker, phoneme), the average cosine distance between distinct pairs
  of tokens of that phoneme produced by that speaker; then averaged over all
  (speaker, phoneme) cells. One delta_0 per (model, layer).

- §8.4 Classification: equivalent / non-equivalent / indeterminate per the
  PDF definitions (CI inside / outside / overlapping ROPE).

- Forest plots. Main neural forest plot uses the layers chosen as
  representative based on Stage 5/6 descriptive results (Whisper L12, XLS-R L04).
  A supplementary forest plot includes all six (model, layer) combinations.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats

import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM


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
REQUIRED_RAW_ACOUSTIC_COLUMNS = {"token_id", "speaker_id", "f1_hz", "f2_hz"}

# Map normalised response column -> raw Hz column in features_acoustic.csv.
RAW_HZ_COLUMN_FOR = {"F1_norm": "f1_hz", "F2_norm": "f2_hz"}


# ---------------------------------------------------------------------
# Generic helpers (mirrored from lme_models.py)
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


def load_acoustic_norm(path: Path, response_cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing normalised acoustic table: {path}")
    df = pd.read_csv(path)
    require_columns(df, REQUIRED_ACOUSTIC_COLUMNS | set(response_cols), context=str(path))
    if df["token_id"].duplicated().any():
        dup = int(df["token_id"].duplicated().sum())
        raise ValueError(f"token_id must be unique in {path}; found {dup} duplicates")
    return df


def load_acoustic_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing raw acoustic table: {path}")
    df = pd.read_csv(path)
    require_columns(df, REQUIRED_RAW_ACOUSTIC_COLUMNS, context=str(path))
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


def select_vowel_tokens(df: pd.DataFrame, configured_vowels: list[str],
                        response_cols: list[str], min_tokens: int = 5) -> tuple[pd.DataFrame, list[str]]:
    """Filter to oral vowels with >= min_tokens complete responses (mirrors Stage 8)."""
    phonemes = [v for v in ORAL_VOWEL_ORDER if v in configured_vowels]
    observed = []
    for ph in phonemes:
        n = df.loc[df["phoneme_base"] == ph, response_cols].dropna().shape[0]
        if n >= min_tokens:
            observed.append(ph)
    if len(observed) < 3:
        raise RuntimeError(f"Too few oral vowels with complete acoustic data: {observed}")
    sub = df[df["phoneme_base"].isin(observed)].copy()
    return sub, observed


def add_design_columns(df: pd.DataFrame, vowel_height: dict[str, str]) -> pd.DataFrame:
    """Binary L2/Male indicators + vowel_height categorical."""
    df = df.copy()
    df["L2"] = (df["L1_status"].astype(str).str.upper() == "L2").astype(int)
    df["Male"] = (df["gender"].astype(str).str.upper() == "M").astype(int)
    df["vowel_height"] = df["phoneme_base"].map(vowel_height).astype("category")
    return df


# ---------------------------------------------------------------------
# Acoustic ROPE: convert Hz units to Lobanov units
# ---------------------------------------------------------------------

def compute_mean_per_speaker_sd(raw: pd.DataFrame, meta: pd.DataFrame,
                                column: str, vowel_set: list[str]) -> float:
    """Mean over speakers of SD(F_j) computed on vowel tokens only.

    This mirrors the Lobanov denominator (PDF §10.1, Eq. 5). We then use this
    average as the conversion factor between Lobanov units and Hz when
    expressing the acoustic ROPE.
    """
    # Use only the phoneme_base label from meta. raw may already carry a
    # phoneme_base column; we drop it before merging to avoid suffix collisions.
    raw_use = raw[["token_id", "speaker_id", column]].copy()
    df = raw_use.merge(
        meta[["token_id", "phoneme_base"]],
        on="token_id",
        how="inner",
    )
    df = df[df["phoneme_base"].isin(vowel_set)].copy()
    df = df.dropna(subset=[column])
    sds = df.groupby("speaker_id")[column].std(ddof=1).dropna()
    if len(sds) == 0:
        raise RuntimeError(f"No speaker SDs computable for {column}")
    return float(sds.mean())


# ---------------------------------------------------------------------
# Acoustic LME for §8.1
# ---------------------------------------------------------------------

def _build_acoustic_design(sub: pd.DataFrame, response: str, baseline_vowel: str
                           ) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Return (df_used, y, fixed_effect_names).

    The design matrix is built explicitly so that we can construct per-vowel
    contrast vectors as linear combinations of the same coefficients across
    bootstrap replicates.
    """
    df = sub.dropna(subset=[response]).copy()

    # Order: baseline first, then sorted others. Used for predictable column
    # naming in the model summary.
    vowel_order = [baseline_vowel] + sorted(
        v for v in df["phoneme_base"].unique() if v != baseline_vowel
    )
    df["phoneme_base"] = pd.Categorical(df["phoneme_base"], categories=vowel_order, ordered=False)

    height_order = ["high", "mid", "low"]
    present_heights = [h for h in height_order if h in df["vowel_height"].cat.categories]
    df["vowel_height"] = pd.Categorical(df["vowel_height"], categories=present_heights, ordered=False)

    return df, df[response].to_numpy(), vowel_order


def _per_vowel_contrast_names(fe_names: list[str], vowels: list[str], baseline: str
                              ) -> dict[str, np.ndarray]:
    """Construct contrast vectors c such that c.T @ beta = L1->L2 effect on vowel v.

    Coding: with the formula
        y ~ 1 + L2 + Male + C(phoneme_base) + L2:C(phoneme_base) + C(vowel_height)
    statsmodels uses baseline-coded dummies. For the baseline vowel the L2
    effect is beta(L2). For other vowels it is beta(L2) + beta(L2:phoneme[v]).
    """
    name_to_idx = {n: i for i, n in enumerate(fe_names)}
    if "L2" not in name_to_idx:
        raise KeyError(f"Fixed-effects vector lacks 'L2'; names={fe_names}")

    contrasts: dict[str, np.ndarray] = {}
    p = len(fe_names)
    for v in vowels:
        c = np.zeros(p, dtype=float)
        c[name_to_idx["L2"]] = 1.0
        if v != baseline:
            interaction_key = f"L2:C(phoneme_base)[T.{v}]"
            if interaction_key not in name_to_idx:
                interaction_key = f"C(phoneme_base)[T.{v}]:L2"
            if interaction_key not in name_to_idx:
                raise KeyError(f"Cannot locate interaction column for vowel {v} in {fe_names}")
            c[name_to_idx[interaction_key]] = 1.0
        contrasts[v] = c
    return contrasts


def _random_var_safe(result) -> float:
    """Sum of random-effect variances; NaN on failure. Mirrors Stage 8 helper."""
    try:
        cov_re = np.asarray(result.cov_re)
        if cov_re.ndim == 0:
            return float(cov_re)
        return float(np.trace(cov_re))
    except Exception:
        return float("nan")


def _fit_with_boundary_retry(model, reml: bool):
    """Fit MixedLM with lbfgs; retry powell if random variance is on the boundary.

    Mirrors the Stage 8 strategy: lbfgs is fast but parks var_speaker ~= 0 on
    the boundary, producing NaN standard errors. powell is slower but
    typically escapes the boundary.
    """
    primary = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            primary = model.fit(method="lbfgs", reml=reml)
    except Exception:
        primary = None

    # Boundary check on the primary fit.
    needs_retry = primary is None
    if primary is not None:
        var_r = _random_var_safe(primary)
        try:
            var_e = float(primary.scale)
        except Exception:
            var_e = float("nan")
        total = var_r + var_e if np.isfinite(var_r) and np.isfinite(var_e) else float("nan")
        if not np.isfinite(var_r) or var_r <= 1e-10:
            needs_retry = True
        elif np.isfinite(total) and total > 0 and (var_r / total) < 0.005:
            needs_retry = True
        else:
            # SE NaNs in fixed effects also indicate a degenerate fit.
            try:
                if np.any(~np.isfinite(np.asarray(primary.bse_fe))):
                    needs_retry = True
            except Exception:
                needs_retry = True

    if not needs_retry:
        return primary

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            retried = model.fit(method="powell", reml=reml)
    except Exception:
        retried = None

    if retried is None:
        return primary  # may itself be None

    if primary is None:
        return retried

    # Keep whichever has the larger random variance (boundary fits underestimate it).
    var_primary = _random_var_safe(primary)
    var_retried = _random_var_safe(retried)
    if np.isfinite(var_retried) and (
        not np.isfinite(var_primary) or var_retried > var_primary
    ):
        return retried
    return primary


def fit_acoustic_lme(sub: pd.DataFrame, response: str, baseline_vowel: str
                     ) -> dict:
    """Fit the pooled acoustic LME and return point estimates + Wald inference.

    Returns a dict with:
        result: statsmodels MixedLMResults
        vowels: ordered list including baseline first
        fe_names: list of fixed-effect parameter names
        contrasts: dict[vowel -> contrast vector]
        per_vowel: list of dicts with keys
            vowel, estimate, se, z, p_value, ci_low_wald, ci_high_wald
    """
    df, y, vowels = _build_acoustic_design(sub, response, baseline_vowel)

    formula = (f"{response} ~ 1 + L2 + Male + C(phoneme_base) + L2:C(phoneme_base) "
               f"+ C(vowel_height)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MixedLM.from_formula(formula, groups="speaker_id", data=df)
        result = _fit_with_boundary_retry(model, reml=False)

    if result is None:
        raise RuntimeError(f"Could not fit acoustic LME for {response}")

    fe_names = list(result.fe_params.index)
    beta = result.fe_params.to_numpy()
    cov = result.cov_params().loc[fe_names, fe_names].to_numpy()

    contrasts = _per_vowel_contrast_names(fe_names, vowels, baseline_vowel)

    per_vowel: list[dict] = []
    alpha = 0.05
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    for v in vowels:
        c = contrasts[v]
        est = float(c @ beta)
        var = float(c @ cov @ c)
        se = float(np.sqrt(var)) if var > 0 else float("nan")
        z = est / se if (se > 0 and np.isfinite(se)) else float("nan")
        p = float(2.0 * (1.0 - stats.norm.cdf(abs(z)))) if np.isfinite(z) else float("nan")
        per_vowel.append({
            "vowel": v,
            "estimate": est,
            "se_wald": se,
            "z_wald": z,
            "p_value_wald": p,
            "ci_low_wald": est - z_crit * se if np.isfinite(se) else float("nan"),
            "ci_high_wald": est + z_crit * se if np.isfinite(se) else float("nan"),
        })

    return {
        "result": result,
        "vowels": vowels,
        "fe_names": fe_names,
        "contrasts": contrasts,
        "per_vowel": per_vowel,
    }


def bootstrap_acoustic(sub: pd.DataFrame, response: str, baseline_vowel: str,
                       contrasts: dict[str, np.ndarray], vowels: list[str],
                       fe_names: list[str], B: int, rng: np.random.Generator
                       ) -> dict[str, np.ndarray]:
    """Speaker-level parametric bootstrap of the per-vowel L1->L2 contrast.

    Returns dict[vowel -> 1D array of length <= B of bootstrap point estimates].
    Failed replicates are silently dropped (rare).
    """
    speakers = np.array(sorted(sub["speaker_id"].unique()))
    n_sp = len(speakers)
    boots: dict[str, list[float]] = {v: [] for v in vowels}
    n_fail = 0

    height_order = ["high", "mid", "low"]
    vowel_order = [baseline_vowel] + sorted(v for v in vowels if v != baseline_vowel)
    sub = sub.copy()

    formula = (f"{response} ~ 1 + L2 + Male + C(phoneme_base) + L2:C(phoneme_base) "
               f"+ C(vowel_height)")

    for b in range(B):
        idx = rng.integers(0, n_sp, size=n_sp)
        chosen = speakers[idx]
        # Re-label resampled speakers to keep groups distinct in MixedLM.
        parts = []
        for new_id, sp in enumerate(chosen):
            block = sub[sub["speaker_id"] == sp].copy()
            block["speaker_id"] = f"b{new_id}"
            parts.append(block)
        boot = pd.concat(parts, ignore_index=True).dropna(subset=[response])
        # Re-impose categorical orderings for stable column names.
        present_v = [v for v in vowel_order if v in boot["phoneme_base"].unique()]
        boot["phoneme_base"] = pd.Categorical(boot["phoneme_base"],
                                              categories=present_v, ordered=False)
        present_h = [h for h in height_order if h in boot["vowel_height"].cat.categories]
        boot["vowel_height"] = pd.Categorical(boot["vowel_height"],
                                              categories=present_h, ordered=False)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mdl = MixedLM.from_formula(formula, groups="speaker_id", data=boot)
                res = mdl.fit(method="lbfgs", reml=False)
        except Exception:
            n_fail += 1
            continue

        b_names = list(res.fe_params.index)
        # If the resampled dataset misses a vowel entirely, that vowel's
        # interaction column will be absent. Skip the affected vowels for
        # this replicate.
        beta = res.fe_params.to_numpy()
        name_to_idx = {n: i for i, n in enumerate(b_names)}
        for v in vowels:
            c_full = contrasts[v]
            est = 0.0
            ok = True
            for global_idx, val in enumerate(c_full):
                if val == 0.0:
                    continue
                target = fe_names[global_idx]
                if target not in name_to_idx:
                    ok = False
                    break
                est += val * float(beta[name_to_idx[target]])
            if ok:
                boots[v].append(est)

    return {v: np.asarray(arr, dtype=float) for v, arr in boots.items()}, n_fail


# ---------------------------------------------------------------------
# Neural CIs and delta_0 for §8.2 + §8.3.2
# ---------------------------------------------------------------------

def speaker_centroids(X: np.ndarray, speaker_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-speaker mean embeddings + the corresponding speaker labels."""
    unique = np.array(sorted(np.unique(speaker_ids)))
    out = np.zeros((len(unique), X.shape[1]), dtype=np.float64)
    for i, sp in enumerate(unique):
        out[i] = X[speaker_ids == sp].mean(axis=0)
    return out, unique


def _cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return float("nan")
    return float(1.0 - (u @ v) / (nu * nv))


def neural_l1_l2_distance(X: np.ndarray, meta: pd.DataFrame, vowel: str) -> float:
    """Cosine distance between L1 and L2 centroids for one vowel."""
    mask_v = (meta["phoneme_base"] == vowel).to_numpy()
    if not mask_v.any():
        return float("nan")
    Xv = X[mask_v]
    L1_mask = (meta.loc[mask_v, "L1_status"].astype(str).str.upper() == "L1").to_numpy()
    L2_mask = ~L1_mask
    if L1_mask.sum() == 0 or L2_mask.sum() == 0:
        return float("nan")
    c_L1 = Xv[L1_mask].mean(axis=0)
    c_L2 = Xv[L2_mask].mean(axis=0)
    return _cosine_distance(c_L1, c_L2)


def bootstrap_neural_l1_l2(X: np.ndarray, meta: pd.DataFrame, vowel: str,
                           B: int, rng: np.random.Generator
                           ) -> np.ndarray:
    """Speaker-level bootstrap of the L1<->L2 cosine distance for a single vowel.

    Resamples speakers within L1 and within L2 separately to guarantee both
    centroids exist on each replicate (PDF §10.2 spirit applied to a two-group
    quantity).
    """
    mask_v = (meta["phoneme_base"] == vowel).to_numpy()
    if not mask_v.any():
        return np.array([], dtype=float)
    Xv = X[mask_v]
    sub_meta = meta.loc[mask_v].reset_index(drop=True)
    sp = sub_meta["speaker_id"].to_numpy()
    L1_sp = np.array(sorted(sub_meta.loc[sub_meta["L1_status"].astype(str).str.upper() == "L1",
                                         "speaker_id"].unique()))
    L2_sp = np.array(sorted(sub_meta.loc[sub_meta["L1_status"].astype(str).str.upper() == "L2",
                                         "speaker_id"].unique()))
    if len(L1_sp) == 0 or len(L2_sp) == 0:
        return np.array([], dtype=float)

    # Precompute per-speaker mean embeddings to make resampling fast.
    speaker_means: dict[str, np.ndarray] = {}
    for s in np.concatenate([L1_sp, L2_sp]):
        speaker_means[s] = Xv[sp == s].mean(axis=0)

    out = np.empty(B, dtype=float)
    for b in range(B):
        s1 = rng.choice(L1_sp, size=len(L1_sp), replace=True)
        s2 = rng.choice(L2_sp, size=len(L2_sp), replace=True)
        c_L1 = np.mean([speaker_means[s] for s in s1], axis=0)
        c_L2 = np.mean([speaker_means[s] for s in s2], axis=0)
        out[b] = _cosine_distance(c_L1, c_L2)
    return out


def intra_speaker_delta0(X: np.ndarray, meta: pd.DataFrame, vowels: list[str]
                         ) -> tuple[float, int]:
    """Mean intra-speaker cosine distance over (speaker, phoneme) cells.

    For each (speaker, phoneme) with >= 2 tokens, compute the mean cosine
    distance among all distinct pairs of tokens; then average over cells.
    Returns (delta0, n_cells_used).
    """
    distances = []
    norms = np.linalg.norm(X, axis=1)
    for v in vowels:
        mask_v = (meta["phoneme_base"] == v).to_numpy()
        if not mask_v.any():
            continue
        Xv = X[mask_v]
        nv = norms[mask_v]
        sp = meta.loc[mask_v, "speaker_id"].to_numpy()
        for s in np.unique(sp):
            mask_s = (sp == s)
            n = int(mask_s.sum())
            if n < 2:
                continue
            U = Xv[mask_s]
            u_norms = nv[mask_s]
            # Pairwise cosine similarity = U @ U.T / (norms outer).
            denom = np.outer(u_norms, u_norms)
            denom[denom == 0.0] = np.nan
            sims = (U @ U.T) / denom
            iu = np.triu_indices(n, k=1)
            cell_dists = 1.0 - sims[iu]
            cell_dists = cell_dists[np.isfinite(cell_dists)]
            if len(cell_dists) > 0:
                distances.append(float(np.mean(cell_dists)))
    if not distances:
        return float("nan"), 0
    return float(np.mean(distances)), len(distances)


# ---------------------------------------------------------------------
# ROPE classification
# ---------------------------------------------------------------------

def classify_rope(ci_low: float, ci_high: float, rope_low: float, rope_high: float
                  ) -> str:
    if not (np.isfinite(ci_low) and np.isfinite(ci_high)):
        return "undefined"
    inside = (ci_low >= rope_low) and (ci_high <= rope_high)
    outside = (ci_high < rope_low) or (ci_low > rope_high)
    if inside:
        return "equivalent"
    if outside:
        return "non_equivalent"
    return "indeterminate"


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def _forest_axes(ax, rows: list[dict], title: str, xlabel: str,
                 rope_low: float | None, rope_high: float | None,
                 group_key: str | None = None) -> None:
    """Generic forest plot: one row per record, error bars from ci_low/ci_high."""
    rows = sorted(rows, key=lambda r: (r.get(group_key, ""), r["label"]))
    y = np.arange(len(rows))
    est = np.array([r["estimate"] for r in rows], dtype=float)
    lo = np.array([r["ci_low"] for r in rows], dtype=float)
    hi = np.array([r["ci_high"] for r in rows], dtype=float)
    err_lo = np.clip(est - lo, a_min=0.0, a_max=None)
    err_hi = np.clip(hi - est, a_min=0.0, a_max=None)

    ax.errorbar(est, y, xerr=[err_lo, err_hi], fmt="o", color="black",
                ecolor="gray", capsize=3, lw=1)
    ax.axvline(0, color="red", lw=0.6, ls="--")
    if rope_low is not None and rope_high is not None:
        ax.axvspan(rope_low, rope_high, color="lightblue", alpha=0.25, zorder=0,
                   label=f"ROPE [{rope_low:.3g}, {rope_high:.3g}]")
        ax.legend(loc="lower right", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", lw=0.3, alpha=0.5)


def plot_acoustic_forest(per_vowel_rows: list[dict], response: str,
                          rope_low: float, rope_high: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(per_vowel_rows) + 1.5)))
    title = f"L1→L2 contrast in {response} (95% bootstrap CI, B=2000)"
    _forest_axes(ax, per_vowel_rows, title=title,
                 xlabel=f"{response} (Lobanov units)",
                 rope_low=rope_low, rope_high=rope_high)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_neural_forest(rows: list[dict], out_path: Path,
                       title: str, delta0_per_block: dict[tuple[str, int], float]
                       ) -> None:
    fig, ax = plt.subplots(figsize=(8, max(3, 0.30 * len(rows) + 1.5)))
    rows = sorted(rows, key=lambda r: (r["model"], r["layer"], r["vowel"]))
    y = np.arange(len(rows))
    est = np.array([r["estimate"] for r in rows], dtype=float)
    lo = np.array([r["ci_low"] for r in rows], dtype=float)
    hi = np.array([r["ci_high"] for r in rows], dtype=float)
    err_lo = np.clip(est - lo, a_min=0.0, a_max=None)
    err_hi = np.clip(hi - est, a_min=0.0, a_max=None)
    blocks = sorted({(r["model"], r["layer"]) for r in rows})
    cmap = plt.get_cmap("tab10")
    color_of = {b: cmap(i % 10) for i, b in enumerate(blocks)}
    point_colors = [color_of[(r["model"], r["layer"])] for r in rows]

    for i in range(len(rows)):
        ax.errorbar(est[i], y[i], xerr=[[err_lo[i]], [err_hi[i]]], fmt="o",
                    color=point_colors[i], ecolor="gray", capsize=3, lw=1)

    # Draw a delta_0 vertical line per (model, layer) block; many blocks ->
    # only show the smallest and largest delta_0 to avoid clutter.
    deltas = sorted(delta0_per_block.items(), key=lambda kv: kv[1])
    if deltas:
        for (b, d) in (deltas[0], deltas[-1]):
            ax.axvline(d, color=color_of[b], lw=0.6, ls=":", alpha=0.7,
                       label=f"δ₀({b[0]} L{b[1]:02d})={d:.4f}")
    ax.axvline(0, color="red", lw=0.6, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['model']} L{r['layer']:02d}  /{r['vowel']}/" for r in rows],
                       fontsize=7)
    ax.set_xlabel("Cosine distance between L1 and L2 centroids")
    ax.set_title(title)
    ax.grid(axis="x", lw=0.3, alpha=0.5)
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    block = cfg["rope_ci"]

    rng = np.random.default_rng(int(block.get("random_state", 42)))
    B = int(block["bootstrap_n_resamples"])
    alpha = float(block["alpha"])
    response_cols = block["acoustic_response_cols"]
    rope_hz = block["rope_acoustic_hz"]

    tab_dir = Path(block["tables_dir"])
    fig_dir = Path(block["figures_dir"]) / "rope_ci"
    ensure_dirs(tab_dir, fig_dir)

    # ---- Load data --------------------------------------------------
    norm = load_acoustic_norm(Path(block["input_acoustic"]), response_cols)
    raw = load_acoustic_raw(Path(block["input_features_acoustic_raw"]))

    sub, observed_vowels = select_vowel_tokens(
        norm, block["oral_vowels"], response_cols, min_tokens=5,
    )
    sub = add_design_columns(sub, block["vowel_height"])
    baseline_vowel = observed_vowels[0]
    print(f"[rope_ci] vowel set: {observed_vowels}")
    print(f"[rope_ci] baseline vowel: {baseline_vowel}")
    print(f"[rope_ci] acoustic LME data: {len(sub)} tokens, "
          f"{sub['speaker_id'].nunique()} speakers")

    # ---- Convert acoustic ROPE Hz -> Lobanov units ------------------
    scale_rows = []
    rope_lobanov: dict[str, tuple[float, float]] = {}
    for response in response_cols:
        if response not in RAW_HZ_COLUMN_FOR:
            raise KeyError(f"No raw Hz column mapping for response {response}")
        raw_col = RAW_HZ_COLUMN_FOR[response]
        mean_sd = compute_mean_per_speaker_sd(raw, norm, column=raw_col,
                                              vowel_set=observed_vowels)
        lo_hz, hi_hz = rope_hz[response]
        lo_lo = lo_hz / mean_sd
        hi_lo = hi_hz / mean_sd
        rope_lobanov[response] = (lo_lo, hi_lo)
        scale_rows.append({
            "response": response,
            "mean_per_speaker_sd_hz": mean_sd,
            "rope_low_hz": lo_hz,
            "rope_high_hz": hi_hz,
            "rope_low_lobanov": lo_lo,
            "rope_high_lobanov": hi_lo,
        })
    pd.DataFrame(scale_rows).to_csv(tab_dir / "tab_rope_acoustic_scale.csv", index=False)

    # ---- §8.1 Acoustic CIs ------------------------------------------
    acoustic_rows: list[dict] = []
    acoustic_forest_rows: dict[str, list[dict]] = {r: [] for r in response_cols}
    for response in response_cols:
        print(f"[rope_ci] acoustic LME bootstrap: {response}")
        fit = fit_acoustic_lme(sub, response, baseline_vowel)
        per_vowel = fit["per_vowel"]
        contrasts = fit["contrasts"]
        vowels = fit["vowels"]
        fe_names = fit["fe_names"]

        boots, n_fail = bootstrap_acoustic(sub, response, baseline_vowel,
                                           contrasts, vowels, fe_names, B, rng)
        print(f"[rope_ci]   bootstrap replicates failed: {n_fail}/{B}")

        rope_lo, rope_hi = rope_lobanov[response]
        lo_hz, hi_hz = rope_hz[response]
        for rec in per_vowel:
            v = rec["vowel"]
            samples = boots.get(v, np.array([]))
            if len(samples) >= 50:
                ci_low = float(np.quantile(samples, alpha / 2.0))
                ci_high = float(np.quantile(samples, 1.0 - alpha / 2.0))
                n_used = int(len(samples))
            else:
                ci_low, ci_high = rec["ci_low_wald"], rec["ci_high_wald"]
                n_used = int(len(samples))

            cls = classify_rope(ci_low, ci_high, rope_lo, rope_hi)

            row = {
                "response": response,
                "vowel": v,
                "contrast_direction": "L2_minus_L1",
                "estimate_lobanov": rec["estimate"],
                "se_wald_lobanov": rec["se_wald"],
                "p_value_wald": rec["p_value_wald"],
                "ci_low_bootstrap_lobanov": ci_low,
                "ci_high_bootstrap_lobanov": ci_high,
                "n_bootstrap_used": n_used,
                "rope_low_lobanov": rope_lo,
                "rope_high_lobanov": rope_hi,
                "rope_low_hz": lo_hz,
                "rope_high_hz": hi_hz,
                "rope_classification": cls,
            }
            acoustic_rows.append(row)
            acoustic_forest_rows[response].append({
                "label": f"/{v}/",
                "estimate": rec["estimate"],
                "ci_low": ci_low,
                "ci_high": ci_high,
            })

    acoustic_df = pd.DataFrame(acoustic_rows)
    acoustic_df.to_csv(tab_dir / "tab_rope_ci_acoustic.csv", index=False)

    # Acoustic forest plots, one per response.
    for response in response_cols:
        rope_lo, rope_hi = rope_lobanov[response]
        out_name = f"fig_rope_acoustic_forest_{response.replace('_norm', '')}.png"
        plot_acoustic_forest(acoustic_forest_rows[response], response,
                             rope_lo, rope_hi, fig_dir / out_name)

    # ---- §8.2 + §8.3.2 Neural CIs + delta_0 -------------------------
    pairs = model_layer_pairs(cfg, block)
    delta0_rows = []
    neural_rows: list[dict] = []
    delta0_by_block: dict[tuple[str, int], float] = {}
    rep_layers = block["representative_layers"]

    interim = Path(block["interim_dir"])
    # Build a metadata table indexed by token_id for all vowel tokens we use.
    vowel_meta = sub[["token_id", "phoneme_base", "speaker_id",
                      "L1_status", "gender"]].set_index("token_id")

    for tag, prefix, layer in pairs:
        npz = interim / f"{prefix}_L{layer:02d}.npz"
        X, ids = load_raw_npz(npz)
        # Keep only embeddings whose token_id is among the vowel subset.
        keep_mask = pd.Index(ids).isin(vowel_meta.index)
        if not keep_mask.any():
            raise RuntimeError(f"No vowel-token embeddings found in {npz}")
        Xv = X[keep_mask]
        kept_ids = pd.Index(ids)[keep_mask]
        meta_v = vowel_meta.loc[kept_ids].reset_index()
        # Note: meta_v is now the metadata aligned with Xv, restricted to vowel
        # tokens of the acoustic subset.
        print(f"[rope_ci] {tag}_L{layer:02d}: {len(meta_v)} vowel tokens, "
              f"{meta_v['speaker_id'].nunique()} speakers")

        delta0, n_cells = intra_speaker_delta0(Xv, meta_v, observed_vowels)
        delta0_by_block[(tag, layer)] = delta0
        delta0_rows.append({
            "model": tag, "layer": layer,
            "delta0_neural": delta0,
            "n_speaker_phoneme_cells": n_cells,
            "rope_low_neural": 0.0,
            "rope_high_neural": delta0,
        })

        for v in observed_vowels:
            est = neural_l1_l2_distance(Xv, meta_v, v)
            samples = bootstrap_neural_l1_l2(Xv, meta_v, v, B=B, rng=rng)
            samples_ok = samples[np.isfinite(samples)]
            if len(samples_ok) >= 50:
                ci_low = float(np.quantile(samples_ok, alpha / 2.0))
                ci_high = float(np.quantile(samples_ok, 1.0 - alpha / 2.0))
            else:
                ci_low, ci_high = float("nan"), float("nan")
            cls = classify_rope(ci_low, ci_high, 0.0, delta0) if np.isfinite(delta0) else "undefined"
            neural_rows.append({
                "model": tag,
                "layer": layer,
                "vowel": v,
                "estimate_cosine_distance": est,
                "ci_low_bootstrap": ci_low,
                "ci_high_bootstrap": ci_high,
                "n_bootstrap_used": int(len(samples_ok)),
                "delta0_neural": delta0,
                "rope_low_neural": 0.0,
                "rope_high_neural": delta0,
                "rope_classification": cls,
            })

    pd.DataFrame(delta0_rows).to_csv(tab_dir / "tab_rope_delta0.csv", index=False)
    neural_df = pd.DataFrame(neural_rows)
    neural_df.to_csv(tab_dir / "tab_rope_ci_neural.csv", index=False)

    # ---- Neural forest plots ----------------------------------------
    rep_rows = []
    for r in neural_rows:
        rep_layer = int(rep_layers.get(r["model"], -1))
        if r["layer"] == rep_layer:
            rep_rows.append({
                "model": r["model"], "layer": r["layer"], "vowel": r["vowel"],
                "estimate": r["estimate_cosine_distance"],
                "ci_low": r["ci_low_bootstrap"], "ci_high": r["ci_high_bootstrap"],
            })
    plot_neural_forest(
        rep_rows,
        fig_dir / "fig_rope_neural_forest.png",
        title="L1 vs L2 centroid cosine distance (representative layers; 95% bootstrap CI)",
        delta0_per_block={k: v for k, v in delta0_by_block.items()
                          if k[1] == int(rep_layers.get(k[0], -1))},
    )

    all_rows = [{
        "model": r["model"], "layer": r["layer"], "vowel": r["vowel"],
        "estimate": r["estimate_cosine_distance"],
        "ci_low": r["ci_low_bootstrap"], "ci_high": r["ci_high_bootstrap"],
    } for r in neural_rows]
    plot_neural_forest(
        all_rows,
        fig_dir / "fig_rope_neural_forest_all_layers.png",
        title="L1 vs L2 centroid cosine distance (all layers; 95% bootstrap CI)",
        delta0_per_block=delta0_by_block,
    )

    # ---- §8.4 Consolidated summary ----------------------------------
    summary_rows: list[dict] = []
    for r in acoustic_rows:
        summary_rows.append({
            "representation": f"acoustic_{r['response'].replace('_norm', '')}",
            "vowel": r["vowel"],
            "estimate": r["estimate_lobanov"],
            "ci_low": r["ci_low_bootstrap_lobanov"],
            "ci_high": r["ci_high_bootstrap_lobanov"],
            "rope_low": r["rope_low_lobanov"],
            "rope_high": r["rope_high_lobanov"],
            "rope_classification": r["rope_classification"],
            "p_value_wald": r["p_value_wald"],
            "scale": "lobanov",
        })
    for r in neural_rows:
        summary_rows.append({
            "representation": f"{r['model']}_L{r['layer']:02d}",
            "vowel": r["vowel"],
            "estimate": r["estimate_cosine_distance"],
            "ci_low": r["ci_low_bootstrap"],
            "ci_high": r["ci_high_bootstrap"],
            "rope_low": r["rope_low_neural"],
            "rope_high": r["rope_high_neural"],
            "rope_classification": r["rope_classification"],
            "p_value_wald": float("nan"),
            "scale": "cosine_distance",
        })
    pd.DataFrame(summary_rows).to_csv(tab_dir / "tab_rope_summary.csv", index=False)

    print("[rope_ci] done")


if __name__ == "__main__":
    main()

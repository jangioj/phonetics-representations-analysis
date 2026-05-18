"""
lme_models.py

Linear Mixed-Effects Models for PDF Section 7.

Covers questions Q8 (ICC of /a/ in acoustic F1 vs Whisper PC1),
Q9 (L1 x Gender interaction in acoustic vs neural), and
Q10 (marginal R^2 for the L1/L2 fixed effect across representations).

Design notes
------------
- Acoustic LME is fit on token-level F1_norm and F2_norm with random
  intercepts for speaker_id. Token-level data is appropriate here because
  the random intercept absorbs speaker-level dependence.
- Neural LME projects each (model, layer) representation to the first
  d = 5 principal components fitted on the vowel-token subset, then fits
  one LME per PC dimension.
- Model selection follows PDF Section 7.3: null, main-effects, full,
  extended (adds vowel_height), random-slope. ML estimation is used so
  that likelihood-ratio tests are valid (PDF box in Section 7.3).
- The random-slope model with L2 by speaker is structurally
  non-identifiable because L2 is between-speaker. We attempt the fit,
  catch convergence problems, and record the outcome.
- R^2 marginal and conditional follow Nakagawa & Schielzeth (2013).
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
from sklearn.decomposition import PCA

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

MODEL_NAMES = ["null", "main", "full", "extended", "random_slope"]


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


# ---------------------------------------------------------------------
# I/O — kept compatible with statistical_tests.py
# ---------------------------------------------------------------------

def load_acoustic(path: Path, response_cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing acoustic table: {path}")
    df = pd.read_csv(path)
    require_columns(df, REQUIRED_ACOUSTIC_COLUMNS | set(response_cols), context=str(path))
    if df["token_id"].duplicated().any():
        dup = int(df["token_id"].duplicated().sum())
        raise ValueError(f"token_id must be unique in {path}; found {dup} duplicates")
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
# Data preparation
# ---------------------------------------------------------------------

def select_vowel_tokens(df: pd.DataFrame, configured_vowels: list[str],
                        response_cols: list[str], min_tokens: int = 5) -> tuple[pd.DataFrame, list[str]]:
    """Mirror statistical_tests.py: filter to oral vowels with >= min_tokens complete responses."""
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
    """Add binary L2/Male indicators and a vowel_height categorical column."""
    df = df.copy()
    df["L2"] = (df["L1_status"].astype(str).str.upper() == "L2").astype(int)
    df["Male"] = (df["gender"].astype(str).str.upper() == "M").astype(int)
    df["L2_x_Male"] = df["L2"] * df["Male"]
    df["vowel_height"] = df["phoneme_base"].map(vowel_height).astype("category")
    return df


# ---------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------

# Each entry is (model_name, formula, re_formula).
MODEL_SPECS: list[tuple[str, str, str | None]] = [
    ("null",         "{y} ~ 1",                              None),
    ("main",         "{y} ~ 1 + L2 + Male",                  None),
    ("full",         "{y} ~ 1 + L2 + Male + L2:Male",        None),
    ("extended",     "{y} ~ 1 + L2 + Male + L2:Male + C(vowel_height)", None),
    ("random_slope", "{y} ~ 1 + L2 + Male + L2:Male + C(vowel_height)", "1 + L2"),
]


def _try_single_fit(model, reml: bool, method_name: str):
    """One attempt at fitting; returns result or None on numerical failure."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return model.fit(reml=reml, method=[method_name], maxiter=400)
    except (np.linalg.LinAlgError, ValueError, AssertionError):
        return None
    except Exception:
        return None


def _random_var(result) -> float:
    """Return the total random-effect variance (trace of cov_re), or NaN."""
    try:
        cov_re = np.asarray(result.cov_re)
        if cov_re.ndim == 0:
            return float(cov_re)
        return float(np.trace(cov_re))
    except Exception:
        return float("nan")


def _se_has_nan(result) -> bool:
    try:
        bse = np.asarray(result.bse_fe, dtype=float)
        return bool(np.any(~np.isfinite(bse)))
    except Exception:
        return True


def fit_mixedlm(data: pd.DataFrame, y: str, formula: str, re_formula: str | None,
                method: str = "ml") -> tuple[object | None, str]:
    """Fit a MixedLM with ML or REML. Returns (result, status).

    status is 'ok', 'singular', 'boundary', 'failed: ...', or 'no_variance'.

    Strategy: try lbfgs first. If the random-effect variance collapses to zero
    (boundary solution, well-known MixedLM failure mode that produces
    var_speaker = 0 and r2_marginal = r2_conditional artefacts), fall back to
    powell which is slower but does not get stuck on the variance boundary.
    """
    fml = formula.format(y=y)
    sub = data.dropna(subset=[y]).copy()
    if sub[y].nunique() < 2:
        return None, "no_variance"

    reml = (method.lower() == "reml")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if re_formula is None:
                model = MixedLM.from_formula(fml, groups="speaker_id", data=sub)
            else:
                model = MixedLM.from_formula(fml, groups="speaker_id",
                                             re_formula=re_formula, data=sub)
    except (np.linalg.LinAlgError, ValueError, AssertionError) as e:
        return None, f"failed: {type(e).__name__}"
    except Exception as e:
        return None, f"failed: {type(e).__name__}"

    # First pass: lbfgs.
    result = _try_single_fit(model, reml=reml, method_name="lbfgs")

    # Boundary detection: if random variance collapsed (absolute OR as a tiny
    # fraction of total variance), retry with powell. The relative check
    # catches cases where lbfgs returns a numerically positive but practically
    # zero variance that produces r2_marginal == r2_conditional artefacts.
    needs_retry = False
    if result is None:
        needs_retry = True
    else:
        var_r = _random_var(result)
        try:
            var_e = float(result.scale)
        except Exception:
            var_e = float("nan")
        total = var_r + var_e if np.isfinite(var_r) and np.isfinite(var_e) else float("nan")
        if not np.isfinite(var_r) or var_r <= 1e-10:
            needs_retry = True
        elif np.isfinite(total) and total > 0 and (var_r / total) < 0.005:
            # Random variance is <0.5% of total — almost certainly a boundary
            # solution from lbfgs. Retry powell to confirm.
            needs_retry = True

    if needs_retry:
        retried = _try_single_fit(model, reml=reml, method_name="powell")
        if retried is not None:
            var_r2 = _random_var(retried)
            # Keep whichever fit has the larger random variance, since
            # boundary-trapped fits underestimate it.
            if result is None:
                result = retried
            else:
                var_r_old = _random_var(result)
                if np.isfinite(var_r2) and (
                    not np.isfinite(var_r_old) or var_r2 > var_r_old
                ):
                    result = retried

    if result is None:
        return None, "failed: all_methods"

    # Numerical sanity.
    try:
        scale = float(result.scale)
    except Exception:
        scale = float("nan")
    if not np.isfinite(scale) or scale <= 0:
        return result, "singular"

    var_r = _random_var(result)
    if not np.isfinite(var_r):
        return result, "singular"

    # If SEs contain NaN (typical for non-identifiable random-slope models),
    # report as singular: estimates may be readable but inference is invalid.
    if _se_has_nan(result):
        # Random-effect boundary at zero is the most common cause; if the
        # variance is essentially zero we tag as boundary, else singular.
        if var_r <= 1e-10:
            return result, "boundary"
        return result, "singular"

    return result, "ok"


def loglik(result) -> float:
    try:
        return float(result.llf)
    except Exception:
        return float("nan")


def k_params(result) -> int:
    try:
        # Fixed effects + variance components.
        n_fe = len(result.fe_params)
        n_re = result.cov_re.shape[0] if hasattr(result, "cov_re") else 1
        n_re_var = n_re * (n_re + 1) // 2
        return int(n_fe + n_re_var + 1)  # + residual
    except Exception:
        return 0


def aic_bic(result, n_obs: int) -> tuple[float, float]:
    ll = loglik(result)
    k = k_params(result)
    if not np.isfinite(ll) or k == 0:
        return float("nan"), float("nan")
    aic = 2.0 * k - 2.0 * ll
    bic = k * np.log(n_obs) - 2.0 * ll
    return aic, bic


def lrt(result_small, result_large) -> tuple[float, int, float]:
    ll_s = loglik(result_small)
    ll_l = loglik(result_large)
    k_s = k_params(result_small)
    k_l = k_params(result_large)
    if not np.isfinite(ll_s) or not np.isfinite(ll_l):
        return float("nan"), 0, float("nan")
    df = max(int(k_l - k_s), 1)
    stat = float(2.0 * (ll_l - ll_s))
    if stat < 0:
        # Numeric noise around zero; treat as zero with 1 df.
        return 0.0, df, 1.0
    p = float(1.0 - stats.chi2.cdf(stat, df=df))
    return stat, df, p


# ---------------------------------------------------------------------
# R^2 (Nakagawa & Schielzeth 2013) for gaussian LMM
# ---------------------------------------------------------------------

def r2_nakagawa(result, data: pd.DataFrame, y: str) -> tuple[float, float, float]:
    """Return (R2_marginal, R2_conditional, variance_random)."""
    try:
        # Fixed-effects prediction (no random effects).
        exog = result.model.exog
        beta = result.fe_params.values
        y_fixed = exog @ beta
        var_f = float(np.var(y_fixed, ddof=0))

        # Random-intercept variance (sum of diagonal of cov_re).
        cov_re = np.asarray(result.cov_re)
        if cov_re.ndim == 0:
            var_r = float(cov_re)
        else:
            var_r = float(np.trace(cov_re))

        var_e = float(result.scale)
        denom = var_f + var_r + var_e
        if denom <= 0 or not np.isfinite(denom):
            return float("nan"), float("nan"), float("nan")
        return float(var_f / denom), float((var_f + var_r) / denom), var_r
    except Exception:
        return float("nan"), float("nan"), float("nan")


def icc_from_null(result) -> float:
    try:
        cov_re = np.asarray(result.cov_re)
        var_u = float(cov_re) if cov_re.ndim == 0 else float(np.trace(cov_re))
        var_e = float(result.scale)
        denom = var_u + var_e
        if denom <= 0 or not np.isfinite(denom):
            return float("nan")
        return float(var_u / denom)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------
# Building blocks: run the full sequence for one response on one frame
# ---------------------------------------------------------------------

def coef_rows(result, representation: str, response: str, model_name: str) -> list[dict]:
    rows: list[dict] = []
    try:
        params = result.fe_params
        bse = result.bse_fe
        try:
            tvalues = result.tvalues
        except Exception:
            tvalues = params / bse
        try:
            pvalues = result.pvalues
        except Exception:
            pvalues = pd.Series(np.full(len(params), np.nan), index=params.index)
        try:
            ci = result.conf_int()
            ci.columns = ["ci_lower", "ci_upper"]
        except Exception:
            ci = pd.DataFrame({"ci_lower": np.full(len(params), np.nan),
                               "ci_upper": np.full(len(params), np.nan)},
                              index=params.index)
        for name in params.index:
            rows.append({
                "representation": representation,
                "response": response,
                "model": model_name,
                "term": name,
                "estimate": float(params[name]),
                "std_error": float(bse[name]),
                "t_value": float(tvalues[name]) if name in tvalues.index else float("nan"),
                "p_value": float(pvalues[name]) if name in pvalues.index else float("nan"),
                "ci_lower": float(ci.loc[name, "ci_lower"]) if name in ci.index else float("nan"),
                "ci_upper": float(ci.loc[name, "ci_upper"]) if name in ci.index else float("nan"),
            })
    except Exception:
        pass
    return rows


def run_model_building(data: pd.DataFrame, y: str, representation: str) -> dict:
    """Fit the five PDF models with ML, compute LRT/AIC/BIC and R^2.

    Returns a dict with keys: coef_rows, comparison_rows, slope_status_row,
    r2_main, r2_full, icc_null, fixed_coef_main (for forest plot).
    """
    fitted: dict[str, object | None] = {}
    statuses: dict[str, str] = {}
    coef_rows_all: list[dict] = []
    n_obs = int(data[y].dropna().shape[0])

    for name, formula, re_formula in MODEL_SPECS:
        result, status = fit_mixedlm(data, y, formula, re_formula, method="ml")
        fitted[name] = result
        statuses[name] = status
        # Accept 'ok' and 'boundary' (variance at zero is still usable for
        # point estimates; we just flag inference as compromised). Reject
        # 'singular' and any 'failed: ...' for inference.
        if result is not None and status in ("ok", "boundary"):
            coef_rows_all.extend(coef_rows(result, representation, y, name))

    # Pairwise comparisons (each model vs the previous one in the sequence).
    comparison_rows: list[dict] = []
    seq_pairs = [
        ("null", "main"),
        ("main", "full"),
        ("full", "extended"),
        ("extended", "random_slope"),
    ]
    for small, large in seq_pairs:
        r_s, r_l = fitted.get(small), fitted.get(large)
        st_s, st_l = statuses.get(small), statuses.get(large)
        usable = ("ok", "boundary")
        if r_s is not None and r_l is not None and st_s in usable and st_l in usable:
            stat, df, p = lrt(r_s, r_l)
        else:
            stat, df, p = float("nan"), 0, float("nan")
        aic_s, bic_s = aic_bic(r_s, n_obs) if r_s is not None else (float("nan"), float("nan"))
        aic_l, bic_l = aic_bic(r_l, n_obs) if r_l is not None else (float("nan"), float("nan"))
        comparison_rows.append({
            "representation": representation,
            "response": y,
            "model_small": small,
            "model_large": large,
            "status_small": st_s,
            "status_large": st_l,
            "loglik_small": loglik(r_s) if r_s is not None else float("nan"),
            "loglik_large": loglik(r_l) if r_l is not None else float("nan"),
            "lrt_chi2": stat,
            "lrt_df": df,
            "lrt_p_value": p,
            "aic_small": aic_s,
            "aic_large": aic_l,
            "bic_small": bic_s,
            "bic_large": bic_l,
        })

    # R^2 from main and full models.
    usable = ("ok", "boundary")
    r2m_main, r2c_main, _ = (
        r2_nakagawa(fitted["main"], data, y) if fitted["main"] is not None and statuses["main"] in usable
        else (float("nan"), float("nan"), float("nan"))
    )
    r2m_full, r2c_full, _ = (
        r2_nakagawa(fitted["full"], data, y) if fitted["full"] is not None and statuses["full"] in usable
        else (float("nan"), float("nan"), float("nan"))
    )
    r2m_ext, r2c_ext, _ = (
        r2_nakagawa(fitted["extended"], data, y) if fitted["extended"] is not None and statuses["extended"] in usable
        else (float("nan"), float("nan"), float("nan"))
    )

    # Partial R^2 for the L2 fixed effect (Q10): fit the extended model
    # WITHOUT L2 (keeping Male, L2:Male, and vowel_height), then take the
    # difference in marginal R^2. This isolates the variance explained by
    # the L1/L2 contrast from the variance explained by gender and vowel
    # height. The model is fit ad-hoc here; it is not part of the PDF's
    # nested model-building sequence.
    ext_no_l2_formula = "{y} ~ 1 + Male + L2:Male + C(vowel_height)"
    res_no_l2, st_no_l2 = fit_mixedlm(data, y, ext_no_l2_formula, None, method="ml")
    if res_no_l2 is not None and st_no_l2 in usable:
        r2m_ext_no_l2, _, _ = r2_nakagawa(res_no_l2, data, y)
    else:
        r2m_ext_no_l2 = float("nan")
    if np.isfinite(r2m_ext) and np.isfinite(r2m_ext_no_l2):
        r2m_ext_l2_partial = float(max(r2m_ext - r2m_ext_no_l2, 0.0))
    else:
        r2m_ext_l2_partial = float("nan")

    icc_val = (
        icc_from_null(fitted["null"]) if fitted["null"] is not None and statuses["null"] in usable
        else float("nan")
    )

    # Slope-status row for the random-slope model.
    slope_status_row = {
        "representation": representation,
        "response": y,
        "model": "random_slope",
        "status": statuses["random_slope"],
        "note": "L2 is between-speaker; random slope for L2 by speaker is structurally non-identifiable",
    }

    # Capture the L2 fixed-effect from the main model for the forest plot / Q10.
    main_result = fitted["main"] if statuses["main"] in usable else None
    if main_result is not None and "L2" in main_result.fe_params.index:
        l2_estimate = float(main_result.fe_params["L2"])
        l2_se = float(main_result.bse_fe["L2"])
        try:
            ci = main_result.conf_int().loc["L2"]
            l2_ci_lo, l2_ci_hi = float(ci[0]), float(ci[1])
        except Exception:
            l2_ci_lo, l2_ci_hi = float("nan"), float("nan")
    else:
        l2_estimate = l2_se = l2_ci_lo = l2_ci_hi = float("nan")

    return {
        "coef_rows": coef_rows_all,
        "comparison_rows": comparison_rows,
        "slope_status_row": slope_status_row,
        "r2m_main": r2m_main, "r2c_main": r2c_main,
        "r2m_full": r2m_full, "r2c_full": r2c_full,
        "r2m_ext": r2m_ext, "r2c_ext": r2c_ext,
        "r2m_ext_l2_partial": r2m_ext_l2_partial,
        "icc_null": icc_val,
        "n_obs": n_obs,
        "n_speakers": int(data["speaker_id"].nunique()),
        "fitted": fitted,
        "statuses": statuses,
        "l2_estimate": l2_estimate,
        "l2_se": l2_se,
        "l2_ci_lower": l2_ci_lo,
        "l2_ci_upper": l2_ci_hi,
    }


# ---------------------------------------------------------------------
# Interaction extraction (Q9)
# ---------------------------------------------------------------------

def interaction_row(fitted_full, representation: str, response: str, status: str) -> dict:
    out = {
        "representation": representation,
        "response": response,
        "term": "L2:Male",
        "estimate": float("nan"),
        "std_error": float("nan"),
        "t_value": float("nan"),
        "p_value": float("nan"),
        "ci_lower": float("nan"),
        "ci_upper": float("nan"),
        "status_full_model": status,
    }
    if fitted_full is None or status not in ("ok", "boundary"):
        return out
    try:
        params = fitted_full.fe_params
        target = None
        for cand in ("L2:Male", "Male:L2", "L2_x_Male"):
            if cand in params.index:
                target = cand
                break
        if target is None:
            return out
        bse = fitted_full.bse_fe
        try:
            tv = fitted_full.tvalues[target]
        except Exception:
            tv = params[target] / bse[target]
        try:
            pv = fitted_full.pvalues[target]
        except Exception:
            pv = float("nan")
        try:
            ci = fitted_full.conf_int().loc[target]
            ci_lo, ci_hi = float(ci[0]), float(ci[1])
        except Exception:
            ci_lo, ci_hi = float("nan"), float("nan")
        out.update({
            "term": target,
            "estimate": float(params[target]),
            "std_error": float(bse[target]),
            "t_value": float(tv),
            "p_value": float(pv),
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
        })
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------
# Forest plot for Q10
# ---------------------------------------------------------------------

def plot_forest_l2(rows: list[dict], out_path: Path) -> None:
    plt.figure(figsize=(8, max(4, 0.45 * len(rows) + 1.5)))
    labels = [r["representation"] for r in rows]
    est = np.array([r["l2_estimate"] for r in rows], dtype=float)
    lo = np.array([r["l2_ci_lower"] for r in rows], dtype=float)
    hi = np.array([r["l2_ci_upper"] for r in rows], dtype=float)

    y_pos = np.arange(len(rows))[::-1]
    err_lo = est - lo
    err_hi = hi - est
    plt.errorbar(est, y_pos, xerr=[err_lo, err_hi], fmt="o", color="black",
                 ecolor="gray", capsize=3)
    plt.axvline(0.0, color="red", linestyle="--", linewidth=0.8)
    plt.yticks(y_pos, labels)
    plt.xlabel("L2 fixed-effect estimate (main-effects model)")
    plt.title("L1/L2 fixed effect across representations\n(error bars: 95% CI)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 8: Linear Mixed-Effects Models")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    block = cfg["lme_models"]

    response_cols = block.get("acoustic_response_cols", ["F1_norm", "F2_norm"])
    alpha = float(block.get("alpha", 0.05))
    seed = int(block.get("random_state", 42))
    n_pcs = int(block.get("neural_pca_n_components", 5))
    icc_phoneme = str(block.get("icc_phoneme", "a"))
    icc_neural_rep = block.get("icc_neural_rep", {"model": "whisper", "layer": 12, "pc": 1})

    in_acoustic = Path(block["input_acoustic"])
    interim_dir = Path(block["interim_dir"])
    tab_dir = Path(block["tables_dir"])
    fig_dir = Path(block["figures_dir"]) / "lme"
    ensure_dirs(tab_dir, fig_dir)

    # Vowel-height mapping (required for the extended model).
    vowel_height = block.get("vowel_height", {})
    if not vowel_height:
        raise ValueError("config.lme_models.vowel_height must be defined for the extended model")

    configured_vowels = block.get("oral_vowels", ORAL_VOWEL_ORDER)

    rng = np.random.default_rng(seed)
    df = load_acoustic(in_acoustic, response_cols)

    # Filter to oral vowels with usable data, mirroring statistical_tests.
    df_v, phonemes = select_vowel_tokens(df, configured_vowels, response_cols, min_tokens=5)
    print(f"[lme_models] vowel set: {phonemes}")

    df_v = add_design_columns(df_v, vowel_height)
    if df_v["vowel_height"].isna().any():
        missing = sorted(set(df_v.loc[df_v["vowel_height"].isna(), "phoneme_base"].unique()))
        raise ValueError(f"vowel_height mapping missing for phonemes: {missing}")

    # ------------------------------------------------------------------
    # Acoustic LME — F1_norm and F2_norm.
    # ------------------------------------------------------------------
    all_coef_rows: list[dict] = []
    all_comparison_rows: list[dict] = []
    all_slope_rows: list[dict] = []
    all_interaction_rows: list[dict] = []
    forest_rows: list[dict] = []
    icc_rows: list[dict] = []
    r2_rows: list[dict] = []

    acoustic_data = df_v.dropna(subset=response_cols).copy()
    print(f"[lme_models] acoustic LME data: {len(acoustic_data)} tokens, "
          f"{acoustic_data['speaker_id'].nunique()} speakers")

    for resp in response_cols:
        rep_label = f"acoustic_{resp}"
        results = run_model_building(acoustic_data, y=resp, representation=rep_label)
        all_coef_rows.extend(results["coef_rows"])
        all_comparison_rows.extend(results["comparison_rows"])
        all_slope_rows.append(results["slope_status_row"])
        all_interaction_rows.append(
            interaction_row(results["fitted"]["full"], rep_label, resp, results["statuses"]["full"])
        )
        forest_rows.append({
            "representation": rep_label,
            "l2_estimate": results["l2_estimate"],
            "l2_se": results["l2_se"],
            "l2_ci_lower": results["l2_ci_lower"],
            "l2_ci_upper": results["l2_ci_upper"],
        })
        r2_rows.append({
            "representation": rep_label,
            "response": resp,
            "r2_marginal_main": results["r2m_main"],
            "r2_conditional_main": results["r2c_main"],
            "r2_marginal_full": results["r2m_full"],
            "r2_conditional_full": results["r2c_full"],
            "r2_marginal_extended": results["r2m_ext"],
            "r2_conditional_extended": results["r2c_ext"],
            "r2_marginal_extended_L2_partial": results["r2m_ext_l2_partial"],
            "n_obs": results["n_obs"],
            "n_speakers": results["n_speakers"],
        })

        # ICC on /a/ specifically (Q8) for F1 only.
        if resp == "F1_norm":
            sub_a = acoustic_data[acoustic_data["phoneme_base"] == icc_phoneme].copy()
            if len(sub_a) >= 10:
                res_a, st_a = fit_mixedlm(sub_a, y=resp,
                                          formula=f"{resp} ~ 1", re_formula=None, method="reml")
                icc_a = icc_from_null(res_a) if (res_a is not None and st_a in ("ok", "boundary")) else float("nan")
                icc_rows.append({
                    "representation": "acoustic_F1",
                    "phoneme": icc_phoneme,
                    "response": resp,
                    "n_tokens": int(len(sub_a)),
                    "n_speakers": int(sub_a["speaker_id"].nunique()),
                    "var_speaker": float(np.trace(np.asarray(res_a.cov_re))) if (res_a is not None and st_a in ("ok", "boundary")) else float("nan"),
                    "var_residual": float(res_a.scale) if (res_a is not None and st_a in ("ok", "boundary")) else float("nan"),
                    "ICC": icc_a,
                    "status": st_a,
                })
            else:
                icc_rows.append({
                    "representation": "acoustic_F1",
                    "phoneme": icc_phoneme,
                    "response": resp,
                    "n_tokens": int(len(sub_a)),
                    "n_speakers": int(sub_a["speaker_id"].nunique()),
                    "var_speaker": float("nan"),
                    "var_residual": float("nan"),
                    "ICC": float("nan"),
                    "status": "insufficient_data",
                })

    # ------------------------------------------------------------------
    # Neural LME — PCA(d=5) per (model, layer), one LME per PC.
    # ------------------------------------------------------------------
    # We need an index from token_id to row in df_v.
    meta_idx = df_v.set_index("token_id")

    for tag, prefix, layer in model_layer_pairs(cfg, block):
        layer_str = f"L{layer:02d}"
        rep_root = f"{tag}_{layer_str}"
        raw_path = interim_dir / f"{prefix}_{layer_str}.npz"
        emb, ids = load_raw_npz(raw_path)

        # Keep only tokens that are in our vowel subset.
        mask_known = pd.Index(ids).isin(meta_idx.index)
        emb_v = emb[mask_known]
        ids_v = ids[mask_known]
        if len(ids_v) < 20:
            print(f"[warning] {rep_root}: too few vowel tokens ({len(ids_v)}); skipped")
            continue

        # Fit PCA on the vowel-token subset for this (model, layer).
        pca = PCA(n_components=n_pcs, random_state=seed)
        X_pcs = pca.fit_transform(emb_v.astype(np.float32))
        ev_ratio = pca.explained_variance_ratio_.astype(float)

        # Standardise each PC to z-scores so that L2 coefficients are on a
        # comparable scale across (model, layer). Without this, XLS-R PCs have
        # huge variance and produce coefficients in the hundreds that are not
        # directly comparable to acoustic Lobanov-normalised F1/F2 effects.
        pc_std = X_pcs.std(axis=0, ddof=0)
        pc_std[pc_std == 0] = 1.0
        X_pcs = (X_pcs - X_pcs.mean(axis=0, keepdims=True)) / pc_std

        # Build a frame aligned to ids_v with the design columns from meta_idx.
        nrows = len(ids_v)
        aligned = meta_idx.loc[ids_v].reset_index()
        for k in range(n_pcs):
            aligned[f"PC{k+1}"] = X_pcs[:, k]

        # Diagnostics print.
        print(f"[lme_models] {rep_root}: {nrows} vowel tokens, "
              f"{aligned['speaker_id'].nunique()} speakers, "
              f"explained variance (first {n_pcs} PCs): "
              f"{np.round(ev_ratio, 3).tolist()}")

        # Run model building on each PC.
        rep_r2_marginal_main = []
        rep_r2_conditional_main = []
        rep_r2_marginal_full = []
        rep_r2_conditional_full = []
        rep_r2_marginal_ext = []
        rep_r2_conditional_ext = []
        rep_r2_marginal_ext_l2_partial = []

        # Forest-plot estimate from PC1 only (most natural single number;
        # the aggregated R^2 below covers Q10 across all PCs).
        rep_forest_added = False

        for k in range(n_pcs):
            y_col = f"PC{k+1}"
            rep_label = f"{rep_root}_PC{k+1}"
            results = run_model_building(aligned, y=y_col, representation=rep_label)
            all_coef_rows.extend(results["coef_rows"])
            all_comparison_rows.extend(results["comparison_rows"])
            all_slope_rows.append(results["slope_status_row"])
            all_interaction_rows.append(
                interaction_row(results["fitted"]["full"], rep_label, y_col, results["statuses"]["full"])
            )
            rep_r2_marginal_main.append(results["r2m_main"])
            rep_r2_conditional_main.append(results["r2c_main"])
            rep_r2_marginal_full.append(results["r2m_full"])
            rep_r2_conditional_full.append(results["r2c_full"])
            rep_r2_marginal_ext.append(results["r2m_ext"])
            rep_r2_conditional_ext.append(results["r2c_ext"])
            rep_r2_marginal_ext_l2_partial.append(results["r2m_ext_l2_partial"])
            r2_rows.append({
                "representation": rep_label,
                "response": y_col,
                "r2_marginal_main": results["r2m_main"],
                "r2_conditional_main": results["r2c_main"],
                "r2_marginal_full": results["r2m_full"],
                "r2_conditional_full": results["r2c_full"],
                "r2_marginal_extended": results["r2m_ext"],
                "r2_conditional_extended": results["r2c_ext"],
                "r2_marginal_extended_L2_partial": results["r2m_ext_l2_partial"],
                "n_obs": results["n_obs"],
                "n_speakers": results["n_speakers"],
            })

            # ICC on /a/ for the configured representation (Q8).
            if (tag == icc_neural_rep.get("model") and
                int(layer) == int(icc_neural_rep.get("layer")) and
                (k + 1) == int(icc_neural_rep.get("pc"))):
                sub_a = aligned[aligned["phoneme_base"] == icc_phoneme].copy()
                if len(sub_a) >= 10:
                    res_a, st_a = fit_mixedlm(sub_a, y=y_col,
                                              formula=f"{y_col} ~ 1", re_formula=None, method="reml")
                    icc_a = icc_from_null(res_a) if (res_a is not None and st_a in ("ok", "boundary")) else float("nan")
                    icc_rows.append({
                        "representation": rep_label,
                        "phoneme": icc_phoneme,
                        "response": y_col,
                        "n_tokens": int(len(sub_a)),
                        "n_speakers": int(sub_a["speaker_id"].nunique()),
                        "var_speaker": float(np.trace(np.asarray(res_a.cov_re))) if (res_a is not None and st_a in ("ok", "boundary")) else float("nan"),
                        "var_residual": float(res_a.scale) if (res_a is not None and st_a in ("ok", "boundary")) else float("nan"),
                        "ICC": icc_a,
                        "status": st_a,
                    })
                else:
                    icc_rows.append({
                        "representation": rep_label,
                        "phoneme": icc_phoneme,
                        "response": y_col,
                        "n_tokens": int(len(sub_a)),
                        "n_speakers": int(sub_a["speaker_id"].nunique()),
                        "var_speaker": float("nan"),
                        "var_residual": float("nan"),
                        "ICC": float("nan"),
                        "status": "insufficient_data",
                    })

            if not rep_forest_added and k == 0:
                forest_rows.append({
                    "representation": rep_root,
                    "l2_estimate": results["l2_estimate"],
                    "l2_se": results["l2_se"],
                    "l2_ci_lower": results["l2_ci_lower"],
                    "l2_ci_upper": results["l2_ci_upper"],
                })
                rep_forest_added = True

        # Aggregated R^2 per (model, layer) across the 5 PCs.
        m_un = float(np.nanmean(rep_r2_marginal_main))
        c_un = float(np.nanmean(rep_r2_conditional_main))
        m_full_un = float(np.nanmean(rep_r2_marginal_full))
        c_full_un = float(np.nanmean(rep_r2_conditional_full))
        m_ext_un = float(np.nanmean(rep_r2_marginal_ext))
        c_ext_un = float(np.nanmean(rep_r2_conditional_ext))
        m_ext_l2p_un = float(np.nanmean(rep_r2_marginal_ext_l2_partial))
        # Weighted by explained variance ratio.
        w = ev_ratio / ev_ratio.sum() if ev_ratio.sum() > 0 else np.full(n_pcs, 1.0 / n_pcs)
        m_w = float(np.nansum(w * np.asarray(rep_r2_marginal_main, dtype=float)))
        c_w = float(np.nansum(w * np.asarray(rep_r2_conditional_main, dtype=float)))
        m_full_w = float(np.nansum(w * np.asarray(rep_r2_marginal_full, dtype=float)))
        c_full_w = float(np.nansum(w * np.asarray(rep_r2_conditional_full, dtype=float)))
        m_ext_w = float(np.nansum(w * np.asarray(rep_r2_marginal_ext, dtype=float)))
        c_ext_w = float(np.nansum(w * np.asarray(rep_r2_conditional_ext, dtype=float)))
        m_ext_l2p_w = float(np.nansum(w * np.asarray(rep_r2_marginal_ext_l2_partial, dtype=float)))
        r2_rows.append({
            "representation": f"{rep_root}_AGG",
            "response": f"PC1..PC{n_pcs}",
            "r2_marginal_main": m_un,
            "r2_conditional_main": c_un,
            "r2_marginal_full": m_full_un,
            "r2_conditional_full": c_full_un,
            "r2_marginal_extended": m_ext_un,
            "r2_conditional_extended": c_ext_un,
            "r2_marginal_extended_L2_partial": m_ext_l2p_un,
            "r2_marginal_main_weighted": m_w,
            "r2_conditional_main_weighted": c_w,
            "r2_marginal_full_weighted": m_full_w,
            "r2_conditional_full_weighted": c_full_w,
            "r2_marginal_extended_weighted": m_ext_w,
            "r2_conditional_extended_weighted": c_ext_w,
            "r2_marginal_extended_L2_partial_weighted": m_ext_l2p_w,
            "n_obs": int(nrows),
            "n_speakers": int(aligned["speaker_id"].nunique()),
            "explained_variance_ratio": ";".join(f"{x:.4f}" for x in ev_ratio),
        })

    # ------------------------------------------------------------------
    # Write all tables.
    # ------------------------------------------------------------------
    pd.DataFrame(all_coef_rows).to_csv(tab_dir / "tab_lme_acoustic_neural_coef.csv", index=False)
    # For convenience: a split version per response domain.
    coef_df = pd.DataFrame(all_coef_rows)
    if not coef_df.empty:
        coef_df[coef_df["representation"].str.startswith("acoustic_")].to_csv(
            tab_dir / "tab_lme_acoustic_coef.csv", index=False
        )
        coef_df[~coef_df["representation"].str.startswith("acoustic_")].to_csv(
            tab_dir / "tab_lme_neural_coef.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(tab_dir / "tab_lme_acoustic_coef.csv", index=False)
        pd.DataFrame().to_csv(tab_dir / "tab_lme_neural_coef.csv", index=False)
    print(f"Wrote LME coefficient tables")

    comp_df = pd.DataFrame(all_comparison_rows)
    if not comp_df.empty:
        comp_df[comp_df["representation"].str.startswith("acoustic_")].to_csv(
            tab_dir / "tab_lme_acoustic_model_comparison.csv", index=False
        )
        comp_df[~comp_df["representation"].str.startswith("acoustic_")].to_csv(
            tab_dir / "tab_lme_neural_model_comparison.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(tab_dir / "tab_lme_acoustic_model_comparison.csv", index=False)
        pd.DataFrame().to_csv(tab_dir / "tab_lme_neural_model_comparison.csv", index=False)
    print(f"Wrote LME model-comparison tables")

    pd.DataFrame(all_slope_rows).to_csv(tab_dir / "tab_lme_random_slope_status.csv", index=False)
    print(f"Wrote {tab_dir / 'tab_lme_random_slope_status.csv'}")

    pd.DataFrame(icc_rows).to_csv(tab_dir / "tab_lme_icc_a.csv", index=False)
    print(f"Wrote {tab_dir / 'tab_lme_icc_a.csv'}")

    pd.DataFrame(all_interaction_rows).to_csv(tab_dir / "tab_lme_l1_gender_interaction.csv", index=False)
    print(f"Wrote {tab_dir / 'tab_lme_l1_gender_interaction.csv'}")

    pd.DataFrame(r2_rows).to_csv(tab_dir / "tab_lme_marginal_r2_l1.csv", index=False)
    print(f"Wrote {tab_dir / 'tab_lme_marginal_r2_l1.csv'}")

    # Forest plot of L2 fixed effect across representations.
    if forest_rows:
        plot_forest_l2(forest_rows, fig_dir / "fig_lme_forest_l2_effect.png")
        print(f"Wrote {fig_dir / 'fig_lme_forest_l2_effect.png'}")

    print("[lme_models] done")


if __name__ == "__main__":
    main()

"""
Snakefile — RU-FR Interference phonetics analysis pipeline.

Stages:
  1. parse_corpus            -> data/interim/tokens.csv
  2. extract_acoustics       -> data/interim/features_acoustic.csv
  3. extract_neural_whisper  -> data/interim/features_whisper_L{NN}.npz
  4. extract_neural_xlsr     -> data/interim/features_xlsr_L{NN}.npz
  5. normalise               -> acoustic normalisation + neural PCA/UMAP
  6. descriptive             -> acoustic, neural, and cross-representation descriptives
"""

from pathlib import Path
import hashlib

configfile: "config.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _whisper_outputs():
    nw = config["extract_neural_whisper"]
    return [
        f"{nw['output_dir']}/{nw['output_prefix']}_L{L:02d}.npz"
        for L in nw["layers"]
    ]


def _xlsr_outputs():
    nx = config["extract_neural_xlsr"]
    return [
        f"{nx['output_dir']}/{nx['output_prefix']}_L{L:02d}.npz"
        for L in nx["layers"]
    ]


def _neural_reduced_outputs():
    """List PCA and UMAP output files: two reduced files per model/layer."""
    nn = config["normalise_neural"]
    out_dir = nn["output_dir"]
    pca_suffix = nn["pca_output_suffix"]
    umap_suffix = nn["umap_output_suffix"]

    files = []

    for key in ("extract_neural_whisper", "extract_neural_xlsr"):
        up = config[key]

        for L in up["layers"]:
            Lstr = f"L{L:02d}"
            files.append(f"{out_dir}/{up['output_prefix']}_{Lstr}_{pca_suffix}.npz")
            files.append(f"{out_dir}/{up['output_prefix']}_{Lstr}_{umap_suffix}.npz")

    return files


def _descriptive_acoustic_outputs():
    """Report-ready acoustic descriptive figures."""
    return [
        "results/figures/descriptive/acoustic/fig_vowel_chart_panels.png",
        "results/figures/descriptive/acoustic/fig_boxplot_F1_norm_clean.png",
        "results/figures/descriptive/acoustic/fig_boxplot_F2_norm_clean.png",
        "results/figures/descriptive/acoustic/fig_intraspeaker_variability_sd.png",
    ]


def _descriptive_neural_layers_to_plot(tag):
    """Return neural layers for which report-ready figures are expected."""
    dn = config["descriptive_neural"]

    key = "extract_neural_whisper" if tag == "whisper" else "extract_neural_xlsr"
    configured = [int(x) for x in config[key]["layers"]]

    if dn.get("plot_all_layers", False):
        return configured

    report_layers = dn.get("report_layers", {}).get(tag, [])
    selected = [int(x) for x in report_layers if int(x) in configured]

    if selected:
        return selected

    return [configured[len(configured) // 2]]


def _neural_color_suffix(color_by):
    return {
        "phoneme_base": "phoneme",
        "L1_status": "L1_status",
        "gender": "gender",
    }.get(color_by, color_by)


def _descriptive_neural_outputs():
    """Report-ready neural projection figures only for selected layers."""
    figs = []

    color_by = config["descriptive_neural"].get(
        "color_by",
        ["phoneme_base", "L1_status", "gender"],
    )

    for tag in ("whisper", "xlsr"):
        for L in _descriptive_neural_layers_to_plot(tag):
            Lstr = f"L{L:02d}"

            for method in ("pca", "umap"):
                for c in color_by:
                    suffix = _neural_color_suffix(c)
                    figs.append(
                        f"results/figures/descriptive/neural/fig_{method}_{tag}_{Lstr}_by_{suffix}.png"
                    )

    return figs


def _descriptive_cross_outputs():
    """Selected RSM figures. Mantel table is computed for all layers."""
    figs = [
        "results/figures/descriptive/cross/fig_rsm_acoustic_main.png",
        "results/figures/descriptive/cross/fig_rsm_acoustic.png",
    ]

    dc = config["descriptive_cross"]

    if dc.get("acoustic_extended_feature_cols", []):
        figs.append("results/figures/descriptive/cross/fig_rsm_acoustic_extended.png")

    plot_layers = dc.get("rsm_plot_layers", {})

    for tag in ("whisper", "xlsr"):
        for L in plot_layers.get(tag, []):
            Lstr = f"L{int(L):02d}"
            figs.append(f"results/figures/descriptive/cross/fig_rsm_{tag}_{Lstr}.png")

    return figs


# ---------------------------------------------------------------------------
# Default target
# ---------------------------------------------------------------------------

rule all:
    input:
        # Stage 1
        config["parse_corpus"]["output"],

        # Stage 2
        config["extract_acoustics"]["output_features"],

        # Stage 3
        _whisper_outputs(),

        # Stage 4
        _xlsr_outputs(),

        # Stage 5a
        config["normalise_acoustic"]["output_features"],

        # Stage 5b
        _neural_reduced_outputs(),

        # Stage 6a: descriptive_acoustic
        "results/tables/tab_acoustic_descriptives.csv",
        "results/tables/tab_acoustic_missingness.csv",
        "results/tables/tab_variance_decomposition.csv",
        "results/tables/tab_intraspeaker_variability_sd.csv",
        _descriptive_acoustic_outputs(),

        # Stage 6b: descriptive_neural
        "results/tables/tab_neural_between_class_ratio.csv",
        "results/tables/tab_neural_cosine_within_between.csv",
        "results/tables/tab_neural_inter_speaker_variability.csv",
        _descriptive_neural_outputs(),

        # Stage 6c: descriptive_cross
        "results/tables/tab_mantel_results.csv",
        _descriptive_cross_outputs(),


# ---------------------------------------------------------------------------
# Stage 1: parse_corpus
# ---------------------------------------------------------------------------

_TG_DIR = Path("data/raw/ru-fr_interference/wav_et_textgrids/FRcorp_textgrids_only")


def _textgrid_digest() -> str:
    """Return a short hash summarising the TextGrid corpus state."""
    if not _TG_DIR.exists():
        return "no-corpus"

    h = hashlib.sha1()

    for p in sorted(_TG_DIR.rglob("*.TextGrid")):
        h.update(str(p).encode("utf-8"))
        h.update(str(p.stat().st_mtime_ns).encode("utf-8"))

    return h.hexdigest()[:12]


rule parse_corpus:
    input:
        script="src/parse_corpus.py",
        config="config.yaml",
        metadata=config["parse_corpus"]["metadata"],
        rufrcorr=config["parse_corpus"]["rufrcorr"],
    output:
        config["parse_corpus"]["output"]
    params:
        tg_digest=_textgrid_digest()
    shell:
        "pixi run python {input.script} --config {input.config}"


# ---------------------------------------------------------------------------
# Stage 2: extract_acoustics
# ---------------------------------------------------------------------------

rule extract_acoustics:
    input:
        script="src/extract_acoustics.py",
        config="config.yaml",
        tokens=config["extract_acoustics"]["input_tokens"],
    output:
        config["extract_acoustics"]["output_features"]
    shell:
        "pixi run python {input.script} --config {input.config}"


# ---------------------------------------------------------------------------
# Stage 3: extract_neural_whisper
# ---------------------------------------------------------------------------

rule extract_neural_whisper:
    input:
        script="src/extract_neural_whisper.py",
        config="config.yaml",
        tokens=config["extract_neural_whisper"]["input_tokens"],
        features_acoustic=config["extract_neural_whisper"]["input_features_acoustic"],
    output:
        _whisper_outputs()
    shell:
        "pixi run python {input.script} --config {input.config}"


# ---------------------------------------------------------------------------
# Stage 4: extract_neural_xlsr
# ---------------------------------------------------------------------------

rule extract_neural_xlsr:
    input:
        script="src/extract_neural_xlsr.py",
        config="config.yaml",
        tokens=config["extract_neural_xlsr"]["input_tokens"],
        features_acoustic=config["extract_neural_xlsr"]["input_features_acoustic"],
    output:
        _xlsr_outputs()
    shell:
        "pixi run python {input.script} --config {input.config}"


# ---------------------------------------------------------------------------
# Stage 5a: normalise_acoustic
# ---------------------------------------------------------------------------

rule normalise_acoustic:
    input:
        script="src/normalise_acoustic.py",
        config="config.yaml",
        features=config["normalise_acoustic"]["input_features"],
    output:
        config["normalise_acoustic"]["output_features"]
    shell:
        "pixi run python {input.script} --config {input.config}"


# ---------------------------------------------------------------------------
# Stage 5b: normalise_neural
# ---------------------------------------------------------------------------

rule normalise_neural:
    input:
        script="src/normalise_neural.py",
        config="config.yaml",
        whisper_npz=_whisper_outputs(),
        xlsr_npz=_xlsr_outputs(),
    output:
        _neural_reduced_outputs()
    shell:
        "pixi run python {input.script} --config {input.config}"


# ---------------------------------------------------------------------------
# Stage 6a: descriptive_acoustic
# ---------------------------------------------------------------------------

rule descriptive_acoustic:
    input:
        script="src/descriptive_acoustic.py",
        config="config.yaml",
        features=config["descriptive_acoustic"]["input_features"],
    output:
        "results/tables/tab_acoustic_descriptives.csv",
        "results/tables/tab_acoustic_missingness.csv",
        "results/tables/tab_variance_decomposition.csv",
        "results/tables/tab_intraspeaker_variability_sd.csv",
        _descriptive_acoustic_outputs(),
    shell:
        "pixi run python {input.script} --config {input.config}"


# ---------------------------------------------------------------------------
# Stage 6b: descriptive_neural
# ---------------------------------------------------------------------------

rule descriptive_neural:
    input:
        script="src/descriptive_neural.py",
        config="config.yaml",
        metadata=config["descriptive_neural"]["input_metadata"],
        whisper_npz=_whisper_outputs(),
        xlsr_npz=_xlsr_outputs(),
        reduced_npz=_neural_reduced_outputs(),
    output:
        "results/tables/tab_neural_between_class_ratio.csv",
        "results/tables/tab_neural_cosine_within_between.csv",
        "results/tables/tab_neural_inter_speaker_variability.csv",
        _descriptive_neural_outputs(),
    shell:
        "pixi run python {input.script} --config {input.config}"


# ---------------------------------------------------------------------------
# Stage 6c: descriptive_cross
# ---------------------------------------------------------------------------

rule descriptive_cross:
    input:
        script="src/descriptive_cross.py",
        config="config.yaml",
        acoustic=config["descriptive_cross"]["input_acoustic"],
        whisper_npz=_whisper_outputs(),
        xlsr_npz=_xlsr_outputs(),
    output:
        "results/tables/tab_mantel_results.csv",
        _descriptive_cross_outputs(),
    shell:
        "pixi run python {input.script} --config {input.config}"

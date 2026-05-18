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

def _lme_models_outputs():
    """Tables and figures written by src/lme_models.py."""
    block = config["lme_models"]
    tab_dir = block["tables_dir"]
    fig_dir = f"{block['figures_dir']}/lme"
    return [
        f"{tab_dir}/tab_lme_acoustic_coef.csv",
        f"{tab_dir}/tab_lme_neural_coef.csv",
        f"{tab_dir}/tab_lme_acoustic_neural_coef.csv",
        f"{tab_dir}/tab_lme_acoustic_model_comparison.csv",
        f"{tab_dir}/tab_lme_neural_model_comparison.csv",
        f"{tab_dir}/tab_lme_random_slope_status.csv",
        f"{tab_dir}/tab_lme_icc_a.csv",
        f"{tab_dir}/tab_lme_l1_gender_interaction.csv",
        f"{tab_dir}/tab_lme_marginal_r2_l1.csv",
        f"{fig_dir}/fig_lme_forest_l2_effect.png",
    ]

def _rope_ci_outputs():
    """Tables and figures written by src/rope_ci.py."""
    block = config["rope_ci"]
    tab_dir = block["tables_dir"]
    fig_dir = f"{block['figures_dir']}/rope_ci"
    return [
        f"{tab_dir}/tab_rope_ci_acoustic.csv",
        f"{tab_dir}/tab_rope_ci_neural.csv",
        f"{tab_dir}/tab_rope_summary.csv",
        f"{tab_dir}/tab_rope_delta0.csv",
        f"{tab_dir}/tab_rope_acoustic_scale.csv",
        f"{fig_dir}/fig_rope_acoustic_forest_F1.png",
        f"{fig_dir}/fig_rope_acoustic_forest_F2.png",
        f"{fig_dir}/fig_rope_neural_forest.png",
        f"{fig_dir}/fig_rope_neural_forest_all_layers.png",
    ]

def _clustering_outputs():
    """Tables and figures written by src/clustering.py."""
    block = config["clustering"]
    tab_dir = block["tables_dir"]
    fig_dir = f"{block['figures_dir']}/clustering"

    files = [
        f"{tab_dir}/tab_clust_consonant_set.csv",
        f"{tab_dir}/tab_clust_vowel_ari.csv",
        f"{tab_dir}/tab_clust_vowel_assignments.csv",
        f"{tab_dir}/tab_clust_cv_ari.csv",
        f"{tab_dir}/tab_clust_cv_assignments.csv",
        f"{tab_dir}/tab_clust_speaker_ari.csv",
        f"{tab_dir}/tab_clust_speaker_assignments.csv",
        f"{tab_dir}/tab_clust_k_selection.csv",
        f"{tab_dir}/tab_clust_q16_systematic_errors.csv",
        f"{fig_dir}/fig_dendro_vowel_acoustic.png",
        f"{fig_dir}/fig_dendro_cv_acoustic.png",
        f"{fig_dir}/fig_dendro_speaker_acoustic.png",
        f"{fig_dir}/fig_dendro_vowel_all_layers.png",
        f"{fig_dir}/fig_silhouette_summary.png",
        f"{fig_dir}/fig_ari_summary.png",
    ]

    rep = block.get("representative_layers", {})
    for tag in ("whisper", "xlsr"):
        L = int(rep.get(tag))
        Lstr = f"L{L:02d}"
        files.append(f"{fig_dir}/fig_dendro_vowel_{tag}_{Lstr}.png")
        files.append(f"{fig_dir}/fig_dendro_cv_{tag}_{Lstr}.png")
        files.append(f"{fig_dir}/fig_dendro_speaker_{tag}_{Lstr}.png")

    return files

def _statistical_tests_outputs():
    """Tables and selected figures written by src/statistical_tests.py."""
    st = config["statistical_tests"]
    tab_dir = st["tables_dir"]
    fig_dir = f"{st['figures_dir']}/statistical_tests"

    files = [
        f"{tab_dir}/tab_stat_acoustic_l1_l2_tests.csv",
        f"{tab_dir}/tab_stat_acoustic_gender_tests.csv",
        f"{tab_dir}/tab_stat_neural_l1_l2_permutation.csv",
        f"{tab_dir}/tab_stat_distance_mantel.csv",
        f"{tab_dir}/tab_stat_distance_bootstrap_ci.csv",
        f"{tab_dir}/tab_stat_classifier_predictions.csv",
        f"{tab_dir}/tab_stat_classifier_accuracy.csv",
        f"{tab_dir}/tab_stat_classifier_f1.csv",
        f"{tab_dir}/tab_stat_mcnemar.csv",
        f"{tab_dir}/tab_stat_distance_matrix_acoustic_euclidean.csv",
        f"{tab_dir}/tab_stat_distance_matrix_acoustic_mahalanobis.csv",
        f"{fig_dir}/distances/fig_distance_matrix_acoustic_euclidean.png",
        f"{fig_dir}/distances/fig_distance_matrix_acoustic_mahalanobis.png",
        f"{fig_dir}/classification/fig_confusion_acoustic_euclidean.png",
    ]

    for feat in st.get("acoustic_feature_cols", ["F1_norm", "F2_norm"]):
        files.append(f"{fig_dir}/acoustic/fig_qq_{feat}_by_vowel_group.png")

    for tag in ("whisper", "xlsr"):
        key = f"extract_neural_{tag}"
        default_layers = config[key]["layers"]
        for L in st.get("neural_layers", {}).get(tag, default_layers):
            files.append(f"{tab_dir}/tab_stat_distance_matrix_{tag}_L{int(L):02d}.csv")

    for tag in ("whisper", "xlsr"):
        for L in st.get("classification_plot_layers", {}).get(tag, []):
            files.append(f"{fig_dir}/classification/fig_confusion_{tag}_L{int(L):02d}.png")

    return files

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

        # Stage 7: statistical_tests
        _statistical_tests_outputs(),

        # Stage 8: lme_models
        _lme_models_outputs(),

        # Stage 9: rope_ci
        _rope_ci_outputs(),

        # Stage 10: clustering
        _clustering_outputs(),


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

# ---------------------------------------------------------------------------
# Stage 7: statistical_tests
# ---------------------------------------------------------------------------

rule statistical_tests:
    input:
        script="src/statistical_tests.py",
        config="config.yaml",
        acoustic=config["statistical_tests"]["input_acoustic"],
        whisper_npz=_whisper_outputs(),
        xlsr_npz=_xlsr_outputs(),
    output:
        _statistical_tests_outputs(),
    shell:
        "pixi run python {input.script} --config {input.config}"


# ---------------------------------------------------------------------------
# Stage 8: lme_models
# ---------------------------------------------------------------------------

rule lme_models:
    input:
        script="src/lme_models.py",
        config="config.yaml",
        acoustic=config["lme_models"]["input_acoustic"],
        whisper_npz=_whisper_outputs(),
        xlsr_npz=_xlsr_outputs(),
    output:
        _lme_models_outputs(),
    shell:
        "pixi run python {input.script} --config {input.config}"

# ---------------------------------------------------------------------------
# Stage 9: rope_ci
# ---------------------------------------------------------------------------

rule rope_ci:
    input:
        script="src/rope_ci.py",
        config="config.yaml",
        acoustic=config["rope_ci"]["input_acoustic"],
        acoustic_raw=config["rope_ci"]["input_features_acoustic_raw"],
        whisper_npz=_whisper_outputs(),
        xlsr_npz=_xlsr_outputs(),
    output:
        _rope_ci_outputs(),
    shell:
        "pixi run python {input.script} --config {input.config}"

# ---------------------------------------------------------------------------
# Stage 10: clustering
# ---------------------------------------------------------------------------

rule clustering:
    input:
        script="src/clustering.py",
        config="config.yaml",
        acoustic_norm=config["clustering"]["input_acoustic_norm"],
        acoustic_raw=config["clustering"]["input_acoustic_raw"],
        whisper_npz=_whisper_outputs(),
        xlsr_npz=_xlsr_outputs(),
        reduced_npz=_neural_reduced_outputs(),
    output:
        _clustering_outputs(),
    shell:
        "pixi run python {input.script} --config {input.config}"
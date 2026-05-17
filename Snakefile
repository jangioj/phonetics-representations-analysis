"""
Snakefile — RU-FR Interference phonetics analysis pipeline.

Stages:
  1. parse_corpus            -> data/interim/tokens.csv
  2. extract_acoustics       -> data/interim/features_acoustic.csv
  3. extract_neural_whisper  -> data/interim/features_whisper_L{NN}.npz
  4. extract_neural_xlsr     -> data/interim/features_xlsr_L{NN}.npz
"""

from pathlib import Path

configfile: "config.yaml"

#Helpers
def _whisper_outputs():
    nw = config["extract_neural_whisper"]
    return [f"{nw['output_dir']}/{nw['output_prefix']}_L{L:02d}.npz" for L in nw["layers"]]

def _xlsr_outputs():
    nx = config["extract_neural_xlsr"]
    return [f"{nx['output_dir']}/{nx['output_prefix']}_L{L:02d}.npz" for L in nx["layers"]]

def _neural_pca_outputs():
    """List of PCA-reduced .npz files: one per (model, layer)."""
    nn = config["normalise_neural"]
    out_dir = nn["output_dir"]
    suffix = nn["output_suffix"]
    files = []
    for key in ("extract_neural_whisper", "extract_neural_xlsr"):
        up = config[key]
        for L in up["layers"]:
            files.append(f"{out_dir}/{up['output_prefix']}_L{L:02d}_{suffix}.npz")
    return files

# Default target: build everything currently defined.
rule all:
    input:
        config["parse_corpus"]["output"],
        config["extract_acoustics"]["output_features"],
        _whisper_outputs(),
        _xlsr_outputs(),
        config["normalise_acoustic"]["output_features"],
        _neural_pca_outputs()


# ---------------------------------------------------------------------------
# Stage 1: parse_corpus
# ---------------------------------------------------------------------------
# Note on TextGrid tracking: with 1482 source files, listing each as an
# explicit input would clutter every Snakemake log. Instead we summarize them
# as a single content-hash digest stored in `params.tg_digest`. Snakemake
# re-runs the rule when this digest changes, which happens iff any TextGrid
# is added, removed, or modified.
import hashlib

_TG_DIR = Path("data/raw/ru-fr_interference/wav_et_textgrids/FRcorp_textgrids_only")

def _textgrid_digest() -> str:
    """Return a short hash summarizing the corpus content (filenames + mtimes)."""
    if not _TG_DIR.exists():
        return "no-corpus"
    h = hashlib.sha1()
    for p in sorted(_TG_DIR.rglob("*.TextGrid")):
        h.update(str(p).encode("utf-8"))
        h.update(str(p.stat().st_mtime_ns).encode("utf-8"))
    return h.hexdigest()[:12]


rule parse_corpus:
    input:
        script = "src/parse_corpus.py",
        config = "config.yaml",
        metadata = config["parse_corpus"]["metadata"],
        rufrcorr = config["parse_corpus"]["rufrcorr"],
    output:
        config["parse_corpus"]["output"]
    params:
        tg_digest = _textgrid_digest()
    shell:
        "pixi run python {input.script} --config {input.config}"

# ---------------------------------------------------------------------------
# Stage 2: extract_acoustics
# ---------------------------------------------------------------------------
rule extract_acoustics:
    input:
        script = "src/extract_acoustics.py",
        config = "config.yaml",
        tokens = config["extract_acoustics"]["input_tokens"],
    output:
        config["extract_acoustics"]["output_features"]
    shell:
        "pixi run python {input.script} --config {input.config}"

# ---------------------------------------------------------------------------
# Stage 3: extract_neural_whisper
# ---------------------------------------------------------------------------
# Runs locally on CPU or on Colab GPU. Running locally is supported for debug
rule extract_neural_whisper:
    input:
        script = "src/extract_neural_whisper.py",
        config = "config.yaml",
        tokens = config["extract_neural_whisper"]["input_tokens"],
        features_acoustic = config["extract_neural_whisper"]["input_features_acoustic"],
    output:
        _whisper_outputs()
    shell:
        "pixi run python {input.script} --config {input.config}"

# ---------------------------------------------------------------------------
# Stage 4: extract_neural_xlsr
# ---------------------------------------------------------------------------
# Runs locally on CPU or on Colab GPU. Running locally is supported for debug.
rule extract_neural_xlsr:
    input:
        script = "src/extract_neural_xlsr.py",
        config = "config.yaml",
        tokens = config["extract_neural_xlsr"]["input_tokens"],
        features_acoustic = config["extract_neural_xlsr"]["input_features_acoustic"],
    output:
        _xlsr_outputs()
    shell:
        "pixi run python {input.script} --config {input.config}"

# ---------------------------------------------------------------------------
# Stage 5a: normalise_acoustic
# ---------------------------------------------------------------------------
rule normalise_acoustic:
    input:
        script = "src/normalise_acoustic.py",
        config = "config.yaml",
        features = config["normalise_acoustic"]["input_features"],
    output:
        config["normalise_acoustic"]["output_features"]
    shell:
        "pixi run python {input.script} --config {input.config}"

# ---------------------------------------------------------------------------
# Stage 5b: normalise_neural
# ---------------------------------------------------------------------------
rule normalise_neural:
    input:
        script = "src/normalise_neural.py",
        config = "config.yaml",
        whisper_npz = _whisper_outputs(),
        xlsr_npz = _xlsr_outputs(),
    output:
        _neural_pca_outputs()
    shell:
        "pixi run python {input.script} --config {input.config}"
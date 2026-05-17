"""
Snakefile — RU-FR Interference phonetics analysis pipeline.

Stages:
  1. parse_corpus        -> data/interim/tokens.csv
  2. extract_acoustics   -> data/interim/features_acoustic.csv
"""

from pathlib import Path

configfile: "config.yaml"

def _whisper_outputs():
    nw = config["extract_neural_whisper"]
    return [f"{nw['output_dir']}/{nw['output_prefix']}_L{L:02d}.npz" for L in nw["layers"]]

# Default target: build everything currently defined.
rule all:
    input:
        config["parse_corpus"]["output"],
        config["extract_acoustics"]["output_features"],
        _whisper_outputs()


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
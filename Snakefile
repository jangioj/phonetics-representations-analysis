"""
Snakefile — Phonetics Representations Analysis
Pipeline orchestration. Run with:  pixi run snakemake --cores 1

Each stage will be added as a rule in its dedicated chat.
"""

configfile: "config.yaml"


# Default target: what `snakemake` builds when no target is specified.
# Will be expanded as stages are added.
rule all:
    input:
        []   # placeholder: nessun output ancora prodotto


# --- Stage 1: parse_corpus (chat [02]) ---
# rule parse_corpus:
#     input:
#         textgrids = config["raw_textgrid_dir"],
#         metadata = "data/metadata.csv"
#     output:
#         tokens = config["interim_dir"] + "/tokens.csv"
#     script:
#         "src/parse_corpus.py"


# --- Stage 2: extract_acoustics (chat [03]) ---
# ...


# --- Stage 3: extract_neural_whisper (chat [04]) ---
# ...
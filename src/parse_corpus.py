"""
parse_corpus.py — Pipeline stage 1.

Reads the RU-FR Interference corpus TextGrids and speaker metadata,
produces data/interim/tokens.csv with one row per target-word phoneme.

Run directly:
    pixi run python src/parse_corpus.py --config config.yaml
Or via Snakemake:
    pixi run snakemake --cores 1 data/interim/tokens.csv

All tunable parameters (paths, debug subset) live in config.yaml so that
Snakemake can detect parameter changes and re-run the stage automatically.
"""
from __future__ import annotations

import csv
import sys
import unicodedata
from pathlib import Path
import argparse

import pandas as pd
import yaml
from praatio import textgrid

# ---------------------------------------------------------------------------
# Phonological lookup
# ---------------------------------------------------------------------------
# French vowel inventory (oral + nasal). Glides /j w ɥ/ are treated as
# consonants (standard phonetic analysis: they are approximants, not
# syllabic nuclei).
FRENCH_VOWELS: set[str] = {
    # oral
    "i", "e", "ɛ", "a", "ɑ", "ɔ", "o", "u",
    "y", "ø", "œ", "ə",
    # nasal
    "ɛ̃", "ɑ̃", "ɔ̃", "œ̃",
}

DISTRACTOR_MARKER = "distractor"


# ---------------------------------------------------------------------------
# String normalization (used for matching target words against TextGrid labels)
# ---------------------------------------------------------------------------
def _normalize_word(s: str) -> str:
    """Fold curly quotes to straight, lowercase, NFC-normalize, strip."""
    s = unicodedata.normalize("NFC", s)
    for ch in ("\u2019", "\u02bc", "\u2018", "`", "´"):
        s = s.replace(ch, "'")
    return s.strip().lower()


# ---------------------------------------------------------------------------
# Lookup tables construction
# ---------------------------------------------------------------------------
def load_speakers_metadata(path: Path) -> dict[str, dict]:
    """Build speaker_id -> {'L1_status', 'gender'} mapping.

    Recoding rule (matches PDF section 2):
      metadata 'L1' column = speaker's native language ('fr' or 'ru').
      Our L1_status is relative to French (the studied language):
        - native French speakers  -> 'L1'
        - native Russian speakers -> 'L2'
    """
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            spk = row["spk"]
            native = row["L1"].strip().lower()
            gender = row["Gender"].strip().lower()
            l1_status = "L1" if native == "fr" else "L2"
            out[spk] = {
                "L1_status": l1_status,
                "gender": gender.upper(),
            }
    return out


def load_slot_to_target(
    path: Path,
    max_repetitions: int | None = None,
) -> dict[int, dict]:
    """Build slot_number (1..78) -> {'target_word', 'repetition', 'target_ipa'}.

    Distractor slots are kept with target_word='distractor' so the caller
    can detect and skip them.

    If `max_repetitions` is given, only the first K occurrences per word
    are kept (debug subset).
    """
    out: dict[int, dict] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            word = row["Word"].strip()
            ipa_raw = row["Ipa"].strip()
            ipa_seq = [] if ipa_raw == "N/A" else ipa_raw.split(";")
            cols = ["occ.1", "occ.2", "occ.3", "occ.4", "occ.5", "occ.6"]
            if max_repetitions is not None:
                cols = cols[:max_repetitions]
            for rep_idx, col in enumerate(cols, start=1):
                try:
                    slot = int(row[col])
                except (ValueError, TypeError, KeyError):
                    continue
                out[slot] = {
                    "target_word": word,
                    "repetition": rep_idx,
                    "target_ipa": ipa_seq,
                }
    return out


# ---------------------------------------------------------------------------
# Debug-mode speaker selection
# ---------------------------------------------------------------------------
def select_speaker_subset(
    speakers: dict[str, dict],
    n_speakers: int,
) -> set[str]:
    """Pick `n_speakers` stratified across the four L1×Gender groups.

    Deterministic: alphabetical within each group, round-robin between
    groups (largest first) until we have `n_speakers`.
    """
    by_group: dict[tuple[str, str], list[str]] = {}
    for spk, meta in speakers.items():
        key = (meta["L1_status"], meta["gender"])
        by_group.setdefault(key, []).append(spk)
    for v in by_group.values():
        v.sort()

    groups_sorted = sorted(by_group.keys(), key=lambda k: -len(by_group[k]))
    selected: list[str] = []
    indices = {g: 0 for g in by_group}
    while len(selected) < n_speakers:
        progressed = False
        for g in groups_sorted:
            if indices[g] < len(by_group[g]):
                selected.append(by_group[g][indices[g]])
                indices[g] += 1
                progressed = True
                if len(selected) == n_speakers:
                    break
        if not progressed:
            break
    return set(selected)


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------
def parse_textgrid_filename(name: str) -> tuple[str, int]:
    """'ab_rus_list1_FRcorp2.TextGrid' -> ('AB', 2)."""
    stem = name.removesuffix(".TextGrid")
    parts = stem.split("_")
    spk = parts[0].upper()
    last = parts[-1]
    assert last.startswith("FRcorp"), f"Unexpected filename: {name}"
    slot = int(last.removeprefix("FRcorp"))
    return spk, slot


# ---------------------------------------------------------------------------
# Target word location inside a TextGrid
# ---------------------------------------------------------------------------
def find_target_word_span(
    words_tier_entries: list[tuple[float, float, str]],
    target_word: str,
) -> tuple[float, float] | None:
    """Find the time span of the target word inside the `words` tier.

    Multi-word targets (e.g. "j'en chie", "cache cache") may appear in the
    TextGrid as a single interval whose text contains spaces, or as several
    consecutive intervals. We match by concatenating non-silent intervals
    (after stripping internal whitespace) until they reproduce the target
    string with all whitespace removed.
    """
    target_concat = _normalize_word(target_word).replace(" ", "")
    non_silent = [
        (on, off, _normalize_word(text).replace(" ", ""))
        for (on, off, text) in words_tier_entries
        if text.strip() != ""
    ]
    for start in range(len(non_silent)):
        acc = ""
        for end in range(start, len(non_silent)):
            acc += non_silent[end][2]
            if acc == target_concat:
                return (non_silent[start][0], non_silent[end][1])
            if not target_concat.startswith(acc):
                break
    return None


# ---------------------------------------------------------------------------
# Phoneme extraction
# ---------------------------------------------------------------------------
def extract_phonemes_in_span(
    phones_tier_entries: list[tuple[float, float, str]],
    on_span: float,
    off_span: float,
) -> list[tuple[float, float, str]]:
    """Return non-silent phonemes whose midpoint falls within [on_span, off_span].

    Midpoint inclusion avoids edge ambiguities at word boundaries.
    """
    out = []
    for on, off, text in phones_tier_entries:
        if text.strip() == "":
            continue
        midpoint = (on + off) / 2
        if on_span <= midpoint <= off_span:
            out.append((on, off, text))
    return out


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------
def process_textgrid(
    tg_path: Path,
    speakers: dict[str, dict],
    slot_lookup: dict[int, dict],
) -> list[dict]:
    """Return a list of token-rows for the given TextGrid, [] if to be skipped."""
    spk, slot = parse_textgrid_filename(tg_path.name)
    info = slot_lookup.get(slot)
    if info is None:
        return []
    target_word = info["target_word"]
    if target_word == DISTRACTOR_MARKER:
        return []

    spk_meta = speakers.get(spk)
    if spk_meta is None:
        print(f"  [WARN] speaker {spk} not in metadata ({tg_path.name})", file=sys.stderr)
        return []

    tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=True)
    if "words" not in tg.tierNames or "phones" not in tg.tierNames:
        print(f"  [WARN] missing tier in {tg_path.name}: {tg.tierNames}", file=sys.stderr)
        return []

    words_entries = [(e.start, e.end, e.label) for e in tg.getTier("words").entries]
    phones_entries = [(e.start, e.end, e.label) for e in tg.getTier("phones").entries]

    span = find_target_word_span(words_entries, target_word)
    if span is None:
        print(
            f"  [WARN] target {target_word!r} not found in {tg_path.name}",
            file=sys.stderr,
        )
        return []
    on_span, off_span = span

    phonemes = extract_phonemes_in_span(phones_entries, on_span, off_span)
    if not phonemes:
        print(f"  [WARN] zero target phonemes in {tg_path.name}", file=sys.stderr)
        return []

    rows = []
    for on, off, label in phonemes:
        rows.append(
            {
                "speaker_id": spk,
                "sentence_id": target_word,
                "repetition": info["repetition"],
                "phoneme": label,
                "onset": on,
                "offset": off,
                "duration": off - on,
                "L1_status": spk_meta["L1_status"],
                "gender": spk_meta["gender"],
                "is_vowel": label in FRENCH_VOWELS,
                "slot": slot,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(config_path: Path) -> None:
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    cfg = config["parse_corpus"]

    corpus_root = Path(cfg["corpus_root"])
    metadata_csv = Path(cfg["metadata"])
    rufrcorr_csv = Path(cfg["rufrcorr"])
    out_csv = Path(cfg["output"])

    debug = cfg.get("debug", {}) or {}
    debug_on = bool(debug.get("enable", False))
    n_spk = debug.get("n_speakers") if debug_on else None
    n_rep = debug.get("n_repetitions") if debug_on else None

    if debug_on:
        print(f"[DEBUG MODE] n_speakers={n_spk}, n_repetitions={n_rep}", file=sys.stderr)

    speakers = load_speakers_metadata(metadata_csv)
    if n_spk is not None:
        keep = select_speaker_subset(speakers, int(n_spk))
        speakers = {k: v for k, v in speakers.items() if k in keep}
        print(f"[DEBUG] selected speakers: {sorted(keep)}", file=sys.stderr)

    slot_lookup = load_slot_to_target(
        rufrcorr_csv,
        max_repetitions=int(n_rep) if n_rep is not None else None,
    )

    tg_root = corpus_root / "wav_et_textgrids" / "FRcorp_textgrids_only"
    assert tg_root.is_dir(), f"Missing: {tg_root}"

    all_rows: list[dict] = []
    n_files = 0
    n_skipped = 0
    for spk_dir in sorted(tg_root.iterdir()):
        if not spk_dir.is_dir():
            continue
        if spk_dir.name not in speakers:
            continue
        for tg_path in sorted(spk_dir.glob("*.TextGrid")):
            n_files += 1
            rows = process_textgrid(tg_path, speakers, slot_lookup)
            if not rows:
                n_skipped += 1
                continue
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")

    # --- Sanity report ---
    print(f"\n=== parse_corpus DONE ===")
    if debug_on:
        print(f"** DEBUG MODE: n_speakers={n_spk}, n_repetitions={n_rep} **")
    print(f"Files visited       : {n_files}")
    print(f"Files skipped       : {n_skipped} (distractors + warnings)")
    print(f"Total phoneme tokens: {len(df)}")
    print(f"Output              : {out_csv}")
    if len(df) == 0:
        print("No tokens emitted; check warnings above.")
        return
    print(f"\nSpeakers ({df.speaker_id.nunique()}): {sorted(df.speaker_id.unique())}")
    print(f"\nTokens per group (L1_status x gender):")
    print(df.groupby(["L1_status", "gender"]).size().to_string())
    print(f"\nTokens per phoneme (top 25):")
    print(df.phoneme.value_counts().head(25).to_string())
    print(f"\nUnique sentence_ids ({df.sentence_id.nunique()}):")
    print(", ".join(sorted(df.sentence_id.unique())))
    print(f"\nRepetitions per (speaker, sentence) - min/max:")
    rep_counts = df.groupby(["speaker_id", "sentence_id"]).repetition.nunique()
    print(f"  min={rep_counts.min()}, max={rep_counts.max()}, mean={rep_counts.mean():.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path, help="Path to config.yaml")
    args = p.parse_args()
    main(args.config)
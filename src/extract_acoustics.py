"""Extract acoustic features for phoneme tokens.

Input:
    data/interim/tokens.csv

Output:
    data/interim/features_acoustic.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth
import yaml
from parselmouth import praat


def read_config(config_path: Path) -> dict:
    """Read the YAML configuration file."""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def as_bool(value) -> bool:
    """Convert CSV values such as True/'True'/1 to a Python boolean."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def get_max_formant(gender: str, acoustic_config: dict) -> int:
    """Return the Praat max_formant parameter based on speaker gender."""
    gender = str(gender).upper()

    if gender == "F":
        return acoustic_config["formants"]["max_formant_female"]
    if gender == "M":
        return acoustic_config["formants"]["max_formant_male"]

    raise ValueError(f"Unknown gender value: {gender}")


def build_wav_path(row: pd.Series, project_root: Path, acoustic_config: dict) -> Path:
    """Find the WAV file corresponding to one token.

    The corpus filenames contain either _fra_ or _rus_ depending on the speaker/file.
    tokens.csv stores speaker_id and slot, but not this language-code part, so we
    search with a wildcard instead of hard-coding it.
    """
    speaker_id = str(row["speaker_id"])
    speaker_prefix = speaker_id.lower()
    slot = int(row["slot"])

    corpus_root = Path(acoustic_config["corpus_root"])
    wav_subdir = Path(acoustic_config["wav_subdir"])

    speaker_dir = project_root / corpus_root / wav_subdir / speaker_id
    pattern = f"{speaker_prefix}_*_list1_FRcorp{slot}.wav"
    candidates = sorted(speaker_dir.glob(pattern))

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        raise ValueError(f"Multiple WAV files found for {speaker_id}, slot {slot}: {candidates}")

    # Fallback path only for a readable missing-file error downstream.
    return speaker_dir / f"{speaker_prefix}_MISSING_list1_FRcorp{slot}.wav"


def clean_number(value: float | None) -> float:
    """Convert Praat missing values to np.nan."""
    if value is None:
        return np.nan
    if not np.isfinite(value):
        return np.nan
    return float(value)

def format_error(error: Exception) -> str:
    """Return a readable error message for debugging."""
    return f"{type(error).__name__}: {error}"

def get_formant_value(formant, formant_number: int, time_point: float) -> float:
    """Read one formant value at one time point using the Praat command."""
    value = praat.call(
        formant,
        "Get value at time",
        formant_number,
        time_point,
        "Hertz",
        "Linear",
    )
    return clean_number(value)


def extract_formants(
    segment: parselmouth.Sound,
    is_vowel: bool,
    max_formant: int,
    n_formants: int,
    long_vowel_threshold_s: float,
) -> dict:
    """Extract F1/F2 at midpoint, F3 for vowels, and trajectories for long vowels."""
    features = {
        "f1_hz": np.nan,
        "f2_hz": np.nan,
        "f3_hz": np.nan,
        "f1_25_hz": np.nan,
        "f2_25_hz": np.nan,
        "f3_25_hz": np.nan,
        "f1_75_hz": np.nan,
        "f2_75_hz": np.nan,
        "f3_75_hz": np.nan,
        "formants_ok": False,
        "formants_error": "",
    }

    try:
        duration = segment.get_total_duration()
        midpoint = duration / 2

        formant = praat.call(
            segment,
            "To Formant (burg)",
            0.0,  # automatic time step
            n_formants,
            max_formant,
            0.025,  # standard Praat window length
            50.0,  # standard pre-emphasis from 50 Hz
        )

        features["f1_hz"] = get_formant_value(formant, 1, midpoint)
        features["f2_hz"] = get_formant_value(formant, 2, midpoint)

        if is_vowel:
            features["f3_hz"] = get_formant_value(formant, 3, midpoint)

        is_long_vowel = is_vowel and duration > long_vowel_threshold_s

        if is_long_vowel:
            t25 = duration * 0.25
            t75 = duration * 0.75

            features["f1_25_hz"] = get_formant_value(formant, 1, t25)
            features["f2_25_hz"] = get_formant_value(formant, 2, t25)
            features["f3_25_hz"] = get_formant_value(formant, 3, t25)

            features["f1_75_hz"] = get_formant_value(formant, 1, t75)
            features["f2_75_hz"] = get_formant_value(formant, 2, t75)
            features["f3_75_hz"] = get_formant_value(formant, 3, t75)

        f1_ok = np.isfinite(features["f1_hz"])
        f2_ok = np.isfinite(features["f2_hz"])
        f3_ok = (not is_vowel) or np.isfinite(features["f3_hz"])

        features["formants_ok"] = bool(f1_ok and f2_ok and f3_ok)


    except Exception as error:
        features["formants_error"] = format_error(error)

    return features


def extract_f0(segment: parselmouth.Sound, pitch_floor_hz: int, pitch_ceiling_hz: int) -> dict:
    """Extract mean f0 from voiced frames only."""
    features = {
        "f0_mean_hz": np.nan,
        "f0_ok": False,
        "f0_error": "",
    }

    try:
        pitch = segment.to_pitch_ac(
            pitch_floor=pitch_floor_hz,
            pitch_ceiling=pitch_ceiling_hz,
        )

        values = pitch.selected_array["frequency"]
        voiced_values = values[values > 0]

        if len(voiced_values) > 0:
            features["f0_mean_hz"] = float(np.mean(voiced_values))
            features["f0_ok"] = True


    except Exception as error:
        features["f0_error"] = format_error(error)

    return features


def extract_spectral_centre_of_gravity(segment: parselmouth.Sound) -> dict:
    """Extract spectral centre of gravity from the segment spectrum."""
    features = {
        "scg_hz": np.nan,
        "scg_ok": False,
        "scg_error": "",
    }

    try:
        spectrum = segment.to_spectrum()
        scg = praat.call(spectrum, "Get centre of gravity", 2.0)

        features["scg_hz"] = clean_number(scg)
        features["scg_ok"] = np.isfinite(features["scg_hz"])


    except Exception as error:
        features["scg_error"] = format_error(error)

    return features


def extract_segment(sound: parselmouth.Sound, onset: float, offset: float) -> parselmouth.Sound:
    """Cut one phoneme segment from the full WAV file."""
    sound_duration = sound.get_total_duration()

    start = max(0.0, float(onset))
    end = min(float(offset), sound_duration)

    if end <= start:
        raise ValueError("Invalid segment boundaries")

    return sound.extract_part(
        from_time=start,
        to_time=end,
        preserve_times=False,
    )


def empty_feature_row(reason: str) -> dict:
    """Return an empty feature row when extraction is impossible."""
    return {
        "wav_path": "",
        "max_formant_hz": np.nan,
        "f1_hz": np.nan,
        "f2_hz": np.nan,
        "f3_hz": np.nan,
        "f1_25_hz": np.nan,
        "f2_25_hz": np.nan,
        "f3_25_hz": np.nan,
        "f1_75_hz": np.nan,
        "f2_75_hz": np.nan,
        "f3_75_hz": np.nan,
        "formants_ok": False,
        "formants_error": reason,
        "f0_mean_hz": np.nan,
        "f0_ok": False,
        "f0_error": reason,
        "scg_hz": np.nan,
        "scg_applicable": False,
        "scg_ok": False,
        "scg_error": reason,
    }


def extract_features_for_token(
    row: pd.Series,
    project_root: Path,
    acoustic_config: dict,
    sound_cache: dict[Path, parselmouth.Sound],
) -> dict:
    """Extract all acoustic features for one token row."""
    wav_path = build_wav_path(row, project_root, acoustic_config)

    if not wav_path.exists():
        return empty_feature_row("missing_wav")

    try:
        if wav_path not in sound_cache:
            sound_cache[wav_path] = parselmouth.Sound(str(wav_path))

        sound = sound_cache[wav_path]
        segment = extract_segment(sound, row["onset"], row["offset"])

    except Exception as error:
        return empty_feature_row(type(error).__name__)

    is_vowel = as_bool(row["is_vowel"])
    phoneme = str(row["phoneme_base"])

    max_formant = get_max_formant(row["gender"], acoustic_config)
    n_formants = acoustic_config["formants"]["n_formants"]
    long_vowel_threshold_s = acoustic_config["formants"]["long_vowel_threshold_s"]

    pitch_floor_hz = acoustic_config["pitch"]["floor_hz"]
    pitch_ceiling_hz = acoustic_config["pitch"]["ceiling_hz"]

    fricatives = set(acoustic_config["scg"]["fricatives"])
    scg_applicable = phoneme in fricatives

    features = {
        "wav_path": str(wav_path),
        "max_formant_hz": max_formant,
    }

    features.update(
        extract_formants(
            segment=segment,
            is_vowel=is_vowel,
            max_formant=max_formant,
            n_formants=n_formants,
            long_vowel_threshold_s=long_vowel_threshold_s,
        )
    )

    features.update(
        extract_f0(
            segment=segment,
            pitch_floor_hz=pitch_floor_hz,
            pitch_ceiling_hz=pitch_ceiling_hz,
        )
    )

    features["scg_applicable"] = scg_applicable

    if scg_applicable:
        features.update(extract_spectral_centre_of_gravity(segment))
    else:
        features.update(
            {
                "scg_hz": np.nan,
                "scg_ok": False,
                "scg_error": "",
            }
        )

    return features


def print_summary(features: pd.DataFrame) -> None:
    """Print a small extraction summary."""
    n_rows = len(features)

    print("\n=== Acoustic extraction summary ===")
    print(f"Rows: {n_rows}")
    print(f"F1 missing: {features['f1_hz'].isna().sum()} / {n_rows}")
    print(f"F2 missing: {features['f2_hz'].isna().sum()} / {n_rows}")
    print(f"F3 missing: {features['f3_hz'].isna().sum()} / {n_rows}")
    print(f"f0 missing: {features['f0_mean_hz'].isna().sum()} / {n_rows}")

    scg_subset = features[features["scg_applicable"]]
    if len(scg_subset) > 0:
        print(f"SCG missing among fricatives: {scg_subset['scg_hz'].isna().sum()} / {len(scg_subset)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    project_root = config_path.parent

    config = read_config(config_path)
    acoustic_config = config["extract_acoustics"]

    input_tokens = project_root / acoustic_config["input_tokens"]
    output_features = project_root / acoustic_config["output_features"]

    tokens = pd.read_csv(input_tokens)

    if acoustic_config["debug"]["enable"]:
        n_tokens = acoustic_config["debug"]["n_tokens"]
        tokens = tokens.head(n_tokens).copy()
        print(f"[DEBUG] Processing only the first {n_tokens} tokens.")

    sound_cache = {}
    feature_rows = []

    for index, row in tokens.iterrows():
        if index % 250 == 0:
            print(f"Processing token {index + 1} / {len(tokens)}")

        feature_row = extract_features_for_token(
            row=row,
            project_root=project_root,
            acoustic_config=acoustic_config,
            sound_cache=sound_cache,
        )

        feature_rows.append(feature_row)

    acoustic_features = pd.DataFrame(feature_rows)

    output = pd.concat(
        [
            tokens.reset_index(drop=True),
            acoustic_features.reset_index(drop=True),
        ],
        axis=1,
    )

    output["duration_ms"] = output["duration"] * 1000
    output_features.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_features, index=False, encoding="utf-8")

    print_summary(output)
    print(f"\nSaved: {output_features}")


if __name__ == "__main__":
    main()
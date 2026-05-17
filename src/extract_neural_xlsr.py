"""Extract XLS-R (wav2vec2) encoder hidden-state embeddings per phoneme token.

Reads tokens.csv + features_acoustic.csv (for wav_path), runs XLS-R forward
once per unique WAV, mean-pools hidden states over each token's frame span,
writes one .npz per requested layer.

Runs locally (CPU) or on Colab (GPU). Device auto-selected if device="auto".

Mirrors src/extract_neural_whisper.py. Key differences:
  - model: Wav2Vec2Model (single encoder, no encoder/decoder split)
  - no fixed-length padding (variable-length input)
  - frame rate is still 50 Hz: CNN total stride 320 over 16 kHz audio
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import soundfile as sf
import librosa


# --------------------------------------------------------------------------
# Constants (wav2vec2/XLS-R architecture, not tunable)
# --------------------------------------------------------------------------
SAMPLE_RATE = 16000        # XLS-R expects 16 kHz mono
FRAME_RATE_HZ = 50.0       # encoder output rate after CNN total stride 320
POOLING = "mean"           # per PDF Eq. (1)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda requested but not available, falling back to cpu")
        return "cpu"
    return requested


# --------------------------------------------------------------------------
# Audio loading
# --------------------------------------------------------------------------
def load_audio_16k(wav_path: Path) -> np.ndarray:
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


# --------------------------------------------------------------------------
# XLS-R forward + layer extraction
# --------------------------------------------------------------------------
def encode_wav(
    audio: np.ndarray,
    feature_extractor: Wav2Vec2FeatureExtractor,
    model: Wav2Vec2Model,
    device: str,
    layers: list[int],
) -> dict[int, np.ndarray]:
    """Return {layer: (T_frames, D)} as float32 numpy arrays."""
    inputs = feature_extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        outputs = model(
            input_values,
            output_hidden_states=True,
            return_dict=True,
        )
    # hidden_states: tuple of (num_layers + 1) tensors of shape (1, T, D).
    # Index 0 = output post-CNN feature extractor (pre-transformer),
    # 1..N = transformer layer outputs.
    hidden_states = outputs.hidden_states
    return {L: hidden_states[L].squeeze(0).cpu().numpy().astype(np.float32) for L in layers}


# --------------------------------------------------------------------------
# Pooling
# --------------------------------------------------------------------------
def pool_token(
    hidden: np.ndarray,
    onset_s: float,
    offset_s: float,
) -> tuple[np.ndarray, int]:
    """Mean-pool hidden states over [onset, offset). Fallback to nearest frame
    at midpoint if span is empty (very short phonemes < 1 frame = 20 ms).
    Returns (vector, n_frames_used).
    """
    T = hidden.shape[0]
    start = int(np.floor(onset_s * FRAME_RATE_HZ))
    end = int(np.ceil(offset_s * FRAME_RATE_HZ))
    start = max(0, min(start, T))
    end = max(0, min(end, T))

    if end > start:
        return hidden[start:end].mean(axis=0), end - start

    mid_s = 0.5 * (onset_s + offset_s)
    mid_idx = int(round(mid_s * FRAME_RATE_HZ))
    mid_idx = max(0, min(mid_idx, T - 1))
    return hidden[mid_idx], 1


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    nx = cfg["extract_neural_xlsr"]

    tokens_path = Path(nx["input_tokens"])
    feats_path = Path(nx["input_features_acoustic"])
    model_name = nx["model_name"]
    layers = list(nx["layers"])
    device = resolve_device(nx["device"])
    dtype_str = nx["embedding_dtype"]
    out_dir = Path(nx["output_dir"])
    out_prefix = nx["output_prefix"]
    out_paths = {L: out_dir / f"{out_prefix}_L{L:02d}.npz" for L in layers}
    debug = nx.get("debug", {"enable": False})

    # ---- Load tokens + wav_path ----
    tokens = pd.read_csv(tokens_path)
    feats = pd.read_csv(feats_path, usecols=["token_id", "wav_path"])
    tokens = tokens.merge(feats, on="token_id", how="left", validate="one_to_one")
    assert tokens["wav_path"].notna().all(), "Missing wav_path for some tokens"

    if debug.get("enable", False):
        n = int(debug["n_tokens"])
        tokens = tokens.head(n).reset_index(drop=True)
        print(f"[debug] truncated to first {n} tokens")

    print(f"[info] {len(tokens)} tokens, {tokens['wav_path'].nunique()} unique WAVs")
    print(f"[info] device = {device}")

    # ---- Load model ----
    print(f"[info] loading {model_name}")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name).to(device)
    model.eval()
    d_model = model.config.hidden_size
    print(f"[info] d_model = {d_model}")

    # ---- Allocate outputs ----
    N = len(tokens)
    out_embeddings = {L: np.zeros((N, d_model), dtype=np.float32) for L in layers}
    n_frames_used = np.zeros(N, dtype=np.int32)
    token_ids = tokens["token_id"].to_numpy(dtype=np.int64)
    token_id_to_row = {tid: i for i, tid in enumerate(token_ids)}

    # ---- Loop over unique WAVs ----
    np_dtype = np.float16 if dtype_str == "float16" else np.float32
    for wav_path_str, grp in tqdm(tokens.groupby("wav_path", sort=False),
                                   total=tokens["wav_path"].nunique()):
        audio = load_audio_16k(Path(wav_path_str))
        hidden_by_layer = encode_wav(audio, feature_extractor, model, device, layers)

        for row in grp.itertuples(index=False):
            i = token_id_to_row[int(row.token_id)]
            for L in layers:
                vec, n_used = pool_token(
                    hidden_by_layer[L],
                    onset_s=float(row.onset),
                    offset_s=float(row.offset),
                )
                out_embeddings[L][i] = vec
            n_frames_used[i] = n_used

    # ---- Write output ----
    for L in layers:
        out_p = out_paths[L]
        out_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_p,
            embeddings=out_embeddings[L].astype(np_dtype),
            token_ids=token_ids,
            n_frames_used=n_frames_used,
            layer=np.int32(L),
            model=np.array(model_name),
            pooling=np.array(POOLING),
            frame_rate_hz=np.float32(FRAME_RATE_HZ),
        )
        print(f"[done] wrote {out_p}  shape={out_embeddings[L].shape}  dtype={np_dtype}")

    print(f"[stats] n_frames_used  min={n_frames_used.min()}  "
          f"median={int(np.median(n_frames_used))}  max={n_frames_used.max()}")
    print(f"[stats] fallback (single-frame) tokens: "
          f"{(n_frames_used == 1).sum()} / {N}")


if __name__ == "__main__":
    main()

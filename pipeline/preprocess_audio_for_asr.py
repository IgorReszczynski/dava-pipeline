#!/usr/bin/env python3
"""
preprocess_audio_for_asr.py
===========================
Clean up an audio (or video) file so it transcribes well with WhisperX / pyannote.

Why this exists
---------------
Produced audio (radio features, podcasts with music beds, jingles/idents) makes
Whisper hallucinate boilerplate over the non-speech parts ("Untertitel im Auftrag
des ZDF...", station idents, etc.) and drives word-confidence down. Removing the
music and normalizing loudness fixes most of that.

Pipeline
--------
  1. Decode the input to PCM via ffmpeg (any format in).
  2. Source-separate with Demucs and keep ONLY the vocal stem (removes music/jingles).
  3. Downmix to mono.
  4. High-pass filter (kills sub-bass rumble / DC offset).
  5. OPTIONAL speech denoise (DeepFilterNet, else noisereduce). Off by default on
     purpose -- see the note in --denoise help.
  6. Loudness-normalize (EBU R128 via pyloudnorm; RMS fallback if it's missing).
  7. Resample to 16 kHz mono and write 16-bit PCM WAV -> ready for ASR.
  8. Optionally also write a full-rate "listenable" WAV for you to QC by ear.

Hard dependencies : numpy, scipy, ffmpeg (on PATH), and demucs (only if separating).
Optional          : pyloudnorm (loudness), noisereduce / deepfilternet (denoise),
                    soundfile (nicer WAV I/O; scipy is used as fallback).

Install
-------
  pip install demucs numpy scipy soundfile pyloudnorm noisereduce
  # optional, stronger speech denoiser:
  pip install deepfilternet
  # torch/torchaudio come in with demucs; match them to your CUDA if needed, e.g.
  # pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

Example
-------
  python preprocess_audio_for_asr.py \
      --input  /local/scratch/mchapa/data-pipeline/data/sample.mp3 \
      --output /local/scratch/mchapa/data-pipeline/data/sample_clean16k.wav \
      --device cuda --demucs-model htdemucs --save-listenable
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from math import gcd

import numpy as np


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def _log(msg: str) -> None:
    print(f"[preprocess] {msg}", flush=True)


def _to_float32(data: np.ndarray) -> np.ndarray:
    """Convert an integer/float PCM array to float32 in [-1, 1]."""
    if data.dtype == np.float32:
        return data
    if data.dtype == np.float64:
        return data.astype(np.float32)
    if data.dtype == np.int16:
        return (data.astype(np.float32)) / 32768.0
    if data.dtype == np.int32:
        return (data.astype(np.float32)) / 2147483648.0
    if data.dtype == np.uint8:
        return (data.astype(np.float32) - 128.0) / 128.0
    return data.astype(np.float32)


def read_wav(path: str):
    """Read a WAV file -> (float32 array [samples, channels], sample_rate)."""
    try:
        import soundfile as sf
        data, sr = sf.read(path, always_2d=True, dtype="float32")
        return data.astype(np.float32), int(sr)
    except Exception:
        from scipy.io import wavfile
        sr, data = wavfile.read(path)
        data = _to_float32(data)
        if data.ndim == 1:
            data = data[:, None]
        return data, int(sr)


def write_wav(path: str, data: np.ndarray, sr: int, subtype: str = "PCM_16") -> None:
    """Write [samples, channels] (or [samples]) float32 -> WAV."""
    if data.ndim == 1:
        data = data[:, None]
    try:
        import soundfile as sf
        sf.write(path, data, sr, subtype=subtype)
    except Exception:
        from scipy.io import wavfile
        x = np.clip(data, -1.0, 1.0)
        wavfile.write(path, sr, (x * 32767.0).astype(np.int16))


def decode_to_array(path: str, tmpdir: str):
    """Decode ANY audio/video file to float32 PCM via ffmpeg -> (array, sr)."""
    tmp = os.path.join(tmpdir, "decoded.wav")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-i", path, "-vn", "-acodec", "pcm_f32le", tmp]
    subprocess.run(cmd, check=True)
    return read_wav(tmp)


# --------------------------------------------------------------------------- #
# DSP building blocks (numpy + scipy only)
# --------------------------------------------------------------------------- #
def to_mono(wav: np.ndarray) -> np.ndarray:
    if wav.ndim == 1:
        return wav[:, None].astype(np.float32)
    if wav.shape[1] == 1:
        return wav.astype(np.float32)
    return wav.mean(axis=1, keepdims=True).astype(np.float32)


def highpass(mono: np.ndarray, sr: int, cutoff: float = 70.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth high-pass. cutoff <= 0 disables it."""
    if cutoff and cutoff > 0:
        from scipy.signal import butter, sosfiltfilt
        sos = butter(order, cutoff / (sr / 2.0), btype="highpass", output="sos")
        y = sosfiltfilt(sos, mono[:, 0]).astype(np.float32)
        return y[:, None]
    return mono


def resample_to(mono: np.ndarray, sr: int, target_sr: int):
    """High-quality polyphase resample. Returns (array, target_sr)."""
    if sr == target_sr:
        return mono, sr
    from scipy.signal import resample_poly
    g = gcd(int(sr), int(target_sr))
    up, down = int(target_sr) // g, int(sr) // g
    y = resample_poly(mono[:, 0], up, down).astype(np.float32)
    return y[:, None], target_sr


def normalize_loudness(mono: np.ndarray, sr: int,
                       target_lufs: float = -23.0, peak_ceiling: float = 0.97) -> np.ndarray:
    """EBU R128 loudness normalization (pyloudnorm); RMS fallback if unavailable.
    A hard peak ceiling is applied afterwards so the file never clips."""
    x = mono[:, 0].astype(np.float64)
    method = "RMS(-20 dBFS)"
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(x)
        if np.isfinite(loudness):
            x = pyln.normalize.loudness(x, loudness, target_lufs)
            method = f"LUFS({target_lufs})"
    except Exception:
        rms = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
        x = x * ((10.0 ** (-20.0 / 20.0)) / rms)  # ~ -20 dBFS RMS
    peak = float(np.max(np.abs(x)) + 1e-12)
    if peak > peak_ceiling:
        x = x * (peak_ceiling / peak)
    _log(f"  loudness: {method}, peak {min(peak, peak_ceiling):.3f}")
    return x.astype(np.float32)[:, None]


def denoise(mono: np.ndarray, sr: int, method: str):
    """Optional speech denoise. Returns (array, sr) -- sr may change (DFN=48 kHz)."""
    if method == "none":
        return mono, sr

    if method == "deepfilternet":
        try:
            from df.enhance import enhance, init_df
            import torch
            model, state, _ = init_df()
            df_sr = int(state.sr())
            x, _ = resample_to(mono, sr, df_sr)
            audio = torch.from_numpy(np.ascontiguousarray(x.T))  # [ch, samples]
            out = enhance(model, state, audio).cpu().numpy().T.astype(np.float32)
            _log(f"  denoise: DeepFilterNet @ {df_sr} Hz")
            return out, df_sr
        except Exception as e:
            _log(f"  DeepFilterNet unavailable ({e!r}); falling back to noisereduce")
            method = "noisereduce"

    if method == "noisereduce":
        try:
            import noisereduce as nr
            y = nr.reduce_noise(y=mono[:, 0], sr=sr, stationary=False, prop_decrease=0.75)
            _log("  denoise: noisereduce (non-stationary spectral gating)")
            return y.astype(np.float32)[:, None], sr
        except Exception as e:
            _log(f"  noisereduce unavailable ({e!r}); skipping denoise")

    return mono, sr


def separate_vocals(path: str, device: str, model_name: str):
    """Run Demucs and return the vocal stem: (array [samples, ch], sample_rate)."""
    from demucs.api import Separator
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"  Demucs '{model_name}' on {device} (slow step; downloads weights on first run)...")
    sep = Separator(model=model_name, device=device, progress=True)
    _origin, stems = sep.separate_audio_file(path)
    sr = int(sep.samplerate)
    if "vocals" not in stems:
        raise RuntimeError(f"model '{model_name}' has no 'vocals' stem; got {list(stems)}")
    vocals = stems["vocals"].cpu().numpy().T.astype(np.float32)  # [samples, ch]
    return vocals, sr


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Music removal + speech enhancement + ASR-friendly normalization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="Input audio/video file.")
    ap.add_argument("--output", default=None,
                    help="Output 16 kHz mono WAV. Default: <input>_clean16k.wav")
    ap.add_argument("--device", default="auto",
                    help="Torch device for Demucs: auto (GPU if available), cuda, cuda:0, "
                         "or cpu. Demucs is very slow on CPU; for clean recordings "
                         "consider --keep-music.")
    ap.add_argument("--demucs-model", default="htdemucs",
                    help="Demucs model (htdemucs, htdemucs_ft = better+slower, mdx_extra).")
    ap.add_argument("--keep-music", action="store_true",
                    help="Skip Demucs separation (only filter/normalize the raw audio).")
    ap.add_argument("--denoise", choices=["none", "noisereduce", "deepfilternet"],
                    default="none",
                    help="Extra denoise AFTER music removal. Leave 'none' unless the "
                         "vocal stem is still noisy: aggressive denoise adds artifacts "
                         "that can RAISE word error rate. 'deepfilternet' is best quality.")
    ap.add_argument("--highpass", type=float, default=70.0,
                    help="High-pass cutoff in Hz (0 disables).")
    ap.add_argument("--target-lufs", type=float, default=-23.0,
                    help="Integrated loudness target (EBU R128).")
    ap.add_argument("--asr-sr", type=int, default=16000,
                    help="Output sample rate for the ASR file.")
    ap.add_argument("--save-listenable", action="store_true",
                    help="Also write a full-rate WAV (<input>_enhanced.wav) to check by ear.")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        _log(f"ERROR: input not found: {args.input}")
        return 2

    stem = os.path.splitext(os.path.abspath(args.input))[0]
    out_asr = args.output or (stem + "_clean16k.wav")
    out_listen = stem + "_enhanced.wav"
    os.makedirs(os.path.dirname(os.path.abspath(out_asr)) or ".", exist_ok=True)

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        # 1 + 2 : get the speech signal ------------------------------------- #
        if args.keep_music:
            _log("Stage: decode (music kept, separation skipped)")
            sig, sr = decode_to_array(args.input, tmp)
        else:
            _log("Stage: source separation (keeping vocal stem)")
            sig, sr = separate_vocals(args.input, args.device, args.demucs_model)
        _log(f"  got {sig.shape[0]/sr:,.1f}s @ {sr} Hz, {sig.shape[1]} ch")

        # 3 : mono ---------------------------------------------------------- #
        sig = to_mono(sig)

        # 4 : high-pass ----------------------------------------------------- #
        _log(f"Stage: high-pass @ {args.highpass} Hz")
        sig = highpass(sig, sr, cutoff=args.highpass)

        # 5 : optional denoise --------------------------------------------- #
        _log(f"Stage: denoise ({args.denoise})")
        sig, sr = denoise(sig, sr, args.denoise)

        # 7 : ASR file = resample -> normalize -> write --------------------- #
        _log(f"Stage: resample -> {args.asr_sr} Hz + normalize (ASR output)")
        asr_sig, asr_sr = resample_to(sig, sr, args.asr_sr)
        asr_sig = normalize_loudness(asr_sig, asr_sr, target_lufs=args.target_lufs)
        write_wav(out_asr, asr_sig, asr_sr, subtype="PCM_16")
        _log(f"  wrote ASR file: {out_asr}")

        # 8 : optional full-rate QC file ----------------------------------- #
        if args.save_listenable:
            listen = normalize_loudness(sig, sr, target_lufs=args.target_lufs)
            write_wav(out_listen, listen, sr, subtype="PCM_24")
            _log(f"  wrote listenable file: {out_listen} ({sr} Hz)")

    _log(f"Done in {time.time() - t0:,.1f}s")
    print("\nNext, feed the cleaned file to your pipeline, e.g.:")
    print(f"  CUDA_VISIBLE_DEVICES=3 python process_single_file_pipeline_AG.py \\")
    print(f"      --episode_path {out_asr} \\")
    print(f"      --out_dir ./output --gpu_index 0 --language de \\")
    print(f"      --asr_backend openai --hf_token <YOUR_NEW_HF_TOKEN>")
    return 0


if __name__ == "__main__":
    sys.exit(main())

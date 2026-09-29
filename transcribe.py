#!/usr/bin/env python3
"""
transcribe.py — add the ASR / speech-to-text step to the dashboard.
================================================================================
The dashboard runs in the ``dava_env`` conda environment, which does NOT contain
WhisperX / torch / pyannote / Demucs — those live only in the ``pipeline`` env.
So instead of importing the ASR code (which would fail here, and would defeat the
reason the two environments are kept apart), this module *shells out* to the
``pipeline`` environment and runs the existing pipeline scripts as subprocesses,
streaming their log output into the UI. When it finishes it drops the resulting
``_whisperx.json`` into the same session keys the sidebar uses, so every other
tab lights up exactly as if the file had been uploaded.

Nothing about the pipeline changes: the same ``preprocess_audio_for_asr.py`` and
``process_single_file_pipeline_AG.py`` you run on the command line are what run
here.

Finding the pipeline interpreter (in order):
  1. the ``DAVA_PIPELINE_PYTHON`` environment variable, if set;
  2. a sibling conda env next to the running interpreter
     (…/envs/dava_env/bin/python  ->  …/envs/pipeline/bin/python);
  3. ``conda run -n pipeline python`` if conda is on PATH.
"""
from __future__ import annotations

import os
import sys
import shutil
import tempfile
import subprocess
from pathlib import Path

import streamlit as st

REPO = Path(__file__).resolve().parent
PIPELINE_DIR = REPO / "pipeline"
PREPROCESS = PIPELINE_DIR / "preprocess_audio_for_asr.py"
TRANSCRIBE = PIPELINE_DIR / "process_single_file_pipeline_AG.py"

AUDIO_EXT = ["wav", "mp3", "m4a", "flac", "ogg", "aac", "wma"]
VIDEO_EXT = ["mp4", "mov", "mkv", "webm", "avi"]


# --------------------------------------------------------------------------- #
# Locating the pipeline environment
# --------------------------------------------------------------------------- #
def _pipeline_python():
    override = os.environ.get("DAVA_PIPELINE_PYTHON")
    if override and Path(override).exists():
        return [override, "-u"]
    here = Path(sys.executable).resolve()          # …/envs/dava_env/bin/python
    sibling = here.parent.parent.parent / "pipeline" / "bin" / "python"
    if sibling.exists():
        return [str(sibling), "-u"]
    if shutil.which("conda"):
        return ["conda", "run", "--no-capture-output", "-n", "pipeline", "python", "-u"]
    return None


def _ffmpeg():
    here = Path(sys.executable).resolve()
    sibling = here.parent.parent.parent / "pipeline" / "bin" / "ffmpeg"
    if sibling.exists():
        return str(sibling)
    return shutil.which("ffmpeg")


# --------------------------------------------------------------------------- #
# Running a subprocess and streaming its output live
# --------------------------------------------------------------------------- #
def _run_streaming(cmd, cwd):
    """Run a command, stream stdout into the page, return (returncode, log)."""
    st.caption("$ " + " ".join(str(c) for c in cmd))
    box = st.empty()
    lines = []
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
    except FileNotFoundError as exc:
        st.error(f"Could not launch the pipeline: {exc}")
        return 1, str(exc)
    for line in proc.stdout:                        # streams line by line
        lines.append(line.rstrip("\n"))
        box.code("\n".join(lines[-40:]) or " ", language="text")
    proc.wait()
    return proc.returncode, "\n".join(lines)


def _newest_whisperx_json(root):
    hits = list(Path(root).rglob("*_whisperx.json"))
    return str(max(hits, key=lambda p: p.stat().st_mtime)) if hits else None


# --------------------------------------------------------------------------- #
# The tab
# --------------------------------------------------------------------------- #
def render_transcribe_ui() -> bool:
    """Render the Transcribe tab. Returns True if a transcript was just produced
    (the caller should then st.rerun() so the other tabs pick it up)."""
    banner = st.session_state.pop("_asr_banner", None)
    if banner:
        st.success("Transcript ready and loaded: **" + os.path.basename(banner) +
                   "**. Open the Overview tab, or press **▶️ Run analysis** in the sidebar "
                   "to add topics, sentiment, emotion and entities.")
        try:
            with open(st.session_state.get("jpath", banner), "rb") as _fh:
                st.download_button("⬇️ Download transcript JSON", _fh.read(),
                                   file_name=os.path.basename(banner),
                                   mime="application/json", key="dl_asr_json")
        except Exception:
            pass
        st.caption("You can also re-download it any time from the ⬇️ Export tab.")

    st.subheader("Create a transcript from audio or video")
    st.caption("Runs the ASR pipeline (Whisper for speech-to-text plus speaker "
               "diarization) in the `pipeline` environment, then loads the result here.")

    py = _pipeline_python()
    if py is None:
        st.error("Could not find the `pipeline` conda environment. Set the environment "
                 "variable `DAVA_PIPELINE_PYTHON` to its interpreter, for example "
                 "`~/miniconda3/envs/pipeline/bin/python`, then restart the dashboard.")
        return False
    if not TRANSCRIBE.exists():
        st.error(f"Transcription script not found at {TRANSCRIBE}.")
        return False
    with st.expander("Engine", expanded=False):
        st.write("Pipeline interpreter: `" + " ".join(py) + "`")
        st.caption("Override with the `DAVA_PIPELINE_PYTHON` environment variable if needed.")

    up = st.file_uploader("Audio or video file", type=AUDIO_EXT + VIDEO_EXT,
                          help="wav / mp3 / m4a / flac / ogg, or mp4 / mov / mkv / webm.")

    c1, c2, c3 = st.columns(3)
    lang = c1.selectbox("Language", ["auto", "de", "fr", "en"], 0,
                        help="Leave on auto to detect it automatically.")
    n_spk = c2.number_input("Speakers (0 = auto)", 0, 20, 0, 1,
                            help="Set this only if you know how many people speak.")
    device = c3.selectbox("Device", ["auto", "cuda", "cpu"], 0,
                          help="`auto` uses the GPU if the pipeline environment finds one, "
                               "otherwise the CPU. Pick `cpu` on a machine without an NVIDIA GPU "
                               "(including Apple Silicon Macs).")
    on_cpu = device == "cpu"

    c4, c5, c6 = st.columns(3)
    engines = ["openai", "transformers", "faster-whisper"]
    backend = c4.selectbox("ASR engine", engines, engines.index("faster-whisper" if on_cpu else "openai"),
                           help="`openai` is the OpenAI-Whisper engine (the GPU default). "
                                "`faster-whisper` is much faster on a CPU.")
    gpu = c5.number_input("GPU index", 0, 8, 0, 1, disabled=on_cpu)
    clean = c6.checkbox("Clean audio first (remove music and noise)", value=not on_cpu,
                        help="Uses Demucs to separate speech, then levels loudness. Recommended "
                             "on a GPU; very slow on a CPU, so leave it off for clean recordings.")

    hf = st.text_input("Hugging Face token (optional, for speaker labels)",
                       value=os.environ.get("HF_TOKEN", ""), type="password",
                       help="Needed to download the diarization models. Leave empty to "
                            "transcribe without speaker labels. Kept only in memory.")

    st.caption("Transcription can take several minutes on a GPU and much longer on a CPU. "
               "The dashboard is busy while it runs, which is normal.")

    go = st.button("▶️ Transcribe", type="primary", disabled=(up is None))
    if not go:
        return False
    if up is None:
        st.warning("Upload an audio or video file first.")
        return False
    if not hf:
        st.info("No Hugging Face token, so the transcript will have no speaker labels. "
                "To add them, paste a token above (huggingface.co, then Settings, then "
                "Access Tokens).")

    workdir = Path(tempfile.mkdtemp(prefix="dava_asr_"))
    raw = workdir / up.name
    raw.write_bytes(up.getbuffer())
    outdir = workdir / "output"
    outdir.mkdir(exist_ok=True)
    ext = raw.suffix.lower().lstrip(".")
    audio_for_asr = raw

    with st.status("Transcribing…", expanded=True) as status:
        # video -> extract a 16 kHz mono wav first (Demucs and Whisper both prefer audio)
        if ext in VIDEO_EXT:
            st.write("**Extracting audio from the video**")
            ff = _ffmpeg()
            if ff is None:
                status.update(label="ffmpeg not found", state="error")
                st.error("ffmpeg was not found, so the audio could not be extracted from "
                         "the video. Install it in the pipeline env: "
                         "`conda install -n pipeline -c conda-forge ffmpeg`.")
                return False
            wav = workdir / (raw.stem + ".wav")
            rc, _ = _run_streaming([ff, "-y", "-i", str(raw), "-vn",
                                    "-ac", "1", "-ar", "16000", str(wav)], REPO)
            if rc != 0 or not wav.exists():
                status.update(label="Audio extraction failed", state="error")
                st.error("Could not extract audio from the video. See the log above.")
                return False
            raw = wav
            audio_for_asr = wav

        # optional cleaning (Demucs + loudness normalisation) -> clean 16 kHz wav
        if clean:
            st.write("**Step 1 — cleaning the audio**")
            clean_wav = workdir / (raw.stem + "_clean16k.wav")
            demucs_device = f"cuda:{int(gpu)}" if device == "cuda" else device
            cmd = py + [str(PREPROCESS), "--input", str(raw), "--output", str(clean_wav),
                        "--device", demucs_device, "--demucs-model", "htdemucs"]
            rc, _ = _run_streaming(cmd, REPO)
            if rc != 0 or not clean_wav.exists():
                status.update(label="Cleaning failed", state="error")
                st.error("Audio cleaning failed (see the log above). You can also untick "
                         "‘Clean audio first’ and try again.")
                return False
            audio_for_asr = clean_wav

        # transcribe + diarize + align
        st.write("**Step 2 — transcribe, identify speakers, align**")
        cmd = py + [str(TRANSCRIBE), "--episode_path", str(audio_for_asr),
                    "--out_dir", str(outdir), "--device", device, "--gpu_index", str(int(gpu)),
                    "--asr_backend", backend]
        if hf:
            cmd += ["--hf_token", hf]
        if lang != "auto":
            cmd += ["--language", lang]
        if int(n_spk) > 0:
            cmd += ["--num_speakers", str(int(n_spk))]
        rc, _ = _run_streaming(cmd, REPO)
        if rc != 0:
            status.update(label="Transcription failed", state="error")
            st.error("Transcription failed. Check the log above. A common cause is a "
                     "Hugging Face token that is invalid or has not accepted the "
                     "diarization model's terms (leave it empty to skip speaker labels).")
            return False

        jpath = _newest_whisperx_json(outdir)
        if not jpath:
            status.update(label="No transcript produced", state="error")
            st.error("The pipeline finished but no `_whisperx.json` was found in the output.")
            return False
        status.update(label="Done", state="complete")

    # hand off to the rest of the dashboard, using the exact keys the sidebar sets
    st.session_state.update(workdir=str(workdir), jpath=jpath,
                            _src_name="asr:" + up.name, enriched=False,
                            summary=None, _asr_banner=jpath)
    return True

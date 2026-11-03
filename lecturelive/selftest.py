"""`Lecture Live.exe --selftest`: check every part the app needs and write selftest.txt next to
config.yaml. Exit code 0 when all checks pass."""
from __future__ import annotations

import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np


def run() -> int:
    from . import config
    lines, failed = [], False

    def check(name, fn):
        nonlocal failed
        t = time.perf_counter()
        try:
            detail = fn()
            lines.append(f"PASS  {name}: {detail}  ({time.perf_counter() - t:.1f} s)")
        except Exception:
            failed = True
            lines.append(f"FAIL  {name}:\n{traceback.format_exc()}")

    cfg = config.load()
    lines.append(f"config: {config.CONFIG_PATH}")

    def vad():
        from .vad import SileroVad
        return f"speech probability on silence {SileroVad()(np.zeros(512, np.float32)):.3f}"

    def mics():
        from .audio import list_input_devices
        devices, _ = list_input_devices()
        return ", ".join(d.name for d in devices)

    def whisper():
        from .engine import WhisperEngine
        eng = WhisperEngine(cfg, lambda _l: None)
        try:
            tr = eng.transcribe(np.zeros(16000, np.float32), "en", "")
            return f"{eng.gpu}; warm-up decode {tr.inference_s:.1f} s"
        finally:
            eng.close()

    def speakers():
        from .speakers import SpeakerTracker
        e = SpeakerTracker(str(cfg.speaker_model_path), cfg.speaker_threshold).embed(
            np.random.default_rng(0).normal(0, 0.05, 32000).astype(np.float32))
        return f"embedding of {len(e)} values"

    def llama():
        r = subprocess.run([str(cfg.llama_server_path), "--list-devices"], capture_output=True, text=True,
                           creationflags=subprocess.CREATE_NO_WINDOW, timeout=60)
        devices = [ln.strip() for ln in (r.stdout + r.stderr).splitlines() if "Vulkan" in ln and ":" in ln]
        if r.returncode != 0 or not devices:
            raise RuntimeError(f"exit {r.returncode}: {r.stdout}{r.stderr}")
        model = cfg.llm_model_path
        if not model.is_file():
            raise FileNotFoundError(model)
        return f"{devices[-1]}; model {model.name}"

    def mp3():
        import soundfile as sf
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.mp3"
            sf.write(str(p), np.zeros(16000, np.float32), 16000, format="MP3", subtype="MPEG_LAYER_III")
            return f"{p.stat().st_size} bytes"

    def qt():
        from PySide6.QtGui import QGuiApplication
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(["selftest", "-platform", "offscreen"])
        return f"Qt {QGuiApplication.platformName()}"

    for name, fn in (("silero vad", vad), ("microphones", mics), ("whisper on the GPU", whisper),
                     ("speaker model", speakers), ("llama.cpp + Qwen", llama), ("mp3 export", mp3), ("qt", qt)):
        check(name, fn)
    (config.CONFIG_PATH.parent / "selftest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if failed else 0

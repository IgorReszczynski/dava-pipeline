"""Whisper on the GPU: whisper.cpp built with Vulkan, through pywhispercpp's C bindings.

pywhispercpp's high-level Model class is not used: it downloads a model on its own when
the file is missing, strips whitespace from every segment and ignores whisper_full's
return code. This module calls the bindings directly, and everything that goes wrong is
raised:

* the model file is missing,
* whisper.cpp logs an error while loading,
* whisper.cpp starts without a GPU (it would otherwise run on the CPU without saying so),
* whisper_full returns non-zero,
* a language code whisper.cpp does not know (the binding would quietly auto-detect),
* text that is not valid UTF-8.

Segment text is returned exactly as whisper.cpp produced it.
"""
from __future__ import annotations

import codecs
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import _pywhispercpp as pw

from .config import Config

LOG_ERROR = 4  # GGML_LOG_LEVEL_ERROR


@dataclass
class Segment:
    start: float  # seconds from the start of the chunk
    end: float
    text: str


@dataclass
class Transcription:
    segments: list[Segment]
    text: str
    language: str
    inference_s: float


def language_codes() -> list[str]:
    return [pw.whisper_lang_str(i) for i in range(pw.whisper_lang_max_id() + 1)]


class WhisperEngine:
    def __init__(self, cfg: Config, log: Callable[[str], None]):
        self._log = log
        self._lock = threading.Lock()
        self._lines: list[tuple[int, str]] = []
        self._ctx = None
        pw.whisper_log_set(self._on_log)
        try:
            self._init(cfg)
        except BaseException:
            # unhook the Python callback, or ggml logging at process exit calls into a dead interpreter
            self.close()
            raise

    def _init(self, cfg: Config):
        path = cfg.model_path
        if not path.is_file():
            raise FileNotFoundError(f"Whisper model not found: {path}\n"
                                    "Run setup.ps1 to download it, or set asr_model / models_dir in config.yaml.")

        cparams = pw.whisper_context_default_params()
        cparams.use_gpu = True
        cparams.gpu_device = cfg.gpu_device
        cparams.flash_attn = cfg.flash_attn

        mark = len(self._lines)
        ctx = pw.whisper_init_from_file_with_params(str(path), cparams)
        init = self._lines[mark:]
        init_text = "".join(t for _, t in init)
        errors = [t.strip() for level, t in init if level == LOG_ERROR]
        if errors or "compute buffer (decode)" not in init_text:
            hint = (f"\n\nThe model file is incomplete ({path.stat().st_size / 1e6:.0f} MB on disk). "
                    "Run setup.ps1 again to finish the download." if "not all tensors loaded" in init_text else "")
            raise RuntimeError(f"whisper.cpp failed to load {path.name}:\n" + ("\n".join(errors) or init_text) + hint)
        self._ctx = ctx

        if "no GPU found" in init_text:
            raise RuntimeError("whisper.cpp found no GPU, so it would run on the CPU.\n"
                               "Check that the Vulkan driver works (run `vulkaninfo --summary`) and that "
                               f"gpu_device={cfg.gpu_device} in config.yaml is a valid Vulkan device.\n\n" + init_text)
        used = re.search(r"whisper_backend_init_gpu: using (\S+) backend", init_text)
        if not used:
            raise RuntimeError("whisper.cpp did not report a GPU backend:\n\n" + init_text)
        backend = used.group(1)
        if not backend.lower().startswith("vulkan"):
            raise RuntimeError(f"whisper.cpp picked the {backend} backend, not Vulkan:\n\n" + init_text)
        dev = re.search(r"ggml_vulkan: \d+ = ([^|\n]+)", init_text)
        self.gpu = f"{backend}: {dev.group(1).strip()}" if dev else backend
        self.model_name = cfg.asr_model
        self.system_info = pw.whisper_print_system_info()

        strategy = (pw.whisper_sampling_strategy.WHISPER_SAMPLING_BEAM_SEARCH if cfg.beam_size > 1
                    else pw.whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY)
        p = pw.whisper_full_default_params(strategy)
        p.n_threads = cfg.n_threads
        p.translate = False
        p.no_context = True          # the prompt is managed by the caller (see Session)
        p.single_segment = False
        p.print_progress = False
        p.print_realtime = False
        p.print_timestamps = False
        p.print_special = False
        p.no_speech_thold = cfg.no_speech_thold
        if cfg.beam_size > 1:
            p.beam_search = {"beam_size": cfg.beam_size, "patience": -1.0}
        self._params = p
        self._codes = set(language_codes())

    def _on_log(self, level: int, text: str):
        self._lines.append((level, text))
        self._log(f"[whisper.cpp] {text.rstrip()}")

    def transcribe(self, audio: np.ndarray, language: str, prompt: str,
                   abort: Callable[[], bool] | None = None) -> Transcription | None:
        """language: a whisper language code, or "" to auto-detect.

        abort: polled by whisper.cpp between the encoder and decoder steps; when it returns
        True the decode stops and None is returned. Any other failure raises."""
        if language and language not in self._codes:
            raise ValueError(f"whisper.cpp does not know the language code {language!r}")
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        with self._lock:
            p = self._params
            p.language = language
            if p.language != language:
                raise RuntimeError(f"whisper.cpp did not accept language {language!r} (got {p.language!r})")
            p.initial_prompt = prompt
            aborted = False

            def check() -> bool:
                nonlocal aborted
                aborted = aborted or bool(abort())
                return aborted

            pw.assign_abort_callback(p, check if abort is not None else None)
            mark = len(self._lines)
            t0 = time.perf_counter()
            rc = pw.whisper_full(self._ctx, p, audio, audio.size)
            elapsed = time.perf_counter() - t0
            if rc != 0:
                if aborted:
                    return None
                log = "".join(t for _, t in self._lines[mark:])
                raise RuntimeError(f"whisper_full failed with code {rc}.\n{log}")

            decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
            segments = []
            for i in range(pw.whisper_full_n_segments(self._ctx)):
                # whisper.cpp can end a segment inside a multi-byte character; the
                # incremental decoder carries the partial bytes into the next segment.
                text = decoder.decode(pw.whisper_full_get_segment_text(self._ctx, i))
                segments.append(Segment(pw.whisper_full_get_segment_t0(self._ctx, i) / 100,
                                        pw.whisper_full_get_segment_t1(self._ctx, i) / 100, text))
            tail = decoder.decode(b"", final=True)  # raises on a truncated character
            if tail:
                segments[-1].text += tail
            lang = pw.whisper_lang_str(pw.whisper_full_lang_id(self._ctx))
        return Transcription(segments, "".join(s.text for s in segments), lang, elapsed)

    def close(self):
        if self._ctx is not None:
            pw.whisper_free(self._ctx)
            self._ctx = None
        pw.whisper_log_set(None)

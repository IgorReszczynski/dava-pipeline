"""Load and save config.yaml.

Every key must be present and of the right type; a missing, unknown or mistyped key
is an error. Saving rewrites only the values on the matching lines, so the comments
in the file survive.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

FROZEN = getattr(sys, "frozen", False)      # running as the packaged Lecture Live.exe
REPO = Path(__file__).resolve().parent.parent
# The packaged app keeps config.yaml (and with it models/ and modules.json) next to the .exe.
APP_DIR = Path(sys.executable).resolve().parent if FROZEN else REPO
CONFIG_PATH = APP_DIR / "config.yaml"


def resolve(path: str) -> Path:
    """A path from config.yaml: ~ is the home folder; relative paths are relative to config.yaml."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else (CONFIG_PATH.parent / p)

SEGMENTATION_MODES = ("vad", "continuous")
COMPUTE_PROFILES = ("accuracy", "balanced", "saver")


@dataclass
class Config:
    # ASR (Whisper)
    asr_backend: str
    asr_model: str
    lang: str
    models_dir: str
    gpu_device: int
    flash_attn: bool
    n_threads: int
    beam_size: int
    no_speech_thold: float
    use_context: bool
    initial_prompt: str
    # audio input
    input_device: str
    raw_capture: bool
    input_gain_db: float
    agc: bool
    agc_target_dbfs: float
    agc_max_gain_db: float
    highpass_hz: float
    # segmentation
    segmentation: str
    vad_threshold: float
    vad_min_silence_ms: int
    vad_speech_pad_ms: int
    max_chunk_s: float
    continuous_chunk_s: float
    compute_profile: str
    # speakers
    speakers: bool
    speaker_model: str
    speaker_threshold: float
    # output
    output_dir: str
    save_audio: bool
    # obsidian notes
    obsidian_vault: str
    obsidian_template: str
    obsidian_modules: str
    obsidian_attachments: str
    llm_model: str
    llama_server: str
    llm_context: int
    # window
    panel_width: float
    background_opacity: float

    @property
    def models_path(self) -> Path:
        return resolve(self.models_dir).resolve()

    @property
    def model_path(self) -> Path:
        return self.models_path / f"ggml-{self.asr_model}.bin"

    @property
    def speaker_model_path(self) -> Path:
        return self.models_path / self.speaker_model

    @property
    def vault_path(self) -> Path | None:
        return Path(self.obsidian_vault) if self.obsidian_vault else None

    @property
    def llm_model_path(self) -> Path:
        return self.models_path / self.llm_model

    @property
    def llama_server_path(self) -> Path:
        return resolve(self.llama_server)

    @property
    def output_path(self) -> Path:
        return resolve(self.output_dir).resolve()


def _check_type(key: str, value, expected: type):
    # YAML reads 1.0 as float and 1 as int; accept an int where a float is expected.
    if expected is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if expected is int and isinstance(value, bool):
        raise TypeError(f"config.yaml: '{key}' must be an integer, got {value!r}")
    if not isinstance(value, expected):
        raise TypeError(f"config.yaml: '{key}' must be {expected.__name__}, got {value!r}")
    return value


def load(path: Path = CONFIG_PATH) -> Config:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} does not contain a mapping of settings")

    expected = {f.name: f.type for f in fields(Config)}
    missing = sorted(set(expected) - set(raw))
    unknown = sorted(set(raw) - set(expected))
    if missing:
        raise KeyError(f"config.yaml is missing: {', '.join(missing)}")
    if unknown:
        raise KeyError(f"config.yaml has unknown keys: {', '.join(unknown)}")

    types = {"str": str, "int": int, "float": float, "bool": bool}
    values = {k: _check_type(k, raw[k], types[t]) for k, t in expected.items()}
    cfg = Config(**values)

    if cfg.asr_backend != "whisper.cpp":
        raise ValueError(f"config.yaml: asr_backend must be 'whisper.cpp', got {cfg.asr_backend!r}")
    if cfg.segmentation not in SEGMENTATION_MODES:
        raise ValueError(f"config.yaml: segmentation must be one of {SEGMENTATION_MODES}, got {cfg.segmentation!r}")
    if cfg.beam_size < 1:
        raise ValueError("config.yaml: beam_size must be 1 or more")
    if cfg.compute_profile not in COMPUTE_PROFILES:
        raise ValueError(f"config.yaml: compute_profile must be one of {COMPUTE_PROFILES}, got {cfg.compute_profile!r}")
    for key in ("panel_width", "background_opacity"):
        if not 0 < getattr(cfg, key) <= 1:
            raise ValueError(f"config.yaml: {key} must be above 0 and at most 1")
    return cfg


def _yaml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)  # a JSON string is a valid YAML string
    return repr(value)


def save(updates: dict, path: Path = CONFIG_PATH) -> None:
    """Write new values for existing keys, keeping every comment and the layout."""
    known = {f.name for f in fields(Config)}
    text = path.read_text(encoding="utf-8")
    for key, value in updates.items():
        if key not in known:
            raise KeyError(f"not a config key: {key}")
        # key: value   # comment   -> keep the padding before the comment
        pattern = re.compile(rf'^({re.escape(key)}:\s*)("(?:[^"\\]|\\.)*"|[^\s#]*)(.*)$', re.M)
        text, n = pattern.subn(lambda m: m.group(1) + _yaml_value(value) + m.group(3), text, count=1)
        if n != 1:
            raise KeyError(f"config.yaml has no line for '{key}'")
    path.write_text(text, encoding="utf-8")

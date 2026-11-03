"""The model files, and downloading the missing ones (first start of the packaged app).

A download goes to "<file>.part", resumes where it stopped, retries after dropped connections,
and is renamed to its real name only when its size matches the server's. So a model file that
exists is complete, and the app needs no network to start once the models are there.
"""
from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import Config

RETRIES = 30


@dataclass
class ModelFile:
    what: str
    path: Path
    url: str


def model_files(cfg: Config) -> list[ModelFile]:
    files = [ModelFile("Whisper speech recognition", cfg.model_path,
                       f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{cfg.model_path.name}")]
    if cfg.speakers:
        files.append(ModelFile("Speaker labels", cfg.speaker_model_path,
                               "https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/resolve/main/"
                               "voxceleb_resnet34_LM.onnx"))
    if cfg.obsidian_vault:
        files.append(ModelFile("Qwen for Obsidian notes", cfg.llm_model_path,
                               f"https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/{cfg.llm_model}"))
    return files


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        size = r.headers.get("Content-Length")
    if size is None:
        raise RuntimeError(f"The server did not say how big {url} is.")
    return int(size)


def absent(cfg: Config) -> list[ModelFile]:
    return [f for f in model_files(cfg) if not f.path.is_file()]


def download(f: ModelFile, total: int, progress: Callable[[int], None]):
    f.path.parent.mkdir(parents=True, exist_ok=True)
    part = f.path.with_name(f.path.name + ".part")
    for attempt in range(1, RETRIES + 1):
        have = part.stat().st_size if part.exists() else 0
        if have == total:
            break
        if have > total:
            raise RuntimeError(f"{part} is larger than the file on the server; delete it and try again.")
        req = urllib.request.Request(f.url, headers={"Range": f"bytes={have}-"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if r.status == 206 else "wb") as out:
                if r.status != 206:
                    have = 0
                while chunk := r.read(1 << 20):
                    out.write(chunk)
                    have += len(chunk)
                    progress(have)
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            if attempt == RETRIES:
                raise RuntimeError(f"Download of {f.path.name} kept failing: {exc}") from exc
            time.sleep(5)
    if part.stat().st_size != total:
        raise RuntimeError(f"Download of {f.path.name} is incomplete ({part.stat().st_size} of {total} bytes).")
    part.replace(f.path)

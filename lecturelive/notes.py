"""Obsidian lecture notes. After a lecture, Qwen (llama.cpp on the Vulkan GPU) fills the
sections of the vault's lecture template from the transcript; the note, a transcript note
and the recording are written into the module's folder in the vault.

The template decides the structure: its front matter, title line and "## " sections are kept
as they are, and the text under each heading (for example "1.\n2.\n3.") is shown to the model
as the format to follow. The model's answer is held to a JSON schema with one field per
section, so every section of the template is always filled. Anything that goes wrong raises.
"""
from __future__ import annotations

import json
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from .config import Config
from .modules import Module

FILE_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f#^\[\]]')   # Windows + Obsidian link syntax
OUTPUT_TOKENS = 3000        # room kept for the model's answer
START_TIMEOUT_S = 300

SYSTEM = ("You write lecture notes into a university student's Obsidian vault. Use only what is said in "
          "the transcript and do not add facts the lecturer did not say. If the lecture says nothing that "
          "fits a section (for example about exams), write that in one short line instead of guessing. "
          "Write in the language of the lecture. Markdown is fine inside a section; do not repeat the "
          "section heading. The transcript comes from speech recognition, so fix obvious mishearings only "
          "where the meaning is clear.")


@dataclass
class Template:
    front: list[str]                    # front-matter lines, without the --- markers
    body: list[str]
    sections: list[tuple[str, str]]     # (heading, the format shown under it)


def parse_template(text: str) -> Template:
    lines = text.splitlines()
    front: list[str] = []
    start = 0
    if lines and lines[0].strip() == "---":
        end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
        if end is None:
            raise ValueError("The lecture template's front matter has no closing ---.")
        front, start = lines[1:end], end + 1
    body = lines[start:]
    sections: list[tuple[str, str]] = []
    heading, hint = None, []
    for ln in body:
        if ln.startswith("## "):
            if heading is not None:
                sections.append((heading, "\n".join(hint).strip()))
            heading, hint = ln[3:].strip(), []
        elif heading is not None:
            hint.append(ln)
    if heading is not None:
        sections.append((heading, "\n".join(hint).strip()))
    if not sections:
        raise ValueError("The lecture template has no '## ' sections to fill in.")
    if len({h for h, _ in sections}) != len(sections):
        raise ValueError("The lecture template has two sections with the same heading.")
    return Template(front, body, sections)


def fill_placeholders(line: str, title: str, when: datetime) -> str:
    """Obsidian core-template placeholders: {{title}}, {{date}}, {{time}}, {{date:YYYY-MM-DD}}."""
    def moment(fmt: str) -> str:
        for token, py in (("YYYY", "%Y"), ("MM", "%m"), ("DD", "%d"), ("HH", "%H"), ("mm", "%M")):
            fmt = fmt.replace(token, py)
        return when.strftime(fmt)

    line = re.sub(r"\{\{date:([^}]+)\}\}", lambda m: moment(m.group(1)), line)
    line = re.sub(r"\{\{time:([^}]+)\}\}", lambda m: moment(m.group(1)), line)
    return (line.replace("{{title}}", title).replace("{{date}}", when.strftime("%Y-%m-%d"))
            .replace("{{time}}", when.strftime("%H:%M")))


def render(tpl: Template, title: str, sections: dict[str, str], module: str, when: datetime,
           footer: list[str]) -> str:
    out = []
    if tpl.front:
        out.append("---")
        for ln in tpl.front:
            key, sep, value = ln.partition(":")
            if sep and key.strip() == "module" and not value.strip():
                ln = f"{key}: {module}"
            out.append(fill_placeholders(ln, title, when))
        out.append("---")
    skipping = False
    for ln in tpl.body:
        if ln.startswith("## "):
            heading = ln[3:].strip()
            out += [ln, sections[heading].strip(), ""]
            skipping = True          # the template's format lines are replaced by the content
            continue
        if not skipping:
            out.append(fill_placeholders(ln, title, when))
    return "\n".join(out + footer).rstrip() + "\n"


def safe_name(text: str) -> str:
    name = re.sub(r"\s+", " ", FILE_UNSAFE.sub("", text)).strip().rstrip(".")
    return name[:90] or "Lecture"


def unique(path: Path) -> Path:
    if not path.exists():
        return path
    for n in range(2, 1000):
        p = path.with_name(f"{path.stem} ({n}){path.suffix}")
        if not p.exists():
            return p
    raise FileExistsError(f"Too many files named like {path}")


class LlamaServer:
    """llama-server (llama.cpp, Vulkan) running Qwen on 127.0.0.1 for the length of a `with`."""

    def __init__(self, exe: Path, model: Path, ctx: int, log_path: Path):
        for p, what in ((exe, "llama-server"), (model, "Qwen model")):
            if not p.is_file():
                raise FileNotFoundError(f"{what} not found: {p}\nRun setup.ps1 to install it.")
        self.exe, self.model, self.ctx, self.log_path = exe, model, ctx, log_path

    def __enter__(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(
            # f16 KV cache: with a quantized cache the Vulkan backend runs flash attention on the CPU
            [str(self.exe), "-m", str(self.model), "--host", "127.0.0.1", "--port", str(self.port),
             "-c", str(self.ctx), "-ngl", "999", "-fa", "on", "--no-webui", "--jinja"],
            stdout=self.log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        deadline = time.monotonic() + START_TIMEOUT_S
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.proc.returncode} while starting:\n"
                                   + self._log_tail())
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        return self
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(f"llama-server did not become ready in {START_TIMEOUT_S} s:\n" + self._log_tail())
            time.sleep(0.5)

    def __exit__(self, *exc):
        self.proc.terminate()
        try:
            self.proc.wait(15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.log.close()

    def _log_tail(self) -> str:
        self.log.flush()
        return "\n".join(self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:])

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", json.dumps(payload).encode("utf-8"),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"llama-server {path} failed ({e.code}): {e.read().decode('utf-8', 'replace')}") from e

    def tokens(self, text: str) -> int:
        return len(self._post("/tokenize", {"content": text})["tokens"])

    def chat(self, user: str, schema: dict | None = None, max_tokens: int = OUTPUT_TOKENS) -> str:
        payload = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                   "temperature": 0.3, "top_p": 0.8, "max_tokens": max_tokens}
        if schema is not None:
            payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "note", "schema": schema}}
        reply = self._post("/v1/chat/completions", payload)
        choice = reply["choices"][0]
        if choice.get("finish_reason") == "length":
            raise RuntimeError(f"Qwen ran out of room ({max_tokens} tokens) before finishing its answer.")
        return choice["message"]["content"]


def summarise(server: LlamaServer, transcript: str, module: str, when: datetime, tpl: Template,
              progress: Callable[[str], None]) -> tuple[str, dict[str, str]]:
    headings = [h for h, _ in tpl.sections]
    schema = {"type": "object", "required": ["title", "sections"], "additionalProperties": False,
              "properties": {
                  "title": {"type": "string"},
                  "sections": {"type": "object", "required": headings, "additionalProperties": False,
                               "properties": {h: {"type": "string"} for h in headings}}}}
    guide = "\n\n".join(f"## {h}\n{hint or '(free text)'}" for h, hint in tpl.sections)
    task = (f"Module: {module}\nDate: {when:%Y-%m-%d}\n\nFill in every section of this lecture-note template. "
            "Under each heading the template shows the format to keep (for example a numbered list):\n\n"
            f"{guide}\n\nCover the key points and the most relevant takeaways of the lecture. "
            "Give the note a short title that names the lecture's main topic.")

    budget = server.ctx - OUTPUT_TOKENS - server.tokens(SYSTEM + task) - 200
    n = server.tokens(transcript)
    if n <= budget:
        material = f"Transcript ([hh:mm:ss] timestamps, speaker labels where known):\n\n{transcript}"
    else:  # too long for one pass: notes on each part first, then the note from those
        lines = transcript.splitlines()
        per_part = max(1, int(len(lines) * (budget * 0.8) / n))
        parts = ["\n".join(lines[i:i + per_part]) for i in range(0, len(lines), per_part)]
        notes = []
        for i, part in enumerate(parts, 1):
            progress(f"Qwen: reading part {i} of {len(parts)}")
            notes.append(server.chat(
                f"This is part {i} of {len(parts)} of the transcript of a {module} lecture. Write detailed "
                f"notes of everything taught in this part: definitions, methods, examples, numbers, "
                f"questions from the room and what the lecturer stressed.\n\n{part}", max_tokens=1500))
        material = "Notes taken from consecutive parts of the lecture:\n\n" + "\n\n".join(
            f"Part {i}:\n{t}" for i, t in enumerate(notes, 1))
    progress("Qwen: writing the note")
    data = json.loads(server.chat(f"{task}\n\n{material}", schema=schema))
    return data["title"].strip(), data["sections"]


def write_note(cfg: Config, module: Module, session: Path, progress: Callable[[str], None]) -> Path:
    vault = cfg.vault_path
    if vault is None:
        raise ValueError("No Obsidian vault is set (obsidian_vault in config.yaml).")
    if not vault.is_dir():
        raise FileNotFoundError(f"The Obsidian vault is not reachable: {vault}\n(Is Google Drive running?)")
    if module.vault_path is None:
        raise ValueError(f"The module “{module.name}” has no Obsidian folder.")
    template_path = vault / cfg.obsidian_template
    if not template_path.is_file():
        raise FileNotFoundError(f"Lecture template not found: {template_path}")
    tpl = parse_template(template_path.read_text(encoding="utf-8"))

    transcript = (session / "transcript.txt").read_text(encoding="utf-8").strip()
    if not transcript:
        raise ValueError("The transcript is empty, so there is nothing to summarise.")
    meta = json.loads((session / "session.json").read_text(encoding="utf-8"))
    when = datetime.fromisoformat(meta["started"])

    progress("Starting Qwen on the GPU")
    with LlamaServer(cfg.llama_server_path, cfg.llm_model_path, cfg.llm_context, session / "llm_log.txt") as srv:
        title, sections = summarise(srv, transcript, module.name, when, tpl, progress)

    progress("Saving to Obsidian")
    base = f"{when:%Y-%m-%d} – {safe_name(title)}"
    folder = module.vault_path
    folder.mkdir(parents=True, exist_ok=True)
    note = unique(folder / f"{base}.md")
    tnote = unique(folder / "Transcripts" / f"{note.stem} – transcript.md")
    tnote.parent.mkdir(exist_ok=True)

    footer = ["", "## Recording"]
    audio = session / "audio.flac"
    if audio.is_file():
        attach = vault / cfg.obsidian_attachments
        attach.mkdir(parents=True, exist_ok=True)
        dest = unique(attach / f"{note.stem}.flac")
        shutil.copyfile(audio, dest)
        footer.append(f"![[{dest.name}]]")
    footer.append(f"Full transcript: [[{tnote.stem}]]")

    tnote.write_text("\n".join([
        "---", f"module: {module.name}", f"date: {when:%Y-%m-%d}", f'lecture: "[[{note.stem}]]"', "---",
        f"# {title} – transcript", "", transcript, ""]), encoding="utf-8")
    note.write_text(render(tpl, title, sections, module.name, when, footer), encoding="utf-8")
    (session / "obsidian.json").write_text(json.dumps(
        {"note": str(note), "transcript": str(tnote), "title": title}, indent=2, ensure_ascii=False), encoding="utf-8")
    return note

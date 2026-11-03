"""Lecture modules (courses). Each module has a name and a folder; its sessions are saved there.

The list lives in modules.json next to config.yaml. Removing a module only takes it off the
list; its folder and recordings are never deleted.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import CONFIG_PATH

STORE_PATH = CONFIG_PATH.parent / "modules.json"
INVALID = set('<>:"/\\|?*')


@dataclass
class Module:
    name: str
    folder: str

    @property
    def path(self) -> Path:
        return Path(self.folder)


def check_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("Give the module a name.")
    bad = sorted(set(name) & INVALID)
    if bad:
        raise ValueError(f"A module name can't contain {' '.join(bad)} (it becomes a folder name).")
    if name.endswith("."):
        raise ValueError("A module name can't end with a dot (it becomes a folder name).")
    return name


class ModuleStore:
    def __init__(self, default_root: Path, path: Path = STORE_PATH):
        self.path = path
        self.default_root = default_root
        self.modules: list[Module] = []
        self.last: str | None = None
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.modules = [Module(m["name"], m["folder"]) for m in data["modules"]]
            self.last = data.get("last")

    def save(self):
        data = {"modules": [asdict(m) for m in self.modules], "last": self.last}
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def get(self, name: str) -> Module:
        for m in self.modules:
            if m.name == name:
                return m
        raise KeyError(f"No module called “{name}”.")

    def add(self, name: str, folder: Path | None) -> Module:
        name = check_name(name)
        if any(m.name.lower() == name.lower() for m in self.modules):
            raise ValueError(f"There is already a module called “{name}”.")
        path = (folder if folder is not None else self.default_root / name).resolve()
        path.mkdir(parents=True, exist_ok=True)
        module = Module(name, str(path))
        self.modules.append(module)
        self.modules.sort(key=lambda m: m.name.lower())
        self.save()
        return module

    def remove(self, name: str):
        self.modules = [m for m in self.modules if m.name != name]
        if self.last == name:
            self.last = None
        self.save()

    def use(self, name: str) -> Module:
        module = self.get(name)
        module.path.mkdir(parents=True, exist_ok=True)
        self.last = name
        self.save()
        return module

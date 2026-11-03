"""Finish a built app folder, or install one for this user.

  prepare_dist.py app <app dir>
      write the app's config.yaml from the repo's, with paths for a self-contained app:
      llama.cpp and models inside the app folder, lectures in Documents\\Lecture Live, and
      the microphone reset to the Windows default.

  prepare_dist.py install <installed app dir>
      give the installed app this PC's module list (sessions in Documents\\Lecture Live\\<module>)
      and copy the lectures recorded so far there. Nothing existing is overwritten.
"""
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from lecturelive import config  # noqa: E402

LECTURES = Path.home() / "Documents" / "Lecture Live"

mode, target = sys.argv[1], Path(sys.argv[2])

if mode == "app":
    dest = target / "config.yaml"
    shutil.copyfile(config.CONFIG_PATH, dest)
    config.save({"models_dir": "models", "llama_server": "llama.cpp/llama-server.exe",
                 "output_dir": "~/Documents/Lecture Live", "input_device": ""}, dest)
    config.load(dest)      # the result must be a valid config
    print("config.yaml written:", dest)

elif mode == "install":
    src_store = REPO / "modules.json"
    dest_store = target / "modules.json"
    if dest_store.exists():
        print("modules.json already installed; left as it is")
    elif src_store.exists():
        data = json.loads(src_store.read_text(encoding="utf-8"))
        for m in data["modules"]:
            old = Path(m["folder"])
            new = LECTURES / m["name"]
            new.mkdir(parents=True, exist_ok=True)
            if old.is_dir():
                for session in old.iterdir():
                    if session.is_dir() and not (new / session.name).exists():
                        shutil.copytree(session, new / session.name)
                        print("copied lecture", session.name, "->", new)
            m["folder"] = str(new)
        dest_store.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print("modules.json installed with", len(data["modules"]), "modules")
else:
    raise SystemExit(f"unknown mode {mode!r}")

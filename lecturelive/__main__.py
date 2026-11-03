"""python -m lecturelive             start the app
python -m lecturelive --icon PATH  write the app icon as a .ico file (used by setup.ps1)"""
import sys

if len(sys.argv) == 3 and sys.argv[1] == "--icon":
    from PySide6.QtWidgets import QApplication
    from .ui import make_icon

    app = QApplication([])
    if not make_icon(256).save(sys.argv[2], "ICO"):
        raise SystemExit(f"could not write {sys.argv[2]}")
else:
    from .ui import main

    raise SystemExit(main())

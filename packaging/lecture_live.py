"""Entry point of the packaged Lecture Live.exe (see build.ps1).
`Lecture Live.exe --selftest` checks every part and writes selftest.txt next to the .exe."""
import sys

if "--selftest" in sys.argv:
    from lecturelive.selftest import run

    raise SystemExit(run())

from lecturelive.ui import main  # noqa: E402

raise SystemExit(main())

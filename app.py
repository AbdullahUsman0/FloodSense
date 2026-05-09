from __future__ import annotations

from pathlib import Path
import runpy


APP_PATH = Path(__file__).resolve().parent / "app" / "main.py"


if __name__ == "__main__":
    runpy.run_path(str(APP_PATH), run_name="__main__")

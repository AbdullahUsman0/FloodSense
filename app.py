from __future__ import annotations

import os
from pathlib import Path
import runpy


# Respect platform-provided port (e.g. Render sets `PORT`).
# If present, set Streamlit environment variables so the server uses that port.
port = os.environ.get("PORT")
if port:
    os.environ.setdefault("STREAMLIT_SERVER_PORT", port)
    os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")

APP_PATH = Path(__file__).resolve().parent / "app" / "main.py"


if __name__ == "__main__":
    runpy.run_path(str(APP_PATH), run_name="__main__")

"""Entry point for Streamlit Community Cloud, which installs app/requirements.txt (the app's few
dependencies, not the training stack) and runs this file from the repository root."""

import runpy
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
runpy.run_path(str(SRC / "virtual_captain" / "app" / "streamlit_app.py"), run_name="__main__")

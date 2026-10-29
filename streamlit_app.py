"""
streamlit_app.py (Root Entrypoint)
----------------------------------
Root delegation entrypoint for Streamlit Community Cloud (share.streamlit.io).
Automatically routes execution to the main application inside `bayesian_heart_diagnosis/app/streamlit_app.py`.
"""

import sys
import runpy
from pathlib import Path

# Resolve path to the package directory and main app
root_dir = Path(__file__).resolve().parent
app_dir = root_dir / "bayesian_heart_diagnosis"
app_file = app_dir / "app" / "streamlit_app.py"

# Add directories to Python module search path
sys.path.insert(0, str(app_dir))
sys.path.insert(0, str(root_dir))

# Execute main Streamlit dashboard
if __name__ == "__main__" or "streamlit" in sys.modules:
    runpy.run_path(str(app_file), run_name="__main__")

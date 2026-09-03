"""Command-line entry point that opens the browser application."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Launch the bundled Streamlit application."""
    app_path = Path(__file__).with_name("app.py")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.maxUploadSize=1024",
        "--theme.base=light",
        "--theme.primaryColor=#0c7c86",
        "--theme.backgroundColor=#f7faf9",
        "--theme.secondaryBackgroundColor=#e8f5f4",
        "--theme.textColor=#17212b",
    ]
    return subprocess.call(command)

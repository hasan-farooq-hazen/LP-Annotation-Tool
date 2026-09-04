"""Launch the compiled application through Streamlit."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from streamlit.web import cli as streamlit_cli


def _bootstrap_path() -> Path:
    """Create the small source file Streamlit requires as its entry point."""
    directory = Path(tempfile.gettempdir()) / "license_plate_annotation_tool"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "streamlit_bootstrap.py"
    path.write_text(
        "import hazen_license_plate_annotation_tool.app\n",
        encoding="utf-8",
    )
    return path


def main() -> int:
    sys.argv = [
        "streamlit",
        "run",
        str(_bootstrap_path()),
        "--server.headless=false",
        "--server.address=127.0.0.1",
        "--server.port=8501",
        "--server.maxUploadSize=1024",
        "--browser.gatherUsageStats=false",
        "--theme.base=light",
        "--theme.primaryColor=#0c7c86",
        "--theme.backgroundColor=#f7faf9",
        "--theme.secondaryBackgroundColor=#e8f5f4",
        "--theme.textColor=#17212b",
    ]
    return streamlit_cli.main()


if __name__ == "__main__":
    raise SystemExit(main())

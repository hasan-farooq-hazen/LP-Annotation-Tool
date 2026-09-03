"""Create the local environment when needed and launch the web interface."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENVIRONMENT = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
INSTALL_MARKER = ENVIRONMENT / ".requirements.sha256"


def environment_python() -> Path:
    if sys.platform == "win32":
        return ENVIRONMENT / "Scripts" / "python.exe"
    return ENVIRONMENT / "bin" / "python"


def requirements_fingerprint() -> str:
    return hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()


def main() -> int:
    python = environment_python()
    if not python.exists():
        print("Preparing the License Plate Annotation Tool for first use…")
        venv.EnvBuilder(with_pip=True).create(ENVIRONMENT)

    fingerprint = requirements_fingerprint()
    installed = INSTALL_MARKER.read_text(encoding="utf-8").strip() if INSTALL_MARKER.exists() else ""
    if installed != fingerprint:
        print("Installing the required components…")
        subprocess.check_call(
            [str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS)]
        )
        INSTALL_MARKER.write_text(fingerprint + "\n", encoding="utf-8")

    print("Opening the License Plate Annotation Tool in your browser…")
    return subprocess.call(
        [str(python), "-m", "streamlit", "run", str(ROOT / "app.py")]
    )


if __name__ == "__main__":
    raise SystemExit(main())

"""Create the local environment when needed and launch the web interface."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENVIRONMENT = ROOT / ".venv"
PACKAGE_CONFIG = ROOT / "pyproject.toml"
INSTALL_MARKER = ENVIRONMENT / ".package.sha256"


def environment_python() -> Path:
    if sys.platform == "win32":
        return ENVIRONMENT / "Scripts" / "python.exe"
    return ENVIRONMENT / "bin" / "python"


def package_fingerprint() -> str:
    return hashlib.sha256(PACKAGE_CONFIG.read_bytes()).hexdigest()


def main() -> int:
    python = environment_python()
    if not python.exists():
        print("Preparing the Hazen License Plate Annotation Tool for first use…")
        venv.EnvBuilder(with_pip=True).create(ENVIRONMENT)

    fingerprint = package_fingerprint()
    installed = (
        INSTALL_MARKER.read_text(encoding="utf-8").strip()
        if INSTALL_MARKER.exists()
        else ""
    )
    if installed != fingerprint:
        print("Installing the required components…")
        subprocess.check_call(
            [str(python), "-m", "pip", "install", "--editable", str(ROOT)]
        )
        INSTALL_MARKER.write_text(fingerprint + "\n", encoding="utf-8")

    print("Opening the Hazen License Plate Annotation Tool in your browser…")
    return subprocess.call(
        [str(python), "-m", "hazen_license_plate_annotation_tool"]
    )


if __name__ == "__main__":
    raise SystemExit(main())

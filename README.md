<p align="center">
  <a href="https://www.hazen.ai/">
    <img src="https://raw.githubusercontent.com/hasan-farooq-hazen/LP-Annotation-Tool/main/assets/hazen-logo.svg" alt="Hazen.ai" width="240">
  </a>
</p>

# License Plate Annotation Tool

An approachable browser tool for turning vehicle footage into clean, useful license-plate
datasets—without requiring computer-vision or command-line expertise.

This tool is open sourced by [Hazen.ai](https://www.hazen.ai/), a privacy-first Urban AI
company building technology that helps organizations understand movement, risk, and changing
conditions across cities, roads, and complex outdoor environments. We released it to make
license-plate dataset preparation more accessible and reduce the time and manual effort needed
to extract frames, remove repeated images, and locate plates.

It is powered by [FastALPR](https://github.com/ankandrew/fast-alpr), using ONNX models for
plate detection and OCR.

## Start in two steps

Python 3.10 or newer is required.

1. Install the tool:

   ```bash
   pip install hazen-license-plate-annotation-tool
   ```

2. Launch it:

   ```bash
   hazen-license-plate-annotation-tool
   ```

The browser interface opens automatically. Uploads, settings, processing, model downloads,
and result downloads are handled inside the app, making it suitable for less-technical users.

## What it does

The app provides three tools that can be used independently:

1. **Video to frames** — upload a video, optionally trim it, and sample it at 5, 10, 15,
   or 20 frames per second.
2. **Reduce similar frames** — upload images/ZIPs or reuse frames from step 1, then retain
   the clearest representatives from visually similar groups.
3. **Extract license plates** — upload images/ZIPs or reuse an earlier result, detect and
   crop plates, and name them with OCR text when available.

The **Guided workflow** runs all three stages from a single video. Every result is provided
as a downloadable ZIP containing the output images, `manifest.csv`, and `run_config.json`.

## Run locally

Python 3.10 or newer is recommended.

For the easiest setup, run:

```bash
python launch.py
```

The launcher creates a private `.venv`, installs or updates the required components, and
opens the web interface. After that, uploads, settings, processing, model downloads, and
result downloads are all handled inside the interface.

If the launch command is not available in your terminal, use the equivalent module command:

```bash
python -m hazen_license_plate_annotation_tool
```

Models are downloaded and cached the first time plate extraction is run. CPU processing is
supported; long videos and higher frame rates naturally take more time and disk space.

## Tests

```bash
pip install --editable .
python -m unittest discover -s tests -v
```

## Privacy and storage

Uploads and generated files are kept in a temporary directory for the active browser session.
Download any results you want to keep before closing or restarting the app.

## Build a release

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

The resulting wheel and source archive in `dist/` are ready for TestPyPI, PyPI, or the
repository's GitHub publishing workflow.

# License Plate Annotation Tool

A browser-based tool that extracts useful source frames containing license plates from a
video.

## Requirements

- Python 3.10 or newer
- An internet connection the first time the recognition models are downloaded

## Run locally

```bash
python launch.py
```

The launcher creates a local virtual environment, installs the required packages, and opens
the app in a browser.

You can also install and run the package manually:

```bash
pip install --editable .
python -m hazen_license_plate_annotation_tool
```

## Workflow

1. Upload a video.
2. Choose the frame rate and optional trim range.
3. Adjust the similarity and plate-detection settings if needed.
4. Select **Process video**.
5. Download the ZIP of source frames containing detected plates.

The workflow samples frames, removes visually similar images, detects license plates, and
keeps the corresponding source frames. The output ZIP also contains `manifest.csv` and
`run_config.json`.

## Tests

```bash
pip install --editable .
python -m unittest discover -s tests -v
```

## Storage

Uploads and generated files are stored in a temporary directory for the active browser
session. Download the result before closing or restarting the app.

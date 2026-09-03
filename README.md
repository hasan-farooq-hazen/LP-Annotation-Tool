# License Plate Annotation Tool

A friendly Streamlit workspace for turning vehicle footage into clean license-plate crops.
It is powered by [FastALPR](https://github.com/ankandrew/fast-alpr), using ONNX models for
plate detection and OCR.

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

For manual setup instead:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Models are downloaded and cached the first time plate extraction is run. CPU processing is
supported; long videos and higher frame rates naturally take more time and disk space.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Privacy and storage

Uploads and generated files are kept in a temporary directory for the active browser session.
Download any results you want to keep before closing or restarting the app.

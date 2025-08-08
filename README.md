# Synergy Photo Editor

This repository contains the Synergy Photo Editor tool, which optimizes and processes photos for OTA listings. It offers both a command-line interface (CLI) and a Gradio-powered web UI.

## Features

- Batch editing and optimization for PNG and JPEG images.
- Room-type presets (bedroom, bathroom, etc.) and OTA presets (Airbnb, Booking.com, Vrbo).
- Auto-leveling, auto-verticals, and auto-crop options.
- SEO-friendly renaming of output files.
- Gradio UI with single and batch modes.

## Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

### Launch Web UI

Install dependencies and run:

```bash
pip install -r requirements.txt
python synergy_photo_editor_v_3.py --ui
```

It will launch a Gradio app at `http://127.0.0.1:7860`. Use it to upload and process images.

### Command Line

Run the tool on a folder of photos:

```bash
python synergy_photo_editor_v_3.py -i ./photos --ota airbnb --auto-level --auto-verticals --auto-crop
```

Use `-h` or `--help` to see all options.

### Self-test

Run internal tests:

```bash
python synergy_photo_editor_v_3.py --selftest
```

## Development

You can develop and edit this project using an IDE like Cursor or VS Code. To run tests automatically, install dependencies and run the self-test command. Contributions are welcome!

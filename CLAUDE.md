# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Tools for querying a vision-language model (VLM) hosted on a remote Ollama server to detect objects in images and video frames, writing the results as bounding-box ("bbox") files. The main model is `qwen3-vl:8b`.

There is no build, test suite, linter or requirements file. Dependencies: `ollama`, `opencv-python` (`cv2`), `pillow`, `pydantic`. Check changes with `python3 -c "import process_video, process_image"`; running the scripts needs an Ollama server, given with `--server <host>` (port 11434 is fixed).

## Commands

```bash
# Single image
python3 process_image.py --image img.png --classes label_classes.json --output out.bbox [--server HOST] [--verbose]

# Video, one model call per frame
python3 process_video.py --video vid.mp4 --classes label_classes.json --output out.bbox [--server HOST] \
    [--frames START-END | --restart] [--temperature T] [--seed N] [--num_ctx N] [--num_predict N] [--verbose]
```

- `--frames` takes `START-END`, `START-` or `-END` (1-based, inclusive).
- `--restart` resumes an interrupted run: it appends to `--output`, starting after the highest frame number in it and continuing bbox ids. It cannot be combined with `--frames`.
- Without `--temperature`/`--seed`, the model's own defaults from the server are used (deliberately).

## Architecture

Pipeline: classes file → prompt + JSON schema → Ollama → parse response → bbox file.

- **Classes file** (`label_classes.json`, read by `label_classes.read_label_classes`): a list of `{code, name, prompt, prompt_order}`. `prompt` is the text shown to the model and is also the label the model must return; it may contain commas. `code` is the class id written to the bbox file.
- **`qwen3vl_support.py`** holds everything model-specific:
  - `get_prompt()` lists the class prompts in `prompt_order` and asks for `{"objects": [{"label", "bbox_2d"}]}`.
  - `get_format()` builds the matching JSON schema for Ollama's `format` parameter, with `label` restricted (enum) to the exact prompt strings. The prompt and schema must stay consistent.
  - `parse_model_response()` accepts the `{"objects": [...]}` object or a raw list (also strips ```json fences), maps labels back to classes with `label_classes.get_entry_from_prompt()` (exact match, else unique prefix), and converts `bbox_2d` from Qwen's 0–1000 grid to pixels. It returns `[]` for no detections and `None` if the response can't be fully parsed (any item dropped → whole frame counted as unparsed).
- **Model gating:** both scripts only call the parser when `model == "qwen3-vl:8b"`; any other model silently yields no bboxes.
- **`process_video.py`** decodes frames with OpenCV, writes each to a temporary PNG for Ollama, and appends each frame's bboxes to the output file as it goes (flushed per frame so a crash loses at most the current frame). It exits if a response's `done_reason` isn't `stop`. It prints average Ollama timing stats, elapsed time and the unparsed-frame rate at the end. Frame timestamps (`fts`) are currently a stub (`0.0`).
- **`process_image.py`** uses `client.chat` without a schema or model options, and writes with `bbox_io.write()`.

### bbox file format (`bbox_io.py`)

One record per line: `bboxid fno ts xmin xmax ymin ymax class_id`.

- Coordinates are normalised to 0–1. Note the order is **xmin xmax ymin ymax**, while in-memory boxes are pixel `[xmin, ymin, xmax, ymax]`.
- `ts` is written as `seconds.microseconds`.
- In-memory records are `[bboxid, fno, ts, box, class_id]`. Use `format_bbox()` to write one record at a time, `write()` for a whole list, `read()` to load.

`modelfiles/` contains Ollama Modelfiles that create larger-context (`num_ctx`) variants of models.

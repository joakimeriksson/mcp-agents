# Candytron 4000, as an MCP service

Shuffling candy on a table with unprecedented precision and dexterity.

This is the MCP server version. Start it with:

```bash
uv run candytron_mcp.py
```

Then you can can use it by starting a generic MCP client such as mcpclient_speech or mcpclient_text.

## Scene logging

Pass `--log-scenes-dir DIR` to record every scene request from a connected
client. For each call to the `get_service_augmentation` prompt, three files
are written into `DIR`:

- `<timestamp>.jpg` — the most recent annotated camera frame (with YOLO
  bounding boxes and labels), the same picture shown in the OpenCV window.
- `<timestamp>_raw.jpg` — the corresponding unannotated camera frame.
- `<timestamp>.json` — the state returned to the caller, containing
  `timestamp`, `lang`, `scene` (the consensus dict of position → candy) and
  `message` (the formatted string the caller receives).

The flag is opt-in. When unset, no extra I/O is performed. In simulated camera
mode no JPGs are produced (there is no real frame), but the JSON is still
written.

```bash
uv run candytron_mcp.py --log-scenes-dir ./scene-log
```

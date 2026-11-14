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

The flag is opt-in. When unset, no extra I/O is performed. With the old
static simulation (`--sim-static`) no JPGs are produced (there is no frame),
but the JSON is still written; the simulated table below writes JPGs too.

```bash
uv run candytron_mcp.py --log-scenes-dir ./scene-log
```

## Simulated table (no robot, no camera)

With `--simulate-robot --simulate-camera` (what `./start-candytron.sh --sim`
and `./start-simulated.sh` pass) the server runs `sim_table.py`: a drawn,
top-down candy table that acts as both the table camera and the arm.

- The paper and its 12 printed boxes are drawn where the arm's taught poses
  (`base-saved-poses.yaml`) are, with real candy pictures cut out of the YOLO
  training photos (`sim_assets/`).
- The frames go through the **real** vision code: the board is calibrated
  from the printed boxes, the YOLO model detects the candy, and the same
  consensus voting runs. It gets about 87 % right; the white and shiny
  wrappers (nut cream, Fazer mint, Marianne) are missed now and then, as on
  the real table. `--sim-vision direct` reports exactly what is on the table
  instead (also the fallback when `models/best-m.pt` is missing).
- `move_between` drives a simple two-joint arm with the same poses the real
  arm gets: it lowers, grips whatever is under the fingers (or nothing),
  carries it and lets go, so the table really changes. A candy left on O0 is
  taken by the "visitor" after 4 seconds. `--sim-speed 2` makes it faster.

Watch it at <http://127.0.0.1:7999/sim> (or in the control panel at
<http://127.0.0.1:8090>): the live picture with YOLO's boxes, what is really
on the table next to what the robot is told, the arm queue, and buttons to
refill the table, drop a candy between the boxes, make the next grip slip,
or clear the table. The OpenCV window has the same keys: `r` refill, `l`
loose candy, `f` next grip slips, `x` clear, `c` recalibrate, `q` quit.

`--sim-static` brings back the old simulation: a fixed list of candies that
never changes and an arm that only waits.

The candy pictures are built once from the YOLO dataset (only needed again
if the dataset changes; each cut-out must be recognised by the model):

```bash
uv run sim_table.py build-assets --dataset ~/work/candy-yolo11-dataset
uv run sim_table.py preview /tmp/table.jpg
```

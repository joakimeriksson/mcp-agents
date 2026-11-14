"""Tests for the simulated table (no YOLO needed):
uv run --with pytest python -m pytest -q test_sim_table.py"""

import time
from types import SimpleNamespace as P

import pytest

import sim_table as st
from camera import find_board_boxes, grid_from_boxes


@pytest.fixture
def sim():
    s = st.SimTable(vision="direct", speed=50, seed=3)
    s.restock()
    return s


def pose(s, tag):
    x, y, z = s.poses[tag]
    return P(x=x, y=y, z=z)


def test_calibration_finds_the_printed_boxes_on_the_drawn_board(sim):
    with sim.cleared():
        boxes = find_board_boxes(sim.render())
    grid = grid_from_boxes(boxes, 3, 4)
    assert grid is not None and sorted(grid) == sorted(sim.boxes)
    for tag, (x, y, _z) in sim.boxes.items():      # where the arm's poses are
        px, py = st.to_px(x, y)
        assert abs(grid[tag][0] - px) < 4 and abs(grid[tag][1] - py) < 4


def test_hand_out_moves_the_candy_and_the_visitor_takes_it(sim, monkeypatch):
    src = next(t for t, names in sim.truth().items() if t in sim.boxes)
    name = sim.truth()[src][0]
    assert sim.pick_and_place(pose(sim, src), pose(sim, "O0"), f"{src} -> O0")
    t = sim.truth()
    assert src not in t and t["O0"] == [name]
    later = time.time() + st.TAKE_AFTER_S + 1
    monkeypatch.setattr(st.time, "time", lambda: later)
    sim.render()
    assert "O0" not in sim.truth()
    assert any("took the" in e for _t, e in sim.events)


def test_gripping_an_empty_box_moves_nothing(sim):
    empty = next(t for t in sim.boxes if t not in sim.truth())
    before = sim.truth()
    assert not sim.pick_and_place(pose(sim, empty), pose(sim, "O0"), "x")
    assert sim.truth() == before
    assert any("closed on nothing" in e for _t, e in sim.events)


def test_a_slipping_grip_drops_the_candy_on_the_way(sim):
    src = next(t for t in sim.truth() if t in sim.boxes)
    sim.set_fail_next()
    assert not sim.pick_and_place(pose(sim, src), pose(sim, "O0"), "x")
    assert "O0" not in sim.truth() and src not in sim.truth()
    assert sum(len(v) for v in sim.truth().values()) == 11      # still on the table
    assert not sim.fail_next


def test_box_to_box_move_and_direct_detections(sim):
    t = sim.truth()
    src = next(k for k in t if k in sim.boxes)
    dst = next(k for k in sim.boxes if k not in t)
    sim.pick_and_place(pose(sim, src), pose(sim, dst), "x")
    assert sim.truth()[dst] == t[src]
    dets = sim.detections()
    assert len(dets) == 11
    x, y, _z = sim.poses[dst]
    px, py = st.to_px(x, y)
    assert any(n == t[src][0] and abs(dx - px) < 15 and abs(dy - py) < 15 for n, dx, dy in dets)


def test_loose_candy_lies_between_boxes(sim):
    sim.drop_loose("Dumle")
    assert sim.truth().get("loose") == ["Dumle"]


def test_frame_and_jpeg(sim):
    ok, fr = sim.read()
    assert ok and fr.shape == (st.FRAME_H, st.FRAME_W, 3)
    assert sim.jpeg()[:2] == b"\xff\xd8"


def test_every_candy_has_pictures():
    assets = st.load_assets()
    assert len(assets) >= 13 and all(len(v) >= 1 for v in assets.values())


def test_server_knows_the_new_model_names():
    from transtable import transtable
    for n in ("NutcremePear", "Nutcreme", "VanillaToffee"):
        assert n in transtable


# ---------------------------------------------------------- missed cases
def _perfect(labels):
    return [(c.name, 0.9, x1, y1, x2, y2) for c, x1, y1, x2, y2 in labels]


def test_judge_finds_each_kind_of_mistake(sim):
    labels = sim.label_boxes(sim.candies)
    boxes = _perfect(labels)
    assert st.judge(labels, boxes) == []
    assert [p[0] for p in st.judge(labels, boxes[1:])] == ["missed"]
    other = "Mars" if labels[0][0].name != "Mars" else "Dumle"
    wrong = [(other,) + boxes[0][1:]] + boxes[1:]
    assert [p[0] for p in st.judge(labels, wrong)] == ["wrong"]
    assert [p[0] for p in st.judge(labels, boxes + [boxes[0]])] == ["double"]
    assert [p[0] for p in st.judge(labels, boxes + [("Mars", 0.5, 5, 5, 25, 25)])] == ["phantom"]
    # the model's old name for the same candy is not a mistake
    lab = [(st.Candy(1, "NutcremePear", 0.3, 0.0, 0), 10, 10, 50, 50)]
    assert st.judge(lab, [("Pearnut", 0.8, 12, 12, 48, 48)]) == []


def test_a_wrong_frame_is_saved_once_with_the_true_labels(tmp_path):
    s = st.SimTable(vision="direct", speed=50, seed=5, misses_dir=str(tmp_path))
    s.restock(6)
    _ok, frame = s.read()
    labels = s.label_boxes(s.candies)
    boxes = _perfect(labels)[1:]                     # the first candy is missed
    assert s.save_if_wrong(boxes, frame)
    assert s.save_if_wrong(boxes, frame) and s.misses_saved == 1   # same mistake, same table
    lab = next((tmp_path / "labels").glob("*.txt")).read_text().split("\n")
    lab = [l.split() for l in lab if l]
    assert len(lab) == 6
    assert sorted(st.DATASET_NAMES[int(l[0])] for l in lab) == sorted(c.name for c in s.candies)
    c, x1, y1, x2, y2 = labels[0]
    assert any(abs(float(l[1]) - (x1 + x2) / 2 / st.FRAME_W) < 1e-4 for l in lab)
    assert (tmp_path / "data.yaml").exists() and len(list((tmp_path / "images").glob("*.jpg"))) == 1
    import json
    rec = json.loads((tmp_path / "misses.jsonl").read_text().splitlines()[0])
    assert rec["problems"][0]["kind"] == "missed"


def test_nothing_is_saved_while_the_arm_moves(tmp_path):
    s = st.SimTable(vision="direct", speed=50, seed=5, misses_dir=str(tmp_path))
    s.restock(6)
    s.arm.moving = True
    _ok, frame = s.read()
    assert s.save_if_wrong([], frame) == [] and s.misses_saved == 0

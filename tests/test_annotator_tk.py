"""The desktop annotator. The edit logic (annotator/model.py) is tested without a display; the window itself is driven
with real mouse and key events when a display exists (skipped otherwise)."""
import pytest

from animalworld import pipeline
from animalworld.annotator.model import Doc, View
from animalworld.stages import review as R

from tkutil import box_grab_point, make_root, pump, shot

KP = ["nose", "left_ear", "right_ear"]


def test_view_transform():
    v = View()
    v.fit(640, 480, 800, 600)
    assert v.scale == pytest.approx(min(784 / 640, 584 / 480))
    for p in ((0, 0), (640, 480), (123.5, 77.25)):
        assert v.to_image(*v.to_canvas(*p)) == pytest.approx(p)
    cx, cy = v.to_canvas(320, 240)
    assert (cx, cy) == pytest.approx((400, 300))                     # centred
    before = v.to_image(250, 130)
    v.zoom_at(250, 130, 2.0)
    assert v.to_image(250, 130) == pytest.approx(before) and v.scale == pytest.approx(2 * min(784 / 640, 584 / 480))
    v.pan(10, -5)
    assert v.to_image(260, 125) == pytest.approx(before)
    x0, y0, x1, y1 = v.visible(640, 480, 800, 600)
    assert 0 <= x0 < x1 <= 640 and 0 <= y0 < y1 <= 480 and (x1 - x0) < 640     # zoomed in: only a part is drawn
    v.zoom_at(0, 0, 1e9)
    assert v.scale == 40.0
    assert View(1, 5000, 5000).visible(10, 10, 100, 100) is None


def test_doc_edits_and_undo():
    d = Doc([{"category": "mouse", "box": [10, 10, 50, 40], "poly": [[10, 10], [50, 10], [50, 40], [10, 40]],
              "keypoints": {"nose": [20, 20, 2]}, "score": 0.9, "track_id": None}], 200, 100, "mouse", KP)
    assert d.selected == 0 and d.add_box(5, 5, 6, 6) is None and not d.dirty        # too small: nothing happens
    i = d.add_box(150, 90, 100, 60)                                                 # any two opposite corners
    assert i == 1 and d.instances[1]["box"] == [100, 60, 150, 90] and d.selected == 1
    assert d.next_keypoint(1) == "nose"
    d.set_keypoint(1, "nose", 110, 70)
    d.set_keypoint(1, "left_ear", 999, -5)                                          # clamped into the image
    assert d.instances[1]["keypoints"]["left_ear"] == [200.0, 0.0, 2] and d.next_keypoint(1) == "right_ear"
    with pytest.raises(KeyError):
        d.set_keypoint(1, "tail", 1, 1)
    assert d.hit(110, 70, 3) == ("kp", 1, "nose") and d.hit(100, 60, 3) == ("handle", 1, "nw")
    assert d.hit(30, 25, 3) == ("box", 0) and d.hit(190, 5, 3) is None
    d.selected = 0
    assert d.hit(50, 10, 3) == ("handle", 0, "ne") and d.hit(20, 20, 3) == ("kp", 0, "nose")
    d.checkpoint()
    d.move_instance(0, 1000, 5)                                                     # stops at the image border
    assert d.instances[0]["box"] == [160, 15, 200, 45] and d.instances[0]["poly"][0] == [160, 15]
    assert d.instances[0]["keypoints"]["nose"] == [170, 25, 2]
    assert d.undo() and d.instances[0]["box"] == [10, 10, 50, 40] and d.instances[0]["keypoints"]["nose"] == [20, 20, 2]
    d.checkpoint()
    d.resize_box(0, "se", 80, 70)
    d.resize_box(0, "nw", 500, 500)                                                 # cannot cross the opposite corner
    assert d.instances[0]["box"] == [77.0, 67.0, 80.0, 70.0]
    d.undo()
    d.checkpoint()
    d.move_vertex(0, 2, 70, 60)
    assert d.instances[0]["box"] == [10, 10, 70, 60]
    d.toggle_visibility(0, "nose")
    assert d.instances[0]["keypoints"]["nose"][2] == 1
    d.delete_keypoint(0, "nose")
    assert d.instances[0]["keypoints"] is None
    assert d.add_poly([(1, 1), (30, 1)]) is None and d.add_poly([(1, 1), (30, 1), (30, 20)]) == 2
    d.delete_instance(2)
    assert len(d.instances) == 2 and d.selected == 1
    n = 0
    while d.undo():
        n += 1
    assert n >= 6 and len(d.instances) == 1 and d.instances[0]["keypoints"] == {"nose": [20, 20, 2]}


@pytest.fixture
def window(project):
    root = make_root()
    root.geometry("200x100+0+0")
    from animalworld.annotator.tkapp import AnnotatorWindow
    pipeline.run_all(project, until="propose")
    w = AnnotatorWindow(root, project, geometry="1180x760+0+0")
    pump(root, settle=0.4)
    w.canvas.focus_force()
    pump(root)
    yield root, w, project
    root.destroy()


def key(root, w, name):
    w.canvas.focus_force()
    w.canvas.event_generate(f"<KeyPress-{name}>")
    pump(root, settle=0.1)


def drag(root, w, a, b, steps=4):
    c = w.canvas
    c.event_generate("<ButtonPress-1>", x=int(a[0]), y=int(a[1]))
    for k in range(1, steps + 1):
        c.event_generate("<B1-Motion>", x=int(a[0] + (b[0] - a[0]) * k / steps), y=int(a[1] + (b[1] - a[1]) * k / steps))
    c.event_generate("<ButtonRelease-1>", x=int(b[0]), y=int(b[1]))
    pump(root, settle=0.1)


def click(root, w, p):
    w.canvas.event_generate("<ButtonPress-1>", x=int(p[0]), y=int(p[1]))
    w.canvas.event_generate("<ButtonRelease-1>", x=int(p[0]), y=int(p[1]))
    pump(root, settle=0.05)


def test_annotator_window_with_mouse_and_keys(window):
    root, w, project = window
    assert w.listbox.size() == 24 and "24 pending" in w.counts_var.get()
    cw, ch = w.canvas.winfo_width(), w.canvas.winfo_height()
    assert cw > 400 and ch > 400 and w.image is not None and w.canvas.find_all()
    # 1. first image: draw a box with the mouse, place two keypoints, accept
    first = w.cur["id"]
    n0 = len(R.proposed(project, first)["instances"])
    a, b = w.view.to_canvas(8, 8), w.view.to_canvas(70, 50)
    key(root, w, "b")
    assert w.mode == "box"
    drag(root, w, a, b)
    assert len(w.doc.instances) == n0 + 1 and w.mode == "select" and w.doc.selected == n0
    key(root, w, "k")
    assert w.mode == "kp" and "nose" in w.msg_var.get()
    click(root, w, w.view.to_canvas(20, 20))
    click(root, w, w.view.to_canvas(30, 15))
    assert w.kp_list.get(0).startswith("o nose") and w.kp_list.get(2).startswith("- ")
    key(root, w, "Escape")
    key(root, w, "a")
    d = R.reviewed(project, first)
    new = d["instances"][-1]
    assert d["status"] == "accepted" and d["origin"] == "human" and len(d["instances"]) == n0 + 1
    assert new["box"] == pytest.approx([8, 8, 70, 50], abs=1.5) and set(new["keypoints"]) == {"nose", "left_ear"}
    assert new["keypoints"]["nose"][:2] == pytest.approx([20, 20], abs=1.5)
    # 2. the window moved on by itself: reject the next image
    second = w.cur["id"]
    assert second != first
    key(root, w, "r")
    assert R.reviewed(project, second)["status"] == "rejected"
    # 3. third image: drag an existing box (outline and box move together), resize it by a handle, then accept
    third = w.cur["id"]
    p0 = R.proposed(project, third)["instances"]
    assert p0, "the mock proposer gives every non-empty image at least one animal"
    i = 0
    w.doc.selected = i
    box = list(w.doc.instances[i]["box"])
    cx, cy = w.view.to_canvas(*box_grab_point(w.doc, i, 8 / w.view.scale))
    dx = -30 if box[0] > w.cur["W"] / 2 else 30                       # move towards the free side
    drag(root, w, (cx, cy), (cx + dx, cy + 12))
    moved = w.doc.instances[i]["box"]
    assert w.doc.selected == i and moved[0] - box[0] == pytest.approx(dx / w.view.scale, abs=1.0)
    assert moved[2] - moved[0] == pytest.approx(box[2] - box[0], abs=0.01)           # moved, not resized
    hx, hy = w.view.to_canvas(moved[2], moved[3])                                     # the "se" handle of the selected box
    drag(root, w, (hx, hy), (hx - 10, hy - 8))
    resized = w.doc.instances[i]["box"]
    assert resized[2] == pytest.approx(moved[2] - 10 / w.view.scale, abs=1.0) and resized[0] == pytest.approx(moved[0], abs=0.01)
    key(root, w, "a")
    saved = R.reviewed(project, third)["instances"][i]
    assert saved["box"] == pytest.approx(resized, abs=0.02) and saved["box"] != pytest.approx(p0[i]["box"], abs=0.5)
    if p0[i].get("poly"):
        assert saved["poly"][0][0] == pytest.approx(p0[i]["poly"][0][0] + dx / w.view.scale, abs=1.0)
    # 4. fourth image: outline with undo, keypoint drag, delete, undo, zoom, pan, revert
    fourth = w.cur["id"]
    n3 = len(w.doc.instances)
    key(root, w, "p")
    pts = [w.view.to_canvas(*p) for p in ((5, 5), (60, 5), (60, 45), (5, 45))]
    for p in pts:
        click(root, w, p)
    w.undo()                                                          # removes the last point while drawing
    assert len(w._poly_pts) == 3
    key(root, w, "Return")
    assert len(w.doc.instances) == n3 + 1 and len(w.doc.instances[-1]["poly"]) == 3 and w.mode == "select"
    key(root, w, "k")
    click(root, w, w.view.to_canvas(20, 12))
    key(root, w, "Escape")
    kx, ky = w.view.to_canvas(20, 12)
    drag(root, w, (kx, ky), (kx + 16, ky + 8))
    kp = w.doc.instances[-1]["keypoints"]["nose"]
    assert w.sel_kp == "nose" and kp[0] == pytest.approx(20 + 16 / w.view.scale, abs=1.0)
    key(root, w, "v")
    assert w.doc.instances[-1]["keypoints"]["nose"][2] == 1
    key(root, w, "Delete")                                            # the selected keypoint only
    assert w.doc.instances[-1]["keypoints"] is None and len(w.doc.instances) == n3 + 1
    key(root, w, "Delete")                                            # now the animal
    assert len(w.doc.instances) == n3
    w.canvas.event_generate("<Control-KeyPress-z>")
    pump(root)
    assert len(w.doc.instances) == n3 + 1
    s0 = w.view.scale
    w.canvas.event_generate("<Button-4>", x=300, y=300)
    pump(root)
    assert w.view.scale == pytest.approx(s0 * 1.2)
    w.canvas.event_generate("<ButtonPress-2>", x=300, y=300)
    w.canvas.event_generate("<B2-Motion>", x=340, y=280)
    pump(root)
    ox = w.view.ox
    key(root, w, "f")
    assert w.view.scale == pytest.approx(s0) and w.view.ox != ox
    w.buttons["sam"].invoke()                                         # not configured: says so, changes nothing
    pump(root)
    assert "needs SAM 2.1" in w.msg_var.get() and w._sam_job is None
    w.buttons["revert"].invoke()
    pump(root)
    assert len(w.doc.instances) == n3 and R.reviewed(project, fourth) is None
    # 5. list filter and navigation
    w.filter_var.set("accepted")
    w._fill_list()
    pump(root)
    assert w.listbox.size() == 3                                      # two accepted + the current (pending) image stays visible
    w.filter_var.set("all")
    w._fill_list()
    key(root, w, "Down")
    assert w.cur["id"] != fourth
    key(root, w, "Up")
    assert w.cur["id"] == fourth
    c = R.counts(project)
    assert (c["accepted_by_human"], c["rejected"], c["pending"]) == (2, 1, 21) and "21 pending" in w.counts_var.get()
    shot(w, "annotator_alone")


def test_annotator_main_refuses_bad_project(tmp_path, capsys):
    from animalworld.annotator import tkapp
    assert tkapp.main([str(tmp_path / "missing")]) == 2
    assert "not an animalworld project" in capsys.readouterr().err

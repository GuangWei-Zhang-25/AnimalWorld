"""Automatic zoom (zeroshot/zoom.py), the stable-geometry bookkeeping and the corrected summary numbers. Mock engine, CPU, no weights."""
import json

import numpy as np
import pytest


def _two_arena_video(path, frames=60, small=True):
    """640 x 240: two 'cameras' side by side (different floor brightness: a hard step at the middle column), one dark animal in
    each half walking a known path. small: 14 x 8 px animals (invisible to the test detector on the whole frame)."""
    cv2 = pytest.importorskip("cv2")
    W, H = 640, 240
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    truth = []
    a, b = (7, 4) if small else (36, 20)
    for f in range(frames):
        im = np.empty((H, W, 3), np.uint8); im[:, :W // 2] = 150; im[:, W // 2:] = 170
        p1 = (70 + 1.5 * f, 110 + 20 * np.sin(f / 9.0)); p2 = (430 + 1.2 * f, 150 - 25 * np.sin(f / 7.0))
        for p in (p1, p2):
            cv2.ellipse(im, (int(round(p[0])), int(round(p[1]))), (a, b), 0, 0, 360, (25, 25, 25), -1, cv2.LINE_AA)
        truth.append((p1, p2)); vw.write(im)
    vw.release()
    return path, truth


def test_seam_regions_and_shape():
    from animalworld.zeroshot import zoom as Z
    L = 20.0
    dets = []
    for t in range(12):                                             # one animal in each half of a 1280 x 480 two-camera frame
        for cx, cy in ((200 + 10 * t, 200), (900 + 5 * t, 260)):
            dets.append(dict(t=t, box=[cx - 10, cy - 5, cx + 10, cy + 5], score=0.9, area=120.0, c=(cx, cy)))
    regs = Z.find_regions(dets, 12, 1280, 480, L, dict(x=640, y=None))
    assert len(regs) == 2 and [r["animals"] for r in regs] == [1, 1]
    assert regs[0]["box"][2] <= 640 <= regs[1]["box"][0]             # never across the seam
    for r in regs:
        w, h = r["box"][2] - r["box"][0], r["box"][3] - r["box"][1]
        assert min(w, h) >= L * Z.MODEL_INPUT / Z.SIZE_MAX_PX - 1 and max(w, h) <= 2 * min(w, h) + 2
        assert Z.size_at_input(L, w, h) <= Z.SIZE_MAX_PX + 1 and 1.0 <= Z.zoom_factor(r["box"]) <= Z.ZOOM_MAX
    # no seam, the two animals never mix and a gap separates them: two regions; five animals milling in one field: one region
    regs = Z.find_regions([dict(d, c=(d["c"][0] / 4 + 300, d["c"][1]), box=[d["box"][0] / 4 + 300, d["box"][1], d["box"][2] / 4 + 300, d["box"][3]]) for d in dets],
                          12, 1280, 480, L, dict(x=None, y=None))
    assert len(regs) == 2
    rng = np.random.default_rng(0)
    field = [dict(t=t, box=[x - 10, y - 5, x + 10, y + 5], score=0.9, area=120.0, c=(float(x), float(y)))
             for t in range(12) for x, y in rng.uniform([400, 150], [560, 300], (5, 2))]
    regs = Z.find_regions(field, 12, 1280, 480, L, dict(x=None, y=None))
    assert len(regs) == 1 and regs[0]["animals"] == 5
    assert Z.split_count([dict(box=[0, 0, 100, 100], animals=1, share=1.0), dict(box=[0, 0, 300, 300], animals=1, share=0.6)], 3) == [1, 2]
    assert Z.split_count([dict(box=[0, 0, 100, 100], animals=1, share=1.0), dict(box=[0, 0, 300, 300], animals=1, share=0.6)], 1) == [1, 0]
    assert Z.find_regions([], 12, 1280, 480, L, dict(x=None, y=None)) == []


def test_map_back_is_exact():
    cv2 = pytest.importorskip("cv2")
    from animalworld.zeroshot import zoom as Z
    mk = np.zeros((20, 40), bool); mk[5:15, 10:30] = True
    row = dict(id=1, x=200.0, y=100.0, box=[160, 80, 200, 100], mask=(np.packbits(mk), 20, 40), kp=np.array([[180.0, 90.0]]), kc=np.array([0.9]), det_box=[160, 80, 200, 100])
    out = Z.map_back([row], [300, 50, 500, 150], 4.0, 4.0, region=2, id0=3, cv2=cv2)[0]
    assert (out["id"], out["region"], out["zoom"]) == (4, 2, 4.0) and (out["x"], out["y"]) == (350.0, 75.0)
    assert out["box"] == [340, 70, 350, 75] and out["kp"].tolist() == [[345.0, 72.5]]
    bits, h, w = out["mask"]
    assert (h, w) == (5, 10) and np.unpackbits(bits)[:50].reshape(5, 10).sum() > 0


def test_jitter_metric_sees_an_alternating_source():
    from animalworld.zeroshot import pipeline as P
    steady, flip = [], []
    for g in range(60):
        x = 100 + 2.0 * g
        steady.append([dict(id=1, x=x, y=50.0, box=[x - 20, 40, x + 20, 60], kp=None, kc=None)])
        w = 20 if g % 3 == 0 else 30                                  # every third frame another box source
        flip.append([dict(id=1, x=x, y=50.0, box=[x - w, 40, x + w, 60], kp=None, kc=None)])
    a, b = P.jitter(steady, 3), P.jitter(flip, 3)
    assert a["edge_px"] == 0.0 and a["alternation"] <= 1.01 and a["transitions"] == 59
    assert b["edge_px"] > 2 and b["by_phase"][1] == 0.0 and b["by_phase"][0] > 4 and b["alternation"] > 100


def test_small_animals_in_two_cameras_are_zoomed_and_mapped_back(tmp_path):
    pytest.importorskip("cv2")
    from animalworld.cli import main
    v, truth = _two_arena_video(tmp_path / "two cameras.mp4")
    out = tmp_path / "out"
    assert main(["zeroshot", str(v), "--out", str(out), "--yes", "--engine", "mock", "--zoom", "off"]) != 0      # whole frame: the test detector sees nothing
    off = json.loads((out / "two_cameras" / "summary.json").read_text()) if (out / "two_cameras" / "summary.json").exists() else None
    assert off is None or off["rows_with_box"] == 0
    out = tmp_path / "out_zoom"
    assert main(["zeroshot", str(v), "--out", str(out), "--yes", "--engine", "mock"]) == 0
    d = out / "two_cameras"
    s = json.loads((d / "summary.json").read_text())
    assert s["zoom"]["used"] and len(s["regions"]) == 2 and s["animals"] == 2 and s["animals_auto_count"] == 2
    assert s["regions"][0]["box"][2] <= 320 <= s["regions"][1]["box"][0] and all(1.0 < r["zoom"] <= 4.0 for r in s["regions"])
    assert s["zoom"]["frame_used_pct"] < 100 and "of the frame in 2 region" in s["zoom"]["short"]
    assert (d / "overlay_regions.mp4").exists() and (d / "overlay.mp4").exists()
    rows = [r.split(",") for r in (d / "tracks.csv").read_text().splitlines()]
    hd = rows[0]; ix = {k: hd.index(k) for k in ("video_frame", "id", "x", "y", "region", "zoom", "geometry")}
    got = {(int(r[ix["video_frame"]]), int(r[ix["id"]])): (float(r[ix["x"]]), float(r[ix["y"]]), int(r[ix["region"]])) for r in rows[1:]}
    assert {i for _, i in got} == {1, 2} and {v_[2] for v_ in got.values()} == {1, 2}
    err = []
    for (f, i), (x, y, reg) in got.items():
        tx, ty = truth[f][i - 1]                                       # identity 1 = left camera, 2 = right camera
        assert reg == i and (x < 320) == (i == 1)
        err.append(np.hypot(x - tx, y - ty))
    assert len(err) >= 100 and np.median(err) < 2.0 and max(err) < 6.0   # full-frame pixels, mapped back from the enlarged crops
    assert s["frames_with_all_animals_pct"] > 90


def test_large_animals_are_left_alone_and_the_output_is_unchanged(tmp_path):
    pytest.importorskip("cv2")
    from animalworld.cli import main
    from test_zeroshot import _video
    v = _video(tmp_path / "big.mp4", 2, 45)
    for mode in ("auto", "off"):
        assert main(["zeroshot", str(v), "--out", str(tmp_path / mode), "--yes", "--engine", "mock", "--zoom", mode]) == 0
    a, b = [(tmp_path / m / "big" / "tracks.csv").read_bytes() for m in ("auto", "off")]
    assert a == b                                                       # no zoom: byte for byte the whole-frame result
    s = json.loads((tmp_path / "auto" / "big" / "summary.json").read_text())
    assert s["zoom"]["used"] is False and "whole frame" in s["zoom"]["text"] and not (tmp_path / "auto" / "big" / "overlay_regions.mp4").exists()
    rows = (tmp_path / "auto" / "big" / "tracks.csv").read_text().splitlines()
    assert rows[0].endswith("region,zoom,geometry,detector_agrees,track_x,track_y") and rows[1].split(",")[-6:-4] == ["0", "1"]


def test_all_animals_number_counts_boxes_not_rows(tmp_path):
    """summary.json: a frame counts as 'all animals' only when every counted animal has a BOX there (a bare position is not found)."""
    pytest.importorskip("cv2")
    from animalworld.zeroshot import pipeline as P
    from test_zeroshot import _video
    v = _video(tmp_path / "pair.mp4", 2, 30)
    job = P.Job(v, tmp_path / "o", P.Options(engine="mock", zoom="off"))
    P.guarded(P.prepare_and_count, job, {"engine": "mock", "device": "CPU"})
    job.n = 2
    res = P.track_simple(job, 2)
    for g in range(0, len(res["frames"]), 2):                          # every second frame: one animal is a position without a box
        if res["frames"][g]:
            res["frames"][g][0].update(box=None, mask=None, kp=None, source="interpolated")
    summ = P.write_outputs(job, res, 2)
    assert summ["frames_with_all_animals_pct"] <= 55 and summ["rows_with_box"] < summ["rows"]
    assert summ["jitter"]["transitions"] > 0 and "alternation" in summ["jitter"] and summ["status"] == "ok"


def test_region_file(tmp_path):
    from animalworld.zeroshot import zoom as Z
    f = tmp_path / "r.json"; f.write_text(json.dumps([{"box": [0, 0, 320, 240], "animals": 1}, {"box": [320, 0, 640, 240], "animals": 2}]))
    r = Z.load_region_file(f, 640, 240)
    assert [x["animals"] for x in r] == [1, 2] and r[1]["box"] == [320, 0, 640, 240]
    f.write_text(json.dumps([{"box": [0, 0, 900, 240]}]))
    with pytest.raises(ValueError):
        Z.load_region_file(f, 640, 240)


def test_fallback_positions_when_the_tracker_cannot_start(tmp_path):
    """No frame with all animals apart: the window still gets positions (detector + simple tracker), to be flagged by the caller."""
    from animalworld.zeroshot import pipeline as P
    T, N = 12, 2
    t, box, c = [], [], []
    for f in range(T):
        for k in range(N):
            x = 100 + 6 * f + 14 * k; t.append(f); box.append([x - 10, 40, x + 10, 60]); c.append([x, 50])       # two animals always touching
    d = tmp_path / "det" / "work" / f"video__w0000_000000_{T - 1:06d}"; d.mkdir(parents=True)
    np.savez(d / "det.npz", t=np.array(t), box=np.array(box, float), score=np.full(len(t), 0.9), c=np.array(c, float), area=np.full(len(t), 300),
             mbuf=np.zeros(0, np.uint8), moff=np.zeros(len(t) + 1, int), T=T, W=640, H=240)
    job = P.Job(tmp_path / "v.mp4", tmp_path / "o", P.Options(engine="mock"))
    X, bl = P._fallback_positions(job, 0, T, N, tmp_path / "det", False, 1)
    assert X.shape == (T, N, 2) and np.isfinite(X).all() and bl > 0
    assert np.allclose(sorted(X[5, :, 0]), [130, 144])
    assert P._fallback_positions(job, 3, T, N, tmp_path / "det", False, 1) == (None, 0.0)                       # no detections at all: nothing to fall back on


def test_masks_only_overlay_is_written(tmp_path):
    pytest.importorskip("cv2")
    from animalworld.cli import main
    from test_zeroshot import _video
    v = _video(tmp_path / "m.mp4", 2, 30)
    assert main(["zeroshot", str(v), "--out", str(tmp_path / "o"), "--yes", "--engine", "mock"]) == 0
    assert (tmp_path / "o" / "m" / "overlay_masks.mp4").stat().st_size > 1000
    assert json.loads((tmp_path / "o" / "m" / "summary.json").read_text())["files"]["overlay_masks"] == "overlay_masks.mp4"


def test_a_detector_box_on_a_reflection_is_rejected_not_drawn():
    """One identity, stable rows, then a filled row whose detector box sits a body length away (a reflection): rejected."""
    from animalworld.zeroshot import pipeline as P
    BL = 140.0
    frames = [[dict(id=1, x=1548.0 + g, y=826.0, box=[1500, 800, 1600, 850], geom="sam2", source="tracked", mask=None, kp=None, kc=None, score=0.9, status="ok")] for g in range(5)]
    frames.append([dict(id=1, x=1586.0, y=690.0, box=[1540, 660, 1640, 720], geom="detector", source="filled", mask=None, kp=None, kc=None, score=0.5, status="ok")])
    frames.append([dict(id=1, x=1554.0, y=826.0, box=[1500, 800, 1600, 850], geom="sam2", source="filled", mask=None, kp=None, kc=None, score=0.9, status="ok")])
    assert P.reject_jumps(frames, BL, 30.0) == 1
    bad = frames[5][0]
    assert bad["box"] is None and bad["geom"] == "rejected" and bad["status"] == "flagged" and (bad["x"], bad["y"]) == (1552.0, 826.0)
    assert frames[6][0]["box"] is not None and P.reject_jumps(frames, BL, 30.0) == 0


def test_empty_camera_half_is_left_out(tmp_path):
    """A 2 : 1 frame with no visible step in the middle, animals large enough, and nothing at all in the right half: the right
    half is dropped (deviation 1 from the pre-registration)."""
    cv2 = pytest.importorskip("cv2")
    from animalworld.cli import main
    W, H = 640, 240
    vw = cv2.VideoWriter(str(tmp_path / "half.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    for f in range(45):
        im = np.full((H, W, 3), 160, np.uint8)
        cv2.ellipse(im, (60 + 3 * f, 120 + int(30 * np.sin(f / 6.0))), (30, 16), 0, 0, 360, (25, 25, 25), -1, cv2.LINE_AA)
        vw.write(im)
    vw.release()
    assert main(["zeroshot", str(tmp_path / "half.mp4"), "--out", str(tmp_path / "o"), "--yes", "--engine", "mock"]) == 0
    s = json.loads((tmp_path / "o" / "half" / "summary.json").read_text())
    assert s["zoom"]["used"] and len(s["regions"]) == 1 and s["regions"][0]["box"][2] <= 320 and s["zoom"]["frame_used_pct"] <= 50


# ---------------------------------------------------------------- regression tests for the bugs of the first collaborator pass
def test_split_count_always_sums_to_the_confirmed_total_and_follows_current_support():
    """Bug 1: with a smaller confirmed total, the old rule ranked the regions once by the support of their PROPOSED count and
    emptied a region that clearly holds an animal (gfp_nor_5boxes_objects [3, 1, 2, 2] -> [0, 1, 2, 2]). Now one animal at a
    time is taken where the CURRENT count is seen least often; the split always sums to the confirmed total."""
    from animalworld.zeroshot import zoom as Z
    pad = lambda s: s + [0.0] * (16 - len(s))                                   # noqa: E731
    five = [dict(box=[0, 0, 200, 200], animals=a, share=s[a - 1], support=pad(s)) for a, s in
            ((3, [1.0, 0.95, 0.3]), (1, [0.9]), (2, [1.0, 0.4]), (2, [1.0, 0.5]))]
    assert Z.split_count(five, 5) == [2, 1, 1, 1]                               # old rule: [0, 1, 2, 2]
    # two mice in one camera, a sporadic detection in the other: the region with the weak single animal goes, not the pair
    two = [dict(box=[0, 0, 640, 480], animals=2, share=0.8, support=pad([1.0, 0.8])), dict(box=[640, 270, 820, 480], animals=1, share=0.3, support=pad([0.3]))]
    assert Z.split_count(two, 2) == [2, 0]
    one_each = [dict(box=[0, 0, 640, 480], animals=2, share=0.2, support=pad([1.0, 0.2])), dict(box=[640, 0, 1280, 480], animals=1, share=0.9, support=pad([0.9]))]
    assert Z.split_count(one_each, 2) == [1, 1]
    rng = np.random.default_rng(1)
    for _ in range(200):
        k = int(rng.integers(1, 6))
        regs = [dict(box=[0, 0, int(rng.integers(50, 500)), int(rng.integers(50, 500))], animals=int(rng.integers(1, 4)), share=float(rng.uniform()),
                     support=sorted(rng.uniform(size=16).round(2).tolist(), reverse=True)) for _ in range(k)]
        for N in range(1, 9):
            got = Z.split_count(regs, N)
            assert sum(got) == N and all(v >= 0 for v in got)


def test_regions_carry_their_per_frame_counts_and_support():
    from animalworld.zeroshot import zoom as Z
    dets = []
    for t in range(12):
        dets.append(dict(t=t, box=[190, 195, 210, 205], score=0.9, area=120.0, c=(200, 200)))
        if t % 3 == 0:                                                 # a second, intermittent detection (reflection) beside it
            dets.append(dict(t=t, box=[230, 195, 250, 205], score=0.9, area=120.0, c=(240, 200)))
    regs = Z.find_regions(dets, 12, 1280, 480, 20.0, dict(x=None, y=None))
    assert len(regs) == 1 and regs[0]["animals"] == 2 and list(regs[0]["counts"])[:3] == [2, 1, 1]
    s = Z.support_of(regs[0]["counts"])
    assert s[0] == 1.0 and abs(s[1] - 4 / 12) < 1e-3 and len(s) == 16 and Z.support_of(None) is None


def test_an_empty_detection_file_has_the_shapes_the_tracker_indexes():
    """Bug 2: a region with no detection wrote det.npz with c of shape (0,) and the vendored core crashed (safe_frames:
    'axes don't match array'). The empty file now has box (0, 4) and c (0, 2); the core returns no safe frame instead of crashing,
    and the pipeline counts zero confident detections (and then skips the SAM steps: the region is dropped)."""
    import sys
    from pathlib import Path
    from animalworld.zeroshot import pipeline as P
    src = (Path(P.__file__).parent / "mwtrack" / "detect_dense.py").read_text()
    assert "reshape(-1, 2)" in src and "reshape(-1, 4)" in src
    sys.path.insert(0, str(Path(P.__file__).parent / "mwtrack"))
    try:
        import core_base as CB
    finally:
        sys.path.pop(0)
    D = dict(t=np.zeros(0, int), box=np.zeros((0, 4)), score=np.zeros(0), c=np.zeros((0, 2)), area=np.zeros(0, int), T=10, W=64, H=64,
             mbuf=np.zeros(0, np.uint8), moff=np.zeros(1, int))
    assert CB.safe_frames(D, 1, 20.0) == [] and CB.safe_frames(D, 2, 20.0) == []
    assert P.count_rule(D)[0] == 0


def test_confident_detection_count(tmp_path):
    from animalworld.zeroshot import pipeline as P
    assert P.confident_detections(tmp_path) == -1
    for k, sc in enumerate(([], [0.2, 0.39], [0.9, 0.41, 0.1])):
        d = tmp_path / f"w{k}"; d.mkdir()
        np.savez(d / "det.npz", score=np.array(sc, float), t=np.zeros(len(sc), int))
    assert P.confident_detections(tmp_path) == 2 and P.confident_detections(tmp_path, 0.15) == 4


def test_a_region_without_any_animal_is_dropped_not_fatal(tmp_path):
    """Bug 2: a region (here given in a region file) that holds no animal on any frame is dropped with a warning; the run
    finishes with the other region, and the identities stay unique."""
    cv2 = pytest.importorskip("cv2")
    from animalworld.cli import main
    v, truth = _two_arena_video(tmp_path / "pair.mp4", small=False)
    # the right half only: repaint it empty
    cap = cv2.VideoCapture(str(v)); frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        im[:, 320:] = 170; frames.append(im)
    vw = cv2.VideoWriter(str(tmp_path / "one.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 240))
    for im in frames:
        vw.write(im)
    vw.release()
    rf = tmp_path / "regions.json"; rf.write_text(json.dumps([{"box": [0, 0, 320, 240], "animals": 1}, {"box": [320, 0, 640, 240], "animals": 1}]))
    assert main(["zeroshot", str(tmp_path / "one.mp4"), "--out", str(tmp_path / "o"), "--yes", "--n-animals", "2", "--engine", "mock", "--zoom", str(rf)]) == 0
    s = json.loads((tmp_path / "o" / "one" / "summary.json").read_text())
    assert len(s["regions"]) == 1 and s["regions"][0]["box"] == [0, 0, 320, 240]
    assert any("region is dropped" in w for w in s["warnings"])


def test_pose_crop_enlargement_is_bicubic_and_shrinking_unchanged():
    """Bug 5: pose_frames.py passed INTER_AREA to warpAffine; OpenCV's warp functions treat INTER_AREA as INTER_LINEAR, so the
    enlargement was bilinear (byte-identical crops). Measured on the demo clips shrunk to 20-60 px animals, bicubic enlargement agrees
    better with the full-size answer (78-98 % vs 76-95 % of keypoints within 0.1 body length), so an enlarged crop is now bicubic;
    a shrunk crop stays bilinear (unchanged for large animals)."""
    cv2 = pytest.importorskip("cv2")
    from pathlib import Path
    from animalworld.zeroshot import pipeline as P
    rng = np.random.default_rng(0)
    im = cv2.GaussianBlur(rng.integers(0, 255, (120, 160, 3)).astype(np.uint8), (5, 5), 0)
    A = np.float32([[256 / 30, 0, -40 * 256 / 30], [0, 256 / 30, -50 * 256 / 30]])
    a = cv2.warpAffine(im, A, (256, 256), flags=cv2.INTER_AREA); b = cv2.warpAffine(im, A, (256, 256), flags=cv2.INTER_LINEAR)
    assert np.array_equal(a, b)
    src = (Path(P.__file__).parent / "mwtrack" / "pose_frames.py").read_text()
    assert "flags=cv2.INTER_AREA" not in src and "flags=cv2.INTER_CUBIC if side < S else cv2.INTER_LINEAR" in src


def test_a_straight_edge_at_the_middle_row_of_one_camera_is_not_a_seam(tmp_path):
    """Deviation 3: the seam rule fired on our own demo clip mouse_x2_cage (1920 x 1080, a cage edge exactly at the middle row)
    and confined two freely moving mice to two halves. A seam now needs two halves that are each an ordinary camera image."""
    cv2 = pytest.importorskip("cv2")
    from animalworld.zeroshot import zoom as Z
    assert Z.seam_halves_ok(1280, 480, "x") and not Z.seam_halves_ok(1280, 480, "y")
    assert not Z.seam_halves_ok(1920, 1080, "y") and not Z.seam_halves_ok(2448, 2048, "y")
    # 21 Nov: portrait halves are not camera images either (public clip damm_track golden_mouse_operant_chamber_..._09, 800 x 600:
    # the middle column was taken for a seam and the new default cost 5 identity fixes on that clip; whole frame: 0)
    assert not Z.seam_halves_ok(800, 600, "x") and not Z.seam_halves_ok(1920, 1080, "x") and not Z.seam_halves_ok(2448, 2048, "x")
    assert Z.seam_halves_ok(2560, 720, "x") and Z.seam_halves_ok(640, 960, "y")
    paths = []
    for k in range(12):
        im = np.full((108, 192, 3), 90, np.uint8); im[54:] = 200                 # hard horizontal step at the middle row of a 16 : 9 frame
        p = tmp_path / f"{k}.jpg"; cv2.imwrite(str(p), im); paths.append(p)
    assert Z.find_seams(paths) == {"x": None, "y": None}
    for k, p in enumerate(paths):
        im = np.full((48, 128, 3), 90, np.uint8); im[:, 64:] = 200               # two 4 : 3 cameras side by side
        cv2.imwrite(str(p), im)
    assert Z.find_seams(paths)["x"] == 64


def test_a_stable_mask_that_does_not_contain_the_identity_is_not_used():
    """Found in the regression run on our demo clip mouse_x2_cage: two mice touching at the start, identity 1's propagated mask
    sat on identity 2 (centre 0.56 body length away, under the 0.6 repair distance) and replaced the tracker's position: one
    swap fix. A SAM row is now used only if the identity's own position lies inside the mask's box (+ 0.1 body length)."""
    from animalworld.zeroshot import pipeline as P
    BL = 170.0
    assert not P.sam_row_consistent((504.4, 213.5), [380, 107, 543, 168], BL)            # frame 0, identity 1 of mouse_x2_cage
    assert P.sam_row_consistent((480.7, 142.7), [381, 111, 553, 262], BL)
    assert P.sam_row_consistent((545.0, 170.0), [380, 107, 543, 168], BL)                # within the margin


def test_frames_between_two_analysed_frames_follow_their_consistency():
    """A filled frame keeps the stable mask unless one of its two neighbouring analysed frames failed the containment test;
    the filled frame's own detector box is not used for the test (on gcamp_5trial_table_hires a wrong detector box between two
    good analysed frames had replaced a correct mask)."""
    from animalworld.zeroshot import pipeline as P
    tr = [(0, True), (3, True), (6, False), (9, True)]
    assert P.bracket_consistent(tr, 0) and P.bracket_consistent(tr, 1) and P.bracket_consistent(tr, 3)
    assert not P.bracket_consistent(tr, 4) and not P.bracket_consistent(tr, 6) and not P.bracket_consistent(tr, 8)
    assert P.bracket_consistent(tr, 10) and P.bracket_consistent([], 5)


def test_a_middle_line_that_animals_cross_is_not_a_camera_seam():
    """Deviation 4: on the public clip damm_track golden_mouse_operant_chamber_..._09 (800 x 600, one camera, two mice) a straight
    edge on the middle column passed the seam test and the chamber was cut in two. Animals walk across it; across a real seam
    between two cameras only an over-stretched box (much larger than an animal) can lie."""
    from animalworld.zeroshot import zoom as Z
    D = lambda t, x0, x1: dict(t=t, box=[x0, 100, x1, 160], score=0.9, c=((x0 + x1) / 2, 130))       # noqa: E731
    walk = [D(t, 300 + 10 * t, 420 + 10 * t) for t in range(20)]          # 120 px animal walking across x = 400
    assert Z.crossings(walk, 400, "x", 20) == 5                          # frames with >= 30 % of the body on each side
    stretched = [D(t, 100, 700) for t in range(20)] + [D(t, 100, 220) for t in range(20)] + [D(t, 500, 620) for t in range(20)]
    assert Z.crossings(stretched, 400, "x", 20) == 0                       # a box over both cameras is not an animal
    edge = [D(t, 380, 500) for t in range(20)]                             # an animal touching the line from one side
    assert Z.crossings(edge, 400, "x", 20) == 0

import json

import pytest
from PIL import Image

from animalworld import pipeline
from animalworld.backends.proposers import pose as P
from animalworld.formats import coco, dlc, sleap, yolo
from animalworld.formats import labels as L
from animalworld.stages import review as R

KP = ["nose", "left_ear", "right_ear", "tail_base"]


def docs():
    a = L.instance("mouse", [10, 20, 60, 80], [[10, 20], [60, 20], [60, 80], [10, 80]],
                   {"nose": [12, 22, 2], "left_ear": [20, 30, 2], "right_ear": [25, 28, 1], "tail_base": [58, 78, 2]}, track_id=0)
    b = L.instance("mouse", [100, 50, 150, 90], [[100, 50], [150, 50], [125, 90]],
                   {"nose": [101, 51, 2], "tail_base": [149, 89, 2]}, track_id=1)
    return [L.label("img_a", "img_a.jpg", 200, 100, [a, b], "t", status="accepted"),
            L.label("img_b", "img_b.jpg", 200, 100, [a], "t", status="accepted"),
            L.label("img_c", "img_c.jpg", 200, 100, [], "t", status="accepted")]


def same_boxes(x, y, tol=0.05):
    assert len(x) == len(y)
    for i, j in zip(x, y):
        assert len(i["instances"]) == len(j["instances"])
        for p, q in zip(i["instances"], j["instances"]):
            assert all(abs(u - v) <= tol for u, v in zip(p["box"], q["box"]))


def test_coco_roundtrip_boxes_masks_keypoints():
    c = coco.to_coco(docs(), ["mouse"], KP, [[0, 1], [0, 2]])
    assert len(c["images"]) == 3 and len(c["annotations"]) == 3 and c["categories"][0]["keypoints"] == KP
    assert c["annotations"][0]["bbox"] == [10, 20, 50, 60] and c["annotations"][0]["num_keypoints"] == 4
    assert c["annotations"][1]["num_keypoints"] == 2 and c["categories"][0]["skeleton"] == [[1, 2], [1, 3]]
    back = coco.from_coco(json.loads(json.dumps(c)))
    same_boxes(docs(), back)
    assert back[0]["instances"][0]["poly"] == docs()[0]["instances"][0]["poly"]
    assert back[0]["instances"][0]["keypoints"]["right_ear"] == [25.0, 28.0, 1]
    assert back[0]["instances"][1]["keypoints"]["left_ear"][2] == 0 and back[0]["id"] == "img_a"


@pytest.mark.parametrize("task", ["detect", "segment", "pose"])
def test_yolo_roundtrip(task):
    for d in docs():
        txt = yolo.to_lines(d, ["mouse"], task, KP)
        back = yolo.from_lines(txt, d["W"], d["H"], ["mouse"], task, KP)
        same_boxes([d], [dict(d, instances=back)], tol=0.01)
        if task == "segment" and d["instances"]:
            assert all(abs(u - v) < 0.01 for p, q in zip(back[0]["poly"], d["instances"][0]["poly"]) for u, v in zip(p, q))
        if task == "pose" and d["instances"]:
            k = back[0]["keypoints"]
            assert abs(k["nose"][0] - 12) < 0.01 and k["nose"][2] == 2 and k["right_ear"][2] == 1
    assert yolo.to_lines(docs()[2], ["mouse"], "segment") == ""
    line = yolo.to_lines(docs()[0], ["mouse"], "pose", KP).splitlines()[1].split()
    assert len(line) == 5 + 3 * len(KP) and line[8:11] == ["0", "0", "0"]        # unlabelled keypoint -> 0 0 0


def test_yolo_data_yaml(tmp_path):
    import yaml as Y
    d = Y.safe_load(yolo.data_yaml(tmp_path, ["mouse"], "pose", KP, [0, 2, 1, 3]))
    assert d["kpt_shape"] == [4, 3] and d["flip_idx"] == [0, 2, 1, 3] and d["names"] == {0: "mouse"} and d["train"] == "train/images"


def test_dlc_roundtrip_multi_and_single(tmp_path):
    for d in docs():
        Image.new("RGB", (200, 100)).save(tmp_path / d["file"])
    res = dlc.export_dlc(docs(), KP, tmp_path / "out", tmp_path, video="vid")
    assert res["multi_animal"] and res["images"] == 3
    rows = open(res["csv"]).read().splitlines()
    assert rows[0].startswith("scorer,,,animalworld") and rows[1].startswith("individuals,,,animal1") and rows[3].startswith("coords,,,x,y")
    assert rows[4].startswith("labeled-data,vid,img_a.jpg,12.00,22.00")
    back = dlc.import_dlc(res["csv"])
    assert back["keypoints"] == KP and len(back["docs"]) == 3
    k = back["docs"][0]["instances"][1]["keypoints"]
    assert k["nose"] == [101.0, 51.0, 2] and k["left_ear"][2] == 0
    assert back["docs"][0]["W"] == 200 and len(back["docs"][2]["instances"]) == 0
    one = dlc.export_dlc(docs()[1:2], KP, tmp_path / "single", tmp_path, video="v1")
    assert not one["multi_animal"] and open(one["csv"]).read().splitlines()[1].startswith("bodyparts,,,nose,nose")
    b1 = dlc.import_dlc(one["csv"])
    assert b1["docs"][0]["instances"][0]["keypoints"]["tail_base"] == [58.0, 78.0, 2]
    # legacy single-column index
    legacy = tmp_path / "CollectedData_old.csv"
    legacy.write_text("scorer,s,s,s,s\nbodyparts,nose,nose,tail,tail\ncoords,x,y,x,y\nlabeled-data/v/img_a.jpg,1,2,,\n")
    lg = dlc.import_dlc(legacy)
    assert lg["keypoints"] == ["nose", "tail"] and lg["docs"][0]["id"] == "img_a" and lg["docs"][0]["instances"][0]["keypoints"]["tail"][2] == 0


def test_sleap_json_route_roundtrip(tmp_path):
    for d in docs():
        Image.new("RGB", (200, 100)).save(tmp_path / d["file"])
    res = sleap.export_sleap(docs(), ["mouse"], KP, [[0, 3]], tmp_path / "sl", tmp_path, prefer_slp=False)
    assert res["route"] == "coco-json"
    back = sleap.import_sleap(res["coco_json"])
    assert back["keypoints"] == KP
    same_boxes(docs(), back["docs"])
    assert back["docs"][0]["instances"][0]["keypoints"]["nose"] == [12.0, 22.0, 2] and back["docs"][0]["status"] == "accepted"


def test_sleap_slp_route_if_sleap_io_installed(tmp_path):
    pytest.importorskip("sleap_io")
    for d in docs():
        Image.new("RGB", (200, 100)).save(tmp_path / d["file"])
    res = sleap.export_sleap(docs(), ["mouse"], KP, [[0, 3]], tmp_path / "sl", tmp_path)
    assert res["route"] == "slp"
    back = sleap.import_sleap(res["slp"])
    assert back["keypoints"] == KP and len(back["docs"]) == 3
    k = back["docs"][0]["instances"][0]["keypoints"]
    assert abs(k["nose"][0] - 12) < 1e-3 and abs(k["tail_base"][1] - 78) < 1e-3
    assert back["docs"][0]["instances"][1]["keypoints"]["left_ear"][2] == 0


def test_pose_crop_and_decode_are_inverse():
    im = Image.new("RGB", (300, 200), (10, 10, 10))
    box = [100, 60, 180, 100]
    crop, X0, Y0, side = P.crop_square(im, box, 1.3)
    assert crop.size == (256, 256) and abs(side - 104) < 1e-6 and abs(X0 - 88) < 1e-6
    kp_frame = {"nose": [110.0, 70.0, 2]}
    in_crop = {"nose": [(110 - X0) * 256 / side, (70 - Y0) * 256 / side, 2]}
    out = P.decode(in_crop, X0, Y0, side)
    assert abs(out["nose"][0] - 110) < 1e-6 and abs(out["nose"][1] - 70) < 1e-6
    fake = lambda c: [{"box": [0, 0, 20, 20], "score": 0.99, "keypoints": {"nose": [5, 5, 2]}},        # noqa: E731
                      {"box": [100, 100, 156, 156], "score": 0.5, "keypoints": in_crop}]
    res = P.pose_for_boxes(im, [L.instance("mouse", box)], fake)
    assert abs(res[0]["keypoints"]["nose"][0] - 110) < 0.01          # the central instance wins over the higher score
    inst = L.instance("mouse", box, None, {"nose": [110, 70, 2], "tail_base": [400, 400, 2]})
    c, line = yolo.pose_crop(im, inst, ["nose", "tail_base"])
    v = line.split()
    assert abs(float(v[5]) - (110 - X0) / side) < 1e-4 and v[8:11] == ["0", "0", "0"]


def test_project_label_exchange_dlc_sleap_coco(reviewed_project):
    pr = reviewed_project
    n = R.counts(pr)["accepted"]
    d = R.export_labels(pr, "dlc")
    s = R.export_labels(pr, "sleap")
    c = R.export_labels(pr, "coco")
    assert d["images"] == n and s["images"] == n and c["images"] == n
    before = {x["id"]: x for x in R.accepted_docs(pr)}
    # labels corrected in DeepLabCut come back: move every nose by +3 px and re-import
    rows = open(d["csv"]).read().splitlines()
    head, body = rows[:4], rows[4:]
    moved = []
    for line in body:
        f = line.split(",")
        if f[3]:
            f[3] = f"{float(f[3]) + 3:.2f}"
        moved.append(",".join(f))
    open(d["csv"], "w").write("\n".join(head + moved) + "\n")
    r = R.import_labels(pr, "dlc", d["csv"])
    assert r["imported"] == n and r["images_not_found"] == 0
    after = {x["id"]: x for x in R.accepted_docs(pr)}
    some = next(i for i in before if before[i]["instances"])
    assert abs(after[some]["instances"][0]["keypoints"]["nose"][0] - before[some]["instances"][0]["keypoints"]["nose"][0] - 3) < 0.02
    assert after[some]["origin"] == "import"
    r = R.import_labels(pr, "sleap", s["coco_json"])
    assert r["imported"] == n
    r = R.import_labels(pr, "coco", c["file"])
    assert r["imported"] == n
    r = R.import_labels(pr, "yolo", pr.path("dataset", "train", "labels"), pr.path("dataset", "train", "images"))
    assert r["imported"] == len(list(pr.path("dataset", "train", "labels").glob("*.txt")))
    assert pipeline.run_stage(pr, "dataset")["mode"] == "executed"

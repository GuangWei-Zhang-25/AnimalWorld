"""ONNX Runtime predictor for exported YOLO (detect / segment / pose) and RF-DETR models.

Pre/post-processing follows the lab's ONNX app (mouseworld_app/src/mouseworld/ops.py, AGPL-3.0):
  YOLO     letterbox to the model's input size (pad value 114), RGB, CHW, /255;
           output0 (1, 4 + nc [+ nm | + 3K], A): cx, cy, w, h in input pixels, class scores, then mask
           coefficients (segment) or K keypoints x, y, conf (pose); protos (1, nm, mh, mw); NMS IoU 0.7.
           End-to-end exports (1, N, 6+: x0, y0, x1, y1, score, class, ...) are also read.
  RF-DETR  plain resize to the input size (no letterbox), /255, ImageNet mean / std;
           boxes (1, Q, 4) cx, cy, w, h as fractions of the image; logits (1, Q, C), score = sigmoid.
Letterboxing uses PIL here (the app uses OpenCV), so numbers can differ from Ultralytics in the last
decimals.

Set ORT_DISABLE_TELEMETRY=1 before starting Python if your site requires it; this module also sets
it (if unset) before importing onnxruntime.
"""
from __future__ import annotations

import ast
import os
from typing import Any, Dict, List

import numpy as np

from ...formats import labels as L
from ...registry import BackendError, need
from .base import Predictor, nms

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox(image, size_wh, pad=114):
    """-> (float32 CHW array in 0..1, ratio, pad_x, pad_y)."""
    from PIL import Image
    tw, th = size_wh
    W, H = image.size
    r = min(tw / W, th / H)
    nw, nh = max(1, int(round(W * r))), max(1, int(round(H * r)))
    canvas = Image.new("RGB", (tw, th), (pad, pad, pad))
    px, py = (tw - nw) // 2, (th - nh) // 2
    canvas.paste(image.resize((nw, nh), Image.BILINEAR), (px, py))
    arr = np.asarray(canvas).astype(np.float32).transpose(2, 0, 1) / 255.0
    return arr, r, px, py


def decode_yolo(out0: np.ndarray, nc: int, task: str, conf: float, iou_thr: float = 0.7):
    """out0 (C, A) or end-to-end (N, 6+) -> (boxes xyxy in input pixels, scores, classes, extra columns)."""
    a = out0
    if a.shape[0] < a.shape[1]:                # (C, A) -> (A, C)
        a = a.T
        cls_scores = a[:, 4:4 + nc]
        scores = cls_scores.max(1)
        classes = cls_scores.argmax(1)
        m = scores >= conf
        a, scores, classes = a[m], scores[m], classes[m]
        boxes = np.stack([a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2, a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2], 1)
        extra = a[:, 4 + nc:]
        keep = nms(boxes, scores, iou_thr)
        return boxes[keep], scores[keep], classes[keep], extra[keep]
    m = a[:, 4] >= conf                         # end-to-end head: already NMS-free
    a = a[m]
    return a[:, :4], a[:, 4], a[:, 5].astype(int), a[:, 6:]


class ONNXPredictor(Predictor):
    name = "onnx"
    tested = False

    def __init__(self, options: Dict[str, Any] = None, **kw: Any):
        super().__init__(options, **kw)
        w = self.opt.get("weights")
        if not w or not os.path.exists(str(w)):
            raise BackendError(f"onnx predictor: weights file not found: {w}")
        os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
        ort = need("onnxruntime", "deploy", "the ONNX predictor")
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(self.opt.get("threads", 2))
        self.sess = ort.InferenceSession(str(w), sess_options=so, providers=self.opt.get("providers") or ["CPUExecutionProvider"])
        meta = self.sess.get_modelmeta().custom_metadata_map or {}
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        shp = inp.shape
        side = int(self.opt.get("imgsz", 640))
        self.th = int(shp[2]) if isinstance(shp[2], int) else side
        self.tw = int(shp[3]) if isinstance(shp[3], int) else side
        self.task = self.opt.get("task") or meta.get("task")
        self.names = {0: self.category}
        if meta.get("names"):
            try:
                self.names = {int(k): str(v) for k, v in ast.literal_eval(meta["names"]).items()}
            except (ValueError, SyntaxError):
                pass
        self.kpt_shape = None
        if meta.get("kpt_shape"):
            try:
                self.kpt_shape = list(ast.literal_eval(meta["kpt_shape"]))
            except (ValueError, SyntaxError):
                pass
        outs = self.sess.get_outputs()
        self.kind = self.opt.get("model_type")
        if not self.kind:
            is_detr = self.task is None and any(len(o.shape) == 3 and o.shape[-1] == 4 for o in outs) and len(outs) >= 2
            self.kind = "rfdetr" if is_detr else "yolo"
        if self.kind == "yolo" and not self.task:
            self.task = "segment" if len(outs) > 1 else "detect"

    # ---- YOLO ---------------------------------------------------------------------------------
    def _yolo(self, image) -> List[dict]:
        arr, r, px, py = letterbox(image, (self.tw, self.th))
        outs = self.sess.run(None, {self.input_name: arr[None]})
        nc = len(self.names)
        boxes, scores, classes, extra = decode_yolo(outs[0][0], nc, self.task, self.conf, float(self.opt.get("iou", 0.7)))
        W, H = image.size
        un = lambda x, y: ((x - px) / r, (y - py) / r)                  # noqa: E731
        inst = []
        for k in range(len(boxes)):
            x0, y0 = un(boxes[k, 0], boxes[k, 1])
            x1, y1 = un(boxes[k, 2], boxes[k, 3])
            box = [float(np.clip(x0, 0, W)), float(np.clip(y0, 0, H)), float(np.clip(x1, 0, W)), float(np.clip(y1, 0, H))]
            if box[2] - box[0] < 1 or box[3] - box[1] < 1:
                continue
            kps, poly = None, None
            if self.task == "pose" and extra.shape[1] >= 3:
                K = extra.shape[1] // 3
                kk = extra[k, :K * 3].reshape(K, 3)
                names = self.kp_names if len(self.kp_names) == K else [f"kp{j}" for j in range(K)]
                kps = {}
                for j, n in enumerate(names):
                    x, y = un(kk[j, 0], kk[j, 1])
                    kps[n] = [float(x), float(y), 2 if kk[j, 2] >= float(self.opt.get("kp_conf", 0.5)) else 1]
            elif self.task == "segment" and len(outs) > 1 and extra.shape[1]:
                poly = self._mask_poly(extra[k], outs[1][0], boxes[k], un)
            inst.append(L.instance(self.names.get(int(classes[k]), self.category), box, poly, kps, score=float(scores[k])))
        return inst

    def _mask_poly(self, coef, protos, box_in, un):
        try:
            from ..proposers.base import mask_to_polygon
            nm, mh, mw = protos.shape
            m = 1.0 / (1.0 + np.exp(-(coef[:nm] @ protos.reshape(nm, -1)).reshape(mh, mw)))
            sx, sy = mw / self.tw, mh / self.th
            crop = np.zeros_like(m, dtype=bool)
            x0, y0, x1, y1 = int(box_in[0] * sx), int(box_in[1] * sy), int(np.ceil(box_in[2] * sx)), int(np.ceil(box_in[3] * sy))
            crop[max(0, y0):y1, max(0, x0):x1] = True
            poly, _, _ = mask_to_polygon((m > 0.5) & crop, eps=0.5)
            if poly is None:
                return None
            return [list(un((x + 0.5) / sx, (y + 0.5) / sy)) for x, y in poly]
        except BackendError:
            return None                        # OpenCV not installed: boxes only

    # ---- RF-DETR ------------------------------------------------------------------------------
    def _rfdetr(self, image) -> List[dict]:
        from PIL import Image
        W, H = image.size
        a = np.asarray(image.resize((self.tw, self.th), Image.BILINEAR)).astype(np.float32) / 255.0
        a = ((a - MEAN) / STD).transpose(2, 0, 1)[None]
        outs = self.sess.run(None, {self.input_name: a})
        boxes = next(o for o in outs if o.ndim == 3 and o.shape[-1] == 4)[0]
        logits = next(o for o in outs if o.ndim == 3 and o.shape[-1] != 4)[0]
        sc = 1.0 / (1.0 + np.exp(-logits))
        scores, classes = sc.max(1), sc.argmax(1)
        m = scores >= self.conf
        boxes, scores, classes = boxes[m], scores[m], classes[m]
        xyxy = np.stack([(boxes[:, 0] - boxes[:, 2] / 2) * W, (boxes[:, 1] - boxes[:, 3] / 2) * H,
                         (boxes[:, 0] + boxes[:, 2] / 2) * W, (boxes[:, 1] + boxes[:, 3] / 2) * H], 1)
        keep = nms(xyxy, scores, float(self.opt.get("iou", 0.5)))
        return [L.instance(self.category, [float(v) for v in np.clip(xyxy[k], 0, [W, H, W, H])], None, None, score=float(scores[k]))
                for k in keep]

    def predict(self, image) -> List[dict]:
        return self._rfdetr(image) if self.kind == "rfdetr" else self._yolo(image)

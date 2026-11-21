"""One geometry source for every output frame: a final SAM 2.1 propagation at the output frame rate.

Written for the package (not part of the frozen rule scripts). It runs AFTER the rule step has resolved the identities
at the analysis rate and never changes them: every identity is seeded from the detector's mask at a frame where the
resolved position and the detection agree, the masks are propagated through all output frames, and at every analysed
frame the mask of each identity is compared with the resolved position. Where a mask has left its identity (centre
further than 0.6 body lengths from the resolved position, or the mask is gone) the identity is anchored again at the
next frame where it has an unambiguous detection and the propagation is repeated from there. Frames that still
disagree are marked not ok, and the caller falls back to the detector's geometry for those rows.

usage: final_masks.py REQUEST.json OUT.npz
REQUEST: {"frames": dir with %06d.jpg (output frames 0 .. n-1), "n": n, "step": output frames per analysed frame,
          "N": identities, "BL": body length px, "X": [[[x, y] or null per identity] per analysed frame],
          "det": det.npz of the detector on these same frames (index t = output frame), "chunk": frames per pass}
OUT.npz:  frame, id (0-based), box (x0, y0, x1, y1), c (centroid), area, ok, mbuf / moff (packed mask inside the box),
          plus seeds, repairs, checked, bad (counts for the summary).
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import core_base as CB          # noqa: E402
import long_baselines as LB     # noqa: E402

LB.setup()
from sam2.build_sam import build_sam2_video_predictor   # noqa: E402

req = json.load(open(sys.argv[1]))
n, S, N, BL = int(req["n"]), int(req["step"]), int(req["N"]), float(req["BL"])
CH = int(req.get("chunk") or 400)
X = np.array([[p if p is not None else [np.nan, np.nan] for p in row] for row in req["X"]], float).reshape(-1, N, 2)
TA = len(X)
D = CB.load_det(req["det"])
by_t = {}
for i, t in enumerate(D["t"]):
    by_t.setdefault(int(t), []).append(i)
FR = Path(req["frames"])
NEAR, APART, MIN_AREA = 0.5 * BL, 1.0 * BL, 30


def attach(a):
    """Detections of analysed frame a attached to the identities by position: one-to-one, nearest first, within half a body length."""
    o = a * S
    cand = sorted((float(np.hypot(*(D["c"][d] - X[a, k]))), k, d) for k in range(N) if np.isfinite(X[a, k, 0]) for d in by_t.get(o, []))
    got, used = {}, set()
    for dist, k, d in cand:
        if dist <= NEAR and k not in got and d not in used:
            got[k] = d; used.add(d)
    return got


def alone(a, k):
    """Identity k is further than one body length from every other identity at analysed frame a."""
    if not np.isfinite(X[a, k, 0]):
        return False
    others = [j for j in range(N) if j != k and np.isfinite(X[a, j, 0])]
    return all(np.hypot(*(X[a, j] - X[a, k])) > APART for j in others)


ATT = [attach(a) for a in range(TA)]
cfg, ck = LB.CK["sam2_large"]
P = build_sam2_video_predictor(cfg, str(ck), device="cuda")
dt = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
out = dict(frame=[], id=[], box=[], c=[], area=[], ok=[]); buf = []; off = [0]
stats = dict(seeds=0, repairs=0, checked=0, bad=0)
prev = {}                          # identity -> mask on the first frame of the next pass (the last frame of the pass before)
c0 = 0
while c0 < n - 1 or (n == 1 and c0 == 0):
    c1 = min(n - 1, c0 + CH)
    wd = Path(tempfile.mkdtemp(prefix="final_"))
    for j, o in enumerate(range(c0, c1 + 1)):
        os.symlink(FR / f"{o:06d}.jpg", wd / f"{j:05d}.jpg")
    an = [a for a in range(TA) if c0 <= a * S <= c1]            # analysed frames inside this pass
    M = {}                                                       # local frame -> (N, H, W) bool

    def run(st, start, reverse, keep=None, lo=None, hi=None, span=None):
        for fl, ids, lg in P.propagate_in_video(st, start_frame_idx=int(start), reverse=reverse, max_frame_num_to_track=span):
            fl = int(fl)
            if (lo is not None and fl < lo) or (hi is not None and fl > hi):
                continue
            L = lg[:, 0].float(); best = L.argmax(0)
            cur = np.stack([((best == j) & (L[j] > 0)).cpu().numpy() for j in range(len(ids))])
            full = M.get(fl)
            if full is None:
                full = np.zeros((N,) + cur.shape[1:], bool); M[fl] = full
            for j, oid in enumerate(ids):
                k = int(oid) - 1
                if keep is None or k in keep:
                    full[k] = cur[j]

    def bad_at(a, k):
        m = M.get(a * S - c0)
        if m is None or not np.isfinite(X[a, k, 0]):
            return False
        ys, xs = np.nonzero(m[k])
        return len(xs) < MIN_AREA or float(np.hypot(xs.mean() - X[a, k, 0], ys.mean() - X[a, k, 1])) > 0.6 * BL

    with torch.inference_mode(), torch.autocast("cuda", dtype=dt):
        st = P.init_state(str(wd), offload_video_to_cpu=True, offload_state_to_cpu=True)
        seed = {}                                                # identity -> local seed frame
        for k, m in prev.items():
            if m.sum() >= MIN_AREA:
                P.add_new_mask(st, frame_idx=0, obj_id=k + 1, mask=m); seed[k] = 0
        for k in range(N):
            if k in seed:
                continue
            pick = [a for a in an if k in ATT[a] and alone(a, k)] or [a for a in an if k in ATT[a]]
            if pick:
                a = pick[0]
                P.add_new_mask(st, frame_idx=a * S - c0, obj_id=k + 1, mask=CB.det_mask(D, ATT[a][k])); seed[k] = a * S - c0; stats["seeds"] += 1
        if seed:
            first = min(seed.values())
            run(st, first, False)
            if max(seed.values()) > 0:
                run(st, max(seed.values()), True)
            # ---- repair: anchor again an identity whose mask left its resolved position
            fixed_until = {k: -1 for k in range(N)}
            for _ in range(8):
                todo = None
                for a in an:
                    ks = [k for k in seed if a > fixed_until[k] and bad_at(a, k)]
                    if ks:
                        todo = (a, ks); break
                if todo is None:
                    break
                a_bad, ks = todo
                done = False
                for a2 in [a for a in an if a >= a_bad]:
                    ok_k = [k for k in ks if k in ATT[a2] and alone(a2, k)]
                    if ok_k:
                        for k in ok_k:
                            P.add_new_mask(st, frame_idx=a2 * S - c0, obj_id=k + 1, mask=CB.det_mask(D, ATT[a2][k]))
                        run(st, a2 * S - c0, False)
                        last_good = max([a for a in an if a < a_bad] or [an[0]])
                        back = a2 * S - c0 - (last_good * S - c0 + 1)
                        if back > 0:
                            run(st, a2 * S - c0, True, keep=set(ok_k), lo=last_good * S - c0 + 1, span=back)
                        for k in ok_k:
                            fixed_until[k] = max(fixed_until[k], a2)          # one repair per incident; what is still wrong before a2 stays marked
                        stats["repairs"] += 1; done = True
                        for k in ks:
                            if k not in ok_k:
                                fixed_until[k] = max(fixed_until[k], a_bad)   # no clean frame for this one here; look again further on
                        break
                if not done:
                    for k in ks:
                        fixed_until[k] = an[-1]
        P.reset_state(st)
    # ---- rows of this pass (the first frame of a later pass repeats the last frame of the pass before: skipped)
    okA = {(a, k): (k in seed and not bad_at(a, k)) for a in an for k in range(N)}
    for a in an:
        for k in range(N):
            if np.isfinite(X[a, k, 0]) and k in seed:
                stats["checked"] += 1; stats["bad"] += not okA[(a, k)]
    for j in sorted(M):
        o = c0 + j
        if o == c0 and c0 > 0:
            continue
        a_lo, r = divmod(o, S)
        for k in range(N):
            ys, xs = np.nonzero(M[j][k])
            if len(xs) < MIN_AREA:
                continue
            good = okA.get((a_lo, k), False) and (r == 0 or okA.get((a_lo + 1, k), False))
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            out["frame"].append(o); out["id"].append(k); out["box"].append([x0, y0, x1, y1]); out["c"].append([float(xs.mean()), float(ys.mean())])
            out["area"].append(len(xs)); out["ok"].append(bool(good))
            b = np.packbits(M[j][k][y0:y1, x0:x1]); buf.append(b); off.append(off[-1] + len(b))
    last = M.get(c1 - c0)
    a_last = max(an) if an else None
    prev = {k: last[k] for k in range(N) if last is not None and a_last is not None and okA.get((a_last, k), False) and last[k].sum() >= MIN_AREA} if last is not None else {}
    for f in wd.iterdir():
        f.unlink()
    wd.rmdir()
    print("final masks", c0, c1, "seeds", stats["seeds"], "repairs", stats["repairs"], "checked", stats["checked"], "bad", stats["bad"], flush=True)
    if c1 >= n - 1:
        break
    c0 = c1
np.savez_compressed(sys.argv[2], frame=np.array(out["frame"], np.int64), id=np.array(out["id"], np.int16), box=np.array(out["box"], np.int32).reshape(-1, 4),
                    c=np.array(out["c"], float).reshape(-1, 2), area=np.array(out["area"], np.int64), ok=np.array(out["ok"], bool),
                    mbuf=np.concatenate(buf) if buf else np.zeros(0, np.uint8), moff=np.array(off, np.int64), **{k: np.int64(v) for k, v in stats.items()})
print("final masks done", len(out["frame"]), "rows", flush=True)

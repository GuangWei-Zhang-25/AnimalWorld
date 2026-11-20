#!/usr/bin/env python3
"""Score a result of Animal World Run on a demo clip against the clip's reference positions. numpy only.

    python score_demo.py RESULT_DIR REFERENCE_DIR [--out accuracy.json]

RESULT_DIR holds tracks.csv (video_frame, id, x, y, ..., source). REFERENCE_DIR holds reference.csv (video_frame, id, x, y)
and reference.json (body_length_px, every_nth_frame, validated, note). The reference exists for every `every_nth_frame`-th
camera frame only. Rules (the same as in the lab's tracker benchmark):
  * identity names are linked ONCE, at the first reference frame where all animals are more than one body length apart;
  * a mouse-frame is correct when the track linked to that animal is within half a body length of it;
  * an identity fix is one run of frames on which a track is on another animal (swap), on nothing (wrong object) or absent
    for 0.3 s or longer (lost); an animal with no track at all is one fix (never found); runs shorter than 0.2 s are merged;
  * detection, per reference frame: output positions are matched one-to-one to reference animals within half a body length
    (ignoring identity): found, missed, false.
Rows between reference frames (source = filled / interpolated) have no reference; they are scored against the straight line
between the two neighbouring reference positions, which is an approximation and is reported separately."""
import csv, itertools, json, sys
from pathlib import Path
import numpy as np


def load(res, ref):
    meta = json.load(open(Path(ref) / 'reference.json')); R = list(csv.DictReader(open(Path(ref) / 'reference.csv')))
    rf = sorted({int(r['video_frame']) for r in R}); ids = sorted({int(r['id']) for r in R}); G = np.full((len(rf), len(ids), 2), np.nan)
    for r in R: G[rf.index(int(r['video_frame'])), ids.index(int(r['id']))] = (float(r['x']), float(r['y']))
    T = list(csv.DictReader(open(Path(res) / 'tracks.csv'))); tid = sorted({int(r['id']) for r in T}); nf = max([int(r['video_frame']) for r in T] + rf) + 1
    X = np.full((nf, len(tid), 2), np.nan); S = np.full((nf, len(tid)), '', dtype=object)
    for r in T: X[int(r['video_frame']), tid.index(int(r['id']))] = (float(r['x']), float(r['y'])); S[int(r['video_frame']), tid.index(int(r['id']))] = r.get('source') or 'tracked'
    return meta, np.array(rf), G, X, S


def link(G, Xr, BL):
    """track index for each reference animal (or -1), from the first reference frame with all animals > 1 body length apart"""
    NG = G.shape[1]; f0 = 0
    for f in range(len(G)):
        if not np.isfinite(G[f]).all(): continue
        d = np.hypot(*(G[f][:, None] - G[f][None]).transpose(2, 0, 1)); d[np.eye(NG, dtype=bool)] = 1e9
        if NG == 1 or d.min() > BL: f0 = f; break
    NT = Xr.shape[1]; first = {}
    for k in range(NT):
        v = np.flatnonzero(np.isfinite(Xr[f0:, k, 0]))
        if len(v): first[k] = f0 + v[0]
    best, bc = {}, 1e18
    for perm in itertools.permutations(list(first) + [-1] * NG, NG):
        if len({p for p in perm if p >= 0}) < sum(p >= 0 for p in perm): continue
        c = 0.0
        for g, k in enumerate(perm):
            if k < 0: c += 1e6; continue
            f = first[k]; d = np.hypot(*(G[f, g] - Xr[f, k])) if np.isfinite(G[f, g, 0]) else 1e9
            c += d + 1e3 * f if d < BL else 1e6
        if c < bc: bc, best = c, perm
    return [k if k >= 0 and True else -1 for k in best], f0


def score(res, ref):
    meta, rf, G, X, S = load(res, ref); BL = float(meta['body_length_px']); n_th = int(meta['every_nth_frame']); fps = float(meta['camera_fps']) / n_th
    NG = G.shape[1]; Xr = X[rf]; perm, f0 = link(G, Xr, BL); MINS, MINL = max(2, int(0.2 * fps)), max(2, int(0.3 * fps))
    ev = dict(swap=0, wrong_object=0, lost=0, never_found=0); correct = total = 0; errors = []
    for g in range(NG):
        gk = np.isfinite(G[:, g, 0]); total += int(gk.sum()); k = perm[g]
        if k < 0: ev['never_found'] += 1; continue
        a = Xr[:, k]; Lb = np.full(len(rf), -2)
        for f in range(len(rf)):
            if not gk[f]: Lb[f] = g; continue
            if not np.isfinite(a[f]).all(): continue
            d = np.hypot(*(G[f] - a[f]).T); d = np.where(np.isfinite(d), d, 1e9); j = int(d.argmin())
            Lb[f] = (g if d[g] < 0.5 * BL else j) if min(d[j], d[g]) < 0.5 * BL else -1
        correct += int(((Lb == g) & gk).sum()); runs = []; f = 0
        while f < len(rf):
            e = f
            while e < len(rf) and Lb[e] == Lb[f]: e += 1
            runs.append([Lb[f], f, e]); f = e
        for i in range(1, len(runs)):
            if runs[i][2] - runs[i][1] < MINS and runs[i][0] != -2: runs[i][0] = runs[i - 1][0]
        M = [runs[0]]
        for r in runs[1:]:
            if r[0] == M[-1][0]: M[-1][2] = r[2]
            else: M.append(r)
        for v, f, e in M:
            if v == g: continue
            kind = 'swap' if v >= 0 else 'wrong_object' if v == -1 else ('lost' if (e - f >= MINL or e == len(rf)) else None)
            if kind: ev[kind] += 1; errors.append(dict(animal=g + 1, kind=kind, from_s=round(rf[f] / meta['camera_fps'], 2), to_s=round(rf[min(e, len(rf) - 1)] / meta['camera_fps'], 2)))
    found = missed = false = 0
    for f in range(len(rf)):
        ref = [p for p in G[f] if np.isfinite(p[0])]; out = [p for p in Xr[f] if np.isfinite(p[0])]; used = set()
        for p in ref:
            d = [np.hypot(*(p - q)) if j not in used else 1e9 for j, q in enumerate(out)]
            if d and min(d) < 0.5 * BL: used.add(int(np.argmin(d))); found += 1
            else: missed += 1
        false += len(out) - len(used)
    # rows between reference frames, against the straight line between the neighbouring reference positions
    fill = dict(filled=0, filled_correct=0, interpolated=0, interpolated_correct=0, no_row=0)
    for i in range(len(rf) - 1):
        for cf in range(rf[i] + 1, rf[i + 1]):
            w = (cf - rf[i]) / (rf[i + 1] - rf[i]); Gi = (1 - w) * G[i] + w * G[i + 1]
            for g in range(NG):
                if not np.isfinite(Gi[g, 0]) or perm[g] < 0: continue
                p = X[cf, perm[g]] if cf < len(X) else np.array([np.nan, np.nan])
                if not np.isfinite(p[0]): fill['no_row'] += 1; continue
                src = 'interpolated' if S[cf, perm[g]] == 'interpolated' else 'filled'
                fill[src] += 1; fill[src + '_correct'] += bool(np.hypot(*(Gi[g] - p)) < 0.5 * BL)
    fixes = sum(ev.values())
    return dict(reference=dict(frames=len(rf), every_nth_camera_frame=n_th, per_second=round(fps, 2), mouse_frames=total, animals=NG, validated=meta.get('validated'), note=meta.get('note')),
                identity_linked_at_s=round(rf[f0] / meta['camera_fps'], 2), tracked_rows=dict(correct_identity_mouse_frames=correct, of=total, pct=round(100.0 * correct / max(1, total), 2)),
                identity_fixes=fixes, fix_kinds=ev, fix_list=errors,
                detection=dict(found=found, missed=missed, false=false, of_reference_mouse_frames=total),
                rows_between_reference_frames=dict(scored_against='straight line between neighbouring reference positions (approximation)', **{k: int(v) for k, v in fill.items()}),
                keypoints='no keypoint reference exists for this clip; not scored',
                perfect=bool(fixes == 0 and correct == total and missed == 0 and false == 0))


if __name__ == '__main__':
    a = score(sys.argv[1], sys.argv[2]); out = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else None
    print(json.dumps(a, indent=1))
    if out: json.dump(a, open(out, 'w'), indent=1)

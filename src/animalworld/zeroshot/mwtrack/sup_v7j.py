# Vendored from the lab's MouseWorld-Track research code (fig8/track_wu/agent_ext7/sup_v7j_ext.py, sha256 41ba71659174439e).
# Only paths were changed (work folder, clip list and checkpoints now come from MWT_* environment variables); the logic is untouched.
"""Agent v5 = v4 contact-graph supervisor + LOCAL TWO-ENDED RE-TRACK witnesses (Lf, Lb: SAM2.1-L re-seeded with detector masks at the
last safe frame before and the first safe frame after EACH contact, native frame rate; track_local.py). When Lf and Lb agree on the
whole permutation of a contact, that permutation is taken (short, independent, two-ended evidence beats long-range runs that drift).
v4 text follows.  (CPU, no reference input).
Principle: identity can only be confused while mice touch. So:
  SAFE frame     the detector sees exactly N mice, all >= 0.8 body length apart -> identity carried by simple proximity, cannot swap
  CONTACT span   everything between two safe frames u (before) and v (after)
For each contact span the only unknown is the permutation linking the N mice at u to the N mice at v. Independent WITNESSES vote
link by link (mouse i at u -> mouse j at v):
  F, B (, M)   SAM2.1 runs seeded by detector masks at different anchors / directions: a track on mouse i at u and on mouse j at v
  S            SAM3.1 text-prompt tracks (different model)
  D            the detector itself: frame-to-frame linking through the span, valid only if N detections every frame and every step unambiguous
A witness whose track is not alone on a mouse at u or v abstains (duplicate / lost tracks never vote).
Decision per link: unanimous (>= 2 witnesses) -> confirmed; majority -> resolved, dissenters logged; tie / single / none -> UNCERTAIN.
Output between u and v: positions from a witness that voted for the winning link.
Evaluation against the validated reference happens only in main()."""
import sys, json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
import os as _os0; _HERE = Path(__file__).resolve().parent; H = Path(_os0.environ.get('MWT_AGENT') or _HERE); L = Path(_os0.environ.get('MWT_ROOT') or H.parent); sys.path.insert(0, str(_HERE)); import core
NAME = 'agent_v7j'


USE_LOCAL = True


def join_fragments(D, N):
    """Two detections in the same frame that are each well under a mouse's size (< 0.75 x the median mouse area) and lie closer than
    0.7 body length are ONE animal seen in two pieces (a clear plastic wall edge or a lid cuts the body), but only on frames where the
    detector (score >= 0.25) sees MORE bodies than there are animals (otherwise two small, close, partly hidden mice would be fused). They are replaced by one
    detection at their area-weighted centre (score = the higher one, area = the sum). Masks are not used afterwards by the supervisor."""
    keep = np.ones(len(D['t']), bool); med = float(np.median(D['area'][D['score'] >= SEED_SCORE_])) if (D['score'] >= SEED_SCORE_).any() else 1.0
    bl = 2 * np.sqrt(med); c = D['c'].copy(); sc = D['score'].copy(); ar = D['area'].astype(float).copy()
    for t in np.unique(D['t']):
        allf = [i for i in np.flatnonzero(D['t'] == t) if sc[i] >= 0.25]
        if len(allf) <= N: continue
        idx = [i for i in allf if ar[i] < 0.75 * med]
        used = set(); extra = len(allf) - N
        for a_ in idx:
            for b_ in idx:
                if a_ >= b_ or a_ in used or b_ in used: continue
                if extra > 0 and np.hypot(*(c[a_] - c[b_])) < 0.7 * bl:
                    w = ar[a_] + ar[b_]; c[a_] = (c[a_] * ar[a_] + c[b_] * ar[b_]) / w; ar[a_] = w; sc[a_] = max(sc[a_], sc[b_]); keep[b_] = False; used |= {a_, b_}; extra -= 1
    out = dict(D); out['c'] = c; out['score'] = np.where(keep, sc, -1.0); out['area'] = ar
    return out


SEED_SCORE_ = 0.4
FAM = {'F': 'sam2 forward', 'Lf': 'sam2 forward', 'F2': 'sam2 forward', 'B': 'sam2 backward', 'Lb': 'sam2 backward', 'B2': 'sam2 backward'}
import os as _os
SEED2 = _os.environ.get('SEED2_WORK')


def run(clip, t0, t1, extra_runs=()):
    import ext_tracks as TE
    o = H / 'work' / f'{clip}_{t0:06d}_{t1:06d}'; D = core.load_det(o / 'det.npz'); Z = np.load(o / 'tracks.npz')
    N, BL = int(Z['N']), float(Z['BL']); T = len(Z['CF']); R = 0.35 * BL
    D = join_fragments(D, N)                                          # v7f: one body split in two (e.g. by a clear wall edge) counts once
    DL = np.load(o / 'dlocal.npz') if (o / 'dlocal.npz').exists() else None; DSTEP = int(DL['step']) if DL is not None else 1
    LOC = np.load(o / 'local.npz') if (o / 'local.npz').exists() and USE_LOCAL else None; STEP = int(LOC['step']) if LOC is not None else 1
    runs = {'F': {k: Z['CF'][:, k] for k in range(N)}, 'B': {k: Z['CB'][:, k] for k in range(N)}}
    for nm in extra_runs:
        if nm in Z.files: runs[nm[1:]] = {k: Z[nm][:, k] for k in range(N)}
    if SEED2:
        p2 = Path(SEED2) / f'{clip}_{t0:06d}_{t1:06d}' / 'tracks.npz'
        if p2.exists():
            Z2 = np.load(p2)
            if int(Z2['N']) == N: runs['F2'] = {k: Z2['CF'][:, k] for k in range(N)}; runs['B2'] = {k: Z2['CB'][:, k] for k in range(N)}
    s3 = TE.load('sam31', clip, f'{t0:06d}_{t1:06d}', t0, T, N)
    if s3: runs['S'] = s3
    conf_d = [np.flatnonzero((D['t'] == t) & (D['score'] >= core.SEED_SCORE)) for t in range(T)]
    all_d = [np.flatnonzero((D['t'] == t) & (D['score'] >= 0.25)) for t in range(T)]

    safe = core.safe_frames(D, N, BL)                                # v6: v5 safe frames, or relaxed ones in a huddled group (core.py)
    if not safe: return dict(clip=clip, window=f'{t0}-{t1}', status='NO SAFE FRAME', ledger=[], N_found=N)
    P = {t: D['c'][conf_d[t]] for t in safe}                         # mouse positions at safe frames (detections)

    def on(r, t, p):
        """track of run r that is alone within R of position p at frame t (None if none or shared)"""
        best = None
        for k, C in runs[r].items():
            if np.isfinite(C[t]).all() and np.hypot(*(C[t] - p)) < R:
                if best is not None: return None
                best = k
        return best

    def d_chain(u, v):
        """detector linking u -> v through every frame; None if any frame lacks N detections or a step is ambiguous"""
        cur = P[u].copy(); path = [cur.copy()]
        for t in range(u + 1, v + 1):
            idx = all_d[t] if t != v else conf_d[t]
            if len(idx) < N: return None, None
            c = D['c'][idx]; d = np.hypot(*(cur[:, None] - c[None]).transpose(2, 0, 1)); rr, cc = linear_sum_assignment(d)
            if d[rr, cc].max() > 0.5 * BL: return None, None
            for i in range(N):                                        # margin: the next-best partner must be clearly worse
                alt = np.delete(d[i], cc[i])
                if len(alt) and alt.min() - d[i, cc[i]] < 0.25 * BL: return None, None
            cur = c[cc]; path.append(cur.copy())
        d = np.hypot(*(cur[:, None] - P[v][None]).transpose(2, 0, 1)); rr, cc = linear_sum_assignment(d)
        return ({int(i): int(j) for i, j in zip(rr, cc)} if d[rr, cc].max() < R else None), np.array(path)
    clean = {}
    def clean_at(r, t):
        if (r, t) not in clean:
            idx = all_d[t]; hit = {}
            for k, C in runs[r].items():
                if not np.isfinite(C[t]).all() or not len(idx): continue
                d = np.hypot(*(D['c'][idx] - C[t]).T); j = int(d.argmin())
                if d[j] < R: hit.setdefault(j, []).append(k)
            clean[r, t] = {ks[0]: runs[r][ks[0]][t] for ks in hit.values() if len(ks) == 1}
        return clean[r, t]

    def step_quorum(u, v):
        """ESCALATION inside one contact span: frame-by-frame votes of clean tracks + elimination (one mouse and one detection
        unaccounted for). Returns path (frames u..v, N, 2) following the mice of P[u], nan where nobody can vouch."""
        d_ = 1 if v > u else -1; fr = list(range(u, v + d_, d_)); path = np.full((len(fr), N, 2), np.nan); path[0] = P[u]
        last = {i: (u, P[u][i].copy()) for i in range(N)}; att = {r: {} for r in runs}

        def attach(t, ids, pos):
            for r in runs:
                c = clean_at(r, t); ks = list(c)
                if not ks: continue
                dd = np.array([[np.hypot(*(c[k] - pos[i])) for i in ids] for k in ks]); rr, cc = linear_sum_assignment(dd)
                for i in ids: att[r].pop(i, None)
                for a_, c_ in zip(rr, cc):
                    if dd[a_, c_] < R: att[r][ids[c_]] = ks[a_]
        attach(u, list(range(N)), {i: P[u][i] for i in range(N)})
        for n_, t in enumerate(fr[1:], 1):
            taken = {}
            cand = {}
            for i in range(N):
                tl, pl = last[i]; reach = BL * min(abs(t - tl), 3); cl = []
                for r in runs:
                    k = att[r].get(i); c = clean_at(r, t)
                    if k is None or k not in c or np.hypot(*(c[k] - pl)) > reach: continue
                    for g in cl:
                        if np.hypot(*(c[k] - g['p'])) < R: g['n'] += 1; g['pts'].append(c[k]); break
                    else: cl.append(dict(p=c[k], n=1, pts=[c[k]]))
                if cl: cl.sort(key=lambda g: (-g['n'], np.hypot(*(g['p'] - pl)))); cand[i] = cl[0]
            for i in sorted(cand, key=lambda i: -cand[i]['n']):
                p = np.mean(cand[i]['pts'], 0)
                if any(np.hypot(*(p - q)) < R for q in taken.values()): continue
                taken[i] = p
            U = [i for i in range(N) if i not in taken]
            if len(U) == 1:
                O = [j for j in conf_d[t] if not any(np.hypot(*(D['c'][j] - q)) < R for q in taken.values())]
                if len(O) == 1 and np.hypot(*(D['c'][O[0]] - last[U[0]][1])) < BL * min(abs(t - last[U[0]][0]), 3): taken[U[0]] = D['c'][O[0]]
            for i, p in taken.items(): path[n_, i] = p; last[i] = (t, p.copy())
            if taken: attach(t, list(taken), taken)
        return path

    # identity labels: lab[t][det index] = fused id ; start at the first safe frame
    SP = []; FLAGS = []
    lab = {safe[0]: list(range(N))}; ledger = []; X = np.full((T, N, 2), np.nan); conf = np.zeros((T, N))
    X[safe[0]] = P[safe[0]]; conf[safe[0]] = 1
    for u, v in zip(safe[:-1], safe[1:]):
        V = np.zeros((N, N)); who = {}
        for r in runs:
            for i in range(N):
                k = on(r, u, P[u][i])
                if k is None: continue
                d = np.hypot(*(P[v] - runs[r][k][v]).T) if np.isfinite(runs[r][k][v]).all() else np.array([9e9])
                j = int(d.argmin())
                if d[j] < R and on(r, v, P[v][j]) == k: V[i, j] += 1; who.setdefault((i, j), []).append(r)
        dm, dpath = d_chain(u, v)
        loc = {}; lf = lb = None
        if LOC is not None and f'f_{u}_{v}' in LOC.files:
            lf, lb = LOC[f'f_{u}_{v}'].astype(float), LOC[f'b_{u}_{v}'].astype(float)
            for nm, C, ref in (('Lf', lf[-1], P[v]), ('Lb', lb[0], P[u])):
                if not np.isfinite(C).all(): continue
                d = np.hypot(*(C[:, None] - ref[None]).transpose(2, 0, 1)); r3, c3 = linear_sum_assignment(d)
                if d[r3, c3].max() < R:
                    loc[nm] = {int(a): int(b) for a, b in zip(r3, c3)} if nm == 'Lf' else {int(b): int(a) for a, b in zip(r3, c3)}
                    for i, j in loc[nm].items(): V[i, j] += 1; who.setdefault((i, j), []).append(nm)
        local_ok = len(loc) == 2 and loc['Lf'] == loc['Lb']
        if DL is not None and f'd_{u}_{v}' in DL.files:               # D30: detector linked frame to frame at the native rate through the contact
            dd_ = DL[f'd_{u}_{v}']; cur = P[u].astype(float).copy(); ok = True; miss = 0; nn = int(dd_[:, 0].max()) if len(dd_) else 0
            for n in range(1, nn + 1):
                c = dd_[(dd_[:, 0] == n) & (dd_[:, 3] >= 0.3)][:, 1:3]
                if len(c) < N:
                    miss += 1
                    if miss > DSTEP: ok = False; break               # more than 0.1 s without N detections: the chain is broken
                    continue
                d = np.hypot(*(cur[:, None] - c[None]).transpose(2, 0, 1)); r3, c3 = linear_sum_assignment(d); lim = 0.2 * BL * (1 + miss)
                if d[r3, c3].max() > lim or any(len(c) > 1 and np.delete(d[i], c3[i]).min() - d[i, c3[i]] < 0.12 * BL for i in range(N)): ok = False; break
                cur = c[c3]; miss = 0
            if ok:
                d = np.hypot(*(cur[:, None] - P[v][None]).transpose(2, 0, 1)); r3, c3 = linear_sum_assignment(d)
                if d[r3, c3].max() < R:
                    for i, j in zip(r3, c3): who.setdefault((int(i), int(j)), []).append('D30')
        if dm:
            for i, j in dm.items(): V[i, j] += 1; who.setdefault((i, j), []).append('D')
        qpath = None
        rr, cc = linear_sum_assignment(-V)
        weak = not local_ok and any(not (V[i, j] >= 2 and max([V[i, x] for x in range(N) if x != j] + [V[x, j] for x in range(N) if x != i] + [0]) == 0) for i, j in zip(rr, cc))
        if weak and v - u > 1:                                        # escalate: step-wise quorum forward and backward through the span
            for nm, (a_, b_) in (('Q>', (u, v)), ('Q<', (v, u))):
                pth = step_quorum(a_, b_)
                if np.isfinite(pth[-1]).all():
                    dd = np.hypot(*(pth[-1][:, None] - P[b_][None]).transpose(2, 0, 1)); r2, c2 = linear_sum_assignment(dd)
                    if dd[r2, c2].max() < R:
                        for i, j in zip(r2, c2):
                            a, b = (int(i), int(j)) if nm == 'Q>' else (int(j), int(i))
                            V[a, b] += 0.1; who.setdefault((a, b), []).append(nm)      # tie-break only: built from the same runs, not independent
                        if nm == 'Q>': qpath = pth
        # FAMILY VOTING: witnesses that share model and direction are not independent (F and Lf both run SAM2.1 forward, B and Lb backward)
        # and tend to fail together. Each family has ONE vote per mouse, split among its members that voted; the step quorum only breaks ties.
        V = np.zeros((N, N)); nfam = {}
        for (i, j), rs in who.items():
            for r in rs: nfam.setdefault((i, FAM.get(r, r)), []).append(r)
        for (i, j), rs in who.items():
            for r in rs: V[i, j] += 0.1 if r in ('Q>', 'Q<') else 1.0 / len(nfam[i, FAM.get(r, r)])
        rr, cc = linear_sum_assignment(-V); pi = {int(i): int(j) for i, j in zip(rr, cc)}
        if v - u == 1 and V.sum() == 0:                               # adjacent safe frames, nobody testified: proximity is exact
            d = np.hypot(*(P[u][:, None] - P[v][None]).transpose(2, 0, 1)); rr, cc = linear_sum_assignment(d); pi = {int(i): int(j) for i, j in zip(rr, cc)}
        famv = {}                                                    # CERTAIN only if >= 2 independent families voted on every mouse, none disagreeing
        for (i, j), rs in who.items():
            for r in rs:
                if r not in ('Q>', 'Q<'): famv.setdefault(i, {}).setdefault(j, set()).add(FAM.get(r, r))
        certain = v - u == 1 or all(i in famv and len(famv[i]) == 1 and len(next(iter(famv[i].values()))) >= 2 for i in range(N))
        if not certain: FLAGS.append(dict(start=int(u), end=int(v), seconds=round((v - u) / 10, 1), votes={f'{i}>{j}': rs for (i, j), rs in who.items()}))
        SP.append(dict(u=int(u), v=int(v), pi=pi, who={f'{i}>{j}': rs for (i, j), rs in who.items()}, Pu=P[u].tolist(), Pv=P[v].tolist()))
        lab[v] = [None] * N
        for i in range(N):
            j = pi[i]; fid = lab[u][i]; lab[v][j] = fid; sup = V[i, j]; ag = max([V[i, x] for x in range(N) if x != j] + [V[x, j] for x in range(N) if x != i] + [0])
            c = 1.0 if (sup >= 2 and ag == 0) or (v - u == 1 and ag == 0) else 0.8 if sup > ag and sup >= 2 else 0.6 if sup > ag else 0.3
            if c < 1.0:
                dis = sorted({r for (a, b), rs in who.items() if (a == i) != (b == j) for r in rs})
                ledger.append(dict(event='contact', start=int(u), end=int(v), seconds=round((v - u) / 10, 1), mouse=int(fid), for_=who.get((i, j), []), against=dis,
                                   status='resolved by majority' if c == 0.8 else 'single witness' if c == 0.6 else 'UNCERTAIN'))
            X[v, fid] = P[v][j]; conf[v, fid] = c
            if v - u > 1:                                             # positions inside the span: median over every witness that supports the link
                srcs = []
                for r in who.get((i, j), []):
                    if r in ('D', 'D30', 'Q>', 'Q<', 'Lf', 'Lb'): continue
                    k = on(r, u, P[u][i])
                    if k is not None: srcs.append(runs[r][k])
                for n_, t in enumerate(range(u + 1, v)):
                    pts = [s_[t] for s_ in srcs if np.isfinite(s_[t]).all()]
                    if lf is not None:                               # local runs that carry this link: forward object i, backward object j
                        if loc.get('Lf', {}).get(i) == j and np.isfinite(lf[STEP * (t - u), i]).all(): pts.append(lf[STEP * (t - u), i])
                        if loc.get('Lb', {}).get(i) == j and np.isfinite(lb[STEP * (t - u), j]).all(): pts.append(lb[STEP * (t - u), j])
                    if qpath is not None and 'Q>' in who.get((i, j), []) and np.isfinite(qpath[n_ + 1][i]).all(): pts.append(qpath[n_ + 1][i])
                    if pts: X[t, fid] = np.median(np.array(pts), 0); conf[t, fid] = c
        if v - u > 1 and N > 1:                                       # JOINT positions inside the contact: one witness for ALL mice per frame,
            cands = {}                                                # the one whose tracks stay most apart (per-mouse medians let two
            for r in runs:                                            # identities collapse onto the same blob)
                ks = [on(r, u, P[u][i]) for i in range(N)]
                if all(k is not None for k in ks) and any(r in who.get((i, pi[i]), []) for i in range(N)) and not any(r in rs for (i_, j_), rs in who.items() if pi.get(i_) != j_): cands[r] = [runs[r][k] for k in ks]
            if lf is not None and loc.get('Lf') == pi: cands['Lf'] = [lf[::STEP][:, i] for i in range(N)]
            if lb is not None and loc.get('Lb') == pi: cands['Lb'] = [lb[::STEP][:, pi[i]] for i in range(N)]
            off = {r: (u if r in ('Lf', 'Lb') else 0) for r in cands}
            # SMOOTHEST CONSISTENT PATH through the contact: per frame every witness offers one joint placement of all mice; dynamic
            # programming picks the sequence with the least total movement from the mice at u to the mice at v (an identity exchange
            # inside the contact costs two jumps, so it is avoided whenever any witness offers a continuous alternative); placements
            # that more independent witnesses agree on are preferred.
            layers = []
            for t in range(u + 1, v):
                opts = []
                for r, tr_ in cands.items():
                    p = np.array([tr_[i][t - off[r]] for i in range(N)], float)
                    if np.isfinite(p).all(): opts.append((r, p))
                cur = np.array([X[t, lab[u][i]] for i in range(N)])
                if np.isfinite(cur).all(): opts.append(('median', cur))
                lay = []
                for r, p in opts:
                    fams = {FAM.get(q, q) for q, p2 in opts if q != 'median' and np.hypot(*(p - p2).T).max() < R}
                    dc_ = D['c'][all_d[t]] if len(all_d[t]) else np.zeros((0, 2))       # DETECTOR SUPPORT: a placement whose mice sit
                    sup_ = sum(min(float(np.hypot(*(dc_ - q).T).min()) if len(dc_) else BL, BL) for q in p)   # on detections is preferred
                    lay.append((p, -0.05 * BL * len(fams) + 0.5 * sup_))
                layers.append(lay)
            end = np.array([P[v][pi[i]] for i in range(N)]); prev = [(P[u].astype(float), 0.0, None)]; back = []
            for lay in layers:
                if not lay: back.append(None); continue
                nxt = []
                for p, nc in lay:
                    costs = [c + np.hypot(*(p - q).T).sum() for q, c, _ in prev]; b = int(np.argmin(costs)); nxt.append((p, costs[b] + nc, b))
                back.append((prev, nxt)); prev = nxt
            if any(b is not None for b in back):
                b = int(np.argmin([c + np.hypot(*(end - q).T).sum() for q, c, _ in prev]))
                for n_ in range(len(layers) - 1, -1, -1):
                    if back[n_] is None: continue
                    pv, nx = back[n_]; p, _, bb = nx[b]
                    for i in range(N): X[u + 1 + n_, lab[u][i]] = p[i]
                    b = bb
    # MERGED AND UNSETTLED: inside a flagged contact, on frames where the detector sees fewer than N animals, mice whose positions are
    # within 0.7 body length of each other are one blob in the image. Which half is which is not supported by evidence there, so each of
    # them is reported at the blob position (their mean) instead of an arbitrary split.
    for fl in FLAGS:
        for t in range(fl['start'] + 1, fl['end']):
            if len(conf_d[t]) >= N: continue
            p = X[t]; ok = np.isfinite(p).all(1); done = set()
            for i in range(N):
                if not ok[i] or i in done: continue
                grp = [k for k in range(N) if ok[k] and np.hypot(*(p[k] - p[i])) < 0.35 * BL]   # v7e: only truly overlapping pairs
                if len(grp) != 2:                                    # v7j: a pair < 0.7 BL apart where the detector (>= 0.25, de-duplicated)
                    near = [k for k in range(N) if k != i and ok[k] and np.hypot(*(p[k] - p[i])) < 0.7 * BL]   # sees ONE body within 0.6 BL
                    dl = D['c'][all_d[t]] if len(all_d[t]) else np.zeros((0, 2))                                  # of the pair's midpoint
                    for k in near:
                        mid = (p[i] + p[k]) / 2; nb = []
                        for q in dl:
                            if np.hypot(*(q - mid)) < 0.6 * BL and all(np.hypot(*(q - r_)) >= 0.3 * BL for r_ in nb): nb.append(q)
                        if len(nb) == 1 and all(len([j for j in range(N) if ok[j] and np.hypot(*(p[j] - q)) < 0.7 * BL]) <= 2 for q in nb): grp = [i, k]; break
                if len(grp) == 2:                                    # v7d: pairs only; 3+ identities are never pulled onto one point
                    m = p[grp].mean(0)
                    # v7c (fix for a four-mouse top-view recording): the blob is only 'merged' if the detector, at its LOW threshold (0.25, duplicates within 0.3 BL
                    # removed), also sees fewer bodies there than identities. A mouse seen dimly (e.g. through a clear plastic box) is
                    # below the 0.4 seed threshold but still a separate body: then each identity is put on its own detection instead.
                    lo = D['c'][all_d[t]] if len(all_d[t]) else np.zeros((0, 2)); lo = lo[np.hypot(*(lo - m).T) < 1.2 * BL] if len(lo) else lo; keep_ = []
                    for q in lo:
                        if all(np.hypot(*(q - r_)) >= 0.3 * BL for r_ in keep_): keep_.append(q)
                    if len(keep_) >= len(grp):
                        dd = np.hypot(*(p[grp][:, None] - np.array(keep_)[None]).transpose(2, 0, 1)); r3, c3 = linear_sum_assignment(dd)
                        for a_, b_ in zip(r3, c3): X[t, grp[a_]] = keep_[b_]; done.add(grp[a_])
                        continue
                    for k in grp: X[t, k] = m; conf[t, k] = min(conf[t, k], 0.5); done.add(k)
    # v7i: inside a flagged contact, a pair the supervisor reported as ONE blob (same point) shortly before AND shortly after (gap <= 3 frames)
    # is reported as one blob in the gap too: a 1-3 frame split in the middle of an unresolved pile carries no identity information.
    for fl in FLAGS:
        for i in range(N):
            for k in range(i + 1, N):
                co = [t for t in range(fl['start'] + 1, fl['end']) if np.isfinite(X[t, i]).all() and np.isfinite(X[t, k]).all() and np.hypot(*(X[t, i] - X[t, k])) < 0.1 * BL]
                for a_, b_ in zip(co[:-1], co[1:]):
                    if 1 < b_ - a_ <= 4:
                        for t in range(a_ + 1, b_):
                            if np.isfinite(X[t, i]).all() and np.isfinite(X[t, k]).all(): m = (X[t, i] + X[t, k]) / 2; X[t, i] = m; X[t, k] = m
    # v7d GLITCH FILTER: a one- or two-frame excursion (position leaves by > 0.5 body length and comes back to within 0.3 body length of
    # where it left) cannot be a mouse moving; the excursion frames are replaced by straight-line interpolation.
    for k in range(N):
        for t in range(1, T - 1):
            for w in (1, 2):
                if t + w >= T: continue
                a_, b_ = X[t - 1, k], X[t + w, k]
                if not (np.isfinite(a_).all() and np.isfinite(b_).all()) or np.hypot(*(a_ - b_)) > 0.3 * BL: continue
                seg = X[t:t + w, k]
                if np.isfinite(seg).all() and (np.hypot(*(seg - a_).T) > 0.5 * BL).all():
                    for q in range(w): X[t + q, k] = a_ + (b_ - a_) * (q + 1) / (w + 1)
    # fill inside spans still empty by linear interpolation when short (<= 0.5 s), flagged
    for fid in range(N):
        t = 0
        while t < T:
            if np.isnan(X[t, fid, 0]):
                e = t
                while e < T and np.isnan(X[e, fid, 0]): e += 1
                if t > 0 and e < T and e - t <= 5:
                    for q in range(t, e): X[q, fid] = X[t - 1, fid] + (X[e, fid] - X[t - 1, fid]) * (q - t + 1) / (e - t + 1); conf[q, fid] = min(conf[t - 1, fid], 0.6)
                elif t > 0 and e < T: ledger.append(dict(event='gap', mouse=fid, start=int(t), end=int(e - 1), seconds=round((e - t) / 10, 1), status='UNCERTAIN (no position)'))
                t = e
            else: t += 1
    # window edges: before the first / after the last safe frame, follow any run attached there
    for edge, rng in ((safe[0], range(safe[0] - 1, -1, -1)), (safe[-1], range(safe[-1] + 1, T))):
        for i in range(N):
            fid = lab[edge][i]
            for r in runs:
                k = on(r, edge, P[edge][i])
                if k is None: continue
                for t in rng:
                    if np.isfinite(runs[r][k][t]).all(): X[t, fid] = runs[r][k][t]; conf[t, fid] = 0.6
                break
    # v7g: window edges, detector check: walking away from the edge safe frame, each identity is matched one-to-one to the detections
    # (score >= 0.25) of the next frame (Hungarian, gate 0.5 body length per frame). An identity the detector supports takes that position;
    # one whose run position coincides (< 0.3 BL) with another identity's while some detection is left unclaimed is moved there.
    for edge, rng in ((safe[0], range(safe[0] - 1, -1, -1)), (safe[-1], range(safe[-1] + 1, T))):
        prev = np.array([X[edge, lab[edge][i]] for i in range(N)], float); fids = [lab[edge][i] for i in range(N)]
        for t in rng:
            dets = D['c'][all_d[t]] if len(all_d[t]) else np.zeros((0, 2)); cur = np.array([X[t, f_] for f_ in fids], float)
            if len(dets):
                dd = np.hypot(*(prev[:, None] - dets[None]).transpose(2, 0, 1)); r3, c3 = linear_sum_assignment(dd); claimed = set()
                for a_, b_ in zip(r3, c3):
                    if dd[a_, b_] < 0.5 * BL: cur[a_] = dets[b_]; claimed.add(b_)
                for a_ in range(N):
                    clash = any(b_ != a_ and np.isfinite(cur[b_]).all() and np.isfinite(cur[a_]).all() and np.hypot(*(cur[a_] - cur[b_])) < 0.3 * BL for b_ in range(N))
                    free = [j for j in range(len(dets)) if j not in claimed]
                    if clash and free:
                        j = min(free, key=lambda j: np.hypot(*(dets[j] - prev[a_])) if np.isfinite(prev[a_]).all() else 0)
                        if not np.isfinite(prev[a_]).all() or np.hypot(*(dets[j] - prev[a_])) < 1.0 * BL: cur[a_] = dets[j]; claimed.add(j)
            for a_, f_ in enumerate(fids):
                if np.isfinite(cur[a_]).all(): X[t, f_] = cur[a_]
            prev = np.where(np.isfinite(cur), cur, prev)
    unc = [l for l in ledger if l['status'].startswith('UNCERTAIN')]
    np.savez_compressed(o / 'fused5_v7j.npz', fused=X, conf=conf)
    od = L / 'runs_clips' / NAME / clip; od.mkdir(parents=True, exist_ok=True); rows = [(t0 + t, k, *X[t, k]) for t in range(T) for k in range(N) if np.isfinite(X[t, k]).all()]
    r_ = np.array(rows, float).reshape(-1, 4)
    np.savez_compressed(od / f'{t0:06d}_{t1:06d}.npz', pt_frame=r_[:, 0].astype(np.int32), pt_track=r_[:, 1].astype(np.int32), pt_x=r_[:, 2], pt_y=r_[:, 3], W=D['W'], H=D['H'], N=T)
    spans = [(u, v) for u, v in zip(safe[:-1], safe[1:]) if v - u > 1]
    return dict(clip=clip, window=f'{t0}-{t1}', N_found=N, witnesses=list(runs) + ['D', 'Q (step quorum, escalation)'], frames=T, safe_frames=len(safe), contact_spans=len(spans),
                longest_contact_s=round(max([v - u for u, v in spans] + [0]) / 10, 1), not_unanimous=len(ledger), uncertain=len(unc), ledger=ledger,
                status='FLAGGED FOR REVIEW' if FLAGS else 'confirmed', flagged_contacts=FLAGS, spans=SP)


def link_apart(TE, tr, G, BL):
    """identity names of a tracker are tied to the reference at the first frame where the reference mice are all > 1 body length apart
    (inside a pile at the window start 'which is which' is not defined for a tracker that got no reference input); same rule for all."""
    NG = G.shape[1]; f0 = 0
    for f in range(len(G)):
        if not np.isfinite(G[f]).all(): continue
        d = np.hypot(*(G[f][:, None] - G[f][None]).transpose(2, 0, 1)); d[np.eye(NG, dtype=bool)] = 1e9
        if NG == 1 or d.min() > BL: f0 = f; break
    keys = list(tr); sub = {k: tr[k][f0:] for k in keys}; m = TE.link(sub, G[f0:], BL, NG)
    return {nk: tr[next(k for k in keys if sub[k] is a)] for nk, a in m.items()}


def main():
    import ext_tracks as TE
    out = []; tot = {}
    sel = getattr(core, sys.argv[1]) if len(sys.argv) > 1 else core.CLIPS
    for clip, t0, t1 in sel:
        if not (H / 'work' / f'{clip}_{t0:06d}_{t1:06d}' / 'tracks.npz').exists(): continue
        r = run(clip, t0, t1, extra_runs=('CM',)); ck = f'{t0:06d}_{t1:06d}'
        z = np.load(TE.gt_file(core.dataset(clip), clip)); G = refcorr.centroids(clip, z)[t0:t1 + 1]; G[G <= 0] = np.nan; BLr = float(z['body_length_px']); FPS = float(z['fps']); r['eval'] = {}
        sc = float(np.load(H / 'work' / f'{clip}_{t0:06d}_{t1:06d}' / 'det.npz')['W']) / float(z['mask_w']); G = G * sc; BLr = BLr * sc   # reference stored at mask resolution
        for m in (NAME, 'agent_v4', 'sam2_large', 'sam31'):
            tr = TE.load(m, clip, ck, t0, len(G), G.shape[1])
            if tr is None: continue
            ev = TE.events(link_apart(TE, tr, G, BLr), G, BLr, FPS)
            r['eval'][m] = {k: ev[k] for k in ('interventions', 'swap', 'wrong_object', 'lost', 'never_found', 'error_mouse_frames')}
            tot[m] = tot.get(m, 0) + ev['interventions']
        nbad = 0                                                     # DIAGNOSTIC ONLY (reference): which witnesses were wrong at which contact
        for sp in r.pop('spans', []):
            if sp['v'] - sp['u'] < 2: continue
            gu, gv = G[sp['u']], G[sp['v']]
            if not (np.isfinite(gu).all() and np.isfinite(gv).all()): continue
            iu = [int(np.hypot(*(gu - p).T).argmin()) for p in sp['Pu']]; iv = [int(np.hypot(*(gv - p).T).argmin()) for p in sp['Pv']]
            if len(set(iu)) < len(iu) or len(set(iv)) < len(iv): continue
            true = {i: iv.index(iu[i]) for i in range(len(iu))}; wrong = sorted({w for k, ws in sp['who'].items() for w in ws if true[int(k.split('>')[0])] != int(k.split('>')[1])})
            right = sorted({w for k, ws in sp['who'].items() for w in ws if true[int(k.split('>')[0])] == int(k.split('>')[1])})
            bad = any(sp['pi'][i] != true[i] for i in true); nbad += bad
            if bad or wrong: r.setdefault('diag', []).append(dict(u=sp['u'], v=sp['v'], decided='WRONG' if bad else 'ok', right=right, wrong=wrong))
        r['wrong_contacts'] = nbad
        out.append(r)
        print(f"\n{clip} {t0}-{t1}: N {r['N_found']}, witnesses {r.get('witnesses')}, safe frames {r.get('safe_frames')}/{r.get('frames')}, contact spans {r.get('contact_spans')} "
              f"(longest {r.get('longest_contact_s')} s), not unanimous {r.get('not_unanimous')}, uncertain {r.get('uncertain')}, status {r['status']}")
        for m, e in r['eval'].items(): print(f'   {m:12s}', e)
        for l in r.get('diag', []): print('   diag', l)
    json.dump(out, open(H / ('results5_' + sys.argv[1] + '.json' if len(sys.argv) > 1 else 'results5.json'), 'w'), indent=1); print('\nTOTAL interventions', tot)


if __name__ == '__main__':
    main()

"""CPU step of MouseWorld-Track: the frozen supervisor (default v7j) on every clip of <MWT_ROOT>/clips.json, run as the lab's
fig8/track_wu/run_supervisors.py runs it (second seed = MWT_SEED2 work folder, extra_runs=('CM',)). No reference is read.
-> <MWT_ROOT>/runs_clips/agent_<ver>/<clip>/<key>.npz and <MWT_ROOT>/supervisor_summary.json.   usage: run_supervisor.py [v7j|v7b]"""
import importlib
import json
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
X = Path(os.environ['MWT_ROOT'])
if os.environ.get('MWT_SEED2'): os.environ['SEED2_WORK'] = os.environ['MWT_SEED2']
ver = (sys.argv[1:] or ['v7j'])[0]
S = importlib.import_module(f'sup_{ver}')
out = {}
for c in json.load(open(X / 'clips.json'))['clips']:
    n = f"{c['dataset']}__{c['clip']}"
    try:
        r = S.run(n, 0, c['frames'] - 1, extra_runs=('CM',))
        out[n] = {k: r.get(k) for k in ('N_found', 'status', 'witnesses', 'safe_frames', 'contact_spans', 'longest_contact_s', 'not_unanimous', 'uncertain')}
        out[n]['flagged_contacts'] = [(int(f['start']), int(f['end'])) for f in r.get('flagged_contacts', [])]
    except Exception as e:
        out[n] = dict(status='NO OUTPUT', error=repr(e)[:300]); traceback.print_exc()
    print(ver, n, out[n], flush=True)
json.dump({ver: out}, open(X / f'supervisor_summary_{ver}.json', 'w'), indent=1, default=str)

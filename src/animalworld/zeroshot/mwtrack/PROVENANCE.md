# Provenance of the vendored MouseWorld-Track scripts

Source: `TrackPose_demonstration/long_runs/PAPER_FIGS/fig8/track_wu/agent_ext7/` (the reference-free copy of the agent, 19 Nov 2026).
`sup_v7j.py` is the frozen v7j supervisor (`supervise5_frozen_v7j_20261112.py` with only the output name changed), the rule set scored as MouseWorld-Track in the manuscript's tracking figure; `sup_v7b.py` is the frozen v7b supervisor.
Patches: work folder / clip list / checkpoint paths read from `MWT_AGENT`, `MWT_ROOT`, `MWT_DET_CKPT`, `MWT_SAM2_*`; nothing else.

| source file | vendored as | sha256 of the source |
|---|---|---|
| core_base.py | core_base.py | fb2649179b35316cac3c42beac2ffd9c2c1a540211240b4c7b609f3f59c4b7f5 |
| core.py | core.py | 7c99039ed6193045fe02064c67af1921a798f4ee631d9064612b31e1cf2935c8 |
| ext_tracks.py | ext_tracks.py | cbcd3ff670dd7e8f74d088a4ab6d506c5c42c147b19bb6216c66bd08e891175c |
| detect_dense.py | detect_dense.py | 31e0b2737937803c297d40d0e1112b29b5dde76837a5615b81a873c2e9cd7820 |
| detect_local.py | detect_local.py | d4cea395c2e25101626e353fd0d532645b27cb5edf3512d6629f6a1590ceea20 |
| track_fb.py | track_fb.py | 17c7d3ea110997c57461d58acf9daf46e8b841e5cd2645049e8b6b6dca0a175a |
| track_local.py | track_local.py | 8f71b6d54c3cc0e912678aa9bfef21f2bc429dfbd7bec60eb35c9c781be721e0 |
| sup_v7j_ext.py | sup_v7j.py | 41ba71659174439e17a9985162a3c17df274198ab8d271c10e05ca5749976004 |
| sup_v7b_ext.py | sup_v7b.py | 450a96d90027c27c20cda8046fae3ed4ba97911c64b1ff5b92cc61ad5b357feb |

New files written for this package: `long_baselines.py` (SAM 2.1 paths from the environment), `sam31_witness.py` (the `sam31` branch of `track_wu/run_ext.py`), `run_supervisor.py` (from `track_wu/run_supervisors.py`), `pose_frames.py` (the crop-pose recipe of `pose_topdown/improve/r3_infer.py`).

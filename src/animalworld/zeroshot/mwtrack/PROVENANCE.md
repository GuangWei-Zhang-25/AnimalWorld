# Provenance of the vendored MouseWorld-Track scripts

Source: `TrackPose_demonstration/long_runs/PAPER_FIGS/fig8/track_wu/agent_ext7/` (the reference-free copy of the agent, 19 Nov 2026).
`sup_v7j.py` is the frozen v7j supervisor (`supervise5_frozen_v7j_20261112.py` with only the output name changed), the rule set scored as MouseWorld-Track in the manuscript's tracking figure; `sup_v7b.py` is the frozen v7b supervisor.
Patches: work folder / clip list / checkpoint paths read from `MWT_AGENT`, `MWT_ROOT`, `MWT_DET_CKPT`, `MWT_SAM2_*`. Two further changes, both outside the rules: `core_base.py` lost the laboratory's clip lists and frame access (the package supplies its own), and since 20 November 2026 `detect_dense.py` writes an empty `det.npz` for a window with no detection. The rule scripts are otherwise as scored; details in `docs/PROVENANCE.md`.

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


Frozen rule file: `supervise5_frozen_v7j_20261112.py`, sha256 `2e956d2cee40179fd54f24dceac11bc52d83c60aa3d4c2f7759795d6b2431bce`. `sup_v7j_ext.py` (hashed above) differs from it only in `NAME` and the output file name.

## Files as shipped in this package

SHA-256 of the vendored files at package version 0.2.1 (they differ from the sources by the patches named above). `tests/test_provenance.py` checks them.

| file | sha256 |
|---|---|
| core_base.py | c6ae10b4ba66dbce88f43fbc5f8393b31be78839cd6b239bedef45ceeec0d0ee |
| core.py | 9bbd3bb017c4e604bacf25a615cbb7104e698d4a04a00a48cf6f0fcd6dd2ebf6 |
| ext_tracks.py | a4d161ed9e9389dfae6a9466509a8a111a28c53861135e09c76167ca060f5599 |
| detect_dense.py | 73329df9f0bb69939c43d341e12c7b57ee5f4497b4c7ddb29302d5952a0a4b68 |
| detect_local.py | 887ffeb0fbd20a5eac49c80d66c9061e8bebaf1a62e517f8341aa6d361a1335b |
| track_fb.py | 51cfbe99fd62938c5330d2266d168b3b970de5fb1023fd5b76c043b6a9df0571 |
| track_local.py | 355a6e6c8d08c99b34905d46cdd6e5cfe3acad9705dd2eba4efddd0397cdd45e |
| sup_v7j.py | cb16f40eed8dc15936978e0d758a2a4b738e696ed37e5f21a0d1b10f5e8d18b4 |
| sup_v7b.py | 5c11cc8183630fae5f6f4a8838aab4a8c08344f798dad88525ee2c3912c908c6 |

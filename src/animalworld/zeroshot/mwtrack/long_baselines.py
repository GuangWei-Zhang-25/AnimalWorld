"""SAM 2.1 location for the vendored agent scripts (replaces the lab's long_baselines.py, which hard-coded cluster paths).

MWT_SAM2_CKPT   path of sam2.1_hiera_large.pt (required)
MWT_SAM2_CFG    hydra config name (default configs/sam2.1/sam2.1_hiera_l.yaml, shipped inside the `sam2` package)
MWT_SAM2_REPO   optional: a source checkout of facebookresearch/sam2 to import from instead of an installed package
"""
import os
import sys
from pathlib import Path

SAM2_REPO = os.environ.get('MWT_SAM2_REPO') or ''
CK = {'sam2_large': (os.environ.get('MWT_SAM2_CFG') or 'configs/sam2.1/sam2.1_hiera_l.yaml', Path(os.environ.get('MWT_SAM2_CKPT') or 'sam2.1_hiera_large.pt'))}


def setup():
    if SAM2_REPO:
        sys.path.insert(0, SAM2_REPO)
        os.chdir(SAM2_REPO)

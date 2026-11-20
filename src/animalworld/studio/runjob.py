"""Run the commands of one Studio job from its JSON file (used by the SLURM wrapper).

    python -m animalworld.studio.runjob JOB.json        JOB.json = {"cmds": [[argv...], ...], "cwd": folder}

Output goes to stdout (sbatch --output points it at the job's log file), in the same format as an in-window job.
"""
from __future__ import annotations

import json
import sys


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    from .core import run_commands
    with open(argv[0], encoding="utf-8") as f:
        job = json.load(f)
    cmds = job.get("cmds")
    if not isinstance(cmds, list) or not all(isinstance(c, list) and c and all(isinstance(a, str) for a in c) for c in cmds):
        print("runjob: the job file does not hold argument lists", file=sys.stderr)
        return 2
    return run_commands(cmds, job.get("cwd") or ".", sys.stdout.buffer)


if __name__ == "__main__":
    raise SystemExit(main())

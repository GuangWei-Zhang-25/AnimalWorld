""""Export for SLEAP / DeepLabCut training": writes the accepted keypoint labels as a SLEAP package
(.slp when sleap-io is installed, COCO keypoint JSON otherwise) and a DeepLabCut labeled-data folder,
and prints the commands to run in your own SLEAP / DeepLabCut environment. Training itself happens
in those tools, not here.
"""
from __future__ import annotations

from pathlib import Path

from .base import Trainer


class SleapDlcExport(Trainer):
    name = "sleap_dlc"
    tested = True

    def plan(self, dataset_dir: Path, out_dir: Path) -> dict:
        return {"backend": self.name, "task": "pose", "writes": [str(Path(out_dir) / "sleap"), str(Path(out_dir) / "dlc")],
                "then_in_sleap": "sleap-label <out>/sleap/labels.slp   (or File > Import > COCO: sleap_coco.json), then train from the GUI or `sleap-train`",
                "then_in_deeplabcut": "copy <out>/dlc/labeled-data/<name> into your DLC project, add config_snippet.yaml entries, "
                                      "deeplabcut.convertcsv2h5(config) if no .h5 was written, deeplabcut.create_training_dataset(config), deeplabcut.train_network(config)"}

    def train(self, dataset_dir: Path, out_dir: Path) -> dict:
        from ...stages import review as R
        s = R.export_labels(self.project, "sleap", Path(out_dir) / "sleap")
        d = R.export_labels(self.project, "dlc", Path(out_dir) / "dlc")
        return {"weights": None, "sleap": s, "dlc": d, "run_dir": str(out_dir)}

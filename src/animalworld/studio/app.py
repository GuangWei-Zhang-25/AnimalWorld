"""Animal World Studio: the whole pipeline in one desktop window (tkinter / ttk, standard library + Pillow; look: animalworld.ui).

    animalworld studio [PROJECT]

A sidebar lists the stages, each with a status dot; a stage is one page: a one-line status, a form of rows (label left,
control right), one primary button and a log that can be opened.
A Run button saves the stage's form into project.yaml and starts the SAME command the command line runs
(`python -m animalworld <stage> <project> ...`, an argument list, never a shell string) as a child process of this
window; its output is written to <project>/logs/ and shown live. The window never blocks: it looks at the job a few
times per second, and Stop ends the job. Everything this window does can also be done headless from the CLI.

There is no server and nothing leaves this computer. An API key typed into the key field is stored with
`animalworld.credentials` (a file outside the project, mode 600) and is never written to project.yaml or a log.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Dict, List, Optional

from .. import __version__, credentials
from . import core
from .core import FormError, Studio

POLL_MS = 200
ROW_ORDER = ("project", "literature", "prompts", "generate", "propose", "review", "dataset", "train", "export", "deploy")
ACTION_ROW = {"literature": "literature", "prompts": "prompts", "generate": "generate", "propose": "propose",
              "propose_keypoints": "propose", "review_auto": "review", "import_labels": "review", "export_labels": "review",
              "dataset": "dataset", "train": "train", "export": "export", "deploy_pretrained": "deploy",
              "deploy_project": "deploy"}
GEN_SOURCES = (("qwen_image", "Qwen-Image (open weights)"), ("http_api", "API"), ("mock", "Mock (try the pipeline)"),
               ("wan_video", "Wan video"))
DEPLOY_MODELS = (("pretrained", "MouseWorld pretrained (zero-shot: MouseWorld-DET + MouseWorld-Track + MouseWorld-Pose)"),
                 ("project", "This project's trained model"))
NO_DISPLAY = ("animalworld studio needs a display and none could be opened ({err}). Start it from a desktop session "
              "or over `ssh -X`; every stage also runs without a window: `animalworld --help`.")


class TextVar:
    """get()/set() for a multi-line Text widget, so a form can treat it like a StringVar."""

    def __init__(self, widget: tk.Text):
        self.widget = widget

    def get(self) -> str:
        return self.widget.get("1.0", "end").strip()

    def set(self, value) -> None:
        self.widget.delete("1.0", "end")
        self.widget.insert("1.0", value if isinstance(value, str) else "\n".join(value))


PAGES = (("project", "Project"), ("literature", "Literature"), ("prompts", "Prompts"), ("generate", "Generate"),
         ("propose", "Propose labels"), ("review", "Review"), ("train", "Train"), ("inference", "Inference"))
PAGE_OF = {"project": "project", "literature": "literature", "prompts": "prompts", "generate": "generate", "propose": "propose",
           "review": "review", "dataset": "train", "train": "train", "export": "train", "deploy": "inference"}
PAGE_CAPTION = {"project": "One folder per species: its settings, images, labels, models and logs.",
                "literature": "What is known about how this animal is recorded: the raw material for the prompts.",
                "prompts": "One prompt per image, varied over the settings found in the literature.",
                "generate": "Images (or clips) from a generator you choose.",
                "propose": "First boxes, outlines and keypoints from models; nothing is accepted here.",
                "review": "A person accepts, corrects or rejects every proposal.",
                "train": "Dataset, training and export of the detector for this species.",
                "inference": "Run a model on your own videos."}


class Row:
    """One stage: a card with a form of rows (label left, control right), its buttons, a message and a log."""

    def __init__(self, app: "StudioApp", parent, key: str, title: str, sub: bool = False, hint: str = ""):
        self.app, self.key = app, key
        th = self.th = app.th
        S = th.S
        self.vars: Dict[str, object] = {}
        self.widgets: Dict[str, tk.Widget] = {}
        self.buttons: Dict[str, ttk.Button] = {}
        self.run_button: Optional[ttk.Button] = None
        self.status_var = tk.StringVar(value="")
        self.msg_var = tk.StringVar(value="")
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill="x", pady=(0, S(24)))
        if sub:
            ttk.Label(self.frame, text=title, style="Section.TLabel").pack(anchor="w")
        st = ttk.Label(self.frame, textvariable=self.status_var, style="Caption.TLabel", justify="left")
        st.pack(fill="x", anchor="w", pady=(S(2), S(10)))
        th.wrap(st)
        self.form = th.card(self.frame, pad=16, fill="x")
        self._group: Optional[ttk.Frame] = None
        self._rows = 0
        if hint:
            self.hint(hint)
        self.actions = ttk.Frame(self.frame)
        self.actions.pack(fill="x", pady=(S(12), 0))
        self.msg_label = ttk.Label(self.frame, textvariable=self.msg_var, style="Error.TLabel", justify="left")
        th.wrap(self.msg_label)
        self.log_frame = ttk.Frame(self.frame, style="Card.TFrame", padding=S(8))
        self.log_text = th.text(self.log_frame, surface="card", height=10, wrap="word", state="disabled")
        sb = ttk.Scrollbar(self.log_frame, orient="vertical", command=self.log_text.yview, style="Card.Vertical.TScrollbar")
        self.log_text.configure(yscrollcommand=sb.set)
        self.log_text.pack(side="left", fill="x", expand=True)
        sb.pack(side="left", fill="y")
        self.log_shown = False
        self.log_button: Optional[ttk.Button] = None

    # ---- building the form ----
    def line(self, parent=None) -> ttk.Frame:
        """Start a new group of rows (in `parent`, e.g. a block that is shown for one choice only)."""
        g = ttk.Frame(parent or self.form, style="In.Card.TFrame")
        g.pack(fill="x")
        g.grid_columnconfigure(0, minsize=self.th.S(250))
        g.grid_columnconfigure(1, weight=1)
        g._n = 0
        g._first = self._rows == 0 and (parent is None)
        self._group = self._line = g
        return g

    def _next(self, label: str):
        """Grid position of the next row in the current group; draws the hairline above it and the label."""
        g = self._group or self.line()
        S = self.th.S
        if g._n or not g._first:
            ln = tk.Frame(g, height=max(1, S(1)), borderwidth=0, highlightthickness=0)
            self.th.reg(ln, background="line")
            ln.grid(row=2 * g._n, column=0, columnspan=3, sticky="ew")
        r = 2 * g._n + 1
        if label:
            ttk.Label(g, text=label, style="Card.TLabel", wraplength=S(236), justify="left").grid(row=r, column=0, sticky="w", pady=S(8), padx=(0, S(14)))
        g._n += 1
        self._rows += 1
        return g, r

    def hint(self, text: str, parent=None) -> None:
        lab = ttk.Label(parent or self.form, text=text, style="Caption.Card.TLabel", justify="left")
        lab.pack(fill="x", anchor="w", pady=(self.th.S(6), self.th.S(6)))
        self.th.wrap(lab)
        self._group = None
        self._rows += 1

    def field(self, key: str, label: str, kind: str = "entry", values=(), width: int = 12, default="", pick: str = None,
              show: str = None):
        """kind: entry | combo | check. pick: "dir" | "file" adds a Choose button. Returns the variable."""
        g, r = self._next(label)
        S = self.th.S
        if kind == "check":
            var = tk.BooleanVar(value=bool(default))
            w = ttk.Checkbutton(g, variable=var, style="Card.TCheckbutton")
            w.grid(row=r, column=1, sticky="w")
        else:
            var = tk.StringVar(value=str(default))
            if kind == "combo":
                w = ttk.Combobox(g, textvariable=var, values=list(values), state="readonly", width=max(width, 12))
                w.grid(row=r, column=1, sticky="w", pady=S(5))
            else:
                w = ttk.Entry(g, textvariable=var, width=max(width, 10), **({"show": show} if show else {}))
                w.grid(row=r, column=1, sticky="ew" if (width >= 18 or pick) else "w", pady=S(5))
            if pick:
                b = ttk.Button(g, text="Choose ...", style="Link.Card.TButton", command=lambda: self.app.browse(var, pick))
                b.grid(row=r, column=2, padx=(S(6), 0))
                self.buttons["browse_" + key] = b
        self.vars[key], self.widgets[key] = var, w
        return var

    def custom(self, label: str) -> ttk.Frame:
        """A row whose control side is a free frame."""
        g, r = self._next(label)
        f = ttk.Frame(g, style="In.Card.TFrame")
        f.grid(row=r, column=1, columnspan=2, sticky="ew", pady=self.th.S(5))
        return f

    def button(self, key: str, text: str, command: Callable, primary: bool = False, quiet: bool = False) -> ttk.Button:
        """primary = the stage's Run button; quiet draws it as a secondary button (a page has ONE accented button)."""
        b = ttk.Button(self.actions, text=text, command=command, style="Primary.TButton" if primary and not quiet else "TButton")
        b.pack(side="right", padx=(self.th.S(8), 0))
        self.buttons[key] = b
        if primary:
            self.run_button = b
        return b

    def log_toggle_button(self) -> None:
        self.log_button = ttk.Button(self.actions, text=" Show log", command=self.toggle_log, style="Link.TButton",
                                     image=self.th.chevron("right", "link"), compound="left")
        self.log_button.pack(side="left")

    # ---- log / messages ----
    def toggle_log(self, show: bool = None) -> None:
        show = (not self.log_shown) if show is None else show
        if show and not self.log_shown:
            self.log_frame.pack(fill="x", pady=(self.th.S(8), 0))
        elif not show and self.log_shown:
            self.log_frame.pack_forget()
        self.log_shown = show
        if self.log_button:
            self.log_button.configure(text=" Hide log" if show else " Show log", image=self.th.chevron("down" if show else "right", "link"))

    def clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def append_log(self, text: str) -> None:
        if not text:
            return
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def log_content(self) -> str:
        return self.log_text.get("1.0", "end")

    def message(self, text: str = "") -> None:
        self.msg_var.set(text)
        if text:
            self.msg_label.pack(fill="x", pady=(self.th.S(8), 0), after=self.actions)
        else:
            self.msg_label.pack_forget()

    def get(self, key: str):
        return self.vars[key].get()


class StudioApp:
    def __init__(self, root: tk.Tk, project=None, geometry: str = "1200x800+0+0"):
        from .. import ui
        self.root, self.ui = root, ui
        self.th = ui.theme(root)
        self.studio = Studio(None)
        self.job = None
        self._offset = 0
        self._job_row: Optional[Row] = None
        self._failed: set = set()
        self.annotator = None
        self.rows: Dict[str, Row] = {}
        self.pages: Dict[str, ttk.Frame] = {}
        self.nav: Dict[str, ttk.Radiobutton] = {}
        self.page_var = tk.StringVar(value="project")
        self.status_var = tk.StringVar(value="Create or open a project.")
        self.confirm: Callable[[str, str], bool] = lambda title, text: messagebox.askyesno(title, text, parent=self.root)
        root.title(f"Animal World Studio {__version__}")
        root.geometry(self.th.geometry(geometry))
        root.minsize(self.th.S(900), self.th.S(500))
        self.bold = self.th.font_bold
        self._build()
        if project:
            self.rows["project"].vars["path"].set(str(project))
            self.open_project()
        self.refresh()
        self._after = root.after(POLL_MS, self._tick)

    # ------------------------------------------------------------------ layout
    def _build(self):
        th, S = self.th, self.th.S
        # sidebar: the stages, each with a status dot
        side = ttk.Frame(self.root, style="Side.TFrame", padding=(S(12), S(20), S(12), S(12)))
        side.pack(side="left", fill="y")
        ttk.Label(side, text="Animal World", style="Section.Side.TLabel").pack(anchor="w", padx=S(8))
        ttk.Label(side, text="Studio", style="Caption.Side.TLabel").pack(anchor="w", padx=S(8), pady=(0, S(16)))
        for key, title in PAGES:
            if key == "literature":
                ttk.Label(side, text="PIPELINE", style="Small.Side.TLabel").pack(anchor="w", padx=S(8), pady=(S(14), S(4)))
            rb = ttk.Radiobutton(side, text=" " + title, value=key, variable=self.page_var, style="Nav.TRadiobutton", image=th.dot("idle"),
                                 compound="left", width=17, command=lambda k=key: self.show(k))
            rb.pack(fill="x", pady=S(1))
            self.nav[key] = rb
        th.mode_button(side, "Link.Side.TButton").pack(side="bottom", anchor="w", padx=S(4))
        self.run_app_button = ttk.Button(side, text="Open Animal World Run", style="Side.TButton", command=self.open_run)
        self.run_app_button.pack(side="bottom", fill="x", pady=(0, S(8)))

        main = ttk.Frame(self.root)
        main.pack(side="left", fill="both", expand=True)
        bottom = ttk.Frame(main, padding=(S(28), S(10), S(28), S(12)))
        bottom.pack(side="bottom", fill="x")
        self.stop_button = ttk.Button(bottom, text="Stop", command=self.stop, state="disabled", width=8)
        self.stop_button.pack(side="right")
        self.status_label = ttk.Label(bottom, textvariable=self.status_var, style="Caption.TLabel", justify="left")
        self.status_label.pack(side="left", fill="x", expand=True, padx=(0, S(12)))
        th.wrap(self.status_label)
        th.hairline(main, side="bottom", fill="x")

        mid = ttk.Frame(main)
        mid.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(mid, highlightthickness=0, borderwidth=0)
        th.reg(self.canvas, background="bg")
        vs = ttk.Scrollbar(mid, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=th.autohide(vs, side="right", fill="y", before=self.canvas))
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = ttk.Frame(self.canvas, padding=(S(28), S(24), S(28), S(8)))
        self._win = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win, width=e.width))
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.root.bind_all(seq, self._wheel, add="+")
        for key, title in PAGES:
            pg = self.pages[key] = ttk.Frame(self.inner)
            ttk.Label(pg, text=title if key != "review" else "Review (human in the loop)", style="Title.TLabel").pack(anchor="w")
            cap = ttk.Label(pg, text=PAGE_CAPTION[key], style="Caption.TLabel", justify="left")
            cap.pack(fill="x", anchor="w", pady=(S(2), S(16) if key == "train" else S(10)))
            th.wrap(cap)
        self._rows()
        self.slurm_card()
        self.show("project")

    def show(self, page: str):
        """Show one stage's page (the sidebar selection)."""
        if page not in self.pages:
            return
        for k, pg in self.pages.items():
            if k == page:
                pg.pack(fill="x")
            else:
                pg.pack_forget()
        self.page_var.set(page)
        self.canvas.yview_moveto(0)

    def slurm_card(self):
        """Where jobs run: this computer (default) or sbatch. Lives on the Project page."""
        th, S = self.th, self.th.S
        pg = self.pages["project"]
        ttk.Label(pg, text="Where the stages run", style="Section.TLabel").pack(anchor="w")
        ttk.Label(pg, text="On this computer, or as SLURM jobs on a cluster.", style="Caption.TLabel").pack(anchor="w", pady=(S(2), S(10)))
        card = th.card(pg, fill="x")
        g = ttk.Frame(card, style="In.Card.TFrame")
        g.pack(fill="x")
        g.grid_columnconfigure(0, minsize=S(250))
        g.grid_columnconfigure(1, weight=1)
        self.slurm_vars = {"on": tk.BooleanVar(value=False)}
        ttk.Label(g, text="Run on SLURM (sbatch)", style="Card.TLabel").grid(row=0, column=0, sticky="w", pady=S(8))
        ttk.Checkbutton(g, variable=self.slurm_vars["on"], style="Card.TCheckbutton").grid(row=0, column=1, sticky="w")
        for i, k in enumerate(("partition", "gres", "cpus", "mem", "time"), start=1):
            ln = tk.Frame(g, height=max(1, S(1)), borderwidth=0, highlightthickness=0)
            th.reg(ln, background="line")
            ln.grid(row=2 * i - 1, column=0, columnspan=2, sticky="ew")
            ttk.Label(g, text=k, style="Card.TLabel").grid(row=2 * i, column=0, sticky="w", pady=S(8))
            self.slurm_vars[k] = tk.StringVar(value=str(core.SLURM_DEFAULTS[k]))
            ttk.Entry(g, textvariable=self.slurm_vars[k], width=18).grid(row=2 * i, column=1, sticky="w", pady=S(5))

    def _wheel(self, e):
        try:
            if e.widget.winfo_toplevel() is not self.root or isinstance(e.widget, (tk.Text, tk.Listbox)):
                return
        except (AttributeError, tk.TclError):
            return
        up = e.num == 4 or getattr(e, "delta", 0) > 0
        if self.inner.winfo_reqheight() > self.canvas.winfo_height():
            self.canvas.yview_scroll(-3 if up else 3, "units")

    def _row(self, key, title, hint="") -> Row:
        page = PAGE_OF[key]
        r = Row(self, self.pages[page], key, title, sub=(page == "train"), hint=hint)
        self.rows[key] = r
        return r

    def _rows(self):
        st = self.studio.state()
        d = st["defaults"]
        S = self.th.S
        # project
        r = self._row("project", "Project")
        r.field("path", "Folder", width=52, default=self.ui.short_path(Path.cwd() / f"{d['species']}_project"), pick="dir")
        r.field("species", "Species", width=18, default=d["species"])
        r.field("description", "Description", width=90, default=d["description"])
        r.button("open", "Create / open", self.open_project, primary=True)
        r.button("save", "Save description", lambda: self.save("project"))
        # literature
        r = self._row("literature", "Literature")
        r.field("backend", "Source", "combo", st["backends"]["literature"], default="offline")
        r.field("limit", "Records", width=10, default=200)
        r.field("force", "Search again", "check")
        r.button("run", "Run", lambda: self.run("literature"), primary=True)
        r.log_toggle_button()
        # prompts
        r = self._row("prompts", "Prompts")
        r.field("n", "Target number of images", width=10, default=40)
        r.field("seed", "Seed", width=10, default=0)
        r.field("megapixels", "Megapixels", width=10, default=1.25)
        r.field("size_multiple", "Size multiple", width=10, default=32)
        r.field("force", "Rewrite all prompts", "check")
        r.hint("Target number = prompts.n in project.yaml: one prompt gives one image (or one clip with a video generator). "
               "Run extracts the taxonomy from the literature, then writes the prompts.")
        r.button("run", "Run", lambda: self.run("prompts"), primary=True)
        r.log_toggle_button()
        # generate
        r = self._row("generate", "Generate")
        f = r.custom("Source")
        src = tk.StringVar(value="mock")
        r.vars["backend"] = src
        r.radios = {}
        for val, text in GEN_SOURCES:
            rb = ttk.Radiobutton(f, text=text, value=val, variable=src, command=self._gen_source, style="Seg.TRadiobutton")
            rb.pack(side="left", padx=(0, S(6)))
            r.radios[val] = rb
        self.gen_groups: Dict[str, ttk.Frame] = {}
        box = ttk.Frame(r.form, style="In.Card.TFrame")
        box.pack(fill="x")
        self._gen_box = box
        g = self.gen_groups["qwen_image"] = ttk.Frame(box, style="In.Card.TFrame")
        q = d["qwen"]
        r.line(g)
        r.field("model_path", "Local weights folder", width=40, default=self.ui.short_path(q["model_path"]), pick="dir")
        r.field("model_id", "or Hugging Face id", width=24, default=q["model_id"])
        r.field("pipeline_class", "Pipeline class", width=24, default=q["pipeline_class"])
        r.field("num_inference_steps", "Steps", width=10, default=q["num_inference_steps"])
        r.field("true_cfg_scale", "CFG", width=10, default=q["true_cfg_scale"])
        r.field("dtype", "dtype", "combo", ("bfloat16", "float16", "float32"), width=12, default=q["dtype"])
        r.field("split", "Encode prompts first, then generate (less GPU memory)", "check", default=q["split"])
        r.hint("Worked example: Qwen-Image 2.1 as in the lab's real run (class QwenImage21Pipeline, 30 steps, true CFG 4.0, bfloat16, "
               "split path, 1.25 MP, sizes in multiples of 32). Weights: a local diffusers folder, by default in the Animal World cache; "
               f"official source {core.QWEN_OFFICIAL} (about 31 GB; Qwen Research License: non-commercial use, and models trained on "
               "its images must say \"Built with Qwen\"). If the folder does not exist, the Hugging Face id is used and the weights "
               "are downloaded from there at first use (that route is untested here). Needs a GPU (24 to 31 GB were used in our runs) and the [generate] extra.", g)
        g = self.gen_groups["http_api"] = ttk.Frame(box, style="In.Card.TFrame")
        r.line(g)
        r.field("endpoint", "Endpoint", width=44, default="https://api.example.com/v1/images/generations")
        r.field("model", "Model name", width=24)
        r.field("style", "Request style", "combo", tuple(core.API_STYLES), width=16, default="openai_images")
        r.field("credential", "Credential name", width=24, default="image_api_key")
        r.field("api_key", "API key", width=34, show="•")
        r.hint("The key is stored outside the project (credentials file, mode 600) under the credential name and this field "
               "is cleared; project.yaml only holds the name. An environment variable ANIMALWORLD_<NAME> works too. "
               "Bring your own account; check the provider's terms for training on generated images.", g)
        g = self.gen_groups["mock"] = ttk.Frame(box, style="In.Card.TFrame")
        r.hint("Flat drawn shapes with known labels. For trying the pipeline only: it says nothing about real image quality.", g)
        g = self.gen_groups["wan_video"] = ttk.Frame(box, style="In.Card.TFrame")
        r.line(g)
        r.field("wan_model_path", "Wan weights folder", width=48, pick="dir")
        r.hint("Text-to-video; frames are extracted for annotation. Needs a whole GPU; never run inside this package's tests.", g)
        r.line()
        r.field("limit", "Only the next N prompts (optional)", width=10)
        r.button("run", "Run", lambda: self.run("generate"), primary=True)
        r.button("save", "Save", lambda: self.save("generate"))
        r.log_toggle_button()
        self._gen_source()
        # propose
        r = self._row("propose", "Propose labels")
        r.field("backend", "Proposal model", "combo", [b for b in st["backends"]["proposers"] if b != "pose"], default="mock")
        r.field("variant", "SAM variant", "combo", ("sam2", "sam3"), width=12, default="sam2")
        r.field("prompt", "Text prompt (empty = species)", width=24)
        r.field("weights", "Detector weights (yolo)", width=28, pick="file")
        r.field("pose_weights", "Keypoint model weights (optional)", width=44, pick="file")
        r.field("force", "Propose again (never touches reviewed images)", "check")
        r.hint("Nothing gated is needed: yolo = a detector file gives boxes and outlines (for mice: mouseworld-det-lite.onnx from the "
               "MouseWorld weights runs on a CPU; later the detector you trained here: the refine loop); sam + sam2 = SAM 2.1 turns existing "
               "boxes into outlines. OPTIONAL: sam + sam3 = SAM 3.1 finds the animals from a text prompt; its weights are gated by Meta's "
               "licence (huggingface.co/facebook/sam3.1). For a new species with no detector yet, the choices are SAM 3.1 or drawing the "
               "first boxes by hand in the annotator. Keypoints come from a pose model, proposed separately. Nothing is accepted automatically here.")
        r.button("run", "Run", lambda: self.run("propose"), primary=True)
        r.button("keypoints", "Propose keypoints", lambda: self.run("propose_keypoints"))
        r.log_toggle_button()
        # review
        r = self._row("review", "Review (human in the loop)")
        r.hint("The annotator shows every image with its proposed boxes, outlines and keypoints; you accept, correct or reject. "
               "Only accepted images go into the training set.")
        r.button("annotate", "Open annotator", self.open_annotator, primary=True)
        r.button("auto", "Accept all that pass the gates (no human look)", self.auto_accept)
        r.log_toggle_button()
        r.line()
        r.field("import_format", "Import labels", "combo", core.LABEL_FORMATS_IN, width=12, default="sleap")
        r.field("import_path", "File or folder", width=30, pick="file")
        r.field("import_images", "Images folder (optional)", width=22, pick="dir")
        f = r.custom("")
        ib = ttk.Button(f, text="Import", command=lambda: self.run("import_labels"), style="Card.TButton")
        ib.pack(side="left")
        r.buttons["import"] = ib
        r.field("export_format", "Export accepted labels", "combo", core.LABEL_FORMATS_OUT, width=12, default="sleap")
        r.field("export_out", "To folder (empty = <project>/exchange)", width=30, pick="dir")
        f = r.custom("")
        eb = ttk.Button(f, text="Export", command=lambda: self.run("export_labels"), style="Card.TButton")
        eb.pack(side="left")
        r.buttons["export"] = eb
        # dataset
        r = self._row("dataset", "Dataset")
        r.field("task", "Task", "combo", core.TASKS, width=12, default="segment")
        r.button("run", "Build dataset", lambda: self.run("dataset"), primary=True, quiet=True)
        r.log_toggle_button()
        # train
        r = self._row("train", "Train")
        r.field("backend", "Trainer", "combo", st["backends"]["trainers"], width=12, default="yolo")
        r.field("task", "Task", "combo", core.TASKS, width=12, default="detect")
        r.field("min_accepted_images", "Train when accepted images reach", width=10, default=200)
        r.field("epochs", "Epochs (optional)", width=10)
        r.field("imgsz", "Image size (optional)", width=10)
        r.field("force", "Train anyway (small model for the refine loop)", "check")
        r.field("dry_run", "Only show the training call", "check")
        r.button("run", "Train", lambda: self.run("train"), primary=True)
        r.log_toggle_button()
        # export
        r = self._row("export", "Export (ONNX)")
        r.hint("Writes the trained model as an ONNX file for `predict` and for other tools.")
        r.button("run", "Export", lambda: self.run("export"), primary=True, quiet=True)
        r.log_toggle_button()
        # inference
        r = self._row("deploy", "Run on your videos")
        f = r.custom("Model")
        mv = tk.StringVar(value="pretrained")
        r.vars["model"] = mv
        r.radios = {}
        for val, text in DEPLOY_MODELS:
            rb = ttk.Radiobutton(f, text=text, value=val, variable=mv, style="Card.TRadiobutton")
            rb.pack(anchor="w")
            r.radios[val] = rb
        f = r.custom("Videos or folders (one per line)")
        well = ttk.Frame(f, style="Well.TFrame", padding=S(3))
        well.pack(side="left", fill="x", expand=True)
        txt = self.th.text(well, surface="log", height=3, width=40, wrap="none")
        txt.configure(font=self.th.font)
        self.th.reg(txt, foreground="text")
        txt.pack(fill="x", expand=True)
        r.vars["videos"], r.widgets["videos"] = TextVar(txt), txt
        bb = ttk.Frame(f, style="In.Card.TFrame")
        bb.pack(side="left", padx=(S(6), 0))
        for k, t, c in (("add_videos", "Add videos ...", lambda: self.browse(r.vars["videos"], "files")),
                        ("add_folder", "Add folder ...", lambda: self.browse(r.vars["videos"], "adddir"))):
            b = ttk.Button(bb, text=t, command=c, style="Link.Card.TButton")
            b.pack(anchor="w")
            r.buttons[k] = b
        r.field("out", "Output folder (empty = <project>/zeroshot)", width=36, pick="dir")
        r.field("n_animals", "Animals (empty = count automatically)", width=10)
        r.field("engine", "Engine", "combo", ("auto", "full", "lite"), width=12, default="auto")
        r.hint("The full MouseWorld-Track strategy currently runs only with the MouseWorld pretrained detectors (mice). "
               "A model trained in this project is run with `predict` + `track` and the simple IoU tracker, one video at a time.")
        r.button("run", "Run", self.run_deploy, primary=True)
        r.button("open_results", "Open results folder", self.open_results)
        r.log_toggle_button()

    def results_folder(self):
        """Where the last 'Run on your videos' wrote: the zero-shot output folder, or the project's tracks / evidence."""
        r = self.rows["deploy"]
        if not self.studio.project:
            return None
        pr = self.studio.project
        if r.get("model") == "pretrained":
            return Path(r.get("out") or pr.path("zeroshot"))
        return pr.path("evidence") if pr.path("evidence").is_dir() else pr.path("tracks")

    def open_results(self) -> bool:
        d = self.results_folder()
        if not d or not Path(d).is_dir():
            self.error("deploy", "no results yet: press Run first")
            return False
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(d))                                                 # noqa: S606
            else:
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(d)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            pass
        self.ok(f"Results are in {self.ui.short_path(d)}")
        return True

    def _gen_source(self):
        cur = self.rows["generate"].vars["backend"].get()
        for k, g in self.gen_groups.items():
            if k == cur:
                g.pack(fill="x")
            else:
                g.pack_forget()

    # ------------------------------------------------------------------ helpers
    def browse(self, var, kind: str):
        if kind == "dir":
            p = filedialog.askdirectory(parent=self.root)
        elif kind == "files":
            p = "\n".join(filedialog.askopenfilenames(parent=self.root))
        elif kind == "adddir":
            p = filedialog.askdirectory(parent=self.root)
        else:
            p = filedialog.askopenfilename(parent=self.root)
        if not p:
            return
        if kind in ("files", "adddir"):
            old = var.get()
            var.set((old + "\n" if old else "") + p)
        else:
            var.set(p)

    def error(self, row_key: str, text: str):
        text = credentials.redact(str(text))
        self.rows[row_key].message(text)
        self.status_var.set(text)
        self.status_label.configure(style="Error.TLabel")

    def ok(self, text: str):
        self.status_var.set(text)
        self.status_label.configure(style="Caption.TLabel")

    @property
    def busy(self) -> bool:
        return bool(self.job and self.job.status == "running")

    # ------------------------------------------------------------------ project / state
    def open_project(self):
        r = self.rows["project"]
        r.message()
        try:
            res = self.studio.open_project({k: r.get(k) for k in ("path", "species", "description")})
        except (FormError, core.ConfigError, credentials.CredentialError, OSError) as e:
            return self.error("project", e)
        self.ok(("Created " if res["created"] else "Opened ") + self.ui.short_path(res["opened"]))
        self.load_values()
        self.refresh()

    def load_values(self):
        """Fill the forms from project.yaml (done when a project is opened, not on every refresh)."""
        st = self.studio.state()
        v = st.get("values")
        if not v:
            return
        self.rows["project"].vars["species"].set(st["species"])
        self.rows["project"].vars["description"].set(v["project"]["description"])
        for stage in ("literature", "prompts", "generate", "propose", "dataset", "train"):
            for k, val in v[stage].items():
                var = self.rows[stage].vars.get(k)
                if var is not None and not (val == "" and k in ("endpoint", "credential")):
                    if isinstance(val, str) and k.endswith(("_path", "weights")) and val:
                        val = self.ui.short_path(val)                 # shown as ~/...; the form expands it again when it is used
                    var.set(val if isinstance(var, tk.BooleanVar) else ("" if val is None else str(val)))
        self.rows["deploy"].vars["n_animals"].set(str(v["deploy"]["n_animals"] or ""))
        self._gen_source()

    def refresh(self):
        """Status lines from the project folder (cheap: counts files and reads small JSON)."""
        try:
            st = self.studio.state()
        except Exception as e:                                    # a broken project.yaml must not kill the window
            self.status_var.set(credentials.redact(f"{type(e).__name__}: {e}"))
            return
        R = self.rows
        if not st.get("project"):
            for k in ROW_ORDER[1:]:
                R[k].status_var.set("")
            R["project"].status_var.set("no project yet")
            self._dots()
            return
        if st.get("error"):
            R["project"].status_var.set("project.yaml: " + st["error"])
            return
        s, modes = st["status"], st["status"]["last_modes"]

        def m(k):
            return f"   [last run: {modes[k]}]" if k in modes else ""
        R["project"].status_var.set(f"{st['species']}   {self.ui.short_path(st['project'])}")
        R["literature"].status_var.set(f"{s['literature']['records']} records ({s['literature']['backend']})" + m("literature"))
        p = s["prompts"]
        R["prompts"].status_var.set(f"{p['items']} prompts" + (f", {100 * p['distinct_fraction']:.0f} % distinct" if p.get("distinct_fraction") is not None else "") + m("prompts"))
        g = s["generate"]
        R["generate"].status_var.set(f"{g['images']} images; {g['units_done']} of {g['units_total']} prompts done ({g['backend']})" + m("generate"))
        R["propose"].status_var.set(f"{s['propose']['proposed']} proposed, {s['propose']['passed_gates']} passed the automatic gates" + m("propose"))
        v = s["review"]
        R["review"].status_var.set(f"{v['pending']} pending, {v['accepted']} accepted ({v['accepted_by_human']} by a person), {v['rejected']} rejected")
        R["dataset"].status_var.set(f"{s['dataset']['train']} train / {s['dataset']['val']} val images" + m("dataset"))
        t = s["train"]
        R["train"].status_var.set(f"{t['accepted_images']} accepted of {t['min_accepted_images']} needed"
                                  + (" (gate passed)" if t["gate_passed"] else " (gate closed)")
                                  + (f"; weights: {t['weights']}" if t.get("weights") else "") + m("train"))
        R["export"].status_var.set((s["export"].get("file") or "nothing exported yet") + m("export"))
        R["deploy"].status_var.set(f"{len(s['predict']['runs'])} prediction run(s), {len(s['track']['runs'])} tracked")
        self._dots(s)

    def _dots(self, s: dict = None):
        """The status dot of every stage in the sidebar: grey = nothing yet, green = has results, orange = running, red = last run failed."""
        kind = {k: "idle" for k, _ in PAGES}
        if self.studio.project:
            kind["project"] = "ok"
        if s:
            try:
                have = {"literature": s["literature"]["records"], "prompts": s["prompts"]["items"], "generate": s["generate"]["images"],
                        "propose": s["propose"]["proposed"], "review": s["review"]["accepted"], "train": bool(s["train"].get("weights")),
                        "inference": len(s["predict"]["runs"]) + len(s["track"]["runs"])}
                for k, v in have.items():
                    if v:
                        kind[k] = "ok"
            except (KeyError, TypeError):
                pass
            self._last_have = {k: v for k, v in kind.items()}
        elif getattr(self, "_last_have", None) and self.studio.project:
            kind.update(self._last_have)
        for k in self._failed:
            kind[k] = "fail"
        if self.job is not None and self._job_row is not None:
            kind[PAGE_OF[self._job_row.key]] = "run"
        for k, rb in self.nav.items():
            rb.configure(image=self.th.dot(kind[k]))

    # ------------------------------------------------------------------ forms -> core
    def values(self, stage: str) -> Optional[dict]:
        r = self.rows.get(stage)
        if stage == "project":
            return {"description": r.get("description")}
        if stage in ("literature", "dataset"):
            return {k: r.get(k) for k in (("backend", "limit") if stage == "literature" else ("task",))}
        if stage == "prompts":
            return {k: r.get(k) for k in ("n", "seed", "megapixels", "size_multiple")}
        if stage == "generate":
            return {k: r.get(k) for k in ("backend", "model_path", "model_id", "pipeline_class", "num_inference_steps",
                                          "true_cfg_scale", "dtype", "split", "endpoint", "model", "style", "credential",
                                          "wan_model_path")}
        if stage == "propose":
            return {k: r.get(k) for k in ("backend", "variant", "prompt", "weights", "pose_weights")}
        if stage == "train":
            return {k: r.get(k) for k in ("backend", "task", "min_accepted_images")}
        return None

    def params(self, action: str) -> dict:
        R = self.rows
        if action in ("literature", "prompts", "propose"):
            return {"force": R[action].get("force")}
        if action == "generate":
            return {"limit": R["generate"].get("limit")}
        if action == "propose_keypoints":
            return {"pose_weights": R["propose"].get("pose_weights")}
        if action == "import_labels":
            r = R["review"]
            return {"format": r.get("import_format"), "path": r.get("import_path"), "images": r.get("import_images")}
        if action == "export_labels":
            return {"format": R["review"].get("export_format"), "out": R["review"].get("export_out")}
        if action == "train":
            r = R["train"]
            return {k: r.get(k) for k in ("force", "dry_run", "epochs", "imgsz")}
        if action in ("deploy_pretrained", "deploy_project"):
            r = R["deploy"]
            return {"videos": [x.strip() for x in r.get("videos").splitlines() if x.strip()], "out": r.get("out"),
                    "n_animals": r.get("n_animals"), "engine": r.get("engine")}
        return {}

    def _store_key(self) -> None:
        """API key field -> credentials file (never project.yaml); the field is cleared whatever happens."""
        r = self.rows["generate"]
        key = r.get("api_key")
        r.vars["api_key"].set("")
        if r.get("backend") == "http_api" and key.strip():
            self.studio.set_credential({"name": r.get("credential"), "value": key})

    def save(self, stage: str) -> bool:
        r = self.rows[stage]
        r.message()
        try:
            if stage == "generate":
                self._store_key()
            self.studio.save(stage, self.values(stage))
        except (FormError, core.ConfigError, credentials.CredentialError, OSError) as e:
            self.error(stage, e)
            return False
        self.ok(f"Saved {stage} settings to project.yaml")
        self.refresh()
        return True

    def slurm(self) -> Optional[dict]:
        if not self.slurm_vars["on"].get():
            return None
        return {k: v.get() for k, v in self.slurm_vars.items() if k != "on"}

    def run(self, action: str) -> bool:
        key = ACTION_ROW[action]
        r = self.rows[key]
        r.message()
        try:
            if action == "generate":
                self._store_key()
            stage = core.SAVE_BEFORE.get(action)
            res = self.studio.run(action, self.values(stage) if stage else None, self.params(action), slurm=self.slurm())
        except (FormError, core.ConfigError, credentials.CredentialError, OSError) as e:
            self.error(key, e)
            return False
        self.job = self.studio.jobs[res["job"]]
        self._offset, self._job_row = 0, r
        self._failed.discard(PAGE_OF[key])
        self.show(PAGE_OF[key])
        self._dots()
        r.clear_log()
        r.toggle_log(True)
        self.stop_button.configure(state="normal")
        self._set_run_buttons("disabled")
        self.ok(f"Running: {action} ...   log: logs/{self.job.log.name}")
        return True

    def run_deploy(self) -> bool:
        return self.run("deploy_pretrained" if self.rows["deploy"].get("model") == "pretrained" else "deploy_project")

    def auto_accept(self) -> bool:
        if not self.confirm("Accept without looking?",
                            "Every proposal that passed the automatic gates will be accepted and every other one rejected, "
                            "with NO person looking at them. Wrong labels that pass the gates go into the training set.\n\n"
                            "Images you already reviewed are not changed. Continue?"):
            return False
        return self.run("review_auto")

    def stop(self):
        if self.job and self.job.status == "running":
            self.job.stop()
            self.ok("Stopping ...")

    def _set_run_buttons(self, state: str):
        for k, r in self.rows.items():
            for name, b in r.buttons.items():
                if k == "project" and name != "save" or name.startswith(("browse", "add_")) or name == "annotate":
                    continue
                b.configure(state=state)

    def _tick(self):
        try:
            job = self.job
            if job is not None:
                info = job.info(self._offset)
                self._offset = info["offset"]
                self._job_row.append_log(info["text"])
                if info["status"] != "running":
                    tail = job.info(self._offset)                 # anything written between the read and the status check
                    self._offset = tail["offset"]
                    self._job_row.append_log(tail["text"])
                    self.job = None
                    self.stop_button.configure(state="disabled")
                    self._set_run_buttons("normal")
                    word = {"done": "Finished", "failed": "FAILED", "stopped": "Stopped"}[info["status"]]
                    self.status_var.set(f"{word}: {job.action} ({info['seconds']:.0f} s)   log: logs/{job.log.name}")
                    self.status_label.configure(style="Error.TLabel" if info["status"] == "failed" else "Caption.TLabel")
                    (self._failed.add if info["status"] == "failed" else self._failed.discard)(PAGE_OF[self._job_row.key])
                    if info["status"] == "failed":
                        self._job_row.message(f"{job.action} failed (exit {info['returncode']}); see the log below.")
                    self.refresh()
                    if self.annotator is not None and self.annotator.winfo_exists():
                        self.annotator.reload_items()
        finally:
            self._after = self.root.after(POLL_MS, self._tick)

    # ------------------------------------------------------------------ other windows
    def open_annotator(self):
        try:
            pr = self.studio.need_project()
        except (FormError, core.ConfigError) as e:
            return self.error("review", e)
        if self.annotator is not None and self.annotator.winfo_exists():
            self.annotator.lift()
            self.annotator.reload_items()
            return self.annotator
        from ..annotator.tkapp import AnnotatorWindow
        self.annotator = AnnotatorWindow(self.root, pr, on_change=self.refresh)
        return self.annotator

    @staticmethod
    def run_app_argv() -> List[str]:
        return [sys.executable, "-m", "animalworld", "run"]

    def open_run(self):
        """Animal World Run (the zero-shot app) as its own process, detached from this window."""
        try:
            kw = {"start_new_session": True} if os.name != "nt" else {}
            subprocess.Popen(self.run_app_argv(), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             env=core.job_env(), shell=False, **kw)
            self.ok("Started Animal World Run in its own window.")
        except OSError as e:
            self.status_var.set(f"could not start Animal World Run: {e}")

    def close(self):
        if self.busy and not self.confirm("A job is running", "Stop the running job and close?"):
            return
        if self.busy:
            self.job.stop()
        try:
            self.root.after_cancel(self._after)
        except tk.TclError:
            pass
        self.root.destroy()


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="animalworld studio", description="Animal World Studio: the pipeline in one desktop window.")
    ap.add_argument("project", nargs="?", help="project folder (optional; create or open one in the window)")
    a = ap.parse_args(argv)
    if a.project and not (Path(a.project) / core.CONFIG_NAME).exists():
        print(f"animalworld: error: {a.project} is not an animalworld project (no {core.CONFIG_NAME}); "
              "start `animalworld studio` without a folder to create one", file=sys.stderr)
        return 2
    try:
        root = tk.Tk()
    except tk.TclError as e:
        print(NO_DISPLAY.format(err=e), file=sys.stderr)
        return 2
    app = StudioApp(root, a.project)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

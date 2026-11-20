"""Animal World Run: a small desktop window (tkinter / ttk, styled by animalworld.ui) for zero-shot analysis of mouse videos.

    animalworld run        (also: mouseworld)

Pick videos (or a folder) and an output folder, press Start. The window decodes each video, proposes the number of
animals with a picture of what it found, waits for you to confirm or correct that number, then runs detection,
identity tracking and keypoints and writes the result files. Nothing else is asked.

Everything runs in this Python process tree: a worker thread drives the engine (zeroshot/pipeline.py, the same code
as `animalworld zeroshot`), each model step is a child process, and log lines come back through a queue. No server.
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional

from .. import __version__, weights as W
from . import pipeline as P

FULL_TEXT = "Full engine: MouseWorld-DET + MouseWorld-Track + MouseWorld-Pose, rows for every camera frame"
LITE_TEXT = ("CPU engine: MouseWorld-DET-Lite with a simple tracker. Tracking quality is LOWER than MouseWorld-Track: "
             "identities are not reliable when animals touch.")


STEPS = ("1  Videos", "2  Animal count", "3  Results")
ENGINE_CHOICES = ("auto", "full", "lite")


class VideoList:
    """The list of videos: one row per video with a status dot, the status text and a rounded progress bar.

    It answers the few Treeview calls the window (and its tests) use: insert, delete, exists, item, selection,
    selection_set."""

    def __init__(self, app: "RunApp", parent):
        self.app, self.th = app, app.th
        tk, ttk, S = app.tk, app.ttk, app.th.S
        self.rows: Dict[str, dict] = {}
        self._sel: Optional[str] = None
        self.outer = ttk.Frame(parent, style="In.Card.TFrame")
        self.canvas = tk.Canvas(self.outer, highlightthickness=0, borderwidth=0, height=S(46))
        self.th.reg(self.canvas, background="card")
        self.scroll = ttk.Scrollbar(self.outer, orient="vertical", command=self.canvas.yview, style="Card.Vertical.TScrollbar")
        self.canvas.configure(yscrollcommand=self.scroll.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(self.canvas, borderwidth=0, highlightthickness=0)
        self.th.reg(self.inner, background="card")
        self._win = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self._resize())
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win, width=e.width))
        self.empty = tk.Label(self.inner, text="No videos yet. Choose mouse videos, or a folder that holds them.", anchor="w",
                              font=self.th.font_caption, pady=S(12))
        self.th.reg(self.empty, background="card", foreground="text2")
        self.empty.pack(fill="x")
        self.th.on_change(self.repaint)

    def _resize(self):
        S = self.th.S
        h = self.inner.winfo_reqheight()
        self.canvas.configure(scrollregion=(0, 0, self.inner.winfo_reqwidth(), h), height=min(h, S(46) * 4))
        if h > S(46) * 4:
            self.scroll.pack(side="right", fill="y")
        else:
            self.scroll.pack_forget()

    def insert(self, _parent, _index, iid: str, values=()):
        tk, ttk, S, th = self.app.tk, self.app.ttk, self.th.S, self.th
        self.empty.pack_forget()
        f = tk.Frame(self.inner, borderwidth=0, highlightthickness=0, padx=S(8), pady=S(6))
        f.pack(fill="x")
        dot = tk.Label(f, image=th.dot("idle"), borderwidth=0)
        dot.grid(row=0, column=0, rowspan=2, padx=(0, S(6)))
        name = tk.Label(f, anchor="w", font=th.font_bold, borderwidth=0)
        name.grid(row=0, column=1, sticky="w")
        status = tk.Label(f, anchor="w", font=th.font_caption, borderwidth=0, width=1)
        status.grid(row=1, column=1, sticky="ew")
        animals = tk.Label(f, anchor="e", font=th.font, borderwidth=0, width=10)
        animals.grid(row=0, column=2, rowspan=2, padx=S(12))
        bar = ttk.Progressbar(f, style="Card.Horizontal.TProgressbar", maximum=100, length=S(150))
        bar.grid(row=0, column=3, rowspan=2)
        pct = tk.Label(f, anchor="e", font=th.font_caption, borderwidth=0, width=5)
        pct.grid(row=0, column=4, rowspan=2, padx=(S(6), 0))
        f.grid_columnconfigure(1, weight=1)
        line = tk.Frame(self.inner, height=max(1, S(1)), borderwidth=0, highlightthickness=0)
        line.pack(fill="x")
        row = dict(frame=f, dot=dot, name=name, status=status, animals=animals, bar=bar, pct=pct, line=line, values=("", "", "", ""))
        self.rows[iid] = row
        for w in (f, dot, name, status, animals, pct):
            w.bind("<Button-1>", lambda e, i=iid: self.selection_set(i))
        self.item(iid, values=values)
        self._paint(iid)

    def delete(self, iid: str):
        row = self.rows.pop(iid, None)
        if row:
            row["frame"].destroy(); row["line"].destroy()
        if self._sel == iid:
            self._sel = None
        if not self.rows:
            self.empty.pack(fill="x")

    def exists(self, iid: str) -> bool:
        return iid in self.rows

    def item(self, iid: str, values=None):
        row = self.rows[iid]
        if values is None:
            return {"values": row["values"]}
        row["values"] = tuple(values)
        video, animals, status, progress = (list(values) + ["", "", "", ""])[:4]
        row["name"].configure(text=video)
        row["status"].configure(text=status)
        row["animals"].configure(text=(f"{animals} animals" if str(animals).isdigit() and int(animals) != 1 else
                                       ("1 animal" if str(animals) == "1" else str(animals))))
        try:
            row["bar"].configure(value=float(str(progress).replace("%", "") or 0))
        except ValueError:
            pass
        row["pct"].configure(text=progress)
        word = str(status).split(":")[0]
        kind = ("ok" if word == "done" else "fail" if word in ("failed", "stopped") else "accent" if word == "CONFIRM THE COUNT"
                else "idle" if word in ("added", "") else "run")
        row["dot"].configure(image=self.th.dot(kind))

    def selection(self):
        return (self._sel,) if self._sel in self.rows else ()

    def selection_set(self, iid: str):
        if iid not in self.rows:
            return
        old, self._sel = self._sel, iid
        for i in (old, iid):
            if i in self.rows:
                self._paint(i)
        self.app._select(iid)

    def _paint(self, iid: str):
        row, c = self.rows[iid], self.th.c
        bg = c("sel") if iid == self._sel else c("card")
        for k in ("frame", "dot", "name", "status", "animals", "pct"):
            row[k].configure(background=bg)
        row["name"].configure(foreground=c("text")); row["animals"].configure(foreground=c("text"))
        row["status"].configure(foreground=c("text2")); row["pct"].configure(foreground=c("text2"))
        row["line"].configure(background=c("line"))
        row["bar"].configure(style=("Sel." if iid == self._sel else "Card.") + "Horizontal.TProgressbar")

    def repaint(self):
        for iid in list(self.rows):
            try:
                self._paint(iid)
                self.item(iid, values=self.rows[iid]["values"])
            except Exception:
                pass


class RunApp:
    def __init__(self, root, engine: str = "auto", out: str = None, videos=(), tracker_detector: str = None, sam3: bool = False):
        import tkinter as tk
        from tkinter import ttk
        from .. import ui
        self.tk, self.ttk, self.root, self.ui = tk, ttk, root, ui
        self.th = th = ui.theme(root)
        S = th.S
        self.want_engine = engine
        self.tracker_detector = tracker_detector
        self.engine: Optional[dict] = None
        self.jobs: Dict[str, P.Job] = {}
        self.order: List[str] = []
        self.events: "queue.Queue" = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self.stop_all = threading.Event()
        self.selected: Optional[str] = None
        self._photo = None
        self._stills: Dict[str, object] = {}
        root.title("Animal World Run")
        root.geometry(th.geometry("1040x800+0+0"))
        root.minsize(S(760), S(560))
        page = ttk.Frame(root, padding=(S(24), S(20), S(24), S(16)))
        page.pack(fill="both", expand=True)

        # ---- header: title, one line of what it does, the three steps
        head = ttk.Frame(page); head.pack(fill="x")
        th.mode_button(head).pack(side="right", anchor="n")
        ttk.Label(head, text="Animal World Run", style="Title.TLabel").pack(anchor="w")
        ttk.Label(head, text="Mouse videos in, tracks out: detection, identity tracking and keypoints with the pretrained MouseWorld models.",
                  style="Caption.TLabel").pack(anchor="w", pady=(S(2), 0))
        steps = ttk.Frame(page); steps.pack(fill="x", pady=(S(12), S(12)))
        self.step_labels = []
        for i, t in enumerate(STEPS):
            if i:
                ttk.Label(steps, image=th.chevron("right"), style="Caption.TLabel").pack(side="left", padx=S(8))
            lab = ttk.Label(steps, text=t, style="Caption.TLabel"); lab.pack(side="left")
            self.step_labels.append(lab)

        # ---- card 1: videos, results folder, Start
        c1 = th.card(page, fill="x")
        row = ttk.Frame(c1, style="In.Card.TFrame"); row.pack(fill="x")
        ttk.Label(row, text="Videos", style="Section.Card.TLabel").pack(side="left")
        self.remove_button = ttk.Button(row, text="Remove", style="Link.Card.TButton", command=self.remove_selected); self.remove_button.pack(side="right")
        self.add_folder_button = ttk.Button(row, text="Add a folder ...", style="Card.TButton", command=self._ask_folder); self.add_folder_button.pack(side="right", padx=S(8))
        self.add_videos_button = ttk.Button(row, text="Choose videos ...", style="Card.TButton", command=self._ask_videos); self.add_videos_button.pack(side="right")
        self.drop_hint = ttk.Label(row, text="", style="Caption.Card.TLabel"); self.drop_hint.pack(side="right", padx=S(12))
        th.hairline(c1, fill="x", pady=(S(10), 0))
        self.tree = VideoList(self, c1)
        self.tree.outer.pack(fill="x")
        row = ttk.Frame(c1, style="In.Card.TFrame"); row.pack(fill="x", pady=(S(10), 0))
        ttk.Label(row, text="Results folder", style="Card.TLabel", width=14).pack(side="left")
        self.out_var = tk.StringVar(value=out or "")
        self.out_button = ttk.Button(row, text="Choose ...", style="Link.Card.TButton", command=self._ask_out); self.out_button.pack(side="right", padx=(S(6), 0))
        self.out_entry = ttk.Entry(row, textvariable=self.out_var); self.out_entry.pack(side="left", fill="x", expand=True)
        row = ttk.Frame(c1, style="In.Card.TFrame"); row.pack(fill="x", pady=(S(12), 0))
        self.start_button = ttk.Button(row, text="Start", style="Primary.Card.TButton", command=self.start, width=10); self.start_button.pack(side="right")
        self.stop_button = ttk.Button(row, text="Stop", style="Card.TButton", command=self.stop, state="disabled", width=8); self.stop_button.pack(side="right", padx=S(8))
        self.device_dot = ttk.Label(row, image=th.dot("idle"), style="Card.TLabel"); self.device_dot.pack(side="left", anchor="n", pady=S(5))
        self.device_var = tk.StringVar(value="Checking this computer (GPU, model libraries, weight files) ...")
        self.device_label = ttk.Label(row, textvariable=self.device_var, style="Caption.Card.TLabel", justify="left")
        self.device_label.pack(side="left", fill="x", expand=True, padx=(S(2), S(12)))
        th.wrap(self.device_label)
        self.msg_var = tk.StringVar(value="")
        self.msg_label = ttk.Label(c1, textvariable=self.msg_var, style="Error.Card.TLabel", justify="left")
        th.wrap(self.msg_label)
        # advanced options, folded away
        self.advanced = adv = ui.Disclosure(th, c1, "Advanced options")
        g = ttk.Frame(adv.body, style="In.Card.TFrame"); g.pack(fill="x")
        g.grid_columnconfigure(1, weight=1)
        ttk.Label(g, text="Engine", style="Card.TLabel").grid(row=0, column=0, sticky="w", pady=S(3), padx=(0, S(16)))
        self.engine_var = tk.StringVar(value=engine)
        self.engine_box = ttk.Combobox(g, textvariable=self.engine_var, state="readonly", width=18,
                                       values=ENGINE_CHOICES + (("mock",) if engine == "mock" else ()))
        self.engine_box.grid(row=0, column=1, sticky="w", pady=S(4))
        self.engine_box.bind("<<ComboboxSelected>>", lambda e: self._options_changed())
        ttk.Label(g, text="Tracker detector", style="Card.TLabel").grid(row=1, column=0, sticky="w", pady=S(3), padx=(0, S(16)))
        self.detector_var = tk.StringVar(value=tracker_detector or P.DEFAULT_TRACKER_DETECTOR)
        self.detector_box = ttk.Combobox(g, textvariable=self.detector_var, state="readonly", width=18, values=tuple(P.TRACKER_DETECTORS))
        self.detector_box.grid(row=1, column=1, sticky="w", pady=S(4))
        self.detector_box.bind("<<ComboboxSelected>>", lambda e: self._options_changed())
        self.ask_var = tk.BooleanVar(value=True)
        self.ask_check = ttk.Checkbutton(g, text="Show me the animal count before tracking (recommended)", variable=self.ask_var, style="Card.TCheckbutton")
        self.ask_check.grid(row=2, column=0, columnspan=2, sticky="w")
        self.sam3_var = tk.BooleanVar(value=bool(sam3) or os.environ.get("ANIMALWORLD_SAM3", "").lower() in ("1", "true", "yes", "on"))
        self.sam3_check = ttk.Checkbutton(g, variable=self.sam3_var, text="Also use SAM 3.1 as a witness (optional)", style="Card.TCheckbutton")
        self.sam3_check.grid(row=3, column=0, columnspan=2, sticky="w")
        lab = ttk.Label(g, style="Caption.Card.TLabel", justify="left",
                        text="SAM 3.1 weights are gated by Meta's licence (huggingface.co/facebook/sam3.1). On the public benchmark the "
                             "tracker needed 20 identity fixes without it and 21 with it.")
        lab.grid(row=4, column=0, columnspan=2, sticky="ew")
        th.wrap(lab)

        # ---- card 2: the selected video: its count to confirm, or its result
        c2 = th.card(page, fill="x", pady=(S(12), 0))
        self.detail_title = tk.StringVar(value="Animal count")
        ttk.Label(c2, textvariable=self.detail_title, style="Section.Card.TLabel").pack(anchor="w")
        body = ttk.Frame(c2, style="In.Card.TFrame"); body.pack(fill="both", expand=True, pady=(S(8), 0))
        self.image_label = ttk.Label(body, style="Card.TLabel"); self.image_label.pack(side="left", anchor="n")
        body.bind("<Configure>", self._refit_image, add="+")
        self.confirm_frame = right = ttk.Frame(body, style="In.Card.TFrame"); right.pack(side="left", fill="both", expand=True, padx=(S(16), 0))
        self.confirm_var = tk.StringVar(value="Select a video to see what was found.")
        lab = ttk.Label(right, textvariable=self.confirm_var, style="Card.TLabel", justify="left"); lab.pack(fill="x", anchor="w")
        th.wrap(lab)
        self.count_row = row = ttk.Frame(right, style="In.Card.TFrame")
        ttk.Label(row, text="Number of animals", style="Card.TLabel").pack(side="left")
        self.count_var = tk.StringVar(value="")
        self.count_spin = ttk.Spinbox(row, from_=1, to=100, width=5, textvariable=self.count_var); self.count_spin.pack(side="left", padx=S(8))
        self.confirm_button = ttk.Button(right, text="Confirm and track", style="Primary.Card.TButton", command=self.confirm_selected, state="disabled")
        self.numbers = ttk.Frame(right, style="In.Card.TFrame")
        self.open_row = ttk.Frame(right, style="In.Card.TFrame")
        self.open_button = ttk.Button(self.open_row, text="Open results", style="Card.TButton", command=self.open_selected, state="disabled")
        self.open_button.pack(side="left")

        # ---- the log, folded away
        self.log_disclosure = d = ui.Disclosure(th, page, "Log", surface="")
        d.button.pack_configure(pady=(S(8), 0))
        self.log_text = th.text(d.body, surface="card", height=8, wrap="none", state="disabled")
        self.log_text.pack(fill="both", expand=True)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._try_drop()
        if videos:
            self.add_paths(videos)
        self._layout_detail(None)
        threading.Thread(target=self._detect_engine, daemon=True).start()
        self.root.after(150, self._poll)

    def _try_drop(self):
        """Dropping files on the window works only where the optional tkinterdnd2 package is installed (not required)."""
        try:
            import tkinterdnd2                                                         # type: ignore
            tkinterdnd2.TkinterDnD._require(self.root)
            self.root.tk.call("tkdnd::drop_target", "register", self.root._w, ("DND_Files",))
            cmd = self.root.register(lambda data: (self.add_paths(self.root.tk.splitlist(data)), "copy")[1])
            self.root.tk.call("bind", self.root._w, "<<Drop>>", cmd + " %D")
            self.drop_hint.configure(text="or drop them on this window")
        except Exception:
            pass

    def _options_changed(self):
        """Engine or tracker detector picked under Advanced options: check the computer again for that choice."""
        if self.busy():
            self._say("finish or stop the running videos before changing the engine")
            self.engine_var.set(self.want_engine); self.detector_var.set(self.tracker_detector or P.DEFAULT_TRACKER_DETECTOR)
            return
        self.want_engine = self.engine_var.get()
        td = self.detector_var.get()
        self.tracker_detector = None if td == P.DEFAULT_TRACKER_DETECTOR and self.tracker_detector is None else td
        self.engine = None
        self.device_var.set("Checking this computer (GPU, model libraries, weight files) ...")
        self.device_dot.configure(image=self.th.dot("idle")); self.device_label.configure(style="Caption.Card.TLabel")
        threading.Thread(target=self._detect_engine, daemon=True).start()

    # ---------------------------------------------------------------- engine choice ------------------------------
    def _detect_engine(self):
        try:
            self.events.put(("engine", W.choose_engine(self.want_engine, self.tracker_detector)))
        except W.WeightsError as e:
            self.events.put(("engine_error", str(e)))

    def _show_engine(self, eng: dict):
        self.engine = eng
        self.device_label.configure(style="Caption.Card.TLabel"); self.device_dot.configure(image=self.th.dot("ok"))
        if eng["engine"] == "full":
            td = eng.get("tracker_detector", P.DEFAULT_TRACKER_DETECTOR)
            self.device_var.set(f"GPU found: {eng['device']}.  {FULL_TEXT}.\nTracker detector: "
                                + {"scored_pair": "the scored pair (default): 20 identity fixes, 73 per hour, on the 34 public benchmark clips (21 with the optional SAM 3.1 witness).",
                                   "strongest": "MouseWorld-DET alone (option 'strongest'): about 25 % less computing, but 42 identity fixes on the 34 "
                                                "public benchmark clips where the default scored pair needed 21.",
                                   "strongest_two_seeds": "MouseWorld-DET with a second seed (option 'strongest_two_seeds'): 39 identity fixes on the 34 "
                                                          "public benchmark clips where the default scored pair needed 21.",
                                   "strongest_recut": "MouseWorld-DET alone with a moved confidence cut (option 'strongest_recut'): experimental."}[td])
        elif eng["engine"] == "lite":
            why = eng["why"]
            short = []
            if any("no CUDA device" in w for w in why) or not W.nvidia_smi():
                short.append("no NVIDIA GPU is visible")
            if any("cannot import" in w for w in why):
                short.append("the GPU libraries are not installed in this environment (bash install.sh --gpu)")
            if any("not found" in w for w in why):
                short.append("model files are missing (animalworld weights)")
            self.device_var.set("No usable GPU setup on this computer.  " + LITE_TEXT + "\nWhy: " + ("; ".join(short) or "see the log below") + ".")
            self.device_label.configure(style="Error.Card.TLabel"); self.device_dot.configure(image=self.th.dot("fail"))
            for w in why:
                self._line("full engine not available: " + w)
        else:
            self.device_var.set("TEST engine (blob detector, no model). For checking the window only.")

    # ---------------------------------------------------------------- inputs -------------------------------------
    def _ask_videos(self):
        from tkinter import filedialog
        ext = " ".join("*" + e for e in P.VIDEO_EXT)
        self.add_paths(filedialog.askopenfilenames(parent=self.root, title="Mouse videos", filetypes=[("Videos", ext), ("All files", "*")]))

    def _ask_folder(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(parent=self.root, title="Folder with mouse videos")
        if d:
            self.add_paths([d])

    def _ask_out(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(parent=self.root, title="Folder for the results")
        if d:
            self.out_var.set(d)

    def add_paths(self, paths) -> int:
        try:
            vids = P.collect([p for p in paths if str(p).strip()])
        except FileNotFoundError as e:
            self._say(str(e))
            return 0
        have = {str(j.video) for j in self.jobs.values()}
        n = 0
        for v in vids:
            if str(v) in have:
                continue
            name = P.slug(v.stem)
            while name in self.jobs:
                name += "_2"
            job = P.Job(v, self.out_var.get() or v.parent / "animalworld_results", P.Options(tracker_detector=self.tracker_detector), name=name)
            job.on_change = lambda j: self.events.put(("job", j.name))
            job.on_line = lambda line, n_=name: self.events.put(("line", f"[{n_}] {line}"))
            self.jobs[name] = job; self.order.append(name)
            self.tree.insert("", "end", iid=name, values=(v.name, "", "added", ""))
            n += 1
        if vids and not self.out_var.get():
            self.out_var.set(str(vids[0].parent / "animalworld_results"))
        if n == 0 and paths:
            self._say("no new video file in the selection (" + ", ".join(P.VIDEO_EXT) + ")")
        return n

    def remove_selected(self):
        j = self.jobs.get(self.selected or "")
        if j and j.status in ("added", "done", "failed", "stopped"):
            self.tree.delete(j.name); self.order.remove(j.name); del self.jobs[j.name]; self._select(None)

    # ---------------------------------------------------------------- run ----------------------------------------
    def start(self):
        if self.worker and self.worker.is_alive():
            return
        if not self.engine:
            self._say("still checking the computer; press Start again in a moment")
            return
        todo = [self.jobs[n] for n in self.order if self.jobs[n].status in ("added", "failed", "stopped")]
        if not todo:
            self._say("add at least one video first")
            return
        out = self.out_var.get().strip()
        if not out:
            self._say("choose a folder for the results first")
            return
        try:
            Path(out).expanduser().mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self._say(f"cannot create the results folder: {e}")
            return
        self._say("")
        for j in todo:
            j.dir = Path(out).expanduser().resolve() / j.name; j.work = j.dir / "_work"
            if self.detector_var.get() != (j.opt.tracker_detector or P.DEFAULT_TRACKER_DETECTOR):
                j.opt.tracker_detector = self.detector_var.get()
            j.opt.engine = self.engine["engine"]; j.stop.clear(); j.warnings = []; j.status = "added"
        self.stop_all.clear()
        self.start_button.configure(state="disabled"); self.stop_button.configure(state="normal")
        self.worker = threading.Thread(target=self._work, args=(todo, bool(self.ask_var.get()), bool(self.sam3_var.get())), daemon=True)
        self.worker.start()

    def _work(self, todo: List[P.Job], ask: bool, sam3: bool = False):
        """Worker thread. Phase 1 for every video (decode + count), then phase 2 for each video once its count is confirmed."""
        if sam3 and self.engine["engine"] == "full":
            ok, why = W.sam3_available()
            if not ok:
                self.events.put(("line", "SAM 3.1 was ticked but cannot be used, so the run continues WITHOUT it: " + why))
                sam3 = False
        for j in todo:
            j.opt.sam3 = bool(sam3) and self.engine["engine"] == "full"
        for j in todo:
            if self.stop_all.is_set():
                break
            P.guarded(P.prepare_and_count, j, self.engine)
            if j.status == "confirm" and not ask:
                j.n = j.n_auto
                j.log(f"automatic count {j.n_auto} accepted without a look")
                j.set("queued", message=f"{j.n} animals (automatic count, not looked at)")
        while not self.stop_all.is_set():
            ready = [j for j in todo if j.status == "queued"]
            if ready:
                P.guarded(P.analyse, ready[0])
                continue
            if not [j for j in todo if j.status == "confirm"]:
                break
            self.stop_all.wait(0.3)
        self.events.put(("finished", None))

    def stop(self):
        self.stop_all.set()
        for j in self.jobs.values():
            j.stop.set()
            if j.status in ("confirm", "queued"):
                j.set("stopped", message="stopped")
        self._line("stopping ...")

    def confirm(self, name: str, n: int) -> bool:
        j = self.jobs.get(name)
        if not j or j.status != "confirm" or not 1 <= int(n) <= 100:
            return False
        j.n = int(n); j.confirmed_by_user = True
        j.log(f"number of animals confirmed by the user: {j.n} (automatic count: {j.n_auto})")
        j.set("queued", message=f"{j.n} animals (confirmed); waiting for its turn")
        self.events.put(("say", ""))
        return True

    def confirm_selected(self):
        v = self.count_var.get().strip()
        if not v.isdigit() or not self.confirm(self.selected or "", int(v)):
            self._say("type the number of animals (1 to 100) for the selected video")

    def open_selected(self):
        j = self.jobs.get(self.selected or "")
        if not j or not j.dir.is_dir():
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(j.dir))                                             # noqa: S606
            else:
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(j.dir)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            self._say(f"results are in {j.dir}")

    # ---------------------------------------------------------------- display ------------------------------------
    def _line(self, text: str):
        t = self.log_text
        t.configure(state="normal"); t.insert("end", text.rstrip("\n") + "\n")
        if int(t.index("end-1c").split(".")[0]) > 3000:
            t.delete("1.0", "500.0")
        t.see("end"); t.configure(state="disabled")

    def _say(self, text: str):
        """A message for the person at the window: shown under the Start button (and kept in the log)."""
        self.msg_var.set(text)
        if text:
            self.msg_label.pack(fill="x", pady=(self.th.S(6), 0), before=self.advanced.button)
            self._line(text)
        else:
            self.msg_label.pack_forget()

    def _refresh(self, name: str):
        j = self.jobs.get(name)
        if not j or not self.tree.exists(name):
            return
        animals = "" if j.n_auto is None else (f"{j.n}" if j.n else f"{j.n_auto} ?")
        label = {"confirm": "CONFIRM THE COUNT", "tracking": "running", "preparing": "reading", "counting": "counting"}.get(j.status, j.status)
        text = j.message if j.message and j.message.lower().startswith(label.lower()) else (f"{label}: {j.message}" if j.message else label)
        self.tree.item(name, values=(j.video.name, animals, text, f"{100 * j.progress:.0f} %"))
        if j.status == "confirm" and (self.selected is None or self.jobs.get(self.selected or "") is None or self.jobs[self.selected].status != "confirm"):
            self.tree.selection_set(name)
        if name == self.selected:
            self._select(name)
        self._steps()

    def _steps(self):
        """Which of the three steps the window is at, and which button is the one to press."""
        st = [j.status for j in self.jobs.values()]
        step = 1 if "confirm" in st else 2 if st and all(x == "done" for x in st) else 0 if not self.busy() else 1 if any(x in ("queued", "tracking") for x in st) else 0
        for i, lab in enumerate(self.step_labels):
            lab.configure(style="Bold.TLabel" if i == step else "Caption.TLabel")
        j = self.jobs.get(self.selected or "")
        todo = any(x in ("added", "failed", "stopped") for x in st)
        self.open_button.configure(style="Primary.Card.TButton" if j and j.status == "done" and not todo and "confirm" not in st else "Card.TButton")
        self.start_button.configure(style="Primary.Card.TButton" if todo or not st else "Card.TButton")

    def _layout_detail(self, j):
        """Card 2 shows the count controls while a count waits, the numbers and Open results when a video is done."""
        S = self.th.S
        for w in (self.count_row, self.confirm_button, self.numbers, self.open_row):
            w.pack_forget()
        for w in self.numbers.winfo_children():
            w.destroy()
        if j is not None and j.status == "confirm":
            self.detail_title.set("Animal count")
            self.count_row.pack(anchor="w", pady=(S(12), 0))
            self.confirm_button.pack(anchor="w", pady=(S(10), 0))
        elif j is not None and j.status == "done":
            self.detail_title.set("Result")
            for k, (cap, val) in enumerate(self._key_numbers(j)):
                self.ttk.Label(self.numbers, text=val, style="Section.Card.TLabel").grid(row=2 * (k // 2), column=k % 2, sticky="w", padx=(0, S(24)), pady=(S(8), 0))
                self.ttk.Label(self.numbers, text=cap, style="Caption.Card.TLabel").grid(row=2 * (k // 2) + 1, column=k % 2, sticky="w", padx=(0, S(24)))
            self.numbers.pack(anchor="w", pady=(S(4), 0))
            self.open_row.pack(anchor="w", pady=(S(16), 0))
        else:
            self.detail_title.set("Animal count" if j is None or j.status != "failed" else "Not finished")

    def _key_numbers(self, j) -> list:
        """The few numbers of summary.json worth a glance (read from the file the run wrote; nothing is computed here)."""
        try:
            import json
            s = json.loads((j.dir / "summary.json").read_text(encoding="utf-8"))
        except Exception:
            return []
        out = [("animals", str(s.get("animals", "")))]
        if s.get("frames_with_all_animals_pct") is not None:
            out.append(("frames with every animal", f"{s['frames_with_all_animals_pct']:.0f} %"))
        if s.get("rows") is not None:
            out.append(("rows in tracks.csv", f"{s['rows']:,}"))
        if s.get("rows_tracked") is not None:
            out.append(("tracked / filled rows", f"{s['rows_tracked']:,} / {s.get('rows_filled', 0):,}"))
        if s.get("rows_flagged") is not None:
            out.append(("rows the tracker flagged", f"{s['rows_flagged']:,}"))
        if s.get("seconds_of_computing") is not None:
            out.append(("seconds of computing", f"{s['seconds_of_computing']:,.0f}"))
        return out

    def _select(self, name: Optional[str]):
        self.selected = name
        j = self.jobs.get(name or "")
        self.confirm_button.configure(state="normal" if j and j.status == "confirm" else "disabled")
        self.open_button.configure(state="normal" if j and j.status == "done" else "disabled")
        shown = (name, j.status if j else None)
        if getattr(self, "_detail_for", None) != shown:
            self._detail_for = shown
            self._layout_detail(j)
        if not j:
            self.confirm_var.set("Select a video to see what was found."); self.image_label.configure(image=""); self._photo = None
            return
        if j.status == "confirm":
            self.confirm_var.set(f"{j.video.name}: {j.n_auto} animal(s) found on the sample frames (green = confident detection). "
                                 "Count the animals yourself, correct the number if needed, then press Confirm.")
            if not self.count_var.get().strip() or getattr(self, "_count_for", None) != name:
                self.count_var.set(str(max(1, j.n_auto or 1))); self._count_for = name
        elif j.status == "done":
            self.confirm_var.set(f"{j.video.name}: {j.message}.  Files: tracks.csv, masks.npz, overlay.mp4, summary.html in {self.ui.short_path(j.dir)}"
                                 + ("\nWarnings: " + " | ".join(j.warnings[:3]) if j.warnings else ""))
        else:
            self.confirm_var.set(f"{j.video.name}: {j.status}. {j.message}")
        if j.status == "done":
            still = self._overlay_still(j)
            if still is not None:
                self._show_image(still)
            else:
                img = j.dir / "contact_sheet.jpg"
                self._show_image(img if img.exists() else None)
        else:
            img = j.count_sheet
            self._show_image(img if img and Path(img).exists() else None)
        self._steps()

    def _refit_image(self, _event=None):
        """The window was resized: draw the picture again at the width that fits now."""
        src = getattr(self, "_image_src", None)
        if src is None:
            return
        room = self.image_label.master.winfo_width()
        want = max(self.th.S(220), min(self.th.S(520), int(room * 0.56)))
        if abs(want - getattr(self, "_image_w", 0)) > self.th.S(12):
            self._show_image(src)

    def _overlay_still(self, j):
        """The middle frame of overlay.mp4 as a picture for the result card (display only; no file is written)."""
        if j.name in self._stills:
            return self._stills[j.name]
        im = None
        try:
            import cv2
            from PIL import Image
            cap = cv2.VideoCapture(str(j.dir / "overlay.mp4"))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            for _ in range(max(0, n // 2)):
                if not cap.grab():
                    break
            ok, frame = cap.read()
            cap.release()
            if ok:
                im = Image.fromarray(frame[:, :, ::-1].copy())
        except Exception:
            im = None
        self._stills[j.name] = im
        return im

    def _show_image(self, path):
        if path is None or path == "":
            self.image_label.configure(image=""); self._photo = None; self._image_src = None
            return
        try:
            from PIL import Image, ImageTk
            if isinstance(path, (str, Path)):
                with Image.open(path) as im:
                    im = im.convert("RGB")
            else:
                im = path.copy()
            self._image_src = path
            room = self.image_label.master.winfo_width()                 # the picture takes at most 56 % of the card
            wmax = self.th.S(520) if room < 50 else max(self.th.S(220), min(self.th.S(520), int(room * 0.56)))
            self._image_w = wmax
            im.thumbnail((wmax, self.th.S(300)))
            self._photo = ImageTk.PhotoImage(im, master=self.root)
            self.image_label.configure(image=self._photo)
        except Exception as e:                      # a picture that cannot be shown must not stop the run
            self._line(f"(cannot show {path}: {type(e).__name__})")

    def _poll(self):
        try:
            for _ in range(400):
                kind, val = self.events.get_nowait()
                if kind == "line":
                    self._line(val)
                elif kind == "job":
                    self._refresh(val)
                elif kind == "say":
                    self._say(val)
                elif kind == "engine":
                    self._show_engine(val)
                elif kind == "engine_error":
                    self.device_var.set("This computer cannot run the analysis yet:\n" + val)
                    self.device_label.configure(style="Error.Card.TLabel"); self.device_dot.configure(image=self.th.dot("fail"))
                elif kind == "finished":
                    self.start_button.configure(state="normal"); self.stop_button.configure(state="disabled")
                    self._line("finished"); self._steps()
        except queue.Empty:
            pass
        self._after = self.root.after(150, self._poll)

    def busy(self) -> bool:
        return bool(self.worker and self.worker.is_alive())

    def close(self):
        self.stop()
        self.root.after(300, self.root.destroy)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="animalworld run", description="Animal World Run: zero-shot analysis of mouse videos in a small window.")
    ap.add_argument("videos", nargs="*", help="videos or folders to list at start (optional)")
    ap.add_argument("--out", help="results folder (optional; can be chosen in the window)")
    ap.add_argument("--engine", default="auto", choices=("auto", "full", "lite", "mock"))
    ap.add_argument("--tracker-detector", choices=P.TRACKER_DETECTORS, help="scored_pair (default), strongest or strongest_two_seeds; see `animalworld zeroshot --help`")
    ap.add_argument("--sam3", action="store_true", help="start with the optional SAM 3.1 witness ticked (gated weights; see `animalworld zeroshot --help`)")
    ap.add_argument("--version", action="version", version=f"animalworld {__version__}")
    a = ap.parse_args(argv)
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                           # no tkinter, or no display
        print(f"Animal World Run needs a desktop display ({type(e).__name__}: {e}).\n"
              "On a remote machine use `ssh -X` or a remote desktop; without a display use the same engine from the command line:\n"
              "  animalworld zeroshot VIDEOS... --out FOLDER", file=sys.stderr)
        return 2
    RunApp(root, a.engine, a.out, a.videos, a.tracker_detector, a.sam3)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

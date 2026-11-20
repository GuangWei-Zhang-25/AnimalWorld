"""Desktop annotator (tkinter + Pillow): the human-in-the-loop review window.

    animalworld review PROJECT --gui          (also opened from Animal World Studio, stage "Review")

Left: the images with their status. Centre: the image on a canvas with boxes, outlines and keypoints.
Keys (same as the earlier web annotator):
    A accept      R reject      B draw a box (drag)      P draw an outline (click points, Enter closes)
    K place keypoints in order on the selected animal    Delete remove the selected animal / keypoint
    V visible <-> hidden for the selected keypoint       Ctrl+Z undo      Esc cancel the tool
    Up / Down (or PgUp / PgDn) previous / next image     wheel or + / - zoom      middle-drag or space-drag pan      F fit
Click selects; drag a box to move it, a corner handle to resize it, a keypoint or an outline vertex to move it.

Nothing is stored in this window: labels are read with `stages.review.reviewed / proposed` and every decision is
written at once with `annotator.model.clean_instances` + `stages.review.save_review(..., "human", "annotator")`,
the same functions the web annotator used.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import List, Optional

from ..formats import labels as L
from ..stages import review as R
from ..stages.generate import index_rows
from .model import Doc, View
from .model import clean_instances

COLORS = ("#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#008080", "#9a6324", "#800000", "#808000", "#000075")
FILTERS = ("all", "pending", "accepted", "rejected")
HIT_PX = 7               # pick radius on screen


class AnnotatorWindow(tk.Toplevel):
    def __init__(self, parent, project, on_change=None, geometry: str = "1180x760+40+40"):
        super().__init__(parent)
        self.project = project
        self.on_change = on_change
        from .. import ui
        self.th = ui.theme(parent)
        self.title(f"Animal World annotator: {project.species}  ({ui.short_path(project.root)})")
        self.geometry(self.th.geometry(geometry))
        self.th.reg(self, background="bg")
        slots = project.slots()
        self.categories = project.categories()
        self.kp_names: List[str] = project.keypoints()
        self.skeleton = [tuple(e) for e in (slots.get("skeleton") or []) if len(e) == 2]
        self.rows: List[dict] = []
        self.shown: List[dict] = []          # rows after the filter
        self.cur: Optional[dict] = None
        self.doc: Optional[Doc] = None
        self.image = None                    # PIL image of the current item
        self._photo = None
        self.view = View()
        self.mode = "select"                 # select | box | poly | kp
        self.sel_kp: Optional[str] = None
        self._drag = None                    # what the pressed mouse button is doing
        self._poly_pts: List[tuple] = []
        self._space = False
        self._pan = (0, 0)
        self._sam_job = None
        self.mode_var = tk.StringVar(value="select")
        self.filter_var = tk.StringVar(value="all")
        self.counts_var = tk.StringVar(value="")
        self.msg_var = tk.StringVar(value="")
        self._build()
        self.reload_items()
        self.after(50, self._first_fit)

    # ------------------------------------------------------------------ layout
    def _build(self):
        th, S = self.th, self.th.S
        left = ttk.Frame(self, style="Side.TFrame", padding=(S(12), S(14), S(12), S(12)))
        left.pack(side="left", fill="y")
        ttk.Label(left, text="Images", style="Section.Side.TLabel").pack(anchor="w")
        top = ttk.Frame(left, style="Side.TFrame")
        top.pack(fill="x", pady=(S(8), S(8)))
        ttk.Label(top, text="Show", style="Side.TLabel").pack(side="left")
        self.filter_box = ttk.Combobox(top, textvariable=self.filter_var, values=FILTERS, state="readonly", width=12)
        self.filter_box.pack(side="left", padx=S(8))
        self.filter_box.bind("<<ComboboxSelected>>", lambda e: (self._fill_list(), self.canvas.focus_set()))
        self.counts_label = ttk.Label(left, textvariable=self.counts_var, wraplength=S(230), justify="left", style="Caption.Side.TLabel")
        self.counts_label.pack(side="bottom", fill="x", pady=(S(8), 0))
        lf = ttk.Frame(left, style="OnSide.Card.TFrame", padding=S(6))
        lf.pack(fill="y", expand=True)
        self.listbox = th.listbox(lf, width=26, exportselection=False, font=th.font_mono)
        sb = ttk.Scrollbar(lf, orient="vertical", command=self.listbox.yview, style="Card.Vertical.TScrollbar")
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.pack(side="left", fill="y", expand=True)
        sb.pack(side="left", fill="y")
        self.listbox.bind("<<ListboxSelect>>", self._on_list)

        right = ttk.Frame(self, padding=(S(14), S(14), S(14), S(12)))
        right.pack(side="right", fill="y")
        self.buttons = {}

        def btn(key, text, cmd, primary=False):
            b = ttk.Button(right, text=text, command=cmd, width=24, style="Primary.TButton" if primary else "TButton")
            b.pack(fill="x", pady=S(2))
            self.buttons[key] = b

        def head(text):
            ttk.Label(right, text=text, style="Small.TLabel").pack(anchor="w", pady=(S(12), S(2)))
        btn("accept", "Accept  (A)", self.accept, primary=True)
        btn("reject", "Reject  (R)", self.reject)
        head("TOOL")
        for key, text in (("select", "Select / move"), ("box", "Draw box  (B)"), ("poly", "Draw outline  (P)"),
                          ("kp", "Place keypoints  (K)")):
            rb = ttk.Radiobutton(right, text=text, value=key, variable=self.mode_var, command=lambda k=key: self.set_mode(k))
            rb.pack(anchor="w")
            self.buttons["mode_" + key] = rb
        head("EDIT")
        btn("delete", "Delete selected  (Del)", self.delete_selected)
        btn("undo", "Undo  (Ctrl+Z)", self.undo)
        btn("revert", "Back to the proposal", self.revert)
        btn("sam", "Mask from box (SAM 2.1)", self.mask_from_box)
        head("IMAGE")
        btn("prev", "Previous image  (Up)", lambda: self.step(-1))
        btn("next", "Next image  (Down)", lambda: self.step(1))
        btn("fit", "Fit to window  (F)", self.fit)
        head("KEYPOINTS OF THE SELECTED ANIMAL")
        kf = ttk.Frame(right, style="Card.TFrame", padding=S(6))
        kf.pack(fill="x")
        self.kp_list = th.listbox(kf, height=max(4, min(12, len(self.kp_names))), exportselection=False, font=th.font_mono, width=26)
        self.kp_list.pack(fill="x")
        ttk.Label(right, textvariable=self.msg_var, wraplength=S(220), justify="left", style="Caption.TLabel").pack(fill="x", pady=S(6))
        ttk.Label(right, text="wheel: zoom   middle-drag: pan\nV: keypoint visible / hidden\nEsc: cancel tool",
                  justify="left", style="Small.TLabel").pack(anchor="w", side="bottom")

        self.canvas = tk.Canvas(self, highlightthickness=0, cursor="tcross")
        th.reg(self.canvas, background="canvas")
        self.canvas.pack(side="left", fill="both", expand=True)
        c = self.canvas
        c.bind("<ButtonPress-1>", self._press)
        c.bind("<B1-Motion>", self._motion)
        c.bind("<ButtonRelease-1>", self._release)
        c.bind("<Double-Button-1>", lambda e: self.close_poly() if self.mode == "poly" else None)
        c.bind("<ButtonPress-2>", self._pan_start)
        c.bind("<B2-Motion>", self._pan_move)
        c.bind("<ButtonPress-3>", self._pan_start)
        c.bind("<B3-Motion>", self._pan_move)
        c.bind("<MouseWheel>", lambda e: self.zoom(1.2 if e.delta > 0 else 1 / 1.2, e.x, e.y))
        c.bind("<Button-4>", lambda e: self.zoom(1.2, e.x, e.y))
        c.bind("<Button-5>", lambda e: self.zoom(1 / 1.2, e.x, e.y))
        c.bind("<Configure>", lambda e: self.redraw())
        c.bind("<Enter>", lambda e: c.focus_set())
        for w in (self, c):
            w.bind("<Key>", self._key)
            w.bind("<KeyRelease-space>", lambda e: setattr(self, "_space", False))
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def _first_fit(self):
        self.fit()
        self.canvas.focus_set()

    # ------------------------------------------------------------------ items
    def status_of(self, iid: str) -> str:
        rv = R.reviewed(self.project, iid)
        return rv["status"] if rv else "pending"

    def reload_items(self, keep: str = None):
        self.rows = index_rows(self.project)
        self._fill_list(keep or (self.cur["id"] if self.cur else None))

    def _fill_list(self, keep: str = None):
        keep = keep or (self.cur["id"] if self.cur else None)
        f = self.filter_var.get()
        self.shown = []
        self.listbox.delete(0, "end")
        for r in self.rows:
            st = self.status_of(r["id"])
            if f != "all" and st != f and r["id"] != keep:
                continue
            self.shown.append(r)
            self.listbox.insert("end", f"{ {'pending': ' ', 'accepted': '+', 'rejected': 'x'}[st] } {r['id']}")
        c = R.counts(self.project)
        self.counts_var.set(f"{c['pending']} pending, {c['accepted']} accepted ({c['accepted_by_human']} by a person), "
                            f"{c['rejected']} rejected, of {c['images']}")
        ids = [r["id"] for r in self.shown]
        if keep in ids:
            self._select_index(ids.index(keep), load=self.cur is None or self.cur["id"] != keep)
        elif self.shown:
            self._select_index(0, load=True)
        else:
            self.cur = self.doc = self.image = None
            self.redraw()

    def _select_index(self, k: int, load: bool = True):
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(k)
        self.listbox.see(k)
        if load:
            self.load(self.shown[k])

    def _on_list(self, _e=None):
        s = self.listbox.curselection()
        if s and (self.cur is None or self.shown[s[0]]["id"] != self.cur["id"]):
            self.load(self.shown[s[0]])
        self.canvas.focus_set()

    def load(self, row: dict, proposal_only: bool = False):
        from PIL import Image
        self.cur = row
        d = (None if proposal_only else R.reviewed(self.project, row["id"])) or R.proposed(self.project, row["id"]) or {}
        self.doc = Doc(d.get("instances", []), row["W"], row["H"], self.categories[0], self.kp_names)
        p = (self.project.root / row["file"]).resolve()
        try:
            with Image.open(p) as im:
                self.image = im.convert("RGB")
        except OSError as e:
            self.image = None
            self.msg_var.set(f"image file missing: {e}")
        self.sel_kp, self._poly_pts, self._drag = None, [], None
        self.set_mode("select")
        why = (R.proposed(self.project, row["id"]) or {}).get("reason", "")
        self.msg_var.set(f"{row['id']}: {self.status_of(row['id'])}" + (f"\nautomatic gate: {why}" if why else ""))
        self.fit()

    def step(self, d: int):
        if not self.shown or not self.cur:
            return
        ids = [r["id"] for r in self.shown]
        k = ids.index(self.cur["id"]) if self.cur["id"] in ids else 0
        k = min(max(0, k + d), len(ids) - 1)
        self._select_index(k, load=True)

    # ------------------------------------------------------------------ decisions
    def _save(self, status: str):
        if not self.cur or not self.doc:
            return
        if self.mode == "poly" and len(self._poly_pts) >= 3:
            self.close_poly()
        inst = clean_instances(self.doc.instances, self.categories, self.kp_names)
        R.save_review(self.project, self.cur, status, inst, "human", "annotator")
        done = self.cur["id"]
        ids = [r["id"] for r in self.shown]
        k = ids.index(done) if done in ids else -1
        nxt = next((r for r in self.shown[k + 1:] if self.status_of(r["id"]) == "pending"), None) \
            or (self.shown[k + 1] if 0 <= k + 1 < len(self.shown) else None)
        self.cur = None
        self._fill_list(keep=nxt["id"] if nxt else done)
        if self.on_change:
            self.on_change()

    def accept(self):
        self._save("accepted")

    def reject(self):
        self._save("rejected")

    def revert(self):
        if self.cur:
            self.load(self.cur, proposal_only=True)

    # ------------------------------------------------------------------ tools
    def set_mode(self, mode: str):
        if mode == "kp":
            if not self.kp_names:
                self.msg_var.set("this species has no keypoints (slots.yaml: keypoints)")
                mode = "select"
            elif not self.doc or self.doc.selected is None:
                self.msg_var.set("select an animal (or draw its box) first, then place keypoints")
                mode = "select"
            else:
                nxt = self.doc.next_keypoint(self.doc.selected)
                self.msg_var.set(f"click: {nxt}" if nxt else "all keypoints are placed; drag them to adjust")
                if not nxt:
                    mode = "select"
        elif mode == "box":
            self.msg_var.set("drag a box around one animal")
        elif mode == "poly":
            self.msg_var.set("click the outline point by point; Enter (or double-click) closes it")
        if mode != "poly":
            self._poly_pts = []
        self.mode = mode
        self.mode_var.set(mode)
        self.redraw()

    def undo(self):
        if self.mode == "poly" and self._poly_pts:
            self._poly_pts.pop()
        elif self.doc and self.doc.undo():
            self.sel_kp = None
        self.redraw()

    def delete_selected(self):
        d = self.doc
        if not d or d.selected is None:
            return
        if self.sel_kp and self.sel_kp in (d.instances[d.selected].get("keypoints") or {}):
            d.delete_keypoint(d.selected, self.sel_kp)
            self.sel_kp = None
        else:
            d.delete_instance(d.selected)
        self.redraw()

    def close_poly(self):
        if self.doc and len(self._poly_pts) >= 3:
            if self.doc.add_poly(self._poly_pts) is None:
                self.msg_var.set("outline too small")
        self._poly_pts = []
        self.set_mode("select")

    # ---- mask from box (SAM 2.1 in a subprocess; optional) ----
    def sam_options(self) -> Optional[dict]:
        o = self.project.options("propose", "sam")
        return o if o.get("config") and o.get("checkpoint") else None

    def mask_from_box(self):
        d = self.doc
        if not d or d.selected is None or not self.cur:
            self.msg_var.set("select an animal first")
            return
        o = self.sam_options()
        if not o:
            self.msg_var.set("Mask from box needs SAM 2.1: set propose.options_for.sam.config and .checkpoint in "
                             "project.yaml (e.g. configs/sam2.1/sam2.1_hiera_l.yaml and the path of sam2.1_hiera_large.pt), "
                             "and install torch + sam2. Until then, draw the outline with P.")
            return
        if self._sam_job:
            return
        tmp = self.project.path("labels", "_annotator_tmp")
        tmp.mkdir(parents=True, exist_ok=True)
        req, res = tmp / "sam2box_request.json", tmp / "sam2box_result.json"
        if res.exists():
            res.unlink()
        req.write_text(json.dumps({"image": str((self.project.root / self.cur["file"]).resolve()),
                                   "boxes": [d.instances[d.selected]["box"]], "config": o["config"],
                                   "checkpoint": str(self.project.resolve(o["checkpoint"])), "device": o.get("device")}))
        argv = [sys.executable, "-m", "animalworld.annotator.sam2box", str(req), str(res)]
        job = {"id": self.cur["id"], "i": d.selected, "res": res, "rc": None, "err": ""}

        def work():
            try:
                p = subprocess.run(argv, capture_output=True, text=True, stdin=subprocess.DEVNULL, shell=False)
                job["rc"], job["err"] = p.returncode, (p.stderr or "").strip()[-400:]
            except OSError as e:
                job["rc"], job["err"] = 127, str(e)
        self._sam_job = job
        self.msg_var.set("SAM 2.1 is computing the mask ... (the window stays usable)")
        threading.Thread(target=work, daemon=True).start()
        self.after(300, self._sam_poll)

    def _sam_poll(self):
        job = self._sam_job
        if not job:
            return
        if job["rc"] is None:
            self.after(300, self._sam_poll)
            return
        self._sam_job = None
        if job["rc"] != 0:
            self.msg_var.set(f"Mask from box failed: {job['err'] or 'exit ' + str(job['rc'])}")
            return
        poly = (json.loads(Path(job["res"]).read_text()).get("polys") or [None])[0]
        if not poly:
            self.msg_var.set("SAM 2.1 returned no mask for this box")
        elif self.cur and self.cur["id"] == job["id"] and self.doc and job["i"] < len(self.doc.instances):
            self.doc.set_poly(job["i"], poly)
            self.msg_var.set("mask replaced from the box (Ctrl+Z undoes it)")
            self.redraw()

    # ------------------------------------------------------------------ view
    def fit(self):
        if self.cur:
            self.update_idletasks()
            self.view.fit(self.cur["W"], self.cur["H"], self.canvas.winfo_width(), self.canvas.winfo_height())
        self.redraw()

    def zoom(self, factor: float, cx: float = None, cy: float = None):
        cx = self.canvas.winfo_width() / 2 if cx is None else cx
        cy = self.canvas.winfo_height() / 2 if cy is None else cy
        self.view.zoom_at(cx, cy, factor)
        self.redraw()

    def _pan_start(self, e):
        self._pan = (e.x, e.y)

    def _pan_move(self, e):
        self.view.pan(e.x - self._pan[0], e.y - self._pan[1])
        self._pan = (e.x, e.y)
        self.redraw()

    # ------------------------------------------------------------------ mouse
    def _press(self, e):
        self.canvas.focus_set()
        d = self.doc
        if not d:
            return
        if self._space:
            self._pan = (e.x, e.y)
            self._drag = ("pan",)
            return
        x, y = self.view.to_image(e.x, e.y)
        if self.mode == "box":
            self._drag = ("newbox", x, y, x, y)
        elif self.mode == "poly":
            self._poly_pts.append((x, y))
        elif self.mode == "kp":
            i = d.selected
            name = d.next_keypoint(i) if i is not None else None
            if name:
                d.set_keypoint(i, name, x, y)
                self.sel_kp = name
            nxt = d.next_keypoint(i) if i is not None else None
            if nxt:
                self.msg_var.set(f"click: {nxt}")
            else:
                self.set_mode("select")
                self.msg_var.set("all keypoints are placed; drag them to adjust")
        else:
            h = d.hit(x, y, HIT_PX * self.th.scale / self.view.scale)
            self.sel_kp = None
            if h is None:
                d.selected = None
                self._drag = None
            else:
                d.selected = h[1]
                if h[0] == "kp":
                    self.sel_kp = h[2]
                self._drag = ("edit", h, x, y, False)
        self.redraw()

    def _motion(self, e):
        d, g = self.doc, self._drag
        if not d or not g:
            return
        if g[0] == "pan":
            self._pan_move(e)
            return
        x, y = self.view.to_image(e.x, e.y)
        if g[0] == "newbox":
            self._drag = ("newbox", g[1], g[2], x, y)
        elif g[0] == "edit":
            _, h, px, py, started = g
            if not started:
                if abs(x - px) * self.view.scale < 2 and abs(y - py) * self.view.scale < 2:
                    return                                   # a click, not a drag
                d.checkpoint()                               # one undo step per drag
            if h[0] == "box":
                d.move_instance(h[1], x - px, y - py)
            elif h[0] == "handle":
                d.resize_box(h[1], h[2], x, y)
            elif h[0] == "kp":
                d.move_keypoint(h[1], h[2], x, y)
            elif h[0] == "vertex":
                d.move_vertex(h[1], h[2], x, y)
            self._drag = ("edit", h, x, y, True)
        self.redraw()

    def _release(self, e):
        d, g = self.doc, self._drag
        self._drag = None
        if d and g and g[0] == "newbox":
            if d.add_box(g[1], g[2], g[3], g[4]) is None:
                self.msg_var.set("box too small: drag from one corner to the opposite one")
            else:
                self.set_mode("select")
                self.msg_var.set("box added" + ("; K places its keypoints" if self.kp_names else ""))
        self.redraw()

    # ------------------------------------------------------------------ keys
    def _key(self, e):
        k = (e.keysym or "").lower()
        ctrl = bool(e.state & 0x4)
        if ctrl and k == "z":
            self.undo()
        elif k == "a":
            self.accept()
        elif k == "r":
            self.reject()
        elif k in ("b", "p", "k"):
            self.set_mode({"b": "box", "p": "poly", "k": "kp"}[k])
        elif k in ("return", "kp_enter"):
            self.close_poly()
        elif k == "escape":
            self.set_mode("select")
        elif k in ("delete", "backspace"):
            self.delete_selected()
        elif k in ("down", "next", "n"):
            self.step(1)
        elif k in ("up", "prior"):
            self.step(-1)
        elif k in ("plus", "equal", "kp_add"):
            self.zoom(1.25)
        elif k in ("minus", "kp_subtract"):
            self.zoom(0.8)
        elif k == "f":
            self.fit()
        elif k == "space":
            self._space = True
        elif k == "v" and self.doc and self.doc.selected is not None and self.sel_kp:
            self.doc.toggle_visibility(self.doc.selected, self.sel_kp)
            self.redraw()
        return "break"

    # ------------------------------------------------------------------ drawing
    def redraw(self):
        from PIL import Image, ImageTk
        c = self.canvas
        c.delete("all")
        cw, ch = c.winfo_width(), c.winfo_height()
        self._fill_kp_list()
        if not self.cur or self.image is None or cw < 4 or ch < 4:
            if not self.cur:
                c.create_text(cw // 2, ch // 2, fill="#dddddd", text="no images in this list", font=self.th.font)
            return
        z = self.th.S                                    # overlay sizes follow the UI scale
        v = self.view
        vis = v.visible(self.cur["W"], self.cur["H"], cw, ch)
        if vis:
            x0, y0, x1, y1 = vis
            w, h = max(1, int(round((x1 - x0) * v.scale))), max(1, int(round((y1 - y0) * v.scale)))
            im = self.image.crop(vis).resize((w, h), Image.NEAREST if v.scale > 3 else Image.BILINEAR)
            self._photo = ImageTk.PhotoImage(im, master=c)
            c.create_image(*v.to_canvas(x0, y0), image=self._photo, anchor="nw")
        d = self.doc
        for i, ins in enumerate(d.instances):
            col = COLORS[i % len(COLORS)]
            sel = i == d.selected
            if ins.get("poly"):
                pts = [q for p in ins["poly"] for q in v.to_canvas(*p)]
                c.create_polygon(*pts, outline=col, fill="", width=z(2))
                if sel:
                    for px, py in ins["poly"]:
                        cx, cy = v.to_canvas(px, py)
                        c.create_rectangle(cx - z(2), cy - z(2), cx + z(2), cy + z(2), outline="white", fill=col)
            bx0, by0 = v.to_canvas(ins["box"][0], ins["box"][1])
            bx1, by1 = v.to_canvas(ins["box"][2], ins["box"][3])
            c.create_rectangle(bx0, by0, bx1, by1, outline=col, width=z(3) if sel else z(1), dash=() if sel else (z(4), z(3)))
            c.create_text(bx0 + z(3), by0 - z(2), anchor="sw", fill=col, text=f"{i + 1}", font=self.th.font_bold)
            if sel:
                for hx, hy in d.handle_points(i).values():
                    cx, cy = v.to_canvas(hx, hy)
                    c.create_rectangle(cx - z(4), cy - z(4), cx + z(4), cy + z(4), outline="black", fill="white")
            kps = ins.get("keypoints") or {}
            for a, b in self.skeleton:
                if a < len(self.kp_names) and b < len(self.kp_names):
                    ka, kb = kps.get(self.kp_names[a]), kps.get(self.kp_names[b])
                    if ka and kb and ka[2] > 0 and kb[2] > 0:
                        c.create_line(*v.to_canvas(ka[0], ka[1]), *v.to_canvas(kb[0], kb[1]), fill=col, width=z(1))
            for n, k in kps.items():
                if not k or k[2] <= 0:
                    continue
                cx, cy = v.to_canvas(k[0], k[1])
                r = z(5) if (sel and n == self.sel_kp) else z(3)
                c.create_oval(cx - r, cy - r, cx + r, cy + r, outline="white", fill=col if k[2] == 2 else "")
                if sel:
                    c.create_text(cx + z(6), cy, anchor="w", fill="white", text=n, font=self.th.font_small)
        g = self._drag
        if g and g[0] == "newbox":
            c.create_rectangle(*v.to_canvas(g[1], g[2]), *v.to_canvas(g[3], g[4]), outline="yellow", dash=(z(3), z(3)), width=z(1))
        if self._poly_pts:
            pts = [q for p in self._poly_pts for q in v.to_canvas(*p)]
            if len(pts) >= 4:
                c.create_line(*pts, fill="yellow", width=z(2))
            for px, py in self._poly_pts:
                cx, cy = v.to_canvas(px, py)
                c.create_oval(cx - z(3), cy - z(3), cx + z(3), cy + z(3), outline="yellow")

    def _fill_kp_list(self):
        self.kp_list.delete(0, "end")
        d = self.doc
        if not d or d.selected is None or d.selected >= len(d.instances):
            return
        kps = d.instances[d.selected].get("keypoints") or {}
        for n in self.kp_names:
            k = kps.get(n)
            self.kp_list.insert("end", f"{'-' if not k or k[2] <= 0 else ('o' if k[2] == 2 else 'h')} {n}")


def main(argv=None) -> int:
    """`animalworld review PROJECT --gui`: the annotator as its own window."""
    import argparse
    from ..config import ConfigError, Project
    ap = argparse.ArgumentParser(prog="animalworld review --gui", description="desktop annotator (tkinter)")
    ap.add_argument("project")
    a = ap.parse_args(argv)
    try:
        pr = Project.load(a.project)
    except ConfigError as e:
        print(f"animalworld: error: {e}", file=sys.stderr)
        return 2
    try:
        root = tk.Tk()
    except tk.TclError as e:
        print(f"animalworld: no display for the annotator window ({e}). Use a desktop session or `ssh -X`; "
              "labels can also be imported from SLEAP / DeepLabCut with `animalworld import-labels`.", file=sys.stderr)
        return 2
    root.withdraw()
    w = AnnotatorWindow(root, pr)
    w.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

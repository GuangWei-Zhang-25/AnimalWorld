"""The shared look of the Animal World windows (Studio, Run, annotator): tkinter / ttk only, no extra package.

A calm, flat style: a soft neutral window, white rounded cards, hairline separators, one accent colour, the platform's
own UI font. Everything is drawn by this module: rounded shapes are small PNG images made in pure Python (zlib + struct
from the standard library) and used as 9-patch ttk "image" elements, so buttons, fields, check boxes and progress bars
stay real ttk widgets (invoke(), state, keyboard) and only their drawing changes. No artwork or font is bundled.

    th = ui.theme(root)          # one Theme per Tk root; installs the styles
    th.S(8)                      # 8 px at the current scale
    th.c("accent")               # a palette colour
    th.toggle()                  # light <-> dark, restyles every widget in place

Light or dark follows the system when it can be found out (macOS, Windows, GNOME), or ANIMALWORLD_THEME=light|dark.
The scale follows the display's pixels per inch, or ANIMALWORLD_UI_SCALE (e.g. 2). If the images cannot be made (a Tk
without PNG support) the windows still work with flat ttk styles in the same colours.
"""
from __future__ import annotations

import base64
import math
import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

PALETTES: Dict[str, Dict[str, str]] = {
    "light": {
        "bg": "#F5F5F7", "side": "#EBEBF0", "card": "#FFFFFF", "field": "#FFFFFF", "fieldline": "#C7C7CC",
        "line": "#E3E3E8", "text": "#1D1D1F", "text2": "#5E5E63", "text3": "#A1A1A6",
        "accent": "#0066CC", "accent_hover": "#005BB8", "accent_press": "#004F9F", "on_accent": "#FFFFFF", "link": "#0066CC",
        "quiet": "#E9E9ED", "quiet_hover": "#DFDFE4", "quiet_press": "#D1D1D6", "danger": "#C0271C",
        "ok": "#248A3D", "warn": "#C9700A", "idle": "#C2C2C8", "log": "#F5F5F7", "canvas": "#2A2A2C", "sel": "#D6E8FB",
    },
    "dark": {
        "bg": "#1C1C1E", "side": "#252527", "card": "#2C2C2E", "field": "#1C1C1E", "fieldline": "#545458",
        "line": "#3D3D40", "text": "#F5F5F7", "text2": "#B4B4BA", "text3": "#77777C",
        "accent": "#0A66CC", "accent_hover": "#1A75DB", "accent_press": "#0858B0", "on_accent": "#FFFFFF", "link": "#6CB4FF",
        "quiet": "#3F3F42", "quiet_hover": "#4A4A4E", "quiet_press": "#57575B", "danger": "#FF7A72",
        "ok": "#30D158", "warn": "#FF9F0A", "idle": "#636366", "log": "#1C1C1E", "canvas": "#141415", "sel": "#16406E",
    },
}
# every pair of (text colour, surface it is written on) used by the windows; tests keep these at >= 4.5 : 1
TEXT_PAIRS = (("text", "bg"), ("text", "side"), ("text", "card"), ("text", "field"), ("text", "quiet"), ("text", "quiet_hover"),
              ("text", "log"), ("text2", "bg"), ("text2", "side"), ("text2", "card"), ("on_accent", "accent"),
              ("on_accent", "accent_hover"), ("on_accent", "accent_press"), ("link", "card"), ("link", "bg"), ("link", "side"),
              ("danger", "card"), ("danger", "bg"), ("text", "sel"))


def _rgb(h: str) -> Tuple[int, int, int]:
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


def contrast(a: str, b: str) -> float:
    """WCAG 2 contrast ratio of two #rrggbb colours."""
    def lum(h):
        v = []
        for x in _rgb(h):
            x /= 255.0
            v.append(x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4)
        return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]
    la, lb = sorted((lum(a), lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def system_mode() -> str:
    """'light' or 'dark': ANIMALWORLD_THEME, else what the desktop says, else light. Never raises, never waits long."""
    env = os.environ.get("ANIMALWORLD_THEME", "").strip().lower()
    if env in ("light", "dark"):
        return env
    try:
        if sys.platform == "darwin":
            r = subprocess.run(["defaults", "read", "-g", "AppleInterfaceStyle"], capture_output=True, text=True, timeout=2)
            return "dark" if "dark" in r.stdout.lower() else "light"
        if sys.platform.startswith("win"):
            import winreg                                                           # type: ignore
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as k:
                return "light" if winreg.QueryValueEx(k, "AppsUseLightTheme")[0] else "dark"
        if "dark" in os.environ.get("GTK_THEME", "").lower():
            return "dark"
        r = subprocess.run(["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"], capture_output=True, text=True, timeout=2)
        return "dark" if "dark" in r.stdout.lower() else "light"
    except Exception:
        return "light"


def short_path(p) -> str:
    """A path as it is SHOWN: the home folder written as ~ (the value itself is never changed)."""
    s = str(p)
    try:
        home = str(Path.home())
    except Exception:
        return s
    if home and home != "/" and (s == home or s.startswith(home.rstrip("/\\") + os.sep)):
        return "~" + s[len(home.rstrip("/\\")):]
    return s


# ------------------------------------------------------------------ tiny rasteriser (signed distances, RGBA, PNG) ----
class _Pix:
    def __init__(self, w: int, h: int):
        self.w, self.h = int(w), int(h)
        self.p = [[0.0, 0.0, 0.0, 0.0] for _ in range(self.w * self.h)]            # premultiplied r g b, alpha

    def fill(self, sd: Callable[[float, float], float], colour: str, alpha: float = 1.0) -> "_Pix":
        r, g, b = _rgb(colour)
        for y in range(self.h):
            for x in range(self.w):
                a = min(1.0, max(0.0, 0.5 - sd(x + 0.5, y + 0.5))) * alpha
                if a <= 0:
                    continue
                q = self.p[y * self.w + x]
                k = 1.0 - a
                q[0], q[1], q[2], q[3] = r * a + q[0] * k, g * a + q[1] * k, b * a + q[2] * k, a + q[3] * k
        return self

    def png(self) -> bytes:
        rows = bytearray()
        for y in range(self.h):
            rows.append(0)
            for x in range(self.w):
                r, g, b, a = self.p[y * self.w + x]
                if a > 0:
                    rows += bytes((min(255, int(r / a + 0.5)), min(255, int(g / a + 0.5)), min(255, int(b / a + 0.5)), min(255, int(a * 255 + 0.5))))
                else:
                    rows += b"\0\0\0\0"

        def chunk(tag, data):
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", self.w, self.h, 8, 6, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(bytes(rows), 9)) + chunk(b"IEND", b""))


def sd_rrect(w: float, h: float, r: float, inset: float = 0.0):
    hw, hh, r = w / 2 - inset, h / 2 - inset, max(0.0, r - inset)

    def f(x, y):
        qx, qy = abs(x - w / 2) - (hw - r), abs(y - h / 2) - (hh - r)
        return math.hypot(max(qx, 0.0), max(qy, 0.0)) + min(max(qx, qy), 0.0) - r
    return f


def sd_circle(cx: float, cy: float, r: float):
    return lambda x, y: math.hypot(x - cx, y - cy) - r


def sd_line(pts, width: float):
    def f(x, y):
        best = 1e9
        for (ax, ay), (bx, by) in zip(pts, pts[1:]):
            dx, dy = bx - ax, by - ay
            t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy or 1.0)))
            best = min(best, math.hypot(x - ax - t * dx, y - ay - t * dy))
        return best - width / 2
    return f


def rounded(w, h, r, fill, border=None, bw=1.0) -> bytes:
    px = _Pix(w, h)
    if border:
        px.fill(sd_rrect(w, h, r), border)
        px.fill(sd_rrect(w, h, r, bw), fill)
    else:
        px.fill(sd_rrect(w, h, r), fill)
    return px.png()


# ------------------------------------------------------------------ the theme ----------------------------------------
class Theme:
    def __init__(self, root, mode: str = None, scale: float = None):
        import tkinter as tk
        import tkinter.font as tkfont
        from tkinter import ttk
        self.tk, self.ttk, self.root = tk, ttk, root
        self.mode = mode if mode in PALETTES else system_mode()
        self.scale = float(scale) if scale else self._scale()
        self.images_ok = True
        self._img: Dict[str, object] = {}
        self._painters: Dict[str, Callable[[], bytes]] = {}
        self._reg: List[tuple] = []
        self._listeners: List[Callable[[], None]] = []
        fam = self._family(tkfont)
        mono = next((f for f in ("SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono", "Liberation Mono") if f in set(tkfont.families(root))), "TkFixedFont")
        mk = lambda px, weight="normal", family=fam: tkfont.Font(root=root, family=family, size=-self.S(px), weight=weight)   # noqa: E731
        self.font = mk(13)
        self.font_bold = mk(13, "bold")
        self.font_title = mk(22, "bold")
        self.font_section = mk(15, "bold")
        self.font_caption = mk(12)
        self.font_small = mk(11)
        self.font_mono = tkfont.Font(root=root, family=mono, size=-self.S(11)) if mono != "TkFixedFont" else tkfont.nametofont("TkFixedFont")
        self.family = fam
        self.style = ttk.Style(root)
        try:
            self.style.theme_use("clam")                 # the one built-in theme whose elements can all be replaced
        except tk.TclError:
            pass
        self._make_elements()
        self.apply()

    # ---- basics ----
    def _scale(self) -> float:
        env = os.environ.get("ANIMALWORLD_UI_SCALE", "").strip()
        try:
            if env:
                return min(4.0, max(0.75, float(env)))
            s = float(self.root.winfo_fpixels("1i")) / 96.0
            if sys.platform == "darwin":
                s = 1.0                                   # Tk on macOS works in points; the system scales
            return min(3.0, max(1.0, round(s * 4) / 4))
        except Exception:
            return 1.0

    def _family(self, tkfont) -> str:
        fams = set(tkfont.families(self.root))
        if sys.platform == "darwin" or sys.platform.startswith("win"):
            sysfam = tkfont.nametofont("TkDefaultFont").actual("family")
            return sysfam
        for f in ("Inter", "Nimbus Sans", "Nimbus Sans L", "Helvetica Neue", "DejaVu Sans", "Liberation Sans", "Noto Sans"):
            if f in fams:
                return f
        return tkfont.nametofont("TkDefaultFont").actual("family")

    def S(self, n: float) -> int:
        return int(round(n * self.scale))

    def c(self, name: str) -> str:
        return PALETTES[self.mode][name]

    def geometry(self, spec: str) -> str:
        """'1200x800+0+0' scaled by the UI scale (position kept)."""
        if self.scale == 1.0 or "x" not in spec:
            return spec
        size, _, pos = spec.partition("+")
        w, h = size.split("x")
        return f"{self.S(int(w))}x{self.S(int(h))}" + ("+" + pos if pos else "")

    def reg(self, widget, **opts):
        """A plain tk widget whose colours follow the theme: reg(w, background="card", foreground="text")."""
        self._reg.append((widget, opts))
        self._paint(widget, opts)
        return widget

    def on_change(self, fn: Callable[[], None]) -> None:
        self._listeners.append(fn)

    def _paint(self, widget, opts):
        try:
            widget.configure(**{k: self.c(v) for k, v in opts.items()})
        except Exception:
            pass

    def toggle(self) -> str:
        self.mode = "dark" if self.mode == "light" else "light"
        self.apply()
        return self.mode

    # ---- images ----
    def image(self, name: str, painter: Callable[[], bytes] = None):
        """A named PhotoImage that is redrawn when the palette changes."""
        if name in self._img:
            return self._img[name]
        self._painters[name] = painter
        try:
            im = self.tk.PhotoImage(master=self.root, data=base64.b64encode(painter()))
        except Exception:
            self.images_ok = False
            im = self.tk.PhotoImage(master=self.root, width=1, height=1)
        self._img[name] = im
        return im

    def dot(self, kind: str):
        """Status dot: idle | ok | run | fail | accent."""
        col = {"idle": "idle", "ok": "ok", "run": "warn", "fail": "danger", "accent": "accent"}[kind]
        d = self.S(9)
        return self.image("dot_" + kind, lambda: _Pix(d + self.S(4), d).fill(sd_circle(d / 2, d / 2, d / 2 - 0.5), self.c(col)).png())

    def chevron(self, direction: str, colour: str = "text2"):
        s = self.S(12)
        a, b, m = 0.28 * s, 0.72 * s, 0.5 * s
        pts = {"down": [(a, 0.38 * s), (m, 0.64 * s), (b, 0.38 * s)], "up": [(a, 0.62 * s), (m, 0.36 * s), (b, 0.62 * s)],
               "right": [(0.38 * s, a), (0.64 * s, m), (0.38 * s, b)]}[direction]
        return self.image(f"chev_{direction}_{colour}", lambda: _Pix(s, s).fill(sd_line(pts, max(1.4, 1.5 * self.scale)), self.c(colour)).png())

    def _make_elements(self):
        S, c, im, st = self.S, self.c, self.image, self.style
        r = S(7)
        n = 2 * r + S(4)

        def rr(name, fill, border=None, bw=None, radius=None, size=None):
            return im(name, lambda: rounded(size or n, size or n, radius or r, c(fill), c(border) if border else None, bw or max(1.0, self.scale)))
        try:
            # buttons
            st.element_create("AW.quiet", "image", rr("b_q", "quiet"), ("disabled", rr("b_qd", "quiet")), ("pressed", rr("b_qp", "quiet_press")),
                              ("active", rr("b_qh", "quiet_hover")), ("focus", rr("b_qf", "quiet", "accent", max(1.5, 1.5 * self.scale))),
                              border=r + 1, padding=0, sticky="nsew")
            st.element_create("AW.primary", "image", rr("b_p", "accent"), ("disabled", rr("b_pd", "quiet")), ("pressed", rr("b_pp", "accent_press")),
                              ("active", rr("b_ph", "accent_hover")), border=r + 1, padding=0, sticky="nsew")
            st.element_create("AW.seg", "image", rr("b_s", "quiet"), ("selected", rr("b_ss", "accent")), ("active", rr("b_sh", "quiet_hover")),
                              border=r + 1, padding=0, sticky="nsew")
            st.element_create("AW.nav", "image", im("nav_0", lambda: _Pix(n, n).png()), ("selected", rr("nav_s", "accent")),
                              ("active", rr("nav_h", "quiet_hover")), border=r + 1, padding=0, sticky="nsew")
            # fields
            st.element_create("AW.field", "image", rr("f", "field", "fieldline"), ("disabled", rr("f_d", "card", "line")),
                              ("focus", rr("f_f", "field", "accent", max(1.5, 1.5 * self.scale))), border=r + 1, padding=0, sticky="nsew")
            # cards
            cr = S(12)
            cn = 2 * cr + S(4)
            st.element_create("AW.card", "image", im("card", lambda: rounded(cn, cn, cr, c("card"), c("line"), max(1.0, self.scale))),
                              border=cr + 1, padding=0, sticky="nsew")
            st.element_create("AW.well", "image", im("well", lambda: rounded(n, n, r, c("log"), c("line"), max(1.0, self.scale))),
                              border=r + 1, padding=0, sticky="nsew")
            # check box and radio
            k = S(18)
            kr = S(5)

            def check(on, dis=False):
                def paint():
                    px = _Pix(k + S(8), k)
                    if on:
                        px.fill(sd_rrect(k, k, kr), c("quiet_press" if dis else "accent"))
                        px.fill(sd_line([(0.27 * k, 0.52 * k), (0.44 * k, 0.69 * k), (0.74 * k, 0.33 * k)], max(1.6, 1.8 * self.scale)), c("on_accent"))
                    else:
                        px.fill(sd_rrect(k, k, kr), c("line" if dis else "fieldline"))
                        px.fill(sd_rrect(k, k, kr, max(1.0, 1.2 * self.scale)), c("card" if dis else "field"))
                    return px.png()
                return paint

            def radio(on):
                def paint():
                    px = _Pix(k + S(8), k)
                    if on:
                        px.fill(sd_circle(k / 2, k / 2, k / 2 - 0.5), c("accent"))
                        px.fill(sd_circle(k / 2, k / 2, k * 0.2), c("on_accent"))
                    else:
                        px.fill(sd_circle(k / 2, k / 2, k / 2 - 0.5), c("fieldline"))
                        px.fill(sd_circle(k / 2, k / 2, k / 2 - 0.5 - max(1.0, 1.2 * self.scale)), c("field"))
                    return px.png()
                return paint
            st.element_create("AW.check", "image", im("ck0", check(False)), ("disabled", "selected", im("ck1d", check(True, True))),
                              ("disabled", im("ck0d", check(False, True))), ("selected", im("ck1", check(True))), sticky="w")
            st.element_create("AW.radio", "image", im("rd0", radio(False)), ("selected", im("rd1", radio(True))), sticky="w")
            # progress bar
            ph = S(6)
            pn = ph * 2 + S(2)
            st.element_create("AW.trough", "image", im("pt", lambda: rounded(pn, ph, ph / 2, c("quiet"))), border=(ph // 2 + 1, 0), padding=0, sticky="ew")
            st.element_create("AW.pbar", "image", im("pb", lambda: rounded(pn, ph, ph / 2, c("accent"))), border=(ph // 2 + 1, 0), padding=0, sticky="ew")
            # arrows of combo boxes and spin boxes (the names must end in ...arrow: ttk's own bindings look for that)
            st.element_create("AW.downarrow", "image", self.chevron("down"), sticky="")
            st.element_create("AW.uparrow", "image", self.chevron("up"), sticky="")
        except self.tk.TclError:
            self.images_ok = False
        self.dot("idle"), self.dot("ok"), self.dot("run"), self.dot("fail")
        self.chevron("right"), self.chevron("down")

    # ---- styles ----
    def apply(self):
        S, c, st = self.S, self.c, self.style
        if self.images_ok:
            for name, painter in self._painters.items():
                try:
                    self._img[name].configure(data=base64.b64encode(painter()))
                except Exception:
                    pass
        ok = self.images_ok
        st.configure(".", background=c("bg"), foreground=c("text"), font=self.font, bordercolor=c("line"), lightcolor=c("line"),
                     darkcolor=c("line"), troughcolor=c("quiet"), focuscolor=c("accent"), selectbackground=c("accent"),
                     selectforeground=c("on_accent"), fieldbackground=c("field"), insertcolor=c("text"), borderwidth=0, relief="flat")
        st.map(".", foreground=[("disabled", c("text3"))], background=[], fieldbackground=[])
        surfaces = {"": "bg", "Card.": "card", "Side.": "side", "Sel.": "sel"}
        for pre, surf in surfaces.items():
            st.configure(pre + "TFrame", background=c(surf))
            st.configure(pre + "TLabel", background=c(surf), foreground=c("text"), font=self.font)
            for role, fg, font in (("Title", "text", self.font_title), ("Section", "text", self.font_section), ("Caption", "text2", self.font_caption),
                                   ("Small", "text2", self.font_small), ("Error", "danger", self.font), ("Bold", "text", self.font_bold),
                                   ("Link", "link", self.font), ("Mono", "text2", self.font_mono)):
                st.configure(f"{role}.{pre}TLabel", background=c(surf), foreground=c(fg), font=font)
            # buttons: quiet (default), primary, link
            for kind, el, fg in (("", "AW.quiet", "text"), ("Primary.", "AW.primary", "on_accent")):
                name = f"{kind}{pre}TButton"
                if ok:
                    st.layout(name, [(el, {"sticky": "nsew", "children": [("Button.padding", {"sticky": "nsew", "children": [("Button.label", {"sticky": "nsew"})]})]})])
                st.configure(name, background=c(surf) if ok else c("accent" if kind else "quiet"), foreground=c(fg), padding=(S(14), S(6)),
                             anchor="center", font=self.font_bold if kind else self.font, borderwidth=0, relief="flat", focusthickness=0)
                st.map(name, foreground=[("disabled", c("text3"))],
                       background=[] if ok else [("disabled", c("quiet")), ("pressed", c("accent_press" if kind else "quiet_press")),
                                                 ("active", c("accent_hover" if kind else "quiet_hover"))])
            name = f"Link.{pre}TButton"
            st.layout(name, [("Button.padding", {"sticky": "nsew", "children": [("Button.label", {"sticky": "nsew"})]})])
            st.configure(name, background=c(surf), foreground=c("link"), padding=(S(4), S(4)), anchor="w", font=self.font, borderwidth=0)
            st.map(name, foreground=[("disabled", c("text3")), ("active", c("text"))], background=[("active", c(surf))])
            for w_, el in (("TCheckbutton", "AW.check"), ("TRadiobutton", "AW.radio")):
                name = pre + w_
                if ok:
                    st.layout(name, [(w_[1:] + ".padding", {"sticky": "nsew", "children": [(el, {"side": "left", "sticky": ""}),
                                                                                           (w_[1:] + ".label", {"side": "left", "sticky": "nsew"})]})])
                st.configure(name, background=c(surf), foreground=c("text"), padding=(0, S(3)), font=self.font, indicatorcolor=c("field"))
                st.map(name, background=[("active", c(surf))], foreground=[("disabled", c("text3"))],
                       indicatorcolor=[("selected", c("accent"))])
        # segmented choice and sidebar navigation (radio buttons without an indicator)
        if ok:
            st.layout("Seg.TRadiobutton", [("AW.seg", {"sticky": "nsew", "children": [("Radiobutton.padding", {"sticky": "nsew", "children": [("Radiobutton.label", {"sticky": "nsew"})]})]})])
            st.layout("Nav.TRadiobutton", [("AW.nav", {"sticky": "nsew", "children": [("Radiobutton.padding", {"sticky": "nsew", "children": [("Radiobutton.label", {"sticky": "nsew"})]})]})])
        else:
            for nm in ("Seg.TRadiobutton", "Nav.TRadiobutton"):
                st.layout(nm, [("Radiobutton.padding", {"sticky": "nsew", "children": [("Radiobutton.label", {"sticky": "nsew"})]})])
        st.configure("Seg.TRadiobutton", background=c("card") if ok else c("quiet"), foreground=c("text"), padding=(S(12), S(5)), anchor="center", font=self.font)
        st.map("Seg.TRadiobutton", foreground=[("selected", c("on_accent"))], background=[("active", c("card"))] if ok else [("selected", c("accent"))])
        st.configure("Nav.TRadiobutton", background=c("side"), foreground=c("text"), padding=(S(10), S(7)), anchor="w", font=self.font)
        st.map("Nav.TRadiobutton", foreground=[("selected", c("on_accent"))], background=[("active", c("side"))] if ok else [("selected", c("accent"))])
        # fields
        fl = [("AW.field", {"sticky": "nsew", "children": [("Entry.padding", {"sticky": "nsew", "children": [("Entry.textarea", {"sticky": "nsew"})]})]})]
        cl = [("AW.field", {"sticky": "nsew", "children": [("AW.downarrow", {"side": "right", "sticky": ""}),
                                                         ("Combobox.padding", {"sticky": "nsew", "children": [("Combobox.textarea", {"sticky": "nsew"})]})]})]
        sl = [("AW.field", {"sticky": "nsew", "children": [("null", {"side": "right", "sticky": "", "children": [
            ("AW.uparrow", {"side": "top", "sticky": "e"}), ("AW.downarrow", {"side": "bottom", "sticky": "e"})]}),
            ("Spinbox.padding", {"sticky": "nsew", "children": [("Spinbox.textarea", {"sticky": "nsew"})]})]})]
        for name, lay, padr in (("TEntry", fl, 8), ("TCombobox", cl, 6), ("TSpinbox", sl, 6)):
            if ok:
                st.layout(name, lay)
            st.configure(name, background=c("card"), foreground=c("text"), fieldbackground=c("field"), insertcolor=c("text"),
                         padding=(S(9), S(6), S(padr), S(6)), selectbackground=c("sel"), selectforeground=c("text"),
                         bordercolor=c("fieldline"), lightcolor=c("field"), darkcolor=c("field"), arrowcolor=c("text2"), borderwidth=0 if ok else 1)
            st.map(name, foreground=[("disabled", c("text3"))], fieldbackground=[("readonly", c("field")), ("disabled", c("card"))],
                   selectbackground=[("readonly", "!focus", c("field")), ("readonly", c("field"))], selectforeground=[("readonly", c("text"))],
                   background=[("readonly", c("card")), ("active", c("card"))], bordercolor=[("focus", c("accent"))])
        try:
            for opt, val in (("background", c("field")), ("foreground", c("text")), ("selectBackground", c("accent")),
                             ("selectForeground", c("on_accent")), ("font", self.font), ("borderWidth", 0), ("relief", "flat")):
                self.root.option_add("*TCombobox*Listbox." + opt, val)
        except Exception:
            pass
        try:
            for cls in ("TEntry", "TCombobox", "TSpinbox"):
                self.root.option_add(f"*{cls}.font", self.font)
        except Exception:
            pass
        # containers
        if ok:
            st.layout("Card.TFrame", [("AW.card", {"sticky": "nsew"})])
            st.layout("Well.TFrame", [("AW.well", {"sticky": "nsew"})])
        st.configure("Card.TFrame", background=c("bg"))                     # what shows behind the rounded corners
        st.configure("OnSide.Card.TFrame", background=c("side"))            # a card that sits on the sidebar colour
        st.configure("In.Card.TFrame", background=c("card"))                # a plain frame inside a card
        st.layout("In.Card.TFrame", [("Frame.border", {"sticky": "nsew"})])
        st.configure("Well.TFrame", background=c("card") if ok else c("log"))
        st.configure("Line.TFrame", background=c("line"))
        # progress bar, scroll bars
        if ok:
            st.layout("Horizontal.TProgressbar", [("AW.trough", {"sticky": "ew", "children": [("AW.pbar", {"side": "left", "sticky": "ew"})]})])
        for pre, surf in surfaces.items():
            st.configure(pre + "Horizontal.TProgressbar", background=c(surf) if ok else c("accent"), troughcolor=c("quiet"), thickness=S(6), borderwidth=0)
            if ok and pre:
                st.layout(pre + "Horizontal.TProgressbar", [("AW.trough", {"sticky": "ew", "children": [("AW.pbar", {"side": "left", "sticky": "ew"})]})])
        for orient in ("Vertical", "Horizontal"):
            name = orient + ".TScrollbar"
            st.layout(name, [(f"{orient}.Scrollbar.trough", {"sticky": "nsew", "children": [(f"{orient}.Scrollbar.thumb", {"expand": "1", "sticky": "nsew"})]})])
            st.configure(name, troughcolor=c("bg"), background=c("idle"), bordercolor=c("bg"), lightcolor=c("idle"), darkcolor=c("idle"),
                         gripcount=0, arrowsize=S(8), width=S(8), borderwidth=0, relief="flat")
            st.map(name, background=[("active", c("text3"))], lightcolor=[("active", c("text3"))], darkcolor=[("active", c("text3"))])
            st.configure("Card." + name, troughcolor=c("card"), bordercolor=c("card"))
            st.layout("Card." + name, [(f"{orient}.Scrollbar.trough", {"sticky": "nsew", "children": [(f"{orient}.Scrollbar.thumb", {"expand": "1", "sticky": "nsew"})]})])
        st.configure("TSeparator", background=c("line"))
        try:
            self.root.configure(background=c("bg"))
        except Exception:
            pass
        alive = []
        for w, opts in self._reg:
            try:
                if w.winfo_exists():
                    self._paint(w, opts)
                    alive.append((w, opts))
            except Exception:
                pass
        self._reg = alive
        for fn in list(self._listeners):
            try:
                fn()
            except Exception:
                pass

    # ---- small building blocks ----
    def card(self, parent, pad: int = 16, **pack):
        """A white rounded card; returns the inner frame to put things in (the card itself is .master)."""
        outer = self.ttk.Frame(parent, style="Card.TFrame", padding=self.S(pad))
        if pack:
            outer.pack(**pack)
        inner = self.ttk.Frame(outer, style="In.Card.TFrame")
        inner.pack(fill="both", expand=True)
        inner.card = outer
        return inner

    def hairline(self, parent, **pack):
        f = self.tk.Frame(parent, height=max(1, self.S(1)), borderwidth=0, highlightthickness=0)
        self.reg(f, background="line")
        f.pack(**({"fill": "x"} if not pack else pack))
        return f

    def text(self, parent, surface: str = "log", **kw):
        """A flat tk.Text in the theme's colours (logs)."""
        t = self.tk.Text(parent, relief="flat", borderwidth=0, highlightthickness=0, font=self.font_mono, padx=self.S(8), pady=self.S(6), **kw)
        self.reg(t, background=surface, foreground="text2", insertbackground="text", selectbackground="sel", selectforeground="text")
        return t

    def listbox(self, parent, surface: str = "card", **kw):
        lb = self.tk.Listbox(parent, relief="flat", borderwidth=0, highlightthickness=0, activestyle="none", **kw)
        self.reg(lb, background=surface, foreground="text", selectbackground="accent", selectforeground="on_accent")
        return lb

    def wrap(self, label, margin: int = 0):
        """Make a label re-wrap its text to its own width."""
        def fit(e, lab=label):
            w = max(self.S(120), e.width - self.S(margin))
            if int(float(lab.cget("wraplength") or 0)) != w:
                lab.configure(wraplength=w)
        label.bind("<Configure>", fit, add="+")
        return label

    def autohide(self, scrollbar, **pack):
        """A set-command for a scrollbar that is only shown while there is something to scroll."""
        def setter(lo, hi, sb=scrollbar):
            try:
                if float(lo) <= 0.0 and float(hi) >= 1.0:
                    if sb.winfo_manager():
                        sb.pack_forget()
                elif not sb.winfo_manager():
                    sb.pack(**pack)
                sb.set(lo, hi)
            except Exception:
                pass
        return setter

    def mode_button(self, parent, style: str = "Link.TButton"):
        """The manual light / dark switch."""
        b = self.ttk.Button(parent, style=style, command=self.toggle)

        def name():
            try:
                b.configure(text="Dark appearance" if self.mode == "light" else "Light appearance")
            except Exception:
                pass
        self.on_change(name)
        name()
        return b


class Disclosure:
    """One line with a chevron that shows / hides a body frame."""

    def __init__(self, th: Theme, parent, text: str, surface: str = "Card.", shown: bool = False, on_toggle=None):
        self.th, self.shown, self.on_toggle = th, False, on_toggle
        self.button = th.ttk.Button(parent, text=" " + text, style=f"Link.{surface}TButton", image=th.chevron("right", "link"),
                                    compound="left", command=self.toggle)
        self.button.pack(anchor="w")
        self.body = th.ttk.Frame(parent, style="In.Card.TFrame" if surface == "Card." else surface + "TFrame")
        if shown:
            self.toggle(True)

    def toggle(self, show: bool = None):
        show = (not self.shown) if show is None else show
        if show and not self.shown:
            self.body.pack(fill="both", expand=True, pady=(self.th.S(8), 0), after=self.button)
        elif not show and self.shown:
            self.body.pack_forget()
        self.shown = show
        self.button.configure(image=self.th.chevron("down" if show else "right", "link"))
        if self.on_toggle:
            self.on_toggle(show)


def theme(root, mode: str = None, scale: float = None) -> Theme:
    """The Theme of this Tk root (made on first use)."""
    top = root.winfo_toplevel() if not hasattr(root, "_aw_theme") else root
    r = top
    while getattr(r, "master", None) is not None:
        r = r.master
    if getattr(r, "_aw_theme", None) is None:
        r._aw_theme = Theme(r, mode, scale)
    return r._aw_theme

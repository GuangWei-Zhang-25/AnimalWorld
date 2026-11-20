"""Helpers for the tkinter tests: a root window (skip without a display), an event pump, screenshots."""
import os
import time

import pytest


def make_root():
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("no DISPLAY: tkinter window tests need a display (e.g. Xvfb)")
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"tkinter cannot open a display: {e}")
    return root


def pump(root, cond=None, timeout=60.0, settle=0.15):
    """Run the event loop by hand (never mainloop) until cond() is true, then a little longer."""
    t0 = time.time()
    while cond is not None and not cond():
        root.update()
        assert time.time() - t0 < timeout, "timed out waiting for the window"
        time.sleep(0.02)
    end = time.time() + settle
    while time.time() < end:
        root.update()
        time.sleep(0.01)


def shot(widget, name):
    """Screenshot of one toplevel into $STUDIO_SHOTS/<name>.png (only when that folder is given and a grab is possible)."""
    out = os.environ.get("STUDIO_SHOTS")
    if not out:
        return None
    from PIL import ImageGrab
    widget.lift()
    widget.update()
    time.sleep(0.2)
    widget.update()
    x, y, w, h = widget.winfo_rootx(), widget.winfo_rooty(), widget.winfo_width(), widget.winfo_height()
    try:
        im = ImageGrab.grab(bbox=(x, y, x + w, y + h), xdisplay=os.environ.get("DISPLAY"))
    except Exception as e:                                           # no grab support on this platform
        print("screenshot not possible:", e)
        return None
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name + ".png")
    im.save(p)
    return p


def box_grab_point(doc, i, tol):
    """An image point inside instance i's box where a press grabs the BOX (not a keypoint, handle or outline point)."""
    x0, y0, x1, y1 = doc.instances[i]["box"]
    for fy in (0.5, 0.35, 0.65, 0.25, 0.75, 0.15, 0.85):
        for fx in (0.5, 0.35, 0.65, 0.25, 0.75, 0.15, 0.85):
            x, y = x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)
            if doc.hit(x, y, tol) == ("box", i):
                return x, y
    raise AssertionError("no free point inside the box")

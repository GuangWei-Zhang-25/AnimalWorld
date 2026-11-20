"""The shared look (animalworld.ui): readable colours, images made without any extra package, live light / dark switch."""
import struct
import zlib

from animalworld import ui

from tkutil import make_root, pump


def test_text_contrast_is_at_least_4_5_in_both_palettes():
    assert set(ui.PALETTES["light"]) == set(ui.PALETTES["dark"])
    for mode, pal in ui.PALETTES.items():
        for fg, bg in ui.TEXT_PAIRS:
            assert ui.contrast(pal[fg], pal[bg]) >= 4.5, (mode, fg, bg, round(ui.contrast(pal[fg], pal[bg]), 2))
    assert round(ui.contrast("#000000", "#FFFFFF"), 1) == 21.0


def test_rounded_image_is_a_valid_png_with_transparent_corners():
    data = ui.rounded(16, 16, 6, "#0066CC")
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", data[16:24]) == (16, 16)
    raw = zlib.decompress(data[data.index(b"IDAT") + 4:data.index(b"IEND") - 8])
    row = 1 + 16 * 4
    assert raw[1:5][3] == 0 and tuple(raw[8 * row + 1 + 8 * 4:8 * row + 1 + 8 * 4 + 4]) == (0, 0x66, 0xCC, 255)


def test_short_path(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert ui.short_path(tmp_path / "videos" / "a.mp4") in ("~/videos/a.mp4", "~\\videos\\a.mp4")
    assert ui.short_path("/somewhere/else") == "/somewhere/else"


def test_theme_styles_real_widgets_and_switches_in_place(monkeypatch):
    monkeypatch.setenv("ANIMALWORLD_THEME", "light")
    root = make_root()
    try:
        from tkinter import ttk
        th = ui.theme(root, scale=1.0)
        assert ui.theme(root) is th and th.mode == "light" and th.S(8) == 8
        b = ttk.Button(root, text="Start", style="Primary.TButton"); b.pack()
        e = ttk.Entry(root); e.pack()
        t = th.text(root, height=2); t.pack()
        pump(root)
        assert th.images_ok, "this Tk cannot read the PNG images; the flat fallback is in use"
        assert str(t.cget("background")).upper() == ui.PALETTES["light"]["log"]
        hit = []
        b.configure(command=lambda: hit.append(1)); b.invoke()
        assert hit == [1]                                                # still a real ttk button
        assert th.toggle() == "dark"; pump(root)
        assert str(t.cget("background")).upper() == ui.PALETTES["dark"]["log"]
        assert str(th.style.lookup("TLabel", "foreground")).upper() == ui.PALETTES["dark"]["text"]
        assert th.geometry("100x50+3+4") == "100x50+3+4" and ui.Theme.geometry(type("T", (), {"scale": 2.0, "S": lambda s, n: int(n * 2)})(), "100x50+3+4") == "200x100+3+4"
    finally:
        root.destroy()


def test_theme_falls_back_to_flat_styles_when_images_cannot_be_made(monkeypatch):
    root = make_root()
    try:
        import tkinter as tk
        real = tk.PhotoImage

        def no_png(*a, **k):
            if "data" in k:
                raise tk.TclError("couldn't recognize image data")
            return real(*a, **k)
        monkeypatch.setattr(tk, "PhotoImage", no_png)
        th = ui.Theme(root, "light", 1.0)
        assert th.images_ok is False
        from tkinter import ttk
        b = ttk.Button(root, text="ok", style="Primary.TButton"); b.pack(); pump(root)
        assert str(th.style.lookup("Primary.TButton", "background")).upper() == ui.PALETTES["light"]["accent"]
    finally:
        root.destroy()

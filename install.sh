#!/usr/bin/env bash
# Animal World: one-command install for Linux and macOS.
#
#   bash install.sh            create ~/animalworld-env, install Animal World with the run extra (CPU engine, video reading)
#   bash install.sh --gpu      the same + the full engine for an NVIDIA GPU (PyTorch, RF-DETR, Ultralytics, SAM 2.1)
#   bash install.sh --gpu --sam3   also the SAM 3.1 code, for the optional --sam3 witness (its weights are gated)
#
# Environment variables (all optional):
#   ANIMALWORLD_ENV          folder of the virtual environment   (default: ~/animalworld-env)
#   ANIMALWORLD_PACKAGE      what pip installs                    (default: the wheel in dist/ next to this script, else this
#                                                                  source folder, else animalworld@git+https://github.com/GuangWei-Zhang-25/AnimalWorld.git
#                                                                  (not on PyPI yet; once it is, "animalworld" also works))
#   ANIMALWORLD_SAM3_SOURCE  where the SAM 3 code comes from      (default: git+https://github.com/facebookresearch/sam3.git)
#   ANIMALWORLD_TORCH_INDEX  PyTorch wheel index for --gpu        (default: https://download.pytorch.org/whl/cu126)
#   PYTHON                   the Python to use                    (default: python3; --gpu needs Python 3.10 or newer)
#
# Nothing is uploaded. Model weights are not installed by this script (see INSTALL.md, "Model files").
set -euo pipefail

GPU=0; SAM3=0
for a in "$@"; do
  case "$a" in
    --gpu) GPU=1 ;;
    --sam3) SAM3=1 ;;
    -h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $a (try --help)" >&2; exit 2 ;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || echo "")"
ENV_DIR="${ANIMALWORLD_ENV:-$HOME/animalworld-env}"
PY="${PYTHON:-python3}"
if [ -n "${ANIMALWORLD_PACKAGE:-}" ]; then PKG="$ANIMALWORLD_PACKAGE"
elif [ -n "$HERE" ] && ls "$HERE"/dist/animalworld-*.whl >/dev/null 2>&1; then PKG="$(ls -t "$HERE"/dist/animalworld-*.whl | head -1)"
elif [ -n "$HERE" ] && [ -f "$HERE/pyproject.toml" ] && [ -d "$HERE/src/animalworld" ]; then PKG="$HERE"
else PKG="animalworld@git+https://github.com/GuangWei-Zhang-25/AnimalWorld.git"; fi    # not on PyPI yet
# pip requirement with an extra: "name[extra]" for a name, path or wheel; "name[extra] @ URL" for name@URL
req() { case "$PKG" in *@*://*) echo "${PKG%%@*}[$1] @ ${PKG#*@}" ;; *) echo "${PKG}[$1]" ;; esac; }

command -v "$PY" >/dev/null 2>&1 || { echo "Python was not found. Install Python 3.9 or newer and run this again." >&2; exit 1; }
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' || { echo "Animal World needs Python 3.9 or newer; $PY is $("$PY" -V 2>&1)." >&2; exit 1; }
if [ "$GPU" = 1 ] && ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  # rfdetr>=1.12 (MouseWorld-DET) is published for Python 3.10 and newer only; the CPU engine still works on 3.9.
  echo "The full engine (--gpu) needs Python 3.10 or newer; $PY is $("$PY" -V 2>&1)." >&2
  echo "Run again with a newer Python, e.g.  PYTHON=python3.11 bash install.sh --gpu  (on a cluster: module load python/3.11 first)," >&2
  echo "or install the CPU engine with this Python:  bash install.sh" >&2
  exit 1
fi
if ! "$PY" -c 'import tkinter' >/dev/null 2>&1; then
  echo "NOTE: this Python has no tkinter, which the two windows need. Debian/Ubuntu: sudo apt install python3-tk; Fedora/Rocky: sudo dnf install python3-tkinter;" >&2
  echo "      macOS: use the python.org installer or 'brew install python-tk'. The command line works without it." >&2
fi

echo "== 1/3  Creating the environment in $ENV_DIR"
"$PY" -m venv "$ENV_DIR"
EPY="$ENV_DIR/bin/python"
"$EPY" -m pip install --quiet --upgrade pip

echo "== 2/3  Installing Animal World from: $PKG"
"$EPY" -m pip install --quiet "$(req run)"

if [ "$GPU" = 1 ]; then
  echo "== 3/3  Installing the full engine for an NVIDIA GPU (several GB; this takes a while)"
  "$EPY" -m pip install --quiet torch torchvision --index-url "${ANIMALWORLD_TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
  "$EPY" -m pip install --quiet "$(req run-gpu)"
  if [ "$SAM3" = 1 ]; then
    # OPTIONAL. SAM 3.1 video support is in Meta's GitHub repository, not in the PyPI package; its weights are gated
    # (accept the licence at https://huggingface.co/facebook/sam3.1). The default tracker does not need it.
    "$EPY" -m pip install --quiet --no-deps "${ANIMALWORLD_SAM3_SOURCE:-git+https://github.com/facebookresearch/sam3.git}"
    "$EPY" -m pip install --quiet ftfy regex iopath einops pycocotools timm
  fi
else
  echo "== 3/3  CPU engine installed. For the full engine on an NVIDIA GPU run:  bash install.sh --gpu"
fi

echo
"$ENV_DIR/bin/animalworld" weights || true
echo
echo "Animal World is installed. Start it with:"
echo
echo "    $ENV_DIR/bin/animalworld studio     # Animal World Studio: the whole pipeline in one window"
echo "    $ENV_DIR/bin/animalworld demo       # fetch the small demo clips and open Animal World Run with them"
echo "    $ENV_DIR/bin/animalworld run        # Animal World Run: pick mouse videos, press Start  (also: $ENV_DIR/bin/mouseworld)"
echo
echo "Without a display:  $ENV_DIR/bin/animalworld zeroshot VIDEOS... --out FOLDER"

"""Stage-3 describers (VLM over an OpenAI-compatible API). See base.py for the answer contract.

The stage-2 scripts in vlm/ import each other as top-level modules (`import imgutil`), so the vlm/
folder is put on sys.path here. Nothing runs at import time.
"""
import sys
from pathlib import Path

_VLM_DIR = str(Path(__file__).resolve().parents[1])
if _VLM_DIR not in sys.path:
    sys.path.insert(0, _VLM_DIR)

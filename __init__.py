"""ComfyUI-H3-ClipColorAnchor - stop colour drift from accumulating across chained MiniMax H3 clips.

License: MIT (this node only).
"""
from .nodes import H3ClipColorAnchorExtension, comfy_entrypoint

__all__ = ["H3ClipColorAnchorExtension", "comfy_entrypoint"]

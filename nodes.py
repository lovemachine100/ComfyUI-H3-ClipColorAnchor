"""ComfyUI-H3-ClipColorAnchor - per-clip colour anchoring for chained MiniMax H3 clips.

Each chained clip drifts a little in colour and the drift accumulates (skin goes pale/lilac, dark
clothes and hair get a purple sheen, the picture darkens). This node maps the per-channel LAB
distribution (CDF / histogram matching) of the masked person region of a clip to that of the anchor
clip (clip 1). Feed the corrected clip both to the output and to ClipContext as `context_frames`
so the next clip starts from corrected colours.

License: MIT (this node only).
"""
from __future__ import annotations

import logging

import numpy as np
import torch

from comfy_api.latest import ComfyExtension, io

from . import color_anchor as ca

_LOG = logging.getLogger("h3_clip_color_anchor")


class _LazyU8:
    """[B,H,W,C] float 0..1 tensor -> uint8 RGB frames converted per access (keeps RAM flat for long clips)."""

    def __init__(self, img):
        self.img = img

    def __len__(self):
        return int(self.img.shape[0])

    def __getitem__(self, i):
        f = (self.img[i, ..., :3].float().clamp(0, 1) * 255.0).round().to(torch.uint8)
        return np.ascontiguousarray(f.cpu().numpy())


def _mask_np(mask):
    if mask is None:
        return None
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    return mask.float().cpu().numpy()


class H3ClipColorAnchor(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ClipColorAnchor",
            display_name="H3 Clip Color Anchor",
            category="MiniMax H3/color",
            description=(
                "Match the colour distribution (per-channel LAB CDF) of the masked region of `images` to "
                "`anchor_images` (clip 1 of a chain) so colour drift does not accumulate over chained H3 clips. "
                "One lookup table per clip (no per-frame flicker); everything outside the soft mask is untouched. "
                "Without a mask the whole frame is used: only safe when the background is static."),
            inputs=[
                io.Image.Input("images", tooltip="Clip to correct (RGB, any frame count)."),
                io.Image.Input("anchor_images", tooltip="Colour reference: clip 1 of the chain (or its head frames)."),
                io.Mask.Input("mask", optional=True,
                              tooltip="Person mask for `images`: one mask per frame (aligned in time) or a single mask "
                                      "broadcast to all frames. Without it the whole frame is used."),
                io.Mask.Input("anchor_mask", optional=True,
                              tooltip="Person mask for `anchor_images`. If omitted the whole anchor frame is sampled "
                                      "(then the statistics of the two sides are not comparable when `mask` is set)."),
                io.Combo.Input("method", options=["histogram", "mean"], default="histogram",
                               tooltip="histogram: per-channel L/a/b CDF matching (recommended). "
                                       "mean: old method, shift mean a/b to the anchor + chroma scale clipped 0.6..1.15."),
                io.Boolean.Input("match_lightness", default=True,
                                 tooltip="histogram: also match L. mean: shift mean L by up to +-12 (8-bit units). "
                                         "Off = lightness untouched."),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.05,
                               tooltip="Blend between original (0) and corrected (1)."),
                io.Int.Input("mask_grow", default=31, min=0, max=201, step=1,
                             tooltip="Dilate the binarised mask with an elliptical kernel of this diameter (px). 0 = off."),
                io.Int.Input("mask_feather", default=31, min=0, max=201, step=1,
                             tooltip="Gaussian blur kernel (px, forced odd) for the soft mask edge. 0 = hard edge."),
                io.Int.Input("sample_stride", default=3, min=1, max=64, step=1,
                             tooltip="Use every Nth frame to build the statistics."),
                io.Float.Input("max_shift", default=0.0, min=0.0, max=128.0, step=1.0,
                               tooltip="0 = off. Otherwise cap |LUT[v]-v| (8-bit LAB units, histogram method) so a clip whose "
                                       "content changed a lot (new object, different pose/lighting) cannot be pushed too far. "
                                       "10-25 is a reasonable guard; drift correction itself is usually within ~10."),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, images, anchor_images, mask=None, anchor_mask=None, method="histogram",
                match_lightness=True, strength=1.0, mask_grow=31, mask_feather=31,
                sample_stride=3, max_shift=0.0) -> io.NodeOutput:
        if images.shape[0] < 1 or anchor_images.shape[0] < 1:
            raise ValueError("[H3ClipColorAnchor] images / anchor_images must not be empty")
        import comfy.utils
        pbar = comfy.utils.ProgressBar(int(images.shape[0]))
        out_u8, report = ca.correct(
            _LazyU8(images), _mask_np(mask), _LazyU8(anchor_images), _mask_np(anchor_mask),
            method=method, match_lightness=match_lightness, strength=strength, grow=mask_grow,
            feather=mask_feather, stride=sample_stride, max_shift=max_shift,
            progress=lambda i, n: pbar.update_absolute(i, n))
        if mask is not None and anchor_mask is None:
            report += " | WARNING: anchor_mask not given, anchor sampled over the whole frame"
        out = torch.from_numpy(out_u8).float().div_(255.0)
        if images.shape[-1] > 3:  # keep alpha untouched
            out = torch.cat([out, images[..., 3:].cpu().float()], dim=-1)
        _LOG.info(report)
        return io.NodeOutput(out, report)


class H3ClipColorAnchorExtension(ComfyExtension):
    async def get_node_list(self):
        return [H3ClipColorAnchor]


async def comfy_entrypoint() -> H3ClipColorAnchorExtension:
    return H3ClipColorAnchorExtension()

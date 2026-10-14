"""Core algorithm (numpy, optional OpenCV, no ComfyUI imports) so it can be tested standalone.

Convention: frames are uint8 RGB [H, W, 3]; LAB is OpenCV's 8-bit LAB
(L = L*255/100, a/b offset by 128, sRGB gamma, D65 white).

OpenCV (cv2) is used when it can be imported; otherwise built-in numpy/torch fallbacks reproduce
COLOR_RGB2LAB / COLOR_LAB2RGB, the elliptical dilate, the gaussian blur and the nearest resize
(differences against cv2 are at the +-1 level of 8-bit LAB; see README). Set H3CCA_FORCE_FALLBACK=1
to force the fallback. Nothing here fails at import time.
"""
from __future__ import annotations

import math
import os

import numpy as np

try:  # optional
    import cv2 as _CV2
except Exception:  # ImportError or a broken binary
    _CV2 = None

MIN_PIXELS = 500  # per-frame minimum selected pixels for the "mean" method


def _cv2():
    """The cv2 module, or None when unavailable / disabled."""
    if os.environ.get("H3CCA_FORCE_FALLBACK") == "1":
        return None
    return _CV2


def backend() -> str:
    return "opencv" if _cv2() is not None else "builtin"


# ---------------------------------------------------------------- built-in fallbacks (match OpenCV's float formulas)
_X = np.arange(256, dtype=np.float64) / 255.0
_GAMMA = np.where(_X <= 0.04045, _X / 12.92, ((_X + 0.055) / 1.055) ** 2.4).astype(np.float32)
_WHITE = np.array([0.950456, 1.0, 1.088754])
_RGB2XYZ = np.array([[0.412453, 0.357580, 0.180423],
                     [0.212671, 0.715160, 0.072169],
                     [0.019334, 0.119193, 0.950227]])
_M_FWD = (_RGB2XYZ / _WHITE[:, None]).T.astype(np.float32)           # rgb_lin @ M -> xyz / white
_XYZ2RGB = np.array([[3.240479, -1.53715, -0.498535],
                     [-0.969256, 1.875991, 0.041556],
                     [0.055648, -0.204043, 1.057311]])
_M_INV = (_XYZ2RGB * _WHITE[None, :]).T.astype(np.float32)           # (xyz/white) @ M -> rgb_lin
_T = 0.008856
_FT = 7.787 * _T + 16.0 / 116.0


def _rgb2lab_np(rgb):
    xyz = _GAMMA[rgb] @ _M_FWD
    f = np.where(xyz > _T, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0).astype(np.float32)
    out = np.empty(rgb.shape, np.float32)
    out[..., 0] = (116.0 * f[..., 1] - 16.0) * (255.0 / 100.0)
    out[..., 1] = 500.0 * (f[..., 0] - f[..., 1]) + 128.0
    out[..., 2] = 200.0 * (f[..., 1] - f[..., 2]) + 128.0
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _lab2rgb_np(lab):
    L = lab[..., 0].astype(np.float32) * (100.0 / 255.0)
    a = lab[..., 1].astype(np.float32) - 128.0
    b = lab[..., 2].astype(np.float32) - 128.0
    fy = np.where(L <= 8.0, 7.787 * (L / 903.3) + 16.0 / 116.0, (L + 16.0) / 116.0)
    f = np.stack([a / 500.0 + fy, fy, fy - b / 200.0], -1).astype(np.float32)
    xyz = np.where(f > _FT, f * f * f, (f - 16.0 / 116.0) / 7.787).astype(np.float32)
    lin = np.clip(xyz @ _M_INV, 0.0, 1.0)
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1.0 / 2.4) - 0.055)
    return np.clip(np.rint(srgb * 255.0), 0, 255).astype(np.uint8)


def rgb2lab(rgb):
    cv2 = _cv2()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB) if cv2 is not None else _rgb2lab_np(rgb)


def lab2rgb(lab):
    cv2 = _cv2()
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) if cv2 is not None else _lab2rgb_np(lab)


def _ellipse_halfwidths(k):
    r = k // 2
    return r, [int(np.rint(math.sqrt(max(r * r - dy * dy, 0)))) for dy in range(-r, r + 1)]


def _dilate_np(m, k):
    """cv2.dilate(m, getStructuringElement(MORPH_ELLIPSE, (k, k))) for a uint8 0/1 mask (zero border)."""
    H, W = m.shape
    r, hw = _ellipse_halfwidths(k)
    csum = np.zeros((H, W + 1), np.int32)
    np.cumsum(m, axis=1, out=csum[:, 1:])
    xs = np.arange(W)
    rows = {}
    for w in set(hw):  # horizontal dilation by half-width w via prefix sums
        lo = np.clip(xs - w, 0, W)
        hi = np.clip(xs + w + 1, 0, W)
        rows[w] = (csum[:, hi] - csum[:, lo]) > 0
    out = np.zeros((H, W), bool)
    for i, w in enumerate(hw):
        dy = i - r
        h = rows[w]
        if dy >= 0:
            out[: H - dy] |= h[dy:]
        else:
            out[-dy:] |= h[: H + dy]
    return out.astype(np.uint8)


def _gaussian_np(m, k):
    """cv2.GaussianBlur(m, (k, k), 0) for float32 [H, W] (sigma from ksize, reflect-101 borders)."""
    import torch
    import torch.nn.functional as F
    sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
    x = np.arange(k, dtype=np.float64) - (k - 1) / 2.0
    ker = np.exp(-(x * x) / (2 * sigma * sigma))
    ker = torch.from_numpy((ker / ker.sum()).astype(np.float32))
    t = torch.from_numpy(m)[None, None]
    p = k // 2
    t = F.pad(t, (p, p, p, p), mode="reflect") if min(m.shape) > p else F.pad(t, (p, p, p, p), mode="replicate")
    t = F.conv2d(t, ker.view(1, 1, 1, k))
    t = F.conv2d(t, ker.view(1, 1, k, 1))
    return t[0, 0].numpy()


def _resize_nearest(m, shape):
    H, W = shape
    h, w = m.shape
    ys = np.minimum((np.arange(H) * (h / H)).astype(np.int64), h - 1)
    xs = np.minimum((np.arange(W) * (w / W)).astype(np.int64), w - 1)
    return m[ys][:, xs]


def soft_mask(mask, shape, grow, feather):
    """mask [h, w] float 0..1 -> soft float32 0..1 at shape (H, W): binarise at 0.5, dilate (ellipse), gaussian blur."""
    cv2 = _cv2()
    m = (mask > 0.5).astype(np.uint8)
    if m.shape != tuple(shape):
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) if cv2 is not None else _resize_nearest(m, shape)
    if grow > 1:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow))) if cv2 is not None else _dilate_np(m, grow)
    m = m.astype(np.float32)
    if feather > 1:
        k = feather if feather % 2 == 1 else feather + 1
        m = cv2.GaussianBlur(m, (k, k), 0) if cv2 is not None else _gaussian_np(m, k)
    return m


def _mask_for(masks, i, shape, grow, feather, cache):
    if masks is None:
        return np.ones(shape, np.float32)
    j = min(i, masks.shape[0] - 1)
    if j not in cache:
        if len(cache) > 8:
            cache.clear()
        cache[j] = soft_mask(masks[j], shape, grow, feather)
    return cache[j]


def collect_stats(frames, masks, stride, grow, feather):
    """3x256 histograms of 8-bit LAB inside soft-mask > 0.5, sampled every `stride` frames,
    plus per-frame-mean a/b/chroma/L (for the 'mean' method)."""
    n = len(frames)
    hist = np.zeros((3, 256), np.int64)
    pa, pb, pc, pl = [], [], [], []
    cache = {}
    total = 0
    for i in range(0, n, max(1, stride)):
        f = frames[i]
        lab = rgb2lab(f)
        m = _mask_for(masks, i, f.shape[:2], grow, feather, cache)
        sel = m > 0.5
        cnt = int(sel.sum())
        if cnt == 0:
            continue
        px = lab[sel]
        total += cnt
        for c in range(3):
            hist[c] += np.bincount(px[:, c], minlength=256)
        if cnt >= MIN_PIXELS:
            a = px[:, 1].astype(np.float32) - 128
            b = px[:, 2].astype(np.float32) - 128
            pa.append(a.mean()); pb.append(b.mean())
            pc.append(np.sqrt(a * a + b * b).mean()); pl.append(px[:, 0].astype(np.float32).mean())
    return {"hist": hist, "n": total,
            "a": float(np.mean(pa)) if pa else 0.0, "b": float(np.mean(pb)) if pb else 0.0,
            "c": float(np.mean(pc)) if pc else 0.0, "L": float(np.mean(pl)) if pl else 0.0}


def build_luts(src_hist, ref_hist, match_lightness=True, max_shift=0.0):
    """CDF matching LUT (3, 256) float32: LUT[c][v] = interp(cdf_src(v) -> cdf_ref)."""
    x = np.arange(256)
    luts = np.zeros((3, 256), np.float32)
    for c in range(3):
        if c == 0 and not match_lightness:
            luts[c] = x
            continue
        cs = src_hist[c].cumsum() / max(1, src_hist[c].sum())
        cr = ref_hist[c].cumsum() / max(1, ref_hist[c].sum())
        lut = np.interp(cs, cr, x).astype(np.float32)
        if max_shift and max_shift > 0:
            lut = (x + np.clip(lut - x, -max_shift, max_shift)).astype(np.float32)
        luts[c] = lut
    return luts


def lab_mean_real(hist):
    """3x256 histograms -> mean (L 0..100, a, b) in real LAB scale."""
    out = []
    for c, (scale, off) in enumerate(((100.0 / 255.0, 0.0), (1.0, -128.0), (1.0, -128.0))):
        h = hist[c]
        mean = (h * np.arange(256)).sum() / max(1, h.sum())
        out.append(float(mean * scale + off))
    return tuple(out)


def apply_hist(frame, mm, luts):
    l8 = rgb2lab(frame)
    lab = l8.astype(np.float32)
    for c in range(3):
        lab[..., c] = lab[..., c] * (1 - mm) + luts[c][l8[..., c]] * mm
    return lab2rgb(np.clip(lab + 0.5, 0, 255).astype(np.uint8))


def apply_mean(frame, mm, ref, src, match_lightness):
    """Old per-clip method: shift mean a/b to the anchor + chroma scale (0.6..1.15), optional L shift (+-12)."""
    k = float(np.clip(ref["c"] / max(src["c"], 1e-3), 0.6, 1.15))
    dL = float(np.clip(ref["L"] - src["L"], -12, 12)) if match_lightness else 0.0
    lab = rgb2lab(frame).astype(np.float32)
    a = lab[..., 1] - 128
    b = lab[..., 2] - 128
    a2 = (a - src["a"]) * k + ref["a"]
    b2 = (b - src["b"]) * k + ref["b"]
    lab[..., 1] = 128 + a * (1 - mm) + a2 * mm
    lab[..., 2] = 128 + b * (1 - mm) + b2 * mm
    if dL:
        lab[..., 0] = lab[..., 0] + dL * mm
    return lab2rgb(np.clip(lab, 0, 255).astype(np.uint8))


def de76(p, q):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(p, q)))


def correct(images, masks, anchor, anchor_masks, method="histogram", match_lightness=True,
            strength=1.0, grow=31, feather=31, stride=3, max_shift=0.0, progress=None):
    """images / anchor: indexable (len + [i]) of uint8 RGB frames. masks / anchor_masks: float [N,h,w] numpy or None.
    Returns (uint8 [N,H,W,3] array, report str)."""
    ref = collect_stats(anchor, anchor_masks, stride, grow, feather)
    src = collect_stats(images, masks, stride, grow, feather)
    if ref["n"] == 0 or src["n"] == 0:
        raise RuntimeError("[H3ClipColorAnchor] mask selects no pixels (anchor=%d, images=%d); check the masks / mask_grow"
                           % (ref["n"], src["n"]))
    luts = build_luts(src["hist"], ref["hist"], match_lightness, max_shift) if method == "histogram" else None
    n = len(images)
    first = images[0]
    out = np.empty((n,) + tuple(first.shape), np.uint8)
    cache = {}
    after = np.zeros((3, 256), np.int64)
    s = float(np.clip(strength, 0.0, 1.0))
    st = max(1, stride)
    for i in range(n):
        f = first if i == 0 else images[i]
        m = _mask_for(masks, i, f.shape[:2], grow, feather, cache)
        mm = m * s
        out[i] = apply_hist(f, mm, luts) if method == "histogram" else apply_mean(f, mm, ref, src, match_lightness)
        if i % st == 0:
            sel = m > 0.5
            if sel.any():
                px = rgb2lab(out[i])[sel]
                for c in range(3):
                    after[c] += np.bincount(px[:, c], minlength=256)
        if progress:
            progress(i + 1, n)
    A, B, C = lab_mean_real(ref["hist"]), lab_mean_real(src["hist"]), lab_mean_real(after)
    shifts = ""
    if luts is not None:
        x = np.arange(256)
        shifts = " | max|LUT shift| L/a/b (8bit): %s" % "/".join("%.1f" % float(np.abs(luts[c] - x).max()) for c in range(3))
    fmt = lambda t: "L=%.2f a=%.2f b=%.2f" % t
    report = ("method=%s strength=%.2f stride=%d | anchor %s (px=%d) | before %s (px=%d) | after %s | "
              "dE76 vs anchor (mask mean): before %.2f -> after %.2f%s"
              % (method, s, st, fmt(A), ref["n"], fmt(B), src["n"], fmt(C), de76(A, B), de76(A, C), shifts))
    report += " | backend=" + backend()
    if masks is None:
        report += " | NOTE: no mask -> whole frame used (only safe for static backgrounds)"
    return out, report

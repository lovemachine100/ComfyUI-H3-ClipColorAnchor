# ComfyUI-H3-ClipColorAnchor

Stop colour drift from **accumulating** across chained MiniMax H3 clips.
MiniMax H3 クリップ連鎖で、1 本ごとに少しずつ乗る色ずれ (肌が青白く / 黒服・黒髪が紫に / 全体が暗く) が積もるのを止めるノード。

When each clip continues from the tail frames of the previous one, the colour of the person drifts a little per
generation (pale/lilac skin, purple sheen on black clothes and hair, darker picture). Changing steps or LoRA
strength did not help. What did: after every clip, map the per-channel **LAB distribution (CDF / histogram
matching)** of the person region to that of **clip 1 (the anchor)**, and use the corrected clip both as the
output and as the context for the next clip.

(Measured in a 5-segment, 12 fps chained-clip test: dE 4.2 -> 1.9, purple-pixel share +0.087 -> +0.035; a second character 4.2 -> 2.5.)

## Node: `H3 Clip Color Anchor` (category `MiniMax H3/color`)

| input | type | |
|---|---|---|
| `images` | IMAGE | clip to correct |
| `anchor_images` | IMAGE | clip 1 of the chain (or its head frames) |
| `mask` | MASK, optional | person mask for `images`; per-frame (time-aligned) or a single mask broadcast |
| `anchor_mask` | MASK, optional | person mask for `anchor_images` (give it whenever you give `mask`) |
| `method` | histogram / mean | histogram = per-channel L/a/b CDF matching. mean = old method (shift mean a/b to the anchor, chroma scale clipped 0.6..1.15) kept for comparison |
| `match_lightness` | bool (True) | histogram: also match L. mean: shift mean L (max +-12). Off = L untouched |
| `strength` | 0..1 (1.0) | blend original -> corrected |
| `mask_grow` | int (31) | elliptical dilation (px) of the binarised mask |
| `mask_feather` | int (31) | gaussian blur (px, forced odd) of the grown mask -> soft 0..1 mask |
| `sample_stride` | int (3) | statistics from every Nth frame, pixels with soft mask > 0.5 |
| `max_shift` | float (0 = off) | cap `abs(LUT[v]-v)` in 8-bit LAB units. A guard for clips whose content changed a lot. 10-25 is a sensible range |

Outputs: `images` (corrected) and `report` (STRING: LAB mean of anchor / before / after, dE76 before -> after, largest LUT shift).

How it works: one 256-entry LUT per channel and per clip (no per-frame flicker), built from the OpenCV 8-bit LAB
cumulative histograms inside the mask: `LUT = interp(cdf_clip, cdf_anchor)`. Per pixel
`out = lab*(1-m) + LUT[lab]*m` with `m` the soft mask * `strength`; the background outside the soft mask is untouched.

### No mask
Without `mask` the whole frame is sampled and corrected. That is only safe when the background is static
(moving monitor content, camera moves, lighting changes etc. bias the statistics and get "corrected" too).
With `mask` but without `anchor_mask` the anchor is sampled over the whole frame (a warning is added to the report).

### Notes
- `max_shift`: histogram matching can have large LUT shifts at the sparse tails of the distribution
  (a/b shifts of 70-90 units appeared at the extreme ends in a test clip, where very few pixels live). With 0 you get the
  plain histogram matching; with 20 the dE result was the same (0.20) in that test.
- Frames: IMAGE float RGB is rounded to 8 bit for the LAB step (as video frames are anyway). Alpha is passed through.
- **No pip install needed.** OpenCV (`cv2`) is used when it can be imported; otherwise a built-in numpy/torch fallback
  does the same work (RGB<->8-bit LAB, mask dilate / blur / resize). The fallback reproduces OpenCV's LAB conversion to
  within +-1..3 LAB units (mean 0.1) and is about 4x slower (about 20 s instead of 5 s for 90 frames of 640x1024); the
  dE result is the same to 0.01. The `report` ends with `backend=opencv` or `backend=builtin`. Set the environment variable
  `H3CCA_FORCE_FALLBACK=1` to force the fallback. No model, no GPU.
- Memory: frames are converted one at a time; the output tensor is one extra copy of the clip.
- Video decoders differ slightly (PyAV vs OpenCV: L differs ~0.8 for the same file). Always take the anchor and the clip
  through the same loader.

## Wiring in a chain (before ClipContext)

```
clip 1 (generated)  ----------------------------------------> anchor_images ----+
clip N (decoded, trimmed)  --> images --> [H3 Clip Color Anchor] --> images ----+--> SaveVideo / concat
person masks (e.g. SAM3 / a mask video) -> mask, anchor_mask                    |
                                                                                +--> ComfyUI-H3-ClipContext.context_frames
                                                                                     (and head_frames if used) for clip N+1
```

Put it **between the VAE decode (after trimming the pinned tail) and everything that consumes the clip**.
The corrected frames must be what ClipContext sees as `context_frames`; correcting only at the end does not stop the
accumulation (a post-only fix reached dE 4.0 vs 1.9 in the chain test). Clip 1 itself is the anchor and is left as is.
Mask videos at a different fps than the clip must be sub-sampled to the clip frames before the node.

Example workflows (`example_workflows/`): `h3_clip_color_anchor_minimal_{ui,api}.json` - two videos (clip 1, clip N)
and two mask videos -> `ImageToMask` -> Color Anchor -> `SaveVideo`, with the report in a Preview as Text node.
Sample videos are **not** shipped. To try it:
1. Put your own clip 1, clip N and the two person-mask videos (white = person; same frame count as the clips) into
   `ComfyUI/input` as `clip1.mp4`, `clipN.mp4`, `mask_clip1.mp4`, `mask_clipN.mp4` - or just pick your files in the four
   `Load Video` nodes.
2. Open the workflow and run. If a SAM3 (or other segmentation) node is installed, feed its mask output into `mask` /
   `anchor_mask` instead of the mask videos.

## Install
```
cd ComfyUI/custom_nodes
git clone https://github.com/lovemachine100/ComfyUI-H3-ClipColorAnchor.git
```
Restart ComfyUI. No pip install is needed.

License: MIT (this node only).

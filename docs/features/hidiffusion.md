---
title: HiDiffusion
---

# HiDiffusion

HiDiffusion is an optional denoising enhancement that can improve detail and structure at higher resolutions for SD 1.5 and SDXL. It modifies the UNet during denoising and is most noticeable at 1536px and above.

Learn more: https://github.com/megvii-research/HiDiffusion

## Where to find the switches

1. Open the **Canvas** tab.
2. Expand **Advanced Settings**.
3. In the **Advanced** grid, enable **HiDiffusion** and optionally adjust the two sub‑toggles:
   - **HiDiffusion: RAU‑Net**
   - **HiDiffusion: Window Attention**

## What the switches do

- **HiDiffusion**  
  Enables the HiDiffusion patch for denoising. Use this for high‑resolution generations; the effect is subtle at lower sizes.

- **HiDiffusion: RAU‑Net**  
  Enables RAU‑Net blocks. This typically improves structure and mid‑frequency detail, especially at larger resolutions.

- **HiDiffusion: Window Attention**  
  Enables windowed attention blocks. This can boost local texture/detail, but may slightly affect global coherence in some prompts.

## Tips

- Try **1536–2048 px** for the clearest benefits (SDXL).
- If results look worse, disable **Window Attention** first, then RAU‑Net.
- Effects vary by scheduler and model; compare with the same seed for a fair test.

---
title: Wan VAE Memory Calibration
lastUpdated: 2026-10-26
---

Before a Wan VAE decode, Invoke estimates the working memory it needs and asks the model cache to keep that much
VRAM free (see [Low-VRAM mode](/configuration/optimization/low-vram-mode/)). The estimate scales with the video's
size through a constant shipped in the code. `scripts/calibrate_wan_vae_working_memory.py` measures a real decode so
you can check that constant on your own GPU.

## Running it

It needs a CUDA or ROCm device and a Wan VAE, as a Diffusers `AutoencoderKLWan` directory or a `.safetensors`
checkpoint:

```sh
python scripts/calibrate_wan_vae_working_memory.py --vae <directory-or-safetensors-file>
```

| Option | Default | Measures |
| --- | --- | --- |
| `--vae` | required | The VAE to load. |
| `--height`, `--width`, `--frames` | `704`, `1280`, `81` | The video's size in pixels and frames. |
| `--dtype` | `float16` | The VAE's dtype. |
| `--streaming` / `--no-streaming` | streaming | The chunked decode that `wan_memory_optimization` streams to MP4, or the default full decode. |
| `--tiling` | off | The spatially tiled full decode, the fallback for decodes too large for the card. Overrides `--streaming`. |
| `--tile-size <pixels>` | the VAE's tile size | The tile size. Requires `--tiling`. |

## Reading the result

The script prints the estimate, the measured allocated and reserved memory deltas, the scaling constant the
reserved delta implies, and `estimate covers: <ratio>x of reserved`.

- The implied constant uses **reserved** memory, which is what the shipped estimator covers: reserved memory is what
  the decode takes from the GPU.
- The shipped constant sits above what the script reports for most sizes, because it has to cover the highest
  measurement, an 832x480 decode inside a running server. `estimate covers` says whether the estimate holds: 1.00x or
  more means it does.
- With `--tiling` the copies of each clip can exceed the whole measurement and leave a negative implied constant;
  read `estimate covers` instead.

## Why streaming holds the VAE lock

Streaming to MP4 keeps the VAE cache lock while chunks are decoded and written. Wan's causal decoder state and weights
must stay live for the whole sequence; releasing the lock between chunks would need a separate, bounded decode and
encode queue.

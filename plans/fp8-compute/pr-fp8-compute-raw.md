# PR-Vorlage — `feat/fp8_compute_raw`

**Base:** `feat/fp8_scaled_compute` (Stacked PR — nicht gegen `main` öffnen)
**Head:** `feat/fp8_compute_raw` @ `e93854e6ae`
**Titel:** `feat(fp8): run raw fp8 checkpoints on the tensor cores`

> Kleinigkeit vor dem Öffnen: die letzten zwei Zeilen der Commit-Message enthalten zwei Fassungen
> derselben Messung (`1.60 -> 1.27 s/it` und `1.297 -> 1.023 s/it`). Die zweite ist die richtige
> (3 warme Läufe). Falls noch nicht gepusht, lohnt ein `git commit --amend`.

---

## Summary

A checkpoint can ship fp8 weights with no `weight_scale`. The runtime already handles them —
`scaled_mm_linear` treats `weight_scale` as optional — but the loaders never let them through:
FLUX, FLUX.2 and Z-Image cast the whole state dict to bf16, discarding both the VRAM saving and the
tensor cores. Krea-2 kept them by accident and said nothing about it.

This routes all four loaders through a shared `cast_state_dict()` that keeps qualifying fp8 weights
quantized when fp8 compute is enabled and the device supports it.

**Only `nn.Linear.weight` is preserved.** That restriction is not cosmetic: a Z-Image checkpoint
quantized everything — 243 of its 453 fp8 tensors are 1-D biases, norm weights and a learned pad
token. Keeping those fp8 saves nothing usable and breaks inference: the value reaches the
activations and the next Linear gets an fp8 input, which dies in `x.abs()` with
`"abs_cuda" not implemented for 'Float8_e4m3fn'`. A model's own
`_skip_layerwise_casting_patterns` is honored on top, for Linears whose forward casts activations to
their weight's dtype.

It also stops fp8 **storage** from silently defeating fp8 **compute**: layerwise casting restores
the compute dtype before every forward, so on an already-fp8 checkpoint the matmul would quietly
fall back and the VRAM toggle would make the model *slower* with no indication why. The loader now
logs and skips layerwise casting in that case.

## Related Issues / Discussions

Stacked on `feat/fp8_scaled_compute`, which adds `torch._scaled_mm` for ComfyUI **scaled**-fp8
checkpoints (fp8 weight + `weight_scale`). This PR covers the other kind: fp8 weight, no scale.

## QA Instructions

Requires a device with fp8 matmul support (SM ≥ 8.9 — Ada/Hopper/Blackwell) and `fp8_compute: true`
in `invokeai.yaml`.

1. Set `enable_partial_loading: false`. fp8 compute needs the model fully resident — under partial
   loading the same seed does not reproduce, and run-to-run noise swamps any measurement.
2. Load a **raw** fp8 checkpoint (fp8 weights, no `weight_scale`) — e.g. Z-Image
   `unstableRevolution_V2Fp8`. The log must show `kept N raw fp8` and **no**
   `FP8 layerwise casting enabled` line.
3. Check `Total model size` in the log: it should roughly halve versus the same checkpoint on the
   base branch.
4. Generate **4 images** (first one discarded as warm-up) and compare the warm per-step times. Do
   not judge from a single image — cold load and residency drift both hide in it.
5. With `fp8_compute: false`, behavior must be unchanged: weights dequantize to bf16 as before.
6. Regression check on a **scaled**-fp8 checkpoint (`weight_scale` present, e.g. FLUX.2
   `flux-2-klein-9b-fp8`): behavior must be byte-identical — no `kept N raw fp8` line, same
   `Total model size` as on the base branch. Verified: 8707.52 MB on both, and
   `FP8 layerwise casting enabled ... param_size=8708MB` present in both logs.

### Measured — Z-Image `unstableRevolution_V2Fp8`, 1024×1024, 30 steps, RTX 4090

| | base | this PR |
|---|---|---|
| Transformer VRAM | 11 740 MB | **5 881 MB** |
| Residency | 95.5–100 % | **100 %** |
| s/it, warm runs | 1.19 / 1.34 / 1.36 | **1.01 / 1.06 / 1.00** |
| s/it, mean | 1.297 | **1.023** (1.27×) |
| Graph total | 40.0 s | **30.9 s** |

Worth noting in the individual numbers: the base **drifts upward** across warm runs while fp8 stays
flat. That is the full residency, not the matmul — and it is invisible in a mean.

## Merge Plan

Merge after `feat/fp8_scaled_compute`. No migration, no schema change, no new dependency. Behavior
is gated behind the existing `fp8_compute` flag, which defaults to `false`.

## Checklist

- [x] The PR has a short but descriptive title
- [x] Tests added / updated (`tests/backend/quantization/test_fp8_scaled.py`, +102 lines; full
      suite 904 passing)
- [ ] Documentation added / updated — covered by the base branch's
      `docs/src/content/docs/configuration/fp8-storage.mdx`
- [x] Updated `What's New` copy — n/a, opt-in developer flag

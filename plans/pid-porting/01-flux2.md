# PiD für FLUX.2 Klein (4B / 9B)

> Voraussetzung: gemeinsame Vorarbeit aus [README.md](README.md) (Abschnitte A–G).
> Backend ist hier fast fertig — es fehlt nur der Decode-Node.

**Status der Bausteine:**
- `_PER_BACKBONE[Flux2]` ✅ vorhanden (`lq_latent_channels=128`, `latent_spatial_down_factor=16`)
- Config `PiDDecoder_Checkpoint_Flux2_Config` ✅ vorhanden (128ch → eindeutig)
- Loader-Register `base=Flux2` ✅ vorhanden
- Decode-Node `flux2_pid_decode` ❌ **fehlt**
- 4B und 9B teilen sich dieselbe FLUX.2-VAE → **ein** Node deckt beide ab.

---

## 1. Nodes

### 1.1 `app/invocations/flux2_pid_decode.py` (NEU)

`flux_pid_decode.py` kopieren, dann:

- `@invocation("flux2_pid_decode", title="Latents to Image - FLUX.2 + PiD (4x SR)", tags=[…,"flux2"], …)`
- `BaseModelType.Flux2` an `estimate_pid_decode_working_memory(latents, BaseModelType.Flux2)` und `PiDDecoder(pid_net, backbone=BaseModelType.Flux2)`.
- **Denormalisierung:** dem `z_image_pid_decode.py`-Muster folgen — optionalen
  `vae: VAEField | None`-Input anbieten und `scaling_factor`/`shift_factor` aus der
  FLUX.2-VAE-Config lesen. Fallback-Konstanten aus der FLUX.2-VAE dokumentieren.
  - Quelle für die Konstanten/das Latent-Handling: vergleiche, wie
    `flux2_vae_decode` den Latent vor dem Decode behandelt (Packing/Norm).

### 1.2 ⚠️ Latent-Layout verifizieren (wichtig)

FLUX.2: `lq_latent_channels=128`, `latent_spatial_down_factor=16`. Der an PiD
gereichte Latent **muss** 128-kanalig im selben Layout wie der `flux2_vae_decode`-Input
sein. Vor dem Vollausbau einen Workflow-Test machen:
- `flux2_denoise` → `flux2_pid_decode` und Shapes loggen (`latents.shape`).
- Erwartung: `[B, 128, H/16, W/16]`. Weicht das ab (z. B. 32ch unpacked), im Node
  das Packing analog `flux2_vae_decode` angleichen, **bevor** denormalisiert wird.

### 1.3 Loader-Node Titel (kosmetisch, optional)

`pid_decoder_loader.py` Titel ist „FLUX / FLUX.2 / SD3" — passt schon; ggf. nichts zu tun.

### 1.4 Registrierung

Nodes werden über das `@invocation`-Decorator automatisch entdeckt. Kein Eintrag
in einer Liste nötig. Nach Anlegen: Backend neu starten, `test_imports.py` läuft mit.

---

## 2. UI

FLUX.2 wird **innerhalb** von `buildFLUXGraph.ts` im `isFlux2`-Zweig gebaut
(eigener Pfad mit `flux2_denoise`, `flux2_vae_decode`, `flux2_klein_model_loader`).

### 2.1 Gemeinsame Frontend-Vorarbeit
Aus [README.md](README.md): Punkt **B** (Chain generalisieren), **C**
(`'flux2_pid_decode'` in `ImageOutputNodes`), **D** (Gating/Filter — `flux2`
ist eigener Decoder-Base), **E** (Readiness im `flux2`-Zweig), **F** (i18n reuse).

### 2.2 `buildFLUXGraph.ts` — `isFlux2`-Pfad
Analog zum Standard-FLUX-Pfad (`addPidDecode` / `buildPidDecodeChain`):

- **Guard** (vor dem `generationMode`-Switch im `isFlux2`-Block): inpaint/outpaint
  sperren (`toast.pidUnsupportedMode`); „Scale Before Processing" off erzwingen.
- **txt2img:** statt `addTextToImage({… l2i: flux2L2i})` →
  `g.deleteNode(flux2L2i.id)` + `addPidDecode({ g, state, mode: pidMode, denoise: flux2Denoise, positivePrompt, seed, decodeNodeType: 'flux2_pid_decode', vaeSource: flux2ModelLoader })`.
- **img2img Fit:** `buildPidDecodeChain({…, mode:'fit', fitSize: originalSize, decodeNodeType:'flux2_pid_decode', vaeSource: flux2ModelLoader})` als `l2i` an `addImageToImage`.
- **img2img Native:** `addPidImageToImageNative({…, decodeNodeType:'flux2_pid_decode', vaeSource: flux2ModelLoader})`.
- **Positive-Prompt-String:** im `isFlux2`-Pfad existiert `positivePrompt` (oben im
  Builder als `string`-Node angelegt) bereits → direkt nutzen. Verifizieren, dass
  er den vollen Prompt trägt.

> FLUX.2 nutzt `getGridSize('flux2', …)` / `getOptimalDimension('flux2', …)` — beide
> sind bereits pidScale-aware, Native funktioniert ohne weitere Änderung.

### 2.3 Gating-Detail
`flux2` in `PID_SUPPORTED_BASES` und `getPidDecoderBaseForMainBase('flux2') = 'flux2'`.
`PidSettings` zeigt damit automatisch FLUX.2-PiD-Decoder im Combobox.

---

## 3. Starter Models

In `starter_models.py`, Region „PiD", analog zu den FLUX-Einträgen:

```python
pid_decoder_flux2_2k = StarterModel(
    name="PiD Decoder FLUX.2 (2K)",
    base=BaseModelType.Flux2,
    source="nvidia/PiD::checkpoints/PiD_res2k_sr4x_official_flux2_distill_4step/model_ema_bf16.pth",
    description="NVIDIA PiD 4x super-resolution decoder for FLUX.2 latents, 2K target preset. ~5GB",
    type=ModelType.PiDDecoder,
    format=ModelFormat.Checkpoint,
    variant=PiDDecoderVariantType.Res2k_Sr4x,
    dependencies=[gemma2_2b_encoder],
)
pid_decoder_flux2_2kto4k = StarterModel(
    name="PiD Decoder FLUX.2 (2K to 4K)",
    base=BaseModelType.Flux2,
    source="nvidia/PiD::checkpoints/PiD_res2kto4k_sr4x_official_flux2_distill_4step/model_ema_bf16.pth",
    description="NVIDIA PiD 4x super-resolution decoder for FLUX.2 latents, 2K-to-4K preset. ~5GB",
    type=ModelType.PiDDecoder,
    format=ModelFormat.Checkpoint,
    variant=PiDDecoderVariantType.Res2kTo4k_Sr4x,
    dependencies=[gemma2_2b_encoder],
)
```

- In `STARTER_MODELS` eintragen.
- Es gibt im Repo zusätzlich `…flux2_distill_4step_2606/` (neuere 2kto4k-Variante).
  Optional als dritten Eintrag aufnehmen — sonst die `_2606`-Version bewusst weglassen.
- 128ch ist **eindeutig** → keine Verzeichnisnamen-Ambiguität (anders als SD3/Qwen).
- Test: `uv run --extra cuda python -c "import invokeai.backend.model_manager.starter_models as s; print(len(s.STARTER_MODELS))"`

---

## Abnahme-Checkliste
- [x] `flux2_pid_decode` Node lädt (Import + `@invocation`-Registrierung verifiziert). Voller
      E2E-Lauf `flux2_denoise → flux2_pid_decode` → Bild: **offen** (braucht Modelle + GPU).
- [x] Latent-Shape verifiziert (128ch / down 16): synthetisch bestätigt
      `[1,32,H/8,W/8] → pack → [1,128,H/16,W/16]`; Shape-Logging im Node ergänzt für Runtime-Check.
- [x] txt2img Fit + Native in Generate-Tab: Graph-Builder verdrahtet (`isFlux2`-Pfad), `pnpm lint:tsc` grün.
      Live-UI-Lauf: **offen**.
- [x] img2img Fit + Native im Canvas (Scale Before Processing = None): Graph-Builder + Readiness-Guards
      verdrahtet, `tsc` grün. Live-UI-Lauf: **offen**.
- [x] Starter-Models installierbar (Pydantic + Unique-Source-Assert grün; 2 Einträge FLUX.2 2K / 2Kto4K).
      Decoder-Combobox filtert base-aware (`getPidDecoderBaseForMainBase('flux2')='flux2'`); Live-Anzeige: **offen**.
- [ ] VRAM-Peak gemessen, `_PID_DECODE_WORKING_MEMORY_SCALING_CONSTANT` ggf. nachjustiert — **offen**
      (kein GPU-Lauf). Analyse: Reservierung skaliert rein mit Output-Pixeln, identisch zu FLUX.1
      (2048px-Output ≈ 4,3 GB), daher a priori keine Anpassung nötig; am gemessenen Peak nachjustieren.

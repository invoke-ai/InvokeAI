# PiD für Qwen-Image

> Voraussetzung: gemeinsame Vorarbeit aus [README.md](README.md) (Abschnitte A–G).
> **Voller Backend-Stack** nötig (neue Config + `_PER_BACKBONE` + Loader-Register + Node).

**Besonderheiten:**
- Qwen-Image-Latent = **16 Kanäle** → 16ch-**Ambiguität** mit FLUX/SD3 (wie SD3).
- Nur **2kto4k**-Checkpoint vorhanden → genau ein Starter-Model.
- Im Repo liegt zusätzlich `QwenImage_VAE_2d.pth` — i. d. R. **nicht** nötig
  (die Qwen-Image-VAE kommt aus dem Hauptmodell), nur zur Kenntnis.

---

## 1. Nodes

### 1.1 `backend/pid/decode.py` — `_PER_BACKBONE`
```python
BaseModelType.QwenImage: {
    "lq_latent_channels": 16,
    "latent_spatial_down_factor": 8,
},
```
⚠️ **Verifizieren:** Qwen-Image-VAE-Latent-Kanäle (16) und Spatial-Down-Factor (8)
gegen die echte Qwen-VAE-Config prüfen, bevor du den Node baust.

### 1.2 `backend/model_manager/configs/pid_decoder.py` — Config + Channel-Map + Filename
- 16ch-Set um Qwen erweitern:
  ```python
  16: {BaseModelType.Flux, BaseModelType.StableDiffusion3, BaseModelType.QwenImage},
  ```
- `_backbone_from_filename` um Qwen ergänzen:
  ```python
  if re.search(r"qwen[_-]?image|qwenimage", n):
      return BaseModelType.QwenImage
  ```
- Neue Config-Klasse:
  ```python
  class PiDDecoder_Checkpoint_QwenImage_Config(PiDDecoder_Checkpoint_Config_Base, Config_Base):
      """PiD decoder for the Qwen-Image backbone (16-channel latent)."""
      base: Literal[BaseModelType.QwenImage] = Field(default=BaseModelType.QwenImage)
      variant: PiDDecoderVariantType = Field(description="Resolution preset of the PiD decoder checkpoint.")
  ```

> **16ch-Ambiguität:** Genau wie SD3 (siehe [02-sd3.md](02-sd3.md) §1.1). Mit drei
> Kandidaten (FLUX/SD3/Qwen) im 16ch-Set wird die Namens-Disambiguierung noch
> wichtiger. **Dringende Empfehlung:** die Probe-Härtung „explizites `base`-Override
> vertrauen" umsetzen, sonst hängt die korrekte Erkennung am erhaltenen Ordnernamen
> (`…official_qwenimage_distill…`).

### 1.3 `backend/model_manager/configs/factory.py`
- Import + Annotated-Union-Eintrag für `PiDDecoder_Checkpoint_QwenImage_Config`.

### 1.4 `backend/model_manager/load/model_loaders/pid_decoder.py`
```python
@ModelLoaderRegistry.register(base=BaseModelType.QwenImage, type=ModelType.PiDDecoder, format=ModelFormat.Checkpoint)
```

### 1.5 `app/invocations/qwen_image_pid_decode.py` (NEU)
`z_image_pid_decode.py` als Vorlage:
- `@invocation("qwen_image_pid_decode", title="Latents to Image - Qwen-Image + PiD (4x SR)", tags=[…,"qwen-image"], …)`
- `BaseModelType.QwenImage` an `estimate_pid_decode_working_memory` und `PiDDecoder`.
- **Denormalisierung:** optionalen `vae: VAEField | None`-Input; scaling/shift aus
  der Qwen-Image-VAE-Config lesen. ⚠️ Fallback-Konstanten erst nach Verifikation
  hartkodieren (Qwen-VAE-Werte aus `Qwen/Qwen-Image-Edit-2511::vae/...` ablesen).

---

## 2. UI

### 2.1 Gemeinsame Frontend-Vorarbeit
[README.md](README.md): **B**, **C** (`'qwen_image_pid_decode'` in `ImageOutputNodes`),
**D** (`qwen-image` eigener Decoder-Base), **E** (Readiness im `qwen-image`-Zweig), **F**.

### 2.2 `buildQwenImageGraph.ts`
Knoten laut Wiring-Map: Loader `qwen_image_model_loader`, Denoise `qwen_image_denoise`, VAE-Decode `qwen_image_l2i`.

- **Guard** (PiD ≠ off): inpaint/outpaint sperren; Scale-Before-Processing off.
- **txt2img:** `g.deleteNode(qwenL2i.id)` + `addPidDecode({…, denoise: qwenDenoise, positivePrompt, seed, decodeNodeType:'qwen_image_pid_decode', vaeSource: <qwen-vae-quelle> })`.
- **img2img Fit / Native:** wie FLUX, `decodeNodeType:'qwen_image_pid_decode'`.
- **VAE-Quelle:** die Qwen-Image-VAE (aus `qwen_image_model_loader` bzw. separat
  konfigurierter VAE-Source — Qwen-GGUF braucht eine VAE-Source) als `vaeSource`.
- **Positive-Prompt-String:** `string`-Node mit Positive-Prompt sicherstellen →
  `qwen_image_pid_decode.prompt`.
- Qwen-Image-Edit (Referenzbild) + PiD: zunächst nur Standard-txt2img/img2img
  abdecken; Edit-Mode separat prüfen.

### 2.3 Gating-Detail
`qwen-image` in `PID_SUPPORTED_BASES`, `getPidDecoderBaseForMainBase('qwen-image') = 'qwen-image'`.
`getGridSize('qwen-image', …)` (Grid 16) / `getOptimalDimension('qwen-image', …)` schon pidScale-aware.

---

## 3. Starter Models

Nur **2kto4k** vorhanden:
```python
pid_decoder_qwenimage_2kto4k = StarterModel(
    name="PiD Decoder Qwen-Image (2K to 4K)",
    base=BaseModelType.QwenImage,
    source="nvidia/PiD::checkpoints/PiD_res2kto4k_sr4x_official_qwenimage_distill_4step/model_ema_bf16.pth",
    description="NVIDIA PiD 4x super-resolution decoder for Qwen-Image latents, 2K-to-4K preset. ~5GB",
    type=ModelType.PiDDecoder,
    format=ModelFormat.Checkpoint,
    variant=PiDDecoderVariantType.Res2kTo4k_Sr4x,
    dependencies=[gemma2_2b_encoder],
)
```
- In `STARTER_MODELS` eintragen.
- **Test (wichtig wegen 16ch-Ambiguität):** real installieren und prüfen, ob als
  `base=qwen-image` erkannt (nicht FLUX). Falls nicht → Probe-Härtung (§1.2 / [02-sd3.md](02-sd3.md) §1.1).

---

## Abnahme-Checkliste
- [x] Latent-Kanäle/Down-Factor der Qwen-VAE **verifiziert** gegen `AutoencoderKLQwenImage`: `z_dim=16`,
      `LATENT_SCALE_FACTOR=8`. ⚠️ Zusätzlich entdeckt: Qwen normalisiert **per-Kanal**
      (`latents_mean`/`latents_std`, 16-Vektoren) statt skalar — der Node macht daher per-Kanal-Denorm
      (`z*std+mean`) + Squeeze der 5D-Temporaldim, exakt wie `qwen_image_l2i` (nicht die skalare z_image-Vorlage).
- [x] `_PER_BACKBONE[QwenImage]` (16/8) + Config `PiDDecoder_Checkpoint_QwenImage_Config` + 16ch-Set-Erweiterung +
      Filename-Heuristik + Factory-Union + Loader-Register + Node `qwen_image_pid_decode` angelegt; Imports grün.
- [x] `base=qwen-image`-Erkennung (16ch, jetzt **3-Wege** FLUX/SD3/Qwen): Probe-Test bestätigt — mit „qwenimage"
      im Ordnernamen → MATCH; bei stillem Namen fängt die **bestehende 16ch-Härtung** (expliziter base-Override,
      vom Starter-Install gesendet) den Fall ab → MATCH; SD3-cfg + qwenimage-Name → korrekt no-match.
      Realer 5-GB-HF-Install: **nicht ausgeführt**.
- [x] `build_pid_net(QwenImage)` baut, `lq_proj.latent_proj.0` erwartet 16 In-Channels; 5D→4D-Squeeze getestet;
      wm 4,19 GB @2048px. Voller E2E-Lauf → Bild: **offen** (GPU+Modelle).
- [x] txt2img Fit + Native; img2img Fit + Native: `buildQwenImageGraph.ts` verdrahtet (PiD-Guard, Qwen-VAE als
      `vaeSource`), `tsc`/`eslint`/`knip`/`dpdm` grün. Live-UI-Lauf: **offen**.
- [ ] VRAM-Peak gemessen — **offen** (kein GPU-Lauf; Schätzung ~4,19 GB @2048px, = FLUX.1).
- Hinweis: **Edit-Mode (Referenzbild) + PiD** nicht separat abgedeckt — der Standard-txt2img/img2img-Pfad deckt
  es strukturell mit ab (Ref-Bilder sind orthogonale Conditioning-Edges), aber ungetestet, wie im Plan vorgesehen.

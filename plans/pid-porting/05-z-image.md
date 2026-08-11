# PiD für Z-Image (+ Z-Image Turbo)

> Voraussetzung: gemeinsame Vorarbeit aus [README.md](README.md) (Abschnitte A–G).
> **Kein neues Backend.** Z-Image teilt sich die 16-Kanal-FLUX-VAE und nutzt daher
> den **FLUX-PiD-Decoder**. Node existiert bereits → nur **UI** + ein **Starter-Pointer**.

**Status der Bausteine:**
- Decode-Node `z_image_pid_decode` ✅ (`app/invocations/z_image_pid_decode.py`)
  - nutzt `backbone=BaseModelType.Flux`, liest VAE scaling/shift aus optionalem `vae`-Input
    (Fallback 0.3611 / 0.1159).
- Eigene Config/Loader/`_PER_BACKBONE` **nicht** nötig — der Decoder ist ein
  **FLUX**-PiD-Decoder-Modell.
- Z-Image **Turbo** und **Base** teilen dieselbe VAE → derselbe Node/Decoder für beide Varianten.

---

## 1. Nodes

Kein neuer Node. **Nur verifizieren:**
- `z_image_denoise → z_image_pid_decode` (mit der Z-Image-VAE am `vae`-Input)
  liefert ein 4×-Bild.
- Der Node nimmt einen `PiDDecoderField`, der auf ein **FLUX**-Base-PiD-Decoder-Modell
  zeigt (z. B. das bereits vorhandene Starter-Model „PiD Decoder FLUX (2K)").

---

## 2. UI

> **Wichtigster Unterschied zu allen anderen:** Der Decoder-Filter muss auf
> **`base === 'flux'`** zeigen, nicht `'z-image'` — Z-Image hat keine eigenen
> PiD-Checkpoints. Das deckt `getPidDecoderBaseForMainBase('z-image') = 'flux'`
> aus [README.md](README.md) §D bereits ab.

### 2.1 Gemeinsame Frontend-Vorarbeit
[README.md](README.md): **B**, **C** (`'z_image_pid_decode'` in `ImageOutputNodes`),
**D** (Gating: `z-image` in `PID_SUPPORTED_BASES`; **Decoder-Base = flux**),
**E** (Readiness im `z-image`-Zweig), **F**.

### 2.2 `buildZImageGraph.ts`
Knoten laut Wiring-Map: Loader `z_image_model_loader`, Denoise `z_image_denoise`, VAE-Decode `z_image_l2i`.

- **Guard** (PiD ≠ off): inpaint/outpaint sperren; Scale-Before-Processing off.
- **txt2img:** `g.deleteNode(zL2i.id)` + `addPidDecode({…, denoise: zDenoise, positivePrompt, seed, decodeNodeType:'z_image_pid_decode', vaeSource: <z-image-vae-quelle> })`.
- **img2img Fit / Native:** wie FLUX, `decodeNodeType:'z_image_pid_decode'`.
- **VAE-Quelle:** unbedingt die **Z-Image-VAE** als `vaeSource` durchreichen —
  der Node liest daraus scaling/shift (sonst Fallback-Konstanten, evtl. ungenau).
- **Positive-Prompt-String:** `string`-Node mit Positive-Prompt → `z_image_pid_decode.prompt`.

### 2.3 Turbo-Besonderheit
Z-Image Turbo läuft mit wenigen Steps / ohne CFG. Das betrifft nur die **Generation**,
nicht den PiD-Decode (PiD-Steps sind separat, Default 4). Keine Sonderbehandlung nötig.

### 2.4 Gating-Detail
`z-image` in `PID_SUPPORTED_BASES`. Im `PidSettings`-Combobox erscheinen damit die
**FLUX**-PiD-Decoder, wenn ein Z-Image-Hauptmodell aktiv ist — das ist beabsichtigt.
Optional: Label/Hinweis ergänzen („nutzt FLUX-PiD-Decoder"), damit Nutzer nicht
verwirrt sind, dass kein „Z-Image"-Decoder gelistet ist.
`getGridSize('z-image', …)` (Grid 16) / `getOptimalDimension('z-image', …)` schon pidScale-aware.

---

## 3. Starter Models

**Kein neues Decoder-Modell.** Z-Image nutzt den bereits eingetragenen FLUX-Decoder
(„PiD Decoder FLUX (2K)" / „… (2K to 4K)") + den gemeinsamen Gemma-2-Encoder.

Optional, zur Auffindbarkeit:
- Im `zimage_bundle` (in `starter_models.py`) auf die FLUX-PiD-Einträge **verweisen**
  bzw. sie aufnehmen, damit „Z-Image installieren" den passenden Decoder mitbringt:
  ```python
  zimage_bundle: list[StarterModel] = [
      z_image_turbo_quantized,
      z_image_qwen3_encoder_quantized,
      z_image_controlnet_union,
      z_image_controlnet_tile,
      flux_vae,
      # PiD (optional): nutzt den FLUX-Decoder + shared Gemma-2
      # pid_decoder_flux_2k,   # ggf. einkommentieren, wenn PiD im Z-Image-Bundle gewünscht
  ]
  ```
  (Abwägen wie bei FLUX: PiD+Gemma ~10GB; eher optional lassen.)
- Keine `starter_models.py`-Pflichtänderung nötig, wenn man PiD individuell installiert.

---

## Abnahme-Checkliste
- [x] `z_image_pid_decode` lädt (Import verifiziert; Backend war fertig). Graph reicht die **Z-Image-VAE**
      (`vaeSource: modelLoader` → `z_image_pid_decode.vae`, Node in `PID_DECODE_NODES_WITH_VAE_INPUT`) durch,
      also werden scaling/shift aus der echten VAE gelesen (nicht die Fallbacks). Voller E2E-Lauf → 4×-Bild: **offen** (GPU).
- [x] PidSettings zeigt bei aktivem Z-Image-Modell die **FLUX**-PiD-Decoder: `getPidDecoderBaseForMainBase('z-image')='flux'`
      → Decoder-Filter zeigt `base==='flux'`. Gating via `getIsPidSupportedBase`. Live-Anzeige: **offen**.
- [x] txt2img Fit + Native; img2img Fit + Native: `buildZImageGraph.ts` verdrahtet (PiD-Guard, `z_image_denoise`
      trägt width/height → kein noise-Node nötig), `tsc`/`eslint`/`knip`/`dpdm` grün. Live-UI-Lauf: **offen**.
- [x] Turbo + PiD: keine Sonderbehandlung — PiD-Steps sind separat (Default 4), Turbo betrifft nur die Generation.
- [ ] (optional) Bundle-/Label-Hinweis auf FLUX-Decoder-Nutzung: **bewusst ausgelassen** (rein optional; PiD nicht
      ins ~10-GB-Z-Image-Bundle aufgenommen, wie bei FLUX). Keine `starter_models.py`-Änderung.

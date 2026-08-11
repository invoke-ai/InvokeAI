# PiD für SDXL

> Voraussetzung: gemeinsame Vorarbeit aus [README.md](README.md) (Abschnitte A–G).
> **Voller Backend-Stack** nötig (neue Config + `_PER_BACKBONE` + Loader-Register + Node).

**Besonderheiten:**
- SDXL-Latent = **4 Kanäle** (alle anderen 16/128) → neuer Channel-Eintrag.
- Nur **2kto4k**-Checkpoint vorhanden (kein 2k) → genau ein Starter-Model.
- SDXL hat einen Negative-Prompt; PiD konditioniert nur auf den Positive-Prompt.

---

## 1. Nodes

### 1.1 `backend/pid/decode.py` — `_PER_BACKBONE` erweitern
```python
BaseModelType.StableDiffusionXL: {
    "lq_latent_channels": 4,
    "latent_spatial_down_factor": 8,
},
```
Damit greifen `build_pid_net`, `estimate_pid_decode_working_memory` und `PiDDecoder`
automatisch für SDXL.

### 1.2 `backend/model_manager/configs/pid_decoder.py` — Config + Channel-Map
- Channel-Map ergänzen (4ch ist eindeutig SDXL):
  ```python
  _LATENT_CHANNELS_TO_BASES: dict[int, set[BaseModelType]] = {
      4: {BaseModelType.StableDiffusionXL},
      16: {BaseModelType.Flux, BaseModelType.StableDiffusion3},
      128: {BaseModelType.Flux2},
  }
  ```
- Filename-Heuristik um SDXL ergänzen (`_backbone_from_filename`):
  ```python
  if re.search(r"\bsdxl\b|sdxl", n):
      return BaseModelType.StableDiffusionXL
  ```
  (vor dem `return None`; Reihenfolge unkritisch, da 4ch ohnehin eindeutig)
- Neue Config-Klasse:
  ```python
  class PiDDecoder_Checkpoint_SDXL_Config(PiDDecoder_Checkpoint_Config_Base, Config_Base):
      """PiD decoder for the SDXL backbone (4-channel latent)."""
      base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)
      variant: PiDDecoderVariantType = Field(description="Resolution preset of the PiD decoder checkpoint.")
  ```

### 1.3 `backend/model_manager/configs/factory.py` — Union ergänzen
- Import: `PiDDecoder_Checkpoint_SDXL_Config`
- In die Annotated-Union (bei den anderen `PiDDecoder_Checkpoint_*`):
  ```python
  Annotated[PiDDecoder_Checkpoint_SDXL_Config, PiDDecoder_Checkpoint_SDXL_Config.get_tag()],
  ```

### 1.4 `backend/model_manager/load/model_loaders/pid_decoder.py` — Register
```python
@ModelLoaderRegistry.register(base=BaseModelType.StableDiffusionXL, type=ModelType.PiDDecoder, format=ModelFormat.Checkpoint)
```
(zur bestehenden Decorator-Kette hinzufügen)

### 1.5 `app/invocations/sdxl_pid_decode.py` (NEU)
`z_image_pid_decode.py` als Vorlage (es liest VAE-Konstanten zur Laufzeit):
- `@invocation("sdxl_pid_decode", title="Latents to Image - SDXL + PiD (4x SR)", tags=[…,"sdxl"], …)`
- `BaseModelType.StableDiffusionXL` an `estimate_pid_decode_working_memory` und `PiDDecoder`.
- **Denormalisierung:** SDXL-VAE hat `scaling_factor = 0.13025`, **kein** Shift (0.0).
  Optionalen `vae: VAEField | None`-Input anbieten und die Konstanten bevorzugt aus
  der VAE-Config lesen (Fallback 0.13025 / 0.0).
- ⚠️ SDXL-Latent ist 4ch — `lq_proj.latent_proj.0` des SDXL-Checkpoints erwartet 4
  In-Channels. Wenn `build_pid_net` mit `lq_latent_channels=4` instanziiert wird,
  passt das. Bei `load_state_dict`-Mismatch die `_PER_BACKBONE`-Werte prüfen.

---

## 2. UI

### 2.1 Gemeinsame Frontend-Vorarbeit
[README.md](README.md): **B**, **C** (`'sdxl_pid_decode'` in `ImageOutputNodes`),
**D** (`sdxl` eigener Decoder-Base), **E** (Readiness im `sdxl`-Zweig), **F**.

### 2.2 `buildSDXLGraph.ts`
Knoten laut Wiring-Map: Loader `sdxl_model_loader`, Denoise `denoise_latents`, VAE-Decode `l2i`.

- **Guard** (PiD ≠ off): inpaint/outpaint sperren; Scale-Before-Processing off.
- **txt2img:** `g.deleteNode(l2i.id)` + `addPidDecode({…, denoise, positivePrompt, seed, decodeNodeType:'sdxl_pid_decode', vaeSource: <vae-quelle> })`.
  - `denoise: Invocation<DenoiseLatentsNodes>` deckt `denoise_latents` ab (generische Signatur aus README-B).
- **img2img Fit / Native:** wie bei FLUX (`buildPidDecodeChain` / `addPidImageToImageNative`) mit `decodeNodeType:'sdxl_pid_decode'`.
- **VAE-Quelle:** SDXL hat eine eigene VAE (separates `vae_loader` oder aus dem
  `sdxl_model_loader`). Diese als `vaeSource` an die Chain geben, damit der Node
  scaling/shift lesen kann.
- **Positive-Prompt-String:** sicherstellen, dass ein `string`-Node mit dem
  Positive-Prompt existiert (SDXL nutzt SDXL-spezifisches Conditioning) und in
  `sdxl_pid_decode.prompt` geht.
- **Refiner:** Falls ein SDXL-Refiner aktiv ist, ist die PiD-Substitution mit dem
  Refiner-Pfad zu klären. **Empfehlung:** PiD + Refiner zunächst ausschließen
  (Guard: wenn `pidMode !== 'off'` und Refiner aktiv → `toast.pidUnsupportedMode`
  oder eigener Hinweis), später separat lösen.

### 2.3 Gating-Detail
`sdxl` in `PID_SUPPORTED_BASES`, `getPidDecoderBaseForMainBase('sdxl') = 'sdxl'`.
`getGridSize('sdxl', …)` (Grid 8, native ×4 → 32) / `getOptimalDimension('sdxl', …)` schon pidScale-aware.

---

## 3. Starter Models

Nur **2kto4k** vorhanden:
```python
pid_decoder_sdxl_2kto4k = StarterModel(
    name="PiD Decoder SDXL (2K to 4K)",
    base=BaseModelType.StableDiffusionXL,
    source="nvidia/PiD::checkpoints/PiD_res2kto4k_sr4x_official_sdxl_distill_4step/model_ema_bf16.pth",
    description="NVIDIA PiD 4x super-resolution decoder for SDXL latents, 2K-to-4K preset. ~5GB",
    type=ModelType.PiDDecoder,
    format=ModelFormat.Checkpoint,
    variant=PiDDecoderVariantType.Res2kTo4k_Sr4x,
    dependencies=[gemma2_2b_encoder],
)
```
- In `STARTER_MODELS` eintragen (4ch eindeutig → keine Ambiguitäts-Sorge).
- Test: `uv run --extra cuda python -c "import invokeai.backend.model_manager.starter_models"`

---

## Abnahme-Checkliste
- [x] `_PER_BACKBONE[SDXL]` (4ch/down8) + Config `PiDDecoder_Checkpoint_SDXL_Config` + Channel-Map[4] +
      Filename-Heuristik + Factory-Union + Loader-Register + Node `sdxl_pid_decode` angelegt; Backend-Imports grün.
- [x] SDXL-Erkennung `base=sdxl`: 4ch ist **eindeutig** → Config-Probe erkennt SDXL auch bei stillem/flachem
      Ordnernamen (getestet: 4ch→SDXL match; 16ch & FLUX-cfg korrekt abgewiesen). Realer 5-GB-HF-Install: **offen**.
- [x] 4ch-Latent verifiziert: `build_pid_net(SDXL)` baut, `lq_proj.latent_proj.0` erwartet **4 In-Channels**
      (Gewicht `[512,4,3,3]`) → echtes 4ch-Checkpoint lädt ohne Shape-Mismatch. Voller E2E-Lauf
      `denoise_latents → sdxl_pid_decode` → 4×-Bild: **offen** (braucht GPU+Modelle).
- [x] txt2img Fit + Native; img2img Fit + Native: `buildSDXLGraph.ts` verdrahtet (PiD-Guard, `noise`-Node-Sizing,
      VAE als `vaeSource`), `tsc`/`eslint`/`knip`/`dpdm` grün. Live-UI-Lauf: **offen**.
- [x] Refiner-Interaktion **gesperrt** (nicht gelöst): PiD + aktiver Refiner → Guard im Graph-Builder
      (`toast.pidUnsupportedMode`) **und** Readiness-Grund (`pidIncompatibleWithRefiner`, neuer i18n-Key).
- [ ] VRAM-Peak gemessen — **offen** (kein GPU-Lauf). Analyse: 2048px-Output ≈ 4,19 GB Reservierung,
      identisch zu FLUX.1; `_PID_DECODE_WORKING_MEMORY_SCALING_CONSTANT` a priori unverändert.

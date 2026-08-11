# PiD (Pixel Diffusion Decoder) — Integrationsplan für InvokeAI

> **Status:** Analyse / Planung
> **Quelle:** https://github.com/nv-tlabs/PiD · https://huggingface.co/nvidia/PiD
> **Sprache:** Deutsch (Plan), Englisch (Code/Identifier)

---

## 1. Was ist PiD?

**Pixel Diffusion Decoder** — ersetzt den klassischen VAE-Decoder durch ein kleines
Diffusionsmodell, das in einem Pass das Latent **dekodiert und gleichzeitig hochskaliert**
(typisch 4×). Effekt: Decode + Super-Resolution in einem Schritt.

**Unterstützte Backbones** (in dieser Integration relevant):

| Backbone | PiD-Decoder vorhanden | Bemerkung |
|---|---|---|
| FLUX1-dev | ✅ `…_flux_distill_4step` | |
| FLUX.2-dev | ✅ `…_flux2_distill_4step` | 32-channel BN-VAE |
| SD3 medium | ✅ `…_sd3_distill_4step` | |
| Z-Image / Z-Image-Turbo | ✅ via FLUX-Decoder | Z-Image teilt FLUX1-VAE |

DINOv2-/SigLIP-Varianten existieren auch, sind für InvokeAI aber irrelevant.

---

## 2. Eckdaten

| | |
|---|---|
| Code-Lizenz | Apache 2.0 |
| Modell-Lizenz | NSCLv1 (non-commercial / research) |
| Vertriebsweg | Kein PyPI, nur GitHub-Source + HF-Weights |
| Decoder-Größe | ~2 GB pro Backbone + Auflösungsvariante |
| `diffusers` Bedarf | `>=0.37` — InvokeAI pinnt `==0.37.0` ✅ |
| `transformers` Bedarf | `>=4.57` — InvokeAI hat `>=4.56.0` ⚠ (minor bump) |
| Custom CUDA-Kernels | Keine |
| Steps pro Decode | 4 (distilled) |

**Lizenz-Einschätzung:** Da InvokeAI ein reines Community-OSS-Projekt ist, ist NSCLv1
konsistent mit dem bereits unterstützten FLUX-dev-Modell (das ebenfalls non-commercial
ist). Die kommerzielle Nutzungsbeschränkung verschiebt sich auf den Endnutzer.

---

## 3. Architektur-Befunde InvokeAI

### Bestehende Decode-Invocations (Vorlagen)

| Backbone | Datei |
|---|---|
| FLUX | [invokeai/app/invocations/flux_vae_decode.py](invokeai/app/invocations/flux_vae_decode.py) |
| FLUX.2 | [invokeai/app/invocations/flux2_vae_decode.py](invokeai/app/invocations/flux2_vae_decode.py) |
| SD3 | [invokeai/app/invocations/sd3_latents_to_image.py](invokeai/app/invocations/sd3_latents_to_image.py) |
| Z-Image | [invokeai/app/invocations/z_image_latents_to_image.py](invokeai/app/invocations/z_image_latents_to_image.py) |

### Model-Manager

- Config-Klassen in [invokeai/backend/model_manager/configs/](invokeai/backend/model_manager/configs/)
- Auto-Detection via State-Dict-Heuristik (siehe [vae.py](invokeai/backend/model_manager/configs/vae.py) als Vorlage: `_is_flux2_vae`, `_is_qwen_image_vae`, `_has_anima_vae_keys`)
- Loader decorator-basiert: `@ModelLoaderRegistry.register(base=..., type=..., format=...)`
- Discriminator-Schema: `{type}.{format}.{base}[.{variant}]` — Variant nur im Tag wenn Default-Wert existiert

---

## 4. Konzept

**PiD ist kein „besserer VAE", sondern ein optionales Hi-Quality-Decode-+-Upscale-Modul.**
Es ersetzt nicht den regulären VAE-Decode in jedem Kontext (img2img / Inpainting brauchen
weiterhin den klassischen Encoder-Roundtrip), sondern bietet eine **alternative
Final-Render-Variante** im Workflow.

Konkurriert UX-mäßig eher mit ESRGAN / SUPIR / Refinern als mit dem normalen VAE.

---

## 5. Foundation-Phasen (einmalig, alle Modelle)

### Phase A — PiD vendoren & aufräumen
**Aufwand:** 3–4 Tage · **blockiert:** alles

> ⚠️ **Revision nach Code-Inspektion:** Der ursprünglich angesetzte Aufwand von
> 1–1.5 T war zu niedrig. PiDs Source hängt tief an NVIDIAs **Imaginaire**-Framework
> (lazy_config, distributed, logging, s3_utils, checkpointer). Direkte
> Entkopplung wäre massiv — pragmatischer Weg: Imaginaire-Subset mit vendoren
> und punktuell strippen (Hard-Deps wie boto3/wandb/fvcore raus).

PiD ist nicht auf PyPI. Subset nach `invokeai/backend/pid/` vendoren
(analog `invokeai/backend/flux/`).

**Was rein muss (`invokeai/backend/pid/_src/`):**
- `models/`: `pid_model.py`, `pid_distill_model.py`, `pixeldit_model.py`, `utils.py`
- `modules/`: `conditioner.py`
- `networks/`: `pid_net.py`, `pixeldit_official.py`, `lq_projection_2d.py`
- `utils/`: `context_parallel.py`, `model_loader.py`
- Aus `inference/`: `pipeline_registry.py` (subset), `checkpoint_registry.py`
- Aus `inference_utils.py`: nur Decode-Pfad → neue Datei `decode.py` mit Minimal-API

**Imaginaire-Subset (`invokeai/backend/pid/_ext/imaginaire/`):**
- `lazy_config/` (LazyCall, LazyDict, instantiate)
- `model.py` (ImaginaireModel Basisklasse)
- `utils/`: `log.py`, `misc.py`, `count_params.py`, `distributed.py` (gestrippt)
- `flags.py`, `types/denoise_prediction.py`, `config.py` (Subset)

**Nicht rein:**
- `tokenizers/*` — InvokeAI hat seine eigenen VAEs ✗
- `configs/*` — Hydra-Configs durch Python-Dataclasses ersetzen ✗
- `_demo_*.py`, `from_*.py`, `create_dataset.py`, `rae_*.py`, `scale_rae_*.py` ✗
- `pixeldit_official.py` Selektiv (großes File, nur was PidNet erbt) ⚠
- Imaginaire `checkpointer/`, `trainer.py`, `visualize/`, `easy_io/`, `s3_utils.py` ✗

**Weitere Schritte:**
- Imports umschreiben: `pid.*` → `invokeai.backend.pid.*`, `pid._ext.imaginaire.*` → `invokeai.backend.pid._ext.imaginaire.*`
- Hydra `LazyDict`-Configs → Python `dataclass`
- Hard-Deps strippen / Lazy-Import: `boto3`, `wandb`, `fvcore`, `iopath`, `loguru` (→ stdlib logging)
- **`transformers>=4.56.0`** bleibt — PiDs `environment.yml` pinnt selbst nicht (Kommentar: "4.x und 5.x"), nur SigLIP/Scale-RAE bräuchte 4.57.x, das nutzen wir nicht
- **Keine neuen Hard-Dependencies** — `omegaconf` initial erwogen, aber `instantiate.py` stdlib-only umgeschrieben, sodass auch dieser Dep wegfällt
- License-Header in jeder vendored Datei beibehalten (sind alle `SPDX: Apache-2.0`)
- `LICENSE-PiD.txt` im Repo-Root ergänzt (analog zu `LICENSE-SD1+SD2.txt`)

**Risiken:**
- Imaginaire-Subset könnte transient mehr Imports ziehen als erwartet
- `pixeldit_official.py` (65 KB) ggf. nicht klein machbar — komplette Datei vendoren

---

### Phase B — Model-Manager: neuer Modelltyp
**Aufwand:** 1 Tag · **blockiert von:** A

Neuer Modelltyp + Auto-Detection, analog zur VAE-Erkennung in
[invokeai/backend/model_manager/configs/vae.py](invokeai/backend/model_manager/configs/vae.py).

**Änderungen:**

- [taxonomy.py](invokeai/backend/model_manager/configs/taxonomy.py):
  - `ModelType.PiDDecoder = "pid_decoder"`
  - `PiDVariantType`-Enum:
    `FLUX_2K`, `FLUX_2K_TO_4K`, `FLUX2_2K`, `FLUX2_2K_TO_4K`, `SD3_2K`, `SD3_2K_TO_4K`
    (1:1 Mapping auf NVIDIAs Checkpoint-Namen)
- Neue Datei `invokeai/backend/model_manager/configs/pid.py`:
  - `PiD_Checkpoint_Config` mit `variant: PiDVariantType` (ohne Default → nicht im Discriminator-Tag)
  - `from_model_on_disk()` mit State-Dict-Heuristik
    (erwartete Layer-Namen aus PiDs DiT-Decoder + Channel-Count zur Backbone-Identifikation)
- [factory.py](invokeai/backend/model_manager/configs/factory.py): in `AnyModelConfig`-Union aufnehmen
- Neuer Loader `invokeai/backend/model_manager/load/model_loaders/pid.py`:
  ```python
  @ModelLoaderRegistry.register(
      base=BaseModelType.Any,
      type=ModelType.PiDDecoder,
      format=ModelFormat.Checkpoint,
  )
  class PiDDecoderLoader(ModelLoader): ...
  ```

---

### Phase B.5 — Gemma-2-2b-it Text Encoder Integration
**Aufwand:** 0.5–1 Tag · **blockiert von:** A · **blockiert:** C

> ⚠️ Bei der Inspektion von `pixeldit_model.py` aufgefallen: PiD ist
> **text-conditional** und nutzt einen eigenen Text-Encoder (Gemma-2-2b-it,
> 2304-dim Caption Channels, ~5 GB bf16). Das ist nicht optional — das Modell
> wurde so trainiert. InvokeAIs vorhandene T5/CLIP/Qwen3-Encoder sind
> inkompatibel (andere Channel-Dimensionen).

Neuer ModelType `Gemma2Encoder` analog zu `Qwen3Encoder`.

**Änderungen:**
- [taxonomy.py](invokeai/backend/model_manager/taxonomy.py):
  - `ModelType.Gemma2Encoder = "gemma2_encoder"`
  - `ModelFormat.Gemma2Encoder = "gemma2_encoder"`
- Neue Datei `invokeai/backend/model_manager/configs/gemma2_encoder.py`:
  - `Gemma2Encoder_Diffusers_Config` (diffusers-Folder mit Gemma-2-2b-it Gewichten)
  - Probing über `config.json` Architecture-Marker (`Gemma2ForCausalLM`)
- [factory.py](invokeai/backend/model_manager/configs/factory.py): Union eintragen
- Neuer Loader `invokeai/backend/model_manager/load/model_loaders/gemma2_encoder.py`:
  - Nutzt `AutoModelForCausalLM.from_pretrained(...).get_decoder()` (PiD nutzt nur den Decoder-Subtree)
  - Optional: BnB-4bit für VRAM-Einsparung

**Lizenz-Hinweis:** Gemma 2 hat eigene Google-Terms (Gemma Terms of Use).
PiD nutzt den Mirror `Efficient-Large-Model/gemma-2-2b-it` (gleiche Gewichte
wie `google/gemma-2-2b-it`, aber ohne Gated-Access).

---

### Phase C — Backend-Decoder-Wrapper
**Aufwand:** 0.5–1 Tag · **blockiert von:** A, B.5 · **parallel zu:** B

Gemeinsame Helper-Klasse für alle modell-spezifischen Invocations.

```
invokeai/backend/pid/
├── __init__.py
├── decode.py          # PiDDecoder.decode(latents, *, scale, steps, dtype, device)
└── registry.py        # backbone -> Latent-Shape/Channels
```

**API-Skizze:**

```python
class PiDDecoder:
    def __init__(self, model: nn.Module, backbone: str): ...

    @torch.no_grad()
    def decode(
        self,
        latents: Tensor,
        *,
        num_inference_steps: int = 4,
        scale: int = 4,
        cfg_scale: float = 1.0,
    ) -> Tensor:
        """Returns image tensor in [-1, 1], shape (B, 3, H*scale, W*scale)."""
```

---

### Phase D — Frontend & Workflow
**Aufwand:** 1–2 Tage · **blockiert von:** A

- **Workflow-Editor:** Neue Invocations erscheinen automatisch im Node-Picker (InvokeAI-Mechanik) → kein Frontend-Code für Workflow-Mode
- **Linear-UI:** Erstmal **nicht** integrieren — bewusste Workflow-Entscheidung
- **Doku:** `docs/features/pid-decoder.md` (kurz) inkl. Lizenz-Hinweis NSCLv1

---

## 6. Per-Modell-Phasen (parallelisierbar nach Foundation)

Pro Backbone eine neue Invocation-Datei. Aufbau identisch zur jeweiligen
VAE-Decode-Vorlage, mit folgenden generellen Änderungen:

```python
class XxxPiDDecodeInvocation(BaseInvocation, WithMetadata, WithBoard):
    latents: LatentsField = InputField(...)
    pid_decoder: PiDDecoderField = InputField(...)   # statt VAEField
    scale: Literal[2, 4] = InputField(default=4)
    num_inference_steps: int = InputField(default=4)

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        latents = context.tensors.load(self.latents.latents_name)
        decoder_info = context.models.load(self.pid_decoder.model)

        with decoder_info.model_on_device(...) as (_, decoder):
            img = PiDDecoder(decoder, backbone="...").decode(
                latents,
                scale=self.scale,
                num_inference_steps=self.num_inference_steps,
            )
        ...
```

### 6.1 Z-Image (Pilot)
**Aufwand:** 0.5 Tag · **Datei:** `invokeai/app/invocations/z_image_pid_decode.py`

- Z-Image nutzt FLUX1-VAE → **FLUX-PiD-Decoder** verwenden
- `id = "z_image_pid_decode"`, `classification=Classification.Prototype`
- **Warum Pilot:** Z-Image ist [bereits Prototype-Status](invokeai/app/invocations/z_image_latents_to_image.py#L36) → niedrigste Regressions-Gefahr

### 6.2 FLUX
**Aufwand:** 0.5 Tag · **Datei:** `invokeai/app/invocations/flux_pid_decode.py`

- Vorlage: [flux_vae_decode.py](invokeai/app/invocations/flux_vae_decode.py)
- Checkpoint-Varianten: `PiDVariantType.FLUX_2K` oder `FLUX_2K_TO_4K`
- Latent-Channels: 16

### 6.3 FLUX.2
**Aufwand:** 0.5 Tag · **Datei:** `invokeai/app/invocations/flux2_pid_decode.py`

- Vorlage: [flux2_vae_decode.py](invokeai/app/invocations/flux2_vae_decode.py)
- Checkpoint-Varianten: `PiDVariantType.FLUX2_2K` / `FLUX2_2K_TO_4K`
- Latent-Channels: 32

### 6.4 SD3
**Aufwand:** 0.5 Tag · **Datei:** `invokeai/app/invocations/sd3_pid_decode.py`

- Vorlage: [sd3_latents_to_image.py](invokeai/app/invocations/sd3_latents_to_image.py)
- Checkpoint-Varianten: `PiDVariantType.SD3_2K` / `SD3_2K_TO_4K`
- Latent-Channels: 16

### 6.5 PiD Upscaler (Schluss-Step)
**Aufwand:** 0.5–1 Tag · **Datei:** `invokeai/app/invocations/pid_upscale.py`

Eigenständige Upscale-Invocation (kein Generator-Latent als Input, sondern ein **Bild**).
Fluss: `Image → InvokeAI VAE-Encode → Latent → PiD-Decode → Image (4× SR)`.

- Inputs:
  - `image: ImageField`
  - `vae: VAEField` (FLUX/FLUX.2/SD3-VAE — Backbone-passend)
  - `pid_decoder: PiDDecoderField`
  - `scale: Literal[2, 4]`
  - `num_inference_steps: int = 4`
- Verkabelt InvokeAIs vorhandenen VAE-Encoder direkt mit dem PiD-Decoder
- Konkurriert UX-mäßig mit ESRGAN/SUPIR
- **Warum am Schluss:** Foundation (Phase A-C) + Decoder-Pfade (6.1-6.4) müssen stehen;
  Upscale ist nur ein dünner Wrapper drüber.

---

## 7. Test- & Validierungs-Phase
**Aufwand:** 1–2 Tage

- **VRAM-Profile** messen: pro Backbone working-memory Schätzung kalibrieren
  (siehe [estimate_vae_working_memory_flux](invokeai/backend/util/vae_working_memory.py))
- **Qualitäts-Vergleich:** Side-by-side VAE-Decode vs. PiD-Decode für 5–10 Prompts pro Backbone
- **Edge Cases:** kleine Latents (256×256), große (1024×1024), Batch > 1
- **Reference-Smoke-Test:** PiDs eigene `from_clean_*.py`-Skripte einmal laufen lassen,
  Output mit unserer Implementation vergleichen

---

## 8. Gesamtübersicht

| Phase | Aufwand | Blockiert von | Parallelisierbar mit |
|---|---|---|---|
| A: Vendoring | **3–4 T** ⚠ | — | — |
| B: Model-Manager | 1 T | A | C |
| C: Backend-Wrapper | 0.5–1 T | A | B |
| D: Frontend | 1–2 T | A | später |
| 6.1 Z-Image | 0.5 T | A, B, C | 6.2 / 6.3 / 6.4 |
| 6.2 FLUX | 0.5 T | A, B, C | 6.1 / 6.3 / 6.4 |
| 6.3 FLUX.2 | 0.5 T | A, B, C | 6.1 / 6.2 / 6.4 |
| 6.4 SD3 | 0.5 T | A, B, C | 6.1 / 6.2 / 6.3 |
| 6.5 Upscaler | 0.5–1 T | A, B, C, 6.x | — |
| 7. Tests | 1–2 T | alle | — |

**Gesamtaufwand:** ~10–13 Personentage. Solo realistisch **2.5–3 Wochen** inkl. Tests,
Upscale-Node und kleinerer Frontend-Polish.

---

## 9. Risiken

| Risiko | Wahrscheinlichkeit | Impact | Gegenmaßnahme |
|---|---|---|---|
| Hydra/Config-Entkopplung größer als angesetzt | mittel | mittel | Spike in Phase A, ggf. Phase A auf 2 T erhöhen |
| `inference_utils.py` monolithisch / verwoben | mittel | mittel | Decode-Pfad isolieren statt komplett zu portieren |
| Scheduler-Inkompatibilität | niedrig | mittel | PiDs eigenen Scheduler im `decode.py`-Wrapper kapseln |
| VRAM zu hoch auf 8 GB-Karten | mittel | niedrig | Sequential-Offload nutzen (vorhanden); 12 GB als Empfehlung |
| Lizenz-Beschwerden trotz OSS-Status | niedrig | niedrig | Klarer NSCLv1-Hinweis in Doku + Model-Card |

---

## 10. Empfohlener Start

1. **Phase A** als Spike auf eigenem Branch → Aufwand validieren
2. Wenn A < 2 Tage: **A + B + C + 6.1 (Z-Image-Pilot)** als erster PR
3. Wenn Z-Image-Pilot OK: FLUX / FLUX.2 / SD3 als Folge-PRs
4. Tests + Doku am Ende

# PiD-Porting: FLUX → weitere Base Models

Ziel: Den bestehenden FLUX-PiD-Support (Pixel Diffusion Decoder, 4× SR) auf
**FLUX.2 Klein (4B/9B), SD3, SDXL, Qwen-Image, Z-Image/Turbo** ausweiten.

Pro Model-Type gibt es eine eigene Datei, jeweils gegliedert in **Nodes → UI →
Starter Models**:

| Datei | Backbone | Decode-Node | Config/Loader | `_PER_BACKBONE` | Starter-Checkpoints |
|---|---|---|---|---|---|
| [01-flux2.md](01-flux2.md) | FLUX.2 Klein 4B/9B | **neu** `flux2_pid_decode` | ✅ vorhanden | ✅ vorhanden | 2k + 2kto4k (+`_2606`) |
| [02-sd3.md](02-sd3.md) | SD3 | ✅ `sd3_pid_decode` | ✅ vorhanden | ✅ vorhanden | 2k + 2kto4k |
| [03-sdxl.md](03-sdxl.md) | SDXL | **neu** `sdxl_pid_decode` | **neu** | **neu** (4ch) | **nur** 2kto4k |
| [04-qwen-image.md](04-qwen-image.md) | Qwen-Image | **neu** `qwen_image_pid_decode` | **neu** | **neu** (16ch) | **nur** 2kto4k |
| [05-z-image.md](05-z-image.md) | Z-Image (+Turbo) | ✅ `z_image_pid_decode` | reuse FLUX | reuse FLUX | reuse FLUX-Decoder |

> **Reihenfolge-Empfehlung:** SD3 und Z-Image zuerst (Node existiert schon → nur
> UI + Starter), dann FLUX.2 (Backend fast fertig), dann SDXL/Qwen (voller
> Backend-Stack inkl. neuer Config + `_PER_BACKBONE`).

Diese README beschreibt die **gemeinsame Infrastruktur**, die alle fünf teilen.
Jede Model-Datei verweist hierauf und beschreibt nur ihre Besonderheiten.

---

## Referenz: Wie der FLUX-Stack aufgebaut ist

**Backend (alles in `invokeai/`):**
- Decode-Pipeline: [`backend/pid/decode.py`](../../invokeai/backend/pid/decode.py)
  - `_PID_SR4X_BASE` (gemeinsame Netz-Hyperparams) + `_PER_BACKBONE` (Deltas pro Backbone)
  - `estimate_pid_decode_working_memory(latent, backbone)`, `build_pid_net`, `load_pid_decoder`, `PiDDecoder`, `encode_caption_for_pid`
- Decode-Nodes: `app/invocations/{flux,sd3,z_image}_pid_decode.py` (Muster, fast identisch)
- Loader-Nodes (generisch, für alle wiederverwendbar):
  - `app/invocations/pid_decoder_loader.py` → `PiDDecoderField`
  - `app/invocations/gemma2_encoder_loader.py` → `Gemma2EncoderField`
- Model-Configs: `backend/model_manager/configs/pid_decoder.py` (FLUX/FLUX2/SD3) + `gemma2_encoder.py`
- Model-Loader: `backend/model_manager/load/model_loaders/pid_decoder.py` (`@ModelLoaderRegistry.register` pro base)
- Config-Union: `backend/model_manager/configs/factory.py` (Annotated-Liste)
- Starter-Models: `backend/model_manager/starter_models.py`

**Frontend (alles in `invokeai/frontend/web/src/`):**
- State (global, schon vorhanden): `features/controlLayers/store/types.ts` (`zPidMode`, `pidMode`/`pidDecoderModel`/`gemma2EncoderModel`/`pidSteps`) + `paramsSlice.ts` (Reducer/Selektoren/Migration `_version: 4`)
- UI-Komponente: `features/parameters/components/Advanced/PidSettings.tsx`
- Gating: `features/settingsAccordions/components/GenerationSettingsAccordion/GenerationSettingsAccordion.tsx:124` (`isFLUX`)
- Graph-Chain: `features/nodes/util/graph/generation/addPidDecode.ts` (`buildPidDecodeChain`, `addPidDecode`, `addPidImageToImageNative`)
- Graph-Builder pro Base: `features/nodes/util/graph/generation/build{FLUX,SD3,SDXL,QwenImage,ZImage}Graph.ts`
- Dimension-Helfer (schon base+pidScale-aware): `features/parameters/util/optimalDimension.ts` (`getGridSize`, `getOptimalDimension`, `getPidScale`, `PID_SCALE`)
- Readiness: `features/queue/store/readiness.ts`
- Model-Hooks/Guards: `services/api/hooks/modelsByType.ts` (`usePiDDecoderModels`, `useGemma2EncoderModels`), `services/api/types.ts` (`isPiDDecoderModelConfig`, `isGemma2EncoderModelConfig`)
- i18n: `public/locales/en.json` (`modelManager.pid*`, `parameters.invoke.*pid*`, `popovers.pidMode`, `toast.pid*`)
- Node-Typ-Union: `features/nodes/util/graph/types.ts` (`ImageOutputNodes`)

---

## Gemeinsame Vorarbeit (einmal, danach von allen genutzt)

### A. Decode-Node-Muster (Backend)

Alle Decode-Nodes sind ~95 % identisch (siehe `flux`/`sd3`/`z_image`). Pro neuem
Backbone kopieren und nur anpassen:
1. `@invocation("<base>_pid_decode", title=…, tags=[…,"<base>"])`
2. den an `estimate_pid_decode_working_memory(...)` und `PiDDecoder(..., backbone=…)` übergebenen `BaseModelType`
3. die **Latent-Denormalisierung** (`z / scaling_factor + shift_factor`).
   - **Empfehlung für neue Nodes (flux2/sdxl/qwen):** dem Muster von
     `z_image_pid_decode.py` folgen und einen **optionalen `vae: VAEField | None`**-Input
     anbieten, der `scaling_factor`/`shift_factor` zur Laufzeit aus der VAE-Config
     liest (mit dokumentiertem Fallback). So vermeidet man hartkodierte, evtl.
     falsche Konstanten.

> Gemma-Offload nach dem Caption-Encode (`context.models.offload_from_vram(...)`)
> und die `working_mem_bytes`-Reservierung **unverändert** aus dem FLUX-Node übernehmen.

### B. `addPidDecode.ts` generalisieren (Frontend, zentral)

Aktuell ist die Chain auf `type: 'flux_pid_decode'` und `denoise: Invocation<'flux_denoise'>`
festverdrahtet. Generalisieren, damit alle Builder sie nutzen können:

```ts
// Neuer Union-Typ
export type PidDecodeNodeType =
  | 'flux_pid_decode' | 'flux2_pid_decode' | 'sd3_pid_decode'
  | 'sdxl_pid_decode' | 'qwen_image_pid_decode' | 'z_image_pid_decode';

type BuildPidDecodeChainArg = {
  g: Graph;
  state: RootState;
  denoise: Invocation<DenoiseLatentsNodes>; // vorher 'flux_denoise'
  decodeNodeType: PidDecodeNodeType;          // NEU
  vaeSource?: Invocation<VaeSourceNodes | MainModelLoaderNodes>; // NEU (für vae-Input der neuen Nodes)
  positivePrompt: Invocation<'string'>;
  seed: Invocation<'integer'>;
  mode: 'fit' | 'native';
  fitSize: Size;
};
```
- `g.addNode({ type: decodeNodeType, … })` statt hartkodiert.
- Falls `vaeSource` gesetzt und der Node einen `vae`-Input hat: `g.addEdge(vaeSource, 'vae', pidDecode, 'vae')`.
- `addPidDecode` und `addPidImageToImageNative` analog auf `DenoiseLatentsNodes` +
  `decodeNodeType` heben. Der FLUX-Aufruf in `buildFLUXGraph.ts` übergibt zusätzlich `decodeNodeType: 'flux_pid_decode'`.

### C. Node-Typ-Union erweitern (Frontend)

In `features/nodes/util/graph/types.ts` jedes neue `*_pid_decode` in **`ImageOutputNodes`**
aufnehmen (FLUX ist schon drin). Sonst akzeptiert `addImageToImage`/`canvasOutput` den Node nicht.

### D. Base-aware Gating + Decoder-Filter (Frontend)

1. **Welche Decoder-Base gehört zu welcher Main-Base?** Z-Image nutzt den
   **FLUX**-Decoder, alle anderen ihren eigenen. Mapping einführen (z. B. in
   `optimalDimension.ts` oder einer neuen `pid.ts`):

   ```ts
   // Main-Model-Base → erlaubte PiD-Decoder-Base(s)
   export const PID_SUPPORTED_BASES = ['flux', 'flux2', 'sd-3', 'sdxl', 'qwen-image', 'z-image'] as const;
   export const getPidDecoderBaseForMainBase = (base?: BaseModelType | null): BaseModelType | null => {
     switch (base) {
       case 'z-image': return 'flux';   // Z-Image teilt sich den FLUX-Decoder
       case 'flux': case 'flux2': case 'sd-3': case 'sdxl': case 'qwen-image':
         return base;
       default: return null;
     }
   };
   ```

2. **Gating** in `GenerationSettingsAccordion.tsx`: `isFLUX` durch einen Selektor
   `selectIsPidSupported` ersetzen (Base ∈ `PID_SUPPORTED_BASES`).

3. **Decoder-Filter** in `PidSettings.tsx` (`ParamPidDecoderModelSelect`):
   den hartkodierten `config.base === 'flux'`-Filter durch
   `config.base === getPidDecoderBaseForMainBase(mainBase)` ersetzen
   (`mainBase` via `selectMainModelConfig`).

### E. Readiness generalisieren (Frontend)

Die PiD-Checks (`pidDecoderModel`/`gemma2EncoderModel` vorhanden; Canvas:
`scaleMethod === 'none'`; PiD-aware Grid via `getGridSize(base, getPidScale(pidMode))`)
in einen Helfer ziehen und in **jedem** unterstützten Base-Zweig in
`getReasonsWhyCannotEnqueueGenerateTab` **und** `getReasonsWhyCannotEnqueueCanvasTab`
aufrufen:

```ts
const pushPidReasons = (reasons: Reason[], params: ParamsState) => {
  if (params.pidMode === 'off') return;
  if (!params.pidDecoderModel) reasons.push({ content: i18n.t('parameters.invoke.noPidDecoderModelSelected') });
  if (!params.gemma2EncoderModel) reasons.push({ content: i18n.t('parameters.invoke.noGemma2EncoderModelSelected') });
};
```
(Canvas zusätzlich: `bbox.scaleMethod !== 'none'` → `pidScaleBeforeProcessingMustBeOff`,
und `getGridSize(base, getPidScale(params.pidMode))` statt fixem Grid.)

### F. i18n

Die bestehenden Keys sind **base-unabhängig** und werden wiederverwendet
(`modelManager.pidMode`, `pidModeOff/Fit/Native`, `pidDecoder`, `gemma2Encoder`,
`parameters.invoke.noPidDecoderModelSelected`, `pidScaleBeforeProcessingMustBeOff`,
`toast.pidUnsupportedMode`, `popovers.pidMode`). I. d. R. **keine** neuen Keys nötig.

### G. Starter-Models (Backend)

Muster steht in `starter_models.py` (Region „PiD (Pixel Diffusion Decoder)"):
- `gemma2_2b_encoder` ist **shared** und schon vorhanden → in jeder neuen Decoder-`dependencies`-Liste wiederverwenden.
- Quellen liegen alle im HF-Repo **`nvidia/PiD`**, Pfadschema:
  `nvidia/PiD::checkpoints/<dir>/model_ema_bf16.pth`.
- `base`, `type=ModelType.PiDDecoder`, `format=ModelFormat.Checkpoint`, `variant` setzen.
- In `STARTER_MODELS` (und optional in den jeweiligen Base-Bundle) eintragen.
- Modul testen: `uv run --extra cuda python -c "import invokeai.backend.model_manager.starter_models"` (validiert Pydantic + Unique-Source-Assert).

---

## ⚠️ Wichtige Stolpersteine

1. **16-Kanal-Ambiguität (SD3, Qwen, Z-Image-Decoder=FLUX):** Der Config-Probe
   (`pid_decoder.py::_validate_base`) unterscheidet FLUX/SD3 (beide 16ch) nur am
   **Verzeichnisnamen** des Checkpoints. Beim Einzeldatei-Download via
   `nvidia/PiD::…/model_ema_bf16.pth` muss der Parent-Ordnername
   (`…official_sd3_distill…`) erhalten bleiben, **sonst defaultet 16ch auf FLUX**
   und der SD3/Qwen-Config-Match scheitert.
   → **Empfehlung:** `_validate_base` so härten, dass ein **explizit übergebenes
   `base`-Override** (Starter-Models setzen das) bei mehrdeutigem 16ch-Fall
   akzeptiert wird, auch wenn der Name schweigt. Details in [02-sd3.md](02-sd3.md)
   und [04-qwen-image.md](04-qwen-image.md).

2. **SDXL = 4 Kanäle:** neuer Eintrag `_LATENT_CHANNELS_TO_BASES[4] = {SDXL}`.

3. **Nur 2kto4k für SDXL/Qwen:** kein 2k-Checkpoint → nur ein Starter-Model, und
   die „Native"-UI sollte den 4×-Charakter klar kommunizieren.

4. **Positive-Prompt-String:** Der PiD-Node braucht den Prompt als `string`-Node.
   FLUX legt dafür `positivePrompt = g.addNode({type:'string'})` an. In anderen
   Buildern ggf. nicht vorhanden → in der jeweiligen UI-Sektion sicherstellen,
   dass ein `string`-Node mit dem (preset-modifizierten) Positive-Prompt existiert
   und in den `prompt`-Input des Decode-Nodes geht.

5. **FLUX.2 Latent-Layout:** 128ch / down-factor 16 — prüfen, dass der an PiD
   gereichte Latent dem `flux2_vae_decode`-Input entspricht (siehe [01-flux2.md](01-flux2.md)).

6. **VRAM:** `_PID_DECODE_WORKING_MEMORY_SCALING_CONSTANT = 250` ist auf FLUX@2048
   kalibriert. Bei FLUX.2 (down 16 → 4096px-Output schon bei 256er-Latent) kann der
   Peak deutlich höher liegen — nach erstem Lauf gegen den gemessenen Peak nachjustieren.

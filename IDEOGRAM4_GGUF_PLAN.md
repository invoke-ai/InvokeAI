# Ideogram 4 — GGUF-Transformer-Support (Follow-up-PR)

> **Status:** Phase 0 (Investigation) läuft — analytischer Teil abgeschlossen, Verifikation am
> echten GGUF ausstehend.
> **Basis:** Ideogram-4-PR #9303, inzwischen **in `main` gemerged** (`7a37c94674`) — dies ist der im
> Merge-Plan ausdrücklich zurückgestellte Follow-up „GGUF loader for the custom DiT".
> **GGUF-Quelle:** https://huggingface.co/molbal/ideogram-4-gguf
> **Sprache:** Deutsch (Plan), Englisch (Code/Identifier)

---

## 1. Ziel

Ideogram-4-Transformer in **GGUF-quantisiertem** Format laden können, inkl. Starter-Models für zwei
Größen.

**Der Nutzen ist Laufzeitspeicher und Portabilität, nicht Plattenplatz** (Zahlen in §2a). Heute gibt
es zwei Wege: `nf4` ist mit ~9,7 GiB pro Transformer-Paar der kleinste, braucht aber **bitsandbytes
und damit CUDA**; `fp8` läuft überall, kostet aber ~17,3 GiB. GGUF schließt die Lücke: ein
4-/5-Bit-Pfad **ohne CUDA-Bindung**, der das residente Paar auf 10,5–13,7 GiB drückt.

Auf der Platte spart es dagegen nichts — gegenüber nf4 ist GGUF sogar etwas größer, und Encoder/VAE
kommen ohnehin aus einem vollständig installierten Diffusers-Modell (Phase 2). Das gehört so in die
PR-Beschreibung.

## 2. Ausgangslage

**Was es schon gibt (aus #9303):**
- `Main_Diffusers_Ideogram4_Config` (`main.py`) — nur Diffusers-Format (nf4/fp8).
- `Ideogram4DiffusersModel`-Loader (`load/model_loaders/ideogram4.py`) mit fp8-/nf4-/unquant-Pfaden
  für **beide** Transformer + `_load_text_encoder` (Qwen3-VL) + `_load_vae`.
- Dual-Branch-CFG: conditional + unconditional Transformer, zusammengehalten via `transformer_pair.py`.
- Starter: `ideogram_4_nf4`, `ideogram_4_fp8` + `StarterModelBundle` (`starter_models.py`).

**Was das molbal-Repo liefert — nur Transformer, keine Pipeline:**
- `ideogram4-transformer-{q4_0,q4_1,q5_0,q5_1,q8_0}.gguf` (conditional)
- `ideogram4-unconditional_transformer-{q4_0,q4_1,q5_0,q5_1,q8_0}.gguf` (unconditional)
- **Kein** Qwen3-VL-Text-Encoder, **keine** VAE.

**Konsequenz:** Eine „Größe" = **zwei GGUF-Dateien** (cond + uncond). Encoder + VAE müssen aus einer
**separaten** Quelle kommen (wie Flux.2-Klein-GGUF, das eine Diffusers-Quelle für VAE/Qwen3 braucht).
molbals README bestätigt das ausdrücklich: „These files are not a complete standalone Ideogram 4
package."

## 2a. Größen — welche Quants sich überhaupt lohnen

`Ideogram4Transformer` hat **9,279 Mrd. Parameter** (gemessen über `state_dict()` auf dem Meta-Device).
Da immer cond **und** uncond geladen werden, zählt die Paar-Größe:

| Quant | pro Datei | **Paar** | Bewertung |
|---|---|---|---|
| q4_0 | 5,64 GB | **10,51 GiB** | kleinste Option; ~0,8 GiB *größer* als nf4 |
| q4_1 | 6,21 GB | 11,56 GiB | |
| q5_0 | 6,77 GB | 12,60 GiB | |
| q5_1 | 7,33 GB | **13,65 GiB** | beste Qualität, die klar unter fp8 bleibt |
| q8_0 | 10,14 GB | 18,88 GiB | **größer als fp8 — sinnlos** |
| *nf4 (Bestand)* | | *9,72 GiB* | CUDA-only |
| *fp8 (Bestand)* | | *17,28 GiB* | läuft überall |

q8_0 (8,5 Bit/Gewicht) liegt über fp8 (8 Bit/Gewicht) und bietet keinen Qualitätsvorteil, der die
1,6 GiB rechtfertigt. **Starter-Empfehlung daher q4_0 + q5_1, nicht q4_0 + q8_0.**

**Was quantisiert wurde:** Aus der Differenz zwischen tatsächlicher Dateigröße und reiner Quantgröße
ergeben sich über alle fünf Quants konsistent **295 M Parameter in F16**. Die Summe aller Tensoren
*außerhalb* der 34 Layer-Blöcke (`llm_cond_proj` 245,4 M + `t_embedding` 42,5 M + `adaln_proj`,
`input_proj`, `final_layer`, `embed_image_indicator`) beträgt 293,8 M — die Übereinstimmung ist
eindeutig. Quantisiert sind also nur die Layer-Gewichte (`qkv`, `o`, `w1/w2/w3`, `adaln_modulation`);
alles außerhalb bleibt F16. Das ist das Standardverhalten von city96s `convert.py`.

## 3. Kernherausforderungen

1. **Dual-Transformer** — das Single-File-GGUF-Muster (wie FLUX) passt nicht direkt. Ein „Modell"
   besteht aus zwei GGUFs. → Config muss einen **Ordner mit beiden** GGUFs erkennen (oder ein
   Paar-Konzept), Loader beide in den `transformer_pair` laden.
2. **Custom DiT** — `Ideogram4Transformer` ist kein `transformers`-Modell, also **kein**
   `from_pretrained(gguf_file=…)` (wie beim Gemma-Encoder). Es muss der GGML-Tensor-Pfad genutzt werden:
   `gguf_sd_loader()` → `GGMLTensor` → in das Custom-Modell laden (wie FLUX-GGUF).
3. **Key-Mapping** — ~~das ist die eigentliche Arbeit~~ **voraussichtlich entschärft, siehe §3a.**
   Nach molbals README wurde „from the original FP8 release" konvertiert, also aus
   `ideogram-ai/ideogram-4-fp8` — und genau die lädt unser Loader heute mit `strict=True` direkt in
   `Ideogram4Transformer`. Die Keys sollten daher **identisch** sein. Muss am Sample-GGUF bestätigt
   werden.
4. **Component-Source** — Encoder (Qwen3-VL) + VAE fehlen im GGUF-Repo → werden zur Laufzeit als
   Submodelle eines **vollständig installierten** Diffusers-Ideogram-4 bezogen (bzw. aus
   Standalone-Modellen), Muster `z_image_model_loader.py`. Siehe Phase 2.

## 3a. Phase-0-Zwischenstand: das Key-Mapping

**Referenzseite (gemessen).** `Ideogram4Transformer(Ideogram4Config())` hat **458 State-Dict-Keys** in
29 Mustern:

```
input_proj.{weight,bias}                    llm_cond_norm.weight
llm_cond_proj.{weight,bias}                 t_embedding.mlp_{in,out}.{weight,bias}
adaln_proj.{weight,bias}                    embed_image_indicator.weight
layers.{0..33}.attention.qkv.weight         (13824, 4608)   <- FUSIONIERT
layers.{0..33}.attention.{norm_q,norm_k}.weight
layers.{0..33}.attention.o.weight
layers.{0..33}.feed_forward.{w1,w2,w3}.weight
layers.{0..33}.{attention_norm1,attention_norm2,ffn_norm1,ffn_norm2}.weight
layers.{0..33}.adaln_modulation.{weight,bias}
final_layer.linear.{weight,bias}            final_layer.adaln_modulation.{weight,bias}
```

`rotary_emb.inv_freq` ist ein `persistent=False`-Buffer und taucht im State-Dict **nicht** auf.

**Die zwei möglichen Quellkonventionen.** diffusers 0.39 bringt eine eigene
`Ideogram4Transformer2DModel` (`models/transformers/transformer_ideogram4.py`) mit. Sie ist
strukturell identisch zu unserer vendorten Klasse — mit **einer** Ausnahme, der Attention:

| unsere vendorte Klasse | diffusers `Ideogram4Transformer2DModel` |
|---|---|
| `attention.qkv.weight` (13824, 4608) | `attention.to_q/to_k/to_v.weight` (je 4608, 4608) |
| `attention.o.weight` | `attention.to_out.0.weight` |

Alles andere (`input_proj`, `llm_cond_*`, `t_embedding.mlp_*`, `adaln_proj`,
`embed_image_indicator`, `feed_forward.w1/w2/w3`, alle vier Norms, `adaln_modulation`, `final_layer`)
heißt in beiden Klassen gleich.

**Erwartung:** identisches Mapping (Konvertierung aus dem fp8-Release, das unsere Namen benutzt).
**Fallback, falls das GGUF doch diffusers-Namen trägt:** q/k/v müssen zu `qkv` fusioniert werden.
Das geht über GGML-Tensoren nur deshalb, weil block-quantisierte Formate zeilenweise arbeiten
(Q4_0: 4608 Elemente/Zeile = 144 Blöcke à 18 Byte) — eine Konkatenation entlang dim 0 ist damit eine
reine Byte-Konkatenation. Die Reihenfolge stimmt: `qkv.view(B, L, 3, num_heads, head_dim)`
([modeling_ideogram4.py:136](invokeai/backend/ideogram4/modeling_ideogram4.py#L136)) zerlegt die
13824 als 3 × 18 × 256, Zeilen 0–4607 sind also q. Sauberer als die Byte-Fummelei wäre in dem Fall,
im GGUF-Loader eine Attention-Variante mit getrennten Projektionen zu bauen —
`cat([to_q(x), to_k(x), to_v(x)], dim=-1)` ist mathematisch exakt `qkv(x)`.

**Werkzeug.** `inspect_ideogram4_gguf.py` (Scratchpad) dumpt Metadaten + Quant-Histogramm und diffed
die GGUF-Tensornamen gegen diese 458 Referenz-Keys, inkl. Shape-Vergleich und — für nicht per Name
zuordenbare Keys — einem shape-basierten Kandidatenvorschlag.

### Ergebnis: Identität bestätigt, kein Mapping nötig

Gemessen an `ideogram4-transformer-q5_0.gguf` **und** `ideogram4-unconditional_transformer-q5_0.gguf`
(je 6,30 GiB):

| Prüfung | Ergebnis |
|---|---|
| Tensornamen vs. `Ideogram4Transformer` | **458 / 458 exakt**, 0 zusätzlich, 0 fehlend — in *beiden* Branches |
| Shapes | alle identisch ✓ |
| `attention.qkv` fusioniert | ✓ (13824, 4608) — kein `to_q/to_k/to_v` |
| fp8-Artefakte (`*.weight_scale`) | **keine** — vor dem Quantisieren wurde dequantisiert |
| Quant-Histogramm | 204 × Q5_0 + 254 × **BF16** |
| GGUF-Metadaten | **`kv_count = 0` — überhaupt keine** |

**Der Loader braucht also keine Key-Übersetzung**: `gguf_sd_loader()` → direkt
`load_state_dict(strict=True)`.

**Die Quantisierungs-Vorhersage aus §2a stimmt exakt.** Die 204 Q5_0-Tensoren sind genau die sechs
Linears pro Layer (`qkv`, `o`, `w1`, `w2`, `w3`, `adaln_modulation.weight`) × 34. Die 254 BF16-Tensoren
sind alles andere: 7 × 34 = 238 In-Layer-Norms/Biases plus die 16 Tensoren außerhalb der Blöcke —
**inklusive `llm_cond_proj.weight` mit 245 M Parametern**, das damit unquantisiert bleibt und allein
0,49 GB jeder Datei ausmacht.

**Zwei Konsequenzen für Phase 1:**

1. **Keine Metadaten → Erkennung nur über Tensornamen.** `general.architecture` existiert nicht.
   Als Fingerprint eignen sich `embed_image_indicator.weight` und `llm_cond_proj.weight` — beide
   Ideogram-4-spezifisch, keine Kollision mit FLUX (`double_blocks`), Qwen-Image (`txt_in/img_in`),
   Z-Image (`cap_embedder`) oder Wan (`patch_embedding`).
2. **BF16 ist nicht in `TORCH_COMPATIBLE_QTYPES`** (dort stehen nur `F32`/`F16` —
   [utils.py:9](invokeai/backend/quantization/gguf/utils.py#L9)). Eine Dequant-Funktion gibt es
   ([:113](invokeai/backend/quantization/gguf/utils.py#L113), registriert in `DEQUANTIZE_FUNCTIONS`),
   und `dequantize_and_run` als `__torch_function__`-Fallback
   ([ggml_tensor.py:13-53](invokeai/backend/quantization/gguf/ggml_tensor.py#L13-L53)) fängt auch
   Nicht-Linear-Module ab — `Ideogram4RMSNorm` (`F.rms_norm`) und `embed_image_indicator`
   (`nn.Embedding`) laufen also über den Dequant-Pfad statt als reine Tensoren. Funktioniert, ist
   aber im Loader-Smoke-Test ausdrücklich mitzuprüfen.

**cond und uncond sind nicht unterscheidbar**: gleiche Namen, gleiche Shapes, gleiches
Quant-Histogramm, byte-gleiche Größe. Nur der Dateiname trennt sie (siehe §6).

---

**Die Quelle ist ebenfalls verifiziert.** Die lokale fp8-Installation (`D:\ModelStuff3\ideogram-4-fp8`) —
also genau das, woraus molbal laut README konvertiert hat — trägt in **beiden** Branches exakt
unseren Namensraum:

```
transformer / unconditional_transformer:  je 669 Tensoren
  = 458 Gewichte (100 % Namensgleichheit mit Ideogram4Transformer, 0 Shape-Abweichungen,
    fusioniertes attention.qkv bestätigt)
  + 211 fp8-Scales (.weight_scale)
dtypes: 211× F8_E4M3 (alle Linears: 6/Layer × 34 + 7 außerhalb) + 247× BF16 (Biases + Norms)
```

Der Konverter hat diese Namen unverändert übernommen — siehe Ergebnistabelle oben. Der
diffusers-Fallback (`to_q/to_k/to_v`) ist damit vom Tisch.

## 4. Referenz-Implementierungen zum Spiegeln

| Aspekt | Vorbild im Repo |
|---|---|
| GGUF Custom-DiT laden | `load/model_loaders/flux.py:662` (`Main`, `Flux`, `GGUFQuantized`) und `:1080` (Flux2) |
| GGUF-Tensor-Loader / Dequant | `backend/quantization/gguf/loaders.py` (`gguf_sd_loader`), `utils.py` (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 bereits vorhanden ✓) |
| **Submodelle aus einem installierten Diffusers-Modell ziehen (Node)** | **`app/invocations/flux2_klein_model_loader.py` (:76-101); gleiche Form in `z_image_model_loader.py` (:91-135)** |
| **Source-Model automatisch auflösen (Frontend)** | **`buildFLUXGraph.ts` (:155-182), `readiness.ts` (:304-312), `selectFlux2DiffusersModels`** |
| Picker-Komponente | Flux.2-Klein: `ParamFlux2KleinModelSelect.tsx`, `paramsSlice` (`kleinVaeModel`/`kleinQwen3EncoderModel`) |
| GGUF-Config-Erkennung | `configs/qwen3_encoder.py` (`Qwen3Encoder_GGUF_Config`, `_has_ggml_tensors`) |
| Dual-Transformer-Handling | `backend/ideogram4/transformer_pair.py`, `load/model_loaders/ideogram4.py` (Diffusers-Pfad) |

**Gut:** Alle im Repo vorkommenden Quant-Typen (q4_0/q4_1/q5_0/q5_1/q8_0) haben bereits Dequant-Kernels
in `gguf/utils.py` — es sind **keine neuen Quant-Kernels** nötig.

## 5. Umsetzungsplan (phasenweise)

### Phase 0 — Investigation — ✅ **abgeschlossen**, Ergebnisse in §3a
- ✅ Referenz-Keys von `Ideogram4Transformer` erhoben (458 Keys / 29 Muster).
- ✅ fp8-Quelle geprüft: identischer Namensraum in beiden Branches.
- ✅ Größen-/Quantisierungsanalyse aus den Dateigrößen (§2a) — durch das Quant-Histogramm bestätigt.
- ✅ Beide q5_0-GGUFs gedumpt: **458/458 exakt, kein Mapping nötig**, keine Metadaten, BF16-Hinweis.
- ✅ VAE-Identität mit FLUX.2 bit-genau nachgewiesen (Phase 2).

### Phase 1 — Backend Config + Loader — ✅ **fertig** (Branch `feat/ideogram4-gguf`)

**Entschieden: zwei getrennte Modell-Records, kein Ordner-Konzept.** Vorbild ist Wan 2.2 A14B, das
seine Dual-Expert-MoE genauso behandelt: jede Datei ist ein eigenes Modell, ein Feld hält fest,
welche Hälfte es ist, und **gepaart wird erst im Loader-Node** (Phase 2). Das passt zu molbals
flacher Repo-Struktur und vermeidet ein Sonder-Installationsformat.

- **`Main_GGUF_Ideogram4_Config`** (`configs/main.py`) — Single-File, `base = Ideogram4`,
  `format = GGUFQuantized`, plus:
  - `branch: Literal["conditional", "unconditional"]`, per Dateiname erkannt
    (`_detect_ideogram4_gguf_branch`). Overridebar, falls jemand umbenennt.
  - Fingerprint `_has_ideogram4_keys`: **beide** von `embed_image_indicator.weight` und
    `llm_cond_proj.weight` müssen da sein; ComfyUI-Prefixe werden toleriert.
  - Registriert in `factory.py` (Import + `AnyModelConfig`-Union).
- **`Ideogram4GGUFModel`**-Loader (`load/model_loaders/ideogram4.py`) für
  `(Ideogram4, Main, GGUFQuantized)`: `gguf_sd_loader()` → `load_state_dict(strict=True, assign=True)`
  in **ein** `Ideogram4Transformer`. Kein Key-Mapping, keine Konvertierung. Gibt bewusst *keinen*
  `Ideogram4TransformerPair` zurück — das Paaren ist Sache des Nodes.

#### Zwei Bugs im Transformer, die der GGUF-Pfad aufgedeckt hat

Beide betrafen `modeling_ideogram4.py` und wären bei fp8/nf4 nie aufgefallen. `GGMLTensor` zeigt
seine echte Shape und Dtype nur über Python-Attribute; `nn.Linear` überlebt das, weil seine Ops in
der Dispatch-Tabelle abgefangen werden — alles andere nicht.

1. **Compute-Dtype-Probe.** `param_dtype = … or self.input_proj.weight.dtype` ergab bei GGUF
   `uint8` (die *gepackte* Speicher-Dtype), worauf der Input nach Byte gecastet wurde:
   `mat1 and mat2 must have the same dtype, but got Byte and BFloat16`. Neu: `_resolve_compute_dtype()`
   — Modul-`compute_dtype` (fp8) → Tensor-`compute_dtype` (GGUF) → `weight.dtype` (unquantisiert).
   Die Reihenfolge lässt den fp8-Pfad unverändert.
2. **Shape-Validierung in C++.** `F.rms_norm` und `F.embedding` prüfen die Gewichts-Shape, *bevor*
   `__torch_dispatch__` greift, und sehen den gepackten Puffer:
   `Expected weight to be of same shape as normalized_shape, but got weight of shape [106496] and
   normalized_shape = [53248]`. Neu: `_dequantized()` in `Ideogram4RMSNorm.forward` und für
   `embed_image_indicator` — ein No-op für unquantisierte Gewichte, betrifft nur wenige kleine
   Tensoren.

#### Verifiziert an den echten q5_0-Dateien

| Prüfung | Ergebnis |
|---|---|
| Identifikation über die volle Registry | beide Dateien → `Main_GGUF_Ideogram4_Config`, `match_count == 1` |
| Branch-Erkennung | `conditional` / `unconditional` korrekt |
| Negativkontrolle (Z-Image-GGUF) | → `Main_GGUF_ZImage_Config`, kein False Positive |
| `load_state_dict(strict=True)` | 458/458, **keine** Meta-Tensoren |
| Resident | 6,30 GiB = Dateigröße → Gewichte bleiben quantisiert |
| Forward-Pass | endlich, nicht degeneriert (std ≈ 0,87) |
| cond vs. uncond | jeder geprüfte Tensor unterscheidet sich (rel. L2 4–10 %) — echte zwei Branches, kein Duplikat-Upload |

Tests: `tests/backend/model_manager/configs/test_ideogram4_gguf_config.py` (18 Tests) — Fingerprint,
Branch-Heuristik, Negativfälle, plus beide Bugfixes als Regression, inklusive Forward-Pass eines
*winzigen* Ideogram-4 mit GGML-Gewichten gegen eine unquantisierte Referenz. Kein 6-GiB-Fixture nötig.
`tests/backend/model_manager` + `tests/backend/ideogram4`: 941 passed.

### Phase 2 — Loader-Node + Frontend — ✅ **fertig**

> **Umgesetzt.** Node-Form und Frontend-Verdrahtung stehen; Details und Begründungen unten
> unverändert. Abweichungen gegenüber der ursprünglichen Skizze:
>
> - Kein separater GGUF-Node: `Ideogram4ModelLoaderInvocation` wurde erweitert (v2.0.0), wie es
>   Z-Image und Wan halten. Der Diffusers-Pfad bleibt der Default (alle Zusatzfelder leer).
> - **Vierter Feldslot** `unconditional_transformer_model` — beim Schreiben von Phase 1 stand noch
>   nicht fest, dass jede GGUF-Datei ein eigener Modell-Record wird. Gefiltert auf
>   `ui_model_format=GGUFQuantized`; das Frontend filtert zusätzlich auf `branch`.
> - Neues Feld `Ideogram4TransformerField` (`model.py`), weil `TransformerField` nur *einen*
>   Identifier trägt. `ideogram4_denoise` (v2.0.0) hält beide Branches über einen `ExitStack`
>   ko-resident — jeder Schritt braucht beide.
> - **Der unconditional Branch wird NICHT automatisch aufgelöst** (anders als das Source-Model).
>   Die Dateien unterscheiden sich nur im Namen, also würde Raten bei mehreren installierten
>   Quant-Stufen still q4 mit q5 mischen. Expliziter Picker, branch-gefiltert.
> - Assertion im Node: gleiche `branch`-Werte auf beiden Feldern → harter Fehler. Fängt den
>   handgebauten Graph/API-Aufruf, den die UI-Filterung nicht abdeckt.
> - `MainModelPicker` blendet unconditional GGUFs aus dem Hauptmodell-Dropdown aus (wie Wan seinen
>   Low-Noise-Experten).
>
> Tests: Frontend 1723 passed (4 neue Graph-Builder-Fälle für die GGUF-Verdrahtung), Backend
> 941 passed, tsc/eslint/ruff sauber.

#### Ursprüngliche Planung

**Entschiedenes Design:** drei eigene Modellfelder — `transformer`, `qwen3_encoder`, `vae` — plus ein
optionales **Source-Model-Feld**, aus dem Encoder und VAE als Submodelle gezogen werden, wenn keine
Standalone-Modelle gesetzt sind.

> **Grundsatz: niemals Teil-Installationen von Diffusers-Modellen.** Ein Modell wird immer
> vollständig installiert. Die Wiederverwendung passiert erst *danach*, zur Laufzeit, indem der
> Graph gezielt einzelne Submodelle eines bereits installierten vollständigen Modells anfordert.

**Vorbild ist FLUX.2 Klein** — `Flux2KleinModelLoaderInvocation`
([flux2_klein_model_loader.py:76-101](invokeai/app/invocations/flux2_klein_model_loader.py#L76-L101)).
`ZImageModelLoaderInvocation` hat dieselbe Node-Form, aber Klein ist das bessere Vorbild, weil dort
die **Frontend-Seite** stimmt (siehe unten).

Node-Form — drei optionale Felder neben dem Hauptmodell:

```python
model               # Transformer (Hauptmodell, z. B. GGUF)
vae_model           # optional, standalone
qwen3_encoder_model # optional, standalone
qwen3_source_model  # optional, VOLLSTÄNDIGES Diffusers-Modell
                    #   -> ui_model_base=…, ui_model_type=Main,
                    #      ui_model_format=ModelFormat.Diffusers
```

Vorrangregel: explizit gesetztes Standalone-Feld gewinnt; sonst wird das Submodell aus dem
Source-Model gezogen (`model_copy(update={"submodel_type": …})`); ist beides leer, gibt es einen
klaren `ValueError` mit Handlungsanweisung. Das Source-Model wird gegen `ModelFormat.Diffusers`
validiert (`_validate_diffusers_format`,
[z_image_model_loader.py:126-135](invokeai/app/invocations/z_image_model_loader.py#L126-L135)).

**Für Ideogram 4 übertragen:**

| Feld | Quelle | Filter |
|---|---|---|
| `model` | Ideogram-4-GGUF (Phase 1) | `base=Ideogram4`, `type=Main` |
| `vae_model` | standalone **FLUX.2-VAE** | `base=Flux2`, `type=VAE` |
| `qwen3_encoder_model` | standalone Qwen3-VL-Encoder | `type=Qwen3VLEncoder` |
| `qwen3_source_model` | **vollständiges** Diffusers-Ideogram-4 | `base=Ideogram4`, `type=Main`, `format=Diffusers` |

Der bestehende Diffusers-Loader-Node
([ideogram4_model_loader.py](invokeai/app/invocations/ideogram4_model_loader.py)) bleibt unverändert.

**Konsequenz für den Nutzen des Features — ehrlich in die PR schreiben.** Wer keine Standalone-
Komponenten hat, braucht eine vollständige Diffusers-Installation (16 GB nf4 bzw. 26 GB fp8) als
Source-Model. GGUF spart hier also **keinen Plattenplatz**, sondern:

- **Laufzeitspeicher** — das residente Transformer-Paar sinkt von 17,28 GiB (fp8) auf 13,65 GiB
  (q5_1) bzw. 10,51 GiB (q4_0). VRAM ist die knappere Ressource.
- **Qualitätsleiter** — q4_0 … q5_1 liegen zwischen nf4 und fp8, ohne die CUDA-Bindung von nf4.

Für die VAE entfällt das Problem ganz, weil eine bereits installierte FLUX.2-VAE genügt (siehe
unten). Für den Encoder ist das Source-Model der Normalfall.

#### VAE — verifiziert: es *ist* die FLUX.2-VAE

Direkter Gewichtsvergleich `ideogram-4-fp8/vae` gegen `flux2-vae.safetensors`: **250 von 250
Float-Tensoren bit-identisch** nach Cast auf bf16, identischer Key-Namensraum, keine
Shape-Abweichung. Einziger Unterschied ist `bn.num_batches_tracked` (int64-BatchNorm-Zähler,
400000 vs. 0 — im Eval-Modus bedeutungslos). Auch die `vae/config.json` sagt es direkt:
`"_class_name": "AutoencoderKLFlux2"`.

InvokeAI hat dafür bereits `VAE_Checkpoint_Flux2_Config` und `VAE_Diffusers_Flux2_Config`
([vae.py:198](invokeai/backend/model_manager/configs/vae.py#L198),
[:453](invokeai/backend/model_manager/configs/vae.py#L453)). **Zu tun ist nur:** der VAE-Picker des
Ideogram-GGUF-Nodes muss `BaseModelType.Flux2` akzeptieren statt `Ideogram4`. Precedent dafür gibt
es beim PiD-Decoder für Z-Image (nutzt ebenfalls die FLUX-Seite). Kein neuer Download, kein neuer
Config-Typ, kein Konverter.

#### Qwen3-VL-Encoder — Normalfall Source-Model, Standalone-Feld ist Kür

Ideogram benutzt **stock Qwen3-VL-8B-Instruct**: `Qwen3VLModel`, `hidden_size` 4096,
36 Layer, vocab 151936 (aus `text_encoder/config.json` der fp8-Installation). Angezapft werden die
Hidden States der Layer 0, 3, …, 35 —
[constants.py:11](invokeai/backend/ideogram4/constants.py#L11) — daher `llm_features_dim`
= 13 × 4096 = 53248.

**Der tragende Pfad ist das Source-Model** und braucht **keine** Config-Änderung: aus einem
installierten vollständigen Diffusers-Ideogram-4 kommen `TextEncoder` und `Tokenizer` als Submodelle,
genau wie es der bestehende Loader heute schon tut
([ideogram4.py:154-213](invokeai/backend/model_manager/load/model_loaders/ideogram4.py#L154-L213)).

Das **optionale Standalone-Feld** ist Zusatz — nützlich für alle, die einen Qwen3-VL-8B ohnehin
liegen haben. Dafür wären zwei Anpassungen nötig, die aber **nicht** auf dem kritischen Pfad liegen
und notfalls in einen Folge-PR können:

1. **Variante einführen** (4B / 8B). Die Config lehnt heute alles ab, was nicht Krea-2s **4B** ist —
   harte Prüfungen auf `_KREA2_QWEN3_VL_HIDDEN_SIZE` und `_KREA2_QWEN3_VL_NUM_HIDDEN_LAYERS`
   ([qwen3_vl_encoder.py:27-35](invokeai/backend/model_manager/configs/qwen3_vl_encoder.py#L27-L35),
   [:78-84](invokeai/backend/model_manager/configs/qwen3_vl_encoder.py#L78-L84)). Ein 8B-Encoder
   wird also schon bei der Installation zurückgewiesen. Ohne Varianten-Feld würde der Krea-2-Picker
   sonst 8B-Encoder anbieten und umgekehrt; ein 4B an Ideogram krachte erst spät
   (`llm_cond_proj` erwartet 53248, bekäme 13 × 2560 = 33280).

   **Vorlage existiert:** Z-Images `Qwen3Encoder_Qwen3Encoder_Config` macht genau das schon —
   `variant: Qwen3VariantType` ([qwen3_encoder.py:229](invokeai/backend/model_manager/configs/qwen3_encoder.py#L229)),
   abgeleitet aus `hidden_size`
   ([:286-297](invokeai/backend/model_manager/configs/qwen3_encoder.py#L286-L297)).
   `_validate_krea2_qwen3_vl_config` wird dabei von einer Ablehnung zu einer Klassifikation.
   **Achtung Discriminator-Tag:** `variant` darf keinen Default bekommen, sonst wandert es in den Tag.

2. **fp8-Zweig im Standalone-Loader.** Die Loader liegen in
   [krea2.py:410](invokeai/backend/model_manager/load/model_loaders/krea2.py#L410) (Ordner) und
   [:509](invokeai/backend/model_manager/load/model_loaders/krea2.py#L509) (Single-File, kann
   ComfyUI-`fp8_scaled`). InvokeAIs eigenes Schema (`ideogram_fp8_weight_only` + `.weight_scale`
   pro Zeile — [quantized_loading.py:27](invokeai/backend/ideogram4/quantized_loading.py#L27))
   können sie nicht. Der Code dafür existiert aber: `swap_linears_to_fp8` + `load_fp8_state_dict`,
   benutzt in
   [ideogram4.py:183-194](invokeai/backend/model_manager/load/model_loaders/ideogram4.py#L183-L194) —
   also übernehmen, nicht neu schreiben.

#### Frontend

**Der entscheidende Punkt: `qwen3_source_model` ist KEIN Picker.** Es gibt nur **zwei** sichtbare
Felder (VAE, Encoder). Das Source-Model wird vom Graph-Builder automatisch aus den installierten
Diffusers-Modellen aufgelöst — der Nutzer wählt es nie
([buildFLUXGraph.ts:155-182](invokeai/frontend/web/src/features/nodes/util/graph/generation/buildFLUXGraph.ts#L155-L182)):

```ts
let qwen3SourceModel: ModelIdentifierField | undefined;
if (model.format !== 'diffusers' && (!kleinVaeModel || !kleinQwen3EncoderModel)) {
  const diffusersModels = selectFlux2DiffusersModels(state);
  …                                     // Kandidat wählen
  qwen3SourceModel = zModelIdentifierField.parse(sourceModel);
}
modelLoader = g.addNode({ type: 'flux2_klein_model_loader', model,
  vae_model: kleinVaeModel ?? undefined,
  qwen3_encoder_model: kleinQwen3EncoderModel ?? undefined,
  qwen3_source_model: qwen3SourceModel ?? undefined });
```

Und die Readiness prüft entsprechend „weder gewählt noch ableitbar"
([readiness.ts:304-312](invokeai/frontend/web/src/features/queue/store/readiness.ts#L304-L312)):

```ts
if (!params.kleinVaeModel && !hasFlux2DiffusersVaeSource) { reasons.push(…) }
if (!params.kleinQwen3EncoderModel && !hasFlux2DiffusersQwen3Source) { reasons.push(…) }
```

**Für Ideogram ist das einfacher als bei Klein.** Klein braucht Varianten-Matching
(`isFlux2KleinQwen3Compatible`, weil klein_4b und klein_9b unterschiedliche Qwen3-Größen haben);
Ideogram 4 hat keine Varianten — jedes installierte Diffusers-Ideogram-4 taugt als Quelle für beide
Komponenten.

Zu bauen (Klein berührt 10 Dateien, hier vergleichbar):

- `paramsSlice` + `types.ts`: `ideogram4VaeModel`, `ideogram4Qwen3EncoderModel` — **kein**
  State-Feld für das Source-Model.
- `modelsByType`-Hook: `selectIdeogram4DiffusersModels` (analog `selectFlux2DiffusersModels`).
- `readiness.ts` (+ Test): blocken, wenn das Hauptmodell GGUF ist und für VAE bzw. Encoder weder ein
  Standalone-Modell gewählt noch ein Diffusers-Ideogram-4 installiert ist.
- Graph-Builder `buildIdeogram4Graph.ts` (+ Test): Source-Model auflösen und den neuen Loader-Node
  einsetzen; VAE/Encoder in die Metadaten schreiben, wenn explizit gewählt.
- Neuer Loader-Node `model` / `vae_model` / `qwen3_encoder_model` / `qwen3_source_model`.
- Zwei Picker-Komponenten (analog `ParamFlux2KleinModelSelect.tsx`).

### Phase 3 — Starter-Models — ✅ **fertig**

Vier Einträge, nach dem Wan-A14B-Muster: die zweite Hälfte ist ein eigener Eintrag, den der primäre
über `dependencies` mitzieht.

| Eintrag | Datei | Größe |
|---|---|---|
| `ideogram_4_gguf_q4_0` | `ideogram4-transformer-q4_0.gguf` | ~11,3 GB als Paar |
| `ideogram_4_gguf_unconditional_q4_0` | `ideogram4-unconditional_transformer-q4_0.gguf` | (Dependency) |
| `ideogram_4_gguf_q5_1` | `ideogram4-transformer-q5_1.gguf` | ~14,7 GB als Paar |
| `ideogram_4_gguf_unconditional_q5_1` | `ideogram4-unconditional_transformer-q5_1.gguf` | (Dependency) |

- **q8_0 bleibt draußen** — als Paar größer als das fp8, das es ersetzen würde, ohne Qualitätsgewinn
  (§2a).
- **Die VAE ist als Dependency dabei** (`flux2_vae`, ~168 MB) — sie ist bit-identisch zur
  Ideogram-VAE, es muss also nichts Ideogram-Spezifisches geladen werden.
- **Keine Teil-Installationen.** Der Starter liefert nur die Transformer.
- Der `ideogram_bundle` bleibt bei `ideogram_4_nf4` — das ist weiterhin die empfohlene
  Erstinstallation; GGUF ist ein Zusatz.
- Der Dateiname überlebt die Installation (`os.path.basename` der URL bzw. `Content-Disposition`,
  [download_default.py:463-472](invokeai/app/services/download/download_default.py#L463-L472)), die
  `branch`-Erkennung greift also auch bei Starter-Installs.

### Phase 5 — Standalone Qwen3-VL 8B — ✅ **fertig**

Der Standalone-Encoder-Weg funktionierte nicht: Ideogram braucht Qwen3-VL-**8B**, und die Config
wies alles zurück, was nicht Krea-2s 4B war. Es gibt beide Größen upstream
(`Qwen/Qwen3-VL-4B-Instruct`, `Qwen/Qwen3-VL-8B-Instruct`, beide verifiziert); die 8B war schlicht
nie verdrahtet, weil Krea-2 der einzige Konsument war.

- **`Qwen3VLVariantType`** (4B/8B) in `taxonomy.py`; die harten 4B-Prüfungen wurden zu
  `_detect_qwen3_vl_variant` / `_detect_qwen3_vl_checkpoint_variant`. Ein unbekannter Hidden-Size
  oder eine Layer-Zahl ≠ 36 wird weiterhin abgelehnt.
- **`variant`-Feld ohne Default** in beiden Configs — mit Default wäre es in den Discriminator-Tag
  gewandert. Verifiziert: die Tags lauten weiterhin `qwen3_vl_encoder.qwen3_vl_encoder.any` bzw.
  `…​.checkpoint.any`.
- **DB-Migration** `2026_08_07_qwen3_vl_encoder_variant`: bestehende Records bekommen `4b`. Das ist
  eindeutig, weil bisher gar nichts anderes installierbar war.
- **Loader**: das HF-Repo für Single-File-Checkpoints hängt jetzt an der Variante (sonst lädt ein
  8B-Checkpoint eine 4B-Config); der Ordner-Loader hat einen Zweig für InvokeAIs eigenes
  `ideogram_fp8_weight_only`-Schema bekommen, der `swap_linears_to_fp8` + `load_fp8_state_dict`
  wiederverwendet statt sie nachzubauen.
- **Starter** `qwen3_vl_encoder_8b` (`Qwen/Qwen3-VL-8B-Instruct`, ~16 GB), zugleich Dependency
  beider GGUF-Starter — ein frischer GGUF-Install ist damit ohne Diffusers-Modell lauffähig.
- **Picker filtern nach Variante**: Krea-2 sieht nur 4B, Ideogram nur 8B. Beide sind derselbe
  Modelltyp, ein Vertauschen würde erst tief in der Inferenz als Shape-Fehler auffallen.

#### Zwei Fallen, die dabei zuschnappten

1. **Enum aufgeschnitten.** `Qwen3VariantType` endet nicht mit der 8B — danach folgt Animas
   `Qwen3_06B`. Die neue Klasse wurde dazwischen eingefügt und hat die 0.6B in den neuen Typ
   verschoben. Gefangen hat das der Typ-Level-Test `common.test-d.ts`, der die Zod-Enums gegen das
   generierte Schema prüft.
2. **`ModelRecordChanges` zählt Varianten von Hand auf** statt `AnyVariant` zu benutzen
   ([model_records_base.py:134-147](invokeai/app/services/model_records/model_records_base.py#L134-L147)).
   Ohne den Eintrag dort schlägt jede Model-Edit-/Install-Form fehl, sobald ein Config die neue
   Variante trägt — vier tsc-Fehler weit weg von der eigentlichen Änderung.

### Phase 4 — Tests + Doku — ✅ **bis auf den E2E-Render fertig**

- ✅ Backend: Config-Identifikation, Fingerprint, Branch-Heuristik, Negativfälle, Loader-Smoke sowie
  beide Bugfixes als Regression — 18 Tests, in Phase 1 entstanden.
- ✅ Frontend Graph-Builder: 4 Fälle für die GGUF-Verdrahtung (Diffusers bleibt unangetastet,
  expliziter uncond-Branch, Source-Fallback, kein Source bei expliziten Komponenten, harter Stopp
  ohne uncond).
- ✅ Frontend Readiness: 5 Fälle — Diffusers meldet nichts; GGUF ohne alles meldet dreifach; ein
  Diffusers-Source deckt Encoder/VAE, **nicht** aber den uncond-Branch; Standalone-Komponenten
  genügen ohne jedes Diffusers-Modell; nur-Encoder-Fehler.
- ✅ Doku: `docs/…/models.mdx`, Ideogram-4-Sektion um „GGUF transformers" erweitert — Zwei-Dateien-
  Prinzip, Encoder-Beschaffung, Warnung zum Quant-Level-Mischen, Begründung für das Weglassen von
  q8_0. README bleibt (Ideogram 4 ist bereits gelistet).
- ✅ **E2E: echter Render über die UI** — gegen `INVOKEAI_ROOT=D:\Entwicklung\InvokeAI_data2`,
  Browser-getrieben. Installiert wurden das q5_0-Paar aus `U:\ModelStuffSSD` (in-place) und
  `D:\ModelStuff3\ideogram-4-fp8` als Encoder-Quelle.

  Bestätigt wurde dabei die ganze Kette:

  | Schritt | Ergebnis |
  |---|---|
  | Identifikation beim Installieren | `branch=conditional` bzw. `unconditional`, korrekt aus dem Dateinamen |
  | Hauptmodell-Dropdown | zeigt nur den conditional Branch — der unconditional ist ausgefiltert |
  | Advanced-Panel bei GGUF | die drei Felder erscheinen; bei Diffusers nicht |
  | Platzhalter | „Required for GGUF models" / „From diffusers model" (da ein Diffusers-Modell installiert ist) |
  | Readiness | Invoke ist **gesperrt**, solange der unconditional Branch fehlt, und gibt frei, sobald er gesetzt ist |
  | Unconditional-Picker | bietet **nur** den unconditional Branch an |
  | Render | erfolgreich, Turbo-Preset (12 Steps), 1024², promptgetreu |

  **Hinweis für spätere Läufe:** der Server liefert das gebaute Frontend aus `dist/`, nicht die
  Quellen. Ohne `pnpm build` testet man die alte UI — das ist beim ersten Versuch genau passiert.

Gesamtstand Tests: Backend 941 passed, Frontend 1728 passed, tsc/eslint/ruff/knip sauber.

## 6. Offene Fragen / Risiken

- ~~**Key-Mapping**~~ — **erledigt.** 458/458 exakt, kein Mapping nötig (§3a).
- **Keine GGUF-Metadaten** (`kv_count = 0`) — die Config-Erkennung kann sich *nur* auf Tensornamen
  stützen. Fingerprint: `embed_image_indicator.weight` + `llm_cond_proj.weight`. Beim Registrieren in
  `factory.py` auf die Reihenfolge achten, damit generische GGUF-Main-Configs nicht vorher greifen.
- ~~**Ordner- vs. Zwei-Datei-Install**~~ — **entschieden:** zwei getrennte Modell-Records mit
  `branch`-Feld, Paarung im Node (Wan-A14B-Muster). Kein Ordner-Konzept, passt zu molbals flacher
  Repo-Struktur.
- **Gegenüber nf4 kein Gewinn** (§2a) — q4_0 ist als Paar sogar ~0,8 GiB größer. Die Rechtfertigung
  ist, dass nf4 CUDA-only ist und GGUF eine Qualitätsleiter ohne diese Bindung liefert. Das gehört so
  in die PR-Beschreibung, sonst ist die erste Rückfrage „warum ist das größer als nf4?".
- **Kein Plattenplatz-Gewinn insgesamt** — Encoder und VAE kommen aus einem vollständig
  installierten Modell (Grundsatz „keine Teil-Installationen", Phase 2). Das GGUF kommt also
  *zusätzlich* zu einer bestehenden Installation. Nutzen ist der Laufzeitspeicher.
- **Qwen3-VL-Encoder-Größe:** bleibt der Hauptspeicherposten; GGUF am Transformer hilft dort nicht.
- **Kein `general.architecture` bekannt** — ob molbals GGUFs überhaupt brauchbare Metadaten tragen,
  ist offen. Fallback für die Config-Erkennung sind die Tensornamen; `embed_image_indicator.weight`
  und `llm_cond_proj.weight` sind Ideogram-4-spezifisch genug, um nicht mit FLUX/Qwen/Z-Image zu
  kollidieren.
- **cond vs. uncond unterscheiden — bestätigt problematisch:** beide Dateien haben nachweislich
  identische Keys, Shapes, Quant-Histogramme und Größen (§3a). Die Unterscheidung kann *nur* über den
  **Dateinamen** (`…-unconditional_transformer-…`) laufen — fragil, wenn Nutzer umbenennen. Braucht
  eine klare Fehlermeldung beim Identifizieren, nicht erst beim Rendern.
- ~~**BF16-Dequant-Pfad**~~ — **erledigt in Phase 1**, hat zwei echte Bugs zutage gefördert
  (Dtype-Probe und C++-Shape-Validierung); beide gefixt und mit Regressionstests abgedeckt.
- **Node muss beide Branches einfordern.** Da jede GGUF-Datei ein eigenes Modell ist, kann ein Nutzer
  eines der beiden vergessen. Der Loader-Node braucht ein zweites Transformer-Feld (Vorbild:
  Wans `transformer_low_noise_model` mit `ui_model_format=GGUFQuantized`) **und** eine
  Readiness-Regel, die auf `branch` prüft — sonst rendert man mit zweimal derselben Hälfte, was
  schlicht CFG ohne Wirkung ergäbe.

## 7. Out of Scope

- GGUF für den Qwen3-VL-Encoder (separat; der Gemma-GGUF-Encoder für PiD ist ein anderes Feature).
- Neue Quant-Kernels (nicht nötig — q4_0/q4_1/q5_0/q5_1/q8_0 bereits unterstützt).
- Änderungen an der bestehenden Diffusers-nf4/fp8-Pipeline.

## 8. Empfohlene Reihenfolge

**Phase 0 ist erledigt** (§3a) — und mit dem besten möglichen Ausgang: kein Key-Mapping, keine neuen
Quant-Kernels, VAE bereits vorhanden.

Damit ist **Phase 1 klein**: Config mit Namens-Fingerprint + Loader, der zweimal `gguf_sd_loader()`
aufruft und direkt lädt. Der **Schwerpunkt liegt auf Phase 2**, weil sie Backend *und* Frontend
anfasst — wobei der Backend-Teil im Wesentlichen `flux2_klein_model_loader.py` abschreibt und die
eigentliche Arbeit im Frontend liegt (zwei Picker, automatische Source-Auflösung, Readiness).

Empfohlene Reihenfolge ab hier:
1. Phase 1 — Config + Loader, mit Unit-Test gegen die lokalen q5_0-Dateien.
2. Phase 2 Backend — Loader-Node mit den vier Feldern.
3. Phase 2 Frontend — Slice, Selektor-Hook, Readiness, Graph-Builder, Picker.
4. Phase 3/4 — Starter-Models, Tests, Doku.

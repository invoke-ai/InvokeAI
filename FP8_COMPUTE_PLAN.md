# fp8-Compute (`torch._scaled_mm`) — Spike-Ergebnisse und Umsetzungsplan

> **Status:** Spike abgeschlossen (Gate bestanden). Implementierung für Krea-2 gelandet auf
> `feat/fp8_scaled_compute`, raw fp8 auf `feat/fp8_compute_raw`; FLUX.2, Z-Image und Qwen-Image
> stehen aus.
>
> **→ Offene Arbeit, Branch-Landkarte, Messwerte und Fallstricke:
> [`plans/fp8-compute/README.md`](plans/fp8-compute/README.md).** Dieses Dokument hier ist der
> Spike-Bericht und bleibt als Begründung stehen.
> **Hardware:** RTX 4090 (Ada, SM 8.9), torch 2.7.1+cu128, diffusers 0.39.0
> **Modelle:** FLUX.2 Klein 9B (Spike), Krea-2 Turbo 12B (Spike + Implementierung)
>
> **Geltungsbereich der Zahlen:** Alle gemessenen Speedups gelten für ComfyUI-**scaled-fp8-
> Checkpoints**. Modelle, die ihr fp8 erst über `fp8_storage`-Layerwise-Casting bekommen,
> profitieren **nicht** — siehe §5.

---

## 1. Ausgangslage

fp8 **Storage** existiert bereits: `default_settings.fp8_storage`
([main.py:59](invokeai/backend/model_manager/configs/main.py#L59)), Layerwise-Casting-Hooks in
[load_default.py:283](invokeai/backend/model_manager/load/load_default.py#L283). Gewichte liegen als
`float8_e4m3fn`, werden aber **pro Forward zu bf16 hochgecastet** und dann durch `F.linear` geschickt.

fp8 **Compute** fehlt: `torch._scaled_mm` rechnet direkt auf den fp8-Tensor-Cores
(fp8×fp8 → bf16-Akkumulation). Genau das ist der ungenutzte Hebel.

Wichtig: fp8 ist **kein** allgemeiner Compute-Dtype. LayerNorm, Softmax, SDPA und Conv haben keine
fp8-Kernels. Nur Matmuls sind betroffen; Attention bleibt in jedem Szenario bf16.

---

## 2. Microbenchmark — echte Layer-Shapes

Alle 9,08B Parameter von Klein 9B stecken in 2-D-Linears (hidden 4096, alle Dimensionen Vielfache
von 16 → `_scaled_mm`-kompatibel). FLOP-gewichtet über alle Linears:

| Pfad | M=1024 | M=4096 |
|---|---|---|
| bf16 nativ | 1,00× | 1,00× |
| fp8-Storage-Dequant | **0,58×** | **0,85×** |
| fp8 `_scaled_mm` | **1,79×** | **1,65×** |

Mit Attention (bf16, 8 Double- + 24 Single-Blocks, 32 Heads à 128) hochgerechnet:

| Auflösung | scaled_mm vs bf16 | vs fp8_storage | Linear-Anteil am Step |
|---|---|---|---|
| 512px | 1,68× | 2,46× | 91% |
| 1024px | 1,50× | 1,70× | 79% |
| 1536px | 1,36× | 1,40× | 66% |

---

## 3. Vollmodell — 1024×1024, 20 Steps, Seed 42, CFG 1.0

Ein Weight-Load, drei Durchläufe. bf16 zuerst, danach Linears **in place** nach fp8 quantisiert —
`dequant` und `scaled_mm` teilen sich dieselben fp8-Gewichte und unterscheiden sich nur in der
Matmul-Ausführung. 151 Linears quantisiert (beide Dimensionen ≥1024).

| Pfad | ms/Step | vs bf16 | vs fp8_storage | Peak VRAM |
|---|---|---|---|---|
| bf16 | 711,6 | 1,00× | — | 17,72 GiB |
| fp8-Dequant (heute) | 790,7 | **0,90×** | 1,00× | 9,47 GiB |
| fp8 `_scaled_mm` | **551,7** | **1,29×** | **1,43×** | **9,43 GiB** |

**Bildqualität:** bei 1:1-Crops fotografisch gleichwertig. Falten, Augenbrauenhaare und Hauttextur
bleiben erhalten; `scaled_mm` ist minimal weicher im Mikrokontrast der Hautporen. Keine Artefakte.

Die relative L2-Abweichung der finalen Latents (dequant 0,444 / scaled_mm 0,572) ist **kein**
Qualitätsmaß — in einer Diffusionsschleife verstärkt jede minimale numerische Änderung chaotisch und
führt zu einem anderen, aber gleichwertigen Bild. Nur der Bildvergleich zählt.

### Gate-Auswertung

| Kriterium | Schwelle | Ergebnis | |
|---|---|---|---|
| Denoise-Speedup | ≥1,25× | 1,29× | ✅ knapp |
| VRAM-Neutralität | nicht schlechter als fp8_storage | 9,43 vs 9,47 GiB | ✅ minimal besser |

**→ GO.**

---

## 3b. Zweites Modell: Krea-2 Turbo (12B), 1024×1024, 8 Steps

Krea-2 Turbo ist in bf16 **~25 GB** und damit auf einer 24-GB-Karte nicht voll resident — es läuft
nur per Partial Loading (Low-VRAM-Mode), also mit ständigem CPU↔GPU-Streaming. Genau hier ist der
fp8-Gewinn am größten: fp8 macht aus einem streamenden Modell ein voll residentes (11,95 GiB).

260 Linears quantisiert (vs. 151 bei FLUX.2 Klein).

| Pfad | ms/Step | Peak VRAM | Anmerkung |
|---|---|---|---|
| bf16 (CPU-Offload) | ~2900 | 18,06 GiB | **keine gültige Speed-Messung** — gestreamt; grob repräsentativ für Low-VRAM |
| fp8-Dequant | 1297,8 | 12,80 GiB | voll resident |
| fp8 `_scaled_mm` | **797,8** | **12,53 GiB** | voll resident |

**`_scaled_mm` vs. Dequant: 1,63×** — deutlich mehr als die 1,43× bei FLUX.2 Klein. Krea-2 hat mehr
und größere Linears bei nur 48 Text-Tokens, der Linear-Anteil am Step ist also höher.

**Bildqualität:** bei 1:1-Crops (Gesicht, Bart, Hautstruktur) von der bf16-Referenz nicht zu
unterscheiden. Kein Detailverlust, keine Artefakte.

### Zusatzbefund: `krea2_turbo_fp8_scaled.safetensors` liegt bereits im richtigen Format

Der ComfyUI-„scaled fp8"-Checkpoint (12,24 GB) speichert fp8-Gewichte **plus Per-Tensor-Scale** —
exakt das, was `_scaled_mm` braucht. InvokeAI castet ihn beim Laden aktuell wieder nach bf16 hoch
(`_dequantize_fp8_weights` im FLUX.2/Krea-2-Loader). Solche Checkpoints könnten den fp8-Pfad direkt
befeuern, ohne Requantisierung — eigener, lohnender Folgeschritt.

---

## 3c. Implementierung: echter Loader-Pfad, offizieller Comfy-Checkpoint

`krea2TurboOfficialComfy_krea2TurboFp8.safetensors` (12,24 GB), 1024×1024, 8 Steps, durch die
Produktions-Helfer (Key-Konvertierung, Metadata-Remap, `extract`/`attach`, `CustomLinear`):

| Pfad | ms/Step | Peak VRAM |
|---|---|---|
| fp8-Gewichte, dequantisiertes Matmul | 1106,8 | 12,95 GiB |
| fp8-Gewichte, `_scaled_mm` | **884,4** | 12,95 GiB |

**1,25×**, resident 12,24 GiB statt ~25 GB in bf16. Bildqualität zwischen beiden Pfaden bei 1:1
nicht unterscheidbar (PSNR 29,95 dB).

Der Wert liegt unter den 1,63× aus §3b, weil der Checkpoint **96 von 256 Layern** per
`full_precision_matrix_mult` vom fp8-Matmul ausnimmt (darunter `attn.to_out.0` und `ff.down`). Das
zu respektieren kostet ~23% Speed und ist richtig so — es ist die Fidelity, die der Quantisierer
für diesen Checkpoint vorgesehen hat. **1,63× war eine Obergrenze, kein Erwartungswert.**

### Zwei Fehler, die erst die Messung aufgedeckt hat

1. **Der fp8-Zweig lief gar nicht.** `apply_custom_layers_to_model()` lässt `device_autocasting`
   für voll residente Modelle **aus**; ein Check nur in `_autocast_forward` wird dann nie erreicht,
   und `forward` fällt in den Dtype-Mismatch-Zweig, der still dequantisiert. Erste Messung: 1,05×.
   Der Check gehört vor die Autocasting-Verzweigung. Alle Unit-Tests waren dabei grün — sie riefen
   `_autocast_forward` direkt auf. Jetzt durch einen Test über beide Autocasting-Zustände abgesichert.
2. **Die Metadata matchte nichts.** `_quantization_metadata` benennt Layer nativ
   (`blocks.0.attn.gate`), die Scales werden nach dem Rename extrahiert
   (`transformer_blocks.0.attn.to_gate`). Ohne Remap wären genau die als fp8-unsicher markierten
   Layer in fp8 gerechnet worden.

### LoRA-Gegenprobe

`PornMaster_Detail_Slider_Krea2_V2.safetensors`, Gewicht 1.0, 256 Patch-Layer:

| Pfad | ms/Step | Sidecar-Module |
|---|---|---|
| fp8, dequantisiertes Matmul | 1241,2 | 256 / 256 |
| fp8, `_scaled_mm` | **1008,0** | 256 / 256 |

**1,231×** — der Sidecar-Overhead frisst den Gewinn also nicht auf (1,251× ohne LoRA). Alle
fp8-Module werden wie vorgesehen zwangsweise auf Sidecar-Patching geroutet
([layer_patcher.py:165](invokeai/backend/patches/layer_patcher.py#L165)), und der Wrapper erreicht
über `_autocast_forward` den fp8-Zweig. Latent-Divergenz gegen den LoRA-losen Lauf: 0,259 — die
LoRA wirkt also wirklich und läuft nicht still ins Leere. Bilder beider Pfade gleichwertig
(PSNR 26,83 dB).

---

## 4. Befunde, die über den Spike hinaus zählen

**1. `fp8_storage` ist heute langsamer als bf16.** 0,90× im Vollmodell, im Microbenchmark bis 0,58×
bei kleinen Token-Zahlen. Die Feature-Beschreibung
([main.py:59](invokeai/backend/model_manager/configs/main.py#L59)) nennt nur die VRAM-Ersparnis und
verschweigt die Zeitkosten. **Sollte unabhängig von diesem Vorhaben dokumentiert werden.**

**2. Ada kann nur Per-Tensor-*Aktivierungs*-Scaling.** `torch._scaled_mm` wirft auf SM 8.9
`Per-row scaling is not supported for this platform`; Per-Row gibt es erst ab Hopper.

Das betrifft aber **nur die Aktivierungen**. Ein Per-Output-Kanal-*Gewichts*-Scale ist separierbar —
Zeile `j` des Gewichts zu skalieren skaliert Ausgabespalte `j` — und wird deshalb nach dem Matmul
angewandt. `scaled_mm_linear` unterstützt beide Formen; die realen Comfy-Checkpoints liefern
ohnehin Per-Tensor-Skalare.

Nachgemessen: Per-Row-Gewichts-Scaling ergibt **denselben** relativen Fehler wie Per-Tensor
(0,03770 vs 0,03770). Der Fehler wird von der Aktivierungs-Quantisierung dominiert. Die frühere
Vermutung, Hopper wäre deshalb *genauer*, wird von der Messung nicht getragen — schneller
vermutlich schon.

**3. torchao wird nicht gebraucht.** `torch._scaled_mm` ist Bordmittel. torchaos Hauptwert sind die
Rowwise-Kernels — die auf Ada gar nicht laufen. **Keine neue Dependency**, eigener Pfad in
`CustomLinear`.

**4. `_scaled_mm` verbraucht weniger transienten Speicher als der Dequant-Pfad**, weil es das bf16-
Gewicht nie materialisiert. Gemessen an `single.linear1` (36864×4096): 343 MB statt 612 MB.

---

## 5. Umsetzung

### Einbauort (wie gebaut)

`CustomLinear._maybe_fp8_forward()`
([custom_linear.py](invokeai/backend/model_manager/load/model_cache/torch_module_autocast/custom_modules/custom_linear.py)),
aufgerufen an **zwei** Stellen:

1. in `forward()`, **vor** der `_device_autocasting_enabled`-Verzweigung
2. in `_autocast_forward()`, weil der LoRA-Sidecar-Wrapper dorthin dispatcht

Nur Stelle 2 zu bedienen war der Fehler aus §3c: für voll residente Modelle ist Autocasting aus,
`forward` fällt dann in den Dtype-Mismatch-Zweig und dequantisiert still.

Vollständige Bedingungsliste, sonst Rückfall auf den Dequant-Pfad:

- `fp8_compute` ist aktiviert
- Gewicht ist `float8_e4m3fn`, Input ist Floating-Point
- der Layer ist **nicht** per `full_precision_matrix_mult` ausgenommen
- Device-Capability ≥ (8, 9)
- `in_features % 16 == 0` und `out_features % 16 == 0`
- Gewicht und Input liegen auf demselben Device (Partial Loading!)
- Token-Dimension wird im Forward auf ein Vielfaches von 16 gepaddet und danach zurückgeschnitten

Der Fallback ist nicht optional: `_scaled_mm` ist bei den Shapes wählerisch, und ein stiller
Rückfall ist besser als ein Laufzeitfehler mitten in der Generierung.

### Exposure (wie gebaut)

Ein **eigenes** App-Config-Feld `fp8_compute` (`invokeai.yaml`), Default `false`, **eine Version
lang opt-in**. Grund: fp8-Compute quantisiert zusätzlich die Aktivierungen, ändert also die Numerik.

Es ist **nicht** an `fp8_storage` gekoppelt — das ist ein Per-Modell-Setting für einen anderen
Mechanismus (siehe unten). Dasselbe Flag entscheidet zusätzlich, ob scaled-fp8-Checkpoints beim
Laden quantisiert bleiben: sie fp8 zu lassen *ohne* fp8-Matmul würde VRAM halbieren, aber langsamer
laufen — die beiden müssen zusammen geschaltet werden.

#### Was muss der Nutzer setzen?

Nur `fp8_compute: true` in der `invokeai.yaml`. `fp8_storage` ist **nicht** zusätzlich nötig und
wird auf diesem Pfad übergangen (der Loader kehrt vor `_apply_fp8_layerwise_casting` zurück).

| Checkpoint | `fp8_compute` | `fp8_storage` | Ergebnis |
|---|---|---|---|
| scaled fp8 (Comfy) | **true** | egal, wird übergangen | fp8 resident + `_scaled_mm` ✅ |
| scaled fp8 | false | false | Dequant nach bf16, voller VRAM (heutiges Verhalten) |
| scaled fp8 | false | true | Dequant nach bf16, dann per Hooks wieder fp8 → VRAM gespart, langsamer |
| normales bf16-Modell | true | false | nichts — der Checkpoint hat keine fp8-Gewichte |
| normales bf16-Modell | egal | true | fp8 nur als Storage, kein `_scaled_mm` (0,90×) |

Ein gesetztes, aber wirkungsloses `fp8_storage` wird beim Laden protokolliert, damit der Fall nicht
stumm bleibt.

> **Korrektur zur früheren Planung:** „kein OpenAPI-Eingriff" trifft nicht zu. `InvokeAIAppConfig`
> wird vom app_info-Router serviert, das Feld steht damit in `schema.ts` (Zeile 18830). Die Änderung
> ist rein additiv und für den v7-Rewrite unkritisch, aber sie *ist* eine Schema-Änderung.

### `fp8_storage` profitiert nicht — und kann es so nicht

`_wrap_forward_with_fp8_cast` ([load_default.py](invokeai/backend/model_manager/load/load_default.py))
registriert einen Pre-Forward-Hook, der `p.data = p.data.to(compute_dtype)` ausführt. Wenn
`CustomLinear.forward` läuft, ist das Gewicht bereits bf16 — der fp8-Zweig kann per Konstruktion
nicht greifen.

**Konsequenz:** die Zahlen in §2/§3 (1,29× gegen bf16) beschreiben, was mit fp8-Gewichten *möglich*
ist. Ausgeliefert wird der Gewinn bislang nur für scaled-fp8-**Checkpoints** (§3c). Wer
`fp8_storage` aktiviert, zahlt weiterhin die gemessenen 0,90× ohne Gegenwert.

Das zu ändern hieße, den Layerwise-Casting-Pfad so umzubauen, dass er die Gewichte fp8 lässt und die
Dequantisierung `CustomLinear` überlässt — ein eigenes, nicht kleines Vorhaben. Bis dahin sollte
mindestens die Feature-Beschreibung von `fp8_storage` die Zeitkosten nennen (§4.1).

### Bestätigung in der echten App (2026-08-01)

Erster Lauf des vollständigen Stacks in InvokeAI selbst — Model-Cache, Partial Loading und LoRA
inklusive — mit `fp8_compute: true` **und** aktiviertem VAE-Tiling (siehe eigener Branch
`feat/qwen_image_i2l_tiling`), Krea-2 Turbo, 2560×1440-Upscale-Workflow:

| Indikator | vorher | mit fp8 + Tiling |
|---|---|---|
| Transformer-Ladezeit pro Pass | 4,41 s / 0,95 s | **0,03 s / 0,01 s** (bleibt resident) |
| `qwen_image_i2l` VRAM-Änderung | −9,939 G | **±0,000 G** |
| VRAM-Warnung (`only -N MB requested`) | ja | **weg** |
| Model-Cache-Misses | 8 | **0** |
| RAM-Zuwachs pro Lauf | +23,1 G | **+0,04 G** |

Der Transformer bleibt sogar über Queue-Items hinweg resident (`VRAM in use: 12.890G`); der
Folgelauf startet mit 0,01 s Ladezeit.

**Kein Gesamtzeit-Vergleich:** die beiden Läufe hatten unterschiedliche Step-Zahlen (16 vs. 12,
weil `denoising_start` im zweiten Denoise abweichend gesetzt war). Die Per-Step-Zeiten sind
erwartungsgemäß unverändert — Tiling beschleunigt den Denoise nicht, es verhindert nur die
Verdrängung. Der dem Fix zurechenbare Gewinn sind die ~5,3 s Transformer-Nachladen pro Lauf plus
das weggefallene Paging, das in keiner Zeitmessung auftaucht, aber die Ursache der Ruckler war.

Damit ist das letzte offene Integrationsrisiko aus der Liste unten praktisch bestätigt: Model-Cache
und LoRA-Patching funktionieren mit dem fp8-Pfad im Normalbetrieb.

### Integrationsrisiken

- [x] **LoRA-Sidecar-Patching** — verifiziert (§3c): 256/256 fp8-Module auf Sidecar geroutet,
      1,231×, Bilder gleichwertig.
- [x] **Transponierte Gewichte nicht als Buffer persistieren** — `scaled_mm_linear` bildet den
      Transpose im Forward. Als Buffer wäre er nach `.to(device)` kein View mehr und würde den
      Gewichtsspeicher verdoppeln (bei Krea-2 12,5 → 25 GB, sofortiger OOM).
- [ ] **Partial Loading** — der Pfad fällt bei Device-Mismatch bewusst zurück, aber ungetestet.
      Bei Krea-2 mit fp8 (12,2 GiB) greift Partial Loading auf 24 GB nicht; auf kleineren Karten schon.
- [ ] **Andere Loader** — nur Krea-2 verdrahtet. FLUX.2, Z-Image und Qwen-Image dequantisieren
      weiterhin eifrig und haben je eine eigene `_dequantize_*`-Kopie, die gegen `fp8_scaled.py`
      getauscht werden muss.
- [ ] **Andere Architekturen** — Shape-Mixe von FLUX.1 und Qwen-Image ungeprüft; die
      16er-Teilbarkeit ist nicht garantiert (der Fallback fängt es ab, kostet dann aber den Gewinn).
- [ ] **Z-Image** ist von `fp8_storage` ausgenommen (Dtype-Bug beim Layerwise-Casting) und bleibt
      es zunächst.

---

## 6. Belastbarkeit der Zahlen

Ehrlich zu benennen:

- **1,29× ist knapp über der Schwelle.** Zwischen zwei bf16-Läufen lagen 711,6 und 739,0 ms/Step —
  ~4% Messrauschen. Die Zahl trägt also ±0,05.
- **Nur 1024×1024 end-to-end gemessen.** 512px und 1536px sind hochgerechnet, nicht verifiziert.
- **Ein Prompt, ein Seed.** Das ist ein Plausibilitätsnachweis für die Bildqualität, keine
  Qualitätsstudie. Vor dem Default-Umschalten (§5) braucht es breitere Stichproben.
- Der hochgerechnete Wert für 1024px war 1,50×, gemessen wurden 1,29×. Die Differenz sind Norms,
  Modulation, RoPE und Scheduler — Fixkosten, die in allen drei Varianten identisch sind und den
  relativen Gewinn verdünnen.

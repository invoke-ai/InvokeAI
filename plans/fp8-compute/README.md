# fp8 Compute — Rollout-Plan (Fortsetzung)

> **Stand:** 2026-08-08. Ergänzt [`FP8_COMPUTE_PLAN.md`](../../FP8_COMPUTE_PLAN.md) im Repo-Root
> (Spike-Ergebnisse, Microbenchmarks, Gate-Auswertung). Dieses Dokument beschreibt **nur**, was noch
> zu tun ist und was man dafür wissen muss.
>
> **Hardware aller Messungen:** RTX 4090 (Ada, SM 8.9), PCIe 3.0, torch 2.7.1+cu128.
> **Datenwurzel für Tests:** `D:\Entwicklung\InvokeAI_data2`.

---

## 1. Branch-Landkarte

Zwei getrennte Stapel. Sie hängen inhaltlich zusammen, aber **nicht** in Git.

### Stapel A — fp8 **Compute** (dieser Plan)

```
main
 └── feat/fp8_scaled_compute      49990d643e   gepusht   ← Basis
      └── feat/fp8_compute_raw    e93854e6ae   gepusht   ← PR-Text fertig, PR NICHT eröffnet
      ├── feat/fp8_compute_flux2       (leer, zeigt auf 49990d643e)
      ├── feat/fp8_compute_qwen_image  (leer, zeigt auf 49990d643e)
      └── feat/fp8_compute_zimage      (leer, zeigt auf 49990d643e)
```

Die drei letzten Branches existieren nur als Platzhalter — **null Commits**. Nicht wundern.

### Stapel B — fp8 **Storage** (fremde Vorarbeit, offene PRs)

```
main
 └── feat/fp8_zimage             cd8f052656   PR #9414
      └── feat/fp8_anima         b47a92cd88   PR #9415
           └── feat/fp8_quantized_guard  3feb027fec
```

Lokal und `origin` sind bei allen fünf gepushten Branches identisch (Stand oben).

---

## 2. Was fertig ist

### `feat/fp8_scaled_compute` — ComfyUI-scaled-fp8 für Krea-2 + Qwen3-VL-Encoder

Gewichte mit `weight_scale`/`scale_weight` bleiben fp8-resident und laufen über
`torch._scaled_mm`. Enthält außerdem:

- `full_precision_matrix_mult`-Hints aus **zwei** Transporten: Safetensors-Header
  `_quantization_metadata` **und** per-Layer `.comfy_quant`-uint8-JSON-Tensoren
  (letzteres markiert 96 von 256 Layern in `PM_Krea2_turboV2FP8` — wurde vorher komplett ignoriert)
- `input_scale`/`scale_input` in beiden Schreibweisen, unkalibrierte Platzhalter (exakt 1.0,
  nicht-endlich, ≤0) werden verworfen
- App-Config-Flag `fp8_compute_full_precision_hints` (Default `true`)
- Doku in `docs/src/content/docs/configuration/fp8-storage.mdx` (umbenannt zu „FP8 Storage &
  Compute", mit `:::danger`-Callout zur Reproduzierbarkeit)

### `feat/fp8_compute_raw` — fp8-Checkpoints **ohne** Scales

Betrifft Checkpoints, die fp8-Gewichte tragen, aber keine `weight_scale`-Tensoren
(z. B. Z-Image-fp8, `getphatFLUXReality_v10FP8`). Diff: 6 Dateien, +272/−15.

| Datei | Was |
|---|---|
| `invokeai/backend/quantization/fp8_scaled.py` | `cast_state_dict()`, `_is_fp8_matmul_weight()`, `count_fp8_weights()`, `should_keep_fp8_weights()` |
| `load/load_default.py` | Guard: fp8-Storage greift nicht mehr, wenn fp8-Compute-Gewichte da sind |
| `load/model_loaders/{flux,krea2,z_image}.py` | pauschales `.to(torch.bfloat16)` → `cast_state_dict(...)` |
| `tests/backend/quantization/test_fp8_scaled.py` | +102 Zeilen; Gesamtsuite 904 grün |

**Der PR-Text liegt fertig in [`pr-fp8-compute-raw.md`](pr-fp8-compute-raw.md)** — nur noch kopieren
und den PR gegen `feat/fp8_scaled_compute` eröffnen.

---

## 3. Messwerte (Stand jetzt, mit ihren Grenzen)

**Krea-2 Turbo, scaled fp8** — 3,117 → 1,483 s/it = **2,10×**.

**Z-Image, raw fp8** (`--runs 4`, 1 Aufwärmer + 3 warme):

| | Basis (bf16) | raw fp8 |
|---|---|---|
| Transformer VRAM | 11 740 MB | **5 881 MB** |
| Residenz | 95,5–100 % | **100 %** |
| s/it warm (einzeln) | 1,19 / 1,34 / 1,36 | **1,01 / 1,06 / 1,00** |
| s/it Mittel | 1,297 | **1,023** (1,27×) |
| Graph gesamt | 40,0 s | **30,9 s** |

Der eigentliche Befund steckt in den Einzelwerten: die **Basis driftet nach oben**, fp8 bleibt
stabil. Das ist die volle Residenz, nicht der Matmul.

**`fp8_compute_full_precision_hints: true`** kostet ~10 % ohne LoRAs (mit LoRAs waren es 57 %) —
und ist trotzdem richtig, es ist die vom Quantisierer vorgesehene Fidelity.

**`enable_partial_loading: true`** kostet **+47 %** (1,027 → 1,507 s/it) und zerstört die
Seed-Reproduzierbarkeit: mittlere Pixeldifferenz 0,000 → 16–22 auf 98,7 % der Pixel. Ursache:
`_can_use_fp8_matmul` verlangt `self.weight.device == input.device`, das kippt pro Layer je nachdem,
was gerade resident ist.

**FLUX.2 `flux-2-klein-9b-fp8` auf dem raw-Branch: kein Effekt — und das ist korrekt.**
Alle 112 fp8-Tensoren dieses Checkpoints tragen Scales, werden von `_dequantize_fp8_weights`
gefaltet und laufen danach als fp8-**Storage**. Beide Arme protokollieren identisch
`Total model size: 8707.52MB` und `param_size=8708MB`, keine `kept N raw fp8`-Zeile. Die gemessenen
~8 % (1,75 → 1,60 s/it) sind **Run-Rauschen unter Partial Loading**, kein Gewinn — nicht als solcher
berichten. Wert des Laufs: Regressionsnachweis, dass scaled-fp8 unverändert bleibt.

---

### 3b. torch-Version vs. CUDA-Version (2026-08-08)

Gemessen auf `feat/fp8_scaled_compute`, Krea-2 `PM_Krea2_turboV2FP8`, 1024×1024, 8 Steps,
`fp8_compute: true`, `enable_partial_loading: false`. Drei Arme, damit die beiden Variablen getrennt
sind — ein reiner „cu128 gegen cu130"-Vergleich hätte den torch-Sprung mitgemessen:

| Arm | torch | CUDA | s/it warm (einzeln) | Mittel | Graph warm |
|---|---|---|---|---|---|
| A | 2.7.1 | 12.8 | 1,417 / 1,514 / 1,367 | 1,433 | 13,6 s |
| B | 2.11.0 | 12.8 | 1,245 / 1,273 / 1,280 | **1,266** | 12,2 s |
| C | 2.11.0 | **13.0** | 1,266 / 1,259 / 1,271 | **1,265** | 12,1 s |

**cu130 bringt auf Ada nichts: 0,001 s/it gegen cu128 (0,08 %).** Die Wertebereiche von B und C
überlappen vollständig — es gibt keinen Effekt zu messen.

Randbedingungen: Residenz in allen drei Armen identisch (Transformer 12 532,85 MB @ 100 %, Encoder
8 464,46 MB @ 100 %). `torch._scaled_mm` verhält sich in 2.11 unverändert (relativer Fehler 0,0378
wie im Spike). Die **Kaltläufe sind nicht vergleichbar** — Arm A lief zuerst mit kaltem
OS-File-Cache (56,4 s Graph gegen 24,3 s), das ist Plattencache, nicht torch.

### Gegenprobe: dieselbe Matrix mit `fp8_compute: false`

Wichtig für die Deutung: Krea-2 fällt mit `fp8_compute: false` **nicht** auf bf16 zurück (25 GB
passen nicht auf 24 GB), sondern auf **fp8 Storage** — `fp8_storage: true` steht in den
`default_settings` des Modells. Der Vergleich ist also fp8-Storage gegen fp8-Compute bei nahezu
gleichem VRAM (12 227,53 MB vs 12 532,85 MB, beide 100 % resident).

| torch / CUDA | fp8 aus (Storage) | fp8 an (Compute) | Gewinn durch Compute |
|---|---|---|---|
| 2.7.1 / 12.8 | 1,431 (1,423 / 1,442 / 1,429) | 1,433 | **1,00× — nichts** |
| 2.11.0 / 12.8 | 1,459 (1,453 / 1,468 / 1,455) | 1,266 | **1,15×** |
| 2.11.0 / 13.0 | 1,479 (1,533 / 1,446 / 1,459) | 1,265 | **1,17×** |

**Der wichtigste Befund der ganzen Reihe:** auf dem heute ausgelieferten torch 2.7.1 bringt fp8
Compute gegenüber fp8 Storage bei voller Residenz **keinen Geschwindigkeitsvorteil**. Erst torch
2.11 liefert ihn — und zwar einseitig: der Storage-Pfad wird dort sogar minimal *langsamer*
(1,431 → 1,459), nur `_scaled_mm` wird schneller. Die 1,13× aus der Tabelle oben sind also kein
allgemeiner torch-Gewinn, sondern eine Verbesserung der fp8-Kernels.

Auch auf dem Storage-Pfad ist **cu130 nicht schneller als cu128**. Der Mittelwert von 1,479 wird von
einem einzelnen Ausreißerlauf getragen (1,533, dessen Encoder-Zeit mit 3,03 s statt 1,65 s ebenfalls
auf externe Kontention deutet); die Läufe 2 und 3 (1,446 / 1,459) liegen mitten in der cu128-Spanne.

**Wofür fp8 Compute auf torch 2.7.1 trotzdem gut ist**, obwohl der Denoise gleich schnell läuft:

- **Kaltladen.** fp8 Storage muss den scaled-fp8-Checkpoint erst nach bf16 dequantisieren und dann
  per Layerwise-Casting zurück nach fp8 quantisieren — gemessen 139–189 s Kaltlauf gegen 12–40 s.
  fp8 Compute lässt die Scales einfach liegen.
- **RAM.** Derselbe Roundtrip trieb den Server-Prozess auf **58,3 GB** Working Set (nur noch 3,8 GB
  von 127 GB frei) für ein 12-GB-Modell. Das ist der bekannte Spike, für den es den Branch
  `fix/scaled_fp8_dequant_ram_spike` gibt — hier unfreiwillig reproduziert.

Verdacht zum Nullgewinn auf 2.7.1, **nicht gemessen**: dieser Checkpoint nimmt 96 von 256 Layern per
`full_precision_matrix_mult` vom fp8-Matmul aus, und `fp8_compute_full_precision_hints` stand auf
`true`. Ein Lauf mit `hints=false` würde zeigen, ob der Vorteil dadurch aufgefressen wird.

Nicht getestet: alles außer Krea-2, eine Auflösung, ein Checkpoint. Der Lauf zeigt, dass torch 2.11
mit diffusers 0.39.0 und transformers 5.5.4 auf diesem Pfad fehlerfrei durchläuft — das ist **kein**
Kompatibilitäts-Audit. `pyproject.toml` pinnt `torch==2.7.1+cu128` im `cuda`-Extra; ein Upgrade wäre
ein eigenes Vorhaben mit Breitentest. Nach diesen Zahlen wäre es allerdings der einzige Hebel, der
fp8 Compute überhaupt einen Denoise-Vorteil verschafft.

Offene Kleinigkeit: unter 2.11 protokolliert torch
`Mismatch dtype between input and weight ... Cannot dispatch to fused implementation` in
`layer_norm`. In allen 2.11-Armen identisch, also für diese Vergleiche irrelevant — aber es deutet
auf einen verpassten Fused-Kernel in InvokeAI hin und wäre einen eigenen Blick wert.

Harness-Hinweis: `bench_commits.py` hat jetzt `--fp8 true|false`, und `wait_for_queue` behandelt
einen fehlgeschlagenen Status-Poll als „noch beschäftigt". Ohne das stirbt der Lauf, weil InvokeAI
während der minutenlangen Dequantisierung den Event-Loop blockiert und HTTP gar nicht bedient.

---

## 4. Offene Arbeit — in dieser Reihenfolge

### 4.1 PR für `feat/fp8_compute_raw` eröffnen

Text liegt bereit. Basis: `feat/fp8_scaled_compute`.

Optional vorher noch: `getphatFLUXReality_v10FP8` testen (FLUX.1, **780 raw-fp8-Tensoren**) — das
wäre der erste FLUX.1-Nachweis auf diesem Pfad. Bisher nur Z-Image verifiziert.

### 4.2 `feat/fp8_compute_flux2` — scaled fp8 für FLUX.2

Der lohnendste offene Schritt, weil `flux-2-klein-9b-fp8` heute nachweislich den langsamen
Storage-Pfad nimmt (§3). Zu tun:

1. Im FLUX.2-Loader `_dequantize_fp8_weights` durch `extract_fp8_scaled_layers` +
   `attach_fp8_scales` ersetzen (Muster: `krea2.py`).
2. **Key-Konvertierung BFL → diffusers** dabei mitziehen — die Scales müssen nach dem Rename
   angehängt werden, sonst matchen die `full_precision_matrix_mult`-Hints nichts.
   Genau dieser Fehler ist bei Krea-2 schon einmal passiert (siehe `FP8_COMPUTE_PLAN.md` §3c).
3. **qkv-Split 1 → 3**: FLUX.2 speichert ein fusioniertes qkv, diffusers will drei. Der
   Per-Tensor-Skalar wird dabei **auf alle drei Teile kopiert**, nicht gesplittet. Alle Scales in
   diesem Checkpoint sind Skalare — das ist geprüft, kein Per-Kanal-Fall.
4. Verifizieren: `Total model size` muss deutlich unter 8 707 MB fallen und die Log-Zeile
   `FP8 layerwise casting enabled` muss **verschwinden**.

### 4.3 Z-Image / Anima — scaled compute

**Blockiert**, aber nicht schwer. Kollision an `_apply_fp8_to_nn_module` in `load_default.py`:

```python
# feat/fp8_compute_raw (meins)
def _apply_fp8_to_nn_module(model, storage_dtype, compute_dtype,
                            skip: Optional[Callable[[str, nn.Module], bool]] = None) -> None:

# feat/fp8_zimage (PR #9414)
def _apply_fp8_to_nn_module(model, storage_dtype, compute_dtype,
                            extra_skip_patterns: tuple[str, ...] = ()) -> None:
```

Auflösung, sobald #9414/#9415 in `main` sind: **beide Parameter behalten**. `extra_skip_patterns`
ist der deklarative Fall, `skip` der programmatische; sie schließen sich nicht aus, im Rumpf werden
sie ver-odert. Nicht versuchen, das eine durch das andere auszudrücken.

### 4.4 `feat/fp8_compute_qwen_image`

Noch nicht angefasst. Vorher prüfen: Qwen-Image-Shapes auf 16er-Teilbarkeit
(`in_features % 16 == 0 and out_features % 16 == 0`). Der Fallback fängt Verstöße ab, kostet dann
aber genau den Gewinn — also vorab messen, nicht hinterher rätseln.

### 4.5 SDNQ-fp8-Handler

Geparkt, bis `feature/svd-quantization` in `main` ist. Siehe `SDNQ_PLAN.md`.

---

## 5. Fallstricke, die schon Blut gekostet haben

**Nicht jeder fp8-Tensor ist ein Matmul-Gewicht.** Der Z-Image-Checkpoint quantisiert *alles*:
243 von 453 fp8-Tensoren sind 1-D-Biases, Norms und `pad_token`. Sie fp8 zu lassen crasht mit
`"abs_cuda" not implemented for 'Float8_e4m3fn'` — erst in `TimestepEmbedder.forward`, nach dem
naheliegenden Skip-Pattern-Pflaster dann in `all_x_embedder`. Die tragfähige Lösung ist
`_is_fp8_matmul_weight()`: Suffix `.weight` **und** `dim() >= 2` **und** das Modul ist wirklich ein
`torch.nn.Linear`. Skip-Patterns allein reichen nicht — man jagt sonst einen Layer nach dem anderen.

**fp8-Storage kann fp8-Compute still auffressen.** Layerwise-Casting castet die fp8-Gewichte pro
Forward nach bf16 hoch; danach kann der `_scaled_mm`-Zweig per Konstruktion nicht mehr greifen.
Deshalb der Guard am Kopf von `_apply_fp8_layerwise_casting`. Wenn eine Messung „keinen Unterschied"
zeigt: **zuerst im Log nach `FP8 layerwise casting enabled` suchen.**

**A/B-Vergleiche sind nur bei 100 % Residenz gültig.** Unter Partial Loading ist das Run-Rauschen so
groß wie jeder Settings-Unterschied. Immer `enable_partial_loading: false` setzen und die
Residenz-Zeile im Log prüfen, bevor man eine Zahl deutet.

**Immer ≥ 3 Bilder messen, Einzelwerte ausweisen.** Der Mittelwert allein hätte bei Z-Image den
eigentlichen Befund (Drift der Basis) unsichtbar gemacht.

---

## 6. Benchmark-Harness

[`bench_commits.py`](bench_commits.py) — Kopie des Session-Skripts, hier durabel abgelegt.

```bash
./.venv/Scripts/python.exe plans/fp8-compute/bench_commits.py \
    --variant feat/fp8_scaled_compute --variant WORKTREE \
    --runs 4 --item 602 --steps 8
```

- `--variant WORKTREE` misst den aktuellen Arbeitsbaum ohne Checkout (auch uncommittet).
- `--item` ist eine Queue-Item-ID aus `databases/invokeai.db`, die als Graph-Vorlage dient.
  Benutzt: **602** (Krea-2, 8 Steps), **467** (FLUX.2, dann `--steps 4`).
- Läuft gegen die echte Datenwurzel, schreibt aber eine **eigene** Config — die `invokeai.yaml`
  des Users wird nie angefasst.

Vier Dinge, die das Skript bereits löst und die man sonst neu lernt: `uv run` ist ein Wrapper
(Prozessbaum per `taskkill /T` beenden, sonst misst die nächste Variante den falschen Code),
`node_cache_size: 0` (sonst misst man den Invocation-Cache), Config pro Commit aus dessen bekannten
Keys bauen (unbekannte Keys sind ein harter Validierungsfehler), und Branch-Namen mit `/` für
Dateinamen escapen.

Python immer als `uv run --extra cuda ...` oder direkt `./.venv/Scripts/python.exe` — **nie** als
nacktes `python`.

---

## 7. Tests

```bash
uv run --extra test --extra cuda pytest tests/backend/quantization/test_fp8_scaled.py --no-cov
```

Relevante Klassen: `TestComfyQuantHints`, `TestQwen3VLKeyRemap`, `TestFullPrecisionHintToggle`,
`TestInputScale`, `TestRawFp8` (darin `test_only_linear_weights_stay_quantized` und
`test_skip_patterns_dequantize_named_modules` — die beiden decken §5 ab).

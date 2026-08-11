# Z-Image — VAE-Tiling für `z_image_l2i` / `z_image_i2l`

> **Status:** Analyse / Planung
> **Auslöser:** User-Report (24 GB VRAM): `estimate_vae_working_memory_flux: 5136384000` — der
> VAE-Decode reserviert 5,14 GB Working Memory, wofür der Model-Cache alles andere aus dem VRAM wirft.
> **Vorbild im Repo:** `39b237565d` (`feat(qwen-image): make VAE tiling usable on both Qwen-Image VAE nodes`)
> **Sprache:** Deutsch (Plan), Englisch (Code/Identifier)

---

## 1. Ziel

`tiled` / `tile_size` als opt-in Input-Felder an beiden Z-Image-VAE-Nodes, sodass der Working-Memory-Bedarf
bei hohen Auflösungen von der vollen Bildfläche auf **eine Kachel** sinkt — und der Model-Cache
entsprechend weniger reserviert.

Ausdrücklich **kein** Ziel: das RAM-Cache-Thrashing des Users (13,92 GB Cache gegen 18,96 GB Modelle).
Das ist ein anderer Cache und wird über `max_cache_ram_gb` gelöst. Tiling adressiert nur den
VRAM-Working-Memory.

## 2. Ausgangslage

Z-Image kann in InvokeAI **zwei verschiedene VAE-Klassen** benutzen —
`z_image_latents_to_image.py:27`:

```python
ZImageVAE = Union[AutoencoderKL, FluxAutoEncoder]
```

| Installationsweg | VAE-Klasse | Tiling vorhanden? |
|---|---|---|
| Z-Image als Diffusers-Modell (eigener `vae/`-Ordner) | `diffusers.AutoencoderKL` | **ja**, vollständig |
| Z-Image GGUF/Checkpoint + separate FLUX-VAE | `invokeai.backend.flux.modules.autoencoder.AutoEncoder` | **nein** |

Der zweite Weg ist der von uns dokumentierte Standard — `starter_models.py:1186`:
„Requires standalone Qwen3 text encoder and **Flux VAE**". Der User im Report fährt genau diesen
(Log: `AutoEncoder`, 159,87 MB).

**`AutoencoderKL`** bringt Tiling mit: `use_tiling`, `tile_sample_min_size`, `tile_latent_min_size`,
`tile_overlap_factor` (`autoencoder_kl.py:130-140`), `tiled_encode()` (:302), `tiled_decode()` (:364).
`enable_tiling()` selbst kommt aus `AutoencoderMixin` (`vae.py:895`) und nimmt **keine Parameter** —
die Kachelgröße wird über die Attribute gesetzt. Genau dafür existiert bei uns bereits
`patch_vae_tiling_params()` in `backend/stable_diffusion/vae_tiling.py`.

**`FluxAutoEncoder`** hat nur `encode()` / `decode()` / `forward()` — keine einzige Tile-Methode.

## 3. Kernherausforderung: der Schätzer, nicht das Tiling

`estimate_vae_working_memory_flux()` (`backend/util/vae_working_memory.py:73`) hat **keinen
`tile_size`-Parameter** und rechnet stur über die volle Ausgabefläche:

```python
working_memory = out_h * out_w * element_size * scaling_constant   # 2200 decode / 1100 encode
```

Ohne Anpassung liefe Z-Image in exakt den Fehler, den `39b237565d` bei `qwen_image_l2i` behoben hat:
der VAE rechnet gekachelt, der Cache reserviert weiter die volle Fläche, die Eviction passiert
trotzdem — Tiling wäre kosmetisch. **Der Schätzer ist der eigentliche Hebel, nicht `enable_tiling()`.**

## 4. Referenz-Implementierungen zum Spiegeln

| Aspekt | Vorbild im Repo |
|---|---|
| `AutoencoderKL`-Tiling im Node (identische VAE-Klasse!) | `app/invocations/latents_to_image.py:45-48, 55, 63, 89, 94-98` (SD/SDXL) |
| Kachelgröße auf `AutoencoderKL` setzen | `backend/stable_diffusion/vae_tiling.py` → `patch_vae_tiling_params()` |
| Schätzer mit `tile_size` (eine Kachel + 25 % Overlap) | `vae_working_memory.py:167` → `estimate_vae_working_memory_qwen_image()` |
| `tile_size=0` = „Modell-Default", vor dem Schätzen auflösen | `qwen_image_latents_to_image.py:48-56`, `qwen_image_image_to_latents.py:58-62` |
| OR mit globalem Flag | `context.config.get().force_tiled_decode` |

**Wichtig:** Der SD/SDXL-Node ist das bessere Vorbild als der Qwen-Node, weil er auf derselben
VAE-Klasse (`AutoencoderKL`) arbeitet. Der Qwen-Node nutzt `AutoencoderKLQwenImage`, dessen
`enable_tiling(tile_sample_min_height=…, tile_sample_min_width=…)` eine **andere Signatur** hat.
Nicht blind kopieren.

## 5. Umsetzungsplan

### Phase 0 — Messung (vor jedem Code)

Ohne Baseline ist der Schätzer-Konstantenwert geraten. Analog zu
`scripts/calibrate_qwen_vae_working_memory.py`:

1. Peak *reserved* Memory beim FLUX-VAE-Decode über ein Auflösungsraster (1024² … 2560×1440), bf16.
2. Dasselbe mit Kachelung (256 / 512 px).
3. Implizite Konstante `reserved / (h*w*element_size)` je Punkt — prüfen, ob `2200` (decode) /
   `1100` (encode) für den Tile-Fall trägt oder ob es wie bei Qwen eine eigene Konstante braucht.

**Abbruchkriterium:** Wenn der gemessene Gewinn bei 1024² unter ~20 % liegt, lohnt nur der
Hochauflösungs-Pfad — dann Phase 2 (FLUX-VAE) streichen und nur Phase 1 liefern.

### Phase 1 — `AutoencoderKL`-Pfad (klein, deckt Diffusers-Installationen ab)

1. `estimate_vae_working_memory_flux()` um `tile_size: int | None = None` erweitern.
   Bei gesetztem `tile_size`: `h = w = tile_size`, Ergebnis × 1,25 für den Overlap, plus das
   residente RGB-Bild — Formel von `estimate_vae_working_memory_qwen_image()` übernehmen.
   Default `None` = heutiges Verhalten, damit `flux_vae_decode` / `flux_vae_encode` /
   `z_image_*` unverändert bleiben, solange sie nichts übergeben.
2. `z_image_latents_to_image.py`: `tiled` + `tile_size` InputFields, `use_tiling = self.tiled or
   context.config.get().force_tiled_decode`, `tile_size` **vor** dem `estimate_…`-Aufruf (:58) auflösen,
   im `with`-Block `vae.enable_tiling()` + `patch_vae_tiling_params()` — aber nur wenn
   `not is_flux_vae` (:55).
3. `z_image_image_to_latents.py`: dasselbe in `vae_encode()` (:44) und `invoke()` (:91).
   `vae_encode` ist `@staticmethod` und wird von `z_image_denoise.py:28` importiert — die neuen
   Parameter müssen dort defaultet werden, sonst bricht der Aufrufer.
4. Node-Versionen `1.1.0` → `1.2.0` (beide, `z_image_latents_to_image.py:35`,
   `z_image_image_to_latents.py:34`).
5. Bei `FluxAutoEncoder` + `tiled=True`: einmalige `logger.warning`, dass Tiling für diese VAE nicht
   verfügbar ist, und ungekachelt weiterrechnen. **Nicht** hart fehlschlagen — der Node kann beide
   Klassen sehen und der User wählt die VAE nicht bewusst nach Tiling-Fähigkeit aus.

### Phase 2 — `FluxAutoEncoder`-Pfad (optional, größer)

Nur angehen, wenn Phase 0 den Nutzen belegt. Zwei Optionen:

- **(a) Tiling in die Klasse:** `tiled_encode()` / `tiled_decode()` + Blend-Logik in
  `backend/flux/modules/autoencoder.py`. Sauber, aber die Klasse wird auch von FLUX.1/FLUX.2
  benutzt — Änderungen dort haben Reichweite über Z-Image hinaus. Braucht eigene Tests für FLUX.
- **(b) Tiling im Node:** Kachelschleife in einer Hilfsfunktion, VAE bleibt unangetastet.
  Dupliziert Blend-Logik, ist aber risikoarm und Z-Image-lokal.

Empfehlung: **(a)**, aber als eigener PR nach Phase 1, damit der Diffusers-Pfad nicht darauf wartet.

### Phase 3 — Frontend / Schema

`openapi.json` + `services/api/schema.ts` regenerieren (läuft per Stop-Hook automatisch).
Prüfen, ob die Nodes im Workflow-Editor die neuen Felder korrekt anzeigen.

## 6. Verifikation

- Tiled vs. untiled bei gleichem Seed: **identische Latent-Dimensionen** über mindestens acht
  Auflösungen (das war der Qwen-Testumfang).
- Reserved Memory vor/nach, gemessen über den Node — nicht geschätzt.
- Relative L2-Abweichung tiled/untiled dokumentieren. Bei Qwen waren es ~1,4 % auf Rausch-Input
  (Worst Case fürs Blending) — das ist der Grund, warum es opt-in bleibt.
- Regression: `tiled=False` muss byte-identisch zum heutigen Verhalten sein.

## 7. Risiken

| Risiko | Gegenmaßnahme |
|---|---|
| Schätzer-Konstante für Tiles falsch → OOM statt Ersparnis | Phase 0 misst; Konstante mit Headroom wie bei Qwen (max. beobachtet + ~8 %) |
| `vae_encode` ist `@staticmethod` und wird extern aufgerufen (`z_image_denoise.py:28`) | Neue Parameter mit Defaults, Aufrufer explizit prüfen |
| Seamless-Pfad (`z_image_latents_to_image.py:65`) kollidiert mit Tiling | Beide sind Kontext-Manager auf demselben VAE; Kombination testen oder ausschließen |
| `FluxAutoEncoder` still ohne Tiling → User wundert sich | Explizite Warnung (Phase 1.5), nicht stillschweigend ignorieren |

## 8. Offene Fragen

1. Soll `tile_size=0` bei `AutoencoderKL` auf `vae.tile_sample_min_size` (Modell-Default) auflösen,
   oder auf einen festen Wert wie bei Qwen (256)? SD/SDXL nutzt den Modell-Default — dem folgen.
2. Lohnt Phase 2 überhaupt, wenn der GGUF-Weg langfristig auf die Diffusers-VAE umgestellt werden
   könnte? Das wäre die Alternative zu 200 Zeilen eigener Kachel-Logik.

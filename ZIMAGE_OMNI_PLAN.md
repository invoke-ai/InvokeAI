# Z-Image Omni — Referenzbild-Konditionierung (SigLIP2 + Condition-Latents)

> **Status:** Analyse / Planung
> **Basis:** diffusers 0.39.0 (`pyproject.toml:39`) liefert `ZImageOmniPipeline`
> (`pipelines/z_image/pipeline_z_image_omni.py`)
> **Sprache:** Deutsch (Plan), Englisch (Code/Identifier)

---

## 1. Ziel

Z-Image-Omni als dritte Variante neben Turbo und ZBase: Bildgenerierung mit **Referenzbildern** als
Konditionierung (Editing / Multi-Ref), analog zu dem, was wir für FLUX.2 Klein schon haben.

## 2. Ausgangslage — die gute Nachricht zuerst

**Wir brauchen keine neue Transformer-Klasse.** `ZImageTransformer2DModel.forward()`
(`transformer_z_image.py:894`) nimmt bereits alles entgegen, was Omni benötigt:

```python
def forward(
    self,
    x,                                              # Zeile 897
    cap_feats: list[torch.Tensor, list[list[torch.Tensor]]],   # :898
    controlnet_block_samples: dict[int, torch.Tensor] | None = None,   # :900
    siglip_feats: list[list[torch.Tensor]] | None = None,      # :901
    image_noise_mask=...,
    ...
)
```

Und unser Denoise-Loop ruft den Transformer bereits in genau dieser Form auf —
`z_image_denoise.py:602-612`:

```python
model_output = transformer(
    x=latent_model_input_list,      # Liste von [C, 1, H, W]
    cap_feats=[pos_prompt_embeds],
    ...
)
```

Der Umbau ist damit **additiv**, nicht strukturell: Condition-Latents vorne an die Liste, zwei neue
Keyword-Argumente, Condition-Outputs hinten wieder abschneiden.

### Was die diffusers-Pipeline zusätzlich mitbringt

`ZImageOmniPipeline.__init__` (`pipeline_z_image_omni.py:148`) registriert gegenüber der
Basis-Pipeline zwei zusätzliche Module:

```python
siglip: Siglip2VisionModel,
siglip_processor: Siglip2ImageProcessorFast,
```

Ablauf im `__call__`:

1. Referenzbilder werden über `Flux2ImageProcessor` (`:173`) auf Zielfläche skaliert —
   `_resize_to_target_area(img, height * width)` bzw. Fallback `1024 * 1024` (`:523-525`).
2. `prepare_siglip_embeds()` (`:319`) — je Bild ein SigLIP2-`last_hidden_state`.
3. `prepare_image_latents()` (`:569`) — je Bild VAE-Latents (`condition_latents`).
4. Sequenz-Zusammenbau (`:668-675`):
   ```python
   x_combined = [condition_latents[i] + [latent_model_input_list[i]] for i in range(bs)]
   image_noise_mask = [[0] * len(condition_latents[i]) + [1] for i in range(bs)]
   ```
   Also: **0 = sauberes Referenzbild, 1 = verrauschtes Ziel.**
5. CFG mit anschließender Renormalisierung über `_cfg_normalization` (`:698-701`) — das haben wir
   heute nicht (`z_image_denoise.py:640` ist plain CFG).

## 3. Kernherausforderungen

1. **SigLIP2 ≠ SigLIP.** Wir haben nur v1: `ModelType.SigLIP` (`taxonomy.py:94`),
   `SigLIP_Diffusers_Config` (`configs/siglip.py:24`), Loader `load/model_loaders/sig_lip.py:4,25`
   (`SiglipVisionModel`), genutzt von `flux_redux.py`. Omni braucht `Siglip2VisionModel` +
   `Siglip2ImageProcessor`. Beides ist in unserem `transformers` vorhanden (Modul
   `transformers/models/siglip2/`) — aber es ist ein **eigener Modelltyp**, kein Drop-in auf den
   bestehenden SigLIP-Configs.
   *Hinweis:* `Siglip2ImageProcessorFast` ist bereits deprecated; direkt `Siglip2ImageProcessor` nutzen.
2. **Gewichte-Quelle unklar.** Die Docstring-Beispiele in diffusers sind widersprüchlich
   (`Z-a-o/Z-Image-Turbo` in `pipeline_z_image_omni.py:41` vs. `Tongyi-MAI/Z-Image-Turbo` in der
   Basis-Pipeline). Ob Omni eigene Transformer-Gewichte hat oder ein Turbo-Checkpoint plus SigLIP2
   reicht, ist **vor jeder Zeile Code** zu klären. Das entscheidet über den gesamten Config-Aufwand.
3. **Variantenerkennung.** `ZImageVariantType` (`taxonomy.py:164`) kennt `Turbo` und `ZBase`, erkannt
   über `scheduler shift >= 5.0` (`configs/main.py:_get_variant_or_raise`, `migration_26.py`). Für
   Omni braucht es ein eigenes Merkmal — vermutlich die Präsenz eines `siglip/`-Ordners im
   Diffusers-Repo, nicht der Scheduler-Shift.
4. **Multi-Image-Batching.** Die Pipeline hält Condition-Latents als verschachtelte Listen
   (`list[list[Tensor]]`) über die Batch-Achse. Unser Denoise arbeitet mit einer flachen Liste. Das
   ist beherrschbar, aber die CFG-Verdopplung (`:654-661`) muss die Condition-Listen mitverdoppeln —
   dort sitzt der wahrscheinlichste Bug.

## 4. Referenz-Implementierungen zum Spiegeln

| Aspekt | Vorbild im Repo |
|---|---|
| Ref-Images als Extension in den Denoise-Loop | `backend/flux2/ref_image_extension.py` → `Flux2RefImageExtension` (`:119`), `_prepare_ref_images` (`:177`), `ensure_batch_size` (`:306`) |
| Ref-Image-Skalierung auf Zielfläche | `ref_image_extension.py:31` → `resize_image_to_max_pixels()` |
| Zusätzliches Vision-Submodell im Model Manager | `configs/siglip.py`, `load/model_loaders/sig_lip.py`, `backend/sig_lip/sig_lip_pipeline.py` (SigLIP v1 für FLUX Redux) |
| Neue Variante samt DB-Migration | `ZImageVariantType` + `migration_26.py` (Turbo/ZBase-Einführung) |
| Bestehender Z-Image-Extension-Punkt | `backend/z_image/z_image_controlnet_extension.py` — zeigt, wo im Denoise zusätzliche Transformer-Argumente eingehängt werden |

## 5. Umsetzungsplan

### Phase 0 — Klärung (blockierend, kein Code)

1. Offizielle Omni-Gewichte identifizieren und lokal ziehen. Prüfen: eigener Transformer, oder
   Turbo-Transformer + separates SigLIP2?
2. State-Dict-Keys gegen `ZImageTransformer2DModel` halten. Falls sie abweichen → der Plan wird
   deutlich größer (eigene Config + Loader-Zweig), falls nicht → reiner Submodell-Zubau.
3. Erkennungsmerkmal für die Variante festlegen (Ordnerstruktur, `model_index.json`).

**Ohne Phase 0 ist der Rest nicht schätzbar.** Alles Weitere setzt „Transformer-Gewichte sind
kompatibel" voraus.

### Phase 1 — SigLIP2 als Modelltyp

1. `ModelType.SigLIP2` in `taxonomy.py` (neuer Typ, nicht v1 erweitern — andere Klasse, andere
   Preprocessing-Semantik).
2. `SigLIP2_Diffusers_Config` in `configs/siglip.py`, Registrierung in `configs/factory.py`.
3. Loader analog `load/model_loaders/sig_lip.py`, aber `Siglip2VisionModel.from_pretrained(...)`.
4. Starter-Model-Eintrag.

### Phase 2 — Backend-Extension

Neue `backend/z_image/z_image_omni_extension.py`, modelliert nach `Flux2RefImageExtension`:

- Nimmt PIL-Bilder + VAE + SigLIP2 entgegen.
- Skalierung auf Zielfläche (`resize_image_to_max_pixels`, Fallback 1024²).
- Liefert `condition_latents: list[Tensor]` und `siglip_feats: list[Tensor]`.
- `ensure_batch_size()` für den CFG-Fall — die Referenz-Konditionierung muss im negativen Pass
  identisch anliegen (`pipeline_z_image_omni.py:587` klont sie).

### Phase 3 — Denoise-Integration

1. Neues optionales InputField `reference_images` an `z_image_denoise.py`.
2. An den vier Transformer-Aufrufstellen (`:602`, `:608`, `:624`, `:630` — plus die Zweitmenge ab
   `:716`): `x` um die Condition-Latents voranstellen, `siglip_feats=` und `image_noise_mask=`
   ergänzen.
3. Model-Output: nur das letzte Listenelement ist das Ziel-Latent, die Condition-Ausgaben verwerfen.
4. Optional `cfg_normalization` als Feld nachziehen (`pipeline_z_image_omni.py:698`) — ist
   unabhängig von Omni auch für ZBase nützlich, kann also ein separater kleiner PR sein.

### Phase 4 — Frontend

Variante in der Model-Auswahl, Ref-Image-Slots im Canvas/Linear-UI. Am ehesten am FLUX.2-Klein-Weg
orientieren, der dieselbe Interaktion schon hat.

## 6. Verifikation

- Ohne Referenzbilder muss `z_image_denoise` **byte-identisch** zum heutigen Verhalten sein
  (gleicher Seed → gleiches Bild). Das ist der wichtigste Regressionstest, weil Phase 3 den
  gemeinsamen Pfad anfasst.
- Mit einem Referenzbild: Vergleich gegen einen direkten `ZImageOmniPipeline`-Lauf in einem Skript,
  gleicher Seed, gleiche Auflösung. Abweichung dokumentieren.
- CFG-Pfad (`guidance_scale != 1.0`) separat prüfen — dort sitzt die Listen-Verdopplung.
- Turbo und ZBase ohne Omni-Gewichte: unverändert lauffähig.

## 7. Risiken

| Risiko | Gegenmaßnahme |
|---|---|
| Omni braucht doch eigene Transformer-Gewichte | Phase 0 klärt das vorab; falls ja, Aufwand neu schätzen statt weiterbauen |
| Phase 3 bricht den bestehenden Z-Image-Pfad | Ref-Images optional, Default `None`, Regressionstest als Abnahmekriterium |
| SigLIP2-Modell füllt den ohnehin knappen Model-Cache | Bekanntes Thema (siehe User-Report 24 GB): SigLIP2 ist klein, aber der Cache-Druck steigt. Working-Memory-Schätzung nicht vergessen |
| `Siglip2ImageProcessorFast` deprecated | Direkt `Siglip2ImageProcessor` verwenden |

## 8. Abgrenzung

Nicht in diesem Plan:

- **ControlNet-Ablösung** durch `ZImageControlNetModel` — eigener Vorgang, siehe Notizen zu Punkt 1.
- **VAE-Tiling** — siehe `ZIMAGE_VAE_TILING_PLAN.md`.
- **`set_attention_backend()`** — modellübergreifend, gehört nicht in einen Z-Image-Plan.

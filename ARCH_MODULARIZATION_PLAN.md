# Backend-Modularisierung: Architektur-Registry

> **Status:** Planung / abgestimmt
> **Scope:** Backend only. Frontend ist bewusst ausgeklammert (v7-Rewrite läuft parallel).
> **Sprache:** Deutsch (Plan), Englisch (Code/Identifier)

---

## 1. Ziel

Einen neuen `BaseModelType` hinzuzufügen soll möglichst **keine Core-Dateien** mehr editieren,
sondern ein Verzeichnis anlegen und sich registrieren. Zweck: kleinere, reviewbare Upstream-PRs
und ein aufgeräumter Core.

**Kein Ziel:**

- Kein Third-Party-Plugin-System. Enums bleiben geschlossen, Imports statisch.
- Kein Frontend. Wenn das Backend sauber ist, kann der v7-Rewrite das Muster direkt übernehmen.
- Keine Verhaltensänderung. Siehe §6.

---

## 2. Ist-Zustand

Gemessen an Z-Image als jüngstem vollständigen Base:

| | eigene neue Dateien | **editierte Core-Dateien** |
|---|---|---|
| Backend | ~23 | **~16** |
| Frontend | ~20 | ~40 (out of scope) |

Die editierten Core-Dateien zerfallen in vier Sorten:

| Sorte | Ort | Beispiel |
|---|---|---|
| Dispatch-Ketten | [step_callback.py:327](invokeai/app/util/step_callback.py#L327) | 11-Zweig-`elif` über `base_model`, wählt nur Daten aus |
| Typ-Streuung | [fields.py](invokeai/app/invocations/fields.py), [conditioning_data.py](invokeai/backend/stable_diffusion/diffusion/conditioning_data.py), [primitives.py](invokeai/app/invocations/primitives.py), [dependencies.py](invokeai/app/api/dependencies.py) | ein Conditioning-Typ kostet 4 Core-Dateien |
| Zentrale Datenlisten | [starter_models.py](invokeai/backend/model_manager/starter_models.py) | 2428 Zeilen, jeder Arch hängt an |
| Loader-Sonderfälle | [load_default.py:241](invokeai/backend/model_manager/load/load_default.py#L241) | `if config.base == ZImage` mitten im generischen Loader |

### Was bereits gut ist und **nicht** angefasst wird

- **`ModelLoaderRegistry`** ([model_loader_registry.py:66](invokeai/backend/model_manager/load/model_loader_registry.py#L66)) —
  funktionierendes Decorator-Registry mit Key `base-type-format`.
- **`Config_Base.CONFIG_CLASSES`** ([base.py:112](invokeai/backend/model_manager/configs/base.py#L112)) —
  Config-Klassen registrieren sich schon selbst via `__init_subclass__`; das Probing iteriert
  darüber, **nicht** über die Union.
- **`AnyModelConfig`** ([factory.py:246](invokeai/backend/model_manager/configs/factory.py#L246)) — bleibt eine
  explizite Union. Der Kommentar dort dokumentiert, dass upstream das absichtlich so hält
  (IDEs/LSPs verlieren bei dynamischem Aufbau die Typen). Kosten: 1 Import + 1 Zeile pro Config.
- **`node_pack`** ([baseinvocation.py:680](invokeai/app/invocations/baseinvocation.py#L680)) — leitet sich aus dem
  Top-Level-Paket ab, ist also für alles unter `invokeai/` immer `"invokeai"`. Invocations umziehen
  ist damit API-neutral.

### Explizit außerhalb des Scopes: Probe / Identification

Die Kaskade in [lora.py:933](invokeai/backend/model_manager/configs/lora.py#L933)
(`if has_qwen_ie_keys and has_lora_suffix and not has_z_image_keys and not has_krea2_keys and not has_flux_keys`)
ist echtes O(n²) — sie entsteht, weil `from_model_on_disk` alle Treffer sammelt und
`matches_sort_key` nur `ModelType`-Priorität kennt, keine Base-Spezifität.

**Bewusste Entscheidung: bleibt unverändert.** Probe-Änderungen können bestehende Installationen
umklassifizieren; das Risiko steht nicht im Verhältnis. Konsequenz: der Cross-Base-Edit an fremden
Probes bleibt Teil jedes neuen Base-PRs. Kandidat für ein späteres, eigenes Vorhaben
(additiver Tie-Break-Score, der bei genau einem Treffer nichts ändert).

---

## 3. Zielarchitektur

Pro Architektur ein Paket `invokeai/backend/<arch>/`, das genau eine `ArchitectureSpec` registriert:

```python
# invokeai/backend/z_image/architecture.py
register_architecture(
    ArchitectureSpec(
        base=BaseModelType.ZImage,
        preview=PreviewSpec(factors=FLUX_LATENT_RGB_FACTORS),
        conditioning=ConditioningSpec(info_cls=ZImageConditioningInfo, ...),
        starter_models=Z_IMAGE_STARTER_MODELS,
        loader_flags=LoaderFlags(supports_fp8_storage=False),
    )
)
```

Der Core liest ausschließlich über Accessoren (`get_preview_spec(base)` etc.) statt über
`elif`-Ketten.

### Registrierung

Eine **explizite Import-Liste** in `invokeai/backend/architectures/__init__.py` — eine Zeile pro Arch.
Deterministisch, LSP-freundlich, konsistent mit der bewusst-expliziten `AnyModelConfig`-Union.
Ein neuer Base editiert damit **eine** Core-Zeile statt 16 Dateien.

> **Harte Randbedingung:** [dependencies.py:161](invokeai/app/api/dependencies.py#L161) baut eine
> `safe_globals`-Allowlist für `torch.load` — sicherheitsrelevant, nicht bloß Bookkeeping. Die
> Registry muss **vollständig sein, bevor `initialize()` läuft**. Lösung: `architectures` wird auf
> Modulebene in `dependencies.py` importiert, nicht lazy innerhalb einer Funktion. Ein Test sichert
> das ab (§6).

### Archs ohne eigenes Paket

Vorhanden: `flux/`, `flux2/`, `z_image/`, `krea2/`, `anima/`, `wan/`, `ernie_image/`, `ideogram4/`, `pid/`.

Fehlend: **CogView4, Qwen-Image, SD3, SD1/SD2/SDXL** (letztere liegen in `stable_diffusion/`).

**Entscheidung:** Die Registry verlangt kein eigenes Verzeichnis — nur einen Registrierungsaufruf.
CogView4 und Qwen-Image bekommen ein schlankes neues Paket (dort existiert bereits Arch-Code, nur
verstreut). SD1/SD2/SDXL/SD3 registrieren sich aus `stable_diffusion/`, ohne Umzug. Damit fällt
kein Zusatzaufwand an, der nicht ohnehin Teil der jeweiligen PRs ist.

---

## 4. PR-Reihenfolge

Jeder PR ist für sich grün, für sich reviewbar und ändert kein Verhalten. Kein PR lässt eine
halb-`elif`-halb-Registry-Kette zurück: **wer eine Registry einführt, migriert alle Bases dieser
Registry im selben PR.**

### PR 0 — Fundament

Legt `invokeai/backend/architectures/` an: `ArchitectureSpec`-Dataclass (zunächst nur `base`),
`register_architecture()`, Accessoren, explizite Import-Liste. Jeder existierende Arch registriert
eine leere Spec.

Dazu die beiden Nachweis-Werkzeuge aus §6 (OpenAPI-Snapshot-Test, Vollständigkeitstest-Gerüst).

Kein Core-Verhalten wird berührt. Reiner Gerüst-PR — bewusst zuerst, damit die vier folgenden PRs
klein bleiben und unabhängig voneinander reviewbar sind.

### PR 1 — Preview-Registry

**Ersetzt:** [step_callback.py:327-370](invokeai/app/util/step_callback.py#L327-L370).

Die `elif`-Kette wählt heute nur Daten aus: `latent_rgb_factors`, `latent_rgb_bias`,
`smooth_matrix`, `spatial_scale`.

> **Design-Detail:** Die Registry darf **nicht rein statisch** sein. Wan wählt zur Laufzeit anhand
> `sample.shape[-3] == 48` zwischen zwei Faktor-Sätzen **und** setzt `spatial_scale=16` statt 8.
> `PreviewSpec` bekommt daher ein optionales `resolve(sample) -> PreviewSpec`; der statische Fall
> ist der Default.

Shared-Daten bleiben Shared: SD1/SD2 teilen sich einen Satz, SDXL/SDXL-Refiner ebenso,
Qwen-Image/Krea-2 ebenso, Z-Image nutzt die FLUX-Faktoren. Die Specs referenzieren dieselben
Konstanten-Objekte.

**Gewinn:** Das heutige `raise ValueError(f"Unsupported base model: {base_model}")` — ein
Laufzeit-Fehler mitten in der Generierung — wird zu einem CI-Fehler (§6).

**Risiko:** niedrig. Reine Datenverschiebung, durch den Vollständigkeitstest abgedeckt.

### PR 2 — Conditioning-Registry

**Ersetzt:** die Streuung über vier Core-Dateien. Bestand: 11 arch-spezifische
`*ConditioningInfo`-Klassen, 11 `*ConditioningField`, 10 `*ConditioningOutput`.

Die Klassendefinitionen ziehen ins Arch-Paket, die `ArchitectureSpec` registriert sie, und
[dependencies.py:161](invokeai/app/api/dependencies.py#L161) baut die `safe_globals`-Liste aus der
Registry statt aus einem handgepflegten Literal.

**Kritisch:** Klassennamen bleiben **bitgleich** — sie sind OpenAPI-`$ref`s und stecken in
gespeicherten Workflows. Es wird ausschließlich der Definitionsort verschoben, plus Re-Exports an
den alten Stellen, damit kein Import-Pfad bricht.

**Risiko:** mittel — wegen des `safe_globals`-Timings. Der Test aus §6 deckt genau das ab.

### PR 3 — Starter-Models pro Arch

**Ersetzt:** [starter_models.py](invokeai/backend/model_manager/starter_models.py), 2428 Zeilen.

> **Komplikation:** Einträge werden **arch-übergreifend geteilt** — `flux_vae` ist 14× referenziert,
> `qwen3_encoder` 11×, `clip_l_encoder` 11×, `t5_base_encoder` 6×, über 56 `dependencies=[...]`-Listen.
> Ein naives Aufteilen pro Arch erzeugt Zyklen.

**Aufteilung:**

- `starter_models/common.py` — geteilte VAEs/Encoder/Image-Encoder. Kein Arch-Import, von allen importierbar.
- `starter_models/<arch>.py` — arch-spezifische Einträge, importieren aus `common`.
- `STARTER_MODELS` und `STARTER_BUNDLES` (heute [Zeile 2097](invokeai/backend/model_manager/starter_models.py#L2097)
  bzw. [2413](invokeai/backend/model_manager/starter_models.py#L2413)) werden aus der Registry aggregiert.

**Nachweis:** Ein Test vergleicht die aggregierte Liste vorher/nachher auf Mengengleichheit
(sortiert nach `source`) — nicht nur auf Länge.

**Risiko:** niedrig-mittel. Reine Datenumschichtung, aber viel Fläche.

### PR 4 — Loader-Co-Location + Sonderfall-Hook

`model_loaders/<arch>.py` zieht ins jeweilige Arch-Paket (die `@ModelLoaderRegistry.register`-Decorators
bleiben unverändert, nur der Modulpfad ändert sich).

Der Sonderfall [load_default.py:241](invokeai/backend/model_manager/load/load_default.py#L241) —
`if config.base == BaseModelType.ZImage: return False` in `_should_use_fp8` — wird zu einem
`LoaderFlags`-Feld der Spec. Die übrigen Ausschlüsse dort (VAE, LoRA, Text-Encoder) sind
**typ**-basiert, nicht base-basiert, und bleiben generisch im Core.

**Risiko:** niedrig.

### PR 5 … N — Invocation-Moves, ein PR pro Arch

`invokeai/app/invocations/<arch>_*.py` → `invokeai/backend/<arch>/invocations/`.

Umfang: FLUX 13, Wan 10, Z-Image 9, FLUX.2 7, Anima 7, Qwen-Image 7, Krea-2 6, SD3 6,
CogView4 5, Ideogram4 5, ERNIE-Image 4, SDXL 1.

Discovery: [invocations/\_\_init\_\_.py](invokeai/app/invocations/__init__.py) globbt heute nur das eigene
Verzeichnis; getrieben wird alles von genau einem Wildcard-Import
([graph.py:40](invokeai/app/services/shared/graph.py#L40)). Die Arch-Pakete werden ohnehin über die
explizite Import-Liste geladen — die Invocations kommen also automatisch mit.

> **Bewusst getrennt und bewusst zuletzt.** Ein 13-Dateien-Move-Diff versteckt echte Änderungen im
> Review. Deshalb: rein mechanischer Move, keine inhaltliche Änderung im selben Commit, und erst
> nachdem die Registries stehen.

---

## 5. Erfolgsmaßstab

Nach Abschluss soll ein **neuer Base** im Backend folgende Core-Dateien anfassen:

| Datei | Grund | vermeidbar? |
|---|---|---|
| `taxonomy.py` | neuer Enum-Wert | nein (bewusst geschlossene Enums) |
| `configs/factory.py` | Union-Einträge | nein (bewusst explizit) |
| `configs/main.py` (+ ggf. `lora.py`, `controlnet.py`) | Config-Klassen | nein |
| `architectures/__init__.py` | 1 Import-Zeile | nein |
| Probe-Kaskade in `lora.py` | Cross-Base-Ausschlüsse | **nein — bewusst außerhalb des Scopes** |

**Von ~16 auf ~5** editierte Core-Dateien. Alles Übrige (Preview, Conditioning, Starter-Models,
Loader, Invocations) liegt dann im eigenen Verzeichnis.

---

## 6. Kompatibilitätsgarantie und Nachweis

**Garantie — hart null Änderung:**

- Invocation-Typ-Strings bitgleich
- Pydantic-Klassennamen (= OpenAPI-`$ref`s) bitgleich
- Enum-Werte und Discriminator-Tags bitgleich
- DB-Spalten und gespeicherte Model-Records unverändert
- `schema.ts` höchstens in der Reihenfolge abweichend

Gespeicherte Workflows, Metadaten und Modell-Records bleiben gültig. Review-Aussage:
*reiner Refactor, kein Verhalten.*

**Nachweis (Teil von PR 0, gilt für jeden Folge-PR):**

1. **OpenAPI-Snapshot-Diff** — `openapi.json` vorher/nachher normalisiert (Schlüssel sortiert)
   diffen. Muss leer sein. Fängt jede versehentliche Klassennamen- oder Feldänderung.
2. **Registry-Vollständigkeitstest** — für **jeden** `BaseModelType` (außer `Any`, `External`,
   `Unknown`) muss eine Spec mit Preview-Eintrag existieren. Ersetzt den heutigen Laufzeit-
   `ValueError` durch einen CI-Fehler.
3. **`safe_globals`-Timing-Test** — prüft, dass die Conditioning-Registry zum Zeitpunkt des
   `ObjectSerializerDisk`-Baus vollständig ist, und vergleicht die erzeugte Allowlist mit der
   heutigen 13-Klassen-Liste.
4. **Starter-Models-Mengenvergleich** — aggregierte Liste vorher/nachher identisch (nach `source` sortiert).
5. Die bestehende pytest-Suite und der `typegen`-Check.

Kein E2E-Lauf pro Architektur — zu teuer (~14 Modelle lokal). Der Vollständigkeitstest plus der
Umstand, dass PR 1 reine Daten verschiebt, deckt das Preview-Risiko ausreichend ab.

---

## 7. Offene Punkte

- [ ] **Re-Export-Strategie bei PR 2** — bleiben `from invokeai.app.invocations.fields import ZImageConditioningField`
      dauerhaft als Re-Export bestehen, oder werden die Importe repo-weit umgestellt? Re-Exports
      halten den Diff klein, hinterlassen aber zwei Wahrheiten. *Vorschlag: Re-Exports in PR 2,
      Aufräum-PR danach.*
- [ ] **Zuschnitt von `LoaderFlags`** — heute nur ein Feld (`supports_fp8_storage`). Ob weitere
      base-spezifische Loader-Sonderfälle existieren, ist noch nicht vollständig auditiert.
- [ ] **Reihenfolge der Import-Liste** — alphabetisch oder nach Abhängigkeit? Relevant nur, falls
      Arch-Pakete sich untereinander importieren (Z-Image nutzt FLUX-Faktoren, Krea-2 den
      Qwen-Image-VAE). *Vorschlag: alphabetisch, geteilte Daten in ein `common`-Modul, das keine
      Arch-Imports hat.*

---

## 8. Nicht Teil dieses Plans

- **Frontend** — die ~40 editierten Frontend-Dateien pro Base. Der v7-Rewrite adressiert das;
  wenn dieses Vorhaben steht, kann v7 dasselbe Registry-Muster übernehmen.
- **Probe/Identification** — siehe §2.
- **fp8-Compute** (`torch._scaled_mm`) — eigenes Vorhaben, beginnt mit einem torchao-Spike auf
  FLUX.2 Klein 9B. Berührt `CustomLinear` und `load_default.py`, ist aber unabhängig; läuft
  zeitlich **vor** diesem Plan, damit die offene Speed-Frage beantwortet ist.

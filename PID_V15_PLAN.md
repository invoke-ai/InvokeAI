# PiD v1.5 Decoder-Architektur — Follow-up-PR

> **Status:** Analyse / Planung
> **Basis:** PiD-PR (#9281). Der aktuelle Loader/`build_pid_net` unterstützt nur die **Legacy**-Architektur
> (`lq_hidden_dim=512`). NVIDIA hat im Juli 2026 **v1.5**-Decoder veröffentlicht (bessere Farbtreue, keine
> Grid-Artefakte) und die alten `res2kto4k`-Checkpoints nach `checkpoints_deprecated/` verschoben.
> **Quelle:** https://github.com/nv-tlabs/PiD/blob/main/pid/_src/configs/common/defaults/net.py
> **Sprache:** Deutsch (Plan), Englisch (Code).

---

## 1. Problem

- v1.5 nutzt eine **andere Netz-Konfiguration**: `lq_hidden_dim=1024` (statt 512), skalare Per-Token-Gates
  (statt per-dim), **PiT-LQ-Injection**, Replicate-Padding, zusätzliche Heads.
- Symptom bei Fehl-Nutzung: `size mismatch for lq_proj.latent_proj.0.weight … [1024, …] vs [512, …]` beim
  `PidNet.load_state_dict`.
- **Aktueller Stand (dieser PR):** Die Starter zeigen auf die **Legacy**-Gewichte in
  `checkpoints_deprecated/` (laden korrekt), und die Config **lehnt** ein 1024-dim-Checkpoint bei der
  Identifikation ab (`_lq_hidden_dim_from_state_dict != 512`) — v1.5 wird also sauber verweigert statt zu
  crashen. Dieser Plan beschreibt die **echte** v1.5-Unterstützung.

## 2. Kernaufgaben

### a) Versions-Discriminator (zwingend zuerst)
Das `PiDDecoderVariantType`-Enum kodiert nur die Auflösung (`res2k_sr4x` / `res2kto4k_sr4x`), **nicht** die
Netz-Version. Legacy-SD3/SDXL sind ebenfalls `res2kto4k` — die Architektur darf also **nicht** allein aus
der Variante abgeleitet werden. Optionen:
- Neues Feld/Enum `PiDDecoderArchVersion` (`legacy` / `v1_5`), abgeleitet aus **Gewichts-Shape**
  (`lq_hidden_dim`: 512→legacy, 1024→v1_5) — nicht aus dem Dateinamen (Single-File-Install verliert ihn).
- Klassifizierer: `lq_hidden_dim` lesen (Helper existiert: `_lq_hidden_dim_from_state_dict`), Version
  pinnen, in der Config speichern.

### b) v1.5-Netz in `build_pid_net`
- NVIDIAs `net.py`-v1.5-Config vollständig nachbauen: `lq_hidden_dim=1024`, PiT-LQ-Injection, skalare
  Gates, Replicate-Padding, zusätzliche Heads.
- `build_pid_net(backbone, arch_version)` → passendes `PidNet` je Version.
- `PidNet` (`_src/networks/pid_net.py`) ggf. um die v1.5-Bausteine erweitern (PiT-Injection etc.).

### c) Klassifizierer strikt validieren (Review P2)
- Nach der Version-/Backbone-Wahl **jeden** Checkpoint-Tensor gegen das gewählte Netz prüfen
  (`load_state_dict(strict=True)` bzw. Shape-Abgleich) **vor** Freigabe — kein „defer to runtime".
- Test: Legacy- **und** v1.5-Checkpoints für FLUX/FLUX.2/Qwen/SD3/SDXL laden; jeder Tensor muss zum
  gewählten Netz passen, dann ein minimaler Decode.

### d) Loader
- `pid_decoder.py`-Loader wählt `build_pid_net(backbone, arch_version)` anhand der gespeicherten Version.

### e) Starter-Models
- v1.5-Starter (empfohlen) für FLUX/FLUX.2/Qwen ergänzen (`checkpoints/PiD_v1pt5_res2kto4k_sr4x_official_…`),
  Legacy als „legacy" kennzeichnen oder entfernen. SD3/SDXL bleiben Legacy (kein v1.5).

### f) Doku
- `docs/…/pid-decode.mdx`: Legacy vs. v1.5 dokumentieren (welche Backbones, welche Presets, VRAM).

## 3. Referenzen
- NVIDIA v1.5 Netz-Config: `pid/_src/configs/common/defaults/net.py` (Repo oben).
- Aktuelle Legacy-Config: `invokeai/backend/pid/decode.py` (`_PER_BACKBONE`, `lq_hidden_dim=512`,
  `build_pid_net`).
- Shape-Guard (schon vorhanden): `configs/pid_decoder.py::_lq_hidden_dim_from_state_dict` +
  `_SUPPORTED_LQ_HIDDEN_DIM`.

## 4. Reihenfolge / Risiko
Phase a) (Versions-Discriminator aus Gewichts-Shape) zuerst — billig und entkoppelt. Der große Brocken ist
b) (v1.5-Netz inkl. PiT-Injection). c) (strikte Validierung) fällt danach leicht. Risiko: die genaue
v1.5-Architektur muss 1:1 aus NVIDIAs `net.py` übernommen werden — empirisch gegen ein v1.5-Checkpoint
mit `strict=True` verifizieren.

## 5. Out of Scope
- GGUF-PiD-Decoder (separates Thema).
- Änderungen am (bereits nativen) Gemma-GGUF-Encoder.

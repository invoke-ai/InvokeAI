# DyPE (Dynamic Position Extrapolation) Implementation Plan for InvokeAI

## Overview

DyPE ermöglicht es vortrainierten FLUX-Modellen, Bilder in Ultra-HD-Auflösungen (4K+) zu generieren, ohne typische Artefakte beim Überschreiten der Trainingsauflösung.

**Kernprinzip:** Die RoPE-Positionscodierungen werden während des Denoising dynamisch skaliert - frühe Steps konzentrieren sich auf niedrige Frequenzen (globale Struktur), späte Steps auf hochfrequente Details.

---

## Phase 1: Backend-Komponenten

### 1.1 DyPE Basis-Modul erstellen

**Datei:** `invokeai/backend/flux/dype/base.py`

```python
class DyPEConfig:
    """Konfiguration für Dynamic Position Extrapolation."""
    enable_dype: bool = True
    base_resolution: int = 1024  # Native Trainingsauflösung
    method: Literal["vision_yarn", "yarn", "ntk", "base"] = "vision_yarn"
    dype_scale: float = 2.0      # Magnitude λs (0.0-8.0)
    dype_exponent: float = 2.0   # Decay-Geschwindigkeit λt (0.0-1000.0)
    dype_start_sigma: float = 1.0  # Wann DyPE-Decay beginnt

class DyPEBasePosEmbed:
    """Basisklasse für dynamische Positionsextrapolation."""

    def _axis_token_span(self, ids: Tensor, axis: int) -> int:
        """Bestimmt die räumliche Ausdehnung einer Achse."""
        ...

    def _get_mscale(self, scale: float, timestep: float) -> float:
        """Berechnet zeitabhängige Skalierungsfaktoren."""
        ...

    def get_components_vision_yarn(self, ids, dim, theta, timestep, scale_h, scale_w):
        """Vision-YARN Methode für Bildmodelle."""
        ...

    def get_components_yarn(self, ids, dim, theta, timestep, scale):
        """Standard YARN Methode."""
        ...

    def get_components_ntk(self, ids, dim, theta, scale):
        """NTK-Faktor Methode."""
        ...
```

**Aufwand:** ~150-200 Zeilen, 1-2 Tage

### 1.2 DyPE-fähige RoPE-Funktion

**Datei:** `invokeai/backend/flux/dype/rope.py`

Erweitert die existierende `rope()`-Funktion aus `math.py`:

```python
def rope_dype(
    pos: Tensor,
    dim: int,
    theta: int,
    # DyPE Parameter
    current_sigma: float,
    target_h: int,
    target_w: int,
    base_resolution: int,
    dype_config: DyPEConfig,
) -> Tensor:
    """RoPE mit dynamischer Positionsextrapolation."""

    # Berechne Skalierungsfaktoren
    scale_h = target_h / base_resolution
    scale_w = target_w / base_resolution

    # Wähle Methode und berechne cos/sin
    if dype_config.method == "vision_yarn":
        cos, sin = get_components_vision_yarn(...)
    elif dype_config.method == "yarn":
        cos, sin = get_components_yarn(...)
    # ...

    # Konstruiere Rotationsmatrix wie in original rope()
    ...
```

**Aufwand:** ~100 Zeilen, 0.5-1 Tag

### 1.3 DyPE-Wrapper für EmbedND

**Datei:** `invokeai/backend/flux/dype/embed.py`

```python
class DyPEEmbedND(nn.Module):
    """Ersetzt EmbedND mit DyPE-Unterstützung."""

    def __init__(
        self,
        dim: int,
        theta: int,
        axes_dim: list[int],
        dype_config: DyPEConfig,
    ):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim
        self.dype_config = dype_config

        # State für aktuellen Step
        self.current_sigma: float = 1.0
        self.target_height: int = 1024
        self.target_width: int = 1024

    def set_step_state(self, sigma: float, height: int, width: int):
        """Wird vor jedem Denoising-Step aufgerufen."""
        self.current_sigma = sigma
        self.target_height = height
        self.target_width = width

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope_dype(
                ids[..., i],
                self.axes_dim[i],
                self.theta,
                self.current_sigma,
                self.target_height,
                self.target_width,
                self.dype_config.base_resolution,
                self.dype_config,
            ) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)
```

**Aufwand:** ~80 Zeilen, 0.5 Tag

---

## Phase 2: Integration in Denoising-Pipeline

### 2.1 DyPE Extension erstellen

**Datei:** `invokeai/backend/flux/extensions/dype_extension.py`

```python
from dataclasses import dataclass

@dataclass
class DyPEExtension:
    """Extension für Dynamic Position Extrapolation."""

    config: DyPEConfig
    target_height: int
    target_width: int

    def patch_model(self, model: Flux) -> DyPEEmbedND:
        """Ersetzt pe_embedder mit DyPE-Version."""
        original_embedder = model.pe_embedder

        dype_embedder = DyPEEmbedND(
            dim=original_embedder.dim,
            theta=original_embedder.theta,
            axes_dim=original_embedder.axes_dim,
            dype_config=self.config,
        )

        # Kopiere keine Weights - EmbedND hat keine lernbaren Parameter
        return dype_embedder

    def update_step_state(
        self,
        embedder: DyPEEmbedND,
        timestep: float,
        timestep_index: int,
        total_steps: int,
    ):
        """Aktualisiert den Step-State im Embedder."""
        embedder.set_step_state(
            sigma=timestep,
            height=self.target_height,
            width=self.target_width,
        )
```

**Aufwand:** ~60 Zeilen, 0.5 Tag

### 2.2 Modifikation von denoise.py

**Datei:** `invokeai/backend/flux/denoise.py`

Änderungen an der `denoise()`-Funktion:

```python
def denoise(
    model: Flux,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    # ... bestehende Parameter ...
    # NEU:
    dype_extension: DyPEExtension | None = None,
):
    # ... bestehender Code ...

    # DyPE Setup
    original_pe_embedder = None
    dype_embedder = None
    if dype_extension is not None:
        original_pe_embedder = model.pe_embedder
        dype_embedder = dype_extension.patch_model(model)
        model.pe_embedder = dype_embedder

    try:
        for step_index, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            # DyPE: Aktualisiere Step-State
            if dype_extension is not None and dype_embedder is not None:
                dype_extension.update_step_state(
                    dype_embedder,
                    t_curr,
                    step_index,
                    total_steps
                )

            # ... Rest des bestehenden Codes ...
    finally:
        # Restore original embedder
        if original_pe_embedder is not None:
            model.pe_embedder = original_pe_embedder
```

**Aufwand:** ~30 Zeilen Änderungen, 0.5 Tag

---

## Phase 3: Direkte Integration in FluxDenoiseInvocation

> **Entscheidung:** DyPE wird direkt als Parameter in FluxDenoise integriert (kein separater Node).

### 3.1 DyPE Preset Enum & Felder

**Datei:** `invokeai/backend/flux/dype/presets.py`

```python
from enum import Enum
from dataclasses import dataclass

class DyPEPreset(str, Enum):
    """Vordefinierte DyPE-Konfigurationen."""
    OFF = "off"
    AUTO = "auto"           # Automatisch basierend auf Auflösung
    PRESET_4K = "4k"        # Optimiert für 3840x2160 / 4096x2160

@dataclass
class DyPEPresetConfig:
    """Preset-Konfigurationswerte."""
    base_resolution: int
    method: str
    dype_scale: float
    dype_exponent: float
    dype_start_sigma: float

DYPE_PRESETS = {
    DyPEPreset.PRESET_4K: DyPEPresetConfig(
        base_resolution=1024,
        method="vision_yarn",
        dype_scale=2.0,
        dype_exponent=2.0,
        dype_start_sigma=1.0,
    ),
}

def get_dype_config_for_resolution(
    width: int,
    height: int,
    base_resolution: int = 1024,
    activation_threshold: int = 1536,  # FLUX kann nativ bis ~1.5x
) -> DyPEConfig | None:
    """Ermittelt automatisch DyPE-Config basierend auf Zielauflösung.

    Args:
        base_resolution: Native Trainingsauflösung des Modells (für Skalierungsberechnung)
        activation_threshold: Ab dieser Auflösung wird DyPE aktiviert (> base_resolution)

    Returns None wenn Auflösung <= activation_threshold (DyPE nicht nötig).
    """
    max_dim = max(width, height)

    if max_dim <= activation_threshold:
        return None  # FLUX kann das nativ, DyPE nicht nötig

    # Skalierungsfaktor basierend auf base_resolution (nicht threshold)
    scale = max_dim / base_resolution

    # Dynamische Parameter basierend auf Skalierung
    return DyPEConfig(
        enable_dype=True,
        base_resolution=base_resolution,
        method="vision_yarn",
        dype_scale=min(2.0 * scale, 8.0),
        dype_exponent=2.0,
        dype_start_sigma=1.0,
    )
```

### 3.2 Integration in FluxDenoiseInvocation

**Datei:** `invokeai/app/invocations/flux_denoise.py`

```python
from invokeai.backend.flux.dype.presets import DyPEPreset, DYPE_PRESETS, get_dype_config_for_resolution

@invocation(
    "flux_denoise",
    title="FLUX Denoise",
    # ...
    version="4.2.0",  # Version bump
)
class FluxDenoiseInvocation(BaseInvocation):
    # ... bestehende Felder ...

    # ===== NEU: DyPE Parameter =====
    dype_preset: DyPEPreset = InputField(
        default=DyPEPreset.OFF,
        description="DyPE preset for high-resolution generation. 'auto' enables automatically for resolutions > 1536px.",
    )

    # Erweiterte DyPE-Optionen (optional, nur wenn preset != off/auto)
    dype_scale: float | None = InputField(
        default=None,
        ge=0.0,
        le=8.0,
        description="DyPE magnitude (λs). Only used when dype_preset is 'off' but you want custom settings.",
    )

    dype_exponent: float | None = InputField(
        default=None,
        ge=0.0,
        le=1000.0,
        description="DyPE decay speed (λt). Only used with custom DyPE settings.",
    )

    def _get_dype_config(self) -> DyPEConfig | None:
        """Ermittelt DyPE-Konfiguration basierend auf Preset oder manuellen Werten."""

        if self.dype_preset == DyPEPreset.OFF:
            # Prüfe ob manuelle Werte gesetzt sind
            if self.dype_scale is not None:
                return DyPEConfig(
                    enable_dype=True,
                    base_resolution=1024,
                    method="vision_yarn",
                    dype_scale=self.dype_scale,
                    dype_exponent=self.dype_exponent or 2.0,
                    dype_start_sigma=1.0,
                )
            return None

        if self.dype_preset == DyPEPreset.AUTO:
            return get_dype_config_for_resolution(
                self.width, self.height,
                base_resolution=1024,
                activation_threshold=1536,  # FLUX kann nativ bis ~1.5x
            )

        # Preset verwenden
        preset_config = DYPE_PRESETS.get(self.dype_preset)
        if preset_config:
            return DyPEConfig(
                enable_dype=True,
                base_resolution=preset_config.base_resolution,
                method=preset_config.method,
                dype_scale=preset_config.dype_scale,
                dype_exponent=preset_config.dype_exponent,
                dype_start_sigma=preset_config.dype_start_sigma,
            )

        return None

    def _run_diffusion(self, context: InvocationContext):
        # ... bestehender Code ...

        # NEU: DyPE Extension vorbereiten
        dype_config = self._get_dype_config()
        dype_extension = None
        if dype_config is not None:
            dype_extension = DyPEExtension(
                config=dype_config,
                target_height=self.height,
                target_width=self.width,
            )
            context.logger.info(
                f"DyPE enabled: {self.width}x{self.height}, scale={dype_config.dype_scale}"
            )

        # ... in denoise() Aufruf einfügen ...
        x = denoise(
            model=transformer,
            # ... bestehende Parameter ...
            dype_extension=dype_extension,  # NEU
        )
```

### 3.3 FLUX Schnell Unterstützung

FLUX Schnell wird unterstützt mit angepasster Basisauflösung:

```python
def _get_dype_config(self) -> DyPEConfig | None:
    # ... bestehender Code ...

    # FLUX Schnell hat gleiche Trainingsauflösung wie Dev
    base_resolution = 1024  # Gilt für beide Varianten

    if self.dype_preset == DyPEPreset.AUTO:
        return get_dype_config_for_resolution(
            self.width, self.height, base_resolution=base_resolution
        )
    # ...
```

**Aufwand:** ~80 Zeilen Änderungen, 0.5 Tag

---

## Phase 4: Tests & Dokumentation

### 4.1 Unit Tests

**Datei:** `tests/backend/flux/dype/test_dype.py`

```python
class TestDyPERope:
    def test_rope_dype_basic(self):
        """Test dass DyPE-RoPE gleiche Shape wie original hat."""
        ...

    def test_vision_yarn_scaling(self):
        """Test Vision-YARN Skalierungsberechnung."""
        ...

    def test_timestep_modulation(self):
        """Test dass Skalierung sich mit Timestep ändert."""
        ...

class TestDyPEEmbedND:
    def test_forward_shape(self):
        """Test Output-Shape."""
        ...

    def test_step_state_update(self):
        """Test Step-State Aktualisierung."""
        ...

class TestDyPEExtension:
    def test_model_patching(self):
        """Test dass Modell korrekt gepatcht wird."""
        ...

    def test_restore_after_patching(self):
        """Test dass Original-Embedder wiederhergestellt wird."""
        ...
```

**Aufwand:** ~150 Zeilen, 1 Tag

### 4.2 Integration Tests

**Datei:** `tests/app/invocations/test_flux_dype.py`

```python
class TestFluxDyPEIntegration:
    def test_dype_node_output(self):
        """Test DyPE Node erzeugt gültige Config."""
        ...

    def test_flux_denoise_with_dype(self):
        """Test FLUX Denoise mit DyPE-Config."""
        ...

    def test_high_resolution_generation(self):
        """Test Generierung bei 2048x2048."""
        ...
```

**Aufwand:** ~100 Zeilen, 0.5 Tag

---

## Zusammenfassung

| Phase | Komponente | Aufwand | Priorität |
|-------|------------|---------|-----------|
| 1.1 | DyPE Basis-Modul | 1-2 Tage | Hoch |
| 1.2 | DyPE RoPE-Funktion | 0.5-1 Tag | Hoch |
| 1.3 | DyPE EmbedND | 0.5 Tag | Hoch |
| 2.1 | DyPE Extension | 0.5 Tag | Hoch |
| 2.2 | denoise.py Integration | 0.5 Tag | Hoch |
| 3.1 | DyPE Presets | 0.5 Tag | Mittel |
| 3.2 | FluxDenoise Integration | 0.5 Tag | Mittel |
| 3.3 | FLUX Schnell Support | 0.25 Tag | Mittel |
| 4.1 | Unit Tests | 1 Tag | Mittel |
| 4.2 | Integration Tests | 0.5 Tag | Niedrig |

**Gesamtaufwand:** 5-7 Entwicklertage

---

## Dateien zu erstellen/ändern

### Neue Dateien (7):
- `invokeai/backend/flux/dype/__init__.py`
- `invokeai/backend/flux/dype/base.py` - DyPEConfig Dataclass
- `invokeai/backend/flux/dype/rope.py` - rope_dype() Funktion
- `invokeai/backend/flux/dype/embed.py` - DyPEEmbedND Klasse
- `invokeai/backend/flux/dype/presets.py` - DyPEPreset Enum, 4K Preset
- `invokeai/backend/flux/extensions/dype_extension.py`
- `tests/backend/flux/dype/test_dype.py`

### Zu ändernde Dateien (2):
- `invokeai/backend/flux/denoise.py` - DyPE Extension Parameter
- `invokeai/app/invocations/flux_denoise.py` - DyPE Preset + Parameter direkt integriert

---

## Risiken & Mitigation

1. **Performance-Impact:** DyPE berechnet RoPE pro Step neu
   - *Mitigation:* Caching der Skalierungsfaktoren, nur Neuberechnung bei Änderung

2. **Speicherverbrauch bei 4K+:** Höhere Auflösungen brauchen mehr VRAM
   - *Mitigation:* Warnung im UI, empfohlene Limits dokumentieren

3. **Kompatibilität mit Extensions:** ControlNet, IP-Adapter etc.
   - *Mitigation:* Testen mit allen bestehenden FLUX Extensions

4. **Z-Image/Lumina 2 Support:** Andere Architektur als FLUX
   - *Mitigation:* Phase 1 nur FLUX, Z-Image als Follow-up

---

## Entscheidungen

| Frage | Entscheidung |
|-------|--------------|
| Integration | Direkt in FluxDenoise (kein separater Node) |
| Presets | Ja - 4K Preset + "auto" Modus |
| FLUX Schnell | Wird unterstützt (gleiche Basisauflösung wie Dev) |

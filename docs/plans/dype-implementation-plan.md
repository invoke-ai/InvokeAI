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

## Phase 3: Node/Invocation

### 3.1 DyPE Configuration Node

**Datei:** `invokeai/app/invocations/flux_dype.py`

```python
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import InputField, OutputField

class DyPEMethod(str, Enum):
    VISION_YARN = "vision_yarn"
    YARN = "yarn"
    NTK = "ntk"
    BASE = "base"

@invocation_output("dype_config_output")
class DyPEConfigOutput(BaseInvocationOutput):
    """Output für DyPE-Konfiguration."""
    dype_config: DyPEConfigField = OutputField(description="DyPE configuration")

@invocation(
    "flux_dype",
    title="FLUX DyPE (Dynamic Position Extrapolation)",
    tags=["flux", "dype", "upscale", "resolution"],
    category="flux",
    version="1.0.0",
)
class FluxDyPEInvocation(BaseInvocation):
    """Konfiguriert Dynamic Position Extrapolation für FLUX-Modelle.

    Ermöglicht die Generierung von Bildern über der Trainingsauflösung (z.B. 4K+)
    ohne typische Artefakte.
    """

    enable_dype: bool = InputField(
        default=True,
        description="Enable Dynamic Position Extrapolation"
    )

    base_resolution: int = InputField(
        default=1024,
        ge=256,
        le=4096,
        description="Native training resolution of the model"
    )

    method: DyPEMethod = InputField(
        default=DyPEMethod.VISION_YARN,
        description="Position extrapolation method. vision_yarn recommended for images."
    )

    dype_scale: float = InputField(
        default=2.0,
        ge=0.0,
        le=8.0,
        description="DyPE magnitude (λs). Higher = stronger extrapolation."
    )

    dype_exponent: float = InputField(
        default=2.0,
        ge=0.0,
        le=1000.0,
        description="DyPE decay speed (λt). Controls transition from low to high frequency."
    )

    dype_start_sigma: float = InputField(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Sigma threshold to start DyPE decay."
    )

    def invoke(self, context: InvocationContext) -> DyPEConfigOutput:
        config = DyPEConfig(
            enable_dype=self.enable_dype,
            base_resolution=self.base_resolution,
            method=self.method.value,
            dype_scale=self.dype_scale,
            dype_exponent=self.dype_exponent,
            dype_start_sigma=self.dype_start_sigma,
        )

        # Speichere Config im Context
        config_name = context.tensors.save(config)  # oder eigener Storage

        return DyPEConfigOutput(
            dype_config=DyPEConfigField(config_name=config_name)
        )
```

**Aufwand:** ~100 Zeilen, 0.5 Tag

### 3.2 Integration in FluxDenoiseInvocation

**Datei:** `invokeai/app/invocations/flux_denoise.py`

Änderungen:

```python
@invocation(
    "flux_denoise",
    title="FLUX Denoise",
    # ...
    version="4.2.0",  # Version bump
)
class FluxDenoiseInvocation(BaseInvocation):
    # ... bestehende Felder ...

    # NEU:
    dype_config: DyPEConfigField | None = InputField(
        default=None,
        description="DyPE configuration for high-resolution generation",
        input=Input.Connection,
    )

    def _run_diffusion(self, context: InvocationContext):
        # ... bestehender Code ...

        # NEU: DyPE Extension vorbereiten
        dype_extension = None
        if self.dype_config is not None:
            config = context.tensors.load(self.dype_config.config_name)
            dype_extension = DyPEExtension(
                config=config,
                target_height=self.height,
                target_width=self.width,
            )

        # ... in denoise() Aufruf einfügen ...
        x = denoise(
            model=transformer,
            # ... bestehende Parameter ...
            dype_extension=dype_extension,  # NEU
        )
```

**Aufwand:** ~20 Zeilen Änderungen, 0.5 Tag

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
| 3.1 | DyPE Node | 0.5 Tag | Mittel |
| 3.2 | FluxDenoise Integration | 0.5 Tag | Mittel |
| 4.1 | Unit Tests | 1 Tag | Mittel |
| 4.2 | Integration Tests | 0.5 Tag | Niedrig |

**Gesamtaufwand:** 5-7 Entwicklertage

---

## Dateien zu erstellen/ändern

### Neue Dateien:
- `invokeai/backend/flux/dype/__init__.py`
- `invokeai/backend/flux/dype/base.py`
- `invokeai/backend/flux/dype/rope.py`
- `invokeai/backend/flux/dype/embed.py`
- `invokeai/backend/flux/extensions/dype_extension.py`
- `invokeai/app/invocations/flux_dype.py`
- `invokeai/app/invocations/fields/dype_fields.py`
- `tests/backend/flux/dype/test_dype.py`
- `tests/app/invocations/test_flux_dype.py`

### Zu ändernde Dateien:
- `invokeai/backend/flux/denoise.py` - DyPE Extension Parameter
- `invokeai/app/invocations/flux_denoise.py` - DyPE Config Input
- `invokeai/app/invocations/fields/__init__.py` - DyPE Field Export

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

## Offene Fragen

1. Soll DyPE als separater Node oder als Parameter in FluxDenoise integriert werden?
2. Sollen Preset-Konfigurationen (z.B. "4K optimiert") angeboten werden?
3. Wie soll mit FLUX Schnell umgegangen werden (hat andere Trainingsauflösung)?

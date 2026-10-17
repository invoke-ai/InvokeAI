"""The manifest of one profile: the server to launch, the cases to run, and how they expand into planned steps."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from scripts.family_e2e.records import PlannedStep, Thresholds

TOOL_CHECKOUT = Path(__file__).resolve().parents[2]
DEFAULT_ARM = "default"
# Profile, family and case names end up in directory names and record ids (`<case_key>.<arm>.<kind>...`).
_NAME = r"^[A-Za-z0-9][A-Za-z0-9_-]*$"

Mode = Literal["txt2img", "img2img", "controlnet"]


class ManifestError(ValueError):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ServerSpec(_Strict):
    template_root: Path
    python: Path
    code_dir: Path = TOOL_CHECKOUT
    env: dict[str, str] = {}


class Defaults(_Strict):
    warmup: int = Field(default=1, ge=1)
    repeats: int = Field(default=3, ge=1)
    seed: int = 1234
    prompt: str
    negative_prompt: str | None = None
    psnr_min_db: float = 30.0
    time_tolerance: float = Field(default=0.10, gt=0)
    peak_tolerance_mb: float = Field(default=256.0, gt=0)
    run_timeout_s: float = Field(default=1800.0, gt=0)


class LoraSpec(_Strict):
    key: str
    weight: float = 1.0


class InitImageSpec(_Strict):
    path: str
    strength: float = Field(gt=0, le=1)


class ControlSpec(_Strict):
    model: str
    image: str
    weight: float = 1.0
    begin: float = Field(default=0.0, ge=0, le=1)
    end: float = Field(default=1.0, ge=0, le=1)


class CaseSpec(_Strict):
    id: str = Field(pattern=_NAME)
    model: str
    components: dict[str, str] = {}
    mode: Mode = "txt2img"
    sizes: list[tuple[int, int]] = Field(min_length=1)
    steps: int = Field(gt=0)
    seed: int | None = None
    prompt: str | None = None
    loras: list[LoraSpec] = []
    reference_images: list[str] = []
    init_image: InitImageSpec | None = None
    control: list[ControlSpec] = []
    settings: dict[str, Any] = {}
    psnr_min_db: float | None = None
    time_tolerance: float | None = Field(default=None, gt=0)
    peak_tolerance_mb: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check(self) -> CaseSpec:
        for width, height in self.sizes:
            if width <= 0 or height <= 0 or width % 8 or height % 8:
                raise ValueError(f"size {width}x{height} must be positive multiples of 8")
        if len(set(self.sizes)) != len(self.sizes):
            raise ValueError("sizes must be unique")
        # A field the mode does not use would be silently ignored by the exporter.
        if (self.mode == "img2img") != (self.init_image is not None):
            raise ValueError("init_image is required for mode img2img and allowed only there")
        if (self.mode == "controlnet") != bool(self.control):
            raise ValueError("control is required for mode controlnet and allowed only there")
        return self

    def input_paths(self) -> list[str]:
        paths = list(self.reference_images) + [c.image for c in self.control]
        if self.init_image is not None:
            paths.append(self.init_image.path)
        return paths


class FamilySpec(_Strict):
    family: str = Field(pattern=_NAME)
    cases: list[CaseSpec] = Field(min_length=1)


class Manifest(_Strict):
    profile: str = Field(pattern=_NAME)
    data_dir: Path
    gpu: str = Field(min_length=1, description="Substring of the adapter description, e.g. 'RTX 4090'")
    server: ServerSpec
    defaults: Defaults
    families: list[FamilySpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_names(self) -> Manifest:
        families = [f.family for f in self.families]
        if len(set(families)) != len(families):
            raise ValueError("family names must be unique")
        ids = [c.id for f in self.families for c in f.cases]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"case ids must be unique across the manifest: {', '.join(duplicates)}")
        return self

    @property
    def inputs_dir(self) -> Path:
        return self.data_dir / "inputs"

    def input_file(self, relative: str) -> Path:
        """The file a manifest input path names: under `<data_dir>/inputs`, never outside it."""
        inputs = self.inputs_dir.resolve()
        path = (inputs / relative).resolve()
        if not path.is_relative_to(inputs):
            raise ManifestError(f"input {relative!r} resolves outside {inputs}")
        return path


@dataclass(frozen=True)
class ResolvedCase:
    """One case at one size: the unit graphs, references and thresholds are keyed by."""

    case_key: str
    family: str
    spec: CaseSpec
    width: int
    height: int
    seed: int
    prompt: str
    negative_prompt: str | None
    thresholds: Thresholds


def case_key(case_id: str, width: int, height: int) -> str:
    return f"{case_id}@{width}x{height}"


def load_manifest(path: Path) -> Manifest:
    """Reads and validates a manifest. Relative paths in it are relative to the manifest file; inputs must exist."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise ManifestError(f"cannot read manifest {path}: {e}") from e
    if not isinstance(raw, dict):
        raise ManifestError(f"manifest {path} must be a mapping")
    try:
        manifest = Manifest.model_validate(raw)
    except ValidationError as e:
        raise ManifestError(f"invalid manifest {path}:\n{e}") from e

    base = path.resolve().parent
    server = manifest.server
    manifest = manifest.model_copy(
        update={
            "data_dir": base / manifest.data_dir,
            "server": server.model_copy(
                update={
                    "template_root": base / server.template_root,
                    "python": base / server.python,
                    "code_dir": base / server.code_dir,
                }
            ),
        }
    )
    missing = [
        f"{case.id}: {relative}"
        for family in manifest.families
        for case in family.cases
        for relative in case.input_paths()
        if not manifest.input_file(relative).is_file()
    ]
    if missing:
        raise ManifestError(f"missing input files under {manifest.inputs_dir}:\n" + "\n".join(missing))
    return manifest


def resolve_cases(manifest: Manifest, families: Sequence[str] = (), case_ids: Sequence[str] = ()) -> list[ResolvedCase]:
    """The selected cases in manifest order: family order, then case order, then size order.

    A case is selected when it passes both filters; an empty filter passes everything.
    """
    known_families = {f.family for f in manifest.families}
    known_cases = {c.id for f in manifest.families for c in f.cases}
    unknown = [f for f in families if f not in known_families] + [c for c in case_ids if c not in known_cases]
    if unknown:
        raise ManifestError(
            f"unknown family or case: {', '.join(unknown)} "
            f"(families: {', '.join(sorted(known_families))}; cases: {', '.join(sorted(known_cases))})"
        )
    defaults = manifest.defaults
    resolved: list[ResolvedCase] = []
    for family in manifest.families:
        if families and family.family not in families:
            continue
        for case in family.cases:
            if case_ids and case.id not in case_ids:
                continue
            thresholds = Thresholds(
                psnr_min_db=defaults.psnr_min_db if case.psnr_min_db is None else case.psnr_min_db,
                time_tolerance=defaults.time_tolerance if case.time_tolerance is None else case.time_tolerance,
                peak_tolerance_mb=(
                    defaults.peak_tolerance_mb if case.peak_tolerance_mb is None else case.peak_tolerance_mb
                ),
            )
            for width, height in case.sizes:
                resolved.append(
                    ResolvedCase(
                        case_key=case_key(case.id, width, height),
                        family=family.family,
                        spec=case,
                        width=width,
                        height=height,
                        seed=defaults.seed if case.seed is None else case.seed,
                        prompt=defaults.prompt if case.prompt is None else case.prompt,
                        negative_prompt=defaults.negative_prompt,
                        thresholds=thresholds,
                    )
                )
    if not resolved:
        raise ManifestError("the family/case filters select no case")
    return resolved


def plan_steps(cases: list[ResolvedCase], warmup: int, repeats: int, arm: str = DEFAULT_ARM) -> list[PlannedStep]:
    """Warm-ups, then measures, per case key, in case order."""
    steps: list[PlannedStep] = []
    for case in cases:
        for kind, count in (("warmup", warmup), ("measure", repeats)):
            steps.extend(
                PlannedStep(
                    case_key=case.case_key, case_id=case.spec.id, family=case.family, arm=arm, kind=kind, rep=rep
                )
                for rep in range(count)
            )
    return steps

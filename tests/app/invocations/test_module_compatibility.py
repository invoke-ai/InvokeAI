"""Compatibility contract for moved invocation modules."""

import json
import subprocess
import sys

import pytest

from invokeai.app.invocations._legacy_module_compatibility import LEGACY_MODULE_MAP


def _run(program: str) -> dict[str, object]:
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr[-4000:]}"
    return json.loads(result.stdout)


def test_compatibility_map_covers_all_moved_modules() -> None:
    assert len(LEGACY_MODULE_MAP) == 106
    assert len(set(LEGACY_MODULE_MAP.values())) == 106


def test_every_legacy_module_imports_and_aliases_canonical() -> None:
    result = _run(
        f"""
import importlib
import json
import logging
import warnings

records = []
handler = logging.Handler()
handler.emit = lambda record: records.append(record.getMessage())
logger = logging.getLogger("InvokeAI")
logger.addHandler(handler)
logger.setLevel(logging.INFO)
mapping = {dict(LEGACY_MODULE_MAP)!r}
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    aliases = {{}}
    for legacy, canonical in mapping.items():
        old = importlib.import_module(legacy)
        repeat = importlib.import_module(legacy)
        aliases[legacy] = [old is importlib.import_module(canonical), repeat is old]
print(json.dumps({{"aliases": aliases, "records": records, "deprecations": [
    {{"message": str(w.message), "filename": str(w.filename)}}
    for w in caught if issubclass(w.category, DeprecationWarning)
]}}))
"""
    )
    assert result["aliases"] == {legacy: [True, True] for legacy in LEGACY_MODULE_MAP}
    assert len(result["records"]) == len(LEGACY_MODULE_MAP)
    for legacy, canonical in LEGACY_MODULE_MAP.items():
        assert sum(legacy in record and canonical in record for record in result["records"]) == 1
    assert not [warning for warning in result["deprecations"] if "invokeai/app/invocations" in warning["filename"]]


def test_canonical_import_silent_unknown_path_stays_missing() -> None:
    result = _run(
        """
import importlib
import json
import logging

records = []
handler = logging.Handler()
handler.emit = lambda record: records.append(record.getMessage())
logger = logging.getLogger("InvokeAI")
logger.addHandler(handler)
logger.setLevel(logging.WARNING)
importlib.import_module("invokeai.app.invocations.flux.flux_denoise")
try:
    importlib.import_module("invokeai.app.invocations.not_a_moved_module")
except ModuleNotFoundError:
    missing = True
else:
    missing = False
print(json.dumps({"missing": missing, "records": records}))
"""
    )
    assert result == {"missing": True, "records": []}


@pytest.mark.parametrize(
    ("legacy", "symbol"),
    [
        ("anima_denoise", "loglinear_timestep_shift"),
        ("cogview4_denoise", "CogView4DenoiseInvocation"),
        ("ernie_image_denoise", "ErnieImageDenoiseInvocation"),
        ("flux_denoise", "FluxDenoiseInvocation"),
        ("flux2_denoise", "Flux2DenoiseInvocation"),
        ("ideogram4_caption", "Ideogram4Region"),
        ("krea2_denoise", "Krea2DenoiseInvocation"),
        ("minimax_h3_denoise", "MiniMaxH3DenoiseOutput"),
        ("flux2_pid_decode", "Flux2PiDDecodeInvocation"),
        ("qwen_image_denoise", "QwenImageDenoiseInvocation"),
        ("controlnet", "ControlField"),
        ("sd3_denoise", "SD3DenoiseInvocation"),
        ("anima_text_encoder", "AnimaTextEncoderInvocation"),
        ("anima_image_to_latents", "AnimaImageToLatentsInvocation"),
        ("wan_denoise", "WanDenoiseInvocation"),
        ("z_image_denoise", "ZImageDenoiseInvocation"),
    ],
)
def test_representative_public_symbols_keep_identity(legacy: str, symbol: str) -> None:
    old = f"invokeai.app.invocations.{legacy}"
    canonical = LEGACY_MODULE_MAP[old]
    result = _run(
        f"import importlib, json; old = importlib.import_module({old!r}); new = importlib.import_module({canonical!r}); assert getattr(old, {symbol!r}) is getattr(new, {symbol!r}); print(json.dumps({{}}))"
    )
    assert result == {}


def test_flux2_klein_text_encoder_legacy_import_explicitly_works() -> None:
    legacy = "invokeai.app.invocations.flux2_klein_text_encoder"
    canonical = LEGACY_MODULE_MAP[legacy]
    assert canonical == "invokeai.app.invocations.text_encoder.flux2_klein_text_encoder"
    result = _run(
        f"import importlib, json; old = importlib.import_module({legacy!r}); new = importlib.import_module({canonical!r}); assert old.Flux2KleinTextEncoderInvocation is new.Flux2KleinTextEncoderInvocation; print(json.dumps({{}}))"
    )
    assert result == {}


def test_core_discovery_excludes_shims_and_emits_no_notice() -> None:
    result = _run(
        f"""
import json
import logging
from invokeai.app.invocations import load_all_modules
from invokeai.app.invocations.baseinvocation import InvocationRegistry

records = []
handler = logging.Handler()
handler.emit = lambda record: records.append(record.getMessage())
logger = logging.getLogger("InvokeAI")
logger.addHandler(handler)
logger.setLevel(logging.INFO)
legacy_names = {set(LEGACY_MODULE_MAP)!r}
modules = load_all_modules()
types = list(InvocationRegistry.get_invocation_types())
print(json.dumps({{"legacy": sorted(set(modules) & legacy_names),
                  "duplicate_types": len(types) - len(set(types)), "records": records}}))
"""
    )
    assert result["legacy"] == []
    assert result["duplicate_types"] == 0
    assert not any("Legacy invocation module" in record for record in result["records"])

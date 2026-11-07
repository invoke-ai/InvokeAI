"""A loader that folds or casts float8 weights must read its file through `load_safetensors_checkpoint`.

`load_file` maps the whole file, so the bf16 tensors that stay views after the float8 ones are widened pin
it for the model's lifetime (see `safetensors_backend`). Which loaders widen float8 is not declared anywhere;
what they have in common is that they reach one of the fold or cast helpers, so this reads the loaders'
source for a function that calls one of those and also a raw `load_file`, parsed rather than grepped so a
comment mentioning either cannot flag or excuse anything.
"""

import ast
import inspect
from pathlib import Path

import pytest

import invokeai.backend.model_manager.load.model_loaders as model_loaders

_FOLDS = {
    "extract_fp8_scaled_layers",
    "cast_state_dict",
    "dequantize_fp8_scaled",
    "_keep_fp8_weights",
    "_dequantize_fp8_weights",
    "_fold_comfy_scaled_weights",
}

#: (file, function) pairs that read with a raw `load_file` and reach a fold, each with the reason it is safe.
_ALLOWED: dict[tuple[str, str], str] = {
    ("ltx2.py", "_load_transformer_from_file"): "refuses float8 weights (reject_float8_weights) before any fold",
    ("ltx2.py", "_load_text_encoder"): "refuses float8 weights (reject_float8_weights) before any cast",
}


def _called_names(function: ast.AST) -> set[str]:
    names = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            names.add(node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", ""))
    return names


def _raw_float8_readers() -> list[tuple[str, str]]:
    found = []
    for path in sorted(Path(inspect.getfile(model_loaders)).parent.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                called = _called_names(node)
                if "load_file" in called and called & _FOLDS:
                    found.append((path.name, node.name))
    return found


@pytest.mark.platform_independent
def test_no_loader_folds_float8_weights_it_read_with_a_raw_load_file() -> None:
    offenders = [site for site in _raw_float8_readers() if site not in _ALLOWED]

    assert offenders == [], (
        "These functions fold or cast weights after a raw `load_file`; read through "
        f"`load_safetensors_checkpoint` or allowlist them with a reason: {offenders}"
    )

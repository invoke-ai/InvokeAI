"""One shared policy for the key lists that ``nn.Module.load_state_dict`` reports.

Checkpoints in the wild routinely carry tensors the model has nowhere to put: runtime-derived
buffers an exporter happened to serialize, bookkeeping the training rig wrote out, leftovers from a
merge. None of them are weights the model needs, and none of them say anything about whether the
weights that *are* present are correct — the official checkpoint for the same architecture loads
fine without them.

Historically each single-file loader made its own call about that, and the loaders that hard-failed
turned every such extra tensor into a user-facing crash that could only be cleared by adding one
more entry to an allowlist and cutting a release (see issue #9437). So the policy here is:

* **Unexpected keys are ignored.** They are logged at ``DEBUG`` — visible when someone turns the log
  level up to diagnose a load, silent otherwise — and never raise.

  The one deliberate exception in the codebase is
  ``model_loaders/wan.py::_raise_for_incompatible_keys``, and it is worth knowing why it is not a
  counter-example to the policy: several Wan 2.2 derivatives (Animate, S2V, Fun-Camera) are
  *supersets* of the plain transformer, so their extra ``audio_injector``/``face_adapter``/
  ``control_adapter`` branches are not exporter noise but the entire feature the checkpoint exists
  for, and running one without them generates silently degraded output. That loader strips the
  genuinely benign extras first, so what reaches its check is a named-variant signal, not junk.
* **Missing keys are still an error** where the caller says the state dict is supposed to be
  complete. A parameter the checkpoint never filled is a real defect: it stays on the meta device
  (or, under ``skip_torch_weight_init()``, holds uninitialised memory) and blows up mid-inference,
  far from the cause.
"""

from collections.abc import Iterable, Mapping
from typing import Any

import torch

from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.get_logger(__name__)

# Enough keys to recognize what the extras are without dumping hundreds of lines for a bundled VAE.
MAX_REPORTED_KEYS = 10


def _format_keys(keys: list[str]) -> str:
    shown = keys[:MAX_REPORTED_KEYS]
    suffix = f" (+{len(keys) - len(shown)} more)" if len(keys) > len(shown) else ""
    return f"{shown}{suffix}"


def log_unexpected_keys(source: str, unexpected_keys: Iterable[Any]) -> None:
    """Report keys that ``load_state_dict`` had nowhere to put, at DEBUG level only.

    Args:
        source: Human-readable name of what was being loaded, e.g. "Anima transformer checkpoint".
        unexpected_keys: The ``unexpected_keys`` reported by ``load_state_dict(strict=False)``.
    """
    # Not every key is guaranteed to be a string: a `.pth` unpickles to whatever it contains.
    keys = sorted(str(key) for key in unexpected_keys)
    if not keys:
        return
    logger.debug(f"{source}: ignoring {len(keys)} key(s) not present in the model: {_format_keys(keys)}")


def reject_incomplete_load(model: torch.nn.Module, *, what: str) -> None:
    """Raise if a ``load_state_dict(strict=False)`` left required tensors on the meta device.

    The completeness check for loaders that cannot use ``load_state_dict_ignoring_extras``'s
    missing-key check — because they legitimately tolerate *some* missing keys, or because the key
    list cannot see the problem. It is strictly stronger than ``missing_keys`` for models built
    under ``accelerate.init_empty_weights()``: it is immune to non-persistent buffers (which never
    appear in ``missing_keys`` at all) and to tied weights (materialized by ``tie_weights()`` after
    the load), and it catches the real failure — a required tensor that was never filled and would
    otherwise blow up mid-inference with "Cannot copy out of meta tensor" instead of at load time.

    Buffers are checked as well as parameters. ``accelerate.init_empty_weights()`` defaults to
    ``include_buffers=False``, so a module's constructor normally materializes them — but a loader
    that opts into ``include_buffers=True``, or that builds the module with ``to_empty()``, leaves
    persistent buffers on meta, and a checkpoint omitting one would slip past a parameters-only
    guard.

    Args:
        model: The module that was just loaded into.
        what: Human-readable name of what was loaded, used in the error message.

    Raises:
        RuntimeError: If any parameter or buffer is still on the meta device.
    """
    still_meta = [
        name
        for name, tensor in (*model.named_parameters(), *model.named_buffers())
        if getattr(tensor, "is_meta", False)
    ]
    if still_meta:
        raise RuntimeError(
            f"{what} is incomplete: {len(still_meta)} tensor(s) were not provided by the checkpoint "
            f"and remain uninitialized (meta device). First few: {still_meta[:8]}. The file is likely "
            "incomplete, misidentified, or uses a key layout that needs conversion."
        )


def load_state_dict_ignoring_extras(
    model: torch.nn.Module,
    state_dict: Mapping[str, Any],
    *,
    source: str,
    assign: bool = False,
    allow_missing: bool = False,
    allowed_missing: Iterable[str] = (),
) -> list[str]:
    """Load ``state_dict`` into ``model``, ignoring keys the model has nowhere to put.

    A drop-in replacement for ``model.load_state_dict(state_dict, strict=True)`` that keeps the
    strictness that matters (every required parameter must be filled) and drops the strictness that
    only produces whack-a-mole (the checkpoint must contain nothing else). Shape mismatches still
    raise from torch, exactly as they do under ``strict=True``.

    Args:
        model: The module to load into.
        state_dict: The state dict to load.
        source: Human-readable name of what is being loaded, used in the log line and the error.
        assign: Passed through to ``load_state_dict``.
        allow_missing: When True, missing keys are tolerated too — for callers that fill them in
            afterwards (tied weights, re-initialised buffers) or that run their own completeness
            check, such as a sweep for tensors left on the meta device.
        allowed_missing: Specific keys that are expected to be absent and must not raise. Ignored
            when ``allow_missing`` is True.

    Returns:
        The missing keys, so callers can materialize what they said they would.

    Raises:
        RuntimeError: If a required key was missing from ``state_dict``.
    """
    incompatible_keys = model.load_state_dict(state_dict, strict=False, assign=assign)
    log_unexpected_keys(source, incompatible_keys.unexpected_keys)

    missing_keys = [str(key) for key in incompatible_keys.missing_keys]
    if not allow_missing:
        allowed = set(allowed_missing)
        required_missing = sorted(key for key in missing_keys if key not in allowed)
        if required_missing:
            raise RuntimeError(
                f"{source} is missing {len(required_missing)} parameter(s) that the model requires: "
                f"{_format_keys(required_missing)}. The checkpoint is likely incomplete, misidentified, "
                "or uses a key layout that needs conversion."
            )
    return missing_keys

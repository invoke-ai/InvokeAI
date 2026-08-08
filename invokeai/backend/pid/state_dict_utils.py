# SPDX-License-Identifier: Apache-2.0
"""The key-space normalisation shared by PiD identification and PiD loading.

NVIDIA's official `.pth` checkpoints serialise `PidDistillModel`, which keeps the student network
under a `net.` prefix and carries the other distill submodules (the EMA copy, the fake-score net,
the discriminator) alongside it. Two consumers have to reduce a checkpoint to the same key space:

- `model_manager/configs/pid_decoder.py`, to decide whether a file is a usable PiD decoder;
- `model_manager/load/model_loaders/pid_decoder.py`, which feeds `PidNet.load_state_dict`.

They used to carry a copy each, and the copies had already diverged (only the loader dropped the
distill-only submodules). The failure mode of that drift is silent in the direction that matters:
identification accepting a file the loader then refuses. Hence one implementation, here.

Deliberately dependency-free — the config imports it during model identification, where pulling in
the vendored PiD network stack would be wasted work.
"""

from typing import Any, TypeVar

# NVIDIA's official PiD `.pth` checkpoints store the student under this prefix (see
# `PidDistillModel.state_dict(prefix="net.")` in the vendored upstream).
NET_PREFIX = "net."

# Sibling submodules of the distillation setup. They are not part of PidNet, and some of them shadow
# its parameter names (`net_ema.lq_proj.…`), so they must be dropped rather than renamed.
_DISTILL_ONLY_PREFIXES = ("net_ema.", "fake_score.", "discriminator.")

T = TypeVar("T")


def _normalized_key(key: Any) -> str | None:
    """The key as `PidNet` sees it, or None if it belongs to a distill-only submodule."""
    if not isinstance(key, str):
        return None
    if key.startswith(NET_PREFIX):
        return key[len(NET_PREFIX) :]
    if key.startswith(_DISTILL_ONLY_PREFIXES):
        return None
    return key


def has_net_prefix(state_dict: dict[Any, Any]) -> bool:
    """Whether this checkpoint is a `PidDistillModel` serialisation rather than a bare `PidNet`."""
    return any(isinstance(k, str) and k.startswith(NET_PREFIX) for k in state_dict)


def strip_net_prefix(state_dict: dict[Any, T]) -> dict[str, T]:
    """Reduce a checkpoint to `PidNet`'s own key space.

    A checkpoint with no `net.` prefix is already in that space and is returned untouched — the
    distill-only filter is not applied there, because without the prefix there is no evidence this
    is a distill serialisation, and a stray `discriminator.*` key should reach the loader's
    "unexpected keys" check rather than be silently dropped.
    """
    if not has_net_prefix(state_dict):
        return state_dict  # type: ignore[return-value]
    return {nk: v for k, v in state_dict.items() if (nk := _normalized_key(k)) is not None}


def pid_net_keys(state_dict: dict[Any, Any]) -> set[str]:
    """The checkpoint's keys as `PidNet` will see them. Mirrors :func:`strip_net_prefix` exactly."""
    return set(strip_net_prefix(state_dict))

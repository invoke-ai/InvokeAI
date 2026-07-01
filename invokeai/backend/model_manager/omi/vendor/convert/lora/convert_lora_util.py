import torch
from torch import Tensor
from typing_extensions import Self


class LoraConversionKeySet:
    def __init__(
        self,
        omi_prefix: str,
        diffusers_prefix: str,
        legacy_diffusers_prefix: str | None = None,
        parent: Self | None = None,
        swap_chunks: bool = False,
        filter_is_last: bool | None = None,
        next_omi_prefix: str | None = None,
        next_diffusers_prefix: str | None = None,
    ):
        if parent is not None:
            self.omi_prefix = combine(parent.omi_prefix, omi_prefix)
            self.diffusers_prefix = combine(parent.diffusers_prefix, diffusers_prefix)
        else:
            self.omi_prefix = omi_prefix
            self.diffusers_prefix = diffusers_prefix

        if legacy_diffusers_prefix is None:
            self.legacy_diffusers_prefix = self.diffusers_prefix.replace(".", "_")
        elif parent is not None:
            self.legacy_diffusers_prefix = combine(parent.legacy_diffusers_prefix, legacy_diffusers_prefix).replace(
                ".", "_"
            )
        else:
            self.legacy_diffusers_prefix = legacy_diffusers_prefix

        self.parent = parent
        self.swap_chunks = swap_chunks
        self.filter_is_last = filter_is_last
        self.prefix = parent

        if next_omi_prefix is None and parent is not None:
            self.next_omi_prefix = parent.next_omi_prefix
            self.next_diffusers_prefix = parent.next_diffusers_prefix
            self.next_legacy_diffusers_prefix = parent.next_legacy_diffusers_prefix
        elif next_omi_prefix is not None and parent is not None:
            self.next_omi_prefix = combine(parent.omi_prefix, next_omi_prefix)
            self.next_diffusers_prefix = combine(parent.diffusers_prefix, next_diffusers_prefix)
            self.next_legacy_diffusers_prefix = combine(parent.legacy_diffusers_prefix, next_diffusers_prefix).replace(
                ".", "_"
            )
        elif next_omi_prefix is not None and parent is None:
            self.next_omi_prefix = next_omi_prefix
            self.next_diffusers_prefix = next_diffusers_prefix
            self.next_legacy_diffusers_prefix = next_diffusers_prefix.replace(".", "_")
        else:
            self.next_omi_prefix = None
            self.next_diffusers_prefix = None
            self.next_legacy_diffusers_prefix = None

    def __get_omi(self, in_prefix: str, key: str) -> str:
        return self.omi_prefix + key.removeprefix(in_prefix)

    def __get_diffusers(self, in_prefix: str, key: str) -> str:
        return self.diffusers_prefix + key.removeprefix(in_prefix)

    def __get_legacy_diffusers(self, in_prefix: str, key: str) -> str:
        key = self.legacy_diffusers_prefix + key.removeprefix(in_prefix)

        suffix = key[key.rfind(".") :]
        if suffix not in [".alpha", ".dora_scale"]:  # some keys only have a single . in the suffix
            suffix = key[key.removesuffix(suffix).rfind(".") :]
        key = key.removesuffix(suffix)

        return key.replace(".", "_") + suffix

    def get_key(self, in_prefix: str, key: str, target: str) -> str:
        if target == "omi":
            return self.__get_omi(in_prefix, key)
        elif target == "diffusers":
            return self.__get_diffusers(in_prefix, key)
        elif target == "legacy_diffusers":
            return self.__get_legacy_diffusers(in_prefix, key)
        return key

    def __str__(self) -> str:
        return f"omi: {self.omi_prefix}, diffusers: {self.diffusers_prefix}, legacy: {self.legacy_diffusers_prefix}"


def combine(left: str, right: str) -> str:
    left = left.rstrip(".")
    right = right.lstrip(".")
    if left == "" or left is None:
        return right
    elif right == "" or right is None:
        return left
    else:
        return left + "." + right


def map_prefix_range(
    omi_prefix: str,
    diffusers_prefix: str,
    parent: LoraConversionKeySet,
) -> list[LoraConversionKeySet]:
    # 100 should be a safe upper bound. increase if it's not enough in the future
    return [
        LoraConversionKeySet(
            omi_prefix=f"{omi_prefix}.{i}",
            diffusers_prefix=f"{diffusers_prefix}.{i}",
            parent=parent,
            next_omi_prefix=f"{omi_prefix}.{i + 1}",
            next_diffusers_prefix=f"{diffusers_prefix}.{i + 1}",
        )
        for i in range(100)
    ]


def __convert(
    state_dict: dict[str, Tensor],
    key_sets: list[LoraConversionKeySet],
    source: str,
    target: str,
) -> dict[str, Tensor]:
    out_states = {}

    if source == target:
        return dict(state_dict)

    # TODO: maybe replace with a non O(n^2) algorithm
    for key, tensor in state_dict.items():
        for key_set in key_sets:
            in_prefix = ""

            if source == "omi":
                in_prefix = key_set.omi_prefix
            elif source == "diffusers":
                in_prefix = key_set.diffusers_prefix
            elif source == "legacy_diffusers":
                in_prefix = key_set.legacy_diffusers_prefix

            if not key.startswith(in_prefix):
                continue

            if key_set.filter_is_last is not None:
                next_prefix = None
                if source == "omi":
                    next_prefix = key_set.next_omi_prefix
                elif source == "diffusers":
                    next_prefix = key_set.next_diffusers_prefix
                elif source == "legacy_diffusers":
                    next_prefix = key_set.next_legacy_diffusers_prefix

                is_last = not any(k.startswith(next_prefix) for k in state_dict)
                if key_set.filter_is_last != is_last:
                    continue

            name = key_set.get_key(in_prefix, key, target)

            can_swap_chunks = target == "omi" or source == "omi"
            if key_set.swap_chunks and name.endswith(".lora_up.weight") and can_swap_chunks:
                chunk_0, chunk_1 = tensor.chunk(2, dim=0)
                tensor = torch.cat([chunk_1, chunk_0], dim=0)

            out_states[name] = tensor

            break  # only map the first matching key set

    return out_states


def __detect_source(
    state_dict: dict[str, Tensor],
    key_sets: list[LoraConversionKeySet],
) -> str:
    omi_count = 0
    diffusers_count = 0
    legacy_diffusers_count = 0

    for key in state_dict:
        for key_set in key_sets:
            if key.startswith(key_set.omi_prefix):
                omi_count += 1
            if key.startswith(key_set.diffusers_prefix):
                diffusers_count += 1
            if key.startswith(key_set.legacy_diffusers_prefix):
                legacy_diffusers_count += 1

    if omi_count > diffusers_count and omi_count > legacy_diffusers_count:
        return "omi"
    if diffusers_count > omi_count and diffusers_count > legacy_diffusers_count:
        return "diffusers"
    if legacy_diffusers_count > omi_count and legacy_diffusers_count > diffusers_count:
        return "legacy_diffusers"

    return ""


def convert_to_omi(
    state_dict: dict[str, Tensor],
    key_sets: list[LoraConversionKeySet],
) -> dict[str, Tensor]:
    source = __detect_source(state_dict, key_sets)
    return __convert(state_dict, key_sets, source, "omi")


def convert_to_diffusers(
    state_dict: dict[str, Tensor],
    key_sets: list[LoraConversionKeySet],
) -> dict[str, Tensor]:
    source = __detect_source(state_dict, key_sets)
    return __convert(state_dict, key_sets, source, "diffusers")


def convert_to_legacy_diffusers(
    state_dict: dict[str, Tensor],
    key_sets: list[LoraConversionKeySet],
) -> dict[str, Tensor]:
    source = __detect_source(state_dict, key_sets)
    return __convert(state_dict, key_sets, source, "legacy_diffusers")

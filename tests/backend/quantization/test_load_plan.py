"""`reserve_for_load` asks for what the whole load is holding, not for half of it.

The reservation a seam makes has to cover what the load is holding. It used to be assembled at each
seam -- fourteen `predict_cast_state_dict_size` call sites and seven `predict_nvfp4_install_size`
ones -- and the term nobody could add was the side channel, because `extract_fp8_scaled_layers` pops
it out of the state dict that every prediction walks.

Each cell below isolates one term by making the others zero, so a total that came out right through
two compensating errors would still fail somewhere. The one thing not asserted here is the wiring:
whether a given loader passes the right mappings is pinned at the seams, in
`test_ideogram4_checkpoint_loader.py` and `test_qwen_image_nvfp4_loader.py`.
"""

from types import SimpleNamespace

import pytest
import torch

from invokeai.backend.quantization.fp8_scaled import (
    extract_fp8_scaled_layers,
    fold_transient_bytes,
    predict_cast_state_dict_size,
)
from invokeai.backend.quantization.load_plan import reserve_for_load, side_channel_bytes
from invokeai.backend.quantization.nvfp4 import pop_nvfp4_layers, predict_nvfp4_install_host_size
from tests.fixtures.quantized_payloads import mxfp8_marker, mxfp8_tensors, nvfp4_signed_tensors, quantize_scaled_fp8


def _linear_model(in_features: int, out_features: int) -> torch.nn.Module:
    model = torch.nn.Module()
    model.add_module("lin", torch.nn.Linear(in_features, out_features, bias=False))
    return model


def _asked() -> tuple[list[int], object]:
    """A `reserve` callable and the list it records into."""
    asked: list[int] = []
    return asked, asked.append


class TestTheSideChannelIsCounted:
    """The term the state dict cannot report, because it is no longer in the state dict."""

    def test_a_decoded_mx_grid_is_inside_the_reservation(self) -> None:
        """The finding this module was written for.

        `decode_mx_block_scales` replaces the stored `uint8` grid with a float32 one four times the
        size, inside the extraction that runs before the reservation. Measured on the released 12.6
        GiB Krea-2 build that is 1.455 GiB across 256 layers; here it is 32768 bytes beside a 524288
        byte weight, and the reservation is their sum.
        """
        exponents = (torch.arange(512 * 16, dtype=torch.int64).reshape(512, 16) % 8) + 124
        tensors, _expected = mxfp8_tensors("lin", exponents)
        sd = dict(tensors)
        model = _linear_model(16 * 32, 512)
        layers = extract_fp8_scaled_layers(sd, layer_hints={"lin": mxfp8_marker()})
        grid = layers["lin"].weight_scale
        asked, reserve = _asked()

        needed = reserve_for_load(
            reserve, sd, torch.bfloat16, keep_fp8=True, model=model, fp8_layers=layers, nvfp4_payloads={}
        )

        assert "lin.weight_scale" not in sd
        assert grid.dtype is torch.float32
        assert needed == sd["lin.weight"].nelement() * 2 + grid.nelement() * grid.element_size()
        assert asked == [needed]

    def test_a_per_tensor_scale_is_counted_too(self) -> None:
        """Four bytes, counted for the same reason. A reservation that special-cased MXFP8 would be
        answering about one scheme rather than about the mapping it was handed."""
        payload = quantize_scaled_fp8(torch.arange(32, dtype=torch.float32).reshape(4, 8))
        layers = extract_fp8_scaled_layers({"lin.weight": payload.codes, "lin.weight_scale": payload.scale})

        # The magnitude only. What the total does with it is `TestKeepFp8IsCarried`'s subject, on
        # this same fixture, so asserting it here as well would be the same cell twice.
        assert side_channel_bytes(layers) == payload.scale.nelement() * payload.scale.element_size()

    def test_an_activation_scale_is_counted_beside_its_weight_scale(self) -> None:
        """Encoder builds ship both halves, and both are resident from extraction onward."""
        payload = quantize_scaled_fp8(torch.arange(32, dtype=torch.float32).reshape(4, 8))
        layers = extract_fp8_scaled_layers(
            {
                "lin.weight": payload.codes,
                "lin.weight_scale": payload.scale,
                "lin.input_scale": torch.tensor(0.5),
            }
        )

        assert layers["lin"].input_scale is not None
        assert side_channel_bytes(layers) == payload.scale.nelement() * payload.scale.element_size() + 4

    def test_a_scale_the_producer_did_not_ship_costs_nothing_and_nothing_else_is_absorbed(self) -> None:
        """`None` is the one absence charged as zero -- an uncalibrated `input_scale` is dropped by
        `_usable_input_scale` and really is not held. Anything else without the tensor interface has
        to raise, because a silent zero is the failure this module removes."""
        payload = quantize_scaled_fp8(torch.arange(32, dtype=torch.float32).reshape(4, 8))
        layers = extract_fp8_scaled_layers(
            {"lin.weight": payload.codes, "lin.weight_scale": payload.scale, "lin.input_scale": torch.tensor(1.0)}
        )

        assert layers["lin"].input_scale is None
        assert side_channel_bytes(layers) == payload.scale.nelement() * payload.scale.element_size()

        layers["lin"].weight_scale = 0.25  # a float, not a tensor
        with pytest.raises(AttributeError):
            side_channel_bytes(layers)


class TestTheDictAndThePayloads:
    def test_mapped_fp8_sources_are_not_credited_as_reclaimed_host_bytes(self) -> None:
        """File-backed source pages remain reclaimable in the live-RAM sample while the widened output is live."""
        sd = {"lin.weight": torch.zeros((4, 8), dtype=torch.float8_e4m3fn)}
        asked: list[tuple[int, dict[str, int]]] = []

        def reserve(bytes_needed: int, **kwargs: int) -> None:
            asked.append((bytes_needed, kwargs))

        needed = reserve_for_load(
            reserve,
            sd,
            torch.bfloat16,
            keep_fp8=False,
            model=_linear_model(8, 4),
            fp8_layers={},
            nvfp4_payloads={},
            source_tensors_mapped=True,
        )

        assert needed == 4 * 8 * torch.bfloat16.itemsize
        assert asked == [(needed, {"host_bytes_needed": needed})]

    def test_retained_mapped_sources_charge_final_output_tensors(self) -> None:
        payload = quantize_scaled_fp8(torch.arange(32, dtype=torch.float32).reshape(4, 8))
        sd = {
            "lin.weight": payload.codes,
            "lin.weight_scale": payload.scale,
            "lin.bias": torch.zeros(2048, dtype=torch.bfloat16),
        }
        layers = extract_fp8_scaled_layers(sd)
        asked: list[tuple[int, dict[str, int]]] = []

        def reserve(bytes_needed: int, **kwargs: int) -> None:
            asked.append((bytes_needed, kwargs))

        needed = reserve_for_load(
            reserve,
            sd,
            torch.bfloat16,
            keep_fp8=False,
            model=_linear_model(8, 4),
            fp8_layers=layers,
            nvfp4_payloads={},
            source_tensors_retained=True,
        )

        assert asked == [(needed, {"host_bytes_needed": 4 * 8 * 2 + 2048 * 2})]

    def test_retained_sources_release_before_decoded_nvfp4_output_is_counted(self) -> None:
        payload = quantize_scaled_fp8(torch.arange(32, dtype=torch.float32).reshape(4, 8))
        nv_tensors, _ = nvfp4_signed_tensors("decoded", torch.randint(0, 2, (128, 64), dtype=torch.bool))
        sd = {
            "lin.weight": payload.codes,
            "lin.weight_scale": payload.scale,
            "lin.bias": torch.zeros(2048, dtype=torch.float32),
            **nv_tensors,
        }
        payloads = pop_nvfp4_layers(sd, {"decoded": {"format": "nvfp4"}})
        layers = extract_fp8_scaled_layers(sd)
        model = torch.nn.Module()
        model.lin = torch.nn.Linear(8, 4, bias=False)
        model.decoded = torch.nn.Linear(64, 128, bias=False)
        asked: list[tuple[int, dict[str, int]]] = []

        def reserve(bytes_needed: int, **kwargs: int) -> None:
            asked.append((bytes_needed, kwargs))

        needed = reserve_for_load(
            reserve,
            sd,
            torch.bfloat16,
            keep_fp8=False,
            model=model,
            fp8_layers=layers,
            nvfp4_payloads=payloads,
            skip_patterns=("decoded",),
            source_tensors_retained=True,
        )

        original_sd_bytes = sd["lin.weight"].nbytes + sd["lin.bias"].nbytes
        final_sd_bytes = sd["lin.weight"].nelement() * torch.bfloat16.itemsize + sd["lin.bias"].nelement() * 2
        nvfp4_host_bytes = predict_nvfp4_install_host_size(model, payloads, torch.bfloat16, ("decoded",))
        released_scale_bytes = layers["lin"].weight_scale.nbytes
        assert asked == [
            (
                needed,
                {
                    "host_bytes_needed": max(
                        final_sd_bytes, final_sd_bytes - original_sd_bytes - released_scale_bytes + nvfp4_host_bytes
                    )
                },
            )
        ]

    def test_packed_nvfp4_payload_has_no_additional_host_demand(self) -> None:
        tensors, _ = nvfp4_signed_tensors("lin", torch.randint(0, 2, (128, 64), dtype=torch.bool))
        sd = dict(tensors)
        payloads = pop_nvfp4_layers(sd, {"lin": {"format": "nvfp4"}})
        asked: list[tuple[int, dict[str, int]]] = []

        def reserve(bytes_needed: int, **kwargs: int) -> None:
            asked.append((bytes_needed, kwargs))

        needed = reserve_for_load(
            reserve,
            sd,
            torch.float32,
            keep_fp8=False,
            model=_linear_model(64, 128),
            fp8_layers={},
            nvfp4_payloads=payloads,
        )

        assert asked == [(needed, {"host_bytes_needed": 0})]

    def test_modelopt_nvfp4_relayout_is_admitted_before_packed_install(self) -> None:
        tensors, _ = nvfp4_signed_tensors("lin", torch.randint(0, 2, (128, 64), dtype=torch.bool))
        sd = dict(tensors)
        payloads = pop_nvfp4_layers(sd, {"lin": {"format": "nvfp4"}})
        asked: list[tuple[int, dict[str, int]]] = []

        def reserve(bytes_needed: int, **kwargs: int) -> None:
            asked.append((bytes_needed, kwargs))

        needed = reserve_for_load(
            reserve,
            sd,
            torch.float32,
            keep_fp8=False,
            model=_linear_model(64, 128),
            fp8_layers={},
            nvfp4_payloads=payloads,
            modelopt_nvfp4_relayout=True,
        )

        payload = payloads["lin"]
        expected = max(3 * payload.weight.nbytes, payload.weight.nbytes + payload.weight_scale.nbytes)
        assert asked == [(needed, {"host_bytes_needed": expected})]

    def test_modelopt_relayout_is_added_after_positive_cast_growth(self) -> None:
        tensors, _ = nvfp4_signed_tensors("lin", torch.randint(0, 2, (128, 64), dtype=torch.bool))
        sd = {"dense.weight": torch.zeros(2048, dtype=torch.float16), **tensors}
        payloads = pop_nvfp4_layers(sd, {"lin": {"format": "nvfp4"}})
        asked: list[tuple[int, dict[str, int]]] = []

        def reserve(bytes_needed: int, **kwargs: int) -> None:
            asked.append((bytes_needed, kwargs))

        needed = reserve_for_load(
            reserve,
            sd,
            torch.float32,
            keep_fp8=False,
            model=_linear_model(64, 128),
            fp8_layers={},
            nvfp4_payloads=payloads,
            modelopt_nvfp4_relayout=True,
        )

        payload = payloads["lin"]
        relayout_peak = max(3 * payload.weight.nbytes, payload.weight.nbytes + payload.weight_scale.nbytes)
        cast_growth = 2048 * (torch.float32.itemsize - torch.float16.itemsize)
        assert asked == [(needed, {"host_bytes_needed": cast_growth + relayout_peak})]

    def test_released_fp8_grid_is_not_charged_again_with_decoded_nvfp4(self) -> None:
        exponents = (torch.arange(512 * 16, dtype=torch.int64).reshape(512, 16) % 8) + 124
        fp8_tensors, _ = mxfp8_tensors("lin", exponents)
        nv_tensors, _ = nvfp4_signed_tensors("decoded", torch.randint(0, 2, (1024, 512), dtype=torch.bool))
        sd = {**fp8_tensors, **nv_tensors}
        payloads = pop_nvfp4_layers(sd, {"decoded": {"format": "nvfp4"}})
        layers = extract_fp8_scaled_layers(sd, layer_hints={"lin": mxfp8_marker()})
        model = torch.nn.Module()
        model.lin = torch.nn.Linear(512, 512, bias=False)
        model.decoded = torch.nn.Linear(512, 1024, bias=False)
        asked: list[tuple[int, dict[str, int]]] = []

        def reserve(bytes_needed: int, **kwargs: int) -> None:
            asked.append((bytes_needed, kwargs))

        needed = reserve_for_load(
            reserve,
            sd,
            torch.bfloat16,
            keep_fp8=False,
            model=model,
            fp8_layers=layers,
            nvfp4_payloads=payloads,
            skip_patterns=("decoded",),
        )

        source_sd_bytes = sd["lin.weight"].nbytes
        target_sd_bytes = sd["lin.weight"].nelement() * torch.bfloat16.itemsize
        released_scale_bytes = layers["lin"].weight_scale.nbytes
        fold_peak = target_sd_bytes + fold_transient_bytes(sd["lin.weight"], layers["lin"].weight_scale)
        nvfp4_peak = predict_nvfp4_install_host_size(model, payloads, torch.bfloat16, ("decoded",))
        expected = max(
            0,
            needed - (source_sd_bytes + side_channel_bytes(layers) + sum(p.nbytes() for p in payloads.values())),
            fold_peak,
            target_sd_bytes - source_sd_bytes - released_scale_bytes + nvfp4_peak,
        )
        assert asked == [(needed, {"host_bytes_needed": expected})]

    def test_retained_fp8_scale_is_not_credited_again_with_decoded_nvfp4(self) -> None:
        payload = quantize_scaled_fp8(torch.arange(32, dtype=torch.float32).reshape(4, 8))
        nv_tensors, _ = nvfp4_signed_tensors("decoded", torch.randint(0, 2, (128, 64), dtype=torch.bool))
        sd = {"lin.weight": payload.codes, "lin.weight_scale": payload.scale, **nv_tensors}
        payloads = pop_nvfp4_layers(sd, {"decoded": {"format": "nvfp4"}})
        layers = extract_fp8_scaled_layers(sd)
        model = torch.nn.Module()
        model.lin = torch.nn.Linear(8, 4, bias=False)
        model.decoded = torch.nn.Linear(64, 128, bias=False)
        asked: list[tuple[int, dict[str, int]]] = []

        def reserve(bytes_needed: int, **kwargs: int) -> None:
            asked.append((bytes_needed, kwargs))

        needed = reserve_for_load(
            reserve,
            sd,
            torch.float32,
            keep_fp8=True,
            model=model,
            fp8_layers=layers,
            nvfp4_payloads=payloads,
            skip_patterns=("decoded",),
        )

        expected = predict_nvfp4_install_host_size(model, payloads, torch.float32, ("decoded",))
        assert layers["lin"].weight_scale is not None
        assert asked == [(needed, {"host_bytes_needed": expected})]

    def test_scaled_fp8_that_cannot_stay_packed_counts_its_dequantized_output(self) -> None:
        exponents = (torch.arange(512 * 16, dtype=torch.int64).reshape(512, 16) % 8) + 124
        tensors, _expected = mxfp8_tensors("lin", exponents)
        sd = dict(tensors)
        layers = extract_fp8_scaled_layers(sd, layer_hints={"lin": mxfp8_marker()})
        asked: list[tuple[int, dict[str, int]]] = []

        def reserve(bytes_needed: int, **kwargs: int) -> None:
            asked.append((bytes_needed, kwargs))

        needed = reserve_for_load(
            reserve,
            sd,
            torch.float32,
            keep_fp8=True,
            model=_linear_model(16 * 32, 512),
            fp8_layers=layers,
            nvfp4_payloads={},
        )

        expected_host = 512 * 512 * torch.float32.itemsize + fold_transient_bytes(
            sd["lin.weight"], layers["lin"].weight_scale
        )
        assert asked == [(needed, {"host_bytes_needed": expected_host})]

    @pytest.mark.parametrize("keep_fp8", [False, True])
    def test_fp8_fold_scratch_is_charged_only_when_the_layer_folds(self, keep_fp8: bool) -> None:
        payload = quantize_scaled_fp8(torch.arange(32, dtype=torch.float32).reshape(4, 8))
        sd = {"lin.weight": payload.codes, "lin.weight_scale": payload.scale}
        layers = extract_fp8_scaled_layers(sd)
        asked: list[tuple[int, dict[str, int]]] = []

        def reserve(bytes_needed: int, **kwargs: int) -> None:
            asked.append((bytes_needed, kwargs))

        needed = reserve_for_load(
            reserve,
            sd,
            torch.float32,
            keep_fp8=keep_fp8,
            model=_linear_model(8, 4),
            fp8_layers=layers,
            nvfp4_payloads={},
        )

        expected_host = 0
        if not keep_fp8:
            expected_host = 4 * 8 * torch.float32.itemsize + fold_transient_bytes(
                sd["lin.weight"], layers["lin"].weight_scale
            )
        assert asked == [(needed, {"host_bytes_needed": expected_host})]

    def test_nvfp4_decoded_host_peak_includes_prior_outputs(self) -> None:
        model = torch.nn.Module()
        model.lin = torch.nn.Linear(64, 128, bias=False)
        model.second = torch.nn.Linear(64, 128, bias=False)
        tensors_a, _ = nvfp4_signed_tensors("lin", torch.randint(0, 2, (128, 64), dtype=torch.bool))
        tensors_b, _ = nvfp4_signed_tensors("second", torch.randint(0, 2, (128, 64), dtype=torch.bool))
        payloads = pop_nvfp4_layers(
            {**tensors_a, **tensors_b}, {"lin": {"format": "nvfp4"}, "second": {"format": "nvfp4"}}
        )

        output_bytes = 128 * 64 * torch.float32.itemsize
        transient_bytes = 128 * 64 * (2 + torch.float32.itemsize) + (128 * 64 // 16) * torch.float32.itemsize
        assert predict_nvfp4_install_host_size(model, payloads, torch.float32, ()) == 0
        assert (
            predict_nvfp4_install_host_size(model, payloads, torch.float32, ("lin", "second"))
            == transient_bytes + output_bytes
        )

    def test_same_dtype_state_dict_uses_no_additional_host_demand(self) -> None:
        sd = {"lin.weight": torch.zeros(4, 8, dtype=torch.float32)}
        asked: list[tuple[int, dict[str, int]]] = []

        def reserve(bytes_needed: int, **kwargs: int) -> None:
            asked.append((bytes_needed, kwargs))

        needed = reserve_for_load(
            reserve, sd, torch.float32, keep_fp8=False, model=_linear_model(8, 4), fp8_layers={}, nvfp4_payloads={}
        )

        assert asked == [(needed, {"host_bytes_needed": 0})]

    def test_in_place_cast_uses_peak_additional_host_demand(self) -> None:
        sd = {"a": torch.zeros(4, dtype=torch.float32), "b": torch.zeros(4, dtype=torch.float32)}
        asked: list[tuple[int, dict[str, int]]] = []

        def reserve(bytes_needed: int, **kwargs: int) -> None:
            asked.append((bytes_needed, kwargs))

        reserve_for_load(reserve, sd, torch.float16, keep_fp8=False, model=None, fp8_layers={}, nvfp4_payloads={})

        assert asked == [(16, {"host_bytes_needed": 8})]

    def test_fold_peak_follows_the_layer_mapping_order(self) -> None:
        first = quantize_scaled_fp8(torch.zeros(2, 8, dtype=torch.float32))
        second = quantize_scaled_fp8(torch.zeros(8, 8, dtype=torch.float32))
        sd = {
            "second.weight": second.codes,
            "first.weight": first.codes,
        }
        layers = {
            "first": SimpleNamespace(weight_scale=first.scale, input_scale=None),
            "second": SimpleNamespace(weight_scale=second.scale, input_scale=None),
        }
        asked: list[tuple[int, dict[str, int]]] = []

        def reserve(bytes_needed: int, **kwargs: int) -> None:
            asked.append((bytes_needed, kwargs))

        needed = reserve_for_load(
            reserve,
            sd,
            torch.bfloat16,
            keep_fp8=False,
            model=None,
            fp8_layers=layers,
            nvfp4_payloads={},
        )

        expected = max(
            2 * 8 * torch.bfloat16.itemsize + fold_transient_bytes(sd["first.weight"], first.scale),
            (2 * 8 * torch.bfloat16.itemsize - 2 * 8)
            + 8 * 8 * torch.bfloat16.itemsize
            + fold_transient_bytes(sd["second.weight"], second.scale),
        )
        assert asked == [(needed, {"host_bytes_needed": expected})]

    def test_plain_dense_weight_remains_in_cast_peak_accounting(self) -> None:
        sd = {"lin.weight": torch.zeros(4, 8, dtype=torch.float32)}
        asked: list[tuple[int, dict[str, int]]] = []

        def reserve(bytes_needed: int, **kwargs: int) -> None:
            asked.append((bytes_needed, kwargs))

        needed = reserve_for_load(
            reserve, sd, torch.bfloat16, keep_fp8=False, model=_linear_model(8, 4), fp8_layers={}, nvfp4_payloads={}
        )

        assert asked == [(needed, {"host_bytes_needed": 64})]

    def test_a_load_with_no_side_channel_asks_for_what_the_predictor_alone_says(self) -> None:
        """A dense checkpoint must not pay for this function. It is a superset, not a surcharge."""
        sd = {"lin.weight": torch.zeros(4, 8)}
        model = _linear_model(8, 4)
        _asked_list, reserve = _asked()

        needed = reserve_for_load(
            reserve, sd, torch.bfloat16, keep_fp8=False, model=model, fp8_layers={}, nvfp4_payloads={}
        )

        assert needed == predict_cast_state_dict_size(sd, torch.bfloat16, keep_fp8=False, model=model)

    def test_nvfp4_payloads_are_counted_although_the_dict_no_longer_holds_them(self) -> None:
        """`pop_nvfp4_layers` takes them out before any prediction, which is why every seam that has
        them used to add a second term by hand."""
        tensors, _expected = nvfp4_signed_tensors("lin", torch.randint(0, 2, (128, 64), dtype=torch.bool))
        sd = dict(tensors)
        dense = torch.zeros(4, 4)
        sd["other.weight"] = dense
        model = _linear_model(64, 128)
        payloads = pop_nvfp4_layers(sd, {"lin": {"format": "nvfp4"}})
        _asked_list, reserve = _asked()

        needed = reserve_for_load(
            reserve, sd, torch.bfloat16, keep_fp8=False, model=model, fp8_layers={}, nvfp4_payloads=payloads
        )

        # Exact, and both terms non-zero: the dense remainder is still in the dict and the packed
        # layer is not, so dropping either changes the number. Inequalities would not have said that
        # -- `needed >= packed` alone is satisfied by the payload term on its own.
        assert needed == dense.nelement() * torch.bfloat16.itemsize + payloads["lin"].nbytes()

    def test_payloads_without_the_model_that_sizes_them_are_refused(self) -> None:
        """`predict_nvfp4_install_size` needs the model to decide which payloads stay packed.

        Absorbing its absence into a zero would drop a GiB-scale term silently -- the exact shape of
        defect this module exists to remove -- and a model-less reservation is a real pattern in this
        tree (`flux.py`, where the architecture is read off the keys before any model exists).
        """
        tensors, _expected = nvfp4_signed_tensors("lin", torch.randint(0, 2, (128, 64), dtype=torch.bool))
        sd = dict(tensors)
        payloads = pop_nvfp4_layers(sd, {"lin": {"format": "nvfp4"}})
        _asked_list, reserve = _asked()

        with pytest.raises(ValueError, match="without the model"):
            reserve_for_load(
                reserve, sd, torch.bfloat16, keep_fp8=False, model=None, fp8_layers={}, nvfp4_payloads=payloads
            )

    def test_failed_ram_admission_stops_before_the_load_can_allocate(self) -> None:
        sd = {"lin.weight": torch.zeros(4, 8)}
        allocated = False

        def allocate_after_reservation() -> None:
            nonlocal allocated
            allocated = True

        with pytest.raises(MemoryError, match="host memory"):
            reserve_for_load(
                lambda _needed: SimpleNamespace(ram_headroom_satisfied=False),
                sd,
                torch.bfloat16,
                keep_fp8=False,
                model=_linear_model(8, 4),
                fp8_layers={},
                nvfp4_payloads={},
            )
            allocate_after_reservation()

        assert not allocated


class TestKeepFp8IsCarried:
    """The flag decides what stays at one byte per element, and it reaches the prediction from here."""

    @pytest.mark.parametrize("keep_fp8", [True, False], ids=["kept", "folded"])
    def test_a_matmul_usable_scale_is_charged_one_byte_when_kept_and_two_when_folded(self, keep_fp8: bool) -> None:
        """A per-tensor scale *is* matmul-usable, so the two branches differ -- unlike an MX grid,
        which is charged the widened size either way and so cannot see this flag at all."""
        payload = quantize_scaled_fp8(torch.arange(32, dtype=torch.float32).reshape(4, 8))
        sd = {"lin.weight": payload.codes, "lin.weight_scale": payload.scale}
        layers = extract_fp8_scaled_layers(sd)
        _asked_list, reserve = _asked()

        needed = reserve_for_load(
            reserve,
            sd,
            torch.bfloat16,
            keep_fp8=keep_fp8,
            model=_linear_model(8, 4),
            fp8_layers=layers,
            nvfp4_payloads={},
        )

        weight_bytes = 4 * 8 * (1 if keep_fp8 else torch.bfloat16.itemsize)
        assert needed == weight_bytes + side_channel_bytes(layers)

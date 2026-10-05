"""Tests for the state-dict side of the int8_convrot scheme: which layers a loader picks up,
and which scale layouts it will act on.

The per-tensor mathematics is covered by ``test_int8_convrot.py``. What is pinned here is the
part a loader can get wrong without anything raising — every failure below would otherwise
surface as a model that loads cleanly and generates noise.
"""

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.quantization.int8_convrot import (
    CONVROT_GROUP_SIZE,
    Int8ConvrotLinear,
    cast_unquantized,
    check_int8_scale_layout,
    drop_unconsumed_quantization_sidecars,
    extract_int8_convrot_markers,
    install_int8_convrot_layers,
    predict_int8_cast_size,
    read_comfy_quant_markers,
    reject_foreign_quantization_scales,
    reject_unmarked_int8_weights,
    split_int8_convrot_layers,
    swap_in_int8_linears,
)
from tests.fixtures.quantized_payloads import comfy_quant_marker

MARKER = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": CONVROT_GROUP_SIZE}


class TestWhichLayersAreClaimed:
    def test_an_int8_marker_is_claimed_and_removed(self) -> None:
        sd = {
            "blocks.0.attn.wq.weight": torch.zeros(4, CONVROT_GROUP_SIZE, dtype=torch.int8),
            "blocks.0.attn.wq.weight_scale": torch.ones(4, 1),
            "blocks.0.attn.wq.comfy_quant": comfy_quant_marker(MARKER),
        }
        markers = extract_int8_convrot_markers(sd)

        assert markers == {"blocks.0.attn.wq": MARKER}
        # The marker is consumed; the weight and its scale stay for the loader to install.
        assert set(sd) == {"blocks.0.attn.wq.weight", "blocks.0.attn.wq.weight_scale"}

    def test_a_marker_for_another_format_is_left_untouched(self) -> None:
        """ComfyUI's fp8_scaled repacks share this key layout and belong to the fp8 path."""
        sd = {
            "layer.weight": torch.zeros(4, 4, dtype=torch.float8_e4m3fn),
            "layer.weight_scale": torch.ones(1),
            "layer.comfy_quant": comfy_quant_marker({"format": "float8_e4m3fn"}),
        }
        assert extract_int8_convrot_markers(sd) == {}
        assert "layer.comfy_quant" in sd

    def test_unmarked_weights_are_not_claimed(self) -> None:
        """Mixed precision is the norm, not the exception: one Krea-2 build leaves 40 weights in
        bf16 with no marker, and the two Qwen3-VL encoders leave over 200 each. A decision made
        per file rather than per tensor would be wrong on all of them."""
        sd = {
            "txtfusion.0.weight": torch.zeros(8, 8, dtype=torch.bfloat16),
            "blocks.0.attn.wq.weight": torch.zeros(4, CONVROT_GROUP_SIZE, dtype=torch.int8),
            "blocks.0.attn.wq.weight_scale": torch.ones(4, 1),
            "blocks.0.attn.wq.comfy_quant": comfy_quant_marker(MARKER),
        }
        assert set(extract_int8_convrot_markers(sd)) == {"blocks.0.attn.wq"}

    def test_a_state_dict_with_no_markers_is_unchanged(self) -> None:
        sd = {"a.weight": torch.zeros(2, 2)}
        assert extract_int8_convrot_markers(sd) == {}
        assert set(sd) == {"a.weight"}

    def test_the_group_size_travels_with_each_marker(self) -> None:
        """Every marker in the Krea-2 build says 256, but the flag is per tensor and another
        producer may vary it, so it is read rather than assumed."""
        sd = {"layer.comfy_quant": comfy_quant_marker({"format": "int8_tensorwise", "convrot_groupsize": 64})}
        assert extract_int8_convrot_markers(sd)["layer"]["convrot_groupsize"] == 64


class TestTheScaleLayoutsSeenInRealCheckpoints:
    """Three appear across the checkpoints this was tested against, so none is hypothetical."""

    def test_per_output_channel(self) -> None:
        check_int8_scale_layout("layer", torch.zeros(64, 256, dtype=torch.int8), torch.ones(64, 1))

    def test_per_output_channel_without_the_trailing_axis(self) -> None:
        check_int8_scale_layout("layer", torch.zeros(64, 256, dtype=torch.int8), torch.ones(64))

    def test_a_scalar_scale(self) -> None:
        """One Qwen3-VL encoder stores every scale as a bare scalar, and Krea-2's
        `txtfusion.projector` does the same."""
        check_int8_scale_layout("layer", torch.zeros(1, 12, dtype=torch.int8), torch.tensor(0.01))

    def test_a_blockwise_grid_is_refused_by_name(self) -> None:
        """Also observed: a 6144x6144 weight with a [48, 48] scale, i.e. a 128x128 block grid.
        Broadcasting would either raise somewhere uninformative or, for an unlucky shape,
        silently scale the wrong axis."""
        with pytest.raises(ValueError, match=r"Blockwise scale grids"):
            check_int8_scale_layout("last.up", torch.zeros(512, 512, dtype=torch.int8), torch.ones(4, 4))

    def test_the_message_names_the_layer_and_both_shapes(self) -> None:
        with pytest.raises(ValueError, match=r"'last\.up' has a \(4, 4\) scale for a \(512, 512\) weight"):
            check_int8_scale_layout("last.up", torch.zeros(512, 512, dtype=torch.int8), torch.ones(4, 4))


class TestCastUnquantized:
    """The cast that runs over an int8 state dict, and what it must leave alone.

    Two kinds of tensor must survive it unchanged: the quantized payloads, which
    `Int8ConvrotLinear` reads as stored, and every integer payload, which is not a weight at all.
    """

    def test_an_ordinary_float_weight_is_cast(self) -> None:
        sd = {"dense.weight": torch.ones(4, 4, dtype=torch.float32)}

        cast_unquantized(sd, torch.bfloat16, {})

        assert sd["dense.weight"].dtype is torch.bfloat16

    def test_the_pinned_int8_weight_and_its_scale_are_left_as_stored(self) -> None:
        """An int8 weight cast to bf16 is no longer int8, and an fp32 scale rounded to bf16 is a
        different scale."""
        sd = {
            "q.weight": torch.ones(4, CONVROT_GROUP_SIZE, dtype=torch.int8),
            "q.weight_scale": torch.full((4, 1), 0.1, dtype=torch.float32),
        }

        cast_unquantized(sd, torch.bfloat16, {"q": MARKER})

        assert sd["q.weight"].dtype is torch.int8
        assert sd["q.weight_scale"].dtype is torch.float32

    @pytest.mark.parametrize(
        ("label", "tensor"),
        [
            ("index_buffer", torch.arange(5, dtype=torch.int64)),
            ("packed_uint8_buffer", torch.zeros(8, dtype=torch.uint8)),
            ("bool_mask", torch.tensor([True, False, True])),
        ],
    )
    def test_an_integer_payload_keeps_its_dtype(self, label, tensor) -> None:
        """Not weights. An index buffer installed as a float raises `IndexError: tensors used as
        indices must be long, byte or bool` inside the forward; a bool mask cast to bf16 changes
        what the mask means and raises nothing at all."""
        sd = {f"module.{label}": tensor}

        cast_unquantized(sd, torch.bfloat16, {})

        assert sd[f"module.{label}"].dtype is tensor.dtype


class TestWhichMarkedLayersMayStayInt8:
    """The int8 branch skips the whole fp8 pipeline, so every filter that pipeline applies before
    leaving a weight quantized has to exist here too, or the int8 path is the lenient one."""

    @staticmethod
    def _model() -> torch.nn.Module:
        model = torch.nn.Module()
        model.t_embedder = torch.nn.Module()
        model.t_embedder.mlp = torch.nn.ModuleList([torch.nn.Linear(CONVROT_GROUP_SIZE, 4, bias=False)])
        model.proj = torch.nn.Linear(CONVROT_GROUP_SIZE, 4, bias=False)
        return model

    @staticmethod
    def _sd() -> dict:
        return {
            "t_embedder.mlp.0.weight": torch.ones(4, CONVROT_GROUP_SIZE, dtype=torch.int8),
            "t_embedder.mlp.0.weight_scale": torch.full((4, 1), 0.5),
            "proj.weight": torch.ones(4, CONVROT_GROUP_SIZE, dtype=torch.int8),
            "proj.weight_scale": torch.full((4, 1), 0.5),
        }

    def test_a_precision_sensitive_layer_is_dequantized_and_dropped_from_the_swap(self) -> None:
        """Diffusers models declare these because their forwards *read their own weight's dtype*.
        On an `Int8ConvrotLinear` that reads `torch.int8`, and the branch that follows has no idea
        what to do with it."""
        sd = self._sd()
        markers = {"t_embedder.mlp.0": dict(MARKER, convrot=False), "proj": dict(MARKER, convrot=False)}

        surviving = split_int8_convrot_layers(
            sd, markers, torch.float32, model=self._model(), skip_patterns=("t_embedder",)
        )

        assert set(surviving) == {"proj"}
        assert sd["t_embedder.mlp.0.weight"].dtype is torch.float32
        assert torch.allclose(sd["t_embedder.mlp.0.weight"], torch.full((4, CONVROT_GROUP_SIZE), 0.5))
        # Its scale is consumed by the dequantization, not left behind as an orphan key.
        assert "t_embedder.mlp.0.weight_scale" not in sd
        assert sd["proj.weight"].dtype is torch.int8

    def test_a_one_dimensional_quantized_weight_is_dequantized_rather_than_refused(self) -> None:
        """Repacks that quantize everything ship 1-D norms. There is no linear for those to become,
        but the scale is right there, so the decode is unambiguous -- the fp8 path makes the same
        call for the same reason."""
        sd = {
            "norm.weight": torch.ones(CONVROT_GROUP_SIZE, dtype=torch.int8),
            "norm.weight_scale": torch.tensor(0.25),
        }
        markers = {"norm": {"format": "int8_tensorwise", "convrot": False}}

        surviving = split_int8_convrot_layers(sd, markers, torch.float32, model=self._model())

        assert surviving == {}
        assert torch.allclose(sd["norm.weight"], torch.full((CONVROT_GROUP_SIZE,), 0.25))

    def test_a_one_dimensional_weight_marked_convrot_is_refused_by_name(self) -> None:
        """The rotation is defined over a weight's input dimension. A 1-D weight claiming it is a
        marker this decode cannot honor either way, and the shape unpack alone said nothing."""
        sd = {"norm.weight": torch.ones(CONVROT_GROUP_SIZE, dtype=torch.int8), "norm.weight_scale": torch.tensor(0.25)}

        with pytest.raises(ValueError, match=r"'norm' is marked convrot but its weight is \(256,\)"):
            split_int8_convrot_layers(sd, {"norm": dict(MARKER, convrot=True)}, torch.float32, model=self._model())

    def test_a_marker_on_a_non_linear_is_dequantized_rather_than_installed(self) -> None:
        """An `Int8ConvrotLinear` cannot stand in for an Embedding: the swap would succeed and the
        failure would surface much later, inside that module's forward."""
        model = self._model()
        model.embed = torch.nn.Embedding(8, CONVROT_GROUP_SIZE)
        sd = {
            "embed.weight": torch.ones(8, CONVROT_GROUP_SIZE, dtype=torch.int8),
            "embed.weight_scale": torch.tensor(0.5),
        }

        surviving = split_int8_convrot_layers(sd, {"embed": dict(MARKER, convrot=False)}, torch.float32, model=model)

        assert surviving == {}
        assert sd["embed.weight"].dtype is torch.float32

    def test_the_reservation_charges_what_the_split_and_the_cast_will_leave_behind(self) -> None:
        """Reserving happens before the split, so the prediction has to apply the same filters:
        a layer the split widens but the prediction charged one byte for is a reservation short by
        exactly that layer, and the widening transient lands on an unreserved cache."""
        sd = self._sd()
        markers = {"t_embedder.mlp.0": dict(MARKER, convrot=False), "proj": dict(MARKER, convrot=False)}

        reserved = predict_int8_cast_size(
            sd, torch.bfloat16, markers, model=self._model(), skip_patterns=("t_embedder",)
        )
        surviving = split_int8_convrot_layers(
            sd, markers, torch.bfloat16, model=self._model(), skip_patterns=("t_embedder",)
        )
        cast_unquantized(sd, torch.bfloat16, surviving)

        assert reserved == sum(t.nelement() * t.element_size() for t in sd.values())

    def test_a_pinned_float32_scale_is_charged_at_float32(self) -> None:
        """`cast_unquantized` pins the scale as stored, so charging it the compute dtype's width
        under-counts it by half."""
        sd = {"proj.weight": torch.ones(4, 8, dtype=torch.int8), "proj.weight_scale": torch.ones(4, 1)}

        reserved = predict_int8_cast_size(sd, torch.bfloat16, {"proj": dict(MARKER, convrot=False)})

        assert reserved == 4 * 8 * 1 + 4 * 1 * 4


class TestWhatTheSwapRefuses:
    """Its preconditions, each of which used to fail somewhere that named neither the layer nor
    the cause. Everything refused here is something `split_int8_convrot_layers` dequantizes, so a
    loader that runs the split never reaches these."""

    @staticmethod
    def _model() -> torch.nn.Module:
        model = torch.nn.Module()
        model.proj = torch.nn.Linear(CONVROT_GROUP_SIZE, 4, bias=False)
        model.embed = torch.nn.Embedding(8, CONVROT_GROUP_SIZE)
        return model

    def test_a_one_dimensional_weight_is_named(self) -> None:
        sd = {"norm.weight": torch.ones(8, dtype=torch.int8), "norm.weight_scale": torch.tensor(0.5)}

        with pytest.raises(ValueError, match=r"'norm'.*\(8,\), not 2-D"):
            swap_in_int8_linears(self._model(), sd, {"norm": dict(MARKER, convrot=False)})

    def test_a_marker_naming_a_module_the_model_lacks_is_named(self) -> None:
        sd = {"absent.weight": torch.ones(4, 8, dtype=torch.int8), "absent.weight_scale": torch.ones(4, 1)}

        with pytest.raises(ValueError, match=r"'absent'.*no such module"):
            swap_in_int8_linears(self._model(), sd, {"absent": dict(MARKER, convrot=False)})

    def test_a_marker_on_a_non_linear_is_named(self) -> None:
        sd = {
            "embed.weight": torch.ones(8, CONVROT_GROUP_SIZE, dtype=torch.int8),
            "embed.weight_scale": torch.ones(8, 1),
        }

        with pytest.raises(ValueError, match=r"'embed'.*Embedding, not an nn\.Linear"):
            swap_in_int8_linears(self._model(), sd, {"embed": dict(MARKER, convrot=False)})

    def test_the_scale_layout_check_is_reached(self) -> None:
        """Nothing else asserted that the swap consults it: deleting the call left every test green
        while a blockwise grid broadcast against the weight."""
        sd = {
            "proj.weight": torch.ones(4, CONVROT_GROUP_SIZE, dtype=torch.int8),
            "proj.weight_scale": torch.ones(2, 2),
        }

        with pytest.raises(ValueError, match=r"Blockwise scale grids"):
            swap_in_int8_linears(self._model(), sd, {"proj": dict(MARKER, convrot=False)})


class TestTheKeysTheOrphanCheckAndTheSidecarDropActOn:
    """Both used to match by substring or by a blind suffix slice, which is wrong in both
    directions: it exempts keys it should flag and drops keys it should keep."""

    def test_an_int8_tensor_whose_suffix_merely_has_the_right_length_is_an_orphan(self) -> None:
        """Seven characters came off every int8 key, so `foo.qkv.scales` read as `foo.qkv` -- and
        the marker on the *weight* at `foo.qkv` then exempted a packed sidecar nothing decodes."""
        sd = {"foo.qkv.scales": torch.zeros(4, 4, dtype=torch.int8)}

        with pytest.raises(ValueError, match=r"foo\.qkv\.scales"):
            reject_unmarked_int8_weights(sd, {"foo.qkv": MARKER}, "Krea-2")

    def test_an_int8_key_shorter_than_the_suffix_is_an_orphan(self) -> None:
        """It collapsed to the empty string, which a stray marker on the model's own root matches."""
        sd = {"w": torch.zeros(4, 4, dtype=torch.int8)}

        with pytest.raises(ValueError, match=r"'w'"):
            reject_unmarked_int8_weights(sd, {"": MARKER}, "Krea-2")

    def test_a_claimed_weight_is_not_an_orphan(self) -> None:
        reject_unmarked_int8_weights({"foo.weight": torch.zeros(4, 4, dtype=torch.int8)}, {"foo": MARKER}, "Krea-2")

    def test_a_module_whose_name_merely_contains_input_scale_is_kept(self) -> None:
        """`"input_scale" in k` is a substring test, so `...input_scaler.weight` was dropped -- and
        under Z-Image's strict load that surfaces as a missing key, not as a dropped sidecar."""
        sd = {
            "blocks.0.input_scaler.weight": torch.ones(4, 4),
            "blocks.0.attn.input_scale": torch.ones(1),
            "blocks.0.attn.comfy_quant": comfy_quant_marker({"format": "float8_e4m3fn"}),
        }

        assert set(drop_unconsumed_quantization_sidecars(sd)) == {"blocks.0.input_scaler.weight"}


class TestTheForeignScaleCheck:
    """`reject_foreign_quantization_scales` decides which unclaimed `weight_scale` is fatal.

    Inside the int8 branch the fp8 pipeline is skipped, so a scaled-fp8 layer that came along would
    be cast without its scale -- off by `1/weight_scale`, silently. Both filters below exist because
    the failure is fatal and a false positive costs a user a checkpoint that worked.
    """

    @staticmethod
    def _model() -> torch.nn.Module:
        model = torch.nn.Module()
        model.proj = torch.nn.Linear(4, 4)
        model.other = torch.nn.Linear(4, 4)
        return model

    def test_an_fp8_weight_this_model_consumes_is_fatal(self) -> None:
        sd = {
            "proj.weight": torch.zeros(4, 4, dtype=torch.int8),
            "proj.weight_scale": torch.ones(4, 1),
            "other.weight": torch.zeros(4, 4, dtype=torch.float8_e4m3fn),
            "other.weight_scale": torch.ones(()),
        }

        with pytest.raises(ValueError, match=r"other\.weight_scale"):
            reject_foreign_quantization_scales(sd, {"proj": MARKER}, "Ideogram 4", self._model())

    def test_a_scale_on_a_dense_weight_is_not(self) -> None:
        """A merged single file carries a bundled encoder too, and these loaders do not prefix-filter
        it out. The weight's dtype is what tells a foreign *quantized* layer from a foreign dense
        one; a dense weight loads correctly whatever sits beside it."""
        sd = {"other.weight": torch.zeros(4, 4), "other.weight_scale": torch.ones(())}

        reject_foreign_quantization_scales(sd, {}, "Ideogram 4", self._model())

    def test_a_scale_on_a_module_this_model_does_not_have_is_not(self) -> None:
        """An all-in-one export bundles a scaled-fp8 *submodel* beside the int8 transformer. Those
        keys are discarded by the load rather than cast, so they cannot load unscaled."""
        sd = {
            "text_encoder.fc.weight": torch.zeros(4, 4, dtype=torch.float8_e4m3fn),
            "text_encoder.fc.weight_scale": torch.ones(()),
        }

        reject_foreign_quantization_scales(sd, {}, "Ideogram 4", self._model())


class TestTheSharedInstall:
    """`install_int8_convrot_layers` owns the order the five steps have to run in.

    Each loader used to write the sequence out by hand, and two of them wrote it out incompletely:
    Z-Image and the PiD decoder both skipped the foreign-scale check and would cast a mixed
    checkpoint's fp8 weights without their scales. The order is what these pin -- every step is
    individually covered above, and a partial sequence raises nothing at all.
    """

    @staticmethod
    def _model() -> torch.nn.Module:
        model = torch.nn.Module()
        model.keeps = torch.nn.Linear(CONVROT_GROUP_SIZE, 4, bias=False)
        model.widens = torch.nn.Linear(CONVROT_GROUP_SIZE, 4, bias=False)
        return model

    @staticmethod
    def _int8_layer(path: str) -> dict[str, torch.Tensor]:
        return {
            f"{path}.weight": torch.ones(4, CONVROT_GROUP_SIZE, dtype=torch.int8),
            f"{path}.weight_scale": torch.ones(4, 1),
        }

    def test_a_foreign_scale_is_refused_before_a_single_byte_is_reserved(self) -> None:
        """The reservation is the expensive step -- it evicts other models to make room. A load that
        is about to be refused must not first ask the cache to free memory for it."""
        sd = {**self._int8_layer("keeps"), "widens.weight": torch.zeros(4, 4, dtype=torch.float8_e4m3fn)}
        sd["widens.weight_scale"] = torch.ones(())
        reserved: list[int] = []

        with pytest.raises(ValueError, match=r"widens\.weight_scale"):
            install_int8_convrot_layers(
                self._model(),
                sd,
                {"keeps": MARKER},
                torch.float32,
                architecture="Z-Image",
                reserve=reserved.append,
            )

        assert reserved == []

    def test_the_reservation_is_made_before_the_split_widens_anything(self) -> None:
        """A locked model cannot be evicted, so the first reservation has to already cover the peak.
        Reserving after the split lets its dequantized tensors land on an unreserved cache."""
        sd = {**self._int8_layer("keeps"), **self._int8_layer("widens")}
        widths: list[torch.dtype] = []

        install_int8_convrot_layers(
            self._model(),
            sd,
            {"keeps": MARKER, "widens": MARKER},
            torch.float32,
            architecture="Z-Image",
            reserve=lambda _bytes: widths.append(sd["widens.weight"].dtype),
            skip_patterns=("widens",),
        )

        assert widths == [torch.int8]
        # And the split did run afterwards, or the assertion above would pass vacuously.
        assert sd["widens.weight"].dtype is torch.float32

    def test_only_the_surviving_layers_are_installed_and_returned(self) -> None:
        """A marker on a layer the split widened must not reach the swap: it would install an
        `Int8ConvrotLinear` over a weight that is no longer int8."""
        model = self._model()
        sd = {**self._int8_layer("keeps"), **self._int8_layer("widens")}

        surviving = install_int8_convrot_layers(
            model,
            sd,
            {"keeps": MARKER, "widens": MARKER},
            torch.float32,
            architecture="Z-Image",
            reserve=lambda _bytes: None,
            skip_patterns=("widens",),
        )

        assert set(surviving) == {"keeps"}
        assert isinstance(model.keeps, Int8ConvrotLinear)
        assert not isinstance(model.widens, Int8ConvrotLinear)

    def test_the_reservation_charges_int8_its_byte_and_adds_the_caller_s_own_bytes(self) -> None:
        """Krea-2 and Z-Image hold nvfp4 layers beside the int8 ones, so one reservation covers the
        whole load. Spelled out rather than recomputed with `predict_int8_cast_size`: an expectation
        derived from the implementation cannot notice the implementation charging the wrong width."""
        reserved: list[int] = []

        install_int8_convrot_layers(
            self._model(),
            self._int8_layer("keeps"),
            {"keeps": MARKER},
            torch.float32,
            architecture="Z-Image",
            reserve=reserved.append,
            extra_reserved_bytes=4096,
        )

        # The layer stays int8, so it is charged one byte per code -- not float32's four -- plus its
        # float32 scale column, plus what the caller asked for on top.
        codes = 4 * CONVROT_GROUP_SIZE
        scale = 4 * 4
        assert reserved == [codes + scale + 4096]


def test_read_comfy_quant_markers_reads_a_marker_off_a_file(tmp_path) -> None:
    """The scheme, from the file rather than from a state dict.

    Two callers need it that way: a loader deciding whether to commit to a ~20 GiB read, and model
    identification, whose state dict is on the meta device and therefore has no bytes to parse.
    Unrelated tensors are not reported, and a weight without a marker contributes nothing.
    """
    marker_json = b'{"format": "fp8_scaled", "convrot": false}'
    path = tmp_path / "tiny.safetensors"
    save_file(
        {
            "blocks.0.mlp.fc2.weight": torch.zeros(2, 2),
            "blocks.0.mlp.fc2.comfy_quant": torch.frombuffer(marker_json, dtype=torch.uint8).clone(),
            "unrelated.weight": torch.zeros(1),
        },
        str(path),
    )

    markers = read_comfy_quant_markers(path)
    assert markers == {"blocks.0.mlp.fc2": {"format": "fp8_scaled", "convrot": False}}

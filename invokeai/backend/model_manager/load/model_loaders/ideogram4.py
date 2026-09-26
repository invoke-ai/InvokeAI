"""Model loading for Ideogram 4 in InvokeAI.

The on-disk model is a diffusers pipeline folder bundling:
  - transformer/                 (Ideogram4Transformer, nf4 or fp8 quantized)
  - unconditional_transformer/   (Ideogram4Transformer, nf4 or fp8 quantized)
  - text_encoder/ + tokenizer/   (Qwen3-VL, nf4 or fp8)
  - vae/                         (FLUX.2-style AutoencoderKL; loaded via the vendored AutoEncoder)

The transformer is our vendored ``Ideogram4Transformer`` (not a diffusers class), so we
build it explicitly and load the prequantized state dict — mirroring how InvokeAI loads
FLUX nf4. Both transformer branches are returned as a single ``Ideogram4TransformerPair``.
"""

import itertools
import json
from pathlib import Path
from typing import Optional

import accelerate
import torch
from safetensors.torch import load_file

from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.main import (
    Main_Checkpoint_Ideogram4_Config,
    Main_Diffusers_Ideogram4_Config,
)
from invokeai.backend.model_manager.load.load_default import ModelLoader, _model_declared_skip_patterns
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)
from invokeai.backend.model_manager.util.qwen3_vl import (
    drop_qwen3vl_visual_tower,
    drop_qwen3vl_visual_tower_keys,
)
from invokeai.backend.quantization.fp8_scaled import (
    Fp8ScaledLayer,
    attach_fp8_scales,
    cast_state_dict,
    dequantize_fp8_scaled,
    extract_comfy_quant_hints,
    extract_fp8_scaled_layers,
    full_precision_hints_respected,
    parse_quantization_metadata,
    read_safetensors_metadata,
    reject_quantized_side_channel,
    should_keep_fp8_weights,
    split_fp8_scaled_layers,
    warn_on_unattached_scales,
)
from invokeai.backend.quantization.int8_convrot import (
    drop_unconsumed_quantization_sidecars,
    extract_int8_convrot_markers,
    install_int8_convrot_layers,
    reject_int8_layers_a_plain_fold_cannot_decode,
    reject_unmarked_int8_weights,
)
from invokeai.backend.quantization.load_plan import reserve_for_load
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.state_dict_loading import load_state_dict_ignoring_extras, log_unexpected_keys


def _load_local_state_dict(folder: Path, basename: str) -> dict[str, torch.Tensor]:
    """Load a (possibly sharded) safetensors checkpoint from a local diffusers component folder."""
    index_path = folder / f"{basename}.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            weight_map: dict[str, str] = json.load(f)["weight_map"]
        sd: dict[str, torch.Tensor] = {}
        for shard in sorted(set(weight_map.values())):
            sd.update(load_file(folder / shard))
        return sd
    return load_file(folder / f"{basename}.safetensors")


def _verify_encoder_fully_materialized(model: torch.nn.Module, *, context: str) -> None:
    """Fail if any parameter is still on the meta device after loading the text encoder.

    The encoder is built under ``accelerate.init_empty_weights()`` (every param starts on the meta
    device) and then filled from the checkpoint. Missing keys are only acceptable for tied weights, which
    ``transformers`` materializes via ``tie_weights()``; any other missing key leaves a meta tensor that
    would pass loading but fail later during device movement or encoding. Re-tie, then hard-fail if any
    meta tensor remains so a bad/mismatched encoder is rejected at load time instead.
    """
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    meta = [
        name
        for name, tensor in itertools.chain(model.named_parameters(), model.named_buffers())
        if getattr(tensor, "is_meta", False)
    ]
    if meta:
        raise RuntimeError(
            f"{context}: {len(meta)} parameter(s) remain on the meta device after loading "
            f"(missing or mismatched weights): {meta[:10]}"
        )


@ModelLoaderRegistry.register(base=BaseModelType.Ideogram4, type=ModelType.Main, format=ModelFormat.Diffusers)
class Ideogram4DiffusersModel(ModelLoader):
    """Loads Ideogram 4 main models (nf4 / fp8) bundled in diffusers layout."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Main_Diffusers_Ideogram4_Config):
            raise ValueError(f"Expected Main_Diffusers_Ideogram4_Config, got {type(config).__name__}.")
        if submodel_type is None:
            raise Exception("A submodel type must be provided when loading Ideogram 4 main pipelines.")

        model_path = Path(config.path)

        match submodel_type:
            case SubModelType.Transformer:
                return self._load_transformer_pair(model_path)
            case SubModelType.TextEncoder:
                return self._load_text_encoder(model_path)
            case SubModelType.Tokenizer:
                from transformers import AutoTokenizer

                return AutoTokenizer.from_pretrained(model_path / "tokenizer", local_files_only=True)
            case SubModelType.VAE:
                return self._load_vae(model_path)

        raise ValueError(
            f"Unsupported submodel for Ideogram 4: {submodel_type.value if submodel_type else 'None'}. "
            "Supported: Transformer, TextEncoder, Tokenizer, VAE."
        )

    def _load_transformer_pair(self, model_path: Path) -> AnyModel:
        from invokeai.backend.ideogram4.transformer_pair import Ideogram4TransformerPair

        conditional = self._load_one_transformer(model_path / "transformer")
        unconditional = self._load_one_transformer(model_path / "unconditional_transformer")
        return Ideogram4TransformerPair(conditional=conditional, unconditional=unconditional)

    def _load_one_transformer(self, folder: Path) -> torch.nn.Module:
        from invokeai.backend.ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer
        from invokeai.backend.ideogram4.quantized_loading import (
            is_bnb4bit_state_dict,
            is_fp8_state_dict,
            load_fp8_state_dict,
            reject_scale_spellings_this_path_drops,
            swap_linears_to_fp8,
        )

        target_device = TorchDevice.choose_torch_device()
        compute_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        sd = _load_local_state_dict(folder, "diffusion_pytorch_model")

        # Both refusals ahead of the reservation: `make_room` evicts to make space, so a refusal
        # below it would flush other resident models for a load that cannot finish. Neither can fire
        # on the nf4 build -- its sidecars are `.absmax` and `.quant_state.*`, and its weights are
        # uint8 rather than int8.
        #
        # int8 first, for the reason `Ideogram4CheckpointModel._load_model` gives at its own probe:
        # an int8 layer ships a `.weight_scale` too, and `is_fp8_state_dict` is satisfied by that
        # suffix alone. Taken for this path's private fp8, a rotated weight gets the scale and never
        # the inverse rotation -- `Fp8Linear.forward` is `weight * weight_scale.unsqueeze(1)`, a fold
        # deferred to forward.
        #
        # What the guard lets through is an *unrotated* marked int8 build, where the scale is the
        # whole decode. Not free even so: `Fp8Linear` stores in e4m3, so the codes are requantized on
        # the way in (1.9% mean weight error, measured). No such build is published for this layout;
        # if one appears, that requantization is the thing to fix, not this refusal.
        reject_int8_layers_a_plain_fold_cannot_decode(sd, f"Ideogram 4 transformer {folder.name}")
        reject_scale_spellings_this_path_drops(sd, f"Ideogram 4 transformer {folder.name}")

        self._ram_cache.make_room(sum(t.nelement() * t.element_size() for t in sd.values()))

        if is_bnb4bit_state_dict(sd):
            # Here rather than at the top: bitsandbytes is not installed on macOS, and importing it
            # up front failed every load through this method there, fp8 and unquantized included.
            from invokeai.backend.quantization.bnb_nf4 import quantize_model_nf4

            # nf4: build the model with InvokeLinearNF4 layers (compress_statistics=False, matching
            # the on-disk single-quant format), then load the prequantized state dict. The model
            # stays on CPU/meta until the cache moves it to the GPU.
            with accelerate.init_empty_weights():
                model: torch.nn.Module = Ideogram4Transformer(Ideogram4Config())
                model = quantize_model_nf4(model, modules_to_not_convert=set(), compute_dtype=compute_dtype)
            load_state_dict_ignoring_extras(model, sd, source="Ideogram 4 nf4 transformer", assign=True)
            return model

        if is_fp8_state_dict(sd):
            # Weight-only fp8 (e4m3): dequantizes to compute dtype at forward time; runs on any device.
            model = Ideogram4Transformer(Ideogram4Config())
            model.to(compute_dtype)
            swap_linears_to_fp8(model, sd, compute_dtype=compute_dtype)
            load_fp8_state_dict(model, sd, device=torch.device("cpu"), dtype=compute_dtype)
            model.eval()
            return model

        # Unquantized fallback.
        with accelerate.init_empty_weights():
            model = Ideogram4Transformer(Ideogram4Config())
        load_state_dict_ignoring_extras(model, sd, source="Ideogram 4 transformer", assign=True)
        return model.to(compute_dtype)

    def _load_text_encoder(self, model_path: Path) -> AnyModel:
        import accelerate
        from transformers import AutoConfig, AutoModel

        from invokeai.backend.ideogram4.quantized_loading import (
            FP8_TEXT_ENCODER_CONFIG_FLAG,
            load_fp8_state_dict,
            reject_scale_spellings_this_path_drops,
            swap_linears_to_fp8,
        )

        encoder_path = model_path / "text_encoder"
        target_device = TorchDevice.choose_torch_device()
        compute_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        raw_cfg = json.loads((encoder_path / "config.json").read_text(encoding="utf-8"))

        # Build the bare architecture from config, then load the prequantized weights ourselves. We must
        # NOT use transformers' native bitsandbytes loading (from_pretrained with a quantization_config)
        # because the resulting bnb Linear4bit layers are not compatible with InvokeAI's partial-loading
        # model cache. This mirrors how the FLUX T5 bnb encoder is loaded.
        cfg = AutoConfig.from_pretrained(encoder_path, local_files_only=True)
        # Drop the quantization_config so from_config builds a plain (unquantized) architecture.
        if hasattr(cfg, "quantization_config"):
            cfg.quantization_config = None

        sd = _load_local_state_dict(encoder_path, "model")
        # Ideogram 4 conditions on the Qwen3-VL language tower only -- `ideogram4/text_encoding.py`
        # drives `text_encoder.language_model` directly and never calls `forward`, so the visual tower
        # is never executed. Dropped before the reservation below, which is sized from this dict.
        sd = drop_qwen3vl_visual_tower_keys(sd)

        # Ahead of the reservation, which evicts other resident models to make space: a refusal below
        # it would flush the cache for a load that cannot finish.
        keeps_fp8 = bool(raw_cfg.get(FP8_TEXT_ENCODER_CONFIG_FLAG, False))
        if keeps_fp8:
            # The fp8 branch reads one spelling of the scale and would drop the other silently.
            reject_scale_spellings_this_path_drops(sd, f"Ideogram 4 text encoder {encoder_path.name}")
        else:
            # Everything without the flag goes to `load_state_dict(assign=True, strict=False)` below,
            # which would make the fp8 codes the parameters themselves and report the orphaned scale
            # at DEBUG. InvokeAI is the only producer that writes the flag, so a third-party fp8
            # repack of this encoder arrives here rather than in the branch above. The nf4 build is
            # unaffected: its sidecars are `.absmax` and `.quant_state.*`, which name no scale.
            reject_quantized_side_channel(sd, f"Ideogram 4 text encoder {encoder_path.name}")

        self._ram_cache.make_room(sum(t.nelement() * t.element_size() for t in sd.values()))

        if keeps_fp8:
            # Weight-only fp8 (e4m3): build the empty architecture, swap the quantized Linears for
            # Fp8Linear (gated on a saved per-row scale), then load. Mirrors the transformer fp8 branch;
            # runs on any device. strict=False tolerates the tied embed weights transformers resolves
            # itself; unexpected keys are logged at DEBUG and ignored (see `load_fp8_state_dict`).
            # assign=True fills the meta params directly.
            with accelerate.init_empty_weights():
                model: torch.nn.Module = AutoModel.from_config(cfg)
                # `required=False`: the architecture comes from the folder's config.json, so an
                # encoder without a vision tower is a shape this loader may legitimately meet.
                drop_qwen3vl_visual_tower(model, required=False)
                swap_linears_to_fp8(model, sd, compute_dtype=compute_dtype)
            load_fp8_state_dict(model, sd, device=torch.device("cpu"), dtype=compute_dtype, assign=True, strict=False)
            _verify_encoder_fully_materialized(model, context="Ideogram 4 fp8 text encoder")
            model.eval()
            return model

        is_bnb_nf4 = "quantization_config" in raw_cfg and bool(raw_cfg["quantization_config"].get("load_in_4bit"))

        with accelerate.init_empty_weights():
            model = AutoModel.from_config(cfg)
            drop_qwen3vl_visual_tower(model, required=False)  # see the fp8 branch above
            if is_bnb_nf4:
                # Only this branch needs bitsandbytes, which macOS does not have; see the transformer.
                from invokeai.backend.quantization.bnb_nf4 import quantize_model_nf4

                model = quantize_model_nf4(model, modules_to_not_convert=set(), compute_dtype=compute_dtype)

        _, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        # Extra keys are exporter noise, not a correctness signal - log them and move on. Missing keys
        # are acceptable only for tied weights (resolved by _verify_encoder_fully_materialized via
        # tie_weights); any genuinely missing non-tied weight is caught there as a leftover meta tensor.
        log_unexpected_keys("Ideogram 4 text encoder", unexpected)
        _verify_encoder_fully_materialized(model, context="Ideogram 4 text encoder")
        if not is_bnb_nf4:
            model = model.to(compute_dtype)
        model.eval()
        return model

    def _load_vae(self, model_path: Path) -> AnyModel:
        from invokeai.backend.ideogram4.autoencoder import (
            AutoEncoder,
            AutoEncoderParams,
            convert_diffusers_state_dict,
        )

        target_device = TorchDevice.choose_torch_device()
        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        sd = load_file(model_path / "vae" / "diffusion_pytorch_model.safetensors")
        # No quantized layout reaches this decoder, and the load below drops what it does not
        # recognise: `load_state_dict_ignoring_extras` reports extras and copies on, so an fp8 conv
        # weight would be cast into a float32 parameter with its scale discarded. On the file's own
        # key names rather than the converted ones, so a rename cannot carry a scale out of sight.
        reject_quantized_side_channel(sd, f"Ideogram 4 VAE {model_path.name}")
        sd = convert_diffusers_state_dict(sd)
        ae = AutoEncoder(AutoEncoderParams())
        load_state_dict_ignoring_extras(ae, sd, source="Ideogram 4 VAE")
        ae.eval()
        return ae.to(model_dtype)


@ModelLoaderRegistry.register(base=BaseModelType.Ideogram4, type=ModelType.Main, format=ModelFormat.Checkpoint)
class Ideogram4CheckpointModel(ModelLoader):
    """Loads ONE branch of Ideogram 4's dual-branch transformer from a single file.

    Comfy-Org publishes the conditional and unconditional transformers as separate files
    (``ideogram4_*.safetensors`` / ``ideogram4_unconditional_*.safetensors``) whose keys are
    exactly this repository's ``Ideogram4Transformer`` state dict — no conversion. Each installs
    as its own model and the loader node pairs them, which is why this returns a bare transformer
    where the diffusers loader returns an ``Ideogram4TransformerPair``.

    Three storage schemes, distinguished by what sits next to the weights:

    - plain bf16/fp16 -- loaded verbatim;
    - ComfyUI "scaled fp8" (fp8 weight + per-tensor ``.weight_scale``, plus ``.comfy_quant``
      markers), kept quantized when either consumer wants it -- the fp8 matmul, or FP8 Storage asked
      of this model, which identification switches on for such a file by itself -- and folded into
      the compute dtype only when neither does. Kept, the 8.6 GiB file is 8.7 GiB resident; folded,
      it is 17.3 GiB. The same trade every other single-file loader here makes;
    - ComfyUI ``int8_tensorwise`` + ``convrot`` (int8 weight + per-output-channel ``.weight_scale``),
      kept in ``Int8ConvrotLinear`` and dequantized per forward, so the file's size is the resident
      size with no setting to enable and no dependence on the device.

    nvfp4 is refused at identification, not here.
    """

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Main_Checkpoint_Ideogram4_Config):
            raise ValueError(f"Expected Main_Checkpoint_Ideogram4_Config, got {type(config).__name__}.")

        if submodel_type is not SubModelType.Transformer:
            raise ValueError(
                "A single-file Ideogram 4 checkpoint holds only a transformer; "
                f"'{submodel_type.value if submodel_type else 'None'}' is not in it. Select a standalone "
                "Qwen3-VL encoder and VAE on the model loader node, or install the diffusers pipeline."
            )

        from invokeai.backend.ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer

        model_path = Path(config.path)
        target_device = TorchDevice.choose_torch_device()
        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        sd: dict[str, torch.Tensor] = load_file(model_path)

        # Which of the two ComfyUI side channels this file carries is decided once, and int8 first:
        # an int8 layer ships a `.weight_scale` too, so probing for scales without ruling int8 out
        # would take the whole checkpoint down the fp8 path -- scaled, but never un-rotated, which
        # loads cleanly and generates noise.
        int8_markers = extract_int8_convrot_markers(sd)
        # Outside the branch on purpose (see the helper): an int8 weight whose marker is missing or
        # unparseable would otherwise be cast to the compute dtype as raw codes and load silently.
        reject_unmarked_int8_weights(sd, int8_markers, "Ideogram 4")

        fp8_layers: dict[str, Fp8ScaledLayer] = {}
        keep_fp8 = False
        if not int8_markers:
            # Per-layer `.comfy_quant` markers are popped out of `sd` here; the header carries the
            # same flags on some repacks and wins where both are present. Read before anything
            # consumes the weights: `extract_fp8_scaled_layers` drops what is left of the side
            # channel.
            layer_hints = {
                **extract_comfy_quant_hints(sd),
                **parse_quantization_metadata(read_safetensors_metadata(model_path, self._logger)),
            }

            fp8_layers = extract_fp8_scaled_layers(sd, layer_hints=layer_hints)
            # Two consumers want these packed and either is enough: the fp8 matmul, which runs on
            # them directly, and FP8 Storage asked of this model. Asking only the matmul -- which is
            # what this did -- makes the storage path *lossy* on a scaled checkpoint: the fold below
            # widens the weights to the compute dtype and drops their scales, and the layerwise cast
            # at the end of this method then re-encodes that result as *unscaled* fp8, for the byte
            # count the file already had. Measured on FLUX.2 Klein 4B, ~3% of the weights flush to
            # zero that way.
            keep_fp8 = self._keep_fp8_weights(config, SubModelType.Transformer)

        with accelerate.init_empty_weights():
            model: torch.nn.Module = Ideogram4Transformer(Ideogram4Config())

        # `input_proj` and `t_embedding` must not stay quantized in either scheme -- the model
        # derives its activation dtype from their weights, and neither float8 nor int8 is a dtype
        # torch computes in (see `Ideogram4Transformer._skip_layerwise_casting_patterns`). The
        # released int8 build leaves both dense anyway; this holds for a repack that does not.
        skip_patterns = _model_declared_skip_patterns(model)

        if int8_markers:
            # W8A8 activation scales and any marker of another format: meaningless on this path,
            # which dequantizes the weight and computes in `model_dtype`. They are dropped rather
            # than tolerated because the load below is strict -- and because they would otherwise be
            # cast and counted against the reservation for nothing.
            # Rebinding is safe here where z_image filters in place: `sd` is the only reference to
            # the dict, and the `sd.clear()` after the load acts on whatever `sd` names then -- the
            # same dict `load_state_dict(assign=True)` aliased its parameters from.
            sd = drop_unconsumed_quantization_sidecars(sd)

            quantized = install_int8_convrot_layers(
                model,
                sd,
                int8_markers,
                model_dtype,
                architecture="Ideogram 4",
                reserve=self._ram_cache.make_room,
                skip_patterns=skip_patterns,
            )
            kept = 0
        else:
            # Reserve before anything below widens a weight: the fold widens every scaled layer and
            # the split dequantizes the ones it cannot keep through float32, so a reservation made
            # afterwards lets either peak land on a cache that was only ever sized for the file.
            # Where the weights are not kept the prediction charges every float at `model_dtype`,
            # folded yet or not, so the number is the same on either side of the fold -- what changes
            # is when the room exists.
            reserve_for_load(
                self._ram_cache.make_room,
                sd,
                model_dtype,
                keep_fp8=keep_fp8,
                model=model,
                skip_patterns=skip_patterns,
                fp8_layers=fp8_layers,
                nvfp4_payloads={},
            )
            if fp8_layers and not keep_fp8:
                # Neither consumer asked. Fold the scales in: staying quantized would halve VRAM but
                # dequantize on every forward, which is a cost nobody asked to pay.
                dequantize_fp8_scaled(sd, fp8_layers, model_dtype)
                fp8_layers = {}
            fp8_layers = split_fp8_scaled_layers(sd, fp8_layers, model_dtype, model=model, skip_patterns=skip_patterns)
            kept = cast_state_dict(sd, model_dtype, keep_fp8=keep_fp8, model=model, skip_patterns=skip_patterns)

        # Strict: the released files are key-for-key this model, so a missing or extra key means a
        # different checkpoint, not a benign naming difference.
        model.load_state_dict(sd, strict=True, assign=True)
        # `assign=True` aliases every parameter to its `sd` tensor; dropping the dict's references
        # keeps the originals from staying reachable beside any fp8 copy made below.
        sd.clear()

        if int8_markers:
            # No setting behind this one and nothing to fall back to: `Int8ConvrotLinear` holds the
            # stored codes and dequantizes per forward, so an 8.9 GiB file stays 8.9 GiB resident on
            # every device. What it does not get is int8 *compute*.
            self._logger.info(
                f"Ideogram 4: kept {len(quantized)} layer(s) in int8 (int8_tensorwise checkpoint, "
                "dequantized per forward)"
            )
            return model

        if kept and not fp8_layers:
            self._logger.info(
                f"Ideogram 4: kept {kept} raw fp8 weight(s) quantized (no weight_scale in the checkpoint), "
                f"kept for {self._fp8_kept_reason()}"
            )

        if fp8_layers:
            attached = attach_fp8_scales(model, fp8_layers)
            self._logger.info(
                f"Ideogram 4: kept {attached} layer(s) in fp8 (scaled fp8 checkpoint, kept for {self._fp8_kept_reason()})"
            )
            warn_on_unattached_scales(self._logger, "Ideogram 4", attached, fp8_layers)
            marked = sum(1 for layer in fp8_layers.values() if layer.full_precision_matmul)
            # Only where the matmul is what runs. Kept for FP8 Storage alone, *every* layer
            # dequantizes per forward, so the hint decides nothing and the advice below would send a
            # user to override the checkpoint producer for a speedup that cannot happen.
            if marked and should_keep_fp8_weights(self._torch_device):
                if full_precision_hints_respected():
                    self._logger.info(
                        f"Ideogram 4: {marked} of {len(fp8_layers)} layer(s) are marked "
                        "full_precision_matrix_mult and will dequantize per forward. Set "
                        "fp8_compute_full_precision_hints=false to run them on the fp8 tensor cores "
                        "instead (faster, but overrides the checkpoint producer's instruction)."
                    )
                else:
                    self._logger.info(
                        f"Ideogram 4: ignoring the full_precision_matrix_mult marker on {marked} layer(s) "
                        "(fp8_compute_full_precision_hints=false)."
                    )
            # The layerwise-casting path exists to *produce* fp8 weights from full-precision ones.
            # These already are fp8, and its hooks would restore the compute dtype without applying
            # `weight_scale` — a silently wrong weight — while also disabling the matmul.
            return model

        return self._apply_fp8_layerwise_casting(model, config, SubModelType.Transformer)

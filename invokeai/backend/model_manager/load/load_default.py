# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Default implementation of model loading in InvokeAI."""

import copy
import itertools
import re
from logging import Logger
from pathlib import Path
from typing import Callable, Optional

import torch

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.model_manager.configs.base import Diffusers_Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.load.load_base import LoadedModel, ModelLoaderBase
from invokeai.backend.model_manager.load.memory_snapshot import GB, MemorySnapshot
from invokeai.backend.model_manager.load.model_cache.cache_record import CacheRecord
from invokeai.backend.model_manager.load.model_cache.model_cache import (
    MODEL_LOAD_LOCK,
    ModelCache,
    get_model_cache_key,
)
from invokeai.backend.model_manager.load.model_util import calc_model_size_by_fs
from invokeai.backend.model_manager.load.optimizations import skip_torch_weight_init
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    SubModelType,
)
from invokeai.backend.quantization.fp8_scaled import count_fp8_weights, should_keep_fp8_weights
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.fp8 import FP8_COMPUTE_DTYPE_ATTR, set_fp8_compute_dtype

# Probe results keyed by concrete device (e.g. "xpu:1"). float8 support is build/driver
# dependent, so it is a per-device property: a discrete Arc may be paired with an integrated
# GPU that answers differently, and keying on the device *type* would let whichever probed
# first decide for both.
_FP8_STORAGE_SUPPORTED: set[str] = set()

# Devices whose probe failure has already been reported. Deliberately separate from the result
# cache above: the probe is retried on every request (a failure may be transient), but repeating
# the warning on every model load would bury the log.
_FP8_PROBE_FAILURE_REPORTED: set[str] = set()


def put_in_eval_mode(model: AnyModel) -> AnyModel:
    """Put a freshly constructed model into inference mode.

    Applied once here rather than in each loader, because it is the loaders that keep getting this
    wrong and there are dozens of them. `from_pretrained` calls `.eval()` on what it returns, but a
    great many loaders build the module themselves — `accelerate.init_empty_weights()` plus
    `load_state_dict()` — which leaves `training` True. Anything dropout- or batchnorm-sensitive in
    such a tree then behaves as if it were training, silently, during inference.

    `_load_model` is the single construction choke point every loader passes through, so covering it
    here also covers loaders that do not exist yet. `.eval()` is idempotent, so the loaders that
    already call it are unaffected. Non-module returns (tokenizers, schedulers, IP-Adapter wrappers,
    pipelines) pass through untouched.
    """
    if isinstance(model, torch.nn.Module):
        model.eval()
    return model


def resolve_submodel_path(config: AnyModelConfig, submodel_type: SubModelType, fallback: Path) -> Path:
    """Where a pipeline component actually lives, preferring what identification recorded.

    `model_index.json` names its components with arbitrary keys, and discovery stores the key it saw
    in `submodels[...].path_or_prefix`. Reconstructing `model_path / "<slot name>"` at load time
    instead assumes the key always equals the slot name, so a pipeline that calls its CLIP encoder
    something else passes discovery and is then loaded from a directory that does not exist.

    `fallback` is used when the config carries no such entry — configs persisted before submodel
    discovery existed, and the layouts where the component folder *is* the model path.
    """
    discovered = (getattr(config, "submodels", None) or {}).get(submodel_type)
    return Path(discovered.path_or_prefix) if discovered else fallback


def _device_supports_fp8_storage(device: torch.device, logger: Optional[Logger] = None) -> bool:
    """Whether FP8 layerwise casting (float8 weight storage + upcast) is usable on this device.

    The feature needs only float8 *storage* and casting to the compute dtype -- not native FP8
    matmul -- so it holds on CUDA and, for current torch builds, on Intel XPU. XPU float8 support is
    build/driver dependent ("emerging" on Xe2), so probe it rather than assume.

    The probe allocates on the *given* device rather than an index-less ``"xpu"``, which would
    resolve through the thread's current XPU device -- not necessarily the device the caller is
    loading onto (see the idle-GPU encoder offload, which re-pins the session device).

    The probe mirrors the runtime path rather than approximating it. At runtime the storage cast
    happens on CPU (``_apply_fp8_to_nn_module`` runs while params are still CPU-resident), the fp8
    tensor is then copied host->device, and the pre-hook upcasts fp8 -> compute_dtype on the
    device. Probing all three steps on the device would pass on a build where the fp8 host->device
    copy or a particular upcast fails, and then break at forward time.

    Only successes are cached. The probe runs during a model load, i.e. exactly when the device
    may be transiently out of memory, and a cached failure would silently disable FP8 for the
    lifetime of the process with no remedy short of a restart.
    """
    if device.type == "cuda":
        return True
    if device.type != "xpu":
        return False

    device = TorchDevice.normalize(device)
    key = str(device)
    if key in _FP8_STORAGE_SUPPORTED:
        return True

    try:
        # 1. Storage cast, on CPU, as _apply_fp8_to_nn_module does.
        stored = torch.zeros(2).to(torch.float8_e4m3fn)
        # 2. fp8 host->device copy.
        stored = stored.to(device)
        # 3. Pre-hook upcast, on device. Both targets are exercised: compute_dtype is bfloat16 for
        #    several supported models (Krea-2, FLUX) and float16 for others, and a build can
        #    support one without the other.
        stored.to(torch.bfloat16)
        stored.to(torch.float16)
    except Exception as exc:
        if logger is not None and key not in _FP8_PROBE_FAILURE_REPORTED:
            _FP8_PROBE_FAILURE_REPORTED.add(key)
            logger.warning(f"FP8 storage probe failed on {device} ({type(exc).__name__}: {exc}); not using FP8.")
        return False

    _FP8_STORAGE_SUPPORTED.add(key)
    return True


# Layer classes that benefit from FP8 storage. Mirrors diffusers'
# `_GO_LC_SUPPORTED_PYTORCH_LAYERS` so the plain-nn.Module fallback path makes the same
# precision/quality trade-offs as the ModelMixin path. Notably excludes norm and embedding
# wrapper modules — those are handled by their direct param types (Embedding is included
# but pos_embed/patch_embed are filtered by `_FP8_DEFAULT_SKIP_PATTERNS`).
_FP8_SUPPORTED_PYTORCH_LAYERS: tuple[type[torch.nn.Module], ...] = (
    torch.nn.Linear,
    torch.nn.Conv1d,
    torch.nn.Conv2d,
    torch.nn.Conv3d,
    torch.nn.ConvTranspose1d,
    torch.nn.ConvTranspose2d,
    torch.nn.ConvTranspose3d,
    torch.nn.Embedding,
)

# Module-path regexes (matched against `named_modules()` dotted paths) for precision-sensitive
# layers that should never be cast to FP8. Mirrors diffusers' `DEFAULT_SKIP_MODULES_PATTERN`
# — without these, FLUX RMSNorm.scale and similar tiny learned scalars get crushed to FP8 and
# inference quality degrades. Includes anything named `norm`, position/patch embeddings, and
# the in/out projection of transformer blocks.
_FP8_DEFAULT_SKIP_PATTERNS: tuple[str, ...] = (
    "pos_embed",
    "patch_embed",
    "norm",
    r"^proj_in$",
    r"^proj_out$",
)

# Model formats whose weights are already quantized. FP8 storage is meaningless for them (the
# payload is packed integers, not values we may re-encode) and actively harmful — see
# `_should_use_fp8`. Declared as strings to keep this module free of a taxonomy import at module
# scope; compared against `config.format`, which is a `ModelFormat` str-enum. Must list every
# quantized member of `ModelFormat`; `test_quantized_format_set_matches_the_taxonomy` pins the
# strings to the enum so a rename cannot silently disable the check.
_QUANTIZED_MODEL_FORMATS: frozenset[str] = frozenset(
    {
        "gguf_quantized",
        "bnb_quantized_nf4b",
        "bnb_quantized_int8b",
        "sdnq_quantized",
    }
)


def _is_quantized_param(param: torch.nn.Parameter) -> bool:
    """Whether `param` holds a quantized payload that must not be re-encoded as FP8.

    Two signals, both observed in practice:

    - Not floating point. bnb's NF4/INT8 weights are packed `uint8` (and `bnb.nn.LinearNF4`
      subclasses `nn.Linear`, so a class check alone does not catch them). Casting those to float8
      succeeds silently and the layer then returns finite garbage.
    - A `torch.Tensor` *subclass*, e.g. `GGMLTensor`, which keeps its quantized payload plus
      metadata and rejects dtype changes outright.
    """
    return not param.data.is_floating_point() or type(param.data) is not torch.Tensor


# The construction path is not thread-safe on its own; it monkey-patches process-global torch state
# (see MODEL_LOAD_LOCK). Concurrent callers must hold the MODEL_LOAD_LOCK write lock (see
# _load_and_cache).
class ModelLoader(ModelLoaderBase):
    """Default implementation of ModelLoaderBase."""

    def __init__(
        self,
        app_config: InvokeAIAppConfig,
        logger: Logger,
        ram_cache: ModelCache,
    ):
        """Initialize the loader."""
        self._app_config = app_config
        self._logger = logger
        self._ram_cache = ram_cache
        self._torch_dtype = TorchDevice.choose_torch_dtype()
        self._torch_device = TorchDevice.choose_torch_device()

    def load_model(self, model_config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> LoadedModel:
        """
        Return a model given its configuration.

        Given a model's configuration as returned by the ModelRecordConfigStore service,
        return a LoadedModel object that can be used for inference.

        :param model config: Configuration record for this model
        :param submodel_type: an ModelType enum indicating the portion of
               the model to retrieve (e.g. ModelType.Vae)
        """
        model_path = self._get_model_path(model_config)

        if not model_path.exists():
            raise FileNotFoundError(f"Files for model '{model_config.name}' not found at {model_path}")

        cache_record = self._load_and_cache(model_config, submodel_type)
        return LoadedModel(config=model_config, cache_record=cache_record, cache=self._ram_cache)

    @property
    def ram_cache(self) -> ModelCache:
        """Return the ram cache associated with this loader."""
        return self._ram_cache

    def _get_model_path(self, config: AnyModelConfig) -> Path:
        model_base = self._app_config.models_path
        return (model_base / config.path).resolve()

    def _get_execution_device(
        self, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None
    ) -> Optional[torch.device]:
        """Determine the execution device for a model based on its configuration.

        CPU-only execution is only applied to text encoder submodels to save VRAM while keeping
        the denoiser on GPU for performance. Conditioning tensors are moved to GPU after encoding.

        Returns:
            torch.device("cpu") if the model should run on CPU only, None otherwise (use cache default).
        """
        # Check if this is a text encoder submodel of a main model with cpu_only setting
        if hasattr(config, "default_settings") and config.default_settings is not None:
            if hasattr(config.default_settings, "cpu_only") and config.default_settings.cpu_only is True:
                # Only apply CPU execution to text encoder submodels
                if submodel_type in [SubModelType.TextEncoder, SubModelType.TextEncoder2, SubModelType.TextEncoder3]:
                    return torch.device("cpu")

        # Check if this is a standalone text encoder config with cpu_only field (T5Encoder, Qwen3Encoder, etc.)
        if hasattr(config, "cpu_only") and config.cpu_only is True:
            return torch.device("cpu")

        return None

    def _load_and_cache(self, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> CacheRecord:
        stats_name = ":".join([config.base, config.type, config.name, (submodel_type or "")])
        cache_key = get_model_cache_key(config.key, submodel_type)
        try:
            return self._ram_cache.get(key=cache_key, stats_name=stats_name)
        except IndexError:
            pass

        # Cache miss: construct the model from disk. This path holds the MODEL_LOAD_LOCK *write*
        # lock because it relies on process-global, non-thread-safe monkey-patches
        # (skip_torch_weight_init and, inside the loaders, accelerate.init_empty_weights / diffusers
        # low_cpu_mem_usage). The write lock excludes both other constructions AND concurrent VRAM
        # load/unload on other workers (which take the read lock); without that, a concurrent move's
        # load_state_dict(assign=True) -> register_parameter gets hijacked onto the `meta` device.
        # See MODEL_LOAD_LOCK for the full explanation.
        #
        # Lock-ordering: the write lock is acquired before any ModelCache._lock taken below
        # (get/make_room/put), matching the readers' order, so there is no AB-BA deadlock.
        with MODEL_LOAD_LOCK.write_lock():
            # Double-checked locking: another worker sharing this cache may have loaded the same
            # entry while we waited for the mutex. (Workers on other devices use a different cache,
            # so they will still miss here and construct their own copy — which is intended.)
            try:
                return self._ram_cache.get(key=cache_key, stats_name=stats_name)
            except IndexError:
                pass

            config.path = str(self._get_model_path(config))

            # Fast path (multi-GPU): if another device already loaded this exact model, its canonical
            # CPU weights are still resident in the shared store along with an empty (meta-weight)
            # clone of the built module. Adopt those weights instead of re-reading the model from
            # disk — this avoids both the redundant disk read and the large transient second copy
            # that would otherwise spike RAM (and, on a RAM-constrained box, drive the system into
            # swap). Any failure falls back to a normal load, so it can never change the result.
            loaded_model = self._try_adopt_shared_weights(cache_key)

            shell_to_register: Optional[torch.nn.Module] = None
            if loaded_model is None:
                # Optional RAM instrumentation for the cold disk-load path (the only place that runs
                # `from_pretrained`, whose construction transient can briefly spike RAM past the
                # cache's retained budget). Gated on `log_memory_usage`; captures process RAM before
                # make_room, after make_room (retained baseline), and after construction (transient
                # peak) so the surge can be attributed without guessing.
                log_mem = self._app_config.log_memory_usage
                ram_before = MemorySnapshot.capture().process_ram if log_mem else 0
                self._ram_cache.make_room(self.get_size_fs(config, Path(config.path), submodel_type))
                ram_after_room = MemorySnapshot.capture().process_ram if log_mem else 0
                with skip_torch_weight_init():
                    loaded_model = put_in_eval_mode(self._load_model(config, submodel_type))
                if log_mem:
                    ram_peak = MemorySnapshot.capture().process_ram
                    self._logger.info(
                        f"Cold load RAM for '{cache_key}': "
                        f"make_room {ram_before / GB:.2f}->{ram_after_room / GB:.2f}GB "
                        f"({(ram_after_room - ram_before) / GB:+.2f}), "
                        f"construct {ram_after_room / GB:.2f}->{ram_peak / GB:.2f}GB "
                        f"({(ram_peak - ram_after_room) / GB:+.2f}) [transient peak]"
                    )
                # Snapshot a meta-weight clone now — before put() applies custom layers or any VRAM
                # move — so the next device to load this model can adopt these weights (see above).
                # Skipped in single-device setups, where no other cache will ever adopt it.
                shared_store = self._ram_cache.shared_cpu_weights
                if shared_store is not None and shared_store.enable_shell_capture:
                    shell_to_register = self._build_meta_shell(loaded_model)

            # Determine execution device from model config, considering submodel type
            execution_device = self._get_execution_device(config, submodel_type)

            self._ram_cache.put(
                cache_key,
                model=loaded_model,
                execution_device=execution_device,
            )
            # Retrieve immediately: the new record carries the cache's post-admission grace until
            # it is locked, and keeping put() and get() adjacent means no failure in between can
            # leave a graced record whose loader never comes back for it.
            cache_record = self._ram_cache.get(key=cache_key, stats_name=stats_name)

            # Register the shell only after put() has created the shared entry (via the wrapper's
            # acquire); it is dropped automatically when that entry's last reference is released.
            if shell_to_register is not None:
                shared_store = self._ram_cache.shared_cpu_weights
                if shared_store is not None:
                    shared_store.set_shell(cache_key, shell_to_register)

            return cache_record

    def get_size_fs(
        self, config: AnyModelConfig, model_path: Path, submodel_type: Optional[SubModelType] = None
    ) -> int:
        """Get the size of the model on disk."""
        # Size the folder the model will actually be loaded from. This has to track
        # `resolve_submodel_path`, or a pipeline whose index calls its CLIP encoder `clip_encoder`
        # gets sized at the non-existent `text_encoder/` — 0 bytes — and `make_room()` reserves
        # nothing before a multi-GB component is read. The conventional case is unchanged: only a
        # component recorded somewhere other than its slot name takes the branch below.
        subfolder = submodel_type.value if submodel_type else None
        if submodel_type is not None:
            conventional = model_path / submodel_type.value
            resolved = resolve_submodel_path(config, submodel_type, conventional)
            if resolved != conventional:
                try:
                    subfolder = resolved.relative_to(model_path).as_posix()
                except ValueError:
                    # Recorded outside this model's directory — size that directory directly.
                    model_path, subfolder = resolved, None

        return calc_model_size_by_fs(
            model_path=model_path,
            subfolder=subfolder,
            variant=config.repo_variant if isinstance(config, Diffusers_Config_Base) else None,
        )

    def _should_use_fp8(self, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> bool:
        """Check if FP8 layerwise casting should be applied to a model."""
        from invokeai.backend.model_manager.taxonomy import ModelType

        # Already-quantized models are excluded. Their weights are packed integer payloads, not
        # values we may re-encode, and casting them is not a no-op:
        #   - GGUF raises `Operation changed the dtype of GGMLTensor unexpectedly`.
        #   - bnb NF4 corrupts *silently* — `bnb.nn.LinearNF4` subclasses `nn.Linear`, so the packed
        #     uint8 payload is cast to float8, inference still returns finite numbers, and the model
        #     just produces garbage.
        # No quantized-format loader calls `_apply_fp8_layerwise_casting` today, so this is a guard
        # against the next loader that gets wired up (they are being added one model at a time)
        # rather than a fix for a live crash.
        if hasattr(config, "format") and config.format in _QUANTIZED_MODEL_FORMATS:
            return False

        # VAEs are excluded — fp8 storage causes noticeable quality degradation in decode.
        if hasattr(config, "type") and config.type == ModelType.VAE:
            return False

        # LoRAs (including ControlLoRA) are excluded — they are not run as a standalone forward pass,
        # they are patched into a base model, so the layerwise-casting hooks would never fire. The
        # toggle is also hidden in the UI for ControlLoRA; this guard handles legacy persisted values.
        if hasattr(config, "type") and config.type in (ModelType.LoRA, ModelType.ControlLoRa):
            return False

        # Don't apply FP8 to text encoders, tokenizers, schedulers, VAEs, etc.
        # The prompt enhancer is a causal LM run autoregressively (one full forward per generated
        # token), so layerwise casting would pay the bf16<->fp8 round trip on every token — and its
        # entire job is text quality, which fp8 rounding degrades. Its tokenizer is listed for
        # symmetry (it is not an nn.Module, so casting is a no-op today only by accident).
        _excluded_submodel_types = {
            SubModelType.TextEncoder,
            SubModelType.TextEncoder2,
            SubModelType.TextEncoder3,
            SubModelType.Tokenizer,
            SubModelType.Tokenizer2,
            SubModelType.Tokenizer3,
            SubModelType.PromptEnhancer,
            SubModelType.PromptEnhancerTokenizer,
            SubModelType.Scheduler,
            SubModelType.SafetyChecker,
            SubModelType.VAE,
            SubModelType.VAEDecoder,
            SubModelType.VAEEncoder,
        }
        if submodel_type in _excluded_submodel_types:
            return False

        # Check default_settings.fp8_storage (Main models, ControlNet)
        if hasattr(config, "default_settings") and config.default_settings is not None:
            if hasattr(config.default_settings, "fp8_storage") and config.default_settings.fp8_storage is True:
                # Device support is probed last, so it runs only for a model that actually wants
                # FP8 -- not on the first load of any tokenizer/VAE/scheduler, and not on API or
                # install threads, where it would force XPU lazy SYCL init on a thread that never
                # generates.
                return _device_supports_fp8_storage(self._torch_device, self._logger)

        return False

    def _apply_fp8_layerwise_casting(
        self, model: AnyModel, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None
    ) -> AnyModel:
        """Apply FP8 layerwise casting to a model if enabled in its config."""
        if not self._should_use_fp8(config, submodel_type):
            return model

        # The cast is not idempotent: on a second pass the first parameter is already fp8, so the
        # compute dtype below would be derived as float8. The marker is set by
        # `_apply_fp8_to_nn_module`, so its presence means this model has already been cast.
        if isinstance(model, torch.nn.Module) and getattr(model, FP8_COMPUTE_DTYPE_ATTR, None) is not None:
            return model

        # A checkpoint that already ships fp8 weights is running (or is about to run) on the fp8
        # tensor cores. Layerwise casting would install hooks that restore the compute dtype before
        # every forward, so `CustomLinear._can_use_fp8_matmul` would no longer see an fp8 weight and
        # would silently fall back to the dequantized path — the VRAM toggle would make the model
        # *slower* with no indication why. Storage has nothing to add here anyway: the weights are
        # already 1 byte per parameter.
        if isinstance(model, torch.nn.Module) and should_keep_fp8_weights(self._torch_device):
            already_fp8 = count_fp8_weights(model)
            if already_fp8:
                self._logger.info(
                    f"FP8 storage skipped for {config.name}: {already_fp8} weight(s) are already fp8 and "
                    "are being run on the fp8 tensor cores (fp8_compute). Layerwise casting would "
                    "disable that matmul without saving any further VRAM."
                )
                return model

        storage_dtype = torch.float8_e4m3fn
        compute_dtype = self._torch_dtype

        # Detect the model's current dtype to use as compute dtype, since models
        # (e.g. Flux) may require a specific dtype (bf16) that differs from the global torch dtype (fp16).
        if isinstance(model, torch.nn.Module):
            first_param = next(model.parameters(), None)
            if first_param is not None:
                compute_dtype = first_param.dtype

        # We use our own hook-based path for every nn.Module — including diffusers ModelMixin —
        # rather than `model.enable_layerwise_casting()`. Diffusers' LayerwiseCastingHook installs
        # an instance-level `forward` attribute that captures the original `Linear.forward` in a
        # closure. `ModelCache.put()` later runs `apply_custom_layers_to_model`, which constructs a
        # new `CustomLinear` sharing the original Linear's `__dict__` — so the diffusers wrapper
        # carries over and routes calls back to the captured original forward, silently bypassing
        # `CustomLinear.forward` and its `cast_to_device` autocast. With partial loading (e.g. FLUX.2
        # Klein 9B) some weights stay on CPU, the diffusers pre_forward only casts dtype, and
        # `F.linear` then sees input on cuda and weight on cpu. Our `register_forward_pre_hook` /
        # `register_forward_hook` path fires around `nn.Module._call_impl` without replacing
        # `forward`, so `CustomLinear.forward` is still reached.
        if isinstance(model, torch.nn.Module):
            # Diffusers models declare their own precision-sensitive modules in
            # `_skip_layerwise_casting_patterns`, and `enable_layerwise_casting()` honors them. Since
            # we no longer call it, we have to apply that list ourselves — it is not cosmetic. Z-Image's
            # `TimestepEmbedder.forward` reads `self.mlp[0].weight.dtype` and casts its *input* to it;
            # with an fp8 weight the input becomes float8 before our pre-hook can restore the weight,
            # and `F.linear` dies with `"addmm_cuda" not implemented for 'Float8_e4m3fn'`. Hence
            # `['t_embedder', 'cap_embedder']` for that model.
            self._apply_fp8_to_nn_module(
                model,
                storage_dtype=storage_dtype,
                compute_dtype=compute_dtype,
                extra_skip_patterns=tuple(getattr(model, "_skip_layerwise_casting_patterns", None) or ()),
            )
        else:
            return model

        param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
        self._logger.info(
            f"FP8 layerwise casting enabled for {config.name} "
            f"(storage=float8_e4m3fn, compute={compute_dtype}, "
            f"param_size={param_bytes / (1024**2):.0f}MB)"
        )
        return model

    @staticmethod
    def _apply_fp8_to_nn_module(
        model: torch.nn.Module,
        storage_dtype: torch.dtype,
        compute_dtype: torch.dtype,
        extra_skip_patterns: tuple[str, ...] = (),
        skip: Optional[Callable[[str, torch.nn.Module], bool]] = None,
    ) -> None:
        """Apply FP8 layerwise casting to a plain nn.Module.

        Mirrors diffusers' `apply_layerwise_casting` semantics: only the layer classes in
        `_FP8_SUPPORTED_PYTORCH_LAYERS` are cast, and modules whose dotted path matches any of
        `_FP8_DEFAULT_SKIP_PATTERNS` (norm, pos_embed, patch_embed, proj_in/out) are skipped.
        Without the skip list, precision-sensitive tiny learned scalars (e.g. FLUX RMSNorm.scale)
        get crushed to FP8 and quality degrades noticeably.

        `extra_skip_patterns` carries the model's own declared exclusions (diffusers'
        `_skip_layerwise_casting_patterns`), which are model-specific and cannot be inferred from
        layer types or generic name patterns.

        `skip` excludes further modules by (dotted name, module). Its one caller uses it to leave
        scaled-fp8 layers alone: those already hold fp8 weights plus a `weight_scale`, and the cast
        hooks installed here would upcast them *without* applying that scale — a silently wrong
        weight. Casting only the remainder lets a partly-quantized checkpoint (fp8 language model,
        bf16 visual tower) end up fully fp8-resident.

        Modules holding already-quantized weights are skipped regardless of their class. This is a
        backstop behind the format check in `_should_use_fp8`, which cannot see quantization that
        is not reflected in the model's format (e.g. a `diffusers`-format checkpoint whose weights
        were quantized by an external tool).

        Records the compute dtype on the model. After the cast, `model.dtype` reports the float8
        storage dtype, which must never be used to create or cast tensors — torch has no arithmetic
        kernels for it (see `get_model_compute_dtype`). The marker is set here rather than at the
        call sites so a new caller cannot forget it.
        """
        set_fp8_compute_dtype(model, compute_dtype)

        skip_patterns = _FP8_DEFAULT_SKIP_PATTERNS + tuple(extra_skip_patterns)
        for module_name, module in model.named_modules():
            if not isinstance(module, _FP8_SUPPORTED_PYTORCH_LAYERS):
                continue
            if any(re.search(pattern, module_name) for pattern in skip_patterns):
                continue
            if skip is not None and skip(module_name, module):
                continue
            params = list(module.parameters(recurse=False))
            if not params:
                continue
            if any(_is_quantized_param(p) for p in params):
                continue

            for param in params:
                param.data = param.data.to(storage_dtype)

            ModelLoader._wrap_forward_with_fp8_cast(module, storage_dtype, compute_dtype)

    @staticmethod
    def _wrap_forward_with_fp8_cast(
        module: torch.nn.Module, storage_dtype: torch.dtype, compute_dtype: torch.dtype
    ) -> None:
        """Register pre/post forward hooks that cast params to compute dtype on entry and back
        to storage dtype on exit.

        We use hooks (rather than overriding `module.forward`) for two reasons:

        1. **Correct dispatch after `apply_custom_layers_to_model`.** `ModelCache.put()` calls
           `apply_custom_layers_to_model`, which creates a NEW `CustomLinear` instance and
           shares the original `Linear.__dict__` (see `wrap_custom_layer`). Anything stored in
           that dict — including an instance-level `forward` attribute — gets carried over to
           the new object. An overridden `forward` would close over the OLD instance, so calls
           to the new `CustomLinear` would silently route to `Linear.forward(old_instance, ...)`
           and bypass the LoRA-patch-aware branch in `CustomLinear.forward`. Hooks, by contrast,
           live in `_forward_hooks` / `_forward_pre_hooks` and are dispatched by
           `nn.Module.__call__` with the *actual* called instance — so they run on the new
           `CustomLinear` and the class's `forward` is still resolved normally.

        2. **Exception safety.** `register_forward_hook(..., always_call=True)` fires the
           post-hook even when `forward` raises. The plain pre-hook/post-hook pair without
           `always_call` would leave params in compute dtype on exception, defeating FP8
           storage savings and making cache size accounting stale.
        """

        def pre_hook(mod: torch.nn.Module, _args: object) -> None:
            for p in mod.parameters(recurse=False):
                p.data = p.data.to(compute_dtype)

        def post_hook(mod: torch.nn.Module, _args: object, _output: object) -> None:
            for p in mod.parameters(recurse=False):
                p.data = p.data.to(storage_dtype)

        module.register_forward_pre_hook(pre_hook)
        module.register_forward_hook(post_hook, always_call=True)

    def _try_adopt_shared_weights(self, cache_key: str) -> Optional[AnyModel]:
        """Build this model by adopting another device's already-resident CPU weights, skipping the
        disk read entirely.

        Returns the constructed model, or None if adoption is unavailable or fails for any reason (in
        which case the caller loads the model from disk normally). Loader-agnostic: it deep-copies the
        meta-weight shell that the first device registered (`_build_meta_shell`) and assigns the
        shared canonical weights into the copy — no per-loader architecture knowledge required, and
        fp8 cast hooks carried by the shell are preserved automatically.

        Must be called while holding the MODEL_LOAD_LOCK write lock (as `_load_and_cache` does), so
        the peeked canonical weights and shell cannot be evicted between the peek and the adopt.
        """
        shared_store = self._ram_cache.shared_cpu_weights
        if shared_store is None:
            return None
        canonical = shared_store.peek(cache_key)
        shell = shared_store.get_shell(cache_key)
        if canonical is None or shell is None:
            return None

        try:
            # Independent module per device (its params will be moved to its own GPU); deep-copying an
            # all-meta shell is cheap (no weight data). assign=True then re-points the copy's
            # parameters at the shared canonical tensors with no allocation.
            model = copy.deepcopy(shell)
            model.load_state_dict(canonical, assign=True)
            # Safety net: if anything is left on the meta device (e.g. a persistent buffer somehow
            # missing from the canonical state dict) the model would silently produce wrong results.
            for tensor in itertools.chain(model.parameters(), model.buffers()):
                if tensor.is_meta:
                    raise RuntimeError("adopted model has tensors left on the meta device")
        except Exception as e:
            # Adoption is best-effort; never let it break a load. Fall back to a normal disk load.
            self._logger.warning(
                f"Could not adopt shared CPU weights for '{cache_key}' ({e!r}); loading from disk instead."
            )
            return None

        self._logger.info(
            f"Adopted shared CPU weights for '{cache_key}' from another device's cache (skipped disk load)."
        )
        return model

    @staticmethod
    def _build_meta_shell(model: AnyModel) -> Optional[torch.nn.Module]:
        """Return an empty, meta-weight structural clone of `model`, or None if it can't be cloned.

        The clone has the identical module structure, registered hooks (e.g. the fp8 layerwise-cast
        hooks), and non-persistent buffers as `model`, but every parameter and persistent buffer is
        replaced by a 0-byte tensor on the `meta` device. A second device adopts it by deep-copying
        and assigning the shared canonical weights — so this works for every model family (diffusers,
        single-file checkpoint, GGUF, transformers) without any per-loader code.

        Best-effort: returns None on any failure (the model then simply isn't adoptable, and the next
        device loads it from disk as before).
        """
        if not isinstance(model, torch.nn.Module):
            return None

        def _meta_like(t: torch.Tensor) -> torch.Tensor:
            # A 0-byte stand-in with the same logical shape/dtype as `t`; replaced by the canonical
            # tensor on adoption (load_state_dict(assign=True)), so only its shape needs to match.
            # `torch.empty_like` is preferred (preserves layout etc.) but is NOT implemented by some
            # tensor subclasses — notably the GGUF `GGMLTensor`, whose `__torch_dispatch__` returns
            # NotImplemented for `aten.empty_like`. That made `_build_meta_shell` throw on the first
            # parameter of every GGUF model (e.g. a Q8_0 quantized transformer), silently disabling
            # cross-device adoption for exactly the largest models. For those, fall back to a plain
            # meta tensor built from the subclass's reported (dequantized) shape and dtype.
            try:
                return torch.empty_like(t, device="meta")
            except TypeError:
                return torch.empty(t.shape, dtype=t.dtype, device="meta")

        try:
            # Persistent buffers come from the canonical state dict on adoption, so they (like params)
            # are replaced by meta placeholders. Non-persistent buffers are NOT in the state dict, so
            # they must be carried over with real data (deepcopy copies them); they are typically
            # small (e.g. rotary-embedding tables, attention masks).
            persistent_names = set(model.state_dict().keys())
            persistent_buffer_ids = {id(b) for n, b in model.named_buffers() if n in persistent_names}

            memo: dict[int, object] = {}
            for param in model.parameters(recurse=True):
                memo[id(param)] = torch.nn.Parameter(_meta_like(param), requires_grad=param.requires_grad)
            for buffer in model.buffers(recurse=True):
                if id(buffer) in persistent_buffer_ids:
                    memo[id(buffer)] = _meta_like(buffer)

            return copy.deepcopy(model, memo)
        except Exception as e:
            # Best-effort: an un-clonable model simply isn't adoptable (the next device loads it from
            # disk). Log at debug so a newly-unadoptable model family can be diagnosed rather than
            # silently double-loading on every device.
            from invokeai.backend.util.logging import InvokeAILogger

            InvokeAILogger.get_logger().debug(
                f"Could not build meta-weight shell for {type(model).__name__} ({e!r}); model won't be adopted."
            )
            return None

    # This needs to be implemented in the subclass
    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        raise NotImplementedError

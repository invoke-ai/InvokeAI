from contextlib import ExitStack
from typing import Literal, Optional

import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    Ideogram4ConditioningField,
    Input,
    InputField,
)
from invokeai.app.invocations.model import TransformerField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.architectures import resolve_latent_space
from invokeai.backend.ideogram4 import run_ideogram4_denoise
from invokeai.backend.ideogram4.latent_norm import get_latent_norm
from invokeai.backend.ideogram4.modeling_ideogram4 import Ideogram4Transformer, compute_dtype_of
from invokeai.backend.ideogram4.sampler_configs import PRESETS
from invokeai.backend.ideogram4.sampling_utils import PIXELS_PER_IMAGE_TOKEN, unpatchify_and_denormalize
from invokeai.backend.ideogram4.transformer_pair import Ideogram4TransformerPair
from invokeai.backend.model_manager.load.load_base import LoadedModel
from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.patches.layer_patcher import LayerPatcher, PatchSpec
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.lora_conversions.ideogram4_lora_constants import IDEOGRAM4_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.quantization.dequantizing_linear import peak_dequant_transient_bytes, requires_sidecar_patching
from invokeai.backend.quantization.fp8_rowwise import Fp8Linear
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import Ideogram4ConditioningInfo
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.fp8 import FP8_STORAGE_DTYPES, get_model_compute_dtype

# Named sampler presets bundle step count, guidance schedule (with polish tail), and the
# logit-normal schedule mean/std. V4_QUALITY_48 is the reference default.
IDEOGRAM4_SAMPLER_PRESETS = Literal["V4_QUALITY_48", "V4_DEFAULT_20", "V4_TURBO_12"]

# The floor the derived residency target is clamped to, so a device short by more than a whole
# branch still keeps a working set instead of streaming every layer of every step. The same 2 GiB
# Wan holds its transformer to, and for the same reason.
IDEOGRAM4_MIN_RESIDENT_SECOND_BRANCH_BYTES = 2 * 2**30

# What the pair is asked to leave unclaimed for everything that is not this generation. On a
# workstation the generation GPU usually also drives the display: measured 1.2-1.8 GiB held by the
# desktop while idle here, and it spikes above that when a browser or the app's own UI repaints.
# It is a term in the target, not a guarantee -- see `_second_branch_residency_target` for what the
# reservation's own error does to it.
IDEOGRAM4_LEAVE_FREE_FOR_THE_SYSTEM_BYTES = 3 * 2**30

# The LoRAs on 'Transformer' are applied to both branches. Every published Ideogram 4 LoRA is one set of weights
# trained on the conditional model, and the references disagree on the unconditional pass: ai-toolkit samples both
# passes through one LoRA-patched transformer, diffusers loads LoRAs into the conditional transformer only. Rendered
# side by side (GGUF pair, same seeds), the conditional-only variant overshoots -- guidance scales the LoRA's whole
# effect, so a style LoRA comes out oversaturated and TurboTime's images crush to dark, high-contrast frames --
# while both-branch patching keeps the style at the same strength without that.
# Listed in patch order: a patch object both branches share ends up on the device of the branch patched last, and the
# conditional branch is the one partial loading and the residency cap leave on the GPU.
_LORA_BRANCHES = ("unconditional", "conditional")


def _sidecar_weight_transient_bytes(patch_specs: list[PatchSpec], model: torch.nn.Module) -> int:
    """Peak bytes a sidecar materializes per forward for the LoRAs' weight-space layers.

    A low-rank layer runs in activation space, and its extra output copy is within the per-token estimate. A LoKR
    layer is applied in weight space instead: the patched module's weight is materialized (dequantized, for a GGUF
    build), its delta built, and the two summed, in up to float32 -- three copies of that module's weight. On
    `attention.qkv` that is ~0.75 GiB, which no activation estimate covers.
    """
    largest = 0
    for spec in patch_specs:
        for key, layer in spec[0].layers.items():
            if isinstance(layer, LoRALayer):
                continue
            try:
                module = model.get_submodule(key.removeprefix(IDEOGRAM4_LORA_TRANSFORMER_PREFIX))
            except AttributeError:
                continue  # skipped here as the patcher skips it, with its own warning
            largest = max(largest, module.in_features * module.out_features)
    return 3 * 4 * largest


def _branch_weights(
    cached_weights: Optional[dict[str, torch.Tensor]], branch: str
) -> Optional[dict[str, torch.Tensor]]:
    """A pair's RAM copy of its weights, narrowed to one branch and keyed like that branch's own parameters."""
    if cached_weights is None:
        return None
    prefix = f"{branch}."
    return {key[len(prefix) :]: value for key, value in cached_weights.items() if key.startswith(prefix)}


def _effective_guidance_schedule(
    base_schedule: tuple[float, ...], preset_num_steps: int, num_steps: int, guidance_scale: Optional[float]
) -> tuple[float, ...]:
    """Build the per-step guidance schedule for the (possibly overridden) step count.

    The preset schedule is ``(polish_gw,)*N_polish + (main_gw,)*N_main`` in loop-index order
    (index 0 = the final/polish step). A ``guidance_scale`` override replaces the main weight while
    the preset's polish tail is preserved; a changed step count rescales the polish tail
    proportionally (always keeping at least one polish and one main step).

    ``num_steps`` must be >= 2 (enforced by the invocation's ``steps`` field) so both a polish and a
    main step always exist — otherwise a single step would be all-polish and silently drop the
    ``guidance_scale`` override.
    """
    polish_gw = base_schedule[0]
    main_gw = float(guidance_scale) if guidance_scale is not None else float(base_schedule[-1])
    if num_steps == preset_num_steps and guidance_scale is None:
        return base_schedule
    n_polish_base = sum(1 for gw in base_schedule if gw == base_schedule[0])
    # Cap the polish tail at num_steps - 1 so at least one main step always remains and the
    # guidance_scale override is never silently dropped.
    polish_count = max(1, min(round(n_polish_base * num_steps / preset_num_steps), num_steps - 1))
    main_count = num_steps - polish_count
    return (polish_gw,) * polish_count + (main_gw,) * main_count


@invocation(
    "ideogram4_denoise",
    title="Denoise - Ideogram 4",
    tags=["image", "ideogram4"],
    category="latents",
    version="1.2.0",
    classification=Classification.Prototype,
)
class Ideogram4DenoiseInvocation(BaseInvocation):
    """Runs the Ideogram 4 dual-branch flow-matching denoising loop (text-to-image)."""

    transformer: TransformerField = InputField(
        description=FieldDescriptions.transformer, input=Input.Connection, title="Transformer"
    )
    unconditional_transformer: Optional[TransformerField] = InputField(
        default=None,
        description="The unconditional branch when it is a separate single-file model. Leave "
        "unconnected for a diffusers pipeline, whose Transformer submodel carries both branches.",
        input=Input.Connection,
        title="Transformer (Unconditional)",
    )
    positive_conditioning: Ideogram4ConditioningField = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    sampler_preset: IDEOGRAM4_SAMPLER_PRESETS = InputField(
        default="V4_QUALITY_48",
        description="Sampler preset (steps + guidance schedule + schedule mean/std).",
        title="Sampler Preset",
    )
    width: int = InputField(default=1024, multiple_of=16, description="Width of the generated image.")
    height: int = InputField(default=1024, multiple_of=16, description="Height of the generated image.")
    seed: int = InputField(default=0, description="Randomness seed for reproducibility.")
    # Optional advanced overrides of the sampler preset. None = use the preset's value.
    steps: Optional[int] = InputField(
        default=None,
        ge=2,
        le=100,
        description="Override the preset's step count (minimum 2, so a polish and a main step both "
        "exist). Leave empty to use the preset.",
    )
    guidance_scale: Optional[float] = InputField(
        default=None,
        ge=1.0,
        le=20.0,
        description="Override the main guidance weight (the preset's polish tail is preserved). "
        "Empty = use the preset.",
    )
    mu: Optional[float] = InputField(
        default=None,
        ge=-4.0,
        le=4.0,
        description="Override the logit-normal schedule mean (resolution-adjusted internally). Empty = use the preset.",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        device = TorchDevice.choose_torch_device()
        preset = PRESETS[self.sampler_preset]

        # Apply optional advanced overrides on top of the preset.
        num_steps = self.steps if self.steps is not None else preset.num_steps
        mu = self.mu if self.mu is not None else preset.mu
        guidance_schedule = _effective_guidance_schedule(
            preset.guidance_schedule, preset.num_steps, num_steps, self.guidance_scale
        )

        # Load conditioning (the stacked Qwen3-VL features).
        cond_data = context.conditioning.load(self.positive_conditioning.conditioning_name)
        assert len(cond_data.conditionings) == 1
        info = cond_data.conditionings[0]
        assert isinstance(info, Ideogram4ConditioningInfo)
        llm_features = info.prompt_embeds.to(device=device, dtype=torch.float32)

        # Denormalization params come from get_latent_norm (no VAE).
        latent_shift, latent_scale = get_latent_norm()

        def step_callback(step: int, total: int, packed_latents: torch.Tensor) -> None:
            # The projection and the downscale come from what this architecture declares, which is
            # the same source the shared denoise callback reads. This was a second copy of the
            # FLUX.2 constants with the 8x downscale hardcoded — and Ideogram 4 was missing from
            # that shared dispatch entirely, so reading either one could not have revealed the other.
            preview = None
            preview_size = None
            try:
                # packed_latents: (1, LATENT_DIM, grid_h, grid_w) -> VAE latent (1, 32, H/8, W/8).
                vae_latent = unpatchify_and_denormalize(
                    packed_latents.float(),
                    latent_shift.to(packed_latents.device),
                    latent_scale.to(packed_latents.device),
                )
                latent_space = resolve_latent_space(BaseModelType.Ideogram4, vae_latent)
                preview = latent_space.preview(vae_latent)
                preview_size = (
                    preview.width * latent_space.spatial_compression,
                    preview.height * latent_space.spatial_compression,
                )
            except Exception:
                # A preview must never break generation — fall back to a plain progress signal.
                preview = None
            if preview is not None:
                context.util.signal_progress(
                    "Running Ideogram 4 denoising",
                    step / total,
                    preview,
                    preview_size,
                )
            else:
                context.util.signal_progress("Running Ideogram 4 denoising", step / total)

        with ExitStack() as stack:
            conditional, unconditional = self._load_branches(
                context, stack, self._estimate_working_memory(int(llm_features.shape[0]))
            )
            packed = run_ideogram4_denoise(
                conditional_transformer=conditional,
                unconditional_transformer=unconditional,
                llm_features=llm_features,
                height=self.height,
                width=self.width,
                num_steps=num_steps,
                mu=mu,
                std=preset.std,
                guidance_schedule=guidance_schedule,
                seed=self.seed,
                device=device,
                step_callback=step_callback,
            )

        packed = packed.detach().to("cpu")
        name = context.tensors.save(tensor=packed)
        return LatentsOutput.build(latents_name=name, latents=packed, seed=None)

    def _estimate_working_memory(self, num_text_tokens: int) -> int:
        """Activation headroom to reserve, in bytes, so the cache does not fill VRAM with weights.

        Without a reservation the cache loads both branches up to the last free byte, and the
        activations then evict the very weights being used: on a 24 GB card the single-file fp8
        pair (17.5 GB resident) spent minutes per step at 94 W, i.e. copying rather than computing.

        Two contributions, both linear in the sequence length:

        * the transformer's own activations (residual stream at 4608 plus one block's attention and
          SwiGLU intermediates) — Krea-2's estimator measures 0.5 MB/token for a comparable MMDiT and
          this model is the same order;
        * Ideogram's conditioning buffers, which are unusually large: `llm_features` is 53248 wide,
          and the loop materializes one buffer over the full packed sequence plus a second over the
          image tokens alone (2 x 53248 x 2 bytes ~ 0.2 MB/token).

        The fixed base covers what does not scale with resolution (fp8 weight-cast transients, the
        VAE-free denormalisation buffers, allocator slack).

        Deliberately not clamped, and the consequence is worth stating. Ideogram 4 is the only
        architecture here that keeps *two* transformers resident, so on a 24 GB card with the fp8
        pair (~17.4 GiB) the headroom runs out somewhere above 1300px: past that the cache cannot
        satisfy the reservation, the second branch loses residency and streams. That is not the
        estimate being wrong -- the memory genuinely is not there -- and reserving less would only
        exchange a slow generation for an out-of-memory error, and the remedy is fewer resident
        bytes rather than a smaller number here. The int8 build does not supply them: 8.9 GiB per
        branch against fp8's 8.7, and fp8 reaches that with `fp8_compute` *or* with FP8 Storage,
        which installation switches on for such a file and which keeps the same weights without the
        matmul. What int8 buys is that this holds on *every* device: with neither of those two the
        fp8 pair expands to 17.3 GiB per branch, and a 24 GB card then runs out of memory during the
        first step.
        """
        image_tokens = (self.height // PIXELS_PER_IMAGE_TOKEN) * (self.width // PIXELS_PER_IMAGE_TOKEN)
        per_token_bytes = 3 * 1024**2 // 4  # 0.75 MiB
        base_bytes = 3 * 1024**3 // 2  # 1.5 GiB
        return (image_tokens + num_text_tokens) * per_token_bytes + base_bytes

    @staticmethod
    def _dequant_transient(model: object) -> int:
        """What an int8 or GGUF build transiently needs to dequantize its largest layer, per forward.

        `Int8ConvrotLinear` keeps the stored codes and a GGUF Linear its packed blocks, and both
        materialize the dequantized weight inside `forward`, so that peak is not part of the model's
        resident size and has to fit inside the caller's reservation. Zero for a bf16 or fp8 build --
        which is why it is measured from the model rather than from the resolution: the two branches
        are separate models and may be different builds.

        The row-wise fp8 build's `Fp8Linear` dequantizes per forward as well, but one fp8 weight cast is
        what `_estimate_working_memory`'s fixed base was sized to hold, for this build and the scaled-fp8
        single files alike. Counting it again cost the 24 GB card its full residency: measured at
        1024x1024, V4_DEFAULT_20, the pair dropped from 100% to 96.8% resident and from 34.0 s to 41-42 s.
        """
        if not isinstance(model, torch.nn.Module):
            return 0
        if any(isinstance(module, Fp8Linear) for module in model.modules()):
            return 0
        return peak_dequant_transient_bytes(model, get_model_compute_dtype(model))

    def _load_branches(
        self, context: InvocationContext, stack: ExitStack, working_mem_bytes: int
    ) -> tuple[Ideogram4Transformer, Ideogram4Transformer]:
        """Put both transformer branches on the device and keep them there for the whole loop.

        A diffusers pipeline yields both in one cache entity (`Ideogram4TransformerPair`); single
        files are two models and are locked simultaneously, because every step runs both and
        releasing one between steps would make the cache stream it back for the next.

        Both locks reserve the same *activation* headroom, even though the dequantization transient
        is per branch and the two branches can be different builds. The cache recomputes what is
        free at each lock and the second cannot claw space back from the first, which is already
        locked: a first branch that reserved less has taken headroom the second one still needs,
        and with the second branch already resident from an earlier run there is nothing left to
        evict. The second lock is then given *more* than that, when the pair would otherwise fill
        the device -- see `_second_branch_residency_target` for what is added and why.

        Taking the maximum up front costs reading `LoadedModel.model` -- documented as returning the
        model unlocked, and it is already constructed by then -- and loading the second branch into
        RAM before the first is locked. Under the default `keep_ram_copy_of_weights` that is not a
        new peak, since both branches hold their RAM copies through the loop anyway; with RAM copies
        off it is one cold load's worth, which the cache absorbs the same way it absorbs any
        overshoot. Neither branch can be evicted meanwhile: `LoadedModel` takes a first-use hold the
        moment it is constructed, and `make_room` skips a record that has one.
        """
        conditional_info = context.models.load(self.transformer.transformer)
        both_in_one = isinstance(conditional_info.model, Ideogram4TransformerPair)

        # Checked before the second model is loaded, so a mis-wired graph costs an error rather than
        # a ~9 GiB read. Real checks, not asserts: a hand-built graph can wire anything here, and
        # under `python -O` an assert would vanish and leave the mismatch to surface inside the loop.
        if both_in_one and self.unconditional_transformer is not None:
            raise ValueError(
                "'Transformer' already carries both Ideogram 4 branches, so 'Transformer (Unconditional)' "
                "must not be connected. Disconnect it, or select a single-file checkpoint as the model."
            )
        if not both_in_one and self.unconditional_transformer is None:
            raise ValueError(
                "This Ideogram 4 transformer holds only one branch, so 'Transformer (Unconditional)' "
                "must be connected as well. The Ideogram 4 model loader emits it when the model is a "
                "single-file checkpoint."
            )

        second = self.unconditional_transformer
        if second is not None and second.loras:
            raise ValueError(
                "LoRAs on 'Transformer (Unconditional)' are not applied. Connect the LoRA loader to 'Transformer'; "
                "the node applies those LoRAs to both branches."
            )
        unconditional_info = None if second is None else context.models.load(second.transformer)
        reservation = working_mem_bytes + max(
            self._dequant_transient(conditional_info.model),
            0 if unconditional_info is None else self._dequant_transient(unconditional_info.model),
        )

        # How each branch takes the LoRAs, decided before the locks: sidecar patches stay on the device for the whole
        # loop, so the reservation has to include them. One patch object serves both branches; it is counted once.
        sidecar = self._lora_sidecar_by_branch(context, conditional_info, unconditional_info)
        if any(sidecar.values()):
            patch_specs = self._materialize_lora_patches(context)
            reservation += sum(spec[0].calc_size() for spec in patch_specs)
            # The branches share their architecture, so either one answers for the module sizes.
            branch_model = conditional_info.model
            if isinstance(branch_model, Ideogram4TransformerPair):
                branch_model = branch_model.conditional
            reservation += _sidecar_weight_transient_bytes(patch_specs, branch_model)

        primary_cached, primary = stack.enter_context(conditional_info.model_on_device(working_mem_bytes=reservation))
        if unconditional_info is None:
            # Narrowing, not validation: `both_in_one` was decided from this same object above, and
            # locking it does not change what it is. The user-facing refusals are the two checks
            # further up, which is why they are `raise` and this is not.
            assert isinstance(primary, Ideogram4TransformerPair)
            # A pipeline's pair is ONE cache record holding both branches, so the residency cap
            # below does not apply to it: it has no second record to hold back, and bounding the
            # single record would stream both branches instead of one. Such a build is also the
            # smaller problem — `transformer_pair.py` notes its co-residency is what makes the nf4
            # build fit in 24 GB in the first place.
            for branch, use_sidecar in sidecar.items():
                branch_model = getattr(primary, branch)
                self._apply_loras(context, stack, branch_model, _branch_weights(primary_cached, branch), use_sidecar)
            return primary.conditional, primary.unconditional

        # Told to the cache *before* the lock, so the branch settles near its target instead of
        # loading whole and being pushed back. It lands near rather than on it: the cache budgets
        # from `free + reclaimable-allocator-bytes - working_mem` and spends that on the weights the
        # model is still *missing*, neither of which this arithmetic models. Measured on the
        # reference card the lock overshoots by a steady 0.23 GiB, cold and warm alike, which the
        # correction after the validation below takes off again.
        target = self._second_branch_residency_target(context, conditional_info, unconditional_info, reservation)
        second_reservation = reservation
        if target is not None:
            free_bytes, _total = torch.cuda.mem_get_info(unconditional_info.compute_device)
            second_reservation = max(reservation, free_bytes - target)

        unconditional_cached, unconditional = stack.enter_context(
            unconditional_info.model_on_device(working_mem_bytes=second_reservation)
        )
        for role, branch in (("Transformer", primary), ("Transformer (Unconditional)", unconditional)):
            if not isinstance(branch, Ideogram4Transformer):
                raise ValueError(
                    f"'{role}' is a {type(branch).__name__}, not an Ideogram 4 transformer. Both inputs "
                    "must come from the Ideogram 4 model loader."
                )
        # After the cheap validation, so a mis-wired graph errors out instead of first paying an
        # unload it will never use.
        if target is not None:
            self._hold_the_second_branch_to(context, unconditional_info, target)
        # Patched only now: a weight patched in place and then streamed out by the hold above would come back from
        # the unpatched RAM copy.
        locked = {"conditional": (primary, primary_cached), "unconditional": (unconditional, unconditional_cached)}
        for branch, use_sidecar in sidecar.items():
            model, cached_weights = locked[branch]
            self._apply_loras(context, stack, model, cached_weights, use_sidecar)
        return primary, unconditional

    def _lora_sidecar_by_branch(
        self, context: InvocationContext, conditional_info: LoadedModel, unconditional_info: Optional[LoadedModel]
    ) -> dict[str, bool]:
        """For each branch the LoRAs go to, whether they are applied as a sidecar (and so stay on the device).

        GGUF, int8 and the other dequantizing builds keep packed weights a patch cannot be added to; fp8 weights have
        no add kernel, so the patcher picks the sidecar for them by itself. Read from the unlocked models. A branch
        that is no Ideogram 4 transformer gets nothing here; the check after the locks refuses it with a message.
        """
        if not self.transformer.loras:
            return {}
        if unconditional_info is not None and not all(
            isinstance(info.model, Ideogram4Transformer) for info in (conditional_info, unconditional_info)
        ):
            return {}
        if unconditional_info is None:
            pair = conditional_info.model
            assert isinstance(pair, Ideogram4TransformerPair)
            model_format = context.models.get_config(self.transformer.transformer).format
            branches = {branch: (getattr(pair, branch), model_format) for branch in _LORA_BRANCHES}
        else:
            assert self.unconditional_transformer is not None
            by_branch = {
                "conditional": (conditional_info.model, self.transformer.transformer),
                "unconditional": (unconditional_info.model, self.unconditional_transformer.transformer),
            }
            branches = {
                branch: (by_branch[branch][0], context.models.get_config(by_branch[branch][1]).format)
                for branch in _LORA_BRANCHES
            }
        return {
            branch: requires_sidecar_patching(model, model_format)
            or any(p.dtype in FP8_STORAGE_DTYPES for p in model.parameters())
            for branch, (model, model_format) in branches.items()
        }

    def _materialize_lora_patches(self, context: InvocationContext) -> list[PatchSpec]:
        """Load every LoRA on 'Transformer' into (patch, weight, cache-pin) specs.

        A pin can be entered only once, so every patcher call gets a list of its own. A pin that is never entered
        (the sizing pass before the locks) takes nothing.
        """
        patch_specs: list[PatchSpec] = []
        for lora in self.transformer.loras:
            lora_info = context.models.load(lora.lora)
            if not isinstance(lora_info.model, ModelPatchRaw):
                raise TypeError(
                    f"Expected ModelPatchRaw for LoRA '{lora.lora.key}', got {type(lora_info.model).__name__}."
                )
            patch_specs.append((lora_info.model, lora.weight, lora_info.model_in_ram()))
        return patch_specs

    def _apply_loras(
        self,
        context: InvocationContext,
        stack: ExitStack,
        model: torch.nn.Module,
        cached_weights: Optional[dict[str, torch.Tensor]],
        use_sidecar: bool,
    ) -> None:
        """Patch one branch for the rest of the loop.

        The patches are cast to the dtype the branch computes in, read the way the loop reads it. A model-level
        guess goes by the first parameter, and on the nf4 build that is a packed uint8 weight: the LoRA would be
        cast to uint8, and the cache's own copy of it come back from the patcher corrupted.
        """
        assert isinstance(model, Ideogram4Transformer)
        stack.enter_context(
            LayerPatcher.apply_smart_model_patches(
                model=model,
                patches=self._materialize_lora_patches(context),
                prefix=IDEOGRAM4_LORA_TRANSFORMER_PREFIX,
                dtype=compute_dtype_of(model.input_proj),
                cached_weights=cached_weights,
                force_sidecar_patching=use_sidecar,
            )
        )

    @staticmethod
    def _second_branch_residency_target(
        context: InvocationContext,
        first_branch: LoadedModel,
        second_branch: LoadedModel,
        reservation: int,
    ) -> Optional[int]:
        """How many of the unconditional branch's weight bytes may stay on the device, or None.

        Ideogram 4 is the only node here that keeps two transformers resident, and the fp8 pair is
        17.4 GiB. On a 24 GB card that plus the loop's reservation is the whole device: measured at
        1024x1024, the two branches settled at 100% and 96.8% residency and the card peaked at 23288
        of 24564 MiB, leaving nothing for whatever else owns the display. Nothing is wrong with the
        cache's arithmetic -- `capacity - working_mem - in_use` is exactly what it promised -- the
        pair simply does not leave room to be a good neighbour, so the node has to decide not to take
        it all. Same purpose as Wan's expert swapper, which bounds its transformer's residency
        through the reservation for the same reason.

        The target is derived, not chosen: whatever is left once the system's share, the reservation
        the cache will really apply, and the *other* branch are subtracted from the device. That
        keeps the cost proportional to how badly the pair overflows, and it makes
        `device_working_mem_gb` the knob for it -- raising the reserve lowers the target one for one,
        with no second setting to discover. Measured on the reference card at 1024x1024, 8 steps:

            uncapped                      38.2s, peak 23288 MiB, 1.2 GiB free
            derived, reserve 5 GiB        35.8s, peak 22117 MiB, 2.4 GiB free (branch B at 84%)
            derived, reserve 8 GiB        40.2s, peak 19009 MiB, 5.4 GiB free (branch B at 50%)

        Stated plainly: at the default reserve this does *not* deliver the whole share the constant
        below names. The reservation is an estimate of the loop's activations and measurably a low
        one -- at 1024x1024 the non-weight bytes at peak were ~8 GiB against an estimate near 5 --
        so the target it yields is correspondingly generous. The arithmetic is still the right shape
        (more reserve, less residency, continuously), and the honest remedy for a machine that needs
        more is the reserve, not a fudge factor hidden in this method.

        Returns None when no cap is wanted or possible: a device that fits the pair with the
        system's share to spare, a branch the cache holds whole (partial loading off, where a
        partial unload would drop every weight instead), or a non-CUDA device.

        Both branch sizes are read rather than doubling one of them. The loader lets an int8 branch
        guide against an fp8 one, and `_estimate_working_memory` records the spread those builds
        have -- 8.7, 8.9, or 17.3 GiB when an fp8 file is expanded -- so a pair can be lopsided
        enough that doubling either side answers the wrong question.
        """
        device = second_branch.compute_device
        if not second_branch.supports_partial_loading or device.type != "cuda":
            return None
        # An explicit `max_cache_vram_gb` makes the cache budget from that cap rather than from the
        # device, so a target derived from physical memory describes a machine the cache is not
        # using: the inflated reservation would be subtracted from the cap a second time and push
        # the branch below the floor this method exists to hold. Such an install has already chosen
        # how much VRAM the cache may take, which is this policy by another route.
        if context.config.get().max_cache_vram_gb is not None:
            return None
        _free, total = torch.cuda.mem_get_info(device)
        # The cache raises any reservation below its configured floor (`_get_vram_available`), so
        # the node's own estimate is not what will be held free. Asking with the smaller number
        # would clear the gate at low resolutions while the cache reserved more and the pair still
        # took everything.
        effective_reservation = max(reservation, int(context.config.get().device_working_mem_gb * 2**30))
        room = total - IDEOGRAM4_LEAVE_FREE_FOR_THE_SYSTEM_BYTES - effective_reservation - first_branch.weight_bytes
        if room >= second_branch.weight_bytes:
            return None
        return max(room, IDEOGRAM4_MIN_RESIDENT_SECOND_BRANCH_BYTES)

    @staticmethod
    def _hold_the_second_branch_to(context: InvocationContext, second_branch: LoadedModel, target: int) -> None:
        """Correct the second branch down to `target` if its lock still settled above it.

        The reservation handed to that lock is the primary lever and usually lands it there, but it
        is computed from `mem_get_info` while the cache decides from its own accounting, and another
        load can slip into a paced lock's gap. Without this the guarantee would be approximate; with
        it the reservation saves the transfer and this only ever trims what is left over.

        `keep_required_weights_in_vram` leaves behind the tensors a streamed forward cannot fetch
        per layer, such as quantization scales.
        """
        excess = second_branch.resident_weight_bytes - target
        if excess <= 0:
            return
        freed = second_branch.unload_from_vram(excess, keep_required_weights_in_vram=True)
        # The cache moves the weights to RAM but does not return the blocks to the driver, so
        # without this the bytes stay in torch's reserve -- invisible to the display, which is who
        # the room was made for.
        TorchDevice.empty_cache()
        context.logger.info(
            f"Ideogram 4: streamed {freed / 2**30:.2f} GiB of the unconditional branch to leave room "
            f"for the rest of the system on {second_branch.compute_device}; this trades generation "
            f"time for VRAM and does not happen on a card that fits both branches."
        )

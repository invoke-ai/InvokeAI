# InvokeAI - New Model Type Integration Checklist

This documentation describes all the steps required to integrate a new model type into InvokeAI. The implementations of FLUX.1, FLUX.2 Klein, SD3, SDXL, and Z-Image serve as references.

---

## Table of Contents

1. [Backend: Model Manager](#1-backend-model-manager)
2. [Backend: Model Configs](#2-backend-model-configs)
3. [Backend: Model Loader](#3-backend-model-loader)
4. [Backend: Invocations](#4-backend-invocations)
5. [Backend: Sampling and Denoise](#5-backend-sampling-and-denoise)
6. [Frontend: Graph Building](#6-frontend-graph-building)
7. [Frontend: State Management](#7-frontend-state-management)
8. [Frontend: Parameter Recall](#8-frontend-parameter-recall)
9. [Metadata and Generation Modes](#9-metadata-and-generation-modes)
10. [Starter Models](#10-starter-models)
11. [Optional Features](#11-optional-features)

---

## 1. Backend: Model Manager

### 1.1 Add BaseModelType

**File:** `invokeai/backend/model_manager/taxonomy.py`

```python
class BaseModelType(str, Enum):
    # Existing types
    StableDiffusion1 = "sd-1"
    StableDiffusion2 = "sd-2"
    StableDiffusionXL = "sdxl"
    Flux = "flux"
    Flux2 = "flux2"        # FLUX.2 Klein
    SD3 = "sd-3"
    ZImage = "z-image"
    # NEW:
    NewModel = "newmodel"
```

### 1.2 Add Variant Type (if needed)

**File:** `invokeai/backend/model_manager/taxonomy.py`

```python
# Examples of existing variants:
class FluxVariantType(str, Enum):
    Schnell = "schnell"
    Dev = "dev"
    DevFill = "dev_fill"

class Flux2VariantType(str, Enum):
    Klein4B = "klein_4b"    # Qwen3 4B encoder
    Klein9B = "klein_9b"    # Qwen3 8B distilled
    Klein9BBase = "klein_9b_base"

# NEW (if needed):
class NewModelVariantType(str, Enum):
    VariantA = "variant_a"
    VariantB = "variant_b"
```

### 1.3 Define Default Settings

**File:** `invokeai/backend/model_manager/configs/main.py`

```python
class MainModelDefaultSettings:
    @staticmethod
    def from_base(base: BaseModelType, variant: AnyVariant | None = None):
        match base:
            case BaseModelType.Flux2:
                if variant == Flux2VariantType.Klein9BBase:
                    return MainModelDefaultSettings(steps=28, cfg_scale=1.0, ...)
                return MainModelDefaultSettings(steps=4, cfg_scale=1.0, ...)
            # NEW:
            case BaseModelType.NewModel:
                return MainModelDefaultSettings(steps=20, cfg_scale=7.0, ...)
```

### Backend Model Manager Checklist

- [ ] Extend `BaseModelType` enum (`taxonomy.py`)
- [ ] Create variant enum if needed (`taxonomy.py`)
- [ ] Update `AnyVariant` union (`taxonomy.py`)
- [ ] Add default settings in `from_base()` (`configs/main.py`)

---

## 2. Backend: Model Configs

### 2.1 Create Main Model Config

**File:** `invokeai/backend/model_manager/configs/main.py`

```python
# Checkpoint Format
@ModelConfigFactory.register
class Main_Checkpoint_NewModel_Config(Checkpoint_Config_Base):
    type: Literal[ModelType.Main] = ModelType.Main
    base: Literal[BaseModelType.NewModel] = BaseModelType.NewModel
    format: Literal[ModelFormat.Checkpoint] = ModelFormat.Checkpoint
    variant: NewModelVariantType = NewModelVariantType.VariantA

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict) -> Self:
        if not cls._validate_is_newmodel(mod):
            raise NotAMatchError("Not a NewModel")
        variant = cls._get_variant_or_raise(mod)
        return cls(..., variant=variant)

# Diffusers Format
@ModelConfigFactory.register
class Main_Diffusers_NewModel_Config(Diffusers_Config_Base):
    type: Literal[ModelType.Main] = ModelType.Main
    base: Literal[BaseModelType.NewModel] = BaseModelType.NewModel
    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers
```

### 2.2 Detection Helper Functions

**File:** `invokeai/backend/model_manager/configs/main.py`

```python
def _is_newmodel(state_dict: dict) -> bool:
    """Detect if state dict belongs to NewModel architecture."""
    # Example FLUX.2 Klein detection:
    # - context_embedder.weight shape[1] > 4096 (Qwen3 vs T5)
    # - img_in.weight shape[1] == 128 (32 latent channels × 4)
    required_keys = ["transformer_blocks.0.attn.to_q.weight", ...]
    return all(key in state_dict for key in required_keys)

def _get_newmodel_variant(state_dict: dict) -> NewModelVariantType:
    """Determine variant from state dict."""
    # Example FLUX.2: context_in_dim distinguishes Klein 4B/9B
    context_dim = state_dict["context_embedder.weight"].shape[1]
    if context_dim == 7680:
        return NewModelVariantType.VariantA
    return NewModelVariantType.VariantB
```

### 2.3 VAE Config (if custom VAE)

**File:** `invokeai/backend/model_manager/configs/vae.py`

```python
@ModelConfigFactory.register
class VAE_Checkpoint_NewModel_Config(VAE_Checkpoint_Base):
    type: Literal[ModelType.VAE] = ModelType.VAE
    base: Literal[BaseModelType.NewModel] = BaseModelType.NewModel

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, ...) -> Self:
        if not _is_newmodel_vae(mod.state_dict):
            raise NotAMatchError()
        return cls(...)

def _is_newmodel_vae(state_dict: dict) -> bool:
    # Example FLUX.2: Check for BN layers (bn.running_mean)
    return "encoder.bn.running_mean" in state_dict
```

### 2.4 Text Encoder Config (if custom encoder)

**File:** `invokeai/backend/model_manager/configs/[encoder_type].py`

```python
def _has_newmodel_encoder_keys(state_dict: dict) -> bool:
    """Check if state dict contains NewModel encoder keys."""
    required_keys = ["model.layers.0.", "model.embed_tokens.weight"]
    return any(
        key.startswith(indicator) or key == indicator
        for key in state_dict.keys()
        for indicator in required_keys
        if isinstance(key, str)
    )

@ModelConfigFactory.register
class NewModelEncoder_Checkpoint_Config(Checkpoint_Config_Base):
    """Configuration for single-file NewModel Encoder models."""

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.NewModelEncoder] = Field(default=ModelType.NewModelEncoder)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict) -> Self:
        raise_if_not_file(mod)
        raise_for_override_fields(cls, override_fields)

        if not _has_newmodel_encoder_keys(mod.load_state_dict()):
            raise NotAMatchError("state dict does not look like a NewModel encoder")

        return cls(**override_fields)

@ModelConfigFactory.register
class NewModelEncoder_Diffusers_Config(Config_Base):
    """Configuration for NewModel Encoder in diffusers directory format."""

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.NewModelEncoder] = Field(default=ModelType.NewModelEncoder)
    format: Literal[ModelFormat.Diffusers] = Field(default=ModelFormat.Diffusers)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict) -> Self:
        raise_if_not_dir(mod)
        raise_for_override_fields(cls, override_fields)

        # Check for text_encoder config
        config_path = mod.path / "text_encoder" / "config.json"
        if not config_path.exists():
            raise NotAMatchError(f"config file not found: {config_path}")

        raise_for_class_name(config_path, {"NewModelForCausalLM"})

        return cls(**override_fields)
```

Examples of existing implementations:
- `t5_encoder.py` - T5 Encoder for FLUX.1, SD3
- `qwen3_encoder.py` - Qwen3 Encoder for FLUX.2 Klein, Z-Image
- `clip_embed.py` - CLIP Encoder for SDXL, SD3

### 2.5 Update AnyModelConfig Union

**File:** `invokeai/backend/model_manager/configs/factory.py`

```python
AnyModelConfig = Annotated[
    # ... existing configs
    Main_Checkpoint_NewModel_Config |
    Main_Diffusers_NewModel_Config |
    VAE_Checkpoint_NewModel_Config,
    Discriminator(...)
]
```

### Backend Model Configs Checklist

- [ ] Create main checkpoint config (`configs/main.py`)
- [ ] Create main diffusers config (`configs/main.py`)
- [ ] Create detection helper functions (`_is_newmodel()`, `_get_variant()`)
- [ ] Create VAE config if custom VAE (`configs/vae.py`)
- [ ] Create text encoder config if custom encoder
- [ ] Update `AnyModelConfig` union (`configs/factory.py`)

---

## 3. Backend: Model Loader

### 3.1 Create Model Loader

**File:** `invokeai/backend/model_manager/load/model_loaders/[newmodel].py`

```python
@ModelLoaderRegistry.register(
    base=BaseModelType.NewModel,
    type=ModelType.Main,
    format=ModelFormat.Checkpoint
)
class NewModelLoader(ModelLoader):
    def _load_model(self, config: AnyModelConfig, submodel_type: SubModelType | None) -> AnyModel:
        # Load and convert state dict
        state_dict = self._load_state_dict(config.path)

        # If format conversion needed (e.g., BFL → Diffusers):
        if self._is_bfl_format(state_dict):
            state_dict = self._convert_bfl_to_diffusers(state_dict)

        # Instantiate model
        model = NewModelTransformer(config=model_config)
        model.load_state_dict(state_dict)
        return model
```

### 3.2 VAE Loader (if custom VAE)

**File:** `invokeai/backend/model_manager/load/model_loaders/[newmodel].py`

```python
@ModelLoaderRegistry.register(
    base=BaseModelType.NewModel,
    type=ModelType.VAE,
    format=ModelFormat.Checkpoint
)
class NewModelVAELoader(ModelLoader):
    def _load_model(self, config, submodel_type) -> AnyModel:
        # Example FLUX.2: AutoencoderKLFlux2 with BN layers
        from diffusers import AutoencoderKLFlux2
        vae = AutoencoderKLFlux2.from_single_file(config.path)
        return vae
```

### 3.3 Text Encoder Loader (if custom encoder)

**File:** `invokeai/backend/model_manager/load/model_loaders/[newmodel].py`

```python
@ModelLoaderRegistry.register(
    base=BaseModelType.Any,
    type=ModelType.NewModelEncoder,
    format=ModelFormat.Checkpoint
)
class NewModelEncoderLoader(ModelLoader):
    """Load single-file NewModel Encoder models."""

    def _load_model(self, config: AnyModelConfig, submodel_type: SubModelType | None) -> AnyModel:
        match submodel_type:
            case SubModelType.TextEncoder:
                return self._load_text_encoder(config)
            case SubModelType.Tokenizer:
                # Load tokenizer from HuggingFace or local path
                return AutoTokenizer.from_pretrained("org/newmodel-base")

        raise ValueError(f"Unsupported submodel: {submodel_type}")

    def _load_text_encoder(self, config: AnyModelConfig) -> AnyModel:
        from safetensors.torch import load_file
        from transformers import NewModelConfig, NewModelForCausalLM

        # Load state dict and determine model configuration
        sd = load_file(config.path)

        # Detect model architecture from weights
        layer_count = self._count_layers(sd)
        hidden_size = sd["model.embed_tokens.weight"].shape[1]

        # Create model with detected configuration
        model_config = NewModelConfig(
            hidden_size=hidden_size,
            num_hidden_layers=layer_count,
            # ... other config parameters
        )

        with accelerate.init_empty_weights():
            model = NewModelForCausalLM(model_config)

        model.load_state_dict(sd, assign=True)
        return model
```

### Backend Model Loader Checklist

- [ ] Create and register main model loader
- [ ] Create VAE loader if custom VAE
- [ ] Create text encoder loader if custom encoder
- [ ] Implement state dict conversion if needed (different formats)
- [ ] Implement submodel loading (Diffusers format)

---

## 4. Backend: Invocations

### 4.1 Model Loader Invocation

**File:** `invokeai/app/invocations/[newmodel]_model_loader.py`

```python
@invocation("newmodel_model_loader", title="NewModel Loader", ...)
class NewModelModelLoaderInvocation(BaseInvocation):
    model: ModelIdentifierField = InputField(description="Main model")
    vae_model: ModelIdentifierField | None = InputField(default=None)
    encoder_model: ModelIdentifierField | None = InputField(default=None)

    def invoke(self, context: InvocationContext) -> NewModelLoaderOutput:
        # Load transformer
        transformer = self.model.model_copy(
            update={"submodel_type": SubModelType.Transformer}
        )
        # Load VAE (from main model or separately)
        if self.vae_model:
            vae = self.vae_model.model_copy(...)
        else:
            vae = self.model.model_copy(
                update={"submodel_type": SubModelType.VAE}
            )
        return NewModelLoaderOutput(transformer=transformer, vae=vae, ...)
```

### 4.2 Text Encoder Invocation

**File:** `invokeai/app/invocations/[newmodel]_text_encoder.py`

```python
@invocation("newmodel_text_encode", title="NewModel Text Encoder", ...)
class NewModelTextEncoderInvocation(BaseInvocation):
    prompt: str = InputField()
    encoder: EncoderField = InputField()

    def invoke(self, context: InvocationContext) -> ConditioningOutput:
        # 1. Tokenize the prompt
        with context.models.load(self.encoder.tokenizer) as tokenizer:
            input_ids = tokenizer(
                self.prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=256,
                truncation=True
            ).input_ids

        # 2. Run encoder and extract hidden states
        # Example FLUX.2 Klein/Z-Image: Extract specific layers and stack them
        # Different models use different layer extraction strategies:
        # - Some use the final hidden state only
        # - Others stack multiple intermediate layers for richer representations
        with context.models.load(self.encoder.text_encoder) as encoder:
            outputs = encoder(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            # Stack layers 9, 18, 27 to create combined text embedding
            # This captures features at different abstraction levels
            # Shape: (batch, seq_len, hidden_size) -> (batch, seq_len, hidden_size * 3)
            stacked_embeddings = torch.cat([
                hidden_states[9],
                hidden_states[18],
                hidden_states[27]
            ], dim=-1)

        # 3. Create conditioning data structure
        # The stacked embeddings become the text conditioning that guides denoising
        conditioning_data = ConditioningFieldData(
            conditionings=[
                BasicConditioningInfo(embeds=stacked_embeddings)
            ]
        )

        # 4. Save conditioning to context and return reference
        conditioning_name = context.conditioning.save(conditioning_data)
        return ConditioningOutput(
            conditioning=ConditioningField(conditioning_name=conditioning_name)
        )
```

### 4.3 Denoise Invocation

**File:** `invokeai/app/invocations/[newmodel]_denoise.py`

```python
@invocation("newmodel_denoise", title="NewModel Denoise", ...)
class NewModelDenoiseInvocation(BaseInvocation):
    # Standard Fields
    latents: LatentsField | None = InputField(default=None)
    positive_conditioning: ConditioningField = InputField()
    negative_conditioning: ConditioningField | None = InputField(default=None)

    # Model Fields
    transformer: TransformerField = InputField()

    # Denoise Parameters
    denoising_start: float = InputField(default=0.0, ge=0, le=1)
    denoising_end: float = InputField(default=1.0, ge=0, le=1)
    steps: int = InputField(default=20, ge=1)
    cfg_scale: float = InputField(default=7.0)

    # Image-to-Image / Inpainting
    denoise_mask: DenoiseMaskField | None = InputField(default=None)

    # Scheduler (if model-specific)
    scheduler: Literal["euler", "heun", "lcm"] = InputField(default="euler")

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        # 1. Generate noise
        noise = get_noise_newmodel(seed, height, width, ...)

        # 2. Pack latents (if needed)
        x = pack_newmodel(latents)

        # 3. Compute schedule
        timesteps = get_schedule_newmodel(num_steps, denoising_start, denoising_end)

        # 4. Denoising loop
        x = denoise(
            model=transformer,
            x=x,
            timesteps=timesteps,
            conditioning=conditioning,
            cfg_scale=self.cfg_scale,
            inpaint_extension=inpaint_extension,  # For inpainting
        )

        # 5. Unpack latents
        latents = unpack_newmodel(x)

        return LatentsOutput(latents=latents)
```

### 4.4 VAE Encode Invocation

**File:** `invokeai/app/invocations/[newmodel]_vae_encode.py`

```python
@invocation("newmodel_vae_encode", title="Image to Latents - NewModel", ...)
class NewModelVaeEncodeInvocation(BaseInvocation):
    image: ImageField = InputField()
    vae: VAEField = InputField()

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        image = context.images.get_pil(self.image.image_name)
        image_tensor = image_resized_to_grid_as_tensor(image.convert("RGB"))

        with context.models.load(self.vae.vae) as vae:
            latent_dist = vae.encode(image_tensor)
            latents = latent_dist.mode()  # Deterministic

        return LatentsOutput(latents=latents)
```

### 4.5 VAE Decode Invocation

**File:** `invokeai/app/invocations/[newmodel]_vae_decode.py`

```python
@invocation("newmodel_vae_decode", title="Latents to Image - NewModel", ...)
class NewModelVaeDecodeInvocation(BaseInvocation):
    latents: LatentsField = InputField()
    vae: VAEField = InputField()

    def invoke(self, context: InvocationContext) -> ImageOutput:
        latents = context.tensors.load(self.latents.latents_name)

        with context.models.load(self.vae.vae) as vae:
            # Example FLUX.2: BN denormalization before decode
            if hasattr(vae, "bn"):
                latents = self._bn_denormalize(latents, vae)

            image = vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)

        return ImageOutput(image=image)
```

### Backend Invocations Checklist

- [ ] Model loader invocation (`[newmodel]_model_loader.py`)
- [ ] Text encoder invocation (`[newmodel]_text_encoder.py`)
- [ ] Denoise invocation (`[newmodel]_denoise.py`)
- [ ] VAE encode invocation (`[newmodel]_vae_encode.py`)
- [ ] VAE decode invocation (`[newmodel]_vae_decode.py`)
- [ ] Define output classes (e.g., `NewModelLoaderOutput`)
- [ ] Define field classes if needed (e.g., `NewModelEncoderField`)

---

## 5. Backend: Sampling and Denoise

### 5.1 Sampling Utilities

**File:** `invokeai/backend/[newmodel]/sampling_utils.py`

```python
def get_noise_newmodel(
    num_samples: int,
    height: int,
    width: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate noise for NewModel.

    Example FLUX.2: 32 latent channels (vs 16 for FLUX.1)
    """
    latent_channels = 32  # Model-specific
    latent_h = height // 8
    latent_w = width // 8

    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(
        (num_samples, latent_channels, latent_h, latent_w),
        generator=generator,
        device=device,
        dtype=dtype,
    )

def pack_newmodel(x: torch.Tensor) -> torch.Tensor:
    """Pack latents for transformer input.

    Example FLUX: 2×2 patches → (B, H/2*W/2, C*4)
    """
    return rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

def unpack_newmodel(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Unpack transformer output to latents."""
    return rearrange(
        x, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=height // 16, w=width // 16, ph=2, pw=2
    )

def get_schedule_newmodel(
    num_steps: int,
    denoising_start: float = 0.0,
    denoising_end: float = 1.0,
) -> list[float]:
    """Create timestep schedule.

    Example FLUX.2 Klein: Linear schedule from 1.0 → 0.0
    """
    start_step = int(num_steps * denoising_start)
    end_step = int(num_steps * denoising_end)

    sigmas = torch.linspace(1.0, 0.0, num_steps + 1)
    return sigmas[start_step:end_step + 1].tolist()

def generate_img_ids_newmodel(batch_size: int, height: int, width: int) -> torch.Tensor:
    """Generate position IDs for transformer.

    Example FLUX.2: 4D position IDs (T, H, W, L)
    """
    # Model-specific position encoding
    pass
```

### 5.2 Denoise Function

**File:** `invokeai/backend/[newmodel]/denoise.py`

```python
def denoise(
    model: nn.Module,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    timesteps: list[float],
    cfg_scale: list[float],
    neg_txt: torch.Tensor | None = None,
    neg_txt_ids: torch.Tensor | None = None,
    scheduler: Any = None,
    inpaint_extension: RectifiedFlowInpaintExtension | None = None,
    step_callback: Callable | None = None,
) -> torch.Tensor:
    """Main denoising loop.

    Example FLUX.2 Klein:
    - No guidance_embeds (unlike FLUX.1 Dev)
    - Supports Euler, Heun, LCM schedulers
    - Integration with RectifiedFlowInpaintExtension
    """
    total_steps = len(timesteps) - 1

    for step_index in range(total_steps):
        t_curr = timesteps[step_index]
        t_prev = timesteps[step_index + 1]

        # CFG
        if cfg_scale[step_index] > 1.0 and neg_txt is not None:
            pred_pos = model(img, t_curr, txt, txt_ids, img_ids)
            pred_neg = model(img, t_curr, neg_txt, neg_txt_ids, img_ids)
            pred = pred_neg + cfg_scale[step_index] * (pred_pos - pred_neg)
        else:
            pred = model(img, t_curr, txt, txt_ids, img_ids)

        # Scheduler step or manual Euler
        if scheduler is not None:
            img = scheduler.step(pred, t_curr, img).prev_sample
        else:
            # Manual Euler: x = x + (t_prev - t_curr) * pred
            img = img + (t_prev - t_curr) * pred

        # Inpainting merge
        if inpaint_extension is not None:
            img = inpaint_extension.merge_intermediate_latents_with_init_latents(img, t_prev)

        # Progress callback
        if step_callback:
            step_callback(PipelineIntermediateState(step=step_index + 1, ...))

    return img
```

### 5.3 Scheduler (if model-specific)

**File:** `invokeai/backend/[newmodel]/schedulers.py` or use existing

```python
# Existing schedulers in invokeai/backend/flux/schedulers.py:
# - FlowMatchEulerDiscreteScheduler
# - FlowMatchHeunDiscreteScheduler
# - FlowMatchLCMScheduler

NEWMODEL_SCHEDULER_MAP = {
    "euler": FlowMatchEulerDiscreteScheduler,
    "heun": FlowMatchHeunDiscreteScheduler,
    "lcm": FlowMatchLCMScheduler,
}
```

### Backend Sampling and Denoise Checklist

- [ ] Noise generation (`get_noise_newmodel()`)
- [ ] Pack/unpack functions (if transformer-based)
- [ ] Schedule generation (`get_schedule_newmodel()`)
- [ ] Position ID generation (if needed)
- [ ] Implement denoise loop
- [ ] Scheduler integration
- [ ] Inpaint extension integration
- [ ] Progress callbacks

---

## 6. Frontend: Graph Building

### 6.1 Create Graph Builder

**File:** `invokeai/frontend/web/src/features/nodes/util/graph/generation/buildNewModelGraph.ts`

```typescript
export const buildNewModelGraph = async (arg: GraphBuilderArg): Promise<GraphBuilderResult> => {
  const { state, manager } = arg;
  const { model } = state.params;

  const g = new Graph();

  // 1. Model Loader
  const modelLoader = g.addNode({
    id: NEWMODEL_MODEL_LOADER,
    type: 'newmodel_model_loader',
    model: Graph.getModelMetadataField(model),
  });

  // 2. Text Encoder
  const positivePrompt = g.addNode({
    id: POSITIVE_CONDITIONING,
    type: 'newmodel_text_encode',
    prompt: positivePromptText,
  });
  g.addEdge(modelLoader, 'encoder', positivePrompt, 'encoder');

  // 3. Denoise Node
  const denoise = g.addNode({
    id: NEWMODEL_DENOISE,
    type: 'newmodel_denoise',
    steps,
    cfg_scale: cfg,
    scheduler: newmodelScheduler,
    denoising_start: 0,
    denoising_end: 1,
  });
  g.addEdge(modelLoader, 'transformer', denoise, 'transformer');
  g.addEdge(positivePrompt, 'conditioning', denoise, 'positive_conditioning');

  // 4. VAE Decode
  const l2i = g.addNode({
    id: NEWMODEL_VAE_DECODE,
    type: 'newmodel_vae_decode',
  });
  g.addEdge(modelLoader, 'vae', l2i, 'vae');
  g.addEdge(denoise, 'latents', l2i, 'latents');

  // 5. Generation Mode Handling
  let canvasOutput: Invocation<ImageOutputNodes> = l2i;

  switch (generationMode) {
    case 'txt2img':
      canvasOutput = addTextToImage({ g, state, denoise, l2i });
      g.upsertMetadata({ generation_mode: 'newmodel_txt2img' });
      break;
    case 'img2img':
      const i2l = g.addNode({ type: 'newmodel_vae_encode' });
      canvasOutput = await addImageToImage({ g, state, manager, denoise, l2i, i2l, ... });
      g.upsertMetadata({ generation_mode: 'newmodel_img2img' });
      break;
    case 'inpaint':
      canvasOutput = await addInpaint({ g, state, manager, denoise, l2i, i2l, ... });
      g.upsertMetadata({ generation_mode: 'newmodel_inpaint' });
      break;
    case 'outpaint':
      canvasOutput = await addOutpaint({ g, state, manager, denoise, l2i, i2l, ... });
      g.upsertMetadata({ generation_mode: 'newmodel_outpaint' });
      break;
  }

  return { g, noise, denoise, posCond: positivePrompt, ... };
};
```

### 6.2 Register Graph Builder

**File:** `invokeai/frontend/web/src/features/queue/hooks/useEnqueueCanvas.ts`

```typescript
// In the buildGraph function (around line 47-64):
switch (base) {
  case 'sd-1':
  case 'sd-2':
  case 'sdxl':
    return buildSD1Graph(arg);
  case 'flux':
    return buildFLUXGraph(arg);
  case 'flux2':
    return buildFLUXGraph(arg);  // FLUX.2 uses the same builder
  case 'sd-3':
    return buildSD3Graph(arg);
  case 'z-image':
    return buildZImageGraph(arg);
  // NEW:
  case 'newmodel':
    return buildNewModelGraph(arg);
}
```

### 6.3 Update Type Definitions

**File:** `invokeai/frontend/web/src/features/nodes/util/graph/types.ts`

```typescript
// Add node types:
export type ImageOutputNodes =
  | 'l2i' | 'flux_vae_decode' | 'flux2_vae_decode'
  | 'sd3_l2i' | 'newmodel_vae_decode';

export type LatentToImageNodes =
  | 'l2i' | 'flux_vae_decode' | 'flux2_vae_decode'
  | 'sd3_l2i' | 'newmodel_vae_decode';

export type ImageToLatentsNodes =
  | 'i2l' | 'flux_vae_encode' | 'flux2_vae_encode'
  | 'sd3_i2l' | 'newmodel_vae_encode';

export type DenoiseLatentsNodes =
  | 'denoise_latents' | 'flux_denoise' | 'flux2_denoise'
  | 'sd3_denoise' | 'newmodel_denoise';

export type MainModelLoaderNodes =
  | 'main_model_loader' | 'flux_model_loader' | 'flux2_klein_model_loader'
  | 'sd3_model_loader' | 'newmodel_model_loader';
```

### 6.4 Update Generation Mode Utilities

**Files:**
- `invokeai/frontend/web/src/features/nodes/util/graph/generation/addImageToImage.ts`
- `invokeai/frontend/web/src/features/nodes/util/graph/generation/addInpaint.ts`
- `invokeai/frontend/web/src/features/nodes/util/graph/generation/addOutpaint.ts`

```typescript
// In addImageToImage.ts - extend type check:
if (
  denoise.type === 'cogview4_denoise' ||
  denoise.type === 'flux_denoise' ||
  denoise.type === 'flux2_denoise' ||
  denoise.type === 'newmodel_denoise'  // NEW
) {
  // Rectified flow models: denoising_start instead of noise
}
```

### Frontend Graph Building Checklist

- [ ] Create graph builder (`buildNewModelGraph.ts`)
- [ ] Register graph builder in useEnqueueCanvas
- [ ] Update type definitions (`types.ts`)
- [ ] Extend node type unions (ImageOutputNodes, etc.)
- [ ] Update `addImageToImage.ts`
- [ ] Update `addInpaint.ts`
- [ ] Update `addOutpaint.ts`

---

## 7. Frontend: State Management

### 7.1 Add Parameter State

**File:** `invokeai/frontend/web/src/features/controlLayers/store/paramsSlice.ts`

```typescript
// Extend state interface:
interface ParamsState {
  // Existing fields
  fluxScheduler: 'euler' | 'heun' | 'lcm';
  zImageScheduler: 'euler' | 'heun' | 'lcm';

  // NEW: NewModel specific parameters
  newmodelScheduler: 'euler' | 'heun' | 'lcm';
  newmodelVaeModel: ParameterVAEModel | null;
  newmodelEncoderModel: ParameterModel | null;
}

// Initial state:
const initialState: ParamsState = {
  newmodelScheduler: 'euler',
  newmodelVaeModel: null,
  newmodelEncoderModel: null,
};

// Reducers:
reducers: {
  setNewmodelScheduler: (state, action: PayloadAction<'euler' | 'heun' | 'lcm'>) => {
    state.newmodelScheduler = action.payload;
  },
  newmodelVaeModelSelected: (state, action: PayloadAction<ParameterVAEModel | null>) => {
    state.newmodelVaeModel = action.payload;
  },
  newmodelEncoderModelSelected: (state, action: PayloadAction<ParameterModel | null>) => {
    state.newmodelEncoderModel = action.payload;
  },
}
```

### 7.2 Create Selectors

**File:** `invokeai/frontend/web/src/features/controlLayers/store/paramsSlice.ts`

```typescript
// Selectors:
export const selectNewmodelScheduler = createSelector(
  selectParamsSlice,
  (params) => params.newmodelScheduler
);

export const selectNewmodelVaeModel = createSelector(
  selectParamsSlice,
  (params) => params.newmodelVaeModel
);

export const selectNewmodelEncoderModel = createSelector(
  selectParamsSlice,
  (params) => params.newmodelEncoderModel
);
```

### Frontend State Management Checklist

- [ ] Extend state interface for model-specific parameters
- [ ] Define initial state
- [ ] Create reducer actions
- [ ] Create selectors
- [ ] Export actions

---

## 8. Frontend: Parameter Recall

### 8.1 Metadata Parsing

**File:** `invokeai/frontend/web/src/features/metadata/parsing.tsx`

```typescript
// Add parameter recall handlers:
const recallNewmodelScheduler = (metadata: CoreMetadata) => {
  if (metadata.scheduler) {
    dispatch(setNewmodelScheduler(metadata.scheduler));
  }
};

const recallNewmodelVaeModel = async (metadata: CoreMetadata) => {
  if (metadata.vae) {
    const vaeModel = await fetchModelConfig(metadata.vae);
    dispatch(newmodelVaeModelSelected(vaeModel));
  }
};

const recallNewmodelEncoderModel = async (metadata: CoreMetadata) => {
  if (metadata.encoder_model) {
    const encoderModel = await fetchModelConfig(metadata.encoder_model);
    dispatch(newmodelEncoderModelSelected(encoderModel));
  }
};
```

### Frontend Parameter Recall Checklist

- [ ] Recall handlers for each model-specific parameter
- [ ] Model config fetching for submodels
- [ ] Dispatch actions for state updates

---

## 9. Metadata and Generation Modes

### 9.1 Add Generation Modes

**File:** `invokeai/app/invocations/metadata.py`

```python
GENERATION_MODES = Literal[
    # Existing modes
    "txt2img", "img2img", "inpaint", "outpaint",
    "sdxl_txt2img", "sdxl_img2img", "sdxl_inpaint", "sdxl_outpaint",
    "flux_txt2img", "flux_img2img", "flux_inpaint", "flux_outpaint",
    "flux2_txt2img", "flux2_img2img", "flux2_inpaint", "flux2_outpaint",
    "sd3_txt2img", "sd3_img2img", "sd3_inpaint", "sd3_outpaint",
    # NEW:
    "newmodel_txt2img",
    "newmodel_img2img",
    "newmodel_inpaint",
    "newmodel_outpaint",
]
```

### 9.2 Extend CoreMetadata (if needed)

**File:** `invokeai/app/invocations/metadata.py`

```python
@invocation_output("core_metadata_output")
class CoreMetadataOutput(BaseInvocationOutput):
    # Existing fields
    model: ModelIdentifierField | None = None
    steps: int | None = None
    cfg_scale: float | None = None

    # NEW: Model-specific metadata fields
    newmodel_encoder: ModelIdentifierField | None = None
    newmodel_custom_param: float | None = None
```

### Metadata and Generation Modes Checklist

- [ ] Add generation modes to `GENERATION_MODES`
- [ ] Extend CoreMetadata if model-specific fields needed
- [ ] Set metadata in graph builder (`g.upsertMetadata({...})`)

---

## 10. Starter Models

### 10.1 Define Starter Models

**File:** `invokeai/backend/model_manager/starter_models.py`

```python
# Main Model
newmodel_main = StarterModel(
    name="NewModel Main",
    base=BaseModelType.NewModel,
    source="organization/newmodel-main",  # HuggingFace repo
    description="NewModel main transformer. ~10GB",
    type=ModelType.Main,
)

# VAE (if separate)
newmodel_vae = StarterModel(
    name="NewModel VAE",
    base=BaseModelType.NewModel,
    source="organization/newmodel::vae",  # Submodel syntax
    description="NewModel VAE. ~500MB",
    type=ModelType.VAE,
)

# Text Encoder (if separate)
newmodel_encoder = StarterModel(
    name="NewModel Encoder",
    base=BaseModelType.Any,
    source="organization/newmodel::text_encoder+tokenizer",
    description="NewModel text encoder. ~5GB",
    type=ModelType.TextEncoder,
)

# Quantized variants
newmodel_fp8 = StarterModel(
    name="NewModel (FP8)",
    base=BaseModelType.NewModel,
    source="https://huggingface.co/org/newmodel-fp8/resolve/main/model.safetensors",
    description="FP8 quantized version. ~5GB",
    type=ModelType.Main,
    dependencies=[newmodel_vae, newmodel_encoder],  # Dependencies!
)

# Add to STARTER_MODELS list:
STARTER_MODELS: list[StarterModel] = [
    # ... existing models
    newmodel_main,
    newmodel_vae,
    newmodel_encoder,
    newmodel_fp8,
]
```

### Starter Models Checklist

- [ ] Define main model StarterModel
- [ ] Define VAE StarterModel if separate
- [ ] Define text encoder StarterModel if separate
- [ ] Define quantized variants (FP8, GGUF, etc.)
- [ ] Set dependencies correctly
- [ ] Add to `STARTER_MODELS` list

---

## 11. Optional Features

### 11.1 ControlNet Support

**Backend Config:**
**File:** `invokeai/backend/model_manager/configs/controlnet.py`

```python
@ModelConfigFactory.register
class ControlNet_Checkpoint_NewModel_Config(ControlNet_Checkpoint_Base):
    base: Literal[BaseModelType.NewModel] = BaseModelType.NewModel

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, ...) -> Self:
        if not _is_newmodel_controlnet(mod.state_dict):
            raise NotAMatchError()
        return cls(...)
```

**Backend Invocation:**
**File:** `invokeai/app/invocations/[newmodel]_controlnet.py`

```python
@invocation("newmodel_controlnet", ...)
class NewModelControlNetInvocation(BaseInvocation):
    image: ImageField = InputField()
    controlnet_model: ControlNetField = InputField()
    control_weight: float = InputField(default=1.0)

    def invoke(self, context) -> ControlNetOutput:
        # Compute ControlNet conditioning
        pass
```

**Frontend Graph:**
```typescript
// In buildNewModelGraph.ts:
const { controlNets } = await addControlNets({ g, manager, denoise });
```

### 11.2 IP-Adapter / Reference Images

**Backend Invocation:**
**File:** `invokeai/app/invocations/[newmodel]_ip_adapter.py`

```python
@invocation("newmodel_ip_adapter", ...)
class NewModelIPAdapterInvocation(BaseInvocation):
    image: ImageField = InputField()
    ip_adapter_model: IPAdapterField = InputField()
    weight: float = InputField(default=1.0)
```

**Frontend Graph:**
```typescript
// In buildNewModelGraph.ts:
const { ipAdapters } = await addIPAdapters({ g, manager, denoise });
```

### 11.3 LoRA Support

**Backend Config:**
**File:** `invokeai/backend/model_manager/configs/lora.py`

```python
@ModelConfigFactory.register
class LoRA_LyCORIS_NewModel_Config(LoRA_LyCORIS_Base):
    base: Literal[BaseModelType.NewModel] = BaseModelType.NewModel
```

**Backend Model Loader Integration:**
```python
# In newmodel_model_loader.py:
class NewModelModelLoaderOutput(BaseInvocationOutput):
    transformer: TransformerField  # TransformerField already contains loras: list[LoRAField]
```

**Frontend Graph:**
```typescript
// In buildNewModelGraph.ts:
const { loras } = await addLoRAs({ g, manager, denoise, modelLoader });
```

### 11.4 Scheduler UI

**Frontend Component:**
**File:** `invokeai/frontend/web/src/features/parameters/components/NewModelScheduler.tsx`

```typescript
export const NewModelSchedulerSelect = () => {
  const dispatch = useAppDispatch();
  const scheduler = useAppSelector(selectNewmodelScheduler);

  return (
    <Select
      value={scheduler}
      onChange={(value) => dispatch(setNewmodelScheduler(value))}
      options={[
        { value: 'euler', label: 'Euler' },
        { value: 'heun', label: 'Heun' },
        { value: 'lcm', label: 'LCM' },
      ]}
    />
  );
};
```

### Optional Features Checklist

**ControlNet:**
- [ ] Create ControlNet config (`configs/controlnet.py`)
- [ ] Create ControlNet invocation
- [ ] Frontend graph integration (`addControlNets`)

**IP-Adapter:**
- [ ] Create IP-Adapter invocation
- [ ] Create image encoder config if needed
- [ ] Frontend graph integration (`addIPAdapters`)

**LoRA:**
- [ ] Create LoRA config (`configs/lora.py`)
- [ ] LoRA loading in model loader
- [ ] Frontend graph integration (`addLoRAs`)

**Scheduler:**
- [ ] Define scheduler constants
- [ ] Frontend UI component
- [ ] State management

---

## Summary: Minimal Integration

For a **minimal txt2img integration**, the following files are required:

### Backend (Python):
1. `invokeai/backend/model_manager/taxonomy.py` - BaseModelType, Variant
2. `invokeai/backend/model_manager/configs/main.py` - Model configs
3. `invokeai/backend/model_manager/configs/factory.py` - AnyModelConfig
4. `invokeai/backend/model_manager/load/model_loaders/[newmodel].py` - Loader
5. `invokeai/app/invocations/[newmodel]_model_loader.py`
6. `invokeai/app/invocations/[newmodel]_text_encoder.py`
7. `invokeai/app/invocations/[newmodel]_denoise.py`
8. `invokeai/app/invocations/[newmodel]_vae_decode.py`
9. `invokeai/backend/[newmodel]/sampling_utils.py`
10. `invokeai/backend/[newmodel]/denoise.py`
11. `invokeai/app/invocations/metadata.py` - Generation modes

### Frontend (TypeScript):
1. `src/features/nodes/util/graph/generation/buildNewModelGraph.ts`
2. `src/features/nodes/util/graph/types.ts`
3. `src/features/queue/hooks/useEnqueueCanvas.ts`
4. `src/features/controlLayers/store/paramsSlice.ts`

### Additional for img2img/inpaint/outpaint:
1. `invokeai/app/invocations/[newmodel]_vae_encode.py`
2. `src/features/nodes/util/graph/generation/addImageToImage.ts`
3. `src/features/nodes/util/graph/generation/addInpaint.ts`
4. `src/features/nodes/util/graph/generation/addOutpaint.ts`

---

## Reference: Existing Implementations

| Feature | FLUX.1 | FLUX.2 Klein | SD3 | SDXL | Z-Image |
|---------|--------|--------------|-----|------|---------|
| Latent Channels | 16 | 32 | 16 | 4 | 32 |
| Text Encoder | CLIP + T5 | Qwen3 | CLIP×3 + T5 | CLIP×2 | Qwen3 |
| VAE | 16ch | 32ch+BN | 16ch | 4ch | 32ch |
| CFG | Optional | Optional | Yes | Yes | Optional |
| Guidance Embed | Dev only | No | No | No | No |
| Pack/Unpack | Yes | Yes | No | No | Yes |

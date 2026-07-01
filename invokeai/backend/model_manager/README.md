# Model Management System

This document describes Invoke's model management system and common tasks for extending model support.

## Overview

The model management system handles the full lifecycle of models: identification, loading, and running. The system is extensible and supports multiple model architectures, formats, and quantization schemes.

### Three Major Subsystems

1. **Model Identification** (`configs/`): Determines model type, architecture, format, and metadata when users install models.
2. **Model Loading** (`load/`): Loads models from disk into memory for inference.
3. **Model Running**: Executes inference on loaded models. Implementation is scattered across the codebase, typically in architecture-specific inference code adjacent to `model_manager/`. The inference code is run in nodes in the graph execution system.

## Core Concepts

### Model Taxonomy

The `taxonomy.py` module defines the type system for models:

- `ModelType`: The kind of model (e.g., `Main`, `LoRA`, `ControlNet`, `VAE`).
- `ModelFormat`: Storage format - may imply a quantization or some other quality (e.g., `Diffusers`, `Checkpoint`, `LyCORIS`, `BnbQuantizednf4b`).
- `BaseModelType`: Associated pipeline architecture (e.g., `StableDiffusion1`, `StableDiffusionXL`, `Flux`). Models without an associated base use `Any` (e.g., `CLIPVision` is its own thing).
- `ModelVariantType`, `FluxVariantType`, `ClipVariantType`: Architecture-specific variants.

These enums form a discriminated union that uniquely identifies each model configuration class.

### Model "Configs"

Model configs are Pydantic models that describe a model on disk. They include the model taxonomy, path, and any metadata needed for loading or running the model.

Model configs are stored in the database.

### Model Identification

When a user installs a model, the system attempts to identify it by trying each registered config class until one matches.

**Config Classes** (`configs/`):

- All config classes inherit from `Config_Base`, either directly or indirectly via some intermediary class (e.g., `Diffusers_Config_Base`, `Checkpoint_Config_Base`, or something narrower).
- Each config class represents a specific, unique combination of `type`, `format`, `base`, and optional `variant`.
- Config classes must implement `from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict) -> Self`. This method inspects the model on disk and raises `NotAMatchError` if the model doesn't match the config class, or returns an instance of the config class if it does.
  - `ModelOnDisk` is a helper class that abstracts the model weights. It should be the entrypoint for inspecting the model (e.g., loading state dicts).
- Override fields allow users to provide hints (e.g., when differentiating between SD1/SD2/SDXL VAEs with identical structures).

**Identification Process**:

1. `ModelConfigFactory.from_model_on_disk()` is called with a path to the model.
2. The factory iterates through all registered config classes, calling `from_model_on_disk()` on each.
3. Each config class inspects the model (state dict keys, tensor shapes, config files, etc.).
4. If a match is found, the config instance is returned. If multiple matches are found, they are prioritized (e.g., main models over LoRAs).
5. If no match is found, an `Unknown_Config` is returned as a fallback.

**Utilities** (`identification_utils.py`):

- `NotAMatchError`: Exception raised when a model doesn't match a config class.
- `get_config_dict_or_raise()`: Load JSON config files from diffusers/transformers models.
- `raise_for_class_name()`: Validate class names in config files.
- `raise_for_override_fields()`: Validate user-provided override fields against the config schema.
- `state_dict_has_any_keys_*()`: Helpers for inspecting state dict keys.

### Model Loading

Model loaders handle instantiating models from disk into memory.

**Loader Classes** (`load/model_loaders/`):

- Loaders register themselves with a decorator `@ModelLoaderRegistry.register(base=..., type=..., format=...)`. The `type`, `format` and `base` indicate which configs classes the loader can handle.
- Each loader implements `_load_model(self, config: AnyModelConfig, submodel_type: Optional[SubModelType]) -> AnyModel`.
- Loaders are responsible for:
  - Loading model weights from the config's path.
  - Instantiating the correct model class (often using diffusers, transformers, or custom implementations).
  - Returning the in-memory model representation.

**Model Cache** (`load/model_cache/`):

> This system typically does not require changes to support new model types, but it is important to understand how it works.

- Manages models in memory with RAM and VRAM limits.
- Handles moving models between CPU (storage device) and GPU (execution device).
- Implements LRU eviction for RAM and smallest-first offload for VRAM.
- Supports partial loading for large models on CUDA.
- Thread-safe with locks on all public methods.

**Loading Process**:

1. The appropriate loader is selected based on the model config's `base`, `type`, and `format` attributes.
2. The loader's `_load_model()` method is called with the model config.
3. The loaded model is added to the model cache via `ModelCache.put()`.
4. When needed, the model is moved into VRAM via `ModelCache.get()` and `ModelCache.lock()`.

### Model Running

Model running is architecture-specific and typically implemented in folders adjacent to `model_manager/`.

Inference code doesn't necessarily follow any specific pattern, and doesn't interact directly with the model management system except to receive model configs and loaded models.

At a high level, when a node needs to run a model, it will:

- Receive a model identifier as an input or constant. This is typically the model's database ID (aka the `key`).
- The node will use the `InvocationContext` API to load the model. The request is dispatched to the model manager which will load the model and return the a model loader with a context manager that yields the in-memory model, mediating VRAM/RAM management as needed.
- The node will run inference using the loaded model using whatever patterns or libraries it needs.

## Common Tasks

### Task 1: Improving Identification for a Supported Model Type

When identification fails or produces incorrect results for a model that should be supported, you may need to refine the identification logic.

**Steps**:

1. Obtain the failing model file or directory.
2. Create a test case for it, following the instructions in `tests/model_identification/README.md`.
3. Review the relevant config class in `configs/` (e.g., `configs/lora.py` for LoRA models).
4. Examine the `from_model_on_disk()` method for some existing models to understand the patterns for identification logic.
5. Inspect the failing model's files and structure:
   - For checkpoint files: Load the state dict and examine keys and tensor shapes.
   - For diffusers models: Examine the config files and directory structure.
6. Update the identification logic to handle the new model variant. Common approaches:
   - Check for specific state dict keys or key patterns.
   - Inspect tensor shapes (e.g., `state_dict[key].shape`).
   - Parse config files for class names or configuration values.
   - Use helper functions from `identification_utils.py`.
7. Run the test suite to verify the new logic works and doesn't break existing tests: `pytest tests/model_identification/test_identification.py`.
   - Make sure you have installed the test dependencies (e.g. `uv pip install -e ".[dev,test]"`).
   - If the model type is complex or has multiple variants, consider adding more test cases to cover edge cases.
8. If, after successfully adding identification support for the model, it still doesn't work, you may need to update loading and/or inference code as well.

**Key Files**:

- Config class: `configs/<model_type>.py`
- Identification utilities: `configs/identification_utils.py`
- Taxonomy: `taxonomy.py`
- Test README: `tests/model_identification/README.md`

### Task 2: Adding Support for a New Model Type

Adding a new model type requires implementing identification and loading logic. Inference and new nodes ("invocations") may be required if the model type doesn't fit into existing architectures or nodes.

**Steps**:

#### 1. Define Taxonomy

- Add a new `ModelType` enum value in `taxonomy.py` if needed.
- Determine the appropriate `BaseModelType` (or use `Any` if not architecture-specific).
- Add a new `ModelFormat` if the model uses a unique storage format.

You may need to add other attributes, depending on the model.

#### 2. Implement Config Class

- Create a new config file in `configs/` (e.g., `configs/new_model.py`).
- Define a config class inheriting from `Config_Base` and appropriate format base class:
  - `Diffusers_Config_Base` for diffusers-style models.
  - `Checkpoint_Config_Base` for single-file checkpoint models.
- Define `type`, `format`, and `base` as `Literal` fields with defaults. Remember, these must uniquely identify the config class.
- Implement `from_model_on_disk()`:
  - Validate the model is the correct format (file vs directory).
  - Inspect state dict keys, tensor shapes, or config files.
  - Raise `NotAMatchError` if the model doesn't match.
  - Extract any additional metadata needed (e.g., variant, prediction type).
  - Return an instance of the config class.
- Register the config in `configs/factory.py`:
  - Add the config class to the `AnyModelConfig` union.
  - Add an `Annotated[YourConfig, YourConfig.get_tag()]` entry.

#### 3. Implement Loader Class

- Create a new loader file in `load/model_loaders/` (e.g., `load/model_loaders/new_model.py`).
- Define a loader class inheriting from `ModelLoader`.
- Decorate with `@ModelLoaderRegistry.register(base=..., type=..., format=...)`.
- Implement `_load_model()`:
  - Load model weights from `config.path`.
  - Instantiate the model using the appropriate library (diffusers, transformers, or custom).
  - Handle `submodel_type` if the model has submodels (e.g., text encoders, VAE).
  - Return the in-memory model representation.

#### 4. Add Tests

Follow the instructions in `tests/model_identification/README.md`.

#### 5. Implement Inference and Nodes (if needed)

- If the model type requires new inference logic, implement it in an appropriate location.
- Create nodes for the model if it doesn't fit into existing nodes. Search for subclasses of `BaseInvocation` for many examples.

### 6. Frontend Support

#### Workflows tab

Typically, you will not need to do anything for the model to work in the Workflow Editor. When you define the node's model field, you can provide constraints for what type of models are selectable. The UI will automatically filter the list of models based on the model taxonomy.

For example, this field definition in a node will allow users to select only "main" (pipeline) Stable Diffusion 1.x or 2.x models:

```py
model: ModelIdentifierField = InputField(
    ui_model_base=[BaseModelType.StableDiffusion1, BaseModelType.StableDiffusion2],
    ui_model_type=ModelType.Main,
)
```

This same pattern works for any combination of `type`, `base`, `format`, and `variant`.

#### Canvas / Generate tabs

The Canvas and Generate tabs use graphs internally, but they don't expose the full graph editor UI. Instead, they provide a simplified interface for common tasks.

They use "graph builder" functions, which take the user's selected settings and build a graph behind the scenes. We have one graph builder for each model architecture.

Updating or adding a graph builder can be a bit complex, and you'd likely need to update other UI components and state management to support the new model type.

The SDXL graph builder is a good example: `invokeai/frontend/web/src/features/nodes/util/graph/generation/buildSDXLGraph.ts`

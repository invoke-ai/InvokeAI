# Model Probe (Identification) Testing

Invoke's model Identification system is tested against example model files. Test cases are lightweight representations of real models which have been "stripped" of their tensor data.

## Setup

Test cases are stored with git lfs. You _must_ [install git lfs](https://git-lfs.com/) to pull down the test cases and add to them.

```bash
# Only need to do this once
git lfs install
# Pull the actual model files down - if you just do `git pull` you'll only get pointers
git lfs pull
```

## Running the Tests

To run the tests use:

```bash
pytest -v tests/test_model_probe/test_identification.py
```

## Stripped Model Files

Invoke abstracts the loading of a model's state dict and metadata in a class called [`ModelOnDisk`](../../invokeai/backend/model_manager/model_on_disk.py). This class loads real model weights. We use it to inspect models and identify them.

For testing purposes, we create a stripped-down version of model weights that contain only the model structure and metadata for each key, without the actual tensor data. The state dict structure is typically all we need to identify models; the tensors themselves are not needed. This allows us to store test cases in the repo without adding many gigabytes of data.

To see how this works, check out [`StrippedModelOnDisk`](./stripped_model_on_disk.py). This class includes logic to strip models and to load these stripped models for testing.

### Some Models Cannot Be Stripped

Certain models cannot be stripped because identification relies on inspecting the actual tensor data. We have to store the full model files for these test cases.

> Currently, the only models that cannot be stripped are [`spandrel`](https://github.com/chaiNNer-org/spandrel/) image-to-image models. `spandrel` supports _many_ model architectures but doesn't provide a way to identify or assert support for a model by its state dict structure alone.
>
> To positively identify these models, we must attempt to load the model using spandrel. If it loads successfully, we assume it is a supported model. Therefore, we cannot strip these models and must store the full model files in the test cases. We only store one such model to keep the test suite size manageable.
>
> `StrippedModelOnDisk` will simply pass-through the "live" tensor data for these models when loading them to test.

## Adding New Test Cases

Run the [`strip_model.py`](./strip_model.py) script to create a new test case. For example:

```bash
python strip_model.py /path/to/your/model --output_dir ./stripped_models
```

It supports single-file models and multi-file models (e.g. diffusers-style models). The output will be a directory named with a UUID, containing the stripped model files and a dummy `__test_metadata__.json` file.

Example output structure for a single-file model:

```
stripped_models/
└── 19fd1a40-c5b7-4734-bd3a-6e0e948cce0b/
    ├── __test_metadata__.json
    └── Standard Reference (XLabs FLUX IP-Adapter v2).safetensors
```

This test metadata file should contain a single JSON dict and must be filled out manually with the expected identification results.

### Structure of `__test_metadata__.json`

This file contains a single JSON dict. Here's an example for a FLUX IP Adapter checkpoint:

```json
{
  "source": "https://huggingface.co/XLabs-AI/flux-ip-adapter-v2/resolve/main/ip_adapter.safetensors",
  "file_name": "Standard Reference (XLabs FLUX IP-Adapter v2).safetensors",
  "expected_config_attrs": {
    "type": "ip_adapter",
    "format": "checkpoint",
    "base": "flux"
  }
}
```

See the details below for each field.

#### `"source"`

A string indicating the source of the model (e.g. a Hugging Face repo ID or URL). This is not used for identification, but is useful for reference so we know where the model came from. Nothing will break if this field is missing or incorrect, but it is good practice to fill it out.

- Example HF Repo ID: `"RunDiffusion/Juggernaut-XL-v9"`
- Example URL: `"https://huggingface.co/XpucT/Deliberate/resolve/main/Deliberate_v5.safetensors"`

#### `"file_name"`

If the model is a single file (e.g. a `.safetensors` file), this is the name of that file. The test suite will look for this file in the test case directory.

If the model is multi-file (e.g. diffusers-style), omit this key or set it to a falsey value like `null` or an empty string.

- Example: `"model.safetensors"`

> The `strip_model.py` script will automatically fill this field in for single-file models.

#### `"expected_config_attrs"`

This field is a dict of expected configuration attributes for the model. It is required for all test cases.

It is used to verify that the model's configuration matches expectations. The keys and values in this dict depend on the specific model and its configuration.

These attributes must be included, as they are the primary discriminators for models:

- `"type"`: The type of the model. This is the value of the `ModelType` enum.
- `"format"`: The format of the model files. This is the value of the `ModelFormat` enum.
- `"base"`: The base model pipeline architecture associated with this model. Many models do not have an associated base. For these, use `"any"`. This is the value of the `BaseModelType` enum.

Depending on the kind of model, these additional keys may be useful:

- `"prediction_type"`: The prediction type used by the model. This is the value of the `SchedulerPredictionType` enum.
- `"variant"`: The variant of the model, if applicable. This is the value of the `ModelVariantType` enum.

To see all possible values for these enums, check out their definitions in [`invokeai/backend/model_manager/taxonomy.py`](../../invokeai/backend/model_manager/taxonomy.py).

For example, for a SD1.5 main (pipeline) inpainting model in diffusers format, you might have:

```json
{
  "expected_config_attrs": {
    "type": "main",
    "format": "diffusers",
    "base": "sd-1",
    "prediction_type": "epsilon",
    "variant": "inpaint"
  }
}
```

#### `"notes"`

This is an optional string field where you can add any notes or comments about the test case. It can be useful for providing context or explaining any special considerations.

#### `"override_fields"`

In some rare cases, we may need to provide additional hints to the identification system to help it identify the model correctly.

Currently, the only known case where we need extra information is to differentiate between single-file SD1.x, SD2.x and SDXL VAEs. These models have identical structures, so we need to provide a hint. Though it is far from ideal, we use simple string matching on the model's name to provide this hint.

For example, when users install the `taesdxl` VAE from the HF repo `madebyollin/taesdxl`, the identification system will get the model name `taesdxl`. It sees "xl" in the name and infers that this is a SDXL VAE. To reproduce this in a test case, we add the following to `__test_metadata__.json`:

```json
{
  "override_fields": {
    "name": "taesdxl"
  }
}
```

# Normalized Model Manager

This is proof-of-principle code that refactors model storage to be
more space efficient. The driving observation is that there is a
significant amount of redundancy in Stable Diffusion models. For
example, the VAE and text encoders are frequently the same across
multiple models derived from the same base models.

Components:

1. Blob folder located in invokeai/models/blob

   This folder contains a series of subfolders with randomly-assigned
   UUIDs. Each subfolder contains the contents of a diffusers folder, e.g.
   the .json and .safetensors files.
   
2. Database located in invokeai/databases/models.db

   This is a database that describes what each model is, and maps
   diffusion pipelines (main models) to the models that compose it.

## Database tables

This illustrates the database schema.

### MODEL_PATH

This provides the type and path of each fundamental model. The type
can be any of the ModelType enums, including clip_vision, etc.

|  **ID**     | **TYPE**      | **REFCOUNT** | **PATH** |
|-------------|---------------|--------------|----------|
| 1           | `vae`         |    3         | /opt/invokeai/models/blob/abc-def-012 |
| 2           | `text_encoder`|    2         | /opt/invokeai/models/blob/482-abc-321 |
| 3           | `unet`        |    1         | /opt/invokeai/models/blob/839-dea-444 |
| 4           | `safety_checker`|  3         | /opt/invokeai/models/blob/982-472-a9e|
| 5           | `lora`        |    1         | /opt/invokeai/models/blob/111-222-333|

Refcount indicates how many pipelines the fundamental is being shared with.

### MODEL_NAME

Holds name and description of the model. Note that an anonymous model
that is a component of a pipeline does not need to have any metadata.

|  **ID**     | **NAME**   | **SOURCE**  | **DESCRIPTION** |
|-------------|------------|-------------|-----------------|
| 5           | LoWRA      | stabilityai/lowra | LoWRA adapted for low light renderings |


### MODEL_BASE

This maps the model UUID to the base model types supported. Some
fundamental models, such as unets, only support a single base. Others,
such as sd-1 VAEs, support more than one, and others, such as the
beloved safety checker, support all model bases (type "any"). So
this table supports one-to-many relationships.

|  **ID**   | **BASE**    |
|-----------|-------------|
| 1         | sd-1        |
| 1         | sd-2        |
| 2         | sd-1        |
| 3         | sd-1        |
| 4         | any         |
| 5         | sd-1        |

### PIPELINE

This is a table of pipeline models. The `toc` field holds the
`index.json` file contained at the top level of a diffusers folder. It
is there only for the purpose of exporting a working diffusers
pipeline folder.


| **ID**  | **NAME**             | **BASE** | **TOC**|
|---------|----------------------|----------|----------|
| 1       | stable-diffusion-1-5 | sd-1     |/opt/invokeai/models/blob/439-aaf-232.json |
| 2       | stable-diffusion-2-1 | sd-2     |/opt/invokeai/models/blob/868-212-11f.json |

### PIPELINE_PARTS

This table describes how to put the fundamental models together in
order to reconstruct the original pipeline.

| **PIPELINE_ID**  | **PART_ID** | **PART_FOLDER** |
|------------------|-------------|-------------------|
| 1               |   1         | `vae`             |
| 1               |   2         | `text_encoder`    |
| 1               |   3         | `unet`            |

## Initializing the normalized model manager

Initialization will look something like this:

```
from invokeai.backend.normalized_mm import normalized_model_manager
from invokeai.app.services.config import InvokeAIAppConfig

config = InvokeAIAppConfig.get_config()
config.parse_args()
nmm = normalized_model_manager(config)
```

## Saving a model to the database

Pass the path to a diffusers model or safetensors file. "main"
safetensors will be converted to diffusers behind the scenes.

```
id = nmm.import('/path/to/folder')
id = nmm.import('/path/to/file.safetensors')
```

## Fetching a model

To fetch a fundamental model, use its name and type:

```
model_info = nmm.get_model_by_name(name='LoWRA', type='lora')
print(model_info.path)
print(model_info.description)
```

To fetch part of a pipeline, use its name, base and the submodel
desired:

```
model_info = nmm.get_pipeline_by_name(name='stable-diffusion-1-5', base='sd-1', submodel='vae')
print(model_info.path)
```

## Exporting a model

To export a model back into its native format (diffusers for main, safetensors for other types), use `export`:

```
nmm.export(name='stable-diffusion-1-5', base='sd-1', destination='/path/to/export/folder')
```

The model will be exported to the indicated folder with the name `stable-diffusion-1-5`.


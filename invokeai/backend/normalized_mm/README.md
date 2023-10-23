# Normalized Model Manager

This is proof-of-principle code that refactors model storage to be
more space efficient and less dependent on the particulars of Stable
Diffusion models. The driving observation is that there is a
significant amount of redundancy in Stable Diffusion models. For
example, the VAE, tokenizer and safety checker are frequently the same
across multiple models derived from the same base models.

The way the normalized model manager works is that when a main
(pipeline) model is ingested, each of its submodels ("vae", "unet" and
so forth) is scanned and hashed using a fast sampling and hashing
algorithm. If the submodel has a hash that hasn't been seen before, it
is copied into a folder within INVOKEAI_ROOT, and we create a new
database entry with the submodel's path and a reference count of "1".
If the submodel has a hash that has previously been seen, then we
update the database to bump up the submodel's reference count.

Checkpoint files (.bin, .ckpt and .safetensors) are converted into
diffusers format prior to ingestion. The system directly imports
simple models, such as LoRAs and standalone VAEs, and normalizes them
if previously seen. This has benefits when a user tries to ingest the
same VAE twice under different names.

Additional database tables map the relationship between main models
and their submodels, and to record which base model(s) a submodel is
compatible with.

## Installation and Testing

To test, checkout the PR and run `pip install -e .`. This will create
a command called `invokeai-nmm` (for "normalized model
manager"). To ingest a single model:


```
invokeai-nmm ingest my_model.safetensors
```

To ingest a whole directory of models:

```
invokeai-nmm ingest my_models/*
```

These commands will create a sqlite3 database of model data in
`INVOKEAI_ROOT/databases/normalized_models.db`, copy the model data
into a blobs directory under `INVOKEAI_ROOT/model_blobs`, and create
appropriate entries in the database. You can then use the API to
retrieve information on pipelines and submodels.

The `invokeai-nmm` tool has a number of other features, including
listing models and examining pipeline subparts. In addition, it has an
`export` command which will reconstitute a diffusers pipeline by
creating a directory containing symbolic links into the blogs
directory.

Use `invokeai-nmm --help` to get a summary of commands and their
flags.

## Benchmarking

To test the performance of the normalied model system, I ingested a
InvokeAI models directory of 117 different models (35 main models, 52
LoRAs, 9 controlnets, 8 embeddings and miscellaneous others). The
ingestion, which included the conversion of multiple checkpoint to
diffusers models, took about 2 minutes. Prior to ingestion, the
directory took up 189.5 GB. After ingestion, it was reduced to 160 GB,
an overall 16% reduction in size and a savings of 29 GB.

I was a surprised at the relatively modest space savings and checked that
submodels were indeed being shared. They were:

```
sqlite> select part_id,type,refcount from simple_model order by refcount desc,type;
┌─────────┬───────────────────┬──────────┐
│ part_id │       type        │ refcount │
├─────────┼───────────────────┼──────────┤
│ 28      │ tokenizer         │ 9        │
│ 67      │ feature_extractor │ 7        │
│ 33      │ feature_extractor │ 5        │
│ 38      │ tokenizer         │ 5        │
│ 26      │ safety_checker    │ 4        │
│ 32      │ safety_checker    │ 4        │
│ 37      │ scheduler         │ 4        │
│ 29      │ vae               │ 3        │
│ 30      │ feature_extractor │ 2        │
│ 72      │ safety_checker    │ 2        │
│ 54      │ scheduler         │ 2        │
│ 100     │ scheduler         │ 2        │
│ 71      │ text_encoder      │ 2        │
│ 90      │ text_encoder      │ 2        │
│ 99      │ text_encoder_2    │ 2        │
│ 98      │ tokenizer_2       │ 2        │
│ 44      │ vae               │ 2        │
│ 73      │ vae               │ 2        │
│ 91      │ vae               │ 2        │
│ 97      │ vae               │ 2        │
│ 1       │ clip_vision       │ 1        │
│ 2       │ clip_vision       │ 1        │
...
```

As expected, submodels that don't change from model to model, such as
the tokenizer and safety checker, are frequently shared across main
models. So were the VAEs, but less frequently than I expected. On
further inspection, the spread of VAEs was explained by the following
formatting differences:

1. Whether the VAE weights are .bin or .safetensors
2. Whether it is an fp16 or fp32 VAE
3. Actual differences in the VAE's training

Ironically, checkpoint models downloaded from Civitai are more likely
to share submodels than diffusers pipelines directly downloaded from
HuggingFace. This is because the checkpoints pass through a uniform
conversion process, while diffusers downloaded directly from
HuggingFace are more likely to have format-related differences.

## Database tables

This illustrates the database schema.

### SIMPLE_MODEL

This provides the type and path of each fundamental model. The type
can be any of the ModelType enums, including clip_vision, etc.

```
┌─────────┬───────────────────┬──────────────────────────────────┬──────────┬──────────────────────────────────────────────────────────────────────────────────────────┐
│ part_id │       type        │               hash               │ refcount │                                           path                                           │
├─────────┼───────────────────┼──────────────────────────────────┼──────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
│ 26      │ safety_checker    │ 76b420d8f641411021ec1dadca767cf7 │ 4        │ /opt/model_blobs/safety_checker-7214b322-1069-4753-a4d5-fe9e18915ca7    │
│ 28      │ tokenizer         │ 44e42c7bf25b5e32e8d7de0b822cf012 │ 9        │ /opt/model_blobs/tokenizer-caeb7f7f-e3db-4d67-8f60-1a4831e1aef2         │
│ 29      │ vae               │ c9aa45f52c5d4e15a22677f34436d373 │ 3        │ /opt/model_blobs/vae-7e7d96ee-074f-45dc-8c43-c9902b0d0671               │
│ 30      │ feature_extractor │ 3240f79383fdf6ea7f24bbd5569cb106 │ 2        │ /opt/model_blobs/feature_extractor-a5bb8ceb-2c15-4b7f-bd43-964396440f6c │
│ 32      │ safety_checker    │ 2e2f7732cff3349350bc99f3e7ab3998 │ 4        │ /opt/model_blobs/safety_checker-ef70c446-e3a1-445c-b216-d7c4acfdbcda    │
└─────────┴───────────────────┴──────────────────────────────────┴──────────┴──────────────────────────────────────────────────────────────────────────────────────────┘
```

The Refcount indicates how many pipelines the fundamental is being
shared with. The path is where the submodel is stored, and uses a
randomly-assigned file/directory name to avoid collisions.

The `type` field is a SQLITE3 ENUM that maps to the values of the
`ModelType` enum.

### MODEL_NAME

The MODEL_NAME table stores the name and other metadata of a top-level
model.  The same table is used for both simple models (one part only)
and pipeline models (multiple parts).

Note that in the current implementation, the model name is forced to
be unique and is currently used as the identifier for retrieving
models from the database. This is a simplifying implementation detail;
in a real system the name would be supplemented with some sort of
anonymous key.

Only top-level models are entered into the MODEL_NAME table. The
models contained in subfolders of a pipeline become unnamed anonymous
parts stored in SIMPLE_MODEL and associated with the named model(s)
that use them in the MODEL_PARTS table described next.

An interesting piece of behavior is that the same simple model can be
both anonymous and named. Consider a VAE that is first imported from
the 'vae' folder of a main model. Because it is part of a larger
pipeline, there will be an entry for the VAE in SIMPLE_MODEL with a
refcount of 1, but not in the MODEL_NAME table. However let's say
that, at a later date, the user ingests the same model as a named
standalone VAE. The system will detect that this is the same model,
and will create a named entry to the VAE in MODEL_NAME that identifies
the VAE as its sole part. In SIMPLE_MODEL, the VAE's refcount will be
bumped up to 2. Thus, the same simple model can be retrieved in two
ways: by requesting the "vae" submodel of the named pipeline, or by
requesting it via its standalone name.

The MODEL_NAME table has fields for the model's name, its source, and
description. The `is_pipeline` field is True if the named model is a
pipeline that contains subparts. In the case of a pipeline, then the
`table_of_contents` field will hold a copy of the contents of
`model_index.json`. This is used for the sole purpose of regenerating
a de-normalized diffusers folder from the database.

```
├──────────┼────────────────────────────────┼────────────────────────────────────────────────────────────┼───────────────────────────────────────────────┼─────────────┼───────────────────┤
│ model_id │              name              │                            source                          │                  description                  │ is_pipeline │ table_of_contents │
├──────────┼────────────────────────────────┼────────────────────────────────────────────────────────────┼───────────────────────────────────────────────┼─────────────┼───────────────────┤
│ 1        │ ip_adapter_sd_image_encoder    │ /opt/models/any/clip_vision/ip_adapter_sd_image_encoder    │ Imported model ip_adapter_sd_image_encoder    │ 0           │                   │
│ 2        │ ip_adapter_sd_image_encoder_01 │ /opt/models/any/clip_vision/ip_adapter_sd_image_encoder_01 │ Imported model ip_adapter_sd_image_encoder_01 │ 0           │                   │
│ 3        │ ip_adapter_sdxl_image_encoder  │ /opt/models/any/clip_vision/ip_adapter_sdxl_image_encoder  │ Imported model ip_adapter_sdxl_image_encoder  │ 0           │                   │
│ 4        │ control_v11e_sd15_ip2p         │ /opt/models/sd-1/controlnet/control_v11e_sd15_ip2p         │ Imported model control_v11e_sd15_ip2p         │ 0           │                   │
│ 5        │ control_v11e_sd15_shuffle      │ /opt/models/sd-1/controlnet/control_v11e_sd15_shuffle      │ Imported model control_v11e_sd15_shuffle      │ 0           │                   │
│ 6        │ control_v11f1e_sd15_tile       │ /opt/models/sd-1/controlnet/control_v11f1e_sd15_tile       │ Imported model control_v11f1e_sd15_tile       │ 0           │                   │
│ 7        │ control_v11f1p_sd15_depth      │ /opt/models/sd-1/controlnet/control_v11f1p_sd15_depth      │ Imported model control_v11f1p_sd15_depth      │ 0           │                   │
│ 8        │ control_v11p_sd15_canny        │ /opt/models/sd-1/controlnet/control_v11p_sd15_canny        │ Imported model control_v11p_sd15_canny        │ 0           │                   │
│ 9        │ control_v11p_sd15_inpaint      │ /opt/models/sd-1/controlnet/control_v11p_sd15_inpaint      │ Imported model control_v11p_sd15_inpaint      │ 0           │                   │
│ 10       │ control_v11p_sd15_lineart      │ /opt/models/sd-1/controlnet/control_v11p_sd15_lineart      │ Imported model control_v11p_sd15_lineart      │ 0           │                   │
└──────────┴────────────────────────────────┴─────────-──────────────────────────────────────────────────┴───────────────────────────────────────────────┴─────────────┴───────────────────┘

```

### MODEL_PARTS

The MODEL_PARTS table maps the `model_id` field from MODEL_NAME to the
`part_id` field of SIMPLE_MODEL, as shown below. The `part_name` field
contains the subfolder name that the part was located in at model
ingestion time.

There is not exactly a one-to-one correspondence between the
MODEL_PARTS `part_name` and the SIMPLE_MODEL `type` fields. For
example, SDXL models have part_names of `text_encoder` and
`text_encoder_2`, both of which point to a simple model of type
`text_encoder`.

For one-part model such as LoRAs, the `part_name` is `root`.

```
┌──────────┬─────────┬───────────────────┐
│ model_id │ part_id │     part_name     │
├──────────┼─────────┼───────────────────┤
│ 6        │ 6       │ root              │
│ 25       │ 25      │ unet              │
│ 25       │ 26      │ safety_checker    │
│ 25       │ 27      │ text_encoder      │
│ 25       │ 28      │ tokenizer         │
│ 25       │ 29      │ vae               │
│ 25       │ 30      │ feature_extractor │
│ 25       │ 31      │ scheduler         │
│ 26       │ 32      │ safety_checker    │
│ 26       │ 33      │ feature_extractor │
│ 26       │ 34      │ unet              │
└──────────┴─────────┴───────────────────┘
```

### MODEL_BASE

The MODEL_BASE table maps simple models to the base models that they
are compatible with. A simple model may be compatible with one base
only (e.g. an SDXL-based `unet`); it may be compatible with multiple
bases (e.g. a VAE that works with either `sd-1` or `sd-2`); or it may
be compatible with all models (e.g. a `clip_vision` model).

This table has two fields, the `part_id` and the `base` it is
compatible with. The base is a SQLITE ENUM that corresponds to the
`BaseModelType` enum.

```
sqlite> select * from model_base limit 8;
┌─────────┬──────────────┐
│ part_id │     base     │
├─────────┼──────────────┤
│ 1       │ sd-1         │
│ 1       │ sd-2         │
│ 1       │ sdxl         │
│ 1       │ sdxl-refiner │
│ 2       │ sd-1         │
│ 2       │ sd-2         │
│ 2       │ sdxl         │
│ 2       │ sdxl-refiner │
└─────────┴──────────────┘
```

At ingestion time, the MODEL_BASE table is populated using the
following algorithm:

1. If the ingested model is a multi-part pipeline, then each of its
   parts is assigned the base determined by probing the pipeline as a
   whole.
   
2. If the ingested model is a single-part simple model, then its part
   is assigned to the base returned by probing the simple model.
   
3. Any models that return `BaseModelType.Any` at probe time will be
   assigned to all four of the base model types as shown in the
   example above.
   
Interestingly, the table will "learn" when the same simple model is
compatible with multiple bases. Consider a sequence of events in which
the user ingests an sd-1 model containing a VAE. The VAE will
initially get a single row in the MODEL_BASE table with base
"sd-1". Next the user ingests an sd-2 model that contains the same
VAE. The system will recognize that the same VAE is being used for a
model with a different base, and will add a new row to the table
indicating that this VAE is compatible with either sd-1 or sd-2.

When retrieving information about a multipart pipeline using the API,
the system will intersect the base compatibility of all the components
of the pipeline until it finds the set of base(s) that all the
subparts are compatible with.

## The API

Initialization will look something like this:

```
from pathlib import Path

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.normalized_mm.normalized_model_manager import NormalizedModelManager

config = InvokeAIAppConfig.get_config()
config.parse_args()

nmm = NormalizedModelManager(config)
```

At the current time, the InvokeAIAppConfig object is used only to
locate the root directory path and the location of the `databases`
subdirectory.

## Ingesting a model

Apply the `ingest()` method to a checkpoint or diffusers folder Path
and an optional model name. If the model name isn't provided, then it
will be derived from the stem of the ingested filename/folder.

```
model_config = nmm.ingest(
           Path('/tmp/models/slick_anime.safetensors'),
		   name="Slick Anime",
		   )
```

Depending on what is being ingested, the call will return either a
`SimpleModelConfig` or a `PipelineConfig` object, which are slightly
different from each other:

```
@dataclass
class SimpleModelConfig:
    """Submodel name, description, type and path."""
    name: str
    description: str
    base_models: Set[BaseModelType]
    type: ExtendedModelType
    path: Path


@dataclass
class PipelineConfig:
    """Pipeline model name, description, type and parts."""
    name: str
    description: str
    base_models: Set[BaseModelType]
    parts: Dict[str, ModelPart]  # part_name -> ModelPart

@dataclass
class ModelPart:
    """Type and path of a pipeline submodel."""
    type: ExtendedModelType
    path: Path
    refcount: int
```

For more control, you can directly call the `ingest_pipeline_model()`
or `ingest_simple_model()` methods, which operate on multi-part
pipelines and single-part models respectively.

Note that the `ExtendedModelType` class is an enum created from the
union of the current model manager's `ModelType` and
`SubModelType`. This was necessary to support the SIMPLE_MODEL table's
`type` field.

## Fetching a model

To fetch a simple model, call `get_model()` with the name of the model
and optionally its part_name. This returns a `SimpleModelConfig` object.

```
model_info = nmm.get_model(name='stable-diffusion-v1-5', part='unet')
print(model_info.path)
print(model_info.description)
print(model_info.base_models)
```

If the model only has one part, leave out the `part` argument, or use
`part=root`:

```
model_info = nmm.get_model(name='detail_slider_v1')
```

To fetch information about a pipeline model, call `get_pipeline()`:

```
model_info = nmm.get_pipeline('stable-diffusion-v1-5')
for part_name, part in model_info.parts.items():
   print(f'{part_name} is located at {part.path}')
```

This returns a `PipelineConfig` object, which you can then interrogate
to get the model's name, description, list of base models it is
compatible with, and its parts. The latter is a dict mapping the
part_name (the original subfolder name) to a `ModelPart` object that
contains the part's type, refcount and path.

## Exporting a model

To export a model back into its native format (diffusers for main,
safetensors for other types), use `export_pipeline`:

```
nmm.export_pipeline(name='stable-diffusion-v1-5', destination='/path/to/export/folder')
```

The model will be exported to the indicated folder as a folder at
`/path/to/export/folder/stable-diffusion-v1-5`. It will contain a copy
of the original `model_index.json` file, and a series of symbolic
links pointing into the model blobs directory for each of the
subfolders.

Despite its name, `export_pipeline()` works as expected with simple
models as well.

## Listing models in the database

There is currently a `list_models()` method that retrieves a list of
all the **named** models in the database. It doesn't currently provide any
way of filtering by name, type or base compatibility, but these are
easy to add in the future.

`list_models()` returns a list of `ModelListing` objects:

```
class ModelListing:
    """Slightly simplified object for generating listings."""
    name: str
    description: str
    source: str
    type: ModelType
    base_models: Set[BaseModelType]
```

An alternative implementation might return a list of
Union[SimpleModelConfig, PipelineConfig], but it seemed cleanest to
return a uniform list.

## Deleting models

Model deletion is not currently fully implemented. When implemented,
deletion of a named model will decrement the refcount of each of its
subparts and then delete parts whose refcount has reached zero. The
appropriate triggers for incrementing and decrementing the refcount
have already been implemented in the database schema.


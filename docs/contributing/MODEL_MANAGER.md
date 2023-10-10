# Introduction to the Model Manager V2

The Model Manager is responsible for organizing the various machine
learning models used by InvokeAI. It consists of a series of
interdependent services that together handle the full lifecycle of a
model. These are the:

* _ModelRecordServiceBase_ Responsible for managing model metadata and
  configuration information. Among other things, the record service
  tracks the type of the model, its provenance, and where it can be
  found on disk.
  
* _ModelLoadServiceBase_ Responsible for loading a model from disk
  into RAM and VRAM and getting it ready for inference/training.
  
* _DownloadQueueServiceBase_ A multithreaded downloader responsible
  for downloading models from a remote source to disk. The download
  queue has special methods for downloading repo_id folders from
  Hugging Face, as well as discriminating among model versions in
  Civitai, but can be used for arbitrary content.
  
* _ModelInstallServiceBase_ A service for installing models to
  disk. It uses `DownloadQueueServiceBase` to download models and
  their metadata, and `ModelRecordServiceBase` to store that
  information. It is also responsible for managing the InvokeAI
  `models` directory and its contents.

## Location of the Code

All four of these services can be found in
`invokeai/app/services` in the following files:

* `invokeai/app/services/model_record_service.py`
* `invokeai/app/services/download_manager.py`  (needs a name change)
* `invokeai/app/services/model_loader_service.py`
* `invokeai/app/services/model_install_service.py`

With the exception of the install service, each of these is a thin
shell around a corresponding implementation located in
`invokeai/backend/model_manager`. The main difference between the
modules found in app services and those in the backend folder is that
the former add support for event reporting and are more tied to the
needs of the InvokeAI API.

Code related to the FastAPI web API can be found in
`invokeai/app/api/routers/models.py`.

***

## What's in a Model? The ModelRecordService

The `ModelRecordService` manages the model's metadata. It supports a
hierarchy of pydantic metadata "config" objects, which become
increasingly specialized to support particular model types.

### ModelConfigBase

All model metadata classes inherit from this pydantic class. it
provides the following fields:

| **Field Name** | **Type**        |  **Description** |
|----------------|-----------------|------------------|
| `key`            | str           | Unique identifier for the model |
| `name`           | str           | Name of the model (not unique) |
| `model_type`     | ModelType     | The type of the model | 
| `model_format`   | ModelFormat   | The format of the model (e.g. "diffusers"); also used as a Union discriminator | 
| `base_model`     | BaseModelType | The base model that the model is compatible with | 
| `path`           | str           | Location of model on disk |
| `hash`           | str           | Most recent hash of the model's contents |
| `description`    | str           | Human-readable description of the model (optional) |
| `author`         | str           | Name of the model's author (optional) |
| `license`        | str           | Model's licensing model, as reported by the download source (optional) |
| `source`         | str           | Model's source URL or repo id (optional) |
| `thumbnail_url`  | str           | A thumbnail preview of model output, as reported by its source (optional) |
| `tags`           | List[str]     | A list of tags associated with the model, as reported by its source (optional) |

The `key` is a unique 32-character hash which is originally obtained
by sampling several parts of the model's files using the `imohash`
library. If the model is altered within InvokeAI (typically by
converting a checkpoint to a diffusers model) the key will remain the
same. The `hash` field holds the current hash of the model. It starts
out being the same as `key`, but may diverge.

`ModelType`, `ModelFormat` and `BaseModelType` are string enums that
are defined in `invokeai.backend.model_manager.config`. They are also
imported by, and can be reexported from,
`invokeai.app.services.model_record_service`:

```
from invokeai.app.services.model_record_service import ModelType, ModelFormat, BaseModelType
```

The `path` field can be absolute or relative. If relative, it is taken
to be relative to the `models_dir` setting in the user's
`invokeai.yaml` file.


### CheckpointConfig

This adds support for checkpoint configurations, and adds the
following field:

| **Field Name** | **Type**        |  **Description** |
|----------------|-----------------|------------------|
| `config`       | str             | Path to the checkpoint's config file |

`config` is the path to the checkpoint's config file. If relative, it
is taken to be relative to the InvokeAI root directory
(e.g. `configs/stable-diffusion/v1-inference.yaml`)

### MainConfig

This adds support for "main" Stable Diffusion models, and adds these
fields:

| **Field Name** | **Type**        |  **Description** |
|----------------|-----------------|------------------|
| `vae`          | str             | Path to a VAE to use instead of the burnt-in one |
| `variant`      | ModelVariantType| Model variant type, such as "inpainting" |

`vae` can be an absolute or relative path. If relative, its base is
taken to be the `models_dir` directory.

`variant` is an enumerated string class with values `normal`,
`inpaint` and `depth`. If needed, it can be imported if needed from
either `invokeai.app.services.model_record_service` or
`invokeai.backend.model_manager.config`.

### ONNXSD2Config

| **Field Name** | **Type**        |  **Description** |
|----------------|-----------------|------------------|
| `prediction_type`   | SchedulerPredictionType   | Scheduler prediction type to use, e.g. "epsilon" |
| `upcast_attention`  | bool | Model requires its attention module to be upcast |

The `SchedulerPredictionType` enum can be imported from either
`invokeai.app.services.model_record_service` or
`invokeai.backend.model_manager.config`.

### Other config classes

There are a series of such classes each discriminated by their
`ModelFormat`, including `LoRAConfig`, `IPAdapterConfig`, and so
forth. These are rarely needed outside the model manager's internal
code, but available in `invokeai.backend.model_manager.config` if
needed. There is also a Union of all ModelConfig classes, called
`AnyModelConfig` that can be imported from the same file.

### Limitations of the Data Model

The config hierarchy has a major limitation in its handling of the
base model type. Each model can only be compatible with one base
model, which breaks down in the event of models that are compatible
with two or more base models. For example, SD-1 VAEs also work with
SD-2 models. A partial workaround is to use `BaseModelType.Any`, which
indicates that the model is compatible with any of the base
models. This works OK for some models, such as the IP Adapter image
encoders, but is an all-or-nothing proposition.

Another issue is that the config class hierarchy is paralleled to some
extent by a `ModelBase` class hierarchy defined in
`invokeai.backend.model_manager.models.base` and its subclasses. These
are classes representing the models after they are loaded into RAM and
include runtime information such as load status and bytes used. Some
of the fields, including `name`, `model_type` and `base_model`, are
shared between `ModelConfigBase` and `ModelBase`, and this is a
potential source of confusion.

** TO DO: ** The `ModelBase` code needs to be revised to reduce the
duplication of similar classes and to support using the `key` as the
primary model identifier.

## Reading and Writing Model Configuration Records

The `ModelRecordService` provides the ability to retrieve model
configuration records from SQL or YAML databases, update them, and
write them back.

### Creating a `ModelRecordService` 

To create a new `ModelRecordService` database or open an existing one,
you can directly create either a `ModelRecordServiceSQL` or a
`ModelRecordServiceFile` object:

```
from invokeai.app.services.model_record_service import ModelRecordServiceSQL, ModelRecordServiceFile

store = ModelRecordServiceSQL.from_connection(connection, lock)
store = ModelRecordServiceSQL.from_db_file('/path/to/sqlite_database.db')
store = ModelRecordServiceFile.from_db_file('/path/to/database.yaml')
```

The `from_connection()` form is only available from the
`ModelRecordServiceSQL` class, and is used to manage records in a
previously-opened SQLITE3 database using a `sqlite3.connection` object
and a `threading.lock` object. It is intended for the specific use
case of storing the record information in the main InvokeAI database,
usually `databases/invokeai.db`.

The `from_db_file()` methods can be used to open new connections to
the named database files. If the file doesn't exist, it will be
created and initialized.

As a convenience, `ModelRecordServiceBase` offers two methods,
`from_db_file` and `open`, which will return either a SQL or File
implementation depending on the context. The former looks at the file
extension to determine whether to open the file as a SQL database
(".db") or as a file database (".yaml"). If the file exists, but is
either the wrong type or does not contain the expected schema
metainformation, then an appropriate `AssertionError` will be raised:

```
store = ModelRecordServiceBase.from_db_file('/path/to/a/file.{yaml,db}')
```

The `ModelRecordServiceBase.open()` method is specifically designed for use in the InvokeAI
web server and to maintain compatibility with earlier iterations of
the model manager. Its signature is:

```
def open(
       cls, 
	   config: InvokeAIAppConfig, 
	   conn: Optional[sqlite3.Connection] = None, 
	   lock: Optional[threading.Lock] = None
    ) -> Union[ModelRecordServiceSQL, ModelRecordServiceFile]:
```

The way it works is as follows:

1. Retrieve the value of the `model_config_db` option from the user's
	`invokeai.yaml` config file.
2. If `model_config_db` is `auto` (the default), then:
   - Use the values of `conn` and `lock` to return a `ModelRecordServiceSQL` object
	 opened on the passed connection and lock.
   - Open up a new connection to `databases/invokeai.db` if `conn`
     and/or `lock` are missing (see note below).
3. If `model_config_db` is a Path, then use `from_db_file`
   to return the appropriate type of ModelRecordService.
4. If `model_config_db` is None, then retrieve the legacy
   `conf_path` option from `invokeai.yaml` and use the Path
   indicated there. This will default to `configs/models.yaml`.
   
So a typical startup pattern would be:

```
import sqlite3
from invokeai.app.services.thread import lock
from invokeai.app.services.model_record_service import ModelRecordServiceBase
from invokeai.app.services.config import InvokeAIAppConfig

config = InvokeAIAppConfig.get_config()
db_conn = sqlite3.connect(config.db_path.as_posix(), check_same_thread=False)
store = ModelRecordServiceBase.open(config, db_conn, lock)
```

_A note on simultaneous access to `invokeai.db`_: The current InvokeAI
service architecture for the image and graph databases is careful to
use a shared sqlite3 connection and a thread lock to ensure that two
threads don't attempt to access the database simultaneously. However,
the default `sqlite3` library used by Python reports using
**Serialized** mode, which allows multiple threads to access the
database simultaneously using multiple database connections (see
https://www.sqlite.org/threadsafe.html and
https://ricardoanderegg.com/posts/python-sqlite-thread-safety/). Therefore
it should be safe to allow the record service to open its own SQLite
database connection. Opening a model record service should then be as
simple as `ModelRecordServiceBase.open(config)`.

### Fetching a Model's Configuration from `ModelRecordServiceBase`

Configurations can be retrieved in several ways.

#### get_model(key) -> AnyModelConfig:

The basic functionality is to call the record store object's
`get_model()` method with the desired model's unique key. It returns
the appropriate subclass of ModelConfigBase:

```
model_conf = store.get_model('f13dd932c0c35c22dcb8d6cda4203764')
print(model_conf.path)

>> '/tmp/models/ckpts/v1-5-pruned-emaonly.safetensors'

```

If the key is unrecognized, this call raises an
`UnknownModelException`.

#### exists(key) -> AnyModelConfig:

Returns True if a model with the given key exists in the databsae.

#### search_by_path(path) -> AnyModelConfig:

Returns the configuration of the model whose path is `path`. The path
is matched using a simple string comparison and won't correctly match
models referred to by different paths (e.g. using symbolic links).

#### search_by_name(name, base, type) -> List[AnyModelConfig]:

This method searches for models that match some combination of `name`,
`BaseType` and `ModelType`. Calling without any arguments will return
all the models in the database.

#### all_models() -> List[AnyModelConfig]:

Return all the model configs in the database. Exactly equivalent to
calling `search_by_name()` with no arguments.

#### search_by_tag(tags) -> List[AnyModelConfig]:

`tags` is a list of strings. This method returns a list of model
configs that contain all of the given tags. Examples:

```
# find all models that are marked as both SFW and as generating
# background scenery
configs = store.search_by_tag(['sfw', 'scenery'])
```

Note that only tags are not searchable in this way. Other fields can
be searched using a filter:

```
commercializable_models = [x for x in store.all_models() \
                           if x.license.contains('allowCommercialUse=Sell')]
```

#### version() -> str:

Returns the version of the database, currently at `3.2`

#### model_info_by_name(name, base_model, model_type) -> ModelConfigBase:

This method exists to ease the transition from the previous version of
the model manager, in which `get_model()` took the three arguments
shown above. This looks for a unique model identified by name, base
model and model type and returns it.

The method will generate a `DuplicateModelException` if there are more
than one models that share the same type, base and name. While
unlikely, it is certainly possible to have a situation in which the
user had added two models with the same name, base and type, one
located at path `/foo/my_model` and the other at `/bar/my_model`. It
is strongly recommended to search for models using `search_by_name()`,
which can return multiple results, and then to select the desired
model and pass its ke to `get_model()`.

### Writing model configs to the database

Several methods allow you to create and update stored model config
records.

#### add_model(key, config) -> ModelConfigBase:

Given a key and a configuration, this will add the model's
configuration record to the database. `config` can either be a subclass of
`ModelConfigBase` (i.e. any class listed in `AnyModelConfig`), or a
`dict` of key/value pairs. In the latter case, the correct
configuration class will be picked by Pydantic's discriminated union
mechanism.

If successful, the method will return the appropriate subclass of
`ModelConfigBase`. It will raise a `DuplicateModelException` if a
model with the same key is already in the database, or an
`InvalidModelConfigException` if a dict was passed and Pydantic
experienced a parse or validation error.

### update_model(key, config) -> AnyModelConfig:

Given a key and a configuration, this will update the model
configuration record in the database. `config` can be either a
instance of `ModelConfigBase`, or a sparse `dict` containing the
fields to be updated. This will return an `AnyModelConfig` on success,
or raise `InvalidModelConfigException` or `UnknownModelException`
exceptions on failure.

***TO DO:*** Investigate why `update_model()` returns an
`AnyModelConfig` while `add_model()` returns a `ModelConfigBase`.

### rename_model(key, new_name) -> ModelConfigBase:

This is a special case of `update_model()` for the use case of
changing the model's name. It is broken out because there are cases in
which the InvokeAI application wants to synchronize the model's name
with its path in the `models` directory after changing the name, type
or base. However, when using the ModelRecordService directly, the call
is equivalent to:

```
store.rename_model(key, {'name': 'new_name'})
```

***TO DO:*** Investigate why `rename_model()` is returning a
`ModelConfigBase` while `update_model()` returns a `AnyModelConfig`.

***

## Let's get loaded, the lowdown on ModelLoadService

The `ModelLoadService` is responsible for loading a named model into
memory so that it can be used for inference. Despite the fact that it
does a lot under the covers, it is very straightforward to use.

### Creating a ModelLoadService object

The class is defined in
`invokeai.app.services.model_loader_service`. It is initialized with
an InvokeAIAppConfig object, from which it gets configuration
information such as the user's desired GPU and precision, and with a
previously-created `ModelRecordServiceBase` object, from which it
loads the requested model's configuration information.

Here is a typical initialization pattern:

```
from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.model_record_service import ModelRecordServiceBase
from invokeai.app.services.model_loader_service import ModelLoadService

config = InvokeAIAppConfig.get_config()
store = ModelRecordServiceBase.open(config)
loader = ModelLoadService(config, store)
```

Note that we are relying on the contents of the application
configuration to choose the implementation of
`ModelRecordServiceBase`.

### get_model(key, [submodel_type], [context]) -> ModelInfo:

The `get_model()` method, like its similarly-named cousin in
`ModelRecordService`, receives the unique key that identifies the
model.  It loads the model into memory, gets the model ready for use,
and returns a `ModelInfo` object. 

The optional second argument, `subtype` is a `SubModelType` string
enum, such as "vae". It is mandatory when used with a main model, and
is used to select which part of the main model to load.

The optional third argument, `invocation_context` can be provided by
an invocation to trigger model load event reporting. See below for
details.

The returned `ModelInfo` object shares some fields in common with
`ModelConfigBase`, but is otherwise a completely different beast:

| **Field Name** | **Type**        |  **Description** |
|----------------|-----------------|------------------|
| `key`          | str                    | The model key derived from the ModelRecordServie database |
| `name`         | str                    | Name of this model |
| `base_model`   | BaseModelType          | Base model for this model |
| `type`         | ModelType or SubModelType   | Either the model type (non-main) or the submodel type (main models)|
| `location`     | Path or str            | Location of the model on the filesystem |
| `precision`    | torch.dtype            | The torch.precision to use for inference |
| `context`      | ModelCache.ModelLocker | A context class used to lock the model in VRAM while in use |

The types for `ModelInfo` and `SubModelType` can be imported from
`invokeai.app.services.model_loader_service`.

To use the model, you use the `ModelInfo` as a context manager using
the following pattern:

```
model_info = loader.get_model('f13dd932c0c35c22dcb8d6cda4203764', SubModelType('vae'))
with model_info as vae:
	image = vae.decode(latents)[0]
```

The `vae` model will stay locked in the GPU during the period of time
it is in the context manager's scope.

`get_model()` may raise any of the following exceptions:

- `UnknownModelException`  -- key not in database
- `ModelNotFoundException` -- key in database but model not found at path
- `InvalidModelException`  -- the model is guilty of a variety of sins
  
** TO DO: ** Resolve discrepancy between ModelInfo.location and
ModelConfig.path.

### Emitting model loading events

When the `context` argument is passed to `get_model()`, it will
retrieve the invocation event bus from the passed `InvocationContext`
object to emit events on the invocation bus. The two events are
"model_load_started" and "model_load_completed". Both carry the
following payload:

```
payload=dict(
	queue_id=queue_id,
	queue_item_id=queue_item_id,
	queue_batch_id=queue_batch_id,
	graph_execution_state_id=graph_execution_state_id,
	model_key=model_key,
	submodel=submodel,
	hash=model_info.hash,
	location=str(model_info.location),
	precision=str(model_info.precision),
)
```


# Introduction to the Model Manager V2

The Model Manager is responsible for organizing the various machine
learning models used by InvokeAI. It consists of a series of
interdependent services that together handle the full lifecycle of a
model. These are the:

* _ModelRecordServiceBase_ Responsible for managing model metadata and
  configuration information. Among other things, the record service
  tracks the type of the model, its provenance, and where it can be
  found on disk.
  
* _ModelInstallServiceBase_ A service for installing models to
  disk. It uses `DownloadQueueServiceBase` to download models and
  their metadata, and `ModelRecordServiceBase` to store that
  information. It is also responsible for managing the InvokeAI
  `models` directory and its contents.
  
* _DownloadQueueServiceBase_
  A multithreaded downloader responsible
  for downloading models from a remote source to disk. The download
  queue has special methods for downloading repo_id folders from
  Hugging Face, as well as discriminating among model versions in
  Civitai, but can be used for arbitrary content.
  
  * _ModelLoadServiceBase_
  Responsible for loading a model from disk
  into RAM and VRAM and getting it ready for inference.

## Location of the Code

The four main services can be found in
`invokeai/app/services` in the following directories:

* `invokeai/app/services/model_records/`
* `invokeai/app/services/model_install/`
* `invokeai/app/services/downloads/`
* `invokeai/app/services/model_load/`

Code related to the FastAPI web API can be found in
`invokeai/app/api/routers/model_manager_v2.py`.

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
| `hash`           | str           | Hash of the model |
| `description`    | str           | Human-readable description of the model (optional) |
| `source`         | str           | Model's source URL or repo id (optional) |

The `key` is a unique 32-character random ID which was generated at
install time. The `hash` field stores a hash of the model's
contents at install time obtained by sampling several parts of the
model's files using the `imohash` library. Over the course of the
model's lifetime it may be transformed in various ways, such as
changing its precision or converting it from a .safetensors to a
diffusers model.

`ModelType`, `ModelFormat` and `BaseModelType` are string enums that
are defined in `invokeai.backend.model_manager.config`. They are also
imported by, and can be reexported from,
`invokeai.app.services.model_manager.model_records`:

```
from invokeai.app.services.model_records import ModelType, ModelFormat, BaseModelType
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
either `invokeai.app.services.model_records` or
`invokeai.backend.model_manager.config`.

### ONNXSD2Config

| **Field Name** | **Type**        |  **Description** |
|----------------|-----------------|------------------|
| `prediction_type`   | SchedulerPredictionType   | Scheduler prediction type to use, e.g. "epsilon" |
| `upcast_attention`  | bool | Model requires its attention module to be upcast |

The `SchedulerPredictionType` enum can be imported from either
`invokeai.app.services.model_records` or
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

## Reading and Writing Model Configuration Records

The `ModelRecordService` provides the ability to retrieve model
configuration records from SQL or YAML databases, update them, and
write them back.

A application-wide `ModelRecordService` is created during API
initialization and can be retrieved within an invocation from the
`InvocationContext` object:

```
store = context.services.model_manager.store
```

or from elsewhere in the code by accessing
`ApiDependencies.invoker.services.model_manager.store`.

### Creating a `ModelRecordService`

To create a new `ModelRecordService` database or open an existing one,
you can directly create either a `ModelRecordServiceSQL` or a
`ModelRecordServiceFile` object:

```
from invokeai.app.services.model_records import ModelRecordServiceSQL, ModelRecordServiceFile

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

The `ModelRecordServiceBase.open()` method is specifically designed
for use in the InvokeAI web server. Its signature is:

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
   * Use the values of `conn` and `lock` to return a `ModelRecordServiceSQL` object
  opened on the passed connection and lock.
   * Open up a new connection to `databases/invokeai.db` if `conn`
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
from invokeai.app.services.model_records import ModelRecordServiceBase
from invokeai.app.services.config import InvokeAIAppConfig

config = InvokeAIAppConfig.get_config()
db_conn = sqlite3.connect(config.db_path.as_posix(), check_same_thread=False)
store = ModelRecordServiceBase.open(config, db_conn, lock)
```

### Fetching a Model's Configuration from `ModelRecordServiceBase`

Configurations can be retrieved in several ways.

#### get_model(key) -> AnyModelConfig

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

#### exists(key) -> AnyModelConfig

Returns True if a model with the given key exists in the databsae.

#### search_by_path(path) -> AnyModelConfig

Returns the configuration of the model whose path is `path`. The path
is matched using a simple string comparison and won't correctly match
models referred to by different paths (e.g. using symbolic links).

#### search_by_name(name, base, type) -> List[AnyModelConfig]

This method searches for models that match some combination of `name`,
`BaseType` and `ModelType`. Calling without any arguments will return
all the models in the database.

#### all_models() -> List[AnyModelConfig]

Return all the model configs in the database. Exactly equivalent to
calling `search_by_name()` with no arguments.

#### search_by_tag(tags) -> List[AnyModelConfig]

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

#### version() -> str

Returns the version of the database, currently at `3.2`

#### model_info_by_name(name, base_model, model_type) -> ModelConfigBase

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
model and pass its key to `get_model()`.

### Writing model configs to the database

Several methods allow you to create and update stored model config
records.

#### add_model(key, config) -> AnyModelConfig

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

### update_model(key, config) -> AnyModelConfig

Given a key and a configuration, this will update the model
configuration record in the database. `config` can be either a
instance of `ModelConfigBase`, or a sparse `dict` containing the
fields to be updated. This will return an `AnyModelConfig` on success,
or raise `InvalidModelConfigException` or `UnknownModelException`
exceptions on failure.

***

## Model installation

The `ModelInstallService` class implements the
`ModelInstallServiceBase` abstract base class, and provides a one-stop
shop for all your model install needs. It provides the following
functionality:

* Registering a model config record for a model already located on the
  local filesystem, without moving it or changing its path.
  
* Installing a model alreadiy located on the local filesystem, by
  moving it into the InvokeAI root directory under the
  `models` folder (or wherever config parameter `models_dir`
  specifies).

* Probing of models to determine their type, base type and other key
  information.
  
* Interface with the InvokeAI event bus to provide status updates on
  the download, installation and registration process.
  
* Downloading a model from an arbitrary URL and installing it in
  `models_dir`.

* Special handling for HuggingFace repo_ids to recursively download
  the contents of the repository, paying attention to alternative
  variants such as fp16.
  
* Saving tags and other metadata about the model into the invokeai database
  when fetching from a repo that provides that type of information,
  (currently only HuggingFace).
  
### Initializing the installer

A default installer is created at InvokeAI api startup time and stored
in `ApiDependencies.invoker.services.model_install` and can
also be retrieved from an invocation's `context` argument with
`context.services.model_install`.

In the event you wish to create a new installer, you may use the
following initialization pattern:

```
from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.model_records import ModelRecordServiceSQL
from invokeai.app.services.model_install import ModelInstallService
from invokeai.app.services.download import DownloadQueueService
from invokeai.app.services.shared.sqlite import SqliteDatabase
from invokeai.backend.util.logging import InvokeAILogger

config = InvokeAIAppConfig.get_config()
config.parse_args()

logger = InvokeAILogger.get_logger(config=config)
db = SqliteDatabase(config, logger)
record_store = ModelRecordServiceSQL(db)
queue = DownloadQueueService()
queue.start()

installer = ModelInstallService(app_config=config, 
                                record_store=record_store,
              download_queue=queue
           )
installer.start()
```

The full form of `ModelInstallService()` takes the following
required parameters:

| **Argument**     | **Type**                     | **Description**              |
|------------------|------------------------------|------------------------------|
| `app_config`         | InvokeAIAppConfig       | InvokeAI app configuration object |
| `record_store`   | ModelRecordServiceBase  | Config record storage database |
| `download_queue`   | DownloadQueueServiceBase  | Download queue object |
|`session`           | Optional[requests.Session]    | Swap in a different Session object (usually for debugging) |

Once initialized, the installer will provide the following methods:

#### install_job = installer.heuristic_import(source, [config], [access_token])

This is a simplified interface to the installer which takes a source
string, an optional model configuration dictionary and an optional
access token.

The `source` is a string that can be any of these forms

1. A path on the local filesystem (`C:\\users\\fred\\model.safetensors`)
2. A Url pointing to a single downloadable model file (`https://civitai.com/models/58390/detail-tweaker-lora-lora`)
3. A HuggingFace repo_id with any of the following formats:
   * `model/name` -- entire model
   * `model/name:fp32` -- entire model, using the fp32 variant
   * `model/name:fp16:vae` -- vae submodel, using the fp16 variant
   * `model/name::vae` -- vae submodel, using default precision
   * `model/name:fp16:path/to/model.safetensors` -- an individual model file, fp16 variant
   * `model/name::path/to/model.safetensors` -- an individual model file, default variant

Note that by specifying a relative path to the top of the HuggingFace
repo, you can download and install arbitrary models files.

The variant, if not provided, will be automatically filled in with
`fp32` if the user has requested full precision, and `fp16`
otherwise. If a variant that does not exist is requested, then the
method will install whatever HuggingFace returns as its default
revision.

`config` is an optional dict of values that will override the
autoprobed values for model type, base, scheduler prediction type, and
so forth. See [Model configuration and
probing](#Model-configuration-and-probing) for details.

`access_token` is an optional access token for accessing resources
that need authentication.

The method will return a `ModelInstallJob`. This object is discussed
at length in the following section.

#### install_job = installer.import_model()

The `import_model()` method is the core of the installer. The
following illustrates basic usage:

```
from invokeai.app.services.model_install import (
 LocalModelSource,
 HFModelSource,
 URLModelSource,
)

source1 = LocalModelSource(path='/opt/models/sushi.safetensors')   # a local safetensors file
source2 = LocalModelSource(path='/opt/models/sushi_diffusers')     # a local diffusers folder

source3 = HFModelSource(repo_id='runwayml/stable-diffusion-v1-5')  # a repo_id
source4 = HFModelSource(repo_id='runwayml/stable-diffusion-v1-5', subfolder='vae')  # a subfolder within a repo_id
source5 = HFModelSource(repo_id='runwayml/stable-diffusion-v1-5', variant='fp16')   # a named variant of a HF model
source6 = HFModelSource(repo_id='runwayml/stable-diffusion-v1-5', subfolder='OrangeMix/OrangeMix1.ckpt')   # path to an individual model file

source7 = URLModelSource(url='https://civitai.com/api/download/models/63006')       # model located at a URL
source8 = URLModelSource(url='https://civitai.com/api/download/models/63006', access_token='letmein') # with an access token

for source in [source1, source2, source3, source4, source5, source6, source7]:
   install_job = installer.install_model(source)
   
source2job = installer.wait_for_installs(timeout=120)
for source in sources:
    job = source2job[source]
 if job.complete:
  model_config = job.config_out
  model_key = model_config.key
  print(f"{source} installed as {model_key}")
 elif job.errored:
     print(f"{source}: {job.error_type}.\nStack trace:\n{job.error}")
 
```

As shown here, the `import_model()` method accepts a variety of
sources, including local safetensors files, local diffusers folders,
HuggingFace repo_ids with and without a subfolder designation,
Civitai model URLs and arbitrary URLs that point to checkpoint files
(but not to folders).

Each call to `import_model()` return a `ModelInstallJob` job,
an object which tracks the progress of the install.

If a remote model is requested, the model's files are downloaded in
parallel across a multiple set of threads using the download
queue. During the download process, the `ModelInstallJob` is updated
to provide status and progress information. After the files (if any)
are downloaded, the remainder of the installation runs in a single
serialized background thread. These are the model probing, file
copying, and config record database update steps.

Multiple install jobs can be queued up. You may block until all
install jobs are completed (or errored) by calling the
`wait_for_installs()` method as shown in the code
example. `wait_for_installs()` will return a `dict` that maps the
requested source to its job. This object can be interrogated
to determine its status. If the job errored out, then the error type
and details can be recovered from `job.error_type` and `job.error`.

The full list of arguments to `import_model()` is as follows:

| **Argument**     | **Type**                     | **Default** | **Description**                           |
|------------------|------------------------------|-------------|-------------------------------------------|
| `source`         | ModelSource                 | None        | The source of the model, Path, URL or repo_id |
| `config`         | Dict[str, Any]               | None        | Override all or a portion of model's probed attributes |

The next few sections describe the various types of ModelSource that
can be passed to `import_model()`.

`config` can be used to override all or a portion of the configuration
attributes returned by the model prober. See the section below for
details.

#### LocalModelSource

This is used for a model that is located on a locally-accessible Posix
filesystem, such as a local disk or networked fileshare.

| **Argument**     | **Type**                     | **Default** | **Description**                           |
|------------------|------------------------------|-------------|-------------------------------------------|
| `path`           | str | Path                   | None        | Path to the model file or directory |
| `inplace`        | bool                         | False       | If set, the model file(s) will be left in their location; otherwise they will be copied into the InvokeAI root's `models` directory |

#### URLModelSource

This is used for a single-file model that is accessible via a URL. The
fields are:

| **Argument**     | **Type**                     | **Default** | **Description**                           |
|------------------|------------------------------|-------------|-------------------------------------------|
| `url`            | AnyHttpUrl                   | None        | The URL for the model file. |
| `access_token`   | str                          | None        | An access token needed to gain access to this file. |

The `AnyHttpUrl` class can be imported from `pydantic.networks`.

Ordinarily, no metadata is retrieved from these sources. However,
there is special-case code in the installer that looks for HuggingFace
and fetches the corresponding model metadata from the corresponding repo.

#### HFModelSource

HuggingFace has the most complicated `ModelSource` structure:

| **Argument**     | **Type**                     | **Default** | **Description**                           |
|------------------|------------------------------|-------------|-------------------------------------------|
| `repo_id`        | str                          | None        | The ID of the desired model. |
| `variant`        | ModelRepoVariant             | ModelRepoVariant('fp16')      | The desired variant. |
| `subfolder`      | Path                         | None        | Look for the model in a subfolder of the repo. |
| `access_token`   | str                          | None        | An access token needed to gain access to a subscriber's-only model. |

The `repo_id` is the repository ID, such as `stabilityai/sdxl-turbo`.

The `variant` is one of the various diffusers formats that HuggingFace
supports and is used to pick out from the hodgepodge of files that in
a typical HuggingFace repository the particular components needed for
a complete diffusers model. `ModelRepoVariant` is an enum that can be
imported from `invokeai.backend.model_manager` and has the following
values:

| **Name**                   | **String Value**          |
|----------------------------|---------------------------|
| ModelRepoVariant.DEFAULT   | "default"                 |
| ModelRepoVariant.FP16      | "fp16"                 |
| ModelRepoVariant.FP32      | "fp32"                 |
| ModelRepoVariant.ONNX      | "onnx"                 |
| ModelRepoVariant.OPENVINO  | "openvino"             |
| ModelRepoVariant.FLAX      | "flax"                 |

You can also pass the string forms to `variant` directly. Note that
InvokeAI may not be able to load and run all variants. At the current
time, specifying `ModelRepoVariant.DEFAULT` will retrieve model files
that are unqualified, e.g. `pytorch_model.safetensors` rather than
`pytorch_model.fp16.safetensors`. These are usually the 32-bit
safetensors forms of the model.

If `subfolder` is specified, then the requested model resides in a
subfolder of the main model repository. This is typically used to
fetch and install VAEs.

Some models require you to be registered with HuggingFace and logged
in. To download these files, you must provide an
`access_token`. Internally, if no access token is provided, then
`HfFolder.get_token()` will be called to fill it in with the cached
one.

#### Monitoring the install job process

When you create an install job with `import_model()`, it launches the
download and installation process in the background and returns a
`ModelInstallJob` object for monitoring the process.

The `ModelInstallJob` class has the following structure:

| **Attribute** | **Type**        |  **Description** |
|----------------|-----------------|------------------|
| `id`           | `int`           | Integer ID for this job |
| `status`       | `InstallStatus`  | An enum of [`waiting`, `downloading`, `running`, `completed`, `error` and `cancelled`]|
| `config_in`    | `dict`          | Overriding configuration values provided by the caller |
| `config_out`   | `AnyModelConfig`| After successful completion, contains the configuration record written to the database |
| `inplace`      | `boolean`       | True if the caller asked to install the model in place using its local path |
| `source`       | `ModelSource`   | The local path, remote URL or repo_id of the model to be installed |
| `local_path`   | `Path`          | If a remote model, holds the path of the model after it is downloaded; if a local model, same as `source` |
| `error_type`   | `str`           | Name of the exception that led to an error status |
| `error`        | `str`           | Traceback of the error |

If the `event_bus` argument was provided, events will also be
broadcast to the InvokeAI event bus. The events will appear on the bus
as an event of type `EventServiceBase.model_event`, a timestamp and
the following event names:

##### `model_install_downloading`

For remote models only, `model_install_downloading` events will be issued at regular
intervals as the download progresses. The event's payload contains the
following keys:

| **Key** | **Type**        |  **Description** |
|----------------|-----------|------------------|
| `source`       | str       | String representation of the requested source |
| `local_path`   | str       | String representation of the path to the downloading model (usually a temporary directory) |
| `bytes`        | int       | How many bytes downloaded so far |
| `total_bytes`  | int       | Total size of all the files that make up the model |
| `parts`        | List[Dict]| Information on the progress of the individual files that make up the model |

The parts is a list of dictionaries that give information on each of
the components pieces of the download. The dictionary's keys are
`source`, `local_path`, `bytes` and `total_bytes`, and correspond to
the like-named keys in the main event.

Note that downloading events will not be issued for local models, and
that downloading events occur _before_ the running event.

##### `model_install_running`

`model_install_running` is issued when all the required downloads have completed (if applicable) and the
model probing, copying and registration process has now started.

The payload will contain the key `source`.

##### `model_install_completed`

`model_install_completed` is issued once at the end of a successful
installation. The payload will contain the keys `source`,
`total_bytes` and `key`, where `key` is the ID under which the model
has been registered.

##### `model_install_error`

`model_install_error` is emitted if the installation process fails for
some reason. The payload will contain the keys `source`, `error_type`
and `error`. `error_type` is a short message indicating the nature of
the error, and `error` is the long traceback to help debug the
problem.

##### `model_install_cancelled`

`model_install_cancelled` is issued if the model installation is
cancelled, or if one or more of its files' downloads are
cancelled. The payload will contain `source`.

##### Following the model status

You may poll the `ModelInstallJob` object returned by `import_model()`
to ascertain the state of the install. The job status can be read from
the job's `status` attribute, an `InstallStatus` enum which has the
enumerated values `WAITING`, `DOWNLOADING`, `RUNNING`, `COMPLETED`,
`ERROR` and `CANCELLED`.

For convenience, install jobs also provided the following boolean
properties: `waiting`, `downloading`, `running`, `complete`, `errored`
and `cancelled`, as well as `in_terminal_state`. The last will return
True if the job is in the complete, errored or cancelled states.

#### Model configuration and probing

The install service uses the `invokeai.backend.model_manager.probe`
module during import to determine the model's type, base type, and
other configuration parameters. Among other things, it assigns a
default name and description for the model based on probed
fields.

When downloading remote models is implemented, additional
configuration information, such as list of trigger terms, will be
retrieved from the HuggingFace and Civitai model repositories.

The probed values can be overriden by providing a dictionary in the
optional `config` argument passed to `import_model()`. You may provide
overriding values for any of the model's configuration
attributes. Here is an example of setting the
`SchedulerPredictionType` and `name` for an sd-2 model:

```
install_job = installer.import_model(
               source=HFModelSource(repo_id='stabilityai/stable-diffusion-2-1',variant='fp32'),
      config=dict(
            prediction_type=SchedulerPredictionType('v_prediction')
      name='stable diffusion 2 base model',
            )
       )
```

### Other installer methods

This section describes additional methods provided by the installer class.

#### jobs = installer.wait_for_installs([timeout])

Block until all pending installs are completed or errored and then
returns a list of completed jobs. The optional `timeout` argument will
return from the call if jobs aren't completed in the specified
time. An argument of 0 (the default) will block indefinitely.

#### jobs = installer.wait_for_job(job, [timeout])

Like `wait_for_installs()`, but block until a specific job has
completed or errored, and then return the job.  The optional `timeout`
argument will return from the call if the job doesn't complete in the
specified time. An argument of 0 (the default) will block
indefinitely.

#### jobs = installer.list_jobs()

Return a list of all active and complete `ModelInstallJobs`.

#### jobs = installer.get_job_by_source(source)

Return a list of `ModelInstallJob` corresponding to the indicated
model source.

#### jobs = installer.get_job_by_id(id)

Return a list of `ModelInstallJob` corresponding to the indicated
model id.

#### jobs = installer.cancel_job(job)

Cancel the indicated job.

#### installer.prune_jobs

Remove jobs that are in a terminal state (i.e. complete, errored or
cancelled) from the job list returned by `list_jobs()` and
`get_job()`.

#### installer.app_config, installer.record_store, installer.event_bus

Properties that provide access to the installer's `InvokeAIAppConfig`,
`ModelRecordServiceBase` and `EventServiceBase` objects.

#### key = installer.register_path(model_path, config), key = installer.install_path(model_path, config)

These methods bypass the download queue and directly register or
install the model at the indicated path, returning the unique ID for
the installed model.

Both methods accept a Path object corresponding to a checkpoint or
diffusers folder, and an optional dict of config attributes to use to
override the values derived from model probing.

The difference between `register_path()` and `install_path()` is that
the former creates a model configuration record without changing the
location of the model in the filesystem. The latter makes a copy of
the model inside the InvokeAI models directory before registering
it.

#### installer.unregister(key)

This will remove the model config record for the model at key, and is
equivalent to `installer.record_store.del_model(key)`

#### installer.delete(key)

This is similar to `unregister()` but has the additional effect of
conditionally deleting the underlying model file(s) if they reside
within the InvokeAI models directory

#### installer.unconditionally_delete(key)

This method is similar to `unregister()`, but also unconditionally
deletes the corresponding model weights file(s), regardless of whether
they are inside or outside the InvokeAI models hierarchy.

#### path = installer.download_and_cache(remote_source, [access_token], [timeout])

This utility routine will download the model file located at source,
cache it, and return the path to the cached file. It does not attempt
to determine the model type, probe its configuration values, or
register it with the models database.

You may provide an access token if the remote source requires
authorization. The call will block indefinitely until the file is
completely downloaded, cancelled or raises an error of some sort. If
you provide a timeout (in seconds), the call will raise a
`TimeoutError` exception if the download hasn't completed in the
specified period.

You may use this mechanism to request any type of file, not just a
model. The file will be stored in a subdirectory of
`INVOKEAI_ROOT/models/.cache`. If the requested file is found in the
cache, its path will be returned without redownloading it.

Be aware that the models cache is cleared of infrequently-used files
and directories at regular intervals when the size of the cache
exceeds the value specified in Invoke's `convert_cache` configuration
variable.

#### List[str]=installer.scan_directory(scan_dir: Path, install: bool)

This method will recursively scan the directory indicated in
`scan_dir` for new models and either install them in the models
directory or register them in place, depending on the setting of
`install` (default False).

The return value is the list of keys of the new installed/registered
models.

#### installer.sync_to_config()

This method synchronizes models in the models directory and autoimport
directory to those in the `ModelConfigRecordService` database. New
models are registered and orphan models are unregistered.

#### installer.start(invoker)

The `start` method is called by the API intialization routines when
the API starts up. Its effect is to call `sync_to_config()` to
synchronize the model record store database with what's currently on
disk.

***

## Get on line: The Download Queue

InvokeAI can download arbitrary files using a multithreaded background
download queue. Internally, the download queue is used for installing
models located at remote locations. The queue is implemented by the
`DownloadQueueService` defined in
`invokeai.app.services.download_manager`. However, most of the
implementation is spread out among several files in
`invokeai/backend/model_manager/download/*`

A default download queue is located in
`ApiDependencies.invoker.services.download_queue`. However, you can
create additional instances if you need to isolate your queue from the
main one.

### A job for every task

The queue operates on a series of download job objects. These objects
specify the source and destination of the download, and keep track of
the progress of the download. Jobs come in a variety of shapes and
colors as they are progressively specialized for particular download
task.

The basic job is the `DownloadJobBase`, a pydantic object with the
following fields:

| **Field**      | **Type**        |  **Default**  | **Description** |
|----------------|-----------------|---------------|-----------------|
| `id`             | int             |               | Job ID, an integer >= 0           |
| `priority`       | int             | 10            | Job priority. Lower priorities run before higher priorities |
| `source`         | str             |               | Where to download from (specialized types used in subclasses)|
| `destination`    | Path            |               | Where to download to              |
| `status`         | DownloadJobStatus| Idle         | Job's status (see below)          |
| `event_handlers` | List[DownloadEventHandler]|     | Event handlers (see below)        |
| `job_started`    | float            |              | Timestamp for when the job started running |
| `job_ended`      | float            |              | Timestamp for when the job completed or errored out |
| `job_sequence`   | int              |              | A counter that is incremented each time a model is dequeued |
| `error`          | Exception        |              | A copy of the Exception that caused an error during download |

When you create a job, you can assign it a `priority`. If multiple
jobs are queued, the job with the lowest priority runs first. (Don't
blame me! The Unix developers came up with this convention.)

Every job has a `source` and a `destination`. `source` is a string in
the base class, but subclassses redefine it more specifically.

The `destination` must be the Path to a file or directory on the local
filesystem. If the Path points to a new or existing file, then the
source will be stored under that filename. If the Path ponts to an
existing directory, then the downloaded file will be stored inside the
directory, usually using the name assigned to it at the remote site in
the `content-disposition` http field.

When the job is submitted, it is assigned a numeric `id`. The id can
then be used to control the job, such as starting, stopping and
cancelling its download.

The `status` field is updated by the queue to indicate where the job
is in its lifecycle. Values are defined in the string enum
`DownloadJobStatus`, a symbol available from
`invokeai.app.services.download_manager`. Possible values are:

| **Value**    |   **String Value**  | **Description** |
|--------------|---------------------|-------------------|
| `IDLE`         | idle              | Job created, but not submitted to the queue |
| `ENQUEUED`     | enqueued          | Job is patiently waiting on the queue       |
| `RUNNING`      | running           | Job is running!                             |
| `PAUSED`       | paused            | Job was paused and can be restarted         |
| `COMPLETED`    | completed         | Job has finished its work without an error |
| `ERROR`        | error             | Job encountered an error and will not run again|
| `CANCELLED`    | cancelled         | Job was cancelled and will not run (again) |

`job_started`, `job_ended` and `job_sequence` indicate when the job
was started (using a python timestamp), when it completed, and the
order in which it was taken off the queue. These are mostly used for
debugging and performance testing.

In case of an error, the Exception that caused the error will be
placed in the `error` field, and the job's status will be set to
`DownloadJobStatus.ERROR`.

After an error occurs, any partially downloaded files will be deleted
from disk, unless `preserve_partial_downloads` was set to True at job
creation time (or set to True any time before the error
occurred). Note that since all InvokeAI model install operations
involve downloading files to a temporary directory that has a limited
lifetime, this flag is not used by the model installer.

There are a series of subclasses of `DownloadJobBase` that provide
support for specific types of downloads. These are:

#### DownloadJobPath

This subclass redefines `source` to be a filesystem Path. It is used
to move a file or directory from the `source` to the `destination`
paths in the background using a uniform event-based infrastructure.

#### DownloadJobRemoteSource

This subclass adds the following fields to the job:

| **Field**      | **Type**        |  **Default**  | **Description** |
|----------------|-----------------|---------------|-----------------|
| `bytes`        | int             |  0            | bytes downloaded so far |
| `total_bytes`  | int             |  0            | total size to download |
| `access_token` | Any             |  None         | an authorization token to present to the remote source |

The job will start out with 0/0 in its bytes/total_bytes fields. Once
it starts running, `total_bytes` will be populated from information
provided in the HTTP download header (if available), and the number of
bytes downloaded so far will be progressively incremented.

#### DownloadJobURL

This is a subclass of `DownloadJobBase`. It redefines `source` to be a
Pydantic `AnyHttpUrl` object, which enforces URL validation checking
on the field.

Note that the installer service defines an additional subclass of
`DownloadJobRemoteSource` that accepts HuggingFace repo_ids in
addition to URLs. This is discussed later in this document.

### Event handlers

While a job is being downloaded, the queue will emit events at
periodic intervals. A typical series of events during a successful
download session will look like this:

* enqueued
* running
* running
* running
* completed

There will be a single enqueued event, followed by one or more running
events, and finally one `completed`, `error` or `cancelled`
events.

It is possible for a caller to pause download temporarily, in which
case the events may look something like this:

* enqueued
* running
* running
* paused
* running
* completed

The download queue logs when downloads start and end (unless `quiet`
is set to True at initialization time) but doesn't log any progress
events. You will probably want to be alerted to events during the
download job and provide more user feedback. In order to intercept and
respond to events you may install a series of one or more event
handlers in the job. Whenever the job's status changes, the chain of
event handlers is traversed and executed in the same thread that the
download job is running in.

Event handlers have the signature `Callable[["DownloadJobBase"],
None]`, i.e.

```
def handler(job: DownloadJobBase):
   pass
```

A typical handler will examine `job.status` and decide if there's
something to be done. This can include cancelling or erroring the job,
but more typically is used to report on the job status to the user
interface or to perform certain actions on successful completion of
the job.

Event handlers can be attached to a job at creation time. In addition,
you can create a series of default handlers that are attached to the
queue object itself. These handlers will be executed for each job
after the job's own handlers (if any) have run.

During a download, running events are issued every time roughly 1% of
the file is transferred. This is to provide just enough granularity to
update a tqdm progress bar smoothly.

Handlers can be added to a job after the fact using the job's
`add_event_handler` method:

```
job.add_event_handler(my_handler)
```

All handlers can be cleared using the job's `clear_event_handlers()`
method. Note that it might be a good idea to pause the job before
altering its handlers.

### Creating a download queue object

The `DownloadQueueService` constructor takes the following arguments:

| **Argument** | **Type**          |  **Default**  | **Description** |
|----------------|-----------------|---------------|-----------------|
| `event_handlers`   | List[DownloadEventHandler]  | []   | Event handlers |
| `max_parallel_dl`  | int                         | 5    | Maximum number of simultaneous downloads allowed |
| `requests_session` | requests.sessions.Session   | None | An alternative requests Session object to use for the download |
| `quiet`            | bool                        | False| Do work quietly without issuing log messages |

A typical initialization sequence will look like:

```
from invokeai.app.services.download_manager import DownloadQueueService

def log_download_event(job: DownloadJobBase):
 logger.info(f'job={job.id}: status={job.status}')

queue = DownloadQueueService(
       event_handlers=[log_download_event]
        )
```

Event handlers can be provided to the queue at initialization time as
shown in the example. These will be automatically appended to the
handler list for any job that is submitted to this queue.

`max_parallel_dl` sets the number of simultaneous active downloads
that are allowed. The default of five has not been benchmarked in any
way, but seems to give acceptable performance.

`requests_session` can be used to provide a `requests` module Session
object that will be used to stream remote URLs to disk. This facility
was added for use in the module's unit tests to simulate a remote web
server, but may be useful in other contexts.

`quiet` will prevent the queue from issuing any log messages at the
INFO or higher levels.

### Submitting a download job

You can submit a download job to the queue either by creating the job
manually and passing it to the queue's `submit_download_job()` method,
or using the `create_download_job()` method, which will do the same
thing on your behalf.

To use the former method, follow this example:

```
job = DownloadJobRemoteSource(
         source='http://www.civitai.com/models/13456',
   destination='/tmp/models/',
   event_handlers=[my_handler1, my_handler2], # if desired
   )
queue.submit_download_job(job, start=True)
```

`submit_download_job()` takes just two arguments: the job to submit,
and a flag indicating whether to immediately start the job (defaulting
to True). If you choose not to start the job immediately, you can
start it later by calling the queue's `start_job()` or
`start_all_jobs()` methods, which are described later.

To have the queue create the job for you, follow this example instead:

```
job = queue.create_download_job(
         source='http://www.civitai.com/models/13456',
   destdir='/tmp/models/',
   filename='my_model.safetensors',
   event_handlers=[my_handler1, my_handler2], # if desired
   start=True,
 )
```

The `filename` argument forces the downloader to use the specified
name for the file rather than the name provided by the remote source,
and is equivalent to manually specifying a destination of
`/tmp/models/my_model.safetensors' in the submitted job.

Here is the full list of arguments that can be provided to
`create_download_job()`:

| **Argument**     | **Type**                     | **Default** | **Description**                           |
|------------------|------------------------------|-------------|-------------------------------------------|
| `source`         | Union[str, Path, AnyHttpUrl] |             | Download remote or local source           |
| `destdir`        | Path                         |             | Destination directory for downloaded file |
| `filename`       | Path                         | None        | Filename for downloaded file              |
| `start`          | bool                         | True        | Enqueue the job immediately               |
| `priority`       | int                          | 10          | Starting priority for this job            |
| `access_token`   | str                          | None        | Authorization token for this resource     |
| `event_handlers` | List[DownloadEventHandler]   | []          | Event handlers for this job                |

Internally, `create_download_job()` has a little bit of internal logic
that looks at the type of the source and selects the right subclass of
`DownloadJobBase` to create and enqueue.

**TODO**: move this logic into its own method for overriding in
subclasses.

### Job control

Prior to completion, jobs can be controlled with a series of queue
method calls. Do not attempt to modify jobs by directly writing to
their fields, as this is likely to lead to unexpected results.

Any method that accepts a job argument may raise an
`UnknownJobIDException` if the job has not yet been submitted to the
queue or was not created by this queue.

#### queue.join()

This method will block until all the active jobs in the queue have
reached a terminal state (completed, errored or cancelled).

#### queue.wait_for_job(job, [timeout])

This method will block until the indicated job has reached a terminal
state (completed, errored or cancelled). If the optional timeout is
provided, the call will block for at most timeout seconds, and raise a
TimeoutError otherwise.

#### jobs = queue.list_jobs()

This will return a list of all jobs, including ones that have not yet
been enqueued and those that have completed or errored out.

#### job = queue.id_to_job(int)

This method allows you to recover a submitted job using its ID.

#### queue.prune_jobs()

Remove completed and errored jobs from the job list.

#### queue.start_job(job)

If the job was submitted with `start=False`, then it can be started
using this method.

#### queue.pause_job(job)

This will temporarily pause the job, if possible. It can later be
restarted and pick up where it left off using `queue.start_job()`.

#### queue.cancel_job(job)

This will cancel the job if possible and clean up temporary files and
other resources that it might have been using.

#### queue.start_all_jobs(), queue.pause_all_jobs(), queue.cancel_all_jobs()

This will start/pause/cancel all jobs that have been submitted to the
queue and have not yet reached a terminal state.

***

## This Meta be Good: Model Metadata Storage

The modules found under `invokeai.backend.model_manager.metadata`
provide a straightforward API for fetching model metadatda from online
repositories. Currently only HuggingFace is supported. However, the
modules are easily extended for additional repos, provided that they
have defined APIs for metadata access.

Metadata comprises any descriptive information that is not essential
for getting the model to run. For example "author" is metadata, while
"type", "base" and "format" are not. The latter fields are part of the
model's config, as defined in `invokeai.backend.model_manager.config`.

### Example Usage

```
from invokeai.backend.model_manager.metadata import (
   AnyModelRepoMetadata,
)
# to access the initialized sql database
from invokeai.app.api.dependencies import ApiDependencies

hf = HuggingFaceMetadataFetch()

# fetch the metadata
model_metadata = hf.from_id("<repo_id>")

assert isinstance(model_metadata, HuggingFaceMetadata)
```

### Structure of the Metadata objects

There is a short class hierarchy of Metadata objects, all of which
descend from the Pydantic `BaseModel`.

#### `ModelMetadataBase`

This is the common base class for metadata:

| **Field Name** | **Type**        |  **Description** |
|----------------|-----------------|------------------|
| `name`        | str           | Repository's name for the model |
| `author`      | str           | Model's author |
| `tags`        | Set[str]      | Model tags |

Note that the model config record also has a `name` field. It is
intended that the config record version be locally customizable, while
the metadata version is read-only. However, enforcing this is expected
to be part of the business logic.

Descendents of the base add additional fields.

#### `HuggingFaceMetadata`

This descends from `ModelMetadataBase` and adds the following fields:

| **Field Name** | **Type**        |  **Description** |
|----------------|-----------------|------------------|
| `type`         | Literal["huggingface"]   | Used for the discriminated union of metadata classes|
| `id`           | str              | HuggingFace repo_id |
| `tag_dict`     | Dict[str, Any]   | A dictionary of tag/value pairs provided in addition to `tags` |
| `last_modified`| datetime         | Date of last commit of this model to the repo |
| `files`        | List[Path]       | List of the files in the model repo |

#### `AnyModelRepoMetadata`

This is a discriminated Union of `HuggingFaceMetadata`.

### Fetching Metadata from Online Repos

The `HuggingFaceMetadataFetch` class will
retrieve metadata from its corresponding repository and return
`AnyModelRepoMetadata` objects. Their base class
`ModelMetadataFetchBase` is an abstract class that defines two
methods: `from_url()` and `from_id()`. The former accepts the type of
model URLs that the user will try to cut and paste into the model
import form. The latter accepts a string ID in the format recognized
by the repository of choice. Both methods return an
`AnyModelRepoMetadata`.

The base class also has a class method `from_json()` which will take
the JSON representation of a `ModelMetadata` object, validate it, and
return the corresponding `AnyModelRepoMetadata` object.

When initializing one of the metadata fetching classes, you may
provide a `requests.Session` argument. This allows you to customize
the low-level HTTP fetch requests and is used, for instance, in the
testing suite to avoid hitting the internet.

The HuggingFace fetcher subclass add additional repo-specific fetching methods:

#### HuggingFaceMetadataFetch

This overrides its base class `from_json()` method to return a
`HuggingFaceMetadata` object directly.

### Metadata Storage

The `ModelConfigBase` stores this response in the `source_api_response` field
as a JSON blob.

***

## The Lowdown on the ModelLoadService

The `ModelLoadService` is responsible for loading a named model into
memory so that it can be used for inference. Despite the fact that it
does a lot under the covers, it is very straightforward to use.

An application-wide model loader is created at API initialization time
and stored in
`ApiDependencies.invoker.services.model_loader`. However, you can
create alternative instances if you wish.

### Creating a ModelLoadService object

The class is defined in
`invokeai.app.services.model_load`. It is initialized with
an InvokeAIAppConfig object, from which it gets configuration
information such as the user's desired GPU and precision, and with a
previously-created `ModelRecordServiceBase` object, from which it
loads the requested model's configuration information.

Here is a typical initialization pattern:

```
from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.model_load import ModelLoadService, ModelLoaderRegistry

config = InvokeAIAppConfig.get_config()
ram_cache = ModelCache(
 max_cache_size=config.ram_cache_size, logger=logger
)
convert_cache = ModelConvertCache(
 cache_path=config.models_convert_cache_path, max_size=config.convert_cache_size
)
loader = ModelLoadService(
 app_config=config,
 ram_cache=ram_cache,
 convert_cache=convert_cache,
 registry=ModelLoaderRegistry
)
```

### load_model(model_config, [submodel_type], [context]) -> LoadedModel

The `load_model()` method takes an `AnyModelConfig` returned by
`ModelRecordService.get_model()` and returns the corresponding loaded
model.  It loads the model into memory, gets the model ready for use,
and returns a `LoadedModel` object.

The optional second argument, `subtype` is a `SubModelType` string
enum, such as "vae". It is mandatory when used with a main model, and
is used to select which part of the main model to load.

The optional third argument, `context` can be provided by
an invocation to trigger model load event reporting. See below for
details.

The returned `LoadedModel` object contains a copy of the configuration
record returned by the model record `get_model()` method, as well as
the in-memory loaded model:

| **Attribute Name** | **Type**        |  **Description** |
|----------------|-----------------|------------------|
| `config`       | AnyModelConfig         | A copy of the model's configuration record for retrieving base type, etc. |
| `model`        | AnyModel               | The instantiated model (details below) |
| `locker`       | ModelLockerBase        | A context manager that mediates the movement of the model into VRAM |

Because the loader can return multiple model types, it is typed to
return `AnyModel`, a Union `ModelMixin`, `torch.nn.Module`,
`IAIOnnxRuntimeModel`, `IPAdapter`, `IPAdapterPlus`, and
`EmbeddingModelRaw`. `ModelMixin` is the base class of all diffusers
models, `EmbeddingModelRaw` is used for LoRA and TextualInversion
models. The others are obvious.

`LoadedModel` acts as a context manager. The context loads the model
into the execution device (e.g. VRAM on CUDA systems), locks the model
in the execution device for the duration of the context, and returns
the model. Use it like this:

```
model_info = loader.get_model_by_key('f13dd932c0c35c22dcb8d6cda4203764', SubModelType('vae'))
with model_info as vae:
 image = vae.decode(latents)[0]
```

`get_model_by_key()` may raise any of the following exceptions:

* `UnknownModelException`   -- key not in database
* `ModelNotFoundException`  -- key in database but model not found at path
* `NotImplementedException` -- the loader doesn't know how to load this type of model
  
### Emitting model loading events

When the `context` argument is passed to `load_model_*()`, it will
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
 submodel_type=submodel,
 hash=model_info.hash,
 location=str(model_info.location),
 precision=str(model_info.precision),
)
```

### Adding Model Loaders

Model loaders are small classes that inherit from the `ModelLoader`
base class. They typically implement one method `_load_model()` whose
signature is:

```
def _load_model(
    self,
    model_path: Path,
    model_variant: Optional[ModelRepoVariant] = None,
    submodel_type: Optional[SubModelType] = None,
) -> AnyModel:
```

`_load_model()` will be passed the path to the model on disk, an
optional repository variant (used by the diffusers loaders to select,
e.g.  the `fp16` variant, and an optional submodel_type for main and
onnx models.

To install a new loader, place it in
`invokeai/backend/model_manager/load/model_loaders`. Inherit from
`ModelLoader` and use the `@ModelLoaderRegistry.register()` decorator to
indicate what type of models the loader can handle.

Here is a complete example from `generic_diffusers.py`, which is able
to load several different diffusers types:

```
from pathlib import Path
from typing import Optional

from invokeai.backend.model_manager import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelRepoVariant,
    ModelType,
    SubModelType,
)
from .. import ModelLoader, ModelLoaderRegistry


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.CLIPVision, format=ModelFormat.Diffusers)
@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.T2IAdapter, format=ModelFormat.Diffusers)
class GenericDiffusersLoader(ModelLoader):
    """Class to load simple diffusers models."""

    def _load_model(
        self,
        model_path: Path,
        model_variant: Optional[ModelRepoVariant] = None,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        model_class = self._get_hf_load_class(model_path)
        if submodel_type is not None:
            raise Exception(f"There are no submodels in models of type {model_class}")
        variant = model_variant.value if model_variant else None
        result: AnyModel = model_class.from_pretrained(model_path, torch_dtype=self._torch_dtype, variant=variant)  # type: ignore
        return result
```

Note that a loader can register itself to handle several different
model types. An exception will be raised if more than one loader tries
to register the same model type.

#### Conversion

Some models require conversion to diffusers format before they can be
loaded. These loaders should override two additional methods:

```
_needs_conversion(self, config: AnyModelConfig, model_path: Path, dest_path: Path) -> bool
_convert_model(self, config: AnyModelConfig, model_path: Path, output_path: Path) -> Path:
```

The first method accepts the model configuration, the path to where
the unmodified model is currently installed, and a proposed
destination for the converted model. This method returns True if the
model needs to be converted. It typically does this by comparing the
last modification time of the original model file to the modification
time of the converted model. In some cases you will also want to check
the modification date of the configuration record, in the event that
the user has changed something like the scheduler prediction type that
will require the model to be re-converted. See `controlnet.py` for an
example of this logic.

The second method accepts the model configuration, the path to the
original model on disk, and the desired output path for the converted
model. It does whatever it needs to do to get the model into diffusers
format, and returns the Path of the resulting model. (The path should
ordinarily be the same as `output_path`.)

## The ModelManagerService object

For convenience, the API provides a `ModelManagerService` object which
gives a single point of access to the major model manager
services. This object is created at initialization time and can be
found in the global `ApiDependencies.invoker.services.model_manager`
object, or in `context.services.model_manager` from within an
invocation.

In the examples below, we have retrieved the manager using:

```
mm = ApiDependencies.invoker.services.model_manager
```

The following properties and methods will be available:

### mm.store

This retrieves the `ModelRecordService` associated with the
manager. Example:

```
configs = mm.store.get_model_by_attr(name='stable-diffusion-v1-5')
```

### mm.install

This retrieves the `ModelInstallService` associated with the manager.
Example:

```
job = mm.install.heuristic_import(`https://civitai.com/models/58390/detail-tweaker-lora-lora`)
```

### mm.load

This retrieves the `ModelLoaderService` associated with the manager. Example:

```
configs = mm.store.get_model_by_attr(name='stable-diffusion-v1-5')
assert len(configs) > 0

loaded_model = mm.load.load_model(configs[0])
```

The model manager also offers a few convenience shortcuts for loading
models:

### mm.load_model_by_config(model_config, [submodel], [context]) -> LoadedModel

Same as `mm.load.load_model()`.

### mm.load_model_by_attr(model_name, base_model, model_type, [submodel], [context]) -> LoadedModel

This accepts the combination of the model's name, type and base, which
it passes to the model record config store for retrieval. If a unique
model config is found, this method returns a `LoadedModel`. It can
raise the following exceptions:

```
UnknownModelException -- model with these attributes not known
NotImplementedException -- the loader doesn't know how to load this type of model
ValueError -- more than one model matches this combination of base/type/name
```

### mm.load_model_by_key(key, [submodel], [context]) -> LoadedModel

This method takes a model key, looks it up using the
`ModelRecordServiceBase` object in `mm.store`, and passes the returned
model configuration to `load_model_by_config()`.  It may raise a
`NotImplementedException`.

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

A application-wide `ModelRecordService` is created during API
initialization and can be retrieved within an invocation from the
`InvocationContext` object:

```
store = context.services.model_record_store
```

or from elsewhere in the code by accessing
`ApiDependencies.invoker.services.model_record_store`.

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

An application-wide model loader is created at API initialization time
and stored in
`ApiDependencies.invoker.services.model_loader`. However, you can
create alternative instances if you wish.

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
| `key`          | str                    | The model key derived from the ModelRecordService database |
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
| `source`         | Any             |               | Where to download from - untyped in base class |
| `destination`    | Path            |               | Where to download to              |
| `status`         | DownloadJobStatus| Idle         | Job's status (see below)          |
| `event_handlers` | List[DownloadEventHandler]|     | Event handlers (see below)        |
| `job_started`    | float            |              | Timestamp for when the job started running |
| `job_ended`      | float            |              | Timestamp for when the job completed or errored out |
| `job_sequence`   | int              |              | A counter that is incremented each time a model is dequeued |
| `preserve_partial_downloads`| bool  | False        | Resume partial downloads when relaunched.   |
| `error`          | Exception        |              | A copy of the Exception that caused an error during download |

When you create a job, you can assign it a `priority`. If multiple
jobs are queued, the job with the lowest priority runs first. (Don't
blame me! The Unix developers came up with this convention.) 

Every job has a `source` and a `destination`. `source` is an untyped
object in the base class, but subclassses redefine it more
specifically.

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

| **Value**    |   **String Value**  | ** Description ** |
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
occurred). Note that since most InvokeAI model install operations
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

- enqueued
- running
- running
- running
- completed

There will be a single enqueued event, followed by one or more running
events, and finally one `completed`, `error` or `cancelled`
events.

It is possible for a caller to pause download temporarily, in which
case the events may look something like this:

- enqueued
- running
- running
- paused
- running
- completed

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

#### queue.change_priority(job, delta)

This will increase (positive delta) or decrease (negative delta) the
priority of the job on the queue. Lower priority jobs will be taken off
the queue and run before higher priority jobs.

Note that this cannot be used to change the job's priority once it has
begun running. However, you can still pause the job and restart it
later.

#### queue.start_all_jobs(), queue.pause_all_jobs(), queue.cancel_all_jobs()

This will start/pause/cancel all jobs that have been submitted to the
queue and have not yet reached a terminal state.

## Model installation

The `ModelInstallService` class implements the
`ModelInstallServiceBase` abstract base class, and provides a one-stop
shop for all your model install needs. It provides the following
functionality:

- Registering a model config record for a model already located on the
  local filesystem, without moving it or changing its path.
  
- Installing a model alreadiy located on the local filesystem, by
  moving it into the InvokeAI root directory under the
  `models` folder (or wherever config parameter `models_dir`
  specifies).
	
- Downloading a model from an arbitrary URL and installing it in
  `models_dir`.
  
- Special handling for Civitai model URLs which allow the user to
  paste in a model page's URL or download link. Any metadata provided
  by Civitai, such as trigger terms, are captured and placed in the
  model config record.
  
- Special handling for HuggingFace repo_ids to recursively download
  the contents of the repository, paying attention to alternative
  variants such as fp16.
  
- Probing of models to determine their type, base type and other key
  information.
  
- Interface with the InvokeAI event bus to provide status updates on
  the download, installation and registration process.
  
### Initializing the installer

A default installer is created at InvokeAI api startup time and stored
in `ApiDependencies.invoker.services.model_install_service` and can
also be retrieved from an invocation's `context` argument with
`context.services.model_install_service`.

In the event you wish to create a new installer, you may use the
following initialization pattern:

```
from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.download_manager import DownloadQueueServive
from invokeai.app.services.model_record_service import ModelRecordServiceBase

config = InvokeAI.get_config()
queue = DownloadQueueService()
store = ModelRecordServiceBase.open(config)
installer = ModelInstallService(config=config, queue=queue, store=store)
```

The full form of `ModelInstallService()` takes the following
parameters. Each parameter will default to a reasonable value, but it
is recommended that you set them explicitly as shown in the above example.

| **Argument**     | **Type**                     | **Default** | **Description**                           |
|------------------|------------------------------|-------------|-------------------------------------------|
| `config`         | InvokeAIAppConfig        | Use system-wide config  | InvokeAI app configuration object |
| `queue`          | DownloadQueueServiceBase  | Create a new download queue for internal use  | Download queue |
| `store`          | ModelRecordServiceBase  | Use config to select the database to open  | Config storage database |
| `event_bus`      | EventServiceBase        | None | An event bus to send download/install progress events to |
| `event_handlers` | List[DownloadEventHandler]  | None | Event handlers for the download queue |

Note that if `store` is not provided, then the class will use
`ModelRecordServiceBase.open(config)` to select the database to use.

Once initialized, the installer will provide the following methods:

#### install_job = installer.install_model()

The `install_model()` method is the core of the installer. The
following illustrates basic usage:

```
sources = [
	Path('/opt/models/sushi.safetensors'),   # a local safetensors file
	Path('/opt/models/sushi_diffusers/'),    # a local diffusers folder
	'runwayml/stable-diffusion-v1-5',        # a repo_id
    'runwayml/stable-diffusion-v1-5:vae',    # a subfolder within a repo_id
	'https://civitai.com/api/download/models/63006', # a civitai direct download link
	'https://civitai.com/models/8765?modelVersionId=10638',  # civitai model page
	'https://s3.amazon.com/fjacks/sd-3.safetensors', # arbitrary URL
]

for source in sources:
   install_job = installer.install_model(source)
   
source2key = installer.wait_for_installs()
for source in sources:
    model_key = source2key[source]
	print(f"{source} installed as {model_key}")
```

As shown here, the `install_model()` method accepts a variety of
sources, including local safetensors files, local diffusers folders,
HuggingFace repo_ids with and without a subfolder designation,
Civitai model URLs and arbitrary URLs that point to checkpoint files
(but not to folders).

Each call to `install_model()` will return a `ModelInstallJob` job, a
subclass of `DownloadJobBase`. The install job has additional
install-specific fields described in the next section.

Each install job will run in a series of background threads using
the object's download queue. You may block until all install jobs are
completed (or errored) by calling the `wait_for_installs()` method as
shown in the code example. `wait_for_installs()` will return a `dict`
that maps the requested source to the key of the installed model. In
the case that a model fails to download or install, its value in the
dict will be None. The actual cause of the error will be reported in
the corresponding job's `error` field.

Alternatively you may install event handlers and/or listen for events
on the InvokeAI event bus in order to monitor the progress of the
requested installs.

The full list of arguments to `model_install()` is as follows:

| **Argument**     | **Type**                     | **Default** | **Description**                           |
|------------------|------------------------------|-------------|-------------------------------------------|
| `source`         | Union[str, Path, AnyHttpUrl] |             | The source of the model, Path, URL or repo_id |
| `inplace`        | bool                         | True        | Leave a local model in its current location |
| `variant`        | str                          | None        | Desired variant, such as 'fp16' or 'onnx' (HuggingFace only) |
| `subfolder`      | str                          | None        | Repository subfolder (HuggingFace only)   |
| `probe_override` | Dict[str, Any]               | None        | Override all or a portion of model's probed attributes |
| `metadata`       | ModelSourceMetadata          | None        | Provide metadata that will be added to model's config |
| `access_token`   | str                          | None        | Provide authorization information needed to download |
| `priority`       | int                          | 10          | Download queue priority for the job       |


The `inplace` field controls how local model Paths are handled. If
True (the default), then the model is simply registered in its current
location by the installer's `ModelConfigRecordService`. Otherwise, the
model will be moved into the location specified by the  `models_dir`
application configuration parameter. 

The `variant` field is used for HuggingFace repo_ids only. If
provided, the repo_id download handler will look for and download
tensors files that follow the convention for the selected variant:

- "fp16" will select files named "*model.fp16.{safetensors,bin}"
- "onnx" will select files ending with the suffix ".onnx"
- "openvino" will select files beginning with "openvino_model"

In the special case of the "fp16" variant, the installer will select
the 32-bit version of the files if the 16-bit version is unavailable.

`subfolder` is used for HuggingFace repo_ids only. If provided, the
model will be downloaded from the designated subfolder rather than the
top-level repository folder. If a subfolder is attached to the repo_id
using the format `repo_owner/repo_name:subfolder`, then the subfolder
specified by the repo_id will override the subfolder argument.

`probe_override` can be used to override all or a portion of the
attributes returned by the model prober. This can be used to overcome
cases in which automatic probing is unable to (correctly) determine
the model's attribute. The most common situation is the
`prediction_type` field for sd-2 (and rare sd-1) models. Here is an
example of how it works:

```
install_job = installer.install_model(
               source='stabilityai/stable-diffusion-2-1',
			   variant='fp16',
			   probe_override=dict(
			         prediction_type=SchedulerPredictionType('v_prediction')
	           )
	      )
```

`metadata` allows you to attach custom metadata to the installed
model. See the next section for details.

`priority` and `access_token` are passed to the download queue and
have the same effect as they do for the DownloadQueueServiceBase.

#### Monitoring the install job process

When you create an install job with `model_install()`, events will be
passed to the list of `DownloadEventHandlers` provided at installer
initialization time. Event handlers can also be added to individual
model install jobs by calling their `add_handler()` method as
described earlier for the `DownloadQueueService`.

If the `event_bus` argument was provided, events will also be
broadcast to the InvokeAI event bus. The events will appear on the bus
as a singular event type named `model_event` with a payload of
`job`. You can then retrieve the job and check its status.

** TO DO: ** consider breaking `model_event` into
`model_install_started`, `model_install_completed`, etc. The event bus
features have not yet been tested with FastAPI/websockets, and it may
turn out that the job object is not serializable.

#### Model metadata and probing

The install service has special handling for HuggingFace and Civitai
URLs that capture metadata from the source and include it in the model
configuration record. For example, fetching the Civitai model 8765
will produce a config record similar to this (using YAML
representation):

```
5abc3ef8600b6c1cc058480eaae3091e:
  path: sd-1/lora/to8contrast-1-5.safetensors
  name: to8contrast-1-5
  base_model: sd-1
  model_type: lora
  model_format: lycoris
  key: 5abc3ef8600b6c1cc058480eaae3091e
  hash: 5abc3ef8600b6c1cc058480eaae3091e
  description: 'Trigger terms: to8contrast style'
  author: theovercomer8
  license: allowCommercialUse=Sell; allowDerivatives=True; allowNoCredit=True
  source: https://civitai.com/models/8765?modelVersionId=10638
  thumbnail_url: null
  tags:
  - model
  - style
  - portraits
```

For sources that do not provide model metadata, you can attach custom
fields by providing a `metadata` argument to `model_install()` using
an initialized `ModelSourceMetadata` object (available for import from
`model_install_service.py`):

```
from invokeai.app.services.model_install_service import ModelSourceMetadata
meta = ModelSourceMetadata(
                name="my model",
				author="Sushi Chef",
				description="Highly customized model; trigger with 'sushi',"
				license="mit",
				thumbnail_url="http://s3.amazon.com/ljack/pics/sushi.png",
				tags=list('sfw', 'food')
			)
install_job = installer.install_model(
               source='sushi_chef/model3',
			   variant='fp16',
			   metadata=meta,
	      )
```

It is not currently recommended to provide custom metadata when
installing from Civitai or HuggingFace source, as the metadata
provided by the source will overwrite the fields you provide. Instead,
after the model is installed you can use
`ModelRecordService.update_model()` to change the desired fields.

** TO DO: ** Change the logic so that the caller's metadata fields take
precedence over those provided by the source.


#### Other installer methods

This section describes additional, less-frequently-used attributes and
methods provided by the installer class.

##### installer.wait_for_installs()

This is equivalent to the `DownloadQueue` `join()` method. It will
block until all the active jobs in the install queue have reached a
terminal state (completed, errored or cancelled).

##### installer.queue, installer.store, installer.config

These attributes provide access to the `DownloadQueueServiceBase`,
`ModelConfigRecordServiceBase`, and `InvokeAIAppConfig` objects that
the installer uses.

For example, to temporarily pause all pending installations, you can
do this:

```
installer.queue.pause_all_jobs()
```
##### key = installer.register_path(model_path, overrides), key = installer.install_path(model_path, overrides)

These methods bypass the download queue and directly register or
install the model at the indicated path, returning the unique ID for
the installed model.

Both methods accept a Path object corresponding to a checkpoint or
diffusers folder, and an optional dict of attributes to use to
override the values derived from model probing.

The difference between `register_path()` and `install_path()` is that
the former will not move the model from its current position, while
the latter will move it into the `models_dir` hierarchy.

##### installer.unregister(key)

This will remove the model config record for the model at key, and is
equivalent to `installer.store.unregister(key)`

##### installer.delete(key)

This is similar to `unregister()` but has the additional effect of
deleting the underlying model file(s) -- even if they were outside the
`models_dir` directory!

##### installer.conditionally_delete(key)

This method will call `unregister()` if the model identified by `key`
is outside the `models_dir` hierarchy, and call `delete()` if the
model is inside.

#### List[str]=installer.scan_directory(scan_dir: Path, install: bool)

This method will recursively scan the directory indicated in
`scan_dir` for new models and either install them in the models
directory or register them in place, depending on the setting of
`install` (default False).

The return value is the list of keys of the new installed/registered
models.

#### installer.scan_models_directory()

This method scans the models directory for new models and registers
them in place. Models that are present in the
`ModelConfigRecordService` database whose paths are not found will be
unregistered.

#### installer.sync_to_config()

This method synchronizes models in the models directory and autoimport
directory to those in the `ModelConfigRecordService` database. New
models are registered and orphan models are unregistered.

#### hash=installer.hash(model_path)

This method is calls the fasthash algorithm on a model's Path
(either a file or a folder) to generate a unique ID based on the
contents of the model.

##### installer.start(invoker)

The `start` method is called by the API intialization routines when
the API starts up. Its effect is to call `sync_to_config()` to
synchronize the model record store database with what's currently on
disk.

This method should not ordinarily be called manually.

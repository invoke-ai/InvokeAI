# The InvokeAI Download Queue

The DownloadQueueService provides a multithreaded parallel download
queue for arbitrary URLs, with queue prioritization, event handling,
and restart capabilities.

## Simple Example

```
from invokeai.app.services.download import DownloadQueueService, TqdmProgress

download_queue = DownloadQueueService()
for url in ['https://github.com/invoke-ai/InvokeAI/blob/main/invokeai/assets/a-painting-of-a-fire.png?raw=true',
            'https://github.com/invoke-ai/InvokeAI/blob/main/invokeai/assets/birdhouse.png?raw=true',
            'https://github.com/invoke-ai/InvokeAI/blob/main/invokeai/assets/missing.png',
            'https://civitai.com/api/download/models/152309?type=Model&format=SafeTensor',
            ]:

    # urls start downloading as soon as download() is called
    download_queue.download(source=url,
                            dest='/tmp/downloads',
                            on_progress=TqdmProgress().update
                            )

download_queue.join()  # wait for all downloads to finish
for job in download_queue.list_jobs():
    print(job.model_dump_json(exclude_none=True, indent=4),"\n")
```

Output:

```
{
    "source": "https://github.com/invoke-ai/InvokeAI/blob/main/invokeai/assets/a-painting-of-a-fire.png?raw=true",
    "dest": "/tmp/downloads",
    "id": 0,
    "priority": 10,
    "status": "completed",
    "download_path": "/tmp/downloads/a-painting-of-a-fire.png",
    "job_started": "2023-12-04T05:34:41.742174",
    "job_ended": "2023-12-04T05:34:42.592035",
    "bytes": 666734,
    "total_bytes": 666734
} 

{
    "source": "https://github.com/invoke-ai/InvokeAI/blob/main/invokeai/assets/birdhouse.png?raw=true",
    "dest": "/tmp/downloads",
    "id": 1,
    "priority": 10,
    "status": "completed",
    "download_path": "/tmp/downloads/birdhouse.png",
    "job_started": "2023-12-04T05:34:41.741975",
    "job_ended": "2023-12-04T05:34:42.652841",
    "bytes": 774949,
    "total_bytes": 774949
}

{
    "source": "https://github.com/invoke-ai/InvokeAI/blob/main/invokeai/assets/missing.png",
    "dest": "/tmp/downloads",
    "id": 2,
    "priority": 10,
    "status": "error",
    "job_started": "2023-12-04T05:34:41.742079",
    "job_ended": "2023-12-04T05:34:42.147625",
    "bytes": 0,
    "total_bytes": 0,
    "error_type": "HTTPError(Not Found)",
    "error": "Traceback (most recent call last):\n  File \"/home/lstein/Projects/InvokeAI/invokeai/app/services/download/download_default.py\", line 182, in _download_next_item\n    self._do_download(job)\n  File \"/home/lstein/Projects/InvokeAI/invokeai/app/services/download/download_default.py\", line 206, in _do_download\n    raise HTTPError(resp.reason)\nrequests.exceptions.HTTPError: Not Found\n"
}

{
    "source": "https://civitai.com/api/download/models/152309?type=Model&format=SafeTensor",
    "dest": "/tmp/downloads",
    "id": 3,
    "priority": 10,
    "status": "completed",
    "download_path": "/tmp/downloads/xl_more_art-full_v1.safetensors",
    "job_started": "2023-12-04T05:34:42.147645",
    "job_ended": "2023-12-04T05:34:43.735990",
    "bytes": 719020768,
    "total_bytes": 719020768
} 
```

##  The API

The default download queue is `DownloadQueueService`, an
implementation of ABC `DownloadQueueServiceBase`. It juggles multiple
background download requests and provides facilities for interrogating
and cancelling the requests. Access to a current or past download task
is mediated via `DownloadJob` objects which report the current status
of a job request

### The Queue Object

A default download queue is located in
`ApiDependencies.invoker.services.download_queue`. However, you can
create additional instances if you need to isolate your queue from the
main one.

```
queue = DownloadQueueService(event_bus=events)
```

`DownloadQueueService()` takes three optional arguments:

| **Argument** | **Type**          |  **Default**  | **Description** |
|----------------|-----------------|---------------|-----------------|
| `max_parallel_dl`  | int                         | 5    | Maximum number of simultaneous downloads allowed |
| `event_bus` | EventServiceBase   | None | System-wide FastAPI event bus for reporting download events |
| `requests_session` | requests.sessions.Session   | None | An alternative requests Session object to use for the download |

`max_parallel_dl` specifies how many download jobs are allowed to run
simultaneously. Each will run in a different thread of execution.

`event_bus` is an EventServiceBase, typically the one created at
InvokeAI startup. If present, download events are periodically emitted
on this bus to allow clients to follow download progress.

`requests_session` is a url library requests Session object. It is
used for testing.

### The Job object

The queue operates on a series of download job objects. These objects
specify the source and destination of the download, and keep track of
the progress of the download.

Two job types are defined. `DownloadJob` and
`MultiFileDownloadJob`. The former is a pydantic object with the
following fields:

| **Field**      | **Type**        |  **Default**  | **Description** |
|----------------|-----------------|---------------|-----------------|
| _Fields passed in at job creation time_                               |
| `source`         | AnyHttpUrl      |               | Where to download from |
| `dest`           | Path            |               | Where to download to              |
| `access_token`   | str             |               | [optional] string containing authentication token for access |
| `on_start`       | Callable        |               | [optional] callback when the download starts |
| `on_progress` | Callable | | [optional] callback called at intervals during download progress |
| `on_complete`    | Callable        |               | [optional] callback called after successful download completion |
| `on_error`       | Callable        |               | [optional] callback called after an error occurs  |
| `id`             | int             | auto assigned | Job ID, an integer >= 0           |
| `priority`       | int             | 10            | Job priority. Lower priorities run before higher priorities |
|                                                                                                        |
| _Fields updated over the course of the download task_
| `status`         | DownloadJobStatus|              | Status code                                |
| `download_path`  | Path |              | Path to the location of the downloaded file |
| `job_started`    | float            |              | Timestamp for when the job started running |
| `job_ended`      | float            |              | Timestamp for when the job completed or errored out |
| `job_sequence`   | int              |              | A counter that is incremented each time a model is dequeued |
| `bytes`          | int              | 0            | Bytes downloaded so far   |
| `total_bytes`    | int              | 0            | Total size of the file at the remote site  |
| `error_type`     | str              |              | String version of the exception that caused an error during download |
| `error`          | str              |              | String version of the traceback associated with an error |
| `cancelled`      | bool             | False        | Set to true if the job was cancelled by the caller|

When you create a job, you can assign it a `priority`. If multiple
jobs are queued, the job with the lowest priority runs first.

Every job has a `source` and a `dest`. `source` is a pydantic.networks AnyHttpUrl object.
The `dest` is a path on the local filesystem that specifies the
destination for the downloaded object. Its semantics are
described below.

When the job is submitted, it is assigned a numeric `id`. The id can
then be used to fetch the job object from the queue.

The `status` field is updated by the queue to indicate where the job
is in its lifecycle. Values are defined in the string enum
`DownloadJobStatus`, a symbol available from
`invokeai.app.services.download_manager`. Possible values are:

| **Value**    |   **String Value**  | ** Description ** |
|--------------|---------------------|-------------------|
| `WAITING`      | waiting           | Job is on the queue but not yet running|
| `RUNNING`      | running           | The download is started                |
| `COMPLETED`    | completed         | Job has finished its work without an error |
| `ERROR`        | error             | Job encountered an error and will not run again|

`job_started` and `job_ended` indicate when the job
was started (using a python timestamp) and when it completed.

In case of an error, the job's status will be set to `DownloadJobStatus.ERROR`, the text of the
Exception that caused the error will be placed in the `error_type`
field and the traceback that led to the error will be in `error`.

A cancelled job will have status `DownloadJobStatus.ERROR` and an
`error_type` field of "DownloadJobCancelledException". In addition,
the job's `cancelled` property will be set to True.

The `MultiFileDownloadJob` is used for diffusers model downloads,
which contain multiple files and directories under a common root:

| **Field**      | **Type**        |  **Default**  | **Description** |
|----------------|-----------------|---------------|-----------------|
| _Fields passed in at job creation time_                               |
| `download_parts` | Set[DownloadJob]|               | Component download jobs |
| `dest`           | Path            |               | Where to download to              |
| `on_start`       | Callable        |               | [optional] callback when the download starts |
| `on_progress` | Callable | | [optional] callback called at intervals during download progress |
| `on_complete`    | Callable        |               | [optional] callback called after successful download completion |
| `on_error`       | Callable        |               | [optional] callback called after an error occurs  |
| `id`             | int             | auto assigned | Job ID, an integer >= 0           |
| _Fields updated over the course of the download task_
| `status`         | DownloadJobStatus|              | Status code                                |
| `download_path`  | Path |              | Path to the root of the downloaded files |
| `bytes`          | int              | 0            | Bytes downloaded so far   |
| `total_bytes`    | int              | 0            | Total size of the file at the remote site  |
| `error_type`     | str              |              | String version of the exception that caused an error during download |
| `error`          | str              |              | String version of the traceback associated with an error |
| `cancelled`      | bool             | False        | Set to true if the job was cancelled by the caller|

Note that the MultiFileDownloadJob does not support the `priority`,
`job_started`, `job_ended` or `content_type` attributes. You can get
these from the individual download jobs in `download_parts`.


### Callbacks

Download jobs can be associated with a series of callbacks, each with
the signature `Callable[["DownloadJob"], None]`. The callbacks are assigned
using optional arguments `on_start`, `on_progress`, `on_complete` and
`on_error`. When the corresponding event occurs, the callback wil be
invoked and passed the job. The callback will be run in a `try:`
context in the same thread as the download job. Any exceptions that
occur during execution of the callback will be caught and converted
into a log error message, thereby allowing the download to continue.

#### `TqdmProgress`

The `invokeai.app.services.download.download_default` module defines a
class named `TqdmProgress` which can be used as an `on_progress`
handler to display a completion bar in the console. Use as follows:

```
from invokeai.app.services.download import TqdmProgress

download_queue.download(source='http://some.server.somewhere/some_file',
                        dest='/tmp/downloads',
                        on_progress=TqdmProgress().update
                        )

```

### Events

If the queue was initialized with the InvokeAI event bus (the case
when using `ApiDependencies.invoker.services.download_queue`), then
download events will also be issued on the bus. The events are:

* `download_started` -- This is issued when a job is taken off the
queue and a request is made to the remote server for the URL headers, but before any data
has been downloaded. The event payload will contain the keys `source`
and `download_path`. The latter contains the path that the URL will be
downloaded to.

* `download_progress -- This is issued periodically as the download
runs. The payload contains the keys `source`, `download_path`,
`current_bytes` and `total_bytes`. The latter two fields can be
used to display the percent complete.

* `download_complete` -- This is issued when the download completes
successfully. The payload contains the keys `source`, `download_path`
and `total_bytes`.

* `download_error` -- This is issued when the download stops because
of an error condition. The payload contains the fields `error_type`
and `error`. The former is the text representation of the exception,
and the latter is a traceback showing where the error occurred.

### Job control

To create a job call the queue's `download()` method. You can list all
jobs using `list_jobs()`, fetch a single job by its with
`id_to_job()`, cancel a running job with `cancel_job()`, cancel all
running jobs with `cancel_all_jobs()`, and wait for all jobs to finish
with `join()`.

#### job = queue.download(source, dest, priority, access_token, on_start, on_progress, on_complete, on_cancelled, on_error)

Create a new download job and put it on the queue, returning the
DownloadJob object.

#### multifile_job = queue.multifile_download(parts, dest, access_token, on_start, on_progress, on_complete, on_cancelled, on_error)

This is similar to download(), but instead of taking a single source,
it accepts a `parts` argument consisting of a list of
`RemoteModelFile` objects. Each part corresponds to a URL/Path pair,
where the URL is the location of the remote file, and the Path is the
destination.

`RemoteModelFile` can be imported from `invokeai.backend.model_manager.metadata`, and
consists of a url/path pair. Note that the path *must* be relative.

The method returns a `MultiFileDownloadJob`.


```
from invokeai.backend.model_manager.metadata import RemoteModelFile
remote_file_1 = RemoteModelFile(url='http://www.foo.bar/my/pytorch_model.safetensors'',
                                path='my_model/textencoder/pytorch_model.safetensors'
			 			  )
remote_file_2 = RemoteModelFile(url='http://www.bar.baz/vae.ckpt',
                                path='my_model/vae/diffusers_model.safetensors'
			 			  )
job = queue.multifile_download(parts=[remote_file_1, remote_file_2],
                               dest='/tmp/downloads',
                               on_progress=TqdmProgress().update)
queue.wait_for_job(job)
print(f"The files were downloaded to {job.download_path}")
```

#### jobs = queue.list_jobs()

Return a list of all active and inactive `DownloadJob`s.

#### job = queue.id_to_job(id)

Return the job corresponding to given ID.

Return a list of all active and inactive `DownloadJob`s.

#### queue.prune_jobs()

Remove inactive (complete or errored) jobs from the listing returned
by `list_jobs()`.

#### queue.join()

Block until all pending jobs have run to completion or errored out.


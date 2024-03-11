---
title: Configuration
---

# :material-tune-variant: InvokeAI Configuration

## Intro

InvokeAI has numerous runtime settings which can be used to adjust
many aspects of its operations, including the location of files and
directories, memory usage, and performance. These settings can be
viewed and customized in several ways:

1. By editing settings in the `invokeai.yaml` file.
2. By setting environment variables.
3. On the command-line, when InvokeAI is launched.

In addition, the most commonly changed settings are accessible
graphically via the `invokeai-configure` script.

### How the Configuration System Works

When InvokeAI is launched, the very first thing it needs to do is to
find its "root" directory, which contains its configuration files,
installed models, its database of images, and the folder(s) of
generated images themselves. In this document, the root directory will
be referred to as ROOT.

#### Finding the Root Directory

To find its root directory, InvokeAI uses the following recipe:

1. It first looks for the argument `--root <path>` on the command line
   it was launched from, and uses the indicated path if present.

2. Next it looks for the environment variable INVOKEAI_ROOT, and uses
   the directory path found there if present.

3. If neither of these are present, then InvokeAI looks for the
   folder containing the `.venv` Python virtual environment directory for
   the currently active environment. This directory is checked for files
   expected inside the InvokeAI root before it is used.

4. Finally, InvokeAI looks for a directory in the current user's home
   directory named `invokeai`.

#### Reading the InvokeAI Configuration File

Once the root directory has been located, InvokeAI looks for a file
named `ROOT/invokeai.yaml`, and if present reads configuration values
from it. The top of this file looks like this:

```
InvokeAI:
  Web Server:
    host: localhost
    port: 9090
    allow_origins: []
    allow_credentials: true
    allow_methods:
    - '*'
    allow_headers:
    - '*'
  Features:
    esrgan: true
    internet_available: true
    log_tokenization: false
    patchmatch: true
    restore: true
...
```

This lines in this file are used to establish default values for
Invoke's settings. In the above fragment, the Web Server's listening
port is set to 9090 by the `port` setting.

You can edit this file with a text editor such as "Notepad" (do not
use Word or any other word processor). When editing, be careful to
maintain the indentation, and do not add extraneous text, as syntax
errors will prevent InvokeAI from launching. A basic guide to the
format of YAML files can be found
[here](https://circleci.com/blog/what-is-yaml-a-beginner-s-guide/).

You can fix a broken `invokeai.yaml` by deleting it and running the
configuration script again -- option [6] in the launcher, "Re-run the
configure script".

#### Reading Environment Variables

Next InvokeAI looks for defined environment variables in the format
`INVOKEAI_<setting_name>`, for example `INVOKEAI_port`. Environment
variable values take precedence over configuration file variables. On
a Macintosh system, for example, you could change the port that the
web server listens on by setting the environment variable this way:

```
export INVOKEAI_port=8000
invokeai-web
```

Please check out these
[Macintosh](https://phoenixnap.com/kb/set-environment-variable-mac)
and
[Windows](https://phoenixnap.com/kb/windows-set-environment-variable)
guides for setting temporary and permanent environment variables.

#### Reading the Command Line

Lastly, InvokeAI takes settings from the command line, which override
everything else. The command-line settings have the same name as the
corresponding configuration file settings, preceded by a `--`, for
example `--port 8000`.

If you are using the launcher (`invoke.sh` or `invoke.bat`) to launch
InvokeAI, then just pass the command-line arguments to the launcher:

```
invoke.bat --port 8000 --host 0.0.0.0
```

The arguments will be applied when you select the web server option
(and the other options as well).

If, on the other hand, you prefer to launch InvokeAI directly from the
command line, you would first activate the virtual environment (known
as the "developer's console" in the launcher), and run `invokeai-web`:

```
> C:\Users\Fred\invokeai\.venv\scripts\activate
(.venv) > invokeai-web --port 8000 --host 0.0.0.0
```

You can get a listing and brief instructions for each of the
command-line options by giving the `--help` argument:

```
(.venv) > invokeai-web --help
usage: InvokeAI [-h] [--host HOST] [--port PORT] [--allow_origins [ALLOW_ORIGINS ...]] [--allow_credentials | --no-allow_credentials] [--allow_methods [ALLOW_METHODS ...]]
                [--allow_headers [ALLOW_HEADERS ...]] [--esrgan | --no-esrgan] [--internet_available | --no-internet_available] [--log_tokenization | --no-log_tokenization]
                [--patchmatch | --no-patchmatch] [--restore | --no-restore]
                [--always_use_cpu | --no-always_use_cpu] [--free_gpu_mem | --no-free_gpu_mem] [--max_loaded_models MAX_LOADED_MODELS] [--max_cache_size MAX_CACHE_SIZE]
                [--max_vram_cache_size MAX_VRAM_CACHE_SIZE] [--gpu_mem_reserved GPU_MEM_RESERVED] [--precision {auto,float16,float32,autocast}]
                [--sequential_guidance | --no-sequential_guidance] [--xformers_enabled | --no-xformers_enabled] [--tiled_decode | --no-tiled_decode] [--root ROOT]
                [--autoimport_dir AUTOIMPORT_DIR] [--lora_dir LORA_DIR] [--embedding_dir EMBEDDING_DIR] [--controlnet_dir CONTROLNET_DIR] [--conf_path CONF_PATH]
                [--models_dir MODELS_DIR] [--legacy_conf_dir LEGACY_CONF_DIR] [--db_dir DB_DIR] [--outdir OUTDIR] [--from_file FROM_FILE]
                [--use_memory_db | --no-use_memory_db] [--model MODEL] [--log_handlers [LOG_HANDLERS ...]] [--log_format {plain,color,syslog,legacy}]
                [--log_level {debug,info,warning,error,critical}] [--version | --no-version]
```

## The Configuration Settings

The config is managed by the `InvokeAIAppConfig` class, which is a pydantic model. The below docs are autogenerated from the class.

When editing your `invokeai.yaml` file, you'll need to put settings under their appropriate group. The group for each setting is denoted in the table below.

Following the table are additional explanations for certain settings.

<!-- prettier-ignore-start -->
::: invokeai.app.services.config.config_default.InvokeAIAppConfig
    options:
        heading_level: 3
<!-- prettier-ignore-end -->

### Model Marketplace API Keys

Some model marketplaces require an API key to download models. You can provide a URL pattern and appropriate token in your `invokeai.yaml` file to provide that API key.

The pattern can be any valid regex (you may need to surround the pattern with quotes):

```yaml
InvokeAI:
  Model Install:
    remote_api_tokens:
      # Any URL containing `models.com` will automatically use `your_models_com_token`
      - url_regex: models.com
        token: your_models_com_token
      # Any URL matching this contrived regex will use `some_other_token`
      - url_regex: '^[a-z]{3}whatever.*\.com$'
        token: some_other_token
```

The provided token will be added as a `Bearer` token to the network requests to download the model files. As far as we know, this works for all model marketplaces that require authorization.

### Model Hashing

Models are hashed during installation, providing a stable identifier for models across all platforms. The default algorithm is `blake3`, with a multi-threaded implementation.

If your models are stored on a spinning hard drive, we suggest using `blake3_single`, the single-threaded implementation. The hashes are the same, but it's much faster on spinning disks.

```yaml
InvokeAI:
  Model Install:
    hashing_algorithm: blake3_single
```

Model hashing is a one-time operation, but it may take a couple minutes to hash a large model collection. You may opt out of model hashing entirely by setting the algorithm to `random`.

```yaml
InvokeAI:
  Model Install:
    hashing_algorithm: random
```

Most common algorithms are supported, like `md5`, `sha256`, and `sha512`. These are typically much, much slower than `blake3`.

### Paths

These options set the paths of various directories and files used by
InvokeAI. Relative paths are interpreted relative to the root directory, so
if root is `/home/fred/invokeai` and the path is
`autoimport/main`, then the corresponding directory will be located at
`/home/fred/invokeai/autoimport/main`.

Note that the autoimport directory will be searched recursively,
allowing you to organize the models into folders and subfolders in any
way you wish.

### Logging

Several different log handler destinations are available, and multiple destinations are supported by providing a list:

```
  log_handlers:
     - console
     - syslog=localhost
     - file=/var/log/invokeai.log
```

- `console` is the default. It prints log messages to the command-line window from which InvokeAI was launched.

- `syslog` is only available on Linux and Macintosh systems. It uses
  the operating system's "syslog" facility to write log file entries
  locally or to a remote logging machine. `syslog` offers a variety
  of configuration options:

```
  syslog=/dev/log`      - log to the /dev/log device
  syslog=localhost`     - log to the network logger running on the local machine
  syslog=localhost:512` - same as above, but using a non-standard port
  syslog=fredserver,facility=LOG_USER,socktype=SOCK_DRAM`
                        - Log to LAN-connected server "fredserver" using the facility LOG_USER and datagram packets.
```

- `http` can be used to log to a remote web server. The server must be
  properly configured to receive and act on log messages. The option
  accepts the URL to the web server, and a `method` argument
  indicating whether the message should be submitted using the GET or
  POST method.

```
 http=http://my.server/path/to/logger,method=POST
```

The `log_format` option provides several alternative formats:

- `color` - default format providing time, date and a message, using text colors to distinguish different log severities
- `plain` - same as above, but monochrome text only
- `syslog` - the log level and error message only, allowing the syslog system to attach the time and date
- `legacy` - a format similar to the one used by the legacy 2.3 InvokeAI releases.

---
title: Configuration
---

# :material-tune-variant: InvokeAI Configuration

## Intro

Runtime settings, including the location of files and
directories, memory usage, and performance, are managed via the
`invokeai.yaml` config file or environment variables. A subset
of settings may be set via commandline arguments.

Settings sources are used in this order:

- CLI args
- Environment variables
- `invokeai.yaml` settings
- Fallback: defaults

### InvokeAI Root Directory

On startup, InvokeAI searches for its "root" directory. This is the directory
that contains models, images, the database, and so on. It also contains
a configuration file called `invokeai.yaml`.

InvokeAI searches for the root directory in this order:

1. The `--root <path>` CLI arg.
2. The environment variable INVOKEAI_ROOT.
3. The directory containing the currently active virtual environment.
4. Fallback: a directory in the current user's home directory named `invokeai`.

### InvokeAI Configuration File

Inside the root directory, we read settings from the `invokeai.yaml` file.

It has two sections - one for internal use and one for user settings:

```yaml
# Internal metadata - do not edit:
schema_version: 4

# Put user settings here - see https://invoke-ai.github.io/InvokeAI/features/CONFIGURATION/:
host: 0.0.0.0 # serve the app on your local network
models_dir: D:\invokeai\models # store models on an external drive
precision: float16 # always use fp16 precision
```

The settings in this file will override the defaults. You only need
to change this file if the default for a particular setting doesn't
work for you.

Some settings, like [Model Marketplace API Keys], require the YAML
to be formatted correctly. Here is a [basic guide to YAML files].

You can fix a broken `invokeai.yaml` by deleting it and running the
configuration script again -- option [6] in the launcher, "Re-run the
configure script".

#### Custom Config File Location

You can use any config file with the `--config` CLI arg. Pass in the path to the `invokeai.yaml` file you want to use.

Note that environment variables will trump any settings in the config file.

### Environment Variables

All settings may be set via environment variables by prefixing `INVOKEAI_`
to the variable name. For example, `INVOKEAI_HOST` would set the `host`
setting.

For non-primitive values, pass a JSON-encoded string:

```sh
export INVOKEAI_REMOTE_API_TOKENS='[{"url_regex":"modelmarketplace", "token": "12345"}]'
```

We suggest using `invokeai.yaml`, as it is more user-friendly.

### CLI Args

A subset of settings may be specified using CLI args:

- `--root`: specify the root directory
- `--config`: override the default `invokeai.yaml` file location

### All Settings

Following the table are additional explanations for certain settings.

<!-- prettier-ignore-start -->
::: invokeai.app.services.config.config_default.InvokeAIAppConfig
    options:
        heading_level: 4
        members: false
        show_docstring_description: false
        group_by_category: true
        show_category_heading: false
<!-- prettier-ignore-end -->

#### Model Marketplace API Keys

Some model marketplaces require an API key to download models. You can provide a URL pattern and appropriate token in your `invokeai.yaml` file to provide that API key.

The pattern can be any valid regex (you may need to surround the pattern with quotes):

```yaml
remote_api_tokens:
  # Any URL containing `models.com` will automatically use `your_models_com_token`
  - url_regex: models.com
    token: your_models_com_token
  # Any URL matching this contrived regex will use `some_other_token`
  - url_regex: '^[a-z]{3}whatever.*\.com$'
    token: some_other_token
```

The provided token will be added as a `Bearer` token to the network requests to download the model files. As far as we know, this works for all model marketplaces that require authorization.

#### Model Hashing

Models are hashed during installation, providing a stable identifier for models across all platforms. Hashing is a one-time operation.

```yaml
hashing_algorithm: blake3_single
```

You might want to change this setting, depending on your system:

- `blake3_single` (default): Single-threaded - best for spinning HDDs, still OK for SSDs
- `blake3`: Parallelized, memory-mapped implementation - best for SSDs, terrible for spinning disks
- `random`: Skip hashing entirely - fastest but of course no hash

During the first startup after upgrading to v4, all of your models will be hashed. This can take a few minutes.

Most common algorithms are supported, like `md5`, `sha256`, and `sha512`. These are typically much, much slower than `blake3`.

#### Path Settings

These options set the paths of various directories and files used by
InvokeAI. Relative paths are interpreted relative to the root directory, so
if root is `/home/fred/invokeai` and the path is
`autoimport/main`, then the corresponding directory will be located at
`/home/fred/invokeai/autoimport/main`.

Note that the autoimport directory will be searched recursively,
allowing you to organize the models into folders and subfolders in any
way you wish.

#### Logging

Several different log handler destinations are available, and multiple destinations are supported by providing a list:

```yaml
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

[basic guide to yaml files]: https://circleci.com/blog/what-is-yaml-a-beginner-s-guide/
[Model Marketplace API Keys]: #model-marketplace-api-keys

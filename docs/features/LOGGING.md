---
title: Controlling Logging
---

# :material-image-off: Controlling Logging

## Controlling How InvokeAI Logs Status Messages

InvokeAI logs status messages using a configurable logging system. You
can log to the terminal window, to a designated file on the local
machine, to the syslog facility on a Linux or Mac, or to a properly
configured web server. You can configure several logs at the same
time, and control the level of message logged and the logging format
(to a limited extent).

Three command-line options control logging:

### `--log_handlers <handler1> <handler2> ...`

This option activates one or more log handlers. Options are "console",
"file", "syslog" and "http". To specify more than one, separate them
by spaces:

```bash
invokeai-web --log_handlers console syslog=/dev/log file=C:\Users\fred\invokeai.log
```

The format of these options is described below.

### `--log_format {plain|color|legacy|syslog}`

This controls the format of log messages written to the console. Only
the "console" log handler is currently affected by this setting.

* "plain" provides formatted messages like this:

```bash

[2023-05-24 23:18:2[2023-05-24 23:18:50,352]::[InvokeAI]::DEBUG --> this is a debug message
[2023-05-24 23:18:50,352]::[InvokeAI]::INFO --> this is an informational messages
[2023-05-24 23:18:50,352]::[InvokeAI]::WARNING --> this is a warning
[2023-05-24 23:18:50,352]::[InvokeAI]::ERROR --> this is an error
[2023-05-24 23:18:50,352]::[InvokeAI]::CRITICAL --> this is a critical error
```

* "color" produces similar output, but the text will be color coded to
indicate the severity of the message.

* "legacy" produces output similar to InvokeAI versions 2.3 and earlier:

```bash
### this is a critical error
*** this is an error
** this is a warning
>> this is an informational messages
   | this is a debug message
```

* "syslog" produces messages suitable for syslog entries:

```bash
InvokeAI [2691178] <CRITICAL> this is a critical error
InvokeAI [2691178] <ERROR> this is an error
InvokeAI [2691178] <WARNING> this is a warning
InvokeAI [2691178] <INFO> this is an informational messages
InvokeAI [2691178] <DEBUG> this is a debug message
```

(note that the date, time and hostname will be added by the syslog
system)

### `--log_level {debug|info|warning|error|critical}`

Providing this command-line option will cause only messages at the
specified level or above to be emitted.

## Console logging

When "console" is provided to `--log_handlers`, messages will be
written to the command line window in which InvokeAI was launched. By
default, the color formatter will be used unless overridden by
`--log_format`.

## File logging

When "file" is provided to `--log_handlers`, entries will be written
to the file indicated in the path argument. By default, the "plain"
format will be used:

```bash
invokeai-web --log_handlers file=/var/log/invokeai.log
```

## Syslog logging

When "syslog" is requested, entries will be sent to the syslog
system. There are a variety of ways to control where the log message
is sent:

* Send to the local machine using the `/dev/log` socket:

```
invokeai-web --log_handlers syslog=/dev/log
```

* Send to the local machine using a UDP message:

```
invokeai-web --log_handlers syslog=localhost
```

* Send to the local machine using a UDP message on a nonstandard
  port:

```
invokeai-web --log_handlers syslog=localhost:512
```

* Send to a remote machine named "loghost" on the local LAN using
  facility LOG_USER and UDP packets:

```
invokeai-web --log_handlers syslog=loghost,facility=LOG_USER,socktype=SOCK_DGRAM
```

This can be abbreviated `syslog=loghost`, as LOG_USER and SOCK_DGRAM
are defaults.

* Send to a remote machine named "loghost" using the facility LOCAL0
  and using a TCP socket:

```
invokeai-web --log_handlers syslog=loghost,facility=LOG_LOCAL0,socktype=SOCK_STREAM
```

If no arguments are specified (just a bare "syslog"), then the logging
system will look for a UNIX socket named `/dev/log`, and if not found
try to send a UDP message to `localhost`. The Macintosh OS used to
support logging to a socket named `/var/run/syslog`, but this feature
has since been disabled.

## Web logging

If you have access to a web server that is configured to log messages
when a particular URL is requested, you can log using the "http"
method:

```
invokeai-web --log_handlers http=http://my.server/path/to/logger,method=POST
```

The optional [,method=] part can be used to specify whether the URL
accepts GET (default) or POST messages.

Currently password authentication and SSL are not supported.

## Using the configuration file

You can set and forget logging options by adding a "Logging" section
to `invokeai.yaml`:

```
InvokeAI:
  [... other settings...]
  Logging:
    log_handlers:
       - console
       - syslog=/dev/log
    log_level: info
    log_format: color
```

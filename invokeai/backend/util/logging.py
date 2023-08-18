# Copyright (c) 2023 Lincoln D. Stein and The InvokeAI Development Team

"""
invokeai.backend.util.logging

Logging class for InvokeAI that produces console messages

Usage:

from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.getLogger(name='InvokeAI') // Initialization
(or)
logger = InvokeAILogger.getLogger(__name__) // To use the filename
logger.configure()

logger.critical('this is critical') // Critical Message
logger.error('this is an error') // Error Message
logger.warning('this is a warning') // Warning Message
logger.info('this is info') // Info Message
logger.debug('this is debugging') // Debug Message

Console messages:
    [12-05-2023 20]::[InvokeAI]::CRITICAL --> This is an info message [In Bold Red]
    [12-05-2023 20]::[InvokeAI]::ERROR --> This is an info message [In Red]
    [12-05-2023 20]::[InvokeAI]::WARNING --> This is an info message [In Yellow]
    [12-05-2023 20]::[InvokeAI]::INFO --> This is an info message [In Grey]
    [12-05-2023 20]::[InvokeAI]::DEBUG --> This is an info message [In Grey]

Alternate Method (in this case the logger name will be set to InvokeAI):
import invokeai.backend.util.logging as IAILogger
IAILogger.debug('this is a debugging message')

## Configuration

The default configuration will print to stderr on the console. To add
additional logging handlers, call getLogger with an initialized InvokeAIAppConfig
object:


 config = InvokeAIAppConfig.get_config()
 config.parse_args()
 logger = InvokeAILogger.getLogger(config=config)

### Three command-line options control logging:

`--log_handlers <handler1> <handler2> ...`

This option activates one or more log handlers. Options are "console", "file", "syslog" and "http". To specify more than one, separate them by spaces:

```
invokeai-web --log_handlers console syslog=/dev/log file=C:\\Users\\fred\\invokeai.log
```

The format of these options is described below.

### `--log_format {plain|color|legacy|syslog}`

This controls the format of log messages written to the console. Only the "console" log handler is currently affected by this setting.

* "plain" provides formatted messages like this:

```bash

[2023-05-24 23:18:2[2023-05-24 23:18:50,352]::[InvokeAI]::DEBUG --> this is a debug message
[2023-05-24 23:18:50,352]::[InvokeAI]::INFO --> this is an informational messages
[2023-05-24 23:18:50,352]::[InvokeAI]::WARNING --> this is a warning
[2023-05-24 23:18:50,352]::[InvokeAI]::ERROR --> this is an error
[2023-05-24 23:18:50,352]::[InvokeAI]::CRITICAL --> this is a critical error
```

* "color" produces similar output, but the text will be color coded to indicate the severity of the message.

* "legacy" produces output similar to InvokeAI versions 2.3 and earlier:

```
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

(note that the date, time and hostname will be added by the syslog system)

### `--log_level {debug|info|warning|error|critical}`

Providing this command-line option will cause only messages at the specified level or above to be emitted.

## Console logging

When "console" is provided to `--log_handlers`, messages will be written to the command line window in which InvokeAI was launched. By default, the color formatter will be used unless overridden by `--log_format`.

## File logging

When "file" is provided to `--log_handlers`, entries will be written to the file indicated in the path argument. By default, the "plain" format will be used:

```bash
invokeai-web --log_handlers file=/var/log/invokeai.log
```

## Syslog logging

When "syslog" is requested, entries will be sent to the syslog system. There are a variety of ways to control where the log message is sent:

* Send to the local machine using the `/dev/log` socket:

```
invokeai-web --log_handlers syslog=/dev/log
```

* Send to the local machine using a UDP message:

```
invokeai-web --log_handlers syslog=localhost
```

* Send to the local machine using a UDP message on a nonstandard port:

```
invokeai-web --log_handlers syslog=localhost:512
```

* Send to a remote machine named "loghost" on the local LAN using  facility LOG_USER and UDP packets:

```
invokeai-web --log_handlers syslog=loghost,facility=LOG_USER,socktype=SOCK_DGRAM
```

This can be abbreviated `syslog=loghost`, as LOG_USER and SOCK_DGRAM are defaults.

* Send to a remote machine named "loghost" using the facility LOCAL0  and using a TCP socket:

```
invokeai-web --log_handlers syslog=loghost,facility=LOG_LOCAL0,socktype=SOCK_STREAM
```

If no arguments are specified (just a bare "syslog"), then the logging system will look for a UNIX socket named `/dev/log`, and if not found try to send a UDP message to `localhost`. The Macintosh OS used to support logging to a socket named `/var/run/syslog`, but this feature has since been disabled.

## Web logging

If you have access to a web server that is configured to log messages when a particular URL is requested, you can log using the "http" method:

```
invokeai-web --log_handlers http=http://my.server/path/to/logger,method=POST
```

The optional [,method=] part can be used to specify whether the URL accepts GET (default) or POST messages.

Currently password authentication and SSL are not supported.

## Using the configuration file

You can set and forget logging options by adding a "Logging" section to `invokeai.yaml`:

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
"""

import logging.handlers
import socket
import urllib.parse

from abc import abstractmethod
from pathlib import Path

from invokeai.app.services.config import InvokeAIAppConfig

try:
    import syslog

    SYSLOG_AVAILABLE = True
except ImportError:
    SYSLOG_AVAILABLE = False


# module level functions
def debug(msg, *args, **kwargs):
    InvokeAILogger.getLogger().debug(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    InvokeAILogger.getLogger().info(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    InvokeAILogger.getLogger().warning(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    InvokeAILogger.getLogger().error(msg, *args, **kwargs)


def critical(msg, *args, **kwargs):
    InvokeAILogger.getLogger().critical(msg, *args, **kwargs)


def log(level, msg, *args, **kwargs):
    InvokeAILogger.getLogger().log(level, msg, *args, **kwargs)


def disable(level=logging.CRITICAL):
    InvokeAILogger.getLogger().disable(level)


def basicConfig(**kwargs):
    InvokeAILogger.getLogger().basicConfig(**kwargs)


def getLogger(name: str = None) -> logging.Logger:
    return InvokeAILogger.getLogger(name)


_FACILITY_MAP = (
    dict(
        LOG_KERN=syslog.LOG_KERN,
        LOG_USER=syslog.LOG_USER,
        LOG_MAIL=syslog.LOG_MAIL,
        LOG_DAEMON=syslog.LOG_DAEMON,
        LOG_AUTH=syslog.LOG_AUTH,
        LOG_LPR=syslog.LOG_LPR,
        LOG_NEWS=syslog.LOG_NEWS,
        LOG_UUCP=syslog.LOG_UUCP,
        LOG_CRON=syslog.LOG_CRON,
        LOG_SYSLOG=syslog.LOG_SYSLOG,
        LOG_LOCAL0=syslog.LOG_LOCAL0,
        LOG_LOCAL1=syslog.LOG_LOCAL1,
        LOG_LOCAL2=syslog.LOG_LOCAL2,
        LOG_LOCAL3=syslog.LOG_LOCAL3,
        LOG_LOCAL4=syslog.LOG_LOCAL4,
        LOG_LOCAL5=syslog.LOG_LOCAL5,
        LOG_LOCAL6=syslog.LOG_LOCAL6,
        LOG_LOCAL7=syslog.LOG_LOCAL7,
    )
    if SYSLOG_AVAILABLE
    else dict()
)

_SOCK_MAP = dict(
    SOCK_STREAM=socket.SOCK_STREAM,
    SOCK_DGRAM=socket.SOCK_DGRAM,
)


class InvokeAIFormatter(logging.Formatter):
    """
    Base class for logging formatter

    """

    def format(self, record):
        formatter = logging.Formatter(self.log_fmt(record.levelno))
        return formatter.format(record)

    @abstractmethod
    def log_fmt(self, levelno: int) -> str:
        pass


class InvokeAISyslogFormatter(InvokeAIFormatter):
    """
    Formatting for syslog
    """

    def log_fmt(self, levelno: int) -> str:
        return "%(name)s [%(process)d] <%(levelname)s> %(message)s"


class InvokeAILegacyLogFormatter(InvokeAIFormatter):
    """
    Formatting for the InvokeAI Logger (legacy version)
    """

    FORMATS = {
        logging.DEBUG: "   | %(message)s",
        logging.INFO: ">> %(message)s",
        logging.WARNING: "** %(message)s",
        logging.ERROR: "*** %(message)s",
        logging.CRITICAL: "### %(message)s",
    }

    def log_fmt(self, levelno: int) -> str:
        return self.FORMATS.get(levelno)


class InvokeAIPlainLogFormatter(InvokeAIFormatter):
    """
    Custom Formatting for the InvokeAI Logger (plain version)
    """

    def log_fmt(self, levelno: int) -> str:
        return "[%(asctime)s]::[%(name)s]::%(levelname)s --> %(message)s"


class InvokeAIColorLogFormatter(InvokeAIFormatter):
    """
    Custom Formatting for the InvokeAI Logger
    """

    # Color Codes
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    cyan = "\x1b[36;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    # Log Format
    log_format = "[%(asctime)s]::[%(name)s]::%(levelname)s --> %(message)s"
    ## More Formatting Options: %(pathname)s, %(filename)s, %(module)s, %(lineno)d

    # Format Map
    FORMATS = {
        logging.DEBUG: cyan + log_format + reset,
        logging.INFO: grey + log_format + reset,
        logging.WARNING: yellow + log_format + reset,
        logging.ERROR: red + log_format + reset,
        logging.CRITICAL: bold_red + log_format + reset,
    }

    def log_fmt(self, levelno: int) -> str:
        return self.FORMATS.get(levelno)


LOG_FORMATTERS = {
    "plain": InvokeAIPlainLogFormatter,
    "color": InvokeAIColorLogFormatter,
    "syslog": InvokeAISyslogFormatter,
    "legacy": InvokeAILegacyLogFormatter,
}


class InvokeAILogger(object):
    loggers = dict()

    @classmethod
    def getLogger(
        cls, name: str = "InvokeAI", config: InvokeAIAppConfig = InvokeAIAppConfig.get_config()
    ) -> logging.Logger:
        if name in cls.loggers:
            logger = cls.loggers[name]
            logger.handlers.clear()
        else:
            logger = logging.getLogger(name)
        logger.setLevel(config.log_level.upper())  # yes, strings work here
        for ch in cls.getLoggers(config):
            logger.addHandler(ch)
            cls.loggers[name] = logger
        return cls.loggers[name]

    @classmethod
    def getLoggers(cls, config: InvokeAIAppConfig) -> list[logging.Handler]:
        handler_strs = config.log_handlers
        handlers = list()
        for handler in handler_strs:
            handler_name, *args = handler.split("=", 2)
            args = args[0] if len(args) > 0 else None

            # console and file get the fancy formatter.
            # syslog gets a simple one
            # http gets no custom formatter
            formatter = LOG_FORMATTERS[config.log_format]
            if handler_name == "console":
                ch = logging.StreamHandler()
                ch.setFormatter(formatter())
                handlers.append(ch)

            elif handler_name == "syslog":
                ch = cls._parse_syslog_args(args)
                handlers.append(ch)

            elif handler_name == "file":
                ch = cls._parse_file_args(args)
                ch.setFormatter(formatter())
                handlers.append(ch)

            elif handler_name == "http":
                ch = cls._parse_http_args(args)
                handlers.append(ch)
        return handlers

    @staticmethod
    def _parse_syslog_args(args: str = None) -> logging.Handler:
        if not SYSLOG_AVAILABLE:
            raise ValueError("syslog is not available on this system")
        if not args:
            args = "/dev/log" if Path("/dev/log").exists() else "address:localhost:514"
        syslog_args = dict()
        try:
            for a in args.split(","):
                arg_name, *arg_value = a.split(":", 2)
                if arg_name == "address":
                    host, *port = arg_value
                    port = 514 if len(port) == 0 else int(port[0])
                    syslog_args["address"] = (host, port)
                elif arg_name == "facility":
                    syslog_args["facility"] = _FACILITY_MAP[arg_value[0]]
                elif arg_name == "socktype":
                    syslog_args["socktype"] = _SOCK_MAP[arg_value[0]]
                else:
                    syslog_args["address"] = arg_name
        except Exception:
            raise ValueError(f"{args} is not a value argument list for syslog logging")
        return logging.handlers.SysLogHandler(**syslog_args)

    @staticmethod
    def _parse_file_args(args: str = None) -> logging.Handler:
        if not args:
            raise ValueError("please provide filename for file logging using format 'file=/path/to/logfile.txt'")
        return logging.FileHandler(args)

    @staticmethod
    def _parse_http_args(args: str = None) -> logging.Handler:
        if not args:
            raise ValueError("please provide destination for http logging using format 'http=url'")
        arg_list = args.split(",")
        url = urllib.parse.urlparse(arg_list.pop(0))
        if url.scheme != "http":
            raise ValueError(f"the http logging module can only log to HTTP URLs, but {url.scheme} was specified")
        host = url.hostname
        path = url.path
        port = url.port or 80

        syslog_args = dict()
        for a in arg_list:
            arg_name, *arg_value = a.split(":", 2)
            if arg_name == "method":
                arg_value = arg_value[0] if len(arg_value) > 0 else "GET"
                syslog_args[arg_name] = arg_value
            else:  # TODO: Provide support for SSL context and credentials
                pass
        return logging.handlers.HTTPHandler(f"{host}:{port}", path, **syslog_args)

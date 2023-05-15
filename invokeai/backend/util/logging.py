# Copyright (c) 2023 Lincoln D. Stein and The InvokeAI Development Team

"""invokeai.util.logging

Logging class for InvokeAI that produces console messages

Usage:

from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.getLogger(name='InvokeAI') // Initialization
(or)
logger = InvokeAILogger.getLogger(__name__) // To use the filename

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
"""

import logging


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


class InvokeAILogFormatter(logging.Formatter):
    '''
    Custom Formatting for the InvokeAI Logger
    '''

    # Color Codes
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    cyan = "\x1b[36;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    # Log Format
    format = "[%(asctime)s]::[%(name)s]::%(levelname)s --> %(message)s"
    ## More Formatting Options: %(pathname)s, %(filename)s, %(module)s, %(lineno)d

    # Format Map
    FORMATS = {
        logging.DEBUG: cyan + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%d-%m-%Y %H:%M:%S")
        return formatter.format(record)


class InvokeAILogger(object):
    loggers = dict()

    @classmethod
    def getLogger(self, name: str = 'InvokeAI') -> logging.Logger:
        if name not in self.loggers:
            logger = logging.getLogger(name)
            logger.setLevel(logging.DEBUG)
            ch = logging.StreamHandler()
            fmt = InvokeAILogFormatter()
            ch.setFormatter(fmt)
            logger.addHandler(ch)
            self.loggers[name] = logger
        return self.loggers[name]

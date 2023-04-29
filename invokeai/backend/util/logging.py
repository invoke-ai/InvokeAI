# Copyright (c) 2023 Lincoln D. Stein and The InvokeAI Development Team

"""invokeai.util.logging

Logging class for InvokeAI that produces console messages that follow
the conventions established in InvokeAI 1.X through 2.X.


One way to use it:

from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.getLogger(__name__)
logger.critical('this is critical')
logger.error('this is an error')
logger.warning('this is a warning')
logger.info('this is info')
logger.debug('this is debugging')

Console messages:
     ### this is critical
     *** this is an error ***
     ** this is a warning
     >> this is info
        | this is debugging

Another way:
import invokeai.backend.util.logging as ialog
ialogger.debug('this is a debugging message')
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

def getLogger(name: str=None)->logging.Logger:
    return InvokeAILogger.getLogger(name)

class InvokeAILogFormatter(logging.Formatter):
    '''
    Repurposed from:
    https://stackoverflow.com/questions/14844970/modifying-logging-message-format-based-on-message-logging-level-in-python3
    '''
    crit_fmt = "### %(msg)s"
    err_fmt = "*** %(msg)s"
    warn_fmt = "** %(msg)s"
    info_fmt = ">> %(msg)s"
    dbg_fmt = "   | %(msg)s"

    def __init__(self):
        super().__init__(fmt="%(levelno)d: %(msg)s", datefmt=None, style='%')

    def format(self, record):
        # Remember the format used when the logging module
        # was installed (in the event that this formatter is
        # used with the vanilla logging module.
        format_orig = self._style._fmt
        if record.levelno == logging.DEBUG:
            self._style._fmt = InvokeAILogFormatter.dbg_fmt
        if record.levelno == logging.INFO:
            self._style._fmt = InvokeAILogFormatter.info_fmt
        if record.levelno == logging.WARNING:
            self._style._fmt = InvokeAILogFormatter.warn_fmt
        if record.levelno == logging.ERROR:
            self._style._fmt = InvokeAILogFormatter.err_fmt
        if record.levelno == logging.CRITICAL:
            self._style._fmt = InvokeAILogFormatter.crit_fmt

        # parent class does the work
        result = super().format(record)
        self._style._fmt = format_orig
        return result

class InvokeAILogger(object):
    loggers = dict()
    
    @classmethod
    def getLogger(self, name:str='invokeai')->logging.Logger:
        if name not in self.loggers:
            logger = logging.getLogger(name)
            logger.setLevel(logging.DEBUG)
            ch = logging.StreamHandler()
            fmt = InvokeAILogFormatter()
            ch.setFormatter(fmt)
            logger.addHandler(ch)
            self.loggers[name] = logger
        return self.loggers[name]

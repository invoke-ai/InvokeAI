import jurigged
from jurigged.codetools import ClassDefinition

from invokeai.app.invocations.baseinvocation import BaseInvocation
from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.getLogger(name=__name__)


def reload_nodes(path: str, codefile: jurigged.CodeFile):
    """Callback function for jurigged post-run events."""
    # Things we have access to here:
    # codefile.module:module - the module object associated with this file
    # codefile.module_name:str - the full module name (its key in sys.modules)
    # codefile.root:ModuleCode - an AST of the current source

    # This is only reading top-level statements, not walking the whole AST, but class definition should be top-level, right?
    class_names = [statement.name for statement in codefile.root.children if isinstance(statement, ClassDefinition)]
    classes = [getattr(codefile.module, name) for name in class_names]
    invocations = [cls for cls in classes if issubclass(cls, BaseInvocation)]
    # outputs = [cls for cls in classes if issubclass(cls, BaseInvocationOutput)]

    # We should assume jurigged has already replaced all references to methods of these classes,
    # but it hasn't re-executed any annotations on them (like @title or @tags).
    # We need to re-do anything that involved introspection like BaseInvocation.get_all_subclasses()
    logger.info("File reloaded: %s contains invocation classes %s", path, invocations)


def start_reloader():
    watcher = jurigged.watch(logger=InvokeAILogger.getLogger(name="jurigged").info)
    watcher.postrun.register(reload_nodes, apply_history=False)

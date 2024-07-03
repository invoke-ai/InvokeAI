from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry


def _build_model_loader_registry():
    return ModelLoaderRegistry()


MODEL_LOADER_REGISTRY = _build_model_loader_registry()

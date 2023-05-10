from invokeai.backend.model_management.model_manager import ModelManager


def choose_model(model_manager: ModelManager, model_name: str):
    """Returns the default model if the `model_name` not a valid model, else returns the selected model."""
    logger = model_manager.logger
    if model_manager.valid_model(model_name):
        model = model_manager.get_model(model_name)
    else:
        model = model_manager.get_model()
        logger.warning(f"{model_name}' is not a valid model name. Using default model \'{model['model_name']}\' instead.")

    return model

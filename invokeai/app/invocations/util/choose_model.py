from invokeai.backend.model_management.model_manager import ModelManager


def choose_model(model_manager: ModelManager, model_name: str):
    """Returns the default model if the `model_name` not a valid model, else returns the selected model."""
    logger = model_manager.logger
    if model_name and not model_manager.valid_model(model_name):
        default_model_name = model_manager.default_model()
        logger.warning(f"\'{model_name}\' is not a valid model name. Using default model \'{default_model_name}\' instead.")
        model = model_manager.get_model()
    else:
        model = model_manager.get_model(model_name)

    return model

from invokeai.app.invocations.baseinvocation import InvocationContext
from invokeai.backend.model_management.model_manager import ModelManager


def choose_model(model_manager: ModelManager, model_name: str):
    """Returns the default model if the `model_name` not a valid model, else returns the selected model."""
    if model_manager.valid_model(model_name):
        return model_manager.get_model(model_name)
    else:
        print(f"* Warning: '{model_name}' is not a valid model name. Using default model instead.")
        return model_manager.get_model()
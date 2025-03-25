import importlib
import pkgutil

import invokeai

KNOWN_IMPORT_ERRORS = {
    "invokeai.backend.image_util.normal_bae.nets.submodules.efficientnet_repo.setup",
    "invokeai.backend.image_util.normal_bae.nets.submodules.efficientnet_repo.validate",
    "invokeai.backend.image_util.normal_bae.nets.submodules.efficientnet_repo.caffe2_benchmark",
    "invokeai.backend.image_util.normal_bae.nets.submodules.efficientnet_repo.caffe2_validate",
    "invokeai.backend.image_util.normal_bae.nets.submodules.efficientnet_repo.onnx_optimize",
    "invokeai.backend.image_util.normal_bae.nets.submodules.efficientnet_repo.onnx_validate",
    "invokeai.backend.image_util.normal_bae.nets.submodules.efficientnet_repo.onnx_to_caffe",
    "invokeai.backend.image_util.normal_bae.nets.submodules.efficientnet_repo.geffnet",
    "invokeai.backend.image_util.normal_bae.nets.submodules.efficientnet_repo.onnx_export",
    "invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_invoke_linear_nf4",
    "invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_invoke_linear_8_bit_lt",
    "invokeai.backend.image_util.normal_bae.nets.submodules.efficientnet_repo.hubconf",
    "invokeai.backend.quantization.bnb_nf4",
    "invokeai.backend.quantization.bnb_llm_int8",
}


def test_invokeai_imports():
    modules = {name for _, name, _ in pkgutil.walk_packages(invokeai.__path__, invokeai.__name__ + ".")}
    failed_to_import = set()

    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception:
            if mod not in KNOWN_IMPORT_ERRORS:
                failed_to_import.add(mod)

    assert not failed_to_import, f"Modules failed to import: {failed_to_import}"

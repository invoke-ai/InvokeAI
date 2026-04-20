from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_docs_json_export_bundle_structure():
    module = _load_module(Path("scripts/generate_docs_json.py"), "generate_docs_json")

    bundle = module.build_docs_bundle()

    assert set(bundle.keys()) == {"invocation_context", "settings"}


def test_docs_json_export_includes_images_interface_and_host_setting():
    module = _load_module(Path("scripts/generate_docs_json.py"), "generate_docs_json")

    bundle = module.build_docs_bundle()

    interface_names = {interface["name"] for interface in bundle["invocation_context"]["interfaces"]}
    assert "ImagesInterface" in interface_names

    setting_names = {setting["name"] for setting in bundle["settings"]["settings"]}
    assert "host" in setting_names
    assert "schema_version" not in setting_names


def test_docs_json_export_includes_rendering_metadata():
    module = _load_module(Path("scripts/generate_docs_json.py"), "generate_docs_json")

    bundle = module.build_docs_bundle()

    images_interface = next(
        interface for interface in bundle["invocation_context"]["interfaces"] if interface["name"] == "ImagesInterface"
    )
    save_method = next(method for method in images_interface["methods"] if method["name"] == "save")
    host_setting = next(setting for setting in bundle["settings"]["settings"] if setting["name"] == "host")

    assert save_method["description"]
    assert "parameters" in save_method
    assert save_method["parameters"]

    assert host_setting["env_var"] == "INVOKEAI_HOST"
    assert host_setting["category"] == "WEB"
    assert "validation" in host_setting
    assert host_setting["validation"] == {}


def test_docs_json_export_writes_expected_files(tmp_path: Path):
    module = _load_module(Path("scripts/generate_docs_json.py"), "generate_docs_json")

    bundle = module.build_docs_bundle()
    module.write_docs_bundle(bundle, tmp_path)

    invocation_context_path = tmp_path / "invocation-context.json"
    settings_path = tmp_path / "settings.json"

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "invocation-context.json",
        "settings.json",
    ]

    invocation_context_payload = json.loads(invocation_context_path.read_text())
    settings_payload = json.loads(settings_path.read_text())

    assert invocation_context_payload == bundle["invocation_context"]
    assert settings_payload == bundle["settings"]

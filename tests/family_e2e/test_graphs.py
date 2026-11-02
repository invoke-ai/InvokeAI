import pytest

from scripts.family_e2e.graphs import (
    IDEOGRAM4_PRESET_STEPS,
    build_export_request,
    output_image_name,
    prepare_batch,
    shareable_source,
)
from scripts.family_e2e.manifest import CaseSpec, ResolvedCase
from scripts.family_e2e.records import Thresholds

MAIN = {"key": "main-key", "hash": "h", "name": "Main", "base": "sdxl", "type": "main"}
LORA = {"key": "lora-key", "hash": "h", "name": "Lora", "base": "sdxl", "type": "lora"}


def _batch(image_name: str = "upload-1.png", seed: int = 1234, reverse: bool = False) -> dict:
    nodes = {
        "model": {"id": "model", "type": "main_model_loader", "model": MAIN, "is_intermediate": True},
        "loras": {"id": "loras", "type": "lora_collection_loader", "loras": [{"lora": LORA, "weight": 0.8}]},
        "init": {"id": "init", "type": "i2l", "image": {"image_name": image_name}, "is_intermediate": True},
        "output": {"id": "output", "type": "l2i", "is_intermediate": False},
    }
    if reverse:
        nodes = {k: dict(reversed(v.items())) for k, v in reversed(nodes.items())}
    data = [[{"node_path": "noise", "field_name": "seed", "items": [seed]}]]
    return {"graph": {"id": "g", "nodes": nodes, "edges": []}, "data": data, "origin": "o", "destination": "gallery"}


def test_the_model_keys_are_the_models_the_server_finds_in_the_nodes() -> None:
    """What the calibrate route takes the identities of: identifiers in node fields, nested ones included."""
    batch = _batch()
    nodes = batch["graph"]["nodes"]
    nodes["metadata"] = {"id": "metadata", "type": "core_metadata", "model": MAIN, "t5_encoder": {**LORA, "key": "enc"}}
    # Neither counts on the server: a mapping without a hash, and a config inside an identifier.
    nodes["other"] = {
        "id": "other",
        "type": "other",
        "value": {"key": "no-hash", "name": "No hash", "base": "sdxl", "type": "main"},
        "loaded": {**MAIN, "key": "outer", "source_config": {**LORA, "key": "inner"}},
    }

    assert prepare_batch(batch, {}, {}).model_keys == {"main-key", "lora-key", "enc", "outer"}


def test_prepare_disables_the_cache_on_every_node_and_runs_once() -> None:
    original = _batch()
    prepared = prepare_batch(original, {"vae": "vae-key"}, {})
    assert all(node["use_cache"] is False for node in prepared.batch["graph"]["nodes"].values())
    assert prepared.batch["runs"] == 1
    assert "use_cache" not in original["graph"]["nodes"]["model"]
    assert prepared.model_keys == {"main-key", "lora-key"}
    assert prepared.components == {"vae": "vae-key"}


def test_hash_ignores_key_order_and_upload_names_but_not_inputs_or_data() -> None:
    first = prepare_batch(_batch("upload-1.png"), {}, {"upload-1.png": "input:abc"})
    reordered = prepare_batch(_batch("upload-7.png", reverse=True), {}, {"upload-7.png": "input:abc"})
    assert first.graph_hash == reordered.graph_hash
    other_input = prepare_batch(_batch("upload-1.png"), {}, {"upload-1.png": "input:def"})
    other_seed = prepare_batch(_batch("upload-1.png", seed=5), {}, {"upload-1.png": "input:abc"})
    assert len({first.graph_hash, other_input.graph_hash, other_seed.graph_hash}) == 3


def _session(results: dict) -> dict:
    return {
        "source_prepared_mapping": {node: [f"{node}-p"] for node in results},
        "results": {f"{k}-p": v for k, v in results.items()},
    }


def test_output_image_comes_from_the_one_non_intermediate_image_node() -> None:
    graph = _batch()["graph"]
    session = _session(
        {
            "init": {"type": "latents_output"},
            "output": {"type": "image_output", "image": {"image_name": "final.png"}},
        }
    )
    assert output_image_name(graph, session) == "final.png"
    # A node without the field is not intermediate, as on the server.
    del graph["nodes"]["init"]["is_intermediate"]
    session = _session(
        {
            "init": {"type": "image_output", "image": {"image_name": "other.png"}},
            "output": {"type": "image_output", "image": {"image_name": "final.png"}},
        }
    )
    with pytest.raises(ValueError, match="found 2"):
        output_image_name(graph, session)
    with pytest.raises(ValueError, match="found 0"):
        output_image_name(graph, _session({}))


def test_export_request_carries_each_case_with_its_uploaded_images() -> None:
    spec = CaseSpec.model_validate(
        {
            "id": "sdxl-cn",
            "model": "main-key",
            "mode": "controlnet",
            "steps": 25,
            "sizes": [[1024, 768]],
            "loras": [{"key": "lora-key", "weight": 0.5}],
            "reference_images": ["ref.png"],
            "control": [{"model": "cn-key", "image": "canny.png", "weight": 0.7, "end": 0.8}],
            "settings": {"scheduler": "euler"},
        }
    )
    case = ResolvedCase(
        case_key="sdxl-cn@1024x768",
        family="sdxl",
        spec=spec,
        width=1024,
        height=768,
        seed=3,
        prompt="a fox",
        negative_prompt="blurry",
        thresholds=Thresholds(psnr_min_db=30, time_tolerance=0.1, peak_tolerance_mb=256),
    )
    names = {"sdxl-cn@1024x768": {("ref.png", False): "upload-1.png", ("canny.png", True): "upload-2.png"}}
    request = build_export_request([case], [MAIN], {"sdxl": {}}, names)
    assert request["version"] == 1
    assert (request["models"], request["capabilities"]) == ([MAIN], {"sdxl": {}})
    assert request["cases"] == [
        {
            "id": "sdxl-cn@1024x768",
            "mode": "controlnet",
            "model_key": "main-key",
            "components": {},
            "width": 1024,
            "height": 768,
            "steps": 25,
            "seed": 3,
            "positive_prompt": "a fox",
            "negative_prompt": "blurry",
            "loras": [{"key": "lora-key", "weight": 0.5}],
            "reference_images": ["upload-1.png"],
            "init_image": None,
            "control": [{"model_key": "cn-key", "image_name": "upload-2.png", "weight": 0.7, "begin": 0.0, "end": 0.8}],
            "settings": {"scheduler": "euler"},
        }
    ]


@pytest.mark.parametrize(
    ("source", "source_type", "shareable"),
    [
        ("Tongyi-MAI/Z-Image-Turbo", "hf_repo_id", True),
        ("https://huggingface.co/x/y/resolve/main/y.safetensors", "url", True),
        ("http://8.8.8.8/y.safetensors", "url", True),
        ("S:/model_library/y.safetensors", "path", False),
        ("https://civitai.com/api/download/models/1?token=abc", "url", False),
        ("https://civitai.com/api/download/models/1#token", "url", False),
        ("https://civitai.com/api/download/models/1;jsessionid=abc", "url", False),
        ("https://user:secret@example.com/y.safetensors", "url", False),
        ("http://cm3588:8090/y.safetensors", "url", False),
        ("http://localhost/y.safetensors", "url", False),
        ("http://nas.local/y.safetensors", "url", False),
        ("http://192.168.1.5/y.safetensors", "url", False),
        ("http://127.0.0.1:9090/y.safetensors", "url", False),
        ("http://[fe80::1]/y.safetensors", "url", False),
        ("file:///S:/y.safetensors", "url", False),
    ],
)
def test_a_source_is_shareable_when_it_names_the_model_for_anyone(
    source: str, source_type: str, shareable: bool
) -> None:
    assert shareable_source({"source": source, "source_type": source_type}) == (source if shareable else None)


def test_each_denoise_node_states_the_step_count_it_runs_over_its_whole_schedule() -> None:
    nodes = {
        # Ideogram 4 runs its sampler preset's count unless it sets `steps`; the case's steps do not reach it.
        "ideogram": {"id": "ideogram", "type": "ideogram4_denoise", "sampler_preset": "V4_TURBO_12"},
        "ideogram_steps": {
            "id": "ideogram_steps",
            "type": "ideogram4_denoise",
            "sampler_preset": "V4_TURBO_12",
            "steps": 30,
        },
        "flux2": {"id": "flux2", "type": "flux2_denoise", "num_steps": 4, "denoising_start": 0, "denoising_end": 1},
        "z_image": {"id": "z_image", "type": "z_image_denoise", "steps": 8},
        # A window of the schedule runs a part of it that each family cuts differently.
        "img2img": {
            "id": "img2img",
            "type": "flux2_denoise",
            "num_steps": 4,
            "denoising_start": 0.3,
            "denoising_end": 1,
        },
        "sdxl_base": {"id": "sdxl_base", "type": "denoise_latents", "steps": 30, "denoising_end": 0.8},
        "output": {"id": "output", "type": "l2i", "is_intermediate": False},
    }
    prepared = prepare_batch({"graph": {"id": "g", "nodes": nodes, "edges": []}, "data": []}, {}, {})
    assert prepared.step_counts == {"ideogram": 12, "ideogram_steps": 30, "flux2": 4, "z_image": 8}


def test_the_ideogram_preset_step_counts_are_the_servers() -> None:
    from invokeai.backend.ideogram4.sampler_configs import PRESETS

    assert IDEOGRAM4_PRESET_STEPS == {name: preset.num_steps for name, preset in PRESETS.items()}

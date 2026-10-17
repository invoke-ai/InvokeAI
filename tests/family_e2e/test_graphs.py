import pytest

from scripts.family_e2e.graphs import build_export_request, output_image_name, prepare_batch
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

from pathlib import Path
from typing import Any

from invokeai.app.invocations.remote_worker.model_transfer import LocalModelFile, model_layout_signature
from invokeai.app.invocations.remote_worker.remote_client import RemoteConfig, RemoteInvokeClient
from invokeai.app.invocations.remote_worker.remote_nodes import _find_compatible_remote_model


def _identifier() -> dict[str, Any]:
    return {
        "key": "local-clip",
        "hash": "blake3:same",
        "name": "clip-vit-large-patch14",
        "base": "any",
        "type": "clip_embed",
        "submodel_type": None,
    }


def test_layout_signature_distinguishes_same_weight_in_different_paths(tmp_path: Path) -> None:
    flat = tmp_path / "flat"
    bundled = tmp_path / "bundled"
    flat.mkdir()
    (bundled / "text_encoder").mkdir(parents=True)
    (bundled / "tokenizer").mkdir()

    (flat / "model.safetensors").write_bytes(b"same weights")
    (flat / "config.json").write_text("{}", encoding="utf-8")
    (bundled / "text_encoder" / "model.safetensors").write_bytes(b"same weights")
    (bundled / "text_encoder" / "config.json").write_text("{}", encoding="utf-8")
    (bundled / "tokenizer" / "vocab.json").write_text("{}", encoding="utf-8")

    assert model_layout_signature(flat)[0] == "directory"
    assert model_layout_signature(flat) != model_layout_signature(bundled)


def test_remap_skips_same_hash_candidate_rejected_by_layout_validator() -> None:
    client = RemoteInvokeClient(RemoteConfig(base_url="http://worker", api_key=""))
    bad = {**_identifier(), "key": "remote-flat"}
    good = {**_identifier(), "key": "remote-bundled"}

    client.list_models = lambda: [bad, good]  # type: ignore[method-assign]
    client.get_model = lambda key: bad if key == "remote-flat" else good  # type: ignore[method-assign]

    graph = {"nodes": {"loader": {"id": "loader", "type": "test", "model": _identifier()}}}
    messages = client.remap_model_identifiers(
        graph,
        model_match_validator=lambda _local, remote: remote.get("key") == "remote-bundled",
    )

    assert graph["nodes"]["loader"]["model"]["key"] == "remote-bundled"
    assert any("remote-bundled" in message for message in messages)


def test_compatible_lookup_rejects_wrong_layout_but_accepts_good_duplicate(tmp_path: Path) -> None:
    model_dir = tmp_path / "clip"
    (model_dir / "text_encoder").mkdir(parents=True)
    (model_dir / "tokenizer").mkdir()
    (model_dir / "text_encoder" / "model.safetensors").write_bytes(b"weights")
    (model_dir / "tokenizer" / "vocab.json").write_text("{}", encoding="utf-8")

    model = LocalModelFile(
        path=model_dir,
        key="local-clip",
        name="clip-vit-large-patch14",
        hash="blake3:same",
        base="any",
        type="clip_embed",
    )
    expected_kind, expected_signature = model_layout_signature(model_dir)

    class Worker:
        def list_models(self) -> list[dict[str, Any]]:
            return [
                {**_identifier(), "key": "remote-flat"},
                {**_identifier(), "key": "remote-bundled"},
            ]

        def get_model_layout(self, key: str) -> dict[str, Any]:
            if key == "remote-flat":
                return {"kind": "directory", "signature": "wrong"}
            return {"kind": expected_kind, "signature": expected_signature}

    found = _find_compatible_remote_model(Worker(), model)  # type: ignore[arg-type]

    assert found is not None
    assert found["key"] == "remote-bundled"

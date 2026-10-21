"""`bundle`: one zip that opens elsewhere, with what the report shows and nothing of this machine."""

import gzip
import json
import os
import re
import urllib.parse
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import psutil
import pytest
import yaml
from PIL import Image, PngImagePlugin

from scripts.family_e2e.__main__ import main
from scripts.family_e2e.graphs import model_identifiers
from scripts.family_e2e.records import (
    Environment,
    ForeignProcess,
    ModelIdentity,
    Reference,
    RunInfo,
    load_records,
    load_run_info,
    write_run_info,
)
from scripts.family_e2e.references import accept, load_reference
from scripts.family_e2e.report import write_report
from scripts.family_e2e.runner import HEARTBEAT_FILE, process_identity
from tests.family_e2e.test_records import make_environment, make_record
from tests.family_e2e.test_report import CASE, FIRST, OTHER_CASE, PROFILE, SECOND, _case_records, _write_run

ENV_SECRET = "hf_bundleTestEnvToken"
REFERENCE_SECRET = "civitaiReferenceOnlyToken"
"""A credential of the reference's run only, not of the bundled run."""
CONFIG_SECRET = "sk-unmaskedByAnOlderServer"
QUERY_SECRET = "bundleTestQueryToken"
PROCESS_NAME = "SecretGame.exe"
ALLOC_CONF = "backend:cudaMallocAsync"
"""A setting the manifest's server env sets, no credential: it must not change a verdict."""
IDENTITY = ModelIdentity(
    key="main-key",
    name="Z-Image Turbo",
    base="z-image",
    type="main",
    format="diffusers",
    variant=None,
    hash="blake3:" + "a" * 64,
    source="Tongyi-MAI/Z-Image-Turbo",
)
SHOWN = f"images/{CASE}.default.measure.0.1.png"
"""The image the bundled report shows for this run: the final attempt of the first measure."""
PREVIOUS = f"previous/{FIRST}/images/{CASE}.default.measure.0.0.png"
REFERENCE = f"reference/{CASE}/image.png"
TEXT_CHUNKS = {b"tEXt", b"zTXt", b"iTXt"}


@dataclass(frozen=True)
class _Bundle:
    data_dir: Path
    run_dir: Path
    root: Path
    """The extracted zip's folder."""
    members: set[str]
    paths: list[Path]
    """This machine's paths the run's files name."""


def _model(key: str, model_type: str, source: str, source_type: str, **fields: str) -> dict[str, str]:
    return {
        "key": key,
        "hash": "blake3:" + "b" * 64,
        "name": key,
        "base": "z-image",
        "type": model_type,
        "source": source,
        "source_type": source_type,
        **fields,
    }


def _graph(main_model: dict[str, str]) -> dict:
    encoder = _model(
        "enc-key", "qwen3_encoder", "Qwen/Qwen3-4B", "hf_repo_id", source_url="https://hf.co/Qwen/Qwen3-4B"
    )
    vae = _model("vae-key", "vae", f"https://example.com/vae.safetensors?download_token={QUERY_SECRET}", "url")
    lora = _model("lora-key", "lora", "https://example.com/lora.safetensors", "url")
    lan = _model("lan-key", "lora", "http://cm3588:8090/lan.safetensors", "url")
    nodes = {
        "loader": {"id": "loader", "type": "z_image_model_loader", "model": main_model, "vae_model": vae},
        "encoder": {"id": "encoder", "type": "z_image_text_encoder", "qwen3_encoder_model": encoder},
        "lora": {"id": "lora", "type": "lora_collection_loader", "loras": [{"lora": lora}, {"lora": lan}]},
    }
    return {"graph": {"id": "g", "nodes": nodes, "edges": []}, "data": [], "origin": "o", "destination": "gallery"}


def _add_metadata(path: Path, text: str) -> None:
    """Re-saves a PNG with the text chunks InvokeAI writes, a private chunk, and a gamma chunk that colours it."""
    info = PngImagePlugin.PngInfo()
    info.add_text("invokeai_metadata", text)
    info.add_text("invokeai_graph", text, zip=True)
    info.add_itxt("Comment", text)
    info.add(b"prVt", text.encode())
    info.add(b"gAMA", (45455).to_bytes(4, "big"))
    with Image.open(path) as image:
        pixels = image.copy()
    pixels.save(path, pnginfo=info)


def _chunk_types(data: bytes) -> list[bytes]:
    types, offset = [], 8
    while offset < len(data):
        types.append(data[offset + 4 : offset + 8])
        offset += 12 + int.from_bytes(data[offset : offset + 4], "big")
    return types


def _pixels(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image)


def _snapshot(directory: Path) -> dict[Path, bytes]:
    return {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*") if path.is_file()}


def _summary_table(summary: str) -> list[str]:
    return [line for line in summary.splitlines() if line.startswith("| ")]


@pytest.fixture
def bundle(tmp_path: Path) -> _Bundle:
    """A run with a reference and a previous run whose files name this machine's paths, secrets and processes."""
    # Spaces, so a path in free text does not end at the first one.
    data_dir, template_root, code_dir = tmp_path / "e2e data", tmp_path / "template root", tmp_path / "code dir"
    model_dir = tmp_path / "model library" / "z image turbo"
    other_model = tmp_path / "other models" / "flux dev.safetensors"
    metadata = json.dumps(
        {"qwen3_source": {"path": str(model_dir)}, "url": f"https://a.example/x?token={QUERY_SECRET}"}
    )
    data_dir.mkdir()
    manifest = data_dir / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "profile": PROFILE,
                "data_dir": ".",
                "gpu": "RTX",
                "server": {"template_root": str(template_root), "python": "python.exe", "code_dir": str(code_dir)},
                "defaults": {"prompt": "a fox"},
                "families": [
                    {"family": "z", "cases": [{"id": "zimage-t2i", "model": "m", "steps": 8, "sizes": [[8, 8]]}]}
                ],
            }
        )
    )
    # The reference and previous run, from another checkout. A credential reaches the report through its
    # environment, and its reference.json carries that environment whole.
    base_checkout, base_python = tmp_path / "base checkout", Path.home() / "venvs" / "invoke base" / "python.exe"
    first_environment = make_environment(
        code_dir=str(base_checkout),
        python=str(base_python),
        invokeai_file=str(base_checkout / "invokeai" / "__init__.py"),
        server_cmd=[str(base_python), "--root", str(data_dir / "scratch")],
        extra_env={"HF_TOKEN": ENV_SECRET, "CIVITAI_API_TOKEN": REFERENCE_SECRET},
        runtime_config={
            "set_fields": [],
            "config": {
                "pytorch_cuda_alloc_conf": "backend:native",
                "attention_backend": f"sage-{ENV_SECRET}",
                "models_dir": str(template_root / "models"),
                "image_index_model": f"model-{REFERENCE_SECRET}",
                "external_gemini_base_url": "http://cm3588:8090/v1",
            },
        },
    )
    first = _write_run(data_dir, FIRST, _case_records(FIRST), environment=first_environment)
    for image in (first / "images").iterdir():
        _add_metadata(image, metadata)
    accept(data_dir, first, None)
    # The report notes it skipped this run, naming its records file in the error.
    unreadable = _write_run(data_dir, "20261001-180000-rtx4090", [])
    (unreadable / "records.jsonl").write_text("{\n")

    run_dir = data_dir / "runs" / SECOND
    python = code_dir / ".venv" / "python.exe"
    environment = make_environment(
        code_dir=str(code_dir),
        python=str(python),
        invokeai_file=str(code_dir / "invokeai" / "__init__.py"),
        server_cmd=[str(python), str(code_dir / "scripts" / "invokeai-web.py"), "--root", str(data_dir / "scratch")],
        extra_env={"HF_TOKEN": ENV_SECRET, "INVOKEAI_PYTORCH_CUDA_ALLOC_CONF": ALLOC_CONF},
        runtime_config={
            "set_fields": [],
            "config": {
                "models_dir": str(template_root / "models"),
                "log_handlers": ["console", f"file={run_dir / 'server-default.log'}"],
                "external_openai_api_key": CONFIG_SECRET,
                "external_gemini_base_url": f"https://user:{CONFIG_SECRET}@gemini.example.com/v1?key={QUERY_SECRET}",
                "log_tokenization": True,
                "pytorch_cuda_alloc_conf": ALLOC_CONF,
                "attention_backend": "sage",
            },
        },
    )
    warmup, measure, other, spoiled = _case_records(SECOND)
    crashed = make_record(
        CASE,
        rep=0,
        attempt=0,
        run_id=SECOND,
        status="error",
        image=False,
        error=f"OSError: cannot open '{model_dir / 'vae' / 'diffusion_pytorch_model.safetensors'}' or '{other_model}' "
        f"with {ENV_SECRET} from {Path.home() / 'private-folder' / 'notes.txt'} in {code_dir / 'invokeai'}, "
        f"fetched from https://user:pw@civitai.com/api/download/models/1;session=s?token={QUERY_SECRET}#top",
        invalid_reasons=[f"server crashed, see {run_dir / 'server-default.log'}"],
    )
    resumed = measure.model_copy(update={"attempt": 1, "record_id": f"{CASE}.default.measure.0.1", "image_path": SHOWN})
    assert spoiled.memory is not None
    foreign = ForeignProcess(pid=4321, name=PROCESS_NAME, dedicated_bytes=2**31)
    spoiled = spoiled.model_copy(
        update={
            "valid_timing": False,
            "valid_memory": False,
            "invalid_reasons": [f"foreign GPU load: pid 4321 ({PROCESS_NAME}) 2048 MiB"],
            "memory": spoiled.memory.model_copy(update={"foreign": [foreign]}),
        }
    )
    _write_run(
        data_dir,
        SECOND,
        [warmup, crashed, resumed, other, spoiled],
        environment=environment,
        manifest_path=str(manifest),
        case_models={CASE: [IDENTITY]},
    )
    _add_metadata(run_dir / SHOWN, metadata)
    main_model = _model(
        "main-key",
        "main",
        str(tmp_path / "downloads" / "z image.safetensors"),
        "path",
        name="Z-Image Turbo",
        path=str(model_dir),
        config_path=str(template_root / "configs" / "z.yaml"),
        source_api_response=json.dumps({"downloadUrl": f"https://civitai.com/api/download/1?token={QUERY_SECRET}"}),
        source_url=f"https://example.com/z?ref={QUERY_SECRET}",
    )
    (run_dir / "graphs").mkdir()
    (run_dir / "graphs" / f"{CASE}.json").write_text(json.dumps(_graph(main_model), indent=2))
    # The exporter request lists every model of the server, also those no graph uses.
    (run_dir / "export").mkdir()
    flux = _model("flux-key", "main", str(other_model), "path", name="Flux Dev", path=str(other_model))
    (run_dir / "export" / "export-request.json").write_text(json.dumps({"version": 1, "models": [flux]}))
    (run_dir / "traces").mkdir()
    with gzip.open(run_dir / "traces" / f"{resumed.record_id}.csv.gz", "wt") as trace:
        trace.write("series,t,bytes\ndedicated,0.0,1024\n")
    (run_dir / "inputs").mkdir()
    (run_dir / "inputs" / "fox@8x8.png").write_bytes(b"input")
    for log in ("runner.log", "server-default.log"):
        (run_dir / log).write_text(f"HF_TOKEN={ENV_SECRET} in {template_root}\n")
    (run_dir / HEARTBEAT_FILE).write_text(json.dumps({"runner": None, "step": "done"}))

    before = _snapshot(run_dir)
    output = tmp_path / "out" / "bundle.zip"
    assert main(["bundle", str(run_dir), "--output", str(output)]) == 0
    assert _snapshot(run_dir) == before
    with zipfile.ZipFile(output) as archive:
        archive.extractall(tmp_path / "unzipped")
        members = {name.removeprefix(f"{SECOND}/") for name in archive.namelist()}
    paths = [data_dir, template_root, code_dir, run_dir, model_dir, other_model, Path.home(), tmp_path]
    return _Bundle(data_dir, run_dir, tmp_path / "unzipped" / SECOND, members, paths)


def test_bundle_holds_the_report_and_the_images_it_shows_with_their_pixels_and_no_logs(bundle: _Bundle) -> None:
    assert bundle.members == {
        "bundle.json",
        "report.html",
        "summary.md",
        "run.json",
        "records.jsonl",
        f"graphs/{CASE}.json",
        f"traces/{CASE}.default.measure.0.1.csv.gz",
        SHOWN,
        REFERENCE,
        f"reference/{CASE}/reference.json",
        PREVIOUS,
        f"diff/{CASE}.png",
    }
    report = (bundle.root / "report.html").read_text(encoding="utf-8")
    links = {urllib.parse.unquote(link) for link in re.findall(r'(?:src|href)="([^"#][^"]*)"', report)}
    assert links == {SHOWN, REFERENCE, PREVIOUS, f"diff/{CASE}.png"}
    sources = {
        SHOWN: bundle.run_dir / SHOWN,
        REFERENCE: bundle.data_dir / "references" / PROFILE / CASE / "image.png",
        PREVIOUS: bundle.data_dir / "runs" / FIRST / "images" / f"{CASE}.default.measure.0.0.png",
    }
    for name, source in sources.items():
        # The same pixels, so a recipient recomputes the PSNR; the colour chunk stays, the metadata goes.
        assert np.array_equal(_pixels(bundle.root / name), _pixels(source)), name
        assert TEXT_CHUNKS < set(_chunk_types(source.read_bytes()))
        bundled = [kind for kind in _chunk_types((bundle.root / name).read_bytes()) if kind != b"IDAT"]
        assert bundled == [b"IHDR", b"gAMA", b"IEND"], name


def test_bundle_holds_no_path_secret_or_process_name_of_this_machine(bundle: _Bundle) -> None:
    forbidden = {
        ENV_SECRET,
        REFERENCE_SECRET,
        CONFIG_SECRET,
        QUERY_SECRET,
        PROCESS_NAME,
        "private-folder",
        "model library",
        "other models",
        "cm3588",
    }
    for path in bundle.paths:
        forbidden |= {str(path), path.as_posix(), json.dumps(str(path))[1:-1]}
    contents = {}
    for path in bundle.root.rglob("*"):
        if path.is_file():
            data = path.read_bytes()
            contents[path.relative_to(bundle.root).as_posix()] = gzip.decompress(data) if path.suffix == ".gz" else data
    needles = {needle.lower().encode("utf-8") for needle in forbidden}

    assert len(contents) == len(bundle.members)
    found = {name: [n for n in needles if n in data.lower()] for name, data in contents.items()}
    assert found == dict.fromkeys(contents, [])
    # The errors are still there and readable, with placeholders.
    summary = (bundle.root / "summary.md").read_text(encoding="utf-8")
    sep = os.sep
    assert (
        f"cannot open '<model Z-Image Turbo>{sep}vae{sep}diffusion_pytorch_model.safetensors' or '<model Flux Dev>'"
        in (summary)
    )
    assert f"with <HF_TOKEN> from <home>{sep}notes.txt in <code_dir>{sep}invokeai" in summary
    assert "fetched from https://civitai.com/api/download/models/1" in summary
    assert "skipped, unreadable: <data_dir>" in summary
    assert "foreign GPU load: pid 4321 (<process>) 2048 MiB" in summary


def test_bundled_verdicts_match_the_report_on_disk(bundle: _Bundle) -> None:
    on_disk = write_report(bundle.data_dir, bundle.run_dir).summary_md.read_text(encoding="utf-8")
    bundled = (bundle.root / "summary.md").read_text(encoding="utf-8")

    assert _summary_table(bundled) == _summary_table(on_disk)
    # A setting from the server env is no credential; the credential an earlier environment shows is replaced.
    change = f"pytorch_cuda_alloc_conf backend:native → {ALLOC_CONF}"
    assert f"environment changed vs reference: {change}; attention_backend sage-{ENV_SECRET} → sage" in on_disk
    assert f"environment changed vs reference: {change}; attention_backend sage-<HF_TOKEN> → sage" in bundled


def test_bundled_run_names_paths_and_variables_only_and_keeps_model_identities(bundle: _Bundle) -> None:
    info = RunInfo.model_validate_json((bundle.root / "run.json").read_bytes())

    environment: Environment = info.environments["default"]
    assert (info.manifest_path, environment.code_dir, environment.python) == ("manifest.yaml", "code dir", "python.exe")
    assert environment.server_cmd == ["python.exe", "invokeai-web.py", "--root", "scratch"]
    assert environment.extra_env == {"HF_TOKEN": "<redacted>", "INVOKEAI_PYTORCH_CUDA_ALLOC_CONF": "<redacted>"}
    assert environment.runtime_config["config"] == {
        "models_dir": "models",
        "log_handlers": ["console", f"file=<run_dir>{os.sep}server-default.log"],
        "external_openai_api_key": "<redacted>",
        "external_gemini_base_url": "https://gemini.example.com/v1",
        "log_tokenization": True,
        "pytorch_cuda_alloc_conf": ALLOC_CONF,
        "attention_backend": "sage",
    }
    foreign = [(p.pid, p.name) for r in load_records(bundle.root) if r.memory for p in r.memory.foreign]
    assert foreign == [(4321, "<process>")]
    assert info.case_models == {CASE: [IDENTITY]}
    assert info.graph_hashes == {CASE: "graph-1"}


def test_bundled_reference_keeps_what_a_reader_compares_and_names_paths_and_variables_only(bundle: _Bundle) -> None:
    bundled = Reference.model_validate_json((bundle.root / "reference" / CASE / "reference.json").read_bytes())
    loaded = load_reference(bundle.data_dir, PROFILE, CASE)
    assert loaded is not None
    original = loaded[0]

    # Its run, graph hash, medians and the hash of the original image: what tells BASELINE from PASS and drives
    # the comparisons.
    assert bundled.model_dump(exclude={"environment"}) == original.model_dump(exclude={"environment"})
    assert (bundled.source_run_id, bundled.graph_hash) == (FIRST, "graph-1")
    environment = bundled.environment
    assert (environment.code_dir, environment.python, environment.server_cmd) == (
        "base checkout",
        "python.exe",
        ["python.exe", "--root", "scratch"],
    )
    assert environment.extra_env == {"HF_TOKEN": "<redacted>", "CIVITAI_API_TOKEN": "<redacted>"}
    assert environment.runtime_config["config"] == {
        "pytorch_cuda_alloc_conf": "backend:native",
        "attention_backend": "sage-<HF_TOKEN>",
        "models_dir": "models",
        "image_index_model": "model-<CIVITAI_API_TOKEN>",
        "external_gemini_base_url": "http://<local host>/v1",
    }


def test_only_cases_with_a_readable_reference_carry_its_reference_json(tmp_path: Path) -> None:
    third = "zimage-t2i@16x16"
    records = [*_case_records(FIRST), *_case_records(FIRST, OTHER_CASE), *_case_records(FIRST, third)]
    run_dir = _write_run(tmp_path, FIRST, records)
    accept(tmp_path, run_dir, [CASE, OTHER_CASE])
    (tmp_path / "references" / PROFILE / OTHER_CASE / "reference.json").write_text("{")

    assert main(["bundle", str(run_dir)]) == 0

    with zipfile.ZipFile(run_dir / f"{FIRST}.zip") as archive:
        names = sorted(name.removeprefix(f"{FIRST}/") for name in archive.namelist() if "/reference/" in name)
    assert names == [f"reference/{CASE}/image.png", f"reference/{CASE}/reference.json"]


def test_bundled_graphs_keep_only_model_sources_anyone_may_see(bundle: _Bundle) -> None:
    graph = json.loads((bundle.root / "graphs" / f"{CASE}.json").read_text(encoding="utf-8"))
    models = {model["key"]: model for model in model_identifiers(graph)}

    assert [key for key, model in models.items() if {"path", "config_path", "source_api_response"} & set(model)] == []
    assert {key: model.get("source") for key, model in models.items()} == {
        "main-key": None,  # a local path
        "vae-key": None,  # a query that may carry a token
        "lan-key": None,  # a host of the local network
        "enc-key": "Qwen/Qwen3-4B",
        "lora-key": "https://example.com/lora.safetensors",
    }
    assert {key: model.get("source_url") for key, model in models.items() if "source_url" in model} == {
        "enc-key": "https://hf.co/Qwen/Qwen3-4B"
    }


@pytest.mark.parametrize(("status", "shown"), [("stopped", "stopped"), ("running", "interrupted")])
def test_a_run_recorded_before_model_identities_existed_bundles_once_its_runner_is_gone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], status: str, shown: str
) -> None:
    run_dir = _write_run(tmp_path, FIRST, _case_records(FIRST), status=status)
    old = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    del old["case_models"]
    (run_dir / "run.json").write_text(json.dumps(old), encoding="utf-8")
    # The pid of the recorded runner now belongs to another process: the runner is gone.
    identity = [os.getpid(), psutil.Process(os.getpid()).create_time() - 100.0]
    (run_dir / HEARTBEAT_FILE).write_text(json.dumps({"runner": identity}))

    assert main(["report", str(run_dir)]) == 0
    assert main(["bundle", str(run_dir)]) == 0

    zip_path = run_dir / f"{FIRST}.zip"
    assert f"bundle  {zip_path}" in capsys.readouterr().out
    with zipfile.ZipFile(zip_path) as archive:
        bundled = RunInfo.model_validate_json(archive.read(f"{FIRST}/run.json"))
        report = archive.read(f"{FIRST}/report.html").decode("utf-8")
        summary = archive.read(f"{FIRST}/summary.md").decode("utf-8")
    assert (bundled.status, bundled.case_models) == (status, {})
    assert "## Models" not in summary
    assert f"run {shown}" in report
    assert f"The run is {shown}; cases it did not reach show as INVALID." in report


@pytest.mark.parametrize("tool_commit", ["0123456789abcdef", None])
def test_bundle_describes_its_format_for_the_upload_hub(tmp_path: Path, tool_commit: str | None) -> None:
    run_dir = _write_run(tmp_path, FIRST, _case_records(FIRST))
    write_run_info(run_dir, load_run_info(run_dir).model_copy(update={"tool_commit": tool_commit}))

    # Bracketed by the clock, not waited for; the timestamp has whole seconds.
    before = datetime.now(timezone.utc).replace(microsecond=0)
    assert main(["bundle", str(run_dir)]) == 0
    after = datetime.now(timezone.utc)

    with zipfile.ZipFile(run_dir / f"{FIRST}.zip") as archive:
        description = json.loads(archive.read(f"{FIRST}/bundle.json"))
    created_at = datetime.fromisoformat(description.pop("created_at"))
    assert description == {"format_version": 2, "tool_commit": tool_commit}
    assert created_at.utcoffset() == timedelta(0)
    assert before <= created_at <= after


def test_bundle_refuses_while_the_runner_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir = _write_run(tmp_path, FIRST, _case_records(FIRST), status="running")
    (run_dir / HEARTBEAT_FILE).write_text(json.dumps({"runner": process_identity(os.getpid())}))

    assert main(["bundle", str(run_dir)]) == 2

    assert f"the runner of {FIRST} is still running" in capsys.readouterr().err
    assert list(run_dir.glob("*.zip")) == []

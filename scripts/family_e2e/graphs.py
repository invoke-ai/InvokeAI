"""Graphs for the cases: the exporter request, the webv2 exporter call, and the runner's batch policy.

The webv2 exporter builds each case's batch with the real frontend graph builders (contract 1 in
`.scratch/agents/family-e2e/contract.md`). The runner then disables the invocation cache on every node, so repeated
runs really execute, and hashes the result so a frontend graph change shows up against the reference.
"""

from __future__ import annotations

import copy
import hashlib
import ipaddress
import json
import shutil
import subprocess
import urllib.parse
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.family_e2e.manifest import TOOL_CHECKOUT, ResolvedCase
from scripts.family_e2e.server import spawn_in_job, terminate

EXPORT_VERSION = 1
WEBV2_DIR = TOOL_CHECKOUT / "invokeai" / "frontend" / "webv2"
EXPORT_TIMEOUT_S = 600
EXPORT_REQUEST_FILE = "export-request.json"
"""In the exporter's work directory; holds every model config of the server, paths included."""
# Hosts that name a machine only on its own network.
LOCAL_HOST_SUFFIXES = (".local", ".localhost")


class ExportError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreparedGraph:
    batch: dict[str, Any]
    """The `batch` body for enqueue_batch."""
    graph_hash: str
    model_keys: frozenset[str]
    """Keys of every model identifier in the graph (main model, components, LoRAs, control models)."""
    components: dict[str, str]
    """The exporter's resolved component choice, for provenance."""


def build_export_request(
    cases: list[ResolvedCase],
    models: list[dict[str, Any]],
    capabilities: Any,
    image_names: dict[str, dict[tuple[str, bool], str]],
) -> dict[str, Any]:
    """`image_names` maps each case key's (manifest input path, used at the case size) to the uploaded image name."""
    request_cases = []
    for case in cases:
        spec = case.spec
        names = image_names[case.case_key]
        request_cases.append(
            {
                "id": case.case_key,
                "mode": spec.mode,
                "model_key": spec.model,
                "components": spec.components,
                "width": case.width,
                "height": case.height,
                "steps": spec.steps,
                "seed": case.seed,
                "positive_prompt": case.prompt,
                "negative_prompt": case.negative_prompt,
                "loras": [{"key": lora.key, "weight": lora.weight} for lora in spec.loras],
                "reference_images": [names[(path, False)] for path in spec.reference_images],
                "init_image": (
                    None
                    if spec.init_image is None
                    else {"image_name": names[(spec.init_image.path, True)], "strength": spec.init_image.strength}
                ),
                "control": [
                    {
                        "model_key": c.model,
                        "image_name": names[(c.image, True)],
                        "weight": c.weight,
                        "begin": c.begin,
                        "end": c.end,
                    }
                    for c in spec.control
                ],
                "settings": spec.settings,
            }
        )
    return {"version": EXPORT_VERSION, "models": models, "capabilities": capabilities, "cases": request_cases}


def run_pnpm_exporter(request: dict[str, Any], work_dir: Path) -> dict[str, Any]:
    """Runs `pnpm graphs:export` in the tool checkout's webv2 package. Per-case errors stay in the response."""
    pnpm = shutil.which("pnpm")
    if pnpm is None:
        raise ExportError("pnpm is not on PATH; the graph exporter runs with pnpm from invokeai/frontend/webv2")
    work_dir.mkdir(parents=True, exist_ok=True)
    # Absolute: pnpm runs the exporter from the webv2 package directory.
    request_path = (work_dir / EXPORT_REQUEST_FILE).resolve()
    response_path = (work_dir / "export-response.json").resolve()
    request_path.write_text(json.dumps(request), encoding="utf-8")
    response_path.unlink(missing_ok=True)
    # pnpm is a shim (pnpm.cmd on Windows) around node: a timeout must kill the whole tree, not just the shim.
    process, job = spawn_in_job(
        [pnpm, "graphs:export", str(request_path), str(response_path)],
        cwd=WEBV2_DIR,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = process.communicate(timeout=EXPORT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise ExportError(f"pnpm graphs:export did not finish within {EXPORT_TIMEOUT_S}s") from None
    finally:
        terminate(process, job)
    if process.returncode != 0 or not response_path.is_file():
        output = (stderr or stdout).decode("utf-8", errors="replace")
        raise ExportError(f"pnpm graphs:export exited with {process.returncode}:\n{output[-4000:]}")
    response = json.loads(response_path.read_text(encoding="utf-8"))
    if response.get("version") != EXPORT_VERSION:
        raise ExportError(f"exporter answered version {response.get('version')!r}, expected {EXPORT_VERSION}")
    return response


def model_identifiers(value: Any) -> Iterator[dict[str, Any]]:
    """Every model identifier in a graph: a ModelIdentifierField {"key", "hash", "name", "base", "type", ...}.

    The webv2 builders put the model's whole config there, path and source included.
    """
    if isinstance(value, dict):
        if isinstance(value.get("key"), str) and "base" in value and "type" in value:
            yield value
        for item in value.values():
            yield from model_identifiers(item)
    elif isinstance(value, list):
        for item in value:
            yield from model_identifiers(item)


def is_public_url(value: object) -> bool:
    """An http(s) URL of a public host, without the places a download token would hide: credentials, `;params`,
    query or fragment."""
    if not isinstance(value, str) or ";" in value:
        return False
    try:
        parts = urllib.parse.urlsplit(value)
        host = parts.hostname
    except ValueError:
        return False
    if parts.scheme not in ("http", "https") or parts.query or parts.fragment or "@" in parts.netloc:
        return False
    return is_public_host(host)


def is_public_host(host: str | None) -> bool:
    """A host anyone can reach: no private, loopback or link-local address, and no name of the local network."""
    if not host:
        return False
    try:
        return ipaddress.ip_address(host).is_global
    except ValueError:
        # A single label (localhost, a LAN machine's name) resolves only on its own network.
        return "." in host and not host.endswith(LOCAL_HOST_SUFFIXES)


def shareable_source(config: dict[str, Any]) -> str | None:
    """A model config's source when it names the model on any machine: an HF repo id or a public URL."""
    source, source_type = config.get("source"), config.get("source_type")
    if source_type == "hf_repo_id" or (source_type == "url" and is_public_url(source)):
        return source if isinstance(source, str) else None
    return None


def _replace_strings(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        return replacements.get(value, value)
    if isinstance(value, dict):
        return {k: _replace_strings(v, replacements) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_strings(v, replacements) for v in value]
    return value


def prepare_batch(batch: dict[str, Any], components: dict[str, str], input_ids: dict[str, str]) -> PreparedGraph:
    """Applies the runner policy to an exported batch and hashes it.

    `input_ids` maps uploaded image names to a stable identity of the input file. Upload names differ on every
    server start, so the hash covers the input's identity instead.
    """
    batch = copy.deepcopy(batch)
    for node in batch["graph"]["nodes"].values():
        node["use_cache"] = False
    batch["runs"] = 1
    hashed = _replace_strings({"graph": batch["graph"], "data": batch.get("data")}, input_ids)
    canonical = json.dumps(hashed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return PreparedGraph(
        batch=batch,
        graph_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        model_keys=frozenset(identifier["key"] for identifier in model_identifiers(batch["graph"])),
        components=components,
    )


def output_image_name(graph: dict[str, Any], session: dict[str, Any]) -> str:
    """The image of the graph's one non-intermediate node that produced an image."""
    mapping = session.get("source_prepared_mapping") or {}
    results = session.get("results") or {}
    names = []
    for node_id, node in graph["nodes"].items():
        # The server's default: a node without the field is not intermediate.
        if node.get("is_intermediate", False):
            continue
        for prepared_id in mapping.get(node_id, []):
            image = (results.get(prepared_id) or {}).get("image")
            if isinstance(image, dict) and isinstance(image.get("image_name"), str):
                names.append(image["image_name"])
    if len(names) != 1:
        raise ValueError(f"expected one output image from the graph's non-intermediate nodes, found {len(names)}")
    return names[0]

"""A run as one zip to attach elsewhere: report, summary, run data, graphs and traces, without this machine's paths.

The run directory stays as it is, apart from the default zip. The report and summary are rendered for the zip. Every
image they show is copied in with its pixels unchanged and its metadata removed, so a recipient can recompute PSNR.
Logs, the heartbeat, inputs and images the report does not show stay out. `Sanitizer` says what is removed from the
rest. `bundle.json` tells a reader, e.g. an upload hub, the zip's format version.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import urllib.parse
import zipfile
from pathlib import Path, PureWindowsPath
from typing import Any, NamedTuple

from scripts.family_e2e.graphs import (
    EXPORT_REQUEST_FILE,
    is_public_host,
    is_public_url,
    model_identifiers,
    shareable_source,
)
from scripts.family_e2e.manifest import ManifestError, load_manifest
from scripts.family_e2e.records import (
    RECORDS_FILE,
    RUN_INFO_FILE,
    Environment,
    Record,
    Reference,
    RunInfo,
    load_records,
    load_run_info,
)
from scripts.family_e2e.references import REFERENCE_FILE, load_reference
from scripts.family_e2e.report import DIFF_DIR, REPORT_FILE, SUMMARY_FILE, render_report
from scripts.family_e2e.runner import EXPORT_DIR, now_iso

BUNDLE_FILE = "bundle.json"
FORMAT_VERSION = 2
"""The layout of the zip and its files; a reader refuses a version it does not know. 2: the CHANGED and BASELINE
case statuses."""
GRAPHS_DIR = "graphs"
TRACES_DIR = "traces"
REFERENCE_DIR = "reference"
"""Per case key, the accepted reference's image.png and reference.json."""
REDACTED = "<redacted>"
PROCESS = "<process>"
INTERRUPTED = "interrupted"
"""How the bundle names a run that run.json calls running: its runner is gone without recording it."""
# Where a model lives on this machine, and the verbatim answer of the API it was installed from.
LOCAL_MODEL_FIELDS = ("path", "config_path", "source_api_response")
# Names of settings and environment variables that hold credentials. The server masks the settings it knows, by
# convention only, and code_dir may be an older checkout.
CREDENTIAL_NAME = re.compile(r"token|secret|key|auth|credential|pass|api|proxy|db_url", re.IGNORECASE)
# A shorter credential would match all over the text, e.g. HF_HUB_DISABLE_IMPLICIT_TOKEN=1.
MIN_SECRET_LENGTH = 8
# The rest of a path in free text, up to the next whitespace or quote.
_PATH_TAIL = r"[^\s\"'<>|*?]*"
DRIVE_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]" + _PATH_TAIL)
# An absolute POSIX path of two or more components. Not after a word character, `.`, `:`, `/`, `~`, `-` or `>`: that
# leaves URL paths, relative paths, ratios such as MiB/s and the rest of a path after a placeholder alone. Nor the
# server's API and socket paths, which the tool's own errors quote (`GET /api/v1/... -> 404`).
POSIX_PATH = re.compile(r"(?<![\w.:/~>-])/(?!(?:api|ws)/)(?:[^\s/\"'<>|*?]+/)+" + _PATH_TAIL)
URL = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# The chunks that make up an image's pixels and colours; every other chunk is metadata. InvokeAI writes the
# generation's metadata and graph into tEXt chunks, with model configs and their paths.
PNG_IMAGE_CHUNKS = frozenset(
    {
        b"IHDR",
        b"PLTE",
        b"IDAT",
        b"IEND",
        b"tRNS",
        b"gAMA",
        b"cHRM",
        b"sRGB",
        b"iCCP",
        b"sBIT",
        b"cICP",
        b"mDCv",
        b"cLLi",
    }
)


class Root(NamedTuple):
    """A directory of this machine whose paths become `placeholder` in a bundle."""

    placeholder: str
    path: str
    keep_subpath: bool
    """Below the tool's own directories the rest of a path names nothing private; below home it can."""


class Sanitizer:
    """Removes this machine's paths, credentials and process names from what goes into a bundle.

    `text` applies, in this order: the value of a credential-named variable in `env` becomes `<VARIABLE>`; an http(s)
    URL loses its credentials, `;params`, query and fragment, and a host of the local network becomes `<local host>`;
    a path below a root becomes the root's placeholder,
    followed by the rest of the path, or by its last component only where the root does not keep subpaths (either
    separator, doubled backslashes and any case match); any other absolute drive path, and any other absolute POSIX
    path of two or more components except the server's `/api/` and `/ws/` paths, keeps only its last component.
    """

    def __init__(self, roots: list[Root], env: dict[str, str]) -> None:
        self._secrets = sorted(
            (
                (value, f"<{name}>")
                for name, value in env.items()
                if CREDENTIAL_NAME.search(name) and len(value) >= MIN_SECRET_LENGTH
            ),
            key=lambda item: -len(item[0]),
        )
        self._roots: dict[str, Root] = {}
        for root in roots:
            key = _normalized(root.path)
            # A bare drive or "/" would swallow every path.
            if key.strip("/") and not re.fullmatch(r"[a-z]:", key):
                self._roots.setdefault(key, root)
        # The longest first, so the run directory wins over the data directory that holds it.
        alternatives = "|".join(_pattern(key) for key in sorted(self._roots, key=len, reverse=True))
        self._root_paths = (
            re.compile("(" + alternatives + r")(?![\w.-])([\\/]" + _PATH_TAIL + ")?", re.IGNORECASE)
            if self._roots
            else None
        )

    def text(self, value: str) -> str:
        for secret, placeholder in self._secrets:
            value = value.replace(secret, placeholder)
        value = URL.sub(lambda match: _strip_url(match.group()), value)
        if self._root_paths is not None:
            value = self._root_paths.sub(self._replace_root, value)
        value = DRIVE_PATH.sub(lambda match: _last_component(match.group()), value)
        return POSIX_PATH.sub(lambda match: _last_component(match.group()), value)

    def _replace_root(self, match: re.Match[str]) -> str:
        root = self._roots[_normalized(match[1])]
        tail = match[2] or ""
        if root.keep_subpath or not tail.strip("\\/"):
            return root.placeholder + tail
        return root.placeholder + tail[0] + _last_component(tail)

    def json(self, value: Any, whole_paths: bool = False) -> Any:
        """Every string through `text`; with `whole_paths`, a string that is an absolute path keeps its last component."""
        if isinstance(value, str):
            return _last_component(value) if whole_paths and _is_absolute(value) else self.text(value)
        if isinstance(value, list):
            return [self.json(item, whole_paths) for item in value]
        if isinstance(value, dict):
            return {key: self.json(item, whole_paths) for key, item in value.items()}
        return value

    def environment(self, environment: Environment) -> Environment:
        """Paths keep their last component; extra_env keeps only its names, runtime_config no credentials."""
        data = environment.model_dump(mode="json")
        data["extra_env"] = dict.fromkeys(data["extra_env"], REDACTED)
        runtime_config = data["runtime_config"]
        config = runtime_config.get("config", runtime_config)
        if isinstance(config, dict):
            for key, value in config.items():
                # Only a value can be a credential; flags such as log_tokenization stay.
                if value and isinstance(value, (str, list, dict)) and CREDENTIAL_NAME.search(key):
                    config[key] = REDACTED
        # The tool and the server write every string here, so one that is an absolute path is a path.
        return Environment.model_validate(self.json(data, whole_paths=True))

    def run_info(self, info: RunInfo) -> RunInfo:
        data = self.json(info.model_dump(mode="json", exclude={"environments"}), whole_paths=True)
        data["environments"] = {arm: self.environment(env) for arm, env in info.environments.items()}
        return RunInfo.model_validate(data)

    def reference(self, reference: Reference) -> Reference:
        data = self.json(reference.model_dump(mode="json", exclude={"environment"}))
        data["environment"] = self.environment(reference.environment)
        return Reference.model_validate(data)

    def record(self, record: Record) -> Record:
        """The text rule; foreign GPU processes keep their pid and memory but not their names."""
        data = record.model_dump(mode="json")
        for process in data["memory"]["foreign"] if data["memory"] else []:
            if process["name"]:
                # The invalid reason quotes the process as `pid N (name)`.
                data["invalid_reasons"] = [
                    reason.replace(f"({process['name']})", f"({PROCESS})") for reason in data["invalid_reasons"]
                ]
                process["name"] = PROCESS
        return Record.model_validate(self.json(data))

    def graph(self, batch: Any) -> Any:
        """Model configs lose LOCAL_MODEL_FIELDS, and their source and source URL unless anyone may see them.

        The run's graph hashes are of the graphs before this.
        """
        for config in list(model_identifiers(batch)):
            for name in LOCAL_MODEL_FIELDS:
                config.pop(name, None)
            if shareable_source(config) is None:
                config.pop("source", None)
            if not is_public_url(config.get("source_url")):
                config.pop("source_url", None)
        return self.json(batch)


def _normalized(path: str) -> str:
    return re.sub(r"[\\/]+", "/", path).rstrip("/").lower()


def _pattern(normalized: str) -> str:
    return r"[\\/]+".join(re.escape(part) for part in normalized.split("/"))


def _is_absolute(value: str) -> bool:
    return PureWindowsPath(value).is_absolute() or value.startswith("/")


def _last_component(path: str) -> str:
    return re.split(r"[\\/]+", path.rstrip("\\/"))[-1]


def _parent(path: str) -> str:
    return "/".join(re.split(r"[\\/]+", path.rstrip("\\/"))[:-1])


def _strip_url(url: str) -> str:
    try:
        parts = urllib.parse.urlsplit(url)
        host = parts.netloc.rpartition("@")[2] if is_public_host(parts.hostname) else "<local host>"
    except ValueError:
        return "<url>"
    return urllib.parse.urlunsplit((parts.scheme, host, parts.path.split(";")[0], "", ""))


def _png_without_metadata(data: bytes) -> bytes:
    """Only the PNG_IMAGE_CHUNKS, byte for byte, so the decoded pixels and colours stay exactly the same."""
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("not a PNG file")
    kept, offset = [PNG_SIGNATURE], len(PNG_SIGNATURE)
    while offset + 8 <= len(data):
        kind = data[offset + 4 : offset + 8]
        end = offset + 12 + int.from_bytes(data[offset : offset + 4], "big")
        if end > len(data):
            break
        if kind in PNG_IMAGE_CHUNKS:
            kept.append(data[offset:end])
        if kind == b"IEND":
            return b"".join(kept)
        offset = end
    raise ValueError("the PNG ends before its IEND chunk")


class _Links:
    """Names each image the report shows inside the zip: its path below the directory it comes from, plus a prefix."""

    def __init__(self, sources: list[tuple[Path, str]]) -> None:
        self._sources = sources
        self.files: dict[str, Path] = {}

    def href(self, path: Path) -> str:
        for directory, prefix in self._sources:
            if path.is_relative_to(directory):
                name = prefix + path.relative_to(directory).as_posix()
                self.files[name] = path
                return urllib.parse.quote(name)
        raise ValueError(f"the report shows {path}, which lies outside the directories a bundle takes images from")


def _model_roots(run_dir: Path) -> list[Root]:
    """The local paths of the models in the run's graphs and in its exporter request, which lists every model.

    Exact paths cover a model path with spaces in an error message. Below a model's own path the rest stays; below
    the directory holding it, only the last component.
    """
    sources = sorted((run_dir / GRAPHS_DIR).glob("*.json"))
    request = run_dir / EXPORT_DIR / EXPORT_REQUEST_FILE
    if request.is_file():
        sources.append(request)
    roots = []
    for source in sources:
        for config in model_identifiers(json.loads(source.read_text(encoding="utf-8"))):
            fields = (
                ("path", "config_path", "source") if config.get("source_type") == "path" else ("path", "config_path")
            )
            for field in fields:
                value = config.get(field)
                if isinstance(value, str) and _is_absolute(value):
                    roots.append(Root(f"<model {config.get('name') or config['key']}>", value, keep_subpath=True))
                    roots.append(Root("<model dir>", _parent(value), keep_subpath=False))
    return roots


def _references(data_dir: Path, info: RunInfo, records: list[Record]) -> dict[str, Reference]:
    """The accepted references of the run's cases, loaded as the report loads them; unreadable ones are left out."""
    references = {}
    for case_key in dict.fromkeys([step.case_key for step in info.plan] + [record.case_key for record in records]):
        try:
            loaded = load_reference(data_dir, info.profile, case_key)
        except (OSError, ValueError):
            continue
        if loaded is not None:
            references[case_key] = loaded[0]
    return references


def _sanitizer(
    data_dir: Path, run_dir: Path, info: RunInfo, references: list[Reference], manifest_path: Path | None
) -> Sanitizer:
    # A reference's environment comes from an earlier run and may hold other paths and credentials.
    environments = [*info.environments.values(), *(reference.environment for reference in references)]
    roots = [Root("<run_dir>", str(run_dir), True), Root("<data_dir>", str(data_dir), True)]
    roots += [Root("<code_dir>", environment.code_dir, True) for environment in environments]
    try:
        manifest = load_manifest(manifest_path or Path(info.manifest_path))
    except ManifestError:
        pass  # the manifest is gone: its template root is caught as a drive path only
    else:
        roots.append(Root("<template_root>", str(manifest.server.template_root), True))
    roots.append(Root("<home>", str(Path.home()), False))
    env = {name: value for environment in environments for name, value in environment.extra_env.items()}
    return Sanitizer(roots + _model_roots(run_dir), env)


def write_bundle(data_dir: Path, run_dir: Path, output: Path, manifest_path: Path | None = None) -> int:
    """Writes the bundle of the run to `output` and returns its size in bytes.

    The run's runner must be gone. `manifest_path` overrides the manifest recorded in run.json, whose template root
    is redacted too.
    """
    info = load_run_info(run_dir)
    run_records = load_records(run_dir)
    references = _references(data_dir, info, run_records)
    sanitizer = _sanitizer(data_dir, run_dir, info, list(references.values()), manifest_path)
    bundled_info = sanitizer.run_info(info)
    records = [sanitizer.record(record) for record in run_records]
    with tempfile.TemporaryDirectory(prefix="family-e2e-bundle-") as staging:
        links = _Links(
            [
                (run_dir, ""),
                (Path(staging), ""),
                (data_dir / "references" / info.profile, f"{REFERENCE_DIR}/"),
                (data_dir / "runs", "previous/"),
            ]
        )
        report = render_report(
            data_dir,
            run_dir,
            bundled_info,
            records,
            diff_dir=Path(staging) / DIFF_DIR,
            href=links.href,
            scrubber=sanitizer,
            status=INTERRUPTED if info.status == "running" else None,
        )
        texts = {
            BUNDLE_FILE: json.dumps(
                {"format_version": FORMAT_VERSION, "tool_commit": info.tool_commit, "created_at": now_iso()}, indent=2
            ),
            REPORT_FILE: report.html,
            SUMMARY_FILE: report.summary,
            RUN_INFO_FILE: bundled_info.model_dump_json(indent=2),
            RECORDS_FILE: "".join(record.model_dump_json() + "\n" for record in records),
        }
        for path in sorted((run_dir / GRAPHS_DIR).glob("*.json")):
            graph = sanitizer.graph(json.loads(path.read_text(encoding="utf-8")))
            texts[f"{GRAPHS_DIR}/{path.name}"] = json.dumps(graph, indent=2)
        for case_key, reference in references.items():
            texts[f"{REFERENCE_DIR}/{case_key}/{REFERENCE_FILE}"] = sanitizer.reference(reference).model_dump_json(
                indent=2
            )
        files = {f"{TRACES_DIR}/{path.name}": path for path in sorted((run_dir / TRACES_DIR).glob("*.csv.gz"))}
        _write_zip(output, info.run_id, texts, links.files | files)
    return output.stat().st_size


def _write_zip(output: Path, folder: str, texts: dict[str, str], files: dict[str, Path]) -> None:
    """Writes the zip, its members under `folder/`, beside `output` first: a failure leaves no partial bundle.

    `files` are compressed already (PNG, gzip) and are stored as they are, PNGs without their metadata.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=output.parent, prefix=f".{output.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, text in texts.items():
                archive.writestr(f"{folder}/{name}", text)
            for name, path in files.items():
                data = path.read_bytes()
                if name.endswith(".png"):
                    try:
                        data = _png_without_metadata(data)
                    except ValueError as e:
                        raise ValueError(f"cannot bundle {name}: {e}") from None
                archive.writestr(f"{folder}/{name}", data, zipfile.ZIP_STORED)
        os.replace(tmp, output)
    finally:
        tmp.unlink(missing_ok=True)

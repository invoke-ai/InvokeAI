"""Capture a checkpoint's key layout as a test fixture module, without reading its weights.

The loader conformance work needs real key layouts: invented names fall through the converters'
format probes as "no match", the converter becomes a no-op, and a test written against them asserts
nothing. But everything such a fixture records -- key names, shapes, dtypes, scale spellings, the
`_quantization_metadata` header block and the per-layer `.comfy_quant` markers -- lives in the
safetensors *header*, which is a few kilobytes at the front of the file. So a layout nobody here
owns is still a layout that can be reproduced: point this at a URL and it reads the header over an
HTTP range request.

Two byte sources, one parser. The local source is deliberately not `safe_open`: a second reading
path would leave the remote one -- the only one that can reach a file we do not have -- as the
untested half.

The output is a Python module in the shape of the fixtures beside
`tests/backend/model_manager/load/state_dicts/`, not JSON, because that is what the tests import.
Its docstring is a stub: it records where the layout came from and what was kept, and leaves the
part no tool can write -- what the layout *means*, and which loader property depends on it. Run
`ruff format` on the result; the checked-in fixtures are formatted.

Usage:

    uv run --no-sync python scripts/capture_state_dict_fixture.py <source> --out <module.py>

`<source>` is a local path, a full `https://...` URL, or `<org>/<repo>/<path-in-repo>`, which is
resolved against Hugging Face. A local path wins when the file exists.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

import httpx
from huggingface_hub.utils import build_hf_headers

from invokeai.backend.quantization.fp8_scaled import COMFY_QUANT_SUFFIX, parse_quantization_metadata
from invokeai.backend.quantization.int8_convrot import parse_comfy_quant_bytes

# Not read from the environment on purpose: `HF_ENDPOINT` often points at a caching proxy here, and
# proxies answer range reads inconsistently (401 on some Comfy-Org files). `build_hf_headers()` does
# read the environment, which is what supplies the token.
HF_ENDPOINT = "https://huggingface.co"

# A sanity bound on the declared header length. The largest checkpoint reachable from this tree
# declares well under a megabyte; anything past this is a file that is not a safetensors.
MAX_HEADER_BYTES = 64 * 1024 * 1024


class ByteSource(Protocol):
    """A file the header can be read out of, byte range by byte range."""

    def read(self, offset: int, length: int) -> bytes: ...

    def close(self) -> None: ...


class LocalFile:
    """A file on disk. The handle stays open: a capture with markers does one read per marker."""

    def __init__(self, path: Path) -> None:
        self._handle = path.open("rb")
        self.display = path.as_posix()

    def read(self, offset: int, length: int) -> bytes:
        self._handle.seek(offset)
        return self._handle.read(length)

    def close(self) -> None:
        self._handle.close()

    def __str__(self) -> str:
        return self.display


class RemoteFile:
    """A file behind an HTTP server that honours `Range`, which Hugging Face's CDN does.

    Nothing but the requested ranges is transferred, so a 20 GiB checkpoint costs a few kilobytes.
    """

    def __init__(self, url: str, display: str | None = None) -> None:
        self._url = url
        self.display = display or url
        self._client = httpx.Client(follow_redirects=True, timeout=180, headers=build_hf_headers())

    def read(self, offset: int, length: int) -> bytes:
        response = self._client.get(self._url, headers={"Range": f"bytes={offset}-{offset + length - 1}"})
        response.raise_for_status()
        if response.status_code != httpx.codes.PARTIAL_CONTENT:
            # 200 means the server ignored `Range` and sent the body -- for a 14 GB checkpoint that
            # is the entire download this tool exists to avoid, already buffered in memory by the
            # time we could notice. Refuse rather than continue: the capture would "work".
            raise SystemExit(
                f"{self._url} answered {response.status_code} to a range request instead of 206; "
                "the server or a redirect target does not honour `Range`, so reading its header "
                "would mean downloading the whole file"
            )
        return response.content

    def close(self) -> None:
        self._client.close()

    def __str__(self) -> str:
        return self.display


def _looks_like_a_path(source: str) -> bool:
    """Whether a source that does not exist was nonetheless meant as a file rather than a repo.

    Without this a typo in a local path has three or more segments and falls through to the
    repository branch, so `S:/ModelStuff3/typo.safetensors` is fetched from
    `huggingface.co/S:/ModelStuff3/...` and the operator is told about an HTTP status instead of a
    missing file. A repository id has none of these shapes.
    """
    return source.startswith(("/", "./", "../", "~")) or "\\" in source or (len(source) > 1 and source[1] == ":")


def open_source(source: str) -> ByteSource:
    """A local path if it exists, a URL as given, otherwise `<org>/<repo>/<path>` on Hugging Face."""
    if source.startswith(("http://", "https://")):
        return RemoteFile(source)
    path = Path(source)
    if path.exists():
        return LocalFile(path)
    if _looks_like_a_path(source):
        raise SystemExit(f"no such file: {source}")
    parts = source.split("/")
    if len(parts) < 3:
        raise SystemExit(f"{source!r} is neither an existing path, a URL, nor <org>/<repo>/<path-in-repo>")
    repo, in_repo = "/".join(parts[:2]), "/".join(parts[2:])
    # The fixture cites the repository, not the resolve URL: that is how a reader finds the file
    # again, and it is the form the hand-written fixtures beside it already use.
    return RemoteFile(f"{HF_ENDPOINT}/{repo}/resolve/main/{in_repo}", display=source)


@dataclass(frozen=True)
class TensorEntry:
    """One header entry. `begin`/`end` are absolute file offsets, not the header's relative ones."""

    dtype: str
    shape: list[int]
    begin: int
    end: int


def read_header(source: ByteSource) -> tuple[dict[str, TensorEntry], dict[str, str]]:
    """The safetensors header: eight bytes of little-endian length, then that much JSON."""
    (header_length,) = struct.unpack("<Q", source.read(0, 8))
    if not 0 < header_length <= MAX_HEADER_BYTES:
        # The length is read from the file, so a URL that is not a safetensors -- an HTML error
        # page, an LFS pointer -- gives a nonsense one, and the next read asks for that many bytes.
        raise SystemExit(
            f"{source} declares a {header_length}-byte header; that is not a safetensors file "
            f"(the limit here is {MAX_HEADER_BYTES} bytes, and the largest checkpoint in this tree "
            "declares under a megabyte)"
        )
    header = json.loads(source.read(8, header_length))
    metadata = header.pop("__metadata__", None) or {}
    data_start = 8 + header_length
    tensors = {
        key: TensorEntry(
            dtype=entry["dtype"],
            shape=list(entry["shape"]),
            begin=data_start + entry["data_offsets"][0],
            end=data_start + entry["data_offsets"][1],
        )
        for key, entry in header.items()
    }
    return tensors, metadata


def read_markers(source: ByteSource, tensors: Mapping[str, TensorEntry]) -> dict[str, dict[str, Any]]:
    """Decode every `<layer>.comfy_quant` blob, keyed by the layer path it names.

    These are 64 to 89 bytes each, so reading them is cheap even over the network -- and it is the
    difference between a fixture that records the marker and one that only records its length.
    """
    markers: dict[str, dict[str, Any]] = {}
    for key, entry in tensors.items():
        if not key.endswith(COMFY_QUANT_SUFFIX):
            continue
        markers[key[: -len(COMFY_QUANT_SUFFIX)]] = parse_comfy_quant_bytes(
            source.read(entry.begin, entry.end - entry.begin)
        )
    return markers


_INDEX = re.compile(r"^\d+$")

# How many distinct indices a prefix needs before it counts as a repeated block stack rather than
# an `nn.Sequential`. Four separates every stack in the captured checkpoints from every Sequential.
MIN_STACK = 4


def stack_indices(keys: Iterable[str]) -> dict[str, set[int]]:
    """Every integer path segment, grouped by the path that precedes it.

    A checkpoint has more than one stack -- FLUX keeps `double_blocks` and `single_blocks` -- and
    they are numbered independently, so the grouping is by prefix rather than by depth.
    """
    found: dict[str, set[int]] = defaultdict(set)
    for key in keys:
        parts = key.split(".")
        for position, part in enumerate(parts):
            if _INDEX.match(part):
                found[".".join(parts[:position])].add(int(part))
    return found


def choose_indices(keys: Iterable[str], block: int | None) -> dict[str, int]:
    """Which index to keep per stack.

    Not every integer segment is a stack. `nn.Sequential` numbers its children too, and those are
    parts of one module rather than repetitions of it -- Krea-2's `tmlp.0`/`tmlp.2` and Anima's
    `adaln_modulation_cross_attn.1`/`.2` are both halves of a thing, and a capture that kept only
    the first would describe a module the file does not contain. A repeated block stack is long, a
    Sequential is two or three deep, so `MIN_STACK` separates them. Where that heuristic is wrong --
    a decoder with four or more Sequential children -- `--no-subset` keeps the file whole, and the
    rendered docstring always names which prefixes were treated as stacks, so the call is visible.

    `block` is a preference, not a demand: a stack that does not have it keeps its lowest index and
    says so. Stacks are different lengths -- asking for block 24 of a 38-deep stack is reasonable
    and of a 19-deep one is not -- and refusing the whole capture over that would help nobody.
    """
    chosen: dict[str, int] = {}
    for prefix, indices in stack_indices(keys).items():
        if len(indices) < MIN_STACK:
            continue
        if block is not None and block not in indices:
            print(
                f"note: {prefix or '<root>'} has no index {block}; keeping {min(indices)} instead",
                file=sys.stderr,
            )
        chosen[prefix] = block if block is not None and block in indices else min(indices)
    return chosen


def select_keys(keys: Iterable[str], chosen: Mapping[str, int]) -> list[str]:
    """Keep the chosen index of every stack, and every key that is not in one."""
    kept = []
    for key in keys:
        parts = key.split(".")
        if all(
            int(part) == chosen[".".join(parts[:position])]
            for position, part in enumerate(parts)
            if _INDEX.match(part) and ".".join(parts[:position]) in chosen
        ):
            kept.append(key)
    return kept


def _py(value: Any) -> str:
    """Render a JSON-ish value as a Python literal. `bool` before `int`: it is a subclass."""
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        # `json.loads` accepts NaN/Infinity, whose `repr` is a bare `nan`/`inf` -- a NameError when
        # the fixture is imported. No producer writes them; refusing beats emitting a broken module.
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            raise SystemExit(f"header carries a non-finite value ({value!r}), which no fixture can record")
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_py(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_py(key)}: {_py(item)}" for key, item in value.items()) + "}"
    raise TypeError(f"cannot render {type(value).__name__} into a fixture module")


def _docstring_safe(text: str) -> str:
    """Neutralise anything in a source name that a docstring would read as syntax."""
    return text.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')


def render_module(
    *,
    source: str,
    tensors: Mapping[str, TensorEntry],
    chosen: Mapping[str, int],
    layer_hints: Mapping[str, Any],
    markers: Mapping[str, Any],
) -> str:
    """The fixture module, in the shape the state_dicts fixtures beside it already have."""
    if chosen:
        rule = ", ".join(f"`{prefix or '<root>'}` index {index}" for prefix, index in sorted(chosen.items()))
        rule = f"{rule}, plus every key outside a stack"
    else:
        rule = "none -- every key in the file"

    lines = [
        '"""TODO: one line naming the build this is the key layout of.',
        "",
        f"Captured on {datetime.date.today().isoformat()} with `scripts/capture_state_dict_fixture.py`,",
        "which reads the header only, from",
        f"`{_docstring_safe(source)}`.",
        "",
        f"Subsetting rule: {rule}. Values are `(shape, dtype)`.",
        "",
        "TODO: what this layout *means* -- which loader property depends on it, and what a re-upload",
        "could change without any loader code moving. A fixture without that is a wall of keys.",
        '"""',
        "",
        "state_dict_keys: dict[str, tuple[list[int], str]] = {",
    ]
    for key in sorted(tensors):
        entry = tensors[key]
        lines.append(f"    {_py(key)}: ({_py(entry.shape)}, {_py(entry.dtype)}),")
    lines.append("}")

    if layer_hints:
        lines += [
            "",
            "# The `_quantization_metadata` header block, layer names exactly as the producer wrote them.",
            "layer_hints: dict[str, dict[str, object]] = {",
        ]
        lines += [f"    {_py(name)}: {_py(hint)}," for name, hint in sorted(layer_hints.items())]
        lines.append("}")

    if markers:
        lines += [
            "",
            "# The per-layer `.comfy_quant` markers, decoded from the blobs themselves.",
            "markers: dict[str, dict[str, object]] = {",
        ]
        lines += [f"    {_py(name)}: {_py(marker)}," for name, marker in sorted(markers.items())]
        lines.append("}")

    return "\n".join(lines) + "\n"


def capture(source_argument: str, *, block: int | None, subset: bool, want_markers: bool) -> str:
    source = open_source(source_argument)
    try:
        tensors, metadata = read_header(source)
        chosen = choose_indices(tensors, block) if subset else {}
        kept = set(select_keys(tensors, chosen)) if chosen else set(tensors)
        subsetted = {key: entry for key, entry in tensors.items() if key in kept}
        markers = read_markers(source, subsetted) if want_markers else {}
    finally:
        source.close()

    hints = parse_quantization_metadata(metadata)
    layer_hints = {name: hint for name, hint in hints.items() if _hint_survived(name, kept)}
    if hints and not any(_hint_survived(name, set(tensors)) for name in hints):
        # Not the subset's doing: no header name matches any tensor in the whole file. That is the
        # header-to-tensor namespace drift `strip_layer_path_prefix` exists for, and a capture that
        # said nothing would record the build as having no header block at all.
        print(
            f"note: none of the {len(hints)} header layer names match a tensor in this file; "
            "the fixture will record no layer_hints",
            file=sys.stderr,
        )
    return render_module(
        source=str(source),
        tensors=subsetted,
        chosen=chosen,
        layer_hints=layer_hints,
        markers=markers,
    )


def _hint_survived(name: str, kept: set[str]) -> bool:
    """A header hint names a module, not a tensor, so it is kept when any of its tensors was.

    The header keeps the producer's own naming, which is the whole point of recording it: it is the
    namespace the loader has to re-key from, and a capture that silently dropped the mismatching
    ones would hide exactly the drift the fixture exists to show.
    """
    return any(key == name or key.startswith(f"{name}.") for key in kept)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="local path, https:// URL, or <org>/<repo>/<path-in-repo>")
    parser.add_argument("--out", type=Path, help="module to write; prints to stdout when omitted")
    parser.add_argument(
        "--block",
        type=int,
        default=None,
        help="stack index to keep (default: the lowest each stack has)",
    )
    parser.add_argument("--no-subset", action="store_true", help="keep every key, stacks and all")
    parser.add_argument("--no-markers", action="store_true", help="skip the .comfy_quant blob reads")
    args = parser.parse_args()

    module = capture(
        args.source,
        block=args.block,
        subset=not args.no_subset,
        want_markers=not args.no_markers,
    )
    if args.out is None:
        print(module, end="")
    else:
        args.out.write_text(module, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

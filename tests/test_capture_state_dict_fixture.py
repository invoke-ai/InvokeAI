"""`scripts/capture_state_dict_fixture.py` against real safetensors files.

The tool exists so a key layout can be recorded without reading a checkpoint's weights, including
one that is not on this machine. Everything it reads it reads by byte offset, so the cells here
write a real file and read it back rather than stubbing the header: an off-by-one in an offset is
the failure this tool can have, and a mocked reader would not have it.

The remote half gets the same treatment. Its cell serves the same file's bytes through an httpx
transport that honours `Range`, and asserts the two paths produce the identical module -- the point
being that the source is swappable and the parser is one.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import httpx
import pytest
import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parent.parent
MARKER = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
HINTS = {"blocks.0.attn.qkv": {"format": "float8_e4m3fn", "full_precision_matrix_mult": True}}


def _load_module(path: Path, name: str) -> ModuleType:
    """Registered in `sys.modules` before execution: `@dataclass` resolves its own module by name."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tool = _load_module(REPO_ROOT / "scripts" / "capture_state_dict_fixture.py", "capture_state_dict_fixture")


def _blob(payload: dict) -> torch.Tensor:
    return torch.frombuffer(bytearray(json.dumps(payload).encode("utf-8")), dtype=torch.uint8).clone()


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    """A file with two stacks of different depth and starting index, a Sequential, and a marker.

    The stacks are five deep so they clear `MIN_STACK`; `refiner` starts at 2 so "keep the lowest"
    is distinguishable from "keep index 0". `mlp.0`/`mlp.2` is the Sequential shape that must not be
    mistaken for a stack. The marker is followed by more tensors, so a slice that reads one byte too
    few or too many lands in a neighbour instead of running off the end of the file.
    """
    tensors: dict[str, torch.Tensor] = {}
    for index in range(5):
        tensors[f"blocks.{index}.attn.qkv.weight"] = torch.zeros(6, 4, dtype=torch.int8)
        tensors[f"blocks.{index}.attn.qkv.weight_scale"] = torch.zeros(6, 1, dtype=torch.float32)
        tensors[f"blocks.{index}.attn.qkv.comfy_quant"] = _blob(MARKER)
        tensors[f"blocks.{index}.mlp.0.weight"] = torch.zeros(4, 4, dtype=torch.bfloat16)
        tensors[f"blocks.{index}.mlp.2.weight"] = torch.zeros(4, 4, dtype=torch.bfloat16)
    for index in range(2, 7):
        tensors[f"refiner.{index}.norm.scale"] = torch.zeros(4, dtype=torch.float32)
    tensors["final.weight"] = torch.zeros(2, 4, dtype=torch.float8_e4m3fn)
    tensors["final.weight_scale"] = torch.zeros((), dtype=torch.float32)
    # One tensor materially larger than a marker, so "no request covers a tensor" has something to
    # be wrong about: every other tensor here is small enough to slip under any sane budget.
    tensors["patch_embedding.weight"] = torch.zeros(128, 128, dtype=torch.bfloat16)

    path = tmp_path / "toy.safetensors"
    save_file(tensors, path, metadata={"_quantization_metadata": json.dumps({"layers": HINTS})})
    return path


@pytest.fixture
def range_server(checkpoint: Path, monkeypatch: pytest.MonkeyPatch):
    """Serve the real file's bytes over `Range`, recording every request the tool makes.

    The recording is the point: what this tool must never do is read a tensor, and the only
    evidence of that is the set of ranges it asked for.
    """
    payload = checkpoint.read_bytes()

    def install(status: int = 206) -> list[tuple[int, int]]:
        seen: list[tuple[int, int]] = []

        def serve(request: httpx.Request) -> httpx.Response:
            first, last = (int(part) for part in request.headers["Range"].removeprefix("bytes=").split("-"))
            seen.append((first, last))
            body = payload if status == 200 else payload[first : last + 1]
            return httpx.Response(status, content=body)

        real_client = httpx.Client
        monkeypatch.setattr(
            tool.httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(serve), **kwargs)
        )
        return seen

    return install


def _undated(module: str) -> str:
    """The rendered docstring carries today's date, which differs between two captures across
    midnight. Everything else about them has to be identical."""
    return chr(10).join(line for line in module.splitlines() if not line.startswith("Captured on "))


def _capture(path: Path, **kwargs) -> str:
    options = {"block": None, "subset": True, "want_markers": True} | kwargs
    return tool.capture(str(path), **options)


def test_the_header_reports_every_key_with_its_shape_and_dtype_spelling(checkpoint: Path) -> None:
    """The dtype strings are safetensors' own, because that is what the fixtures record.

    Inventing a spelling here would make every captured fixture disagree with the eleven that were
    written by hand, and the disagreement would only show up in whichever test read both.
    """
    source = tool.open_source(str(checkpoint))
    try:
        tensors, metadata = tool.read_header(source)
    finally:
        source.close()

    assert tensors["blocks.0.attn.qkv.weight"].shape == [6, 4]
    assert tensors["blocks.0.attn.qkv.weight"].dtype == "I8"
    assert tensors["blocks.0.mlp.0.weight"].dtype == "BF16"
    assert tensors["final.weight"].dtype == "F8_E4M3"
    assert tensors["final.weight_scale"].shape == []
    assert tensors["blocks.0.attn.qkv.comfy_quant"].dtype == "U8"
    assert json.loads(metadata["_quantization_metadata"])["layers"] == HINTS


def test_a_marker_is_decoded_from_the_bytes_its_header_entry_points_at(checkpoint: Path) -> None:
    """Reading the blob is the whole reason the offsets are absolute rather than header-relative.

    The fixtures that exist today reconstruct the marker by hand and can only assert its *length*,
    which cannot tell `convrot: true` from `convrot: false`.
    """
    source = tool.open_source(str(checkpoint))
    try:
        tensors, _metadata = tool.read_header(source)
        markers = tool.read_markers(source, tensors)
    finally:
        source.close()

    assert markers["blocks.0.attn.qkv"] == MARKER
    assert set(markers) == {f"blocks.{index}.attn.qkv" for index in range(5)}


def test_the_lowest_index_of_each_stack_is_kept_and_a_sequential_is_not_one(checkpoint: Path) -> None:
    """`refiner` starts at 2, so this fails for an implementation that hard-codes index 0.

    `mlp.0`/`mlp.2` are two halves of one module. Subsetting them would describe a model the file
    does not contain -- and it would do so silently, since the result is still valid Python.
    """
    module = _capture(checkpoint)

    assert '"blocks.0.attn.qkv.weight"' in module
    assert '"blocks.1.attn.qkv.weight"' not in module
    assert '"refiner.2.norm.scale"' in module
    assert '"refiner.3.norm.scale"' not in module
    assert '"blocks.0.mlp.0.weight"' in module
    assert '"blocks.0.mlp.2.weight"' in module


def test_a_requested_block_is_kept_where_the_stack_has_it(checkpoint: Path) -> None:
    """Anima's captured fixture holds block 24, not block 0, because that is where its only
    `full_precision_matrix_mult` layer sits. Choosing the block is the normal case, not an escape."""
    module = _capture(checkpoint, block=3)

    assert '"blocks.3.attn.qkv.weight"' in module
    assert '"blocks.0.attn.qkv.weight"' not in module


def test_a_stack_without_the_requested_block_keeps_its_own_lowest(checkpoint: Path, capsys) -> None:
    """One `--block` meets stacks of different depths. Refusing the capture would help nobody, so
    the stack that cannot honour it says what it did instead -- on stderr and in the docstring."""
    module = _capture(checkpoint, block=0)

    assert '"blocks.0.attn.qkv.weight"' in module
    assert '"refiner.2.norm.scale"' in module
    assert "`refiner` index 2" in module
    assert "has no index 0; keeping 2 instead" in capsys.readouterr().err


def test_the_rendered_module_imports_back_to_what_was_captured(checkpoint: Path, tmp_path: Path) -> None:
    """The output is a Python module the tests import, so "it is valid Python" is a property of the
    product and not of the renderer -- a stray quote in a key name would only show up here."""
    written = tmp_path / "captured_keys.py"
    written.write_text(_capture(checkpoint), encoding="utf-8")

    fixture = _load_module(written, "captured_keys")

    assert fixture.state_dict_keys["blocks.0.attn.qkv.weight"] == ([6, 4], "I8")
    assert fixture.state_dict_keys["final.weight_scale"] == ([], "F32")
    assert fixture.markers["blocks.0.attn.qkv"] == MARKER
    assert fixture.layer_hints == HINTS


def test_hints_and_markers_are_absent_when_the_file_carries_none(tmp_path: Path) -> None:
    """Krea-2's int8 build has no `__metadata__` at all. Emitting an empty `layer_hints` there would
    read as "the header names no layers" rather than "there is no header block"."""
    path = tmp_path / "plain.safetensors"
    save_file({"final.weight": torch.zeros(2, 4, dtype=torch.bfloat16)}, path)

    module = _capture(path)

    assert "state_dict_keys" in module
    assert "layer_hints: " not in module
    assert "markers: " not in module


def test_a_hint_naming_a_dropped_block_does_not_survive_the_subset(checkpoint: Path, tmp_path: Path) -> None:
    """A header hint names a module; it is kept exactly when one of that module's tensors was.

    Left in, it would name a layer the fixture no longer contains, and a test pairing hints with
    weights would read that as a lost flag -- the defect this fixture family exists to catch.
    """
    path = tmp_path / "later_block.safetensors"
    save_file(
        {key: torch.zeros(2, 2, dtype=torch.bfloat16) for key in (f"blocks.{i}.attn.qkv.weight" for i in range(5))},
        path,
        metadata={"_quantization_metadata": json.dumps({"layers": {"blocks.4.attn.qkv": {"format": "x"}}})},
    )

    assert "layer_hints: " not in _capture(path)
    assert "layer_hints: " in _capture(path, block=4)


def test_the_remote_source_captures_the_same_module_as_the_local_one(checkpoint: Path, range_server) -> None:
    """The remote half is the one that can reach a layout nobody here owns, so it is the half that
    must not drift. Serving the same bytes through `Range` has to produce the identical module."""
    seen = range_server()

    remote = tool.capture("https://example.invalid/toy.safetensors", block=None, subset=True, want_markers=True)

    assert seen[0] == (0, 7)
    assert _undated(remote.replace("https://example.invalid/toy.safetensors", checkpoint.as_posix())) == _undated(
        _capture(checkpoint)
    )


def test_a_source_name_cannot_close_the_docstring_it_lands_in(range_server) -> None:
    """The source name is written into the generated module's docstring, so it is code, not text.

    A local path is rendered with posix separators and so cannot carry an escape, but a URL is
    whatever the operator passed. Driven through the remote source because that is the one whose
    name the tool does not normalise.
    """
    range_server()

    module = tool.capture(r'https://example.invalid/a"""b\c.safetensors', block=None, subset=True, want_markers=False)

    compile(module, "<captured>", "exec")


def test_a_bare_repo_path_resolves_against_hugging_face(tmp_path: Path) -> None:
    """`<org>/<repo>/<path>` is how the checkpoints are named everywhere else in this work. An
    existing local path wins, so a relative filename is never mistaken for a repository."""
    argument = "Comfy-Org/Wan_2.2_Repackaged/split_files/diffusion_models/wan.safetensors"
    source = tool.open_source(argument)
    try:
        assert (
            source._url
            == f"https://huggingface.co/Comfy-Org/Wan_2.2_Repackaged/resolve/main/{argument.split('/', 2)[2]}"
        )
        # The docstring cites the repository, which is what a reader needs to find the file again.
        assert str(source) == argument
    finally:
        source.close()

    local = tmp_path / "local.safetensors"
    save_file({"w": torch.zeros(2)}, local)
    source = tool.open_source(str(local))
    try:
        assert isinstance(source, tool.LocalFile)
    finally:
        source.close()


def test_only_the_markers_that_survived_the_subset_are_read(checkpoint: Path, tmp_path: Path, range_server) -> None:
    """Reading markers for dropped blocks costs a request each and emits layers the fixture does
    not contain.

    The toy file has one marker per block and five blocks; a capture that kept block 0 must ask for
    exactly one. On a 40-block checkpoint the same slip is 40 round trips and a `markers` mapping
    whose keys no `state_dict_keys` entry backs -- which is what the consumer cells assert against,
    so it would be found only after the bad fixture was already checked in.
    """
    seen = range_server()

    written = tmp_path / "subset_markers.py"
    written.write_text(
        tool.capture("https://example.invalid/toy.safetensors", block=None, subset=True, want_markers=True),
        encoding="utf-8",
    )

    assert set(_load_module(written, "subset_markers").markers) == {"blocks.0.attn.qkv"}
    assert len(seen) == 3  # the eight length bytes, the header, one marker


def test_no_request_ever_covers_a_tensor(checkpoint: Path, range_server) -> None:
    """The whole promise of the tool: a capture of a multi-gigabyte file costs kilobytes.

    Asserted as a property of the requests rather than of the runtime, so it holds for a file this
    suite is too small to contain. The header read is exempt and named; everything else is a marker.
    """
    seen = range_server()

    tool.capture("https://example.invalid/toy.safetensors", block=None, subset=True, want_markers=True)

    lengths = [last - first + 1 for first, last in seen[2:]]
    assert lengths
    assert max(lengths) < 1024, seen


def test_a_server_that_ignores_the_range_is_refused(range_server) -> None:
    """A 200 means the body is the whole file, already buffered by the time we could look at it.

    That is the one cost this tool exists to avoid, and it is invisible otherwise: the capture
    succeeds, it just downloaded 14 GB to read four kilobytes of it.
    """
    range_server(status=200)

    with pytest.raises(SystemExit, match="does not honour"):
        tool.capture("https://example.invalid/toy.safetensors", block=None, subset=True, want_markers=True)


def test_something_that_is_not_a_safetensors_is_refused_before_the_second_read(tmp_path: Path, range_server) -> None:
    """The header length comes out of the file, so an HTML error page or an LFS pointer yields a
    nonsense one -- and the next read asks for exactly that many bytes."""
    not_a_checkpoint = tmp_path / "index.html"
    not_a_checkpoint.write_bytes(b"<!DOCTYPE html><title>404</title>" + b"x" * 64)

    with pytest.raises(SystemExit, match="not a safetensors file"):
        tool.capture(str(not_a_checkpoint), block=None, subset=True, want_markers=False)


def test_a_mistyped_local_path_is_not_fetched_from_hugging_face() -> None:
    """`S:/ModelStuff3/typo.safetensors` has three segments and does not exist, so without a guard
    it becomes `huggingface.co/S:/ModelStuff3/resolve/main/typo.safetensors` and the operator is
    told about an HTTP status instead of a missing file."""
    for mistyped in (r"S:/ModelStuff3/typo.safetensors", "/home/me/models/typo.safetensors"):
        with pytest.raises(SystemExit, match="no such file"):
            tool.open_source(mistyped)


def test_a_non_finite_header_value_is_refused_rather_than_rendered(tmp_path: Path) -> None:
    """Python's JSON accepts `NaN` and `Infinity`; Python's `repr` of them is a bare name.

    Rendered into a fixture that is a `NameError` at import, i.e. a module that looks captured and
    cannot be read. No producer writes them, so this is about what happens when one does.
    """
    path = tmp_path / "nan.safetensors"
    save_file(
        {"final.weight": torch.zeros(2, 2, dtype=torch.bfloat16)},
        path,
        metadata={"_quantization_metadata": '{"layers": {"final": {"scale": NaN}}}'},
    )

    with pytest.raises(SystemExit, match="non-finite"):
        _capture(path)

"""Tests for the MP4 keyed-metadata reader/writer (mp4_metadata.py)."""

import json
import struct
from pathlib import Path

import imageio.v2 as iio2
import numpy as np
import pytest

from invokeai.app.util.mp4_metadata import (
    Mp4MetadataError,
    read_ftyp_major_brand,
    read_mp4_tags,
    write_mp4_tags,
)
from invokeai.app.util.video_encoding import make_mp4_writer


@pytest.fixture(scope="module")
def tiny_mp4(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("mp4") / "tiny.mp4"
    writer = make_mp4_writer(path, fps=8.0)
    try:
        for i in range(4):
            writer.append_data(np.full((32, 48, 3), i * 40, dtype=np.uint8))
    finally:
        writer.close()
    return path


def _box(box_type: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(payload), box_type) + payload


def _keyed_metadata_boxes(tags: dict[str, str], *, full_meta_box: bool) -> bytes:
    """A hand-built ``udta/meta`` carrying ``tags`` in either ``meta`` layout."""
    keys_payload = struct.pack(">II", 0, len(tags))
    ilst_payload = b""
    for index, (key, value) in enumerate(tags.items(), start=1):
        name = key.encode("utf-8")
        keys_payload += struct.pack(">I4s", 8 + len(name), b"mdta") + name
        data = _box(b"data", struct.pack(">II", 1, 0) + value.encode("utf-8"))
        ilst_payload += _box(index.to_bytes(4, "big"), data)
    hdlr = _box(b"hdlr", struct.pack(">II4s4s", 0, 0, b"", b"mdta") + b"\x00" * 12)
    meta_payload = (b"\x00\x00\x00\x00" if full_meta_box else b"") + hdlr + _box(b"keys", keys_payload)
    meta_payload += _box(b"ilst", ilst_payload)
    return _box(b"udta", _box(b"meta", meta_payload))


def _fake_mp4(path: Path, udta: bytes, *, moov_first: bool = True) -> Path:
    ftyp = _box(b"ftyp", b"isom" + struct.pack(">I", 512) + b"isomiso2avc1mp41")
    moov = _box(b"moov", _box(b"mvhd", b"\x00" * 100) + udta)
    mdat = _box(b"mdat", b"\xff" * 64)
    path.write_bytes(ftyp + (moov + mdat if moov_first else mdat + moov))
    return path


def test_round_trip_through_ffmpeg_preserves_every_special_character(tiny_mp4: Path, tmp_path: Path) -> None:
    metadata = json.dumps({"positive_prompt": "a=b;c#d\\e\nf", "seed": 7, "nested": {"k": [1, 2]}})
    graph = json.dumps({"nodes": {}, "edges": []})
    out = tmp_path / "tagged.mp4"

    write_mp4_tags(tiny_mp4, out, {"invokeai_metadata": metadata, "invokeai_graph": graph})

    tags = read_mp4_tags(out)
    assert tags["invokeai_metadata"] == metadata
    assert tags["invokeai_graph"] == graph
    # The remux must leave a playable H.264 stream behind and keep moov in front (faststart).
    reader = iio2.get_reader(str(out))
    try:
        assert reader.get_data(0).shape[:2] == (32, 48)
    finally:
        reader.close()
    with open(out, "rb") as fh:
        head = fh.read(4096)
    assert b"moov" in head


def test_round_trip_of_a_payload_larger_than_a_megabyte(tiny_mp4: Path, tmp_path: Path) -> None:
    graph = json.dumps({"nodes": {"x": "y" * 1_500_000}, "edges": []})
    out = tmp_path / "big.mp4"

    write_mp4_tags(tiny_mp4, out, {"invokeai_graph": graph})

    assert read_mp4_tags(out, keys=["invokeai_graph"]) == {"invokeai_graph": graph}


def test_keys_filter_returns_only_the_requested_tags(tiny_mp4: Path, tmp_path: Path) -> None:
    out = tmp_path / "filtered.mp4"
    write_mp4_tags(tiny_mp4, out, {"invokeai_metadata": "{}", "invokeai_workflow": "{}"})

    assert read_mp4_tags(out, keys=["invokeai_workflow"]) == {"invokeai_workflow": "{}"}
    assert read_mp4_tags(out, keys=["absent"]) == {}


def test_untagged_mp4_and_non_mp4_read_as_no_tags(tiny_mp4: Path, tmp_path: Path) -> None:
    assert read_mp4_tags(tiny_mp4) == {}
    garbage = tmp_path / "garbage.mp4"
    garbage.write_bytes(b"\x00\x00\x00\x18ftypmp42 not a real mp4")
    assert read_mp4_tags(garbage) == {}
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    assert read_mp4_tags(empty) == {}


@pytest.mark.parametrize("full_meta_box", [True, False], ids=["iso-fullbox-meta", "quicktime-plain-meta"])
@pytest.mark.parametrize("moov_first", [True, False], ids=["moov-first", "moov-last"])
def test_reads_both_meta_layouts_in_either_box_order(tmp_path: Path, full_meta_box: bool, moov_first: bool) -> None:
    tags = {"invokeai_metadata": '{"seed": 3}', "com.apple.quicktime.make": "Apple"}
    path = _fake_mp4(
        tmp_path / "fake.mp4", _keyed_metadata_boxes(tags, full_meta_box=full_meta_box), moov_first=moov_first
    )

    assert read_mp4_tags(path) == tags


def test_truncated_box_size_reads_as_no_tags(tmp_path: Path) -> None:
    path = _fake_mp4(tmp_path / "trunc.mp4", _keyed_metadata_boxes({"invokeai_metadata": "{}"}, full_meta_box=True))
    data = bytearray(path.read_bytes())
    # Claim the moov box is far larger than the file: the walker must not trust it.
    moov_at = data.index(b"moov") - 4
    data[moov_at : moov_at + 4] = struct.pack(">I", 1 << 30)
    path.write_bytes(bytes(data))

    assert read_mp4_tags(path) == {}


def test_oversized_tag_is_skipped_without_allocating_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tags = {"invokeai_metadata": "x" * 64, "invokeai_workflow": "{}"}
    path = _fake_mp4(tmp_path / "oversized.mp4", _keyed_metadata_boxes(tags, full_meta_box=True))
    monkeypatch.setattr("invokeai.app.util.mp4_metadata.MAX_TAG_BYTES", 32)

    assert read_mp4_tags(path) == {"invokeai_workflow": "{}"}


def test_non_utf8_data_atom_is_skipped(tmp_path: Path) -> None:
    boxes = bytearray(_keyed_metadata_boxes({"invokeai_metadata": "{}"}, full_meta_box=True))
    at = boxes.index(b"data" + struct.pack(">I", 1)) + 4  # the atom, not the "metadata" key text
    boxes[at : at + 4] = struct.pack(">I", 0)  # type indicator 0 = binary, not UTF-8 text
    path = _fake_mp4(tmp_path / "binary.mp4", bytes(boxes))

    assert read_mp4_tags(path) == {}


def test_carriage_returns_round_trip(tiny_mp4: Path, tmp_path: Path) -> None:
    # CRLF-formatted JSON from an API client reaches the writer verbatim.
    value = '{\r\n  "seed": 1\r\n}'
    out = tmp_path / "crlf.mp4"

    write_mp4_tags(tiny_mp4, out, {"invokeai_metadata": value, "invokeai_graph": "{}"})

    assert read_mp4_tags(out, keys=["invokeai_metadata", "invokeai_graph"]) == {
        "invokeai_metadata": value,
        "invokeai_graph": "{}",
    }


def test_an_empty_mapping_strips_existing_tags(tiny_mp4: Path, tmp_path: Path) -> None:
    tagged = tmp_path / "tagged.mp4"
    write_mp4_tags(tiny_mp4, tagged, {"invokeai_workflow": "{}"})

    write_mp4_tags(tagged, tmp_path / "stripped.mp4", {})

    assert read_mp4_tags(tmp_path / "stripped.mp4", keys=["invokeai_workflow"]) == {}


def test_write_rejects_tags_the_ffmetadata_grammar_cannot_carry(tiny_mp4: Path, tmp_path: Path) -> None:
    # ffmpeg reads a backslash before the line break as a continuation even when escaped, so a
    # value ending in one would swallow the following tag.
    with pytest.raises(ValueError):
        write_mp4_tags(tiny_mp4, tmp_path / "out.mp4", {"invokeai_metadata": "abc\\", "invokeai_graph": "{}"})
    with pytest.raises(ValueError):
        write_mp4_tags(tiny_mp4, tmp_path / "out.mp4", {"bad\nkey": "{}"})


def test_oversized_key_name_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tags = {"x" * 64: "long", "invokeai_workflow": "{}"}
    path = _fake_mp4(tmp_path / "longkey.mp4", _keyed_metadata_boxes(tags, full_meta_box=True))
    monkeypatch.setattr("invokeai.app.util.mp4_metadata._MAX_KEY_NAME_BYTES", 32)

    assert read_mp4_tags(path) == {"invokeai_workflow": "{}"}


def test_top_level_walk_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    udta = _keyed_metadata_boxes({"invokeai_metadata": "{}"}, full_meta_box=True)
    path = tmp_path / "junk.mp4"
    junk = _box(b"free", b"") * 100
    path.write_bytes(junk + _box(b"moov", udta) + _box(b"mdat", b""))
    monkeypatch.setattr("invokeai.app.util.mp4_metadata._MAX_TOP_LEVEL_BOXES", 50)

    assert read_mp4_tags(path) == {}


def test_write_raises_on_an_undecodable_source(tmp_path: Path) -> None:
    garbage = tmp_path / "garbage.mp4"
    garbage.write_bytes(b"\x00\x00\x00\x18ftypmp42 not a real mp4")

    with pytest.raises(Mp4MetadataError):
        write_mp4_tags(garbage, tmp_path / "out.mp4", {"invokeai_metadata": "{}"})


def test_read_ftyp_major_brand(tiny_mp4: Path, tmp_path: Path) -> None:
    assert read_ftyp_major_brand(tiny_mp4) == b"isom"
    qt = tmp_path / "qt.mov"
    qt.write_bytes(_box(b"ftyp", b"qt  " + b"\x00" * 8) + _box(b"mdat", b""))
    assert read_ftyp_major_brand(qt) == b"qt  "
    late = tmp_path / "late.mp4"
    late.write_bytes(_box(b"free", b"\x00" * (70 * 1024)) + _box(b"ftyp", b"isom" + b"\x00" * 8))
    assert read_ftyp_major_brand(late) is None
    (tmp_path / "none.bin").write_bytes(b"not boxes at all")
    assert read_ftyp_major_brand(tmp_path / "none.bin") is None

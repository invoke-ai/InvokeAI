"""Tests for the video route's HTTP Range header parser."""

import pytest

from invokeai.app.api.routers.videos import _parse_range_header

FILE_SIZE = 1000


@pytest.mark.parametrize(
    "header,expected",
    [
        ("bytes=0-499", (0, 499)),
        ("bytes=500-999", (500, 999)),
        ("bytes=500-", (500, 999)),  # open-ended
        ("bytes=-500", (500, 999)),  # suffix: last 500 bytes
        ("bytes=-2000", (0, 999)),  # suffix longer than file clamps to whole file
        ("bytes=0-1999", (0, 999)),  # end clamps to file size
        (" bytes=0-0 ", (0, 0)),  # whitespace tolerated, single byte
    ],
)
def test_valid_ranges(header: str, expected: tuple[int, int]) -> None:
    assert _parse_range_header(header, FILE_SIZE) == expected


@pytest.mark.parametrize(
    "header",
    [
        "bytes=-",  # no start, no end
        "bytes=abc-def",
        "bytes=100",  # missing dash
        "0-499",  # missing bytes= prefix
        "bytes=500-100",  # start > end
        "bytes=1000-",  # start beyond EOF
        "bytes=-0",  # zero-length suffix
        "bytes=0-499,600-699",  # multipart ranges unsupported
    ],
)
def test_malformed_or_unsatisfiable_ranges(header: str) -> None:
    assert _parse_range_header(header, FILE_SIZE) is None


@pytest.mark.parametrize("header", ["bytes=-500", "bytes=0-", "bytes=0-0"])
def test_empty_file_is_never_satisfiable(header: str) -> None:
    """Regression: a suffix range against a zero-length file used to return (0, -1),
    producing a 206 with `Content-Range: bytes 0--1/0` instead of a 416."""
    assert _parse_range_header(header, 0) is None

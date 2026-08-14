"""Unit tests for the download-URL SSRF guard."""

import http.server
import ipaddress
import logging
import threading
from typing import Any, Generator

import pytest
import requests
from requests import Request
from requests.models import PreparedRequest
from requests.utils import select_proxy

from invokeai.app.util import ssrf
from invokeai.app.util.ssrf import UnsafeDownloadURLException, build_guarded_session, validate_download_url


@pytest.fixture
def fake_dns(monkeypatch: Any):
    """Resolve hostnames from a table so the tests never touch a real resolver."""
    table: dict[str, list[str]] = {}

    def _resolve(host: str, port: int | None):
        if host not in table:
            raise OSError(f"no fake DNS record for {host}")
        return [ipaddress.ip_address(a) for a in table[host]]

    monkeypatch.setattr(ssrf, "_resolve", _resolve)
    return table


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/x",
        "http://127.1.2.3/x",
        "https://10.0.0.1/x",
        "http://172.16.4.5/x",
        "http://192.168.1.1/x",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://0.0.0.0/x",
        "http://[::1]/x",
        "http://[fe80::1]/x",
        "http://[fc00::1]/x",
        "http://[::ffff:127.0.0.1]/x",  # IPv4-mapped loopback
        "http://[::ffff:10.0.0.1]/x",
        "http://[2002:7f00:0001::]/x",  # 6to4-wrapped 127.0.0.1
        "http://0177.0.0.1/x",  # legacy octal IPv4 literal for 127.0.0.1
        "http://0177.0.0.1./x",  # trailing-dot legacy octal literal
        "http://2130706433/x",  # integer literal
        "http://100.64.0.1/x",  # RFC 6598 shared address space -- not is_private
        "http://198.18.0.1/x",  # benchmarking range -- not is_reserved
        # `requests` percent-decodes unreserved characters in the host before connecting,
        # so these reach loopback / the metadata service.
        "http://127%2e0%2e0%2e1/x",
        "http://169.254.169%2e254/latest/meta-data/",
        "http://%31%30.0.0.5/x",
        # IPv6 wrappers that reach an IPv4 destination. These are all `is_global == True`,
        # so a "not is_global" test on its own lets them through.
        "http://[64:ff9b::a9fe:a9fe]/latest/meta-data/",  # NAT64 for 169.254.169.254
        "http://[64:ff9b::a00:1]/x",  # NAT64 for 10.0.0.1
        "http://[64:ff9b:1::7f00:1]/x",  # NAT64 local-use prefix for 127.0.0.1
        "http://[::127.0.0.1]/x",  # IPv4-compatible loopback
        "http://[5f00::1]/x",  # unallocated
        "http://[ff02::1]/x",  # link-local all-nodes multicast
        # Teredo: the embedded client is public, but the wrapper itself is not global.
        "http://[2001:0:53aa:64c:0:5bfe:5f00:1]/x",
    ],
)
def test_rejects_non_public_addresses(url: str):
    with pytest.raises(UnsafeDownloadURLException):
        validate_download_url(url)


@pytest.mark.parametrize(
    "url",
    ["http://93.184.216.34/x", "https://8.8.8.8/x", "http://[2606:4700:4700::1111]/x"],
)
def test_allows_public_literals(url: str):
    validate_download_url(url)


def test_allows_public_ipv4_mapped_ipv6_literal():
    validate_download_url("http://[::ffff:8.8.8.8]/x")


def test_rejects_invalid_port_before_dns():
    with pytest.raises(UnsafeDownloadURLException, match="invalid port"):
        validate_download_url("http://example.com:65536/x")


def test_rejects_hostname_resolving_to_loopback(fake_dns: dict[str, list[str]]):
    fake_dns["localtest.me"] = ["127.0.0.1"]
    with pytest.raises(UnsafeDownloadURLException):
        validate_download_url("http://localtest.me/x")


@pytest.mark.parametrize("host", ["0177.0.0.1", "0177.0.0.1.", "2130706433", "0x7f000001", "127.0.0.1."])
def test_rejects_legacy_ipv4_literal_without_dns(monkeypatch: Any, host: str):
    """Legacy numeric IPv4 spellings must not depend on resolver normalization."""

    def fail_resolve(host: str, port: int | None) -> list[ipaddress.IPv4Address]:
        raise AssertionError(f"unexpected DNS lookup for numeric host {host}")

    monkeypatch.setattr(ssrf, "_resolve", fail_resolve)
    with pytest.raises(UnsafeDownloadURLException):
        validate_download_url(f"http://{host}/x")


def test_rejects_hostname_with_any_non_public_record(fake_dns: dict[str, list[str]]):
    """A split-horizon name must not slip through on the strength of its public record."""
    fake_dns["mixed.example.com"] = ["93.184.216.34", "127.0.0.1"]
    with pytest.raises(UnsafeDownloadURLException):
        validate_download_url("http://mixed.example.com/x")


def test_blocked_address_error_hides_resolved_ip(fake_dns: dict[str, list[str]], caplog: Any):
    fake_dns["private.example.com"] = ["10.0.0.7"]
    with pytest.raises(UnsafeDownloadURLException) as excinfo:
        validate_download_url("http://private.example.com/x")

    assert "10.0.0.7" not in str(excinfo.value)
    assert "10.0.0.7" in caplog.text


def test_allows_hostname_resolving_to_public_address(fake_dns: dict[str, list[str]]):
    fake_dns["cdn.example.com"] = ["93.184.216.34", "2606:4700:4700::1111"]
    validate_download_url("http://cdn.example.com/x")


def test_unresolvable_host_is_left_to_the_http_client(fake_dns: dict[str, list[str]]):
    """We share a resolver with `requests`; if we cannot resolve it, neither can it."""
    validate_download_url("http://no-such-host.invalid/x")


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "gopher://example.com/x"])
def test_rejects_non_http_schemes(url: str):
    with pytest.raises(UnsafeDownloadURLException):
        validate_download_url(url)


def test_opt_in_allows_private_addresses():
    validate_download_url("http://127.0.0.1/x", allow_private_urls=True)


def test_opt_in_still_rejects_bad_schemes():
    with pytest.raises(UnsafeDownloadURLException):
        validate_download_url("file:///etc/passwd", allow_private_urls=True)


# --------------- Socket-level guard: the check that survives rebinding ---------------


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = b"internal-only"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def loopback_server() -> Generator[int, None, None]:
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield srv.server_address[1]
    finally:
        srv.shutdown()


def _unsafe_in_chain(exc: BaseException) -> bool:
    seen = exc
    while seen is not None:
        if isinstance(seen, UnsafeDownloadURLException):
            return True
        seen = seen.__cause__ or seen.__context__
    return False


def test_guarded_session_refuses_loopback(loopback_server: int):
    session = build_guarded_session()
    with pytest.raises(Exception) as excinfo:
        session.get(f"http://127.0.0.1:{loopback_server}/x", timeout=5)
    assert _unsafe_in_chain(excinfo.value)


def test_guarded_session_blocks_dns_rebinding(loopback_server: int, monkeypatch: Any):
    """The up-front check sees a public record; the socket still lands on loopback.

    An attacker who controls the zone can answer our lookup and the HTTP client's lookup
    differently, so only the connected-socket check can catch this.
    """
    monkeypatch.setattr(ssrf, "_resolve", lambda host, port: [ipaddress.ip_address("93.184.216.34")])

    url = f"http://localhost:{loopback_server}/x"
    validate_download_url(url)  # up-front check is fooled, by construction

    session = build_guarded_session()
    with pytest.raises(Exception) as excinfo:
        session.get(url, timeout=5)
    assert _unsafe_in_chain(excinfo.value)


def test_guarded_session_blocks_percent_encoded_host(loopback_server: int, monkeypatch: Any):
    """Even with the up-front check disabled, `%6cocalhost` cannot reach loopback."""
    monkeypatch.setattr(ssrf, "_host_spellings", lambda host: [host])
    url = f"http://%6cocalhost:{loopback_server}/x"
    validate_download_url(url)  # unresolvable as spelled -> fails open, by construction

    session = build_guarded_session()
    with pytest.raises(Exception) as excinfo:
        session.get(url, timeout=5)
    assert _unsafe_in_chain(excinfo.value)


def test_guarded_session_is_installed_for_both_schemes():
    session = build_guarded_session()
    for prefix in ("http://", "https://"):
        adapter = session.get_adapter(prefix + "example.com")
        assert isinstance(adapter, ssrf.SsrfGuardedAdapter)
        assert adapter.poolmanager.pool_classes_by_scheme["http"] is ssrf._GuardedHTTPConnectionPool
        assert adapter.poolmanager.pool_classes_by_scheme["https"] is ssrf._GuardedHTTPSConnectionPool


@pytest.mark.parametrize(
    "url",
    [
        "http://100.63.255.255/x",  # just below the shared address space
        "http://100.128.0.1/x",  # just above it
        "http://[2606:4700:4700::1111]/x",
    ],
)
def test_public_neighbours_of_blocked_ranges_still_allowed(url: str):
    """The `is_global` tightening must not spill over into genuinely public space."""
    validate_download_url(url)


# ----------------------------- Proxy visibility -----------------------------


def test_guarded_session_ignores_environment_proxy(monkeypatch: Any, caplog: Any):
    """Ambient proxies must not move destination resolution outside the socket guard."""
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.internal:3128")
    session = build_guarded_session()
    assert session.trust_env is True
    assert ssrf.proxies_in_effect(session) == {}
    assert session.merge_environment_settings("https://example.com", {}, None, None, None)["proxies"] == {}

    logger = logging.getLogger("test-ssrf-proxy")
    with caplog.at_level(logging.WARNING, logger="test-ssrf-proxy"):
        ssrf.warn_if_proxied(session, logger)
    assert "proxy settings are ignored" in caplog.text


def test_guarded_session_preserves_environment_ca_bundle(monkeypatch: Any):
    ca_bundle = "/tmp/invokeai-test-ca.pem"
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", ca_bundle)
    session = build_guarded_session()
    settings = session.merge_environment_settings("https://example.com", {}, None, None, None)
    assert settings["verify"] == ca_bundle


def test_guarded_session_preserves_environment_netrc(monkeypatch: Any, tmp_path: Any):
    netrc = tmp_path / "netrc"
    netrc.write_text("machine example.com login netrc-user password netrc-pass\n")
    monkeypatch.setenv("NETRC", str(netrc))
    prepared = build_guarded_session().prepare_request(Request("GET", "https://example.com/file.bin"))
    assert prepared.headers["Authorization"] == "Basic bmV0cmMtdXNlcjpuZXRyYy1wYXNz"


def test_guarded_session_supports_explicit_proxy(monkeypatch: Any):
    monkeypatch.setenv("HTTPS_PROXY", "http://ambient.internal:3128")
    session = build_guarded_session(proxy="http://proxy.internal:3128")
    assert ssrf.proxies_in_effect(session) == {
        "http": "http://proxy.internal:3128",
        "https": "http://proxy.internal:3128",
    }
    assert session.merge_environment_settings("https://example.com", {}, None, None, None)["proxies"] == {
        "http": "http://proxy.internal:3128",
        "https": "http://proxy.internal:3128",
    }
    request = PreparedRequest()
    request.prepare(method="GET", url="https://example.com/file.bin")
    assert session.rebuild_proxies(request, {}) == {}
    assert session.rebuild_proxies(request, session.proxies) == session.proxies

    captured: dict[str, Any] = {}

    def capture_send(_session: requests.Session, _request: PreparedRequest, **kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(requests.Session, "send", capture_send)
    session.send(request)
    assert captured["proxies"] == session.proxies


def test_no_warning_without_a_proxy(monkeypatch: Any, caplog: Any):
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    session = build_guarded_session()
    assert ssrf.proxies_in_effect(session) == {}

    logger = logging.getLogger("test-ssrf-noproxy")
    with caplog.at_level(logging.WARNING, logger="test-ssrf-noproxy"):
        ssrf.warn_if_proxied(session, logger)
    assert caplog.text == ""


@pytest.mark.parametrize("var", ["ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy"])
def test_proxy_detection_covers_every_env_spelling(monkeypatch: Any, var: str):
    """`ALL_PROXY` becomes the key "all", which requests honours as a catch-all.

    Filtering `getproxies()` by scheme silently missed it, so the warning did not fire in
    the setup where it matters most.
    """
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(var, "http://proxy.internal:3128")

    session = ssrf.requests.Session()
    detected = ssrf.proxies_in_effect(session)
    assert detected, f"{var} was not detected"
    assert select_proxy("http://example.com/x", detected) == "http://proxy.internal:3128"


@pytest.mark.parametrize(
    ("url", "blocked"),
    [
        ("http://[2a01:1:2:3:0:5efe:a00:1]/x", True),  # ISATAP wrapping 10.0.0.1
        ("http://[2a01:1:2:3:200:5efe:7f00:1]/x", True),  # ISATAP wrapping 127.0.0.1
        ("http://[2a01:1:2:3:0:5efe:5db8:d822]/x", False),  # ISATAP wrapping a public v4
        ("http://[64:ff9b::a9fe:a9fe]/x", True),  # NAT64 WKP -- caught as ::/8 reserved
    ],
)
def test_ipv6_transition_wrappers(url: str, blocked: bool):
    if blocked:
        with pytest.raises(UnsafeDownloadURLException):
            validate_download_url(url)
    else:
        validate_download_url(url)

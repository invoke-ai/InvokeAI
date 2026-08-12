"""Guards against server-side request forgery (SSRF) via user-supplied download URLs.

The download queue fetches URLs supplied over the API. Without a check, a caller can
point the server at addresses only the server can reach — loopback services, other
hosts on the private network, or the cloud metadata endpoint at 169.254.169.254 — and
use the download job as a proxy for them.

There are two layers here, and the second is the one that actually holds:

`build_guarded_session()` returns a `requests.Session` that validates the peer address
of every socket it opens, before a single byte of the request is written. Checking the
connected socket is the only check that cannot be desynchronised from what the HTTP
client does: it is immune to DNS rebinding (the client resolves again after any
up-front check, and an attacker controlling the zone can answer the two lookups
differently) and to hostname spellings that we and `requests` disagree about (it
percent-decodes unreserved characters in the host, so `http://%6cocalhost/` and
`http://169.254.169%2e254/` reach loopback and the metadata service respectively).

`validate_download_url()` is the cheap up-front check. It rejects the obvious cases
before any packet is sent — which keeps the API's error messages useful and stops the
socket layer being used as a port-existence oracle — but it resolves the host itself,
so it must never be relied on alone.

One case gets only the weaker layer: when a request goes through an HTTP proxy (including
one picked up from ambient `HTTP_PROXY`/`ALL_PROXY`), the socket goes to the proxy and the
proxy resolves the destination, so there is no peer address for us to inspect. Address
policy belongs to the proxy there. This is decided per request — a host excluded by
`no_proxy` is still fully guarded — and IP-literal URLs are still rejected up front,
because those need no DNS. What is left uncovered is hostnames that resolve differently
for us than for the proxy, or that only resolve proxy-side and so hit the fail-open path.
`warn_if_proxied()` says this out loud at startup rather than letting the control look
stronger than it is.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from collections.abc import Iterator
from typing import Any
from urllib.parse import unquote, urlsplit
from urllib.request import getproxies

import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool

ALLOWED_SCHEMES = ("http", "https")

IpAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


class UnsafeDownloadURLException(ValueError):
    """Raised when a URL points at a network location the download service refuses to fetch."""


# RFC 5214 reserves this interface-identifier marker for ISATAP; the 32 bits after it are
# the tunnelled IPv4 address. `u/g` may be 0x0000 or 0x0200.
_ISATAP_MARKERS = (0x00005EFE, 0x02005EFE)


def _candidates(ip: IpAddress) -> Iterator[IpAddress]:
    """The address itself, plus every IPv4 address it can carry traffic to.

    IPv6 has several ways to wrap an IPv4 destination, and on a host with the matching
    transition mechanism enabled, connecting to the wrapper reaches the embedded IPv4
    address. `IPv6Address("::ffff:127.0.0.1").is_loopback` is False, so the inner address
    has to be pulled out and judged on its own.

    Both the wrapper and what it wraps must be acceptable, which is why this yields rather
    than substitutes: a Teredo address whose embedded client is public is still a Teredo
    address.

    Only wrappers identifiable from the address alone are enumerated: v4-mapped, 6to4,
    Teredo and ISATAP. The NAT64 well-known prefixes (`64:ff9b::/96`, `64:ff9b:1::/48`)
    need no special case — they sit in `::/8` and so are already `is_reserved`. NAT64 with
    an operator-chosen network-specific prefix, 6rd, and SIIT-EAM are out of scope by
    construction: their prefix length and bit offsets are site-local configuration, so a
    translated address is indistinguishable from any other global one. 6over4 is likewise
    excluded — its "IPv4 in the low 32 bits" layout is what an ordinary hand-assigned
    address looks like, so screening for it would reject a great deal of real traffic.
    """
    yield ip
    if isinstance(ip, ipaddress.IPv6Address):
        teredo = ip.teredo
        for inner in (ip.ipv4_mapped, ip.sixtofour, teredo[1] if teredo else None):
            if inner is not None:
                yield inner
        if (int(ip) >> 32) & 0xFFFFFFFF in _ISATAP_MARKERS:
            yield ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)


def _is_blocked(ip: IpAddress) -> bool:
    """Allow only globally-routable unicast addresses, wrappers included.

    `is_global` rather than a hand-written range list, so that the shared address space
    (`100.64.0.0/10`, RFC 6598) and the benchmarking range are covered — neither is
    `is_private` or `is_reserved`, but both reach a provider's internal infrastructure.

    `is_reserved` and `is_multicast` are still needed on top of it: for IPv6, CPython
    defines `is_global` as `not is_private`, and the IPv6 private list covers neither the
    reserved/unallocated space (`::/8`, `4000::/3`, the NAT64 prefixes) nor multicast, all
    of which report `is_global == True`.
    """
    return any(not c.is_global or c.is_reserved or c.is_multicast for c in _candidates(ip))


def _resolve(host: str, port: int | None) -> list[IpAddress]:
    infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    return [ipaddress.ip_address(info[4][0]) for info in infos]


def _host_spellings(host: str) -> list[str]:
    """Every host string the HTTP client might end up connecting to.

    `requests` runs the URL through `requote_uri()`, which decodes percent-escapes of
    unreserved characters. So `169.254.169%2e254` is checked by us as-is but connected to
    as `169.254.169.254`. Check the decoded form too; the socket-level guard is what
    finally catches anything this misses.
    """
    spellings = [host]
    decoded = unquote(host)
    if decoded != host:
        spellings.append(decoded)
    return spellings


def _parse_ipv4_literal(host: str) -> IpAddress | None:
    """Parse dotted and legacy numeric IPv4 spellings accepted by HTTP clients."""
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        pass

    parts = host.split(".")
    if not 1 <= len(parts) <= 4:
        return None

    values: list[int] = []
    for part in parts:
        if not part or part[0] in "+-" or not part.isascii():
            return None
        if part.lower().startswith("0x"):
            base, digits = 16, part[2:]
        elif len(part) > 1 and part.startswith("0"):
            base, digits = 8, part[1:]
        else:
            base, digits = 10, part
        alphabet = "0123456789abcdef"[:base]
        if not digits or any(char.lower() not in alphabet for char in digits):
            return None
        try:
            values.append(int(digits, base))
        except ValueError:
            return None

    widths = {1: (32,), 2: (8, 24), 3: (8, 8, 16), 4: (8, 8, 8, 8)}[len(values)]
    if any(value >= 1 << width for value, width in zip(values, widths, strict=True)):
        return None

    address = 0
    for value, width in zip(values, widths, strict=True):
        address = (address << width) | value
    return ipaddress.IPv4Address(address)


def check_address(ip: IpAddress, host: str) -> None:
    """Raise if `ip` is not an address we are willing to connect to."""
    if _is_blocked(ip):
        raise UnsafeDownloadURLException(
            f"Refusing to download from '{host}': it resolves to the non-public address {ip}. "
            "Set `allow_private_download_urls` in invokeai.yaml to permit downloads from loopback "
            "and private-network addresses."
        )


def validate_download_url(url: str, allow_private_urls: bool = False) -> None:
    """Reject `url` up front if it obviously points somewhere only the server can reach.

    Every address the host resolves to must be public — a hostname with both a public and a
    loopback record is rejected, because we cannot control which one the HTTP client picks.

    An unresolvable host is allowed through to the HTTP client, so that offline test
    environments and mocked sessions keep working. That is only safe because the session
    from `build_guarded_session()` re-checks the address it actually connects to.
    """
    parts = urlsplit(str(url))

    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        raise UnsafeDownloadURLException(f"Unsupported URL scheme '{parts.scheme}'. Only http and https are allowed.")

    host = parts.hostname
    if not host:
        raise UnsafeDownloadURLException(f"Download URL '{url}' has no host.")

    if allow_private_urls:
        return

    for spelling in _host_spellings(host):
        literal = _parse_ipv4_literal(spelling)
        if literal is not None:
            candidates = [literal]
        else:
            try:
                candidates = _resolve(spelling, parts.port)
            except (OSError, UnicodeError, ValueError):
                continue
        for candidate in candidates:
            check_address(candidate, host)


def _check_socket(sock: socket.socket) -> None:
    """Validate the address a socket is actually connected to."""
    peer = sock.getpeername()
    # IPv6 peers come back as (host, port, flowinfo, scope_id) and the host may carry a
    # "%iface" suffix; ip_address() will not parse that.
    address = ipaddress.ip_address(str(peer[0]).partition("%")[0])
    check_address(address, str(peer[0]))


class _GuardedHTTPConnection(HTTPConnection):
    def _new_conn(self) -> socket.socket:
        sock = super()._new_conn()
        try:
            _check_socket(sock)
        except BaseException:
            sock.close()
            raise
        return sock


class _GuardedHTTPSConnection(HTTPSConnection):
    def _new_conn(self) -> socket.socket:
        # Checked here rather than after `connect()` so that we bail out before the TLS
        # handshake, not just before the HTTP request.
        sock = super()._new_conn()
        try:
            _check_socket(sock)
        except BaseException:
            sock.close()
            raise
        return sock


class _GuardedHTTPConnectionPool(HTTPConnectionPool):
    ConnectionCls = _GuardedHTTPConnection


class _GuardedHTTPSConnectionPool(HTTPSConnectionPool):
    ConnectionCls = _GuardedHTTPSConnection


class SsrfGuardedAdapter(HTTPAdapter):
    """A `requests` adapter that refuses to talk to non-public addresses."""

    def init_poolmanager(self, *args: Any, **kwargs: Any) -> None:
        super().init_poolmanager(*args, **kwargs)
        self.poolmanager.pool_classes_by_scheme = {
            "http": _GuardedHTTPConnectionPool,
            "https": _GuardedHTTPSConnectionPool,
        }

    # `proxy_manager_for` is deliberately left alone. When an HTTP proxy is configured the
    # socket goes to the proxy, so guarding it would only reject proxies on the local
    # network; address selection is the proxy's business in that setup.


def build_guarded_session() -> requests.Session:
    """A `requests.Session` that will not open a connection to a non-public address."""
    session = requests.Session()
    adapter = SsrfGuardedAdapter()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def proxies_in_effect(session: requests.Session) -> dict[str, str]:
    """Proxies this session would use, including ones inherited from the environment."""
    proxies = dict(session.proxies)
    if session.trust_env:
        # Every key, not just http/https: `ALL_PROXY` arrives as the key "all", which
        # `requests.utils.select_proxy` honours as a catch-all. Filtering by scheme would
        # miss the form most egress-proxy setups actually use.
        for scheme, proxy_url in getproxies().items():
            proxies.setdefault(scheme, proxy_url)
    return proxies


def warn_if_proxied(session: requests.Session, logger: logging.Logger) -> None:
    """Say plainly when the socket-level guard cannot apply.

    `trust_env` is deliberately left on: an operator behind a mandatory egress proxy would
    otherwise lose the ability to download models at all, and their DNS may well only
    resolve proxy-side, so failing closed here would break them rather than protect them.
    """
    proxies = proxies_in_effect(session)
    if proxies:
        logger.warning(
            "An HTTP proxy is configured (%s). Downloads that are not excluded by no_proxy "
            "are resolved by the proxy, so they cannot be checked against loopback/private "
            "addresses at the socket. Restrict outbound access at the proxy instead.",
            ", ".join(sorted(proxies)),
        )

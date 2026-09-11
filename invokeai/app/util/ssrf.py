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

Ambient `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY` settings are ignored by the guarded session.
That keeps destination resolution inside this process, where the socket-level policy can
inspect the connected peer, while Requests' CA-bundle and netrc environment support remains
enabled. A caller that explicitly adds a proxy to a session accepts the proxy's destination
policy instead; `warn_if_proxied()` reports that weaker configuration.
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
from requests.auth import _basic_auth_str
from requests.models import PreparedRequest
from requests.utils import get_auth_from_url, resolve_proxies
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool

ALLOWED_SCHEMES = ("http", "https")
logger = logging.getLogger(__name__)

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

    An IPv4-mapped IPv6 address is an alternate spelling of its embedded IPv4 address, so
    only the embedded address is judged. Other wrappers must be acceptable themselves and
    what they wrap, which is why those addresses are yielded rather than substituted: a
    Teredo address whose embedded client is public is still a Teredo address.

    Only wrappers identifiable from the address alone are enumerated: v4-mapped, 6to4,
    Teredo and ISATAP. The NAT64 well-known prefixes (`64:ff9b::/96`, `64:ff9b:1::/48`)
    need no special case — they sit in `::/8` and so are already `is_reserved`. NAT64 with
    an operator-chosen network-specific prefix, 6rd, and SIIT-EAM are out of scope by
    construction: their prefix length and bit offsets are site-local configuration, so a
    translated address is indistinguishable from any other global one. 6over4 is likewise
    excluded — its "IPv4 in the low 32 bits" layout is what an ordinary hand-assigned
    address looks like, so screening for it would reject a great deal of real traffic.
    """
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            yield ip.ipv4_mapped
            return
        yield ip
        teredo = ip.teredo
        for inner in (ip.sixtofour, teredo[1] if teredo else None):
            if inner is not None:
                yield inner
        if (int(ip) >> 32) & 0xFFFFFFFF in _ISATAP_MARKERS:
            yield ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    else:
        yield ip


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
    host = host.removesuffix(".")
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
        logger.warning("Blocked download host %s resolving to non-public address %s", host, ip)
        raise UnsafeDownloadURLException(
            f"Refusing to download from '{host}': it resolves to a non-public address. "
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

    try:
        port = parts.port
    except ValueError as e:
        raise UnsafeDownloadURLException(f"Download URL '{url}' has an invalid port.") from e

    if allow_private_urls:
        return

    for spelling in _host_spellings(host):
        literal = _parse_ipv4_literal(spelling)
        if literal is not None:
            candidates = [literal]
        else:
            try:
                candidates = _resolve(spelling, port)
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

    def proxy_manager_for(self, proxy: str, **proxy_kwargs: Any) -> Any:
        """Install the socket guard on proxy pools as well as direct pools.

        Requests creates proxy managers separately from the adapter's direct pool
        manager. Without replacing their pool classes, an explicit download proxy
        would use urllib3's ordinary connection classes and bypass the peer-address
        check entirely.
        """
        manager = super().proxy_manager_for(proxy, **proxy_kwargs)
        manager.pool_classes_by_scheme = {
            "http": _GuardedHTTPConnectionPool,
            "https": _GuardedHTTPSConnectionPool,
        }
        return manager


class _SsrfGuardedSession(requests.Session):
    """Session that keeps Requests environment support but drops ambient proxies."""

    _ignore_environment_proxies = True

    def merge_environment_settings(
        self,
        url: str,
        proxies: dict[str, str] | None,
        stream: bool | None,
        verify: bool | str | None,
        cert: str | tuple[str, str] | None,
    ) -> dict[str, Any]:
        request_proxies = dict(proxies or {})
        settings = super().merge_environment_settings(url, proxies, stream, verify, cert)
        # `super()` supplies CA-bundle and netrc environment settings, which we want to
        # retain. Replace only the merged proxy map so ambient proxies cannot bypass the
        # peer-address check. Explicit session/request proxies remain supported but weaker.
        effective_proxies = dict(self.proxies)
        effective_proxies.update(request_proxies)
        settings["proxies"] = effective_proxies
        return settings

    def rebuild_proxies(self, prepared_request: PreparedRequest, proxies: dict[str, str] | None) -> dict[str, str]:
        """Rebuild explicit proxies across redirects without consulting the environment."""
        new_proxies = resolve_proxies(prepared_request, proxies, trust_env=False)
        headers = prepared_request.headers
        if "Proxy-Authorization" in headers:
            del headers["Proxy-Authorization"]

        scheme = urlsplit(prepared_request.url).scheme
        try:
            username, password = get_auth_from_url(new_proxies[scheme])
        except KeyError:
            username, password = None, None

        # urllib3 handles proxy authorization for HTTPS tunnels. Avoid putting these
        # credentials in the tunneled request headers.
        if not scheme.startswith("https") and username and password:
            headers["Proxy-Authorization"] = _basic_auth_str(username, password)
        return new_proxies

    def send(self, request: PreparedRequest, **kwargs: Any) -> requests.Response:
        """Keep direct `send()` calls from reintroducing ambient proxies."""
        if kwargs.get("proxies") is None:
            kwargs["proxies"] = dict(self.proxies)
        return super().send(request, **kwargs)


def build_guarded_session(proxy: str | None = None) -> requests.Session:
    """A `requests.Session` that will not open a connection to a non-public address."""
    session = _SsrfGuardedSession()
    if proxy:
        session.proxies.update({"http": proxy, "https": proxy})
    adapter = SsrfGuardedAdapter()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def proxies_in_effect(session: requests.Session) -> dict[str, str]:
    """Proxies this session would use, including ones inherited from the environment."""
    proxies = dict(session.proxies)
    if session.trust_env and not getattr(session, "_ignore_environment_proxies", False):
        # Every key, not just http/https: `ALL_PROXY` arrives as the key "all", which
        # `requests.utils.select_proxy` honours as a catch-all. Filtering by scheme would
        # miss the form most egress-proxy setups actually use.
        for scheme, proxy_url in getproxies().items():
            proxies.setdefault(scheme, proxy_url)
    return proxies


def warn_if_proxied(session: requests.Session, logger: logging.Logger) -> None:
    """Report explicit proxies or ignored ambient proxy settings."""
    proxies = proxies_in_effect(session)
    if proxies:
        logger.warning(
            "An explicit HTTP proxy is configured (%s). Downloads through it cannot be checked "
            "against loopback/private addresses at the socket; restrict outbound access at the proxy.",
            ", ".join(sorted(proxies)),
        )
    elif getattr(session, "_ignore_environment_proxies", False) and getproxies():
        logger.warning("Ambient HTTP proxy settings are ignored for guarded downloads.")

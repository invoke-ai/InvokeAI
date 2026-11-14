import os
import sys
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Mount, Route
from starlette.testclient import TestClient

from invokeai.app.api.desktop_theme import DESKTOP_THEME_MARKER
from invokeai.app.api.frontend import mount_frontend
from invokeai.frontend.cli.arg_parser import InvokeAIArgs


@pytest.fixture
def frontend_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    for name, marker in [("webv2", "default frontend"), ("webv1", "legacy frontend")]:
        bundle = tmp_path / name / "dist"
        (bundle / "assets").mkdir(parents=True)
        (bundle / "index.html").write_text(marker)
        (bundle / "assets" / "main.js").write_text(f"console.log('{marker}')")
    root = tmp_path / "webv1"
    (root / "static" / "docs").mkdir(parents=True)
    (root / "static" / "docs" / "invoke-favicon-docs.svg").write_text("<svg>docs favicon</svg>")
    return tmp_path / "webv2"


@pytest.mark.parametrize("prefix", ["", "/invoke"])
@pytest.mark.parametrize(
    ("flags", "marker"),
    [([], "default frontend"), (["--webv2"], "default frontend"), (["--web-legacy"], "legacy frontend")],
)
def test_launch_serves_selected_bundle_and_preserves_api_and_docs(
    frontend_root: Path, monkeypatch: pytest.MonkeyPatch, flags: list[str], marker: str, prefix: str
) -> None:
    monkeypatch.setattr(sys, "argv", ["invokeai-web", *flags])
    monkeypatch.setattr(InvokeAIArgs, "args", None)
    monkeypatch.setattr(InvokeAIArgs, "did_parse", False)
    args = InvokeAIArgs.parse_args()
    app = Starlette(routes=[Route("/api/ping", lambda request: PlainTextResponse("backend"))])
    mount_frontend(app, frontend_root, legacy=getattr(args, "web_legacy", False))
    host = Starlette(routes=[Mount(prefix, app=app)]) if prefix else app
    with TestClient(host) as client:
        for path in ["/", "/projects/example"]:
            response = client.get(f"{prefix}{path}")
            assert response.status_code == 200
            assert response.text == marker
            assert "no-store" in response.headers["cache-control"]
        assert client.get(f"{prefix}/assets/main.js").text == f"console.log('{marker}')"
        favicon = client.get(f"{prefix}/static/docs/invoke-favicon-docs.svg")
        assert favicon.status_code == 200
        assert favicon.text == "<svg>docs favicon</svg>"
        assert client.get(f"{prefix}/static/missing.svg").status_code == 404
        assert client.get(f"{prefix}/api/ping").text == "backend"


def test_conflicting_frontend_flags_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["invokeai-web", "--webv2", "--web-legacy"])
    with pytest.raises(SystemExit, match="2"):
        InvokeAIArgs.parse_args()


def test_missing_legacy_bundle_does_not_silently_serve_default(frontend_root: Path) -> None:
    (frontend_root.parent / "webv1" / "dist").rename(frontend_root.parent / "old-dist")
    with pytest.raises(RuntimeError, match="webv1"):
        mount_frontend(Starlette(), frontend_root, legacy=True)


def test_missing_default_bundle_preserves_docs_without_falling_back_to_legacy(frontend_root: Path) -> None:
    (frontend_root / "dist").rename(frontend_root.parent / "old-dist")
    app = Starlette()
    with pytest.raises(RuntimeError, match="webv2"):
        mount_frontend(app, frontend_root)
    with TestClient(app) as client:
        assert client.get("/").status_code == 404
        assert client.get("/static/docs/invoke-favicon-docs.svg").text == "<svg>docs favicon</svg>"


@pytest.fixture
def desktop_palette(frontend_root: Path, tmp_path: Path) -> Path:
    index = frontend_root / "dist" / "index.html"
    index.write_text(f"<html><head>{DESKTOP_THEME_MARKER}</head><body>workbench</body></html>")
    palette = tmp_path / "state" / "omarchy" / "current" / "theme" / "colors.toml"
    palette.parent.mkdir(parents=True)
    palette.write_text('mode="dark"\nbackground="#2c2525"\nforeground="#e6d9db"\naccent="#f38d70"\n')
    return palette


def test_frontend_supplies_fresh_desktop_palette_without_caching_its_index(
    frontend_root: Path, desktop_palette: Path
) -> None:
    app = Starlette()
    mount_frontend(app, frontend_root)
    with TestClient(app) as client:
        first = client.get("/")
        assert first.status_code == 200
        assert first.text.count("--invokeai-desktop-") == 8
        assert "--invokeai-desktop-background:#2c2525" in first.text
        assert "--invokeai-desktop-accent:#f38d70" in first.text
        assert "--invokeai-desktop-accent-contrast:#2c2525" in first.text
        assert "--invokeai-desktop-color-scheme:dark" in first.text
        assert '<meta name="darkreader-lock">' in first.text
        assert "--invokeai-desktop-surface:#373030" in first.text
        assert "--invokeai-desktop-selection:#3f3737" in first.text
        assert DESKTOP_THEME_MARKER not in first.text
        assert str(desktop_palette) not in first.text
        assert "no-store" in first.headers["cache-control"]
        assert "last-modified" not in first.headers
        assert int(first.headers["content-length"]) == len(first.content)
        assert client.get("/", headers={"If-None-Match": first.headers["etag"]}).status_code == 304
        head = client.head("/projects/example")
        assert head.content == b""
        assert head.headers["content-length"] == first.headers["content-length"]
        assert head.headers["etag"] == first.headers["etag"]
        # HTML's mtime and even a preserved palette mtime cannot stand in for the rendered document's validator.
        timestamp = desktop_palette.stat().st_mtime_ns
        desktop_palette.write_text('mode="light"\nbackground="#ffffff"\nforeground="#222222"\naccent="#006b5f"\n')
        os.utime(desktop_palette, ns=(timestamp, timestamp))
        changed = client.get("/projects/example", headers={"If-None-Match": first.headers["etag"]})
        assert changed.status_code == 200
        assert "--invokeai-desktop-background:#ffffff" in changed.text
        assert "--invokeai-desktop-accent:#006b5f" in changed.text
        assert "--invokeai-desktop-color-scheme:light" in changed.text
        assert changed.headers["etag"] != first.headers["etag"]
        assert (
            client.get(
                "/", headers={"If-Modified-Since": first.headers.get("date", "Wed, 01 Jan 2100 00:00:00 GMT")}
            ).status_code
            == 200
        )
        asset = client.get("/assets/main.js")
        assert client.get("/assets/main.js", headers={"If-None-Match": asset.headers["etag"]}).status_code == 304
        assert "last-modified" in asset.headers
        assert "invokeai-desktop" not in asset.text


@pytest.mark.parametrize("prefix", ["", "/invoke"])
def test_desktop_palette_keeps_spa_subpaths_and_uses_only_whitelisted_values(
    frontend_root: Path, desktop_palette: Path, prefix: str
) -> None:
    with desktop_palette.open("a") as source:
        source.write('unrelated="</style><script>untrusted()</script>"\n')
    app = Starlette()
    mount_frontend(app, frontend_root)
    host = Starlette(routes=[Mount(prefix, app=app)]) if prefix else app
    with TestClient(host) as client:
        response = client.get(f"{prefix}/projects/example")
        assert response.status_code == 200
        assert "--invokeai-desktop-accent:#f38d70" in response.text
        assert "untrusted" not in response.text


@pytest.mark.parametrize(
    "content",
    [
        'mode="dark"\nbackground="#ffffff"\nforeground="#000000"\naccent="red;display:none"',
        'mode="dark;display:none"\nbackground="#ffffff"\nforeground="#000000"\naccent="#123456"',
        'mode="dark"\nbackground="#ffffff"\nforeground="#eeeeee"\naccent="#123456"',
        'mode="dark"\nbackground="#ffffff"\nforeground="#000000"\naccent=123456',
        'mode="dark"\nbackground="#ffffff"\naccent="#123456"',
        "not valid toml!",
        "#" * 16_385,
    ],
)
def test_invalid_desktop_palette_leaves_original_frontend_unchanged(
    frontend_root: Path, desktop_palette: Path, content: str
) -> None:
    desktop_palette.write_text(content)
    original = (frontend_root / "dist" / "index.html").read_text()
    app = Starlette()
    mount_frontend(app, frontend_root)
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.text == original
        assert "darkreader-lock" not in response.text
        assert "last-modified" in response.headers
        assert client.get("/", headers={"If-None-Match": response.headers["etag"]}).status_code == 304


def test_missing_desktop_palette_leaves_original_frontend_unchanged(frontend_root: Path, desktop_palette: Path) -> None:
    desktop_palette.unlink()
    app = Starlette()
    mount_frontend(app, frontend_root)
    with TestClient(app) as client:
        assert client.get("/").text == (frontend_root / "dist" / "index.html").read_text()


def test_desktop_accent_uses_readable_black_when_neither_native_text_color_contrasts(
    frontend_root: Path, desktop_palette: Path
) -> None:
    desktop_palette.write_text('mode="dark"\nbackground="#222222"\nforeground="#eeeeee"\naccent="#777777"\n')
    app = Starlette()
    mount_frontend(app, frontend_root)
    with TestClient(app) as client:
        assert "--invokeai-desktop-accent-contrast:#000000" in client.get("/").text


@pytest.mark.skipif(sys.platform == "win32", reason="Theme-directory symlinks require Windows privileges")
def test_frontend_follows_active_desktop_theme_symlink_on_refresh(frontend_root: Path, desktop_palette: Path) -> None:
    active = desktop_palette.parent
    previous = active.with_name("previous")
    active.rename(previous)
    active.symlink_to(previous, target_is_directory=True)
    app = Starlette()
    mount_frontend(app, frontend_root)
    with TestClient(app) as client:
        assert "--invokeai-desktop-accent:#f38d70" in client.get("/").text
        next_theme = active.with_name("next")
        next_theme.mkdir()
        (next_theme / "colors.toml").write_text(
            'mode="light"\nbackground="#ffffff"\nforeground="#000000"\naccent="#006b5f"\n'
        )
        active.unlink()
        active.symlink_to(next_theme, target_is_directory=True)
        assert "--invokeai-desktop-accent:#006b5f" in client.get("/").text


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="Named pipes are Unix-only")
def test_nonregular_desktop_source_does_not_block_frontend(frontend_root: Path, desktop_palette: Path) -> None:
    desktop_palette.unlink()
    os.mkfifo(desktop_palette)
    app = Starlette()
    mount_frontend(app, frontend_root)
    with TestClient(app) as client:
        assert client.get("/").text == (frontend_root / "dist" / "index.html").read_text()


@pytest.mark.parametrize(
    ("mode", "background", "foreground", "accent", "surface", "inset", "selection"),
    [
        ("dark", "#2c2525", "#e6d9db", "#f38d70", "#3d2f2a", "#211b1b", "#403e41"),
        ("light", "#faf4ed", "#575279", "#56949f", "#f2e9e1", "#ede7e1", "#dfdad9"),
    ],
)
def test_frontend_preserves_native_surface_roles_and_explicit_light_or_dark_mode(
    frontend_root: Path,
    desktop_palette: Path,
    mode: str,
    background: str,
    foreground: str,
    accent: str,
    surface: str,
    inset: str,
    selection: str,
) -> None:
    desktop_palette.write_text(
        f'mode="{mode}"\nbackground="{background}"\nforeground="{foreground}"\naccent="{accent}"\n'
        f'lighter_background="{surface}"\ndark_background="{inset}"\nselection="{selection}"\n'
    )
    app = Starlette()
    mount_frontend(app, frontend_root)
    with TestClient(app) as client:
        html = client.get("/").text
        assert f"--invokeai-desktop-color-scheme:{mode}" in html
        assert f"--invokeai-desktop-surface:{surface}" in html
        assert f"--invokeai-desktop-inset:{inset}" in html
        assert f"--invokeai-desktop-selection:{selection}" in html
        assert '<meta name="darkreader-lock">' in html


def test_unreadable_native_surface_roles_use_safe_restrained_fallbacks(
    frontend_root: Path, desktop_palette: Path
) -> None:
    with desktop_palette.open("a") as source:
        source.write('lighter_background="#e6d9db"\ndark_background="#e6d9db"\nselection="#e6d9db"\n')
    app = Starlette()
    mount_frontend(app, frontend_root)
    with TestClient(app) as client:
        html = client.get("/").text
        assert "--invokeai-desktop-surface:#373030" in html
        assert "--invokeai-desktop-inset:#2c2525" in html
        assert "--invokeai-desktop-selection:#3f3737" in html


def test_invalid_native_surface_color_cannot_inject_css_or_disable_browser_fallback(
    frontend_root: Path, desktop_palette: Path
) -> None:
    with desktop_palette.open("a") as source:
        source.write('lighter_background="</style><script>untrusted()</script>"\n')
    app = Starlette()
    mount_frontend(app, frontend_root)
    with TestClient(app) as client:
        html = client.get("/").text
        assert DESKTOP_THEME_MARKER in html
        assert "untrusted" not in html
        assert "darkreader-lock" not in html

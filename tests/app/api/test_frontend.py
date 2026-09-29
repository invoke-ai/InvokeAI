import sys
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Mount, Route
from starlette.testclient import TestClient

from invokeai.app.api.frontend import mount_frontend
from invokeai.frontend.cli.arg_parser import InvokeAIArgs


@pytest.fixture
def frontend_root(tmp_path: Path) -> Path:
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

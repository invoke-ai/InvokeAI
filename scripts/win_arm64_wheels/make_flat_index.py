"""Write the flat (find-links style) index page for the Windows ARM64 wheel release.

The page is uploaded as a release asset next to the wheels. ``pyproject.toml`` points the
``invoke-win-arm64-wheels`` index at it, and the manual-install docs pass the same URL to
``uv pip install --find-links``. Every link is absolute because the asset is served from a redirect
target, so relative links would resolve against the wrong host, and carries ``#sha256=`` so installers
verify what they download.

Usage: make_flat_index.py --repo owner/name --tag <release tag> wheel [wheel ...] > index.html
"""

import argparse
import hashlib
import html
import sys
from pathlib import Path


def asset_url(repo: str, tag: str, filename: str) -> str:
    return f"https://github.com/{repo}/releases/download/{tag}/{filename}"


def render(repo: str, tag: str, wheels: list[Path]) -> str:
    lines = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>{html.escape(tag)}</title>",
        "</head><body>",
    ]
    for wheel in sorted(wheels, key=lambda p: p.name):
        if wheel.suffix != ".whl":
            raise ValueError(f"not a wheel: {wheel}")
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        href = f"{asset_url(repo, tag, wheel.name)}#sha256={digest}"
        lines.append(f'<a href="{html.escape(href, quote=True)}">{html.escape(wheel.name)}</a><br>')
    lines.append("</body></html>")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="GitHub repository, owner/name")
    parser.add_argument("--tag", required=True, help="release tag the wheels are attached to")
    parser.add_argument("wheels", nargs="+", type=Path)
    args = parser.parse_args(argv[1:])
    sys.stdout.write(render(args.repo, args.tag, args.wheels))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

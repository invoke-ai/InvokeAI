"""Decide which files the Windows ARM64 wheel release may receive, without ever replacing an asset.

Every Invoke release locks the assets of the tag its ``pyproject.toml`` names by SHA-256, and the wheel builds
are not byte-reproducible, so replacing an asset breaks every install of every release that locked it. A file is
therefore uploaded only when the release has no asset of that name. An asset that already exists with the same
SHA-256 is skipped: that is how "Re-run failed jobs" finishes an interrupted upload, since a rerun of the publish
job reuses the build artifacts of the original run. An existing asset with different contents, or one whose
digest GitHub does not report, fails the publish. A new dispatch that reuses a published tag therefore fails,
because its rebuilt wheels differ from the published ones.

Usage: plan_release_upload.py RELEASE_JSON file [file ...]

RELEASE_JSON is the output of ``gh api repos/<repo>/releases/tags/<tag>`` (``{"assets": []}`` for a release
that does not exist yet). The names of the files to upload are printed one per line, as the given paths.
"""

import hashlib
import json
import sys
from pathlib import Path


class UploadConflict(Exception):
    pass


def plan_upload(local: dict[str, str], remote: dict[str, str | None]) -> list[str]:
    """The local asset names to upload.

    ``local`` maps an asset name to its SHA-256 hex digest; ``remote`` maps each existing release asset to the
    digest GitHub reports (``sha256:<hex>``), or None when it reports none. Raises UploadConflict naming every
    asset that is already published with other (or unverifiable) contents.
    """
    conflicts = []
    upload = []
    for name, sha256 in sorted(local.items()):
        if name not in remote:
            upload.append(name)
            continue
        published = remote[name]
        if published is None:
            conflicts.append(f"{name}: published without a digest, so it cannot be compared")
        elif published.lower() != f"sha256:{sha256}":
            conflicts.append(f"{name}: published as {published}, this build is sha256:{sha256}")
    if conflicts:
        raise UploadConflict(
            "The release already holds different contents for:\n  "
            + "\n  ".join(conflicts)
            + "\nPublished assets are locked by hash and are never replaced. Dispatch with a new release tag."
        )
    return upload


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        sys.stderr.write(__doc__ or "")
        return 2
    release = json.loads(Path(argv[1]).read_text())
    remote = {asset["name"]: asset.get("digest") for asset in release.get("assets", [])}
    paths = {Path(arg).name: Path(arg) for arg in argv[2:]}
    try:
        names = plan_upload({name: sha256_file(path) for name, path in paths.items()}, remote)
    except UploadConflict as e:
        sys.stderr.write(f"{e}\n")
        return 1
    for name in sorted(set(paths) - set(names)):
        sys.stderr.write(f"already published with identical contents, skipped: {name}\n")
    for name in names:
        print(paths[name])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

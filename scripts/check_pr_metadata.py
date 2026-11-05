"""Check a pull request title and body against the repository's contribution rules.

Reads PR_TITLE and PR_BODY from the environment, prints every failure and exits 1 if there is any.

- The title is a Conventional Commit subject: `type(scope)!: subject`. Squash merges use it as the commit subject.
- The title does not name the frontend package.
- The body keeps the headings of .github/pull_request_template.md in template order. Summary, QA Instructions,
  Review and Checklist are required; the other sections are optional.
"""

import os
import re
import sys

TYPES = ("feat", "fix", "docs", "chore", "ci", "test", "refactor", "perf", "build", "style", "revert")

# The template's `## ` headings in order. The second element says whether the section is required.
SECTIONS = (
    ("Summary", True),
    ("Related Issues / Discussions", False),
    ("QA Instructions", True),
    ("Review", True),
    ("Compatibility / Rollout", False),
    ("Checklist", True),
)

# GitHub's Revert button titles a PR `Revert "<original title>"`.
REVERT_RE = re.compile(r'^Revert ".+"$')
TITLE_RE = re.compile(r"^(?P<type>[a-z]+)(\([^()\s][^()]*\))?(?P<breaking>!)?:(?: (?P<subject>.*))?$")
FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
COMMENT_START_RE = re.compile(r"^ {0,3}<!--")
FORBIDDEN_TITLE_WORD = "webv2"


def check_title(title: str) -> list[str]:
    failures = []
    match = TITLE_RE.match(title.strip())
    if REVERT_RE.match(title.strip()):
        pass
    elif match is None or match["type"] not in TYPES:
        failures.append(
            f"Title {title!r} is not a Conventional Commit title. Use `type(scope): subject` or `type: subject` "
            f"with type one of {', '.join(TYPES)}, e.g. `fix(queue): keep paused jobs on restart`."
        )
    elif not (match["subject"] or "").strip():
        failures.append("Title has an empty subject. Put a short description after the colon.")
    if FORBIDDEN_TITLE_WORD in title.lower():
        failures.append(
            f"Title mentions `{FORBIDDEN_TITLE_WORD}`. Name the feature area instead (e.g. `fix(canvas): ...`), "
            "not the frontend package."
        )
    return failures


def body_headings(body: str) -> list[str]:
    """Return the text of every `## ` heading that renders as a heading, in order.

    Line by line, as CommonMark: fenced code blocks and HTML comment blocks hide their lines. A comment block
    starts at a line beginning with `<!--` (up to three spaces of indent) and ends with the first line containing
    `-->`, which belongs to the block; an unclosed one runs to the end. A fence is literal inside a comment
    and `<!--` is literal inside a fence.
    """
    headings = []
    fence = None
    in_comment = False
    for line in body.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if in_comment:
            in_comment = "-->" not in line
            continue
        if fence is None and COMMENT_START_RE.match(line):
            in_comment = "-->" not in line
            continue
        fence_match = FENCE_RE.match(line)
        if fence_match:
            marker = fence_match[1]
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence) and line.strip() == marker:
                fence = None
            continue
        if fence is None and line.startswith("## "):
            headings.append(line[3:].strip())
    return headings


def check_body(body: str) -> list[str]:
    failures = []
    headings = body_headings(body)
    names = [name for name, _ in SECTIONS]
    for name, required in SECTIONS:
        if required and name not in headings:
            failures.append(f"Body is missing the `## {name}` section. Copy it from .github/pull_request_template.md.")
    present = [h for h in headings if h in names]
    for name in names:
        if present.count(name) > 1:
            failures.append(f"Body has more than one `## {name}` section. Keep exactly one.")
    ranks = [names.index(h) for h in dict.fromkeys(present)]
    if ranks != sorted(ranks):
        expected = " > ".join(n for n in names if n in present)
        failures.append(
            f"Body sections are out of order. Order them as in .github/pull_request_template.md: {expected}."
        )
    return failures


def main() -> int:
    failures = check_title(os.environ.get("PR_TITLE", "")) + check_body(os.environ.get("PR_BODY", ""))
    for failure in failures:
        print(f"FAIL: {failure}")
    if not failures:
        print("PR title and description follow the repository conventions.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

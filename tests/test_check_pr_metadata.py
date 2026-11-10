"""scripts/check_pr_metadata.py must enforce the PR title and template rules, driven through its environment entry."""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_pr_metadata.py"
TEMPLATE = REPO_ROOT / ".github" / "pull_request_template.md"

spec = importlib.util.spec_from_file_location("check_pr_metadata", SCRIPT)
assert spec is not None and spec.loader is not None
check_pr_metadata = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = check_pr_metadata
spec.loader.exec_module(check_pr_metadata)

GOOD_TITLE = "fix(queue): keep paused jobs on restart"
REQUIRED = ["Summary", "QA Instructions", "Review", "Checklist"]
ALL_SECTIONS = [
    "Summary",
    "Related Issues / Discussions",
    "QA Instructions",
    "Review",
    "Compatibility / Rollout",
    "Checklist",
]


def body(*headings: str, sep: str = "\n") -> str:
    return sep.join(f"## {h}{sep}{sep}text{sep}" for h in headings)


def run(monkeypatch, capsys, title: str | None, body_text: str | None) -> tuple[int, str]:
    for key, value in (("PR_TITLE", title), ("PR_BODY", body_text)):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    code = check_pr_metadata.main()
    return code, capsys.readouterr().out


@pytest.mark.parametrize(
    "title",
    [GOOD_TITLE, "feat: add thing", "feat(ui)!: drop old route", "revert(api): undo change", "ci: pin action"],
)
def test_valid_titles_pass(monkeypatch, capsys, title):
    assert run(monkeypatch, capsys, title, body(*REQUIRED))[0] == 0


@pytest.mark.parametrize(
    "title",
    [
        "",
        "Fix the queue",
        "fix(queue) keep jobs",
        "fix:no space",
        "feature: add thing",
        "Fix: add thing",
        "fix(): x",
        'Revert ""',
        "Revert fix(x): y",
    ],
)
def test_malformed_titles_fail_as_not_conventional(monkeypatch, capsys, title):
    code, out = run(monkeypatch, capsys, title, body(*REQUIRED))
    assert code == 1
    assert "not a Conventional Commit title" in out
    assert "empty subject" not in out


@pytest.mark.parametrize("title", ["fix: ", "fix:  "])
def test_empty_subject_fails_as_empty_subject(monkeypatch, capsys, title):
    code, out = run(monkeypatch, capsys, title, body(*REQUIRED))
    assert code == 1
    assert "empty subject" in out
    assert "not a Conventional Commit title" not in out


def test_github_revert_button_title_passes(monkeypatch, capsys):
    assert run(monkeypatch, capsys, 'Revert "fix(queue): keep paused jobs"', body(*REQUIRED))[0] == 0


def test_revert_title_naming_frontend_package_still_fails(monkeypatch, capsys):
    code, out = run(monkeypatch, capsys, 'Revert "fix(webv2): thing"', body(*REQUIRED))
    assert code == 1
    assert "webv2" in out


@pytest.mark.parametrize("title", ["fix(webv2): thing", "feat: WebV2 thing"])
def test_frontend_package_name_in_title_fails(monkeypatch, capsys, title):
    code, out = run(monkeypatch, capsys, title, body(*REQUIRED))
    assert code == 1
    assert "webv2" in out


def test_webv2_in_body_is_allowed(monkeypatch, capsys):
    text = body(*REQUIRED) + "\nTouches webv2 canvas.\n"
    assert run(monkeypatch, capsys, GOOD_TITLE, text)[0] == 0


@pytest.mark.parametrize("missing", REQUIRED)
def test_missing_required_section_fails(monkeypatch, capsys, missing):
    code, out = run(monkeypatch, capsys, GOOD_TITLE, body(*[h for h in REQUIRED if h != missing]))
    assert code == 1
    assert f"## {missing}" in out


def test_conditional_sections_are_accepted_in_template_order(monkeypatch, capsys):
    assert run(monkeypatch, capsys, GOOD_TITLE, body(*ALL_SECTIONS))[0] == 0


def test_conditional_section_out_of_order_fails(monkeypatch, capsys):
    text = body("Summary", "QA Instructions", "Related Issues / Discussions", "Review", "Checklist")
    code, out = run(monkeypatch, capsys, GOOD_TITLE, text)
    assert code == 1
    assert "out of order" in out


def test_required_sections_out_of_order_fail(monkeypatch, capsys):
    code, out = run(monkeypatch, capsys, GOOD_TITLE, body("Summary", "Review", "QA Instructions", "Checklist"))
    assert code == 1
    assert "out of order" in out


def test_duplicate_section_fails(monkeypatch, capsys):
    code, out = run(monkeypatch, capsys, GOOD_TITLE, body("Summary", *REQUIRED))
    assert code == 1
    assert "more than one" in out


def test_all_failures_reported_together(monkeypatch, capsys):
    code, out = run(monkeypatch, capsys, "Fix webv2 thing", body("Summary", "Checklist"))
    assert code == 1
    assert "Conventional Commit" in out
    assert "webv2" in out
    assert "## QA Instructions" in out
    assert "## Review" in out


def test_crlf_body_passes(monkeypatch, capsys):
    assert run(monkeypatch, capsys, GOOD_TITLE, body(*REQUIRED, sep="\r\n"))[0] == 0


@pytest.mark.parametrize("fence", ["```", "~~~", "````"])
def test_heading_inside_code_fence_is_ignored(monkeypatch, capsys, fence):
    text = body("Summary", "QA Instructions", "Checklist") + f"\n{fence}\n## Review\n{fence}\n"
    code, out = run(monkeypatch, capsys, GOOD_TITLE, text)
    assert code == 1
    assert "## Review" in out


def missing(name: str) -> str:
    return f"missing the `## {name}` section"


def test_all_required_headings_commented_out_fail_multiline(monkeypatch, capsys):
    text = "<!--\n" + body(*REQUIRED) + "-->\n"
    code, out = run(monkeypatch, capsys, GOOD_TITLE, text)
    assert code == 1
    assert all(missing(name) in out for name in REQUIRED)


def test_all_required_headings_commented_out_fail_single_line(monkeypatch, capsys):
    text = "\n".join(f"<!-- ## {name} -->" for name in REQUIRED)
    code, out = run(monkeypatch, capsys, GOOD_TITLE, text)
    assert code == 1
    assert all(missing(name) in out for name in REQUIRED)


def test_fence_marker_inside_comment_does_not_hide_later_headings(monkeypatch, capsys):
    text = "<!--\n```\n-->\n" + body(*REQUIRED)
    assert run(monkeypatch, capsys, GOOD_TITLE, text)[0] == 0


def test_comment_opener_inside_fence_does_not_hide_later_headings(monkeypatch, capsys):
    text = "```\n<!--\n```\n" + body(*REQUIRED)
    assert run(monkeypatch, capsys, GOOD_TITLE, text)[0] == 0


def test_unclosed_comment_hides_later_headings(monkeypatch, capsys):
    text = body("Summary", "QA Instructions") + "<!-- never closed\n" + body("Review", "Checklist")
    code, out = run(monkeypatch, capsys, GOOD_TITLE, text)
    assert code == 1
    assert missing("Review") in out
    assert missing("Checklist") in out
    assert missing("Summary") not in out


def test_text_after_comment_end_on_closing_line_is_not_a_heading(monkeypatch, capsys):
    text = body("Summary", "QA Instructions", "Checklist") + "<!--\nnote\n--> ## Review\n"
    code, out = run(monkeypatch, capsys, GOOD_TITLE, text)
    assert code == 1
    assert missing("Review") in out


def test_mid_line_comment_opener_does_not_hide_next_heading(monkeypatch, capsys):
    text = "## Summary\ntext <!-- start\n" + body("QA Instructions", "Review", "Checklist")
    assert run(monkeypatch, capsys, GOOD_TITLE, text)[0] == 0


def test_shorter_fence_line_does_not_close_longer_fence(monkeypatch, capsys):
    text = body("Summary", "QA Instructions", "Checklist") + "\n````\n```\n## Review\n````\n"
    code, out = run(monkeypatch, capsys, GOOD_TITLE, text)
    assert code == 1
    assert "## Review" in out


def test_tilde_line_does_not_close_backtick_fence(monkeypatch, capsys):
    text = body("Summary", "QA Instructions", "Checklist") + "\n```\n~~~\n## Review\n```\n"
    code, out = run(monkeypatch, capsys, GOOD_TITLE, text)
    assert code == 1
    assert "## Review" in out


def test_unclosed_fence_swallows_later_headings(monkeypatch, capsys):
    text = body("Summary", "QA Instructions") + "\n```\n" + body("Review", "Checklist")
    code, out = run(monkeypatch, capsys, GOOD_TITLE, text)
    assert code == 1
    assert "## Review" in out
    assert "## Checklist" in out


def test_third_level_heading_is_not_the_section(monkeypatch, capsys):
    text = body("Summary", "QA Instructions", "Checklist") + "\n### Review\n"
    code, out = run(monkeypatch, capsys, GOOD_TITLE, text)
    assert code == 1
    assert "## Review" in out


def test_heading_after_closed_fence_counts(monkeypatch, capsys):
    text = "## Summary\n```\n## Review\n```\n" + body("QA Instructions", "Review", "Checklist")
    assert run(monkeypatch, capsys, GOOD_TITLE, text)[0] == 0


@pytest.mark.parametrize("empty", ["", None])
def test_empty_or_missing_body_fails(monkeypatch, capsys, empty):
    code, out = run(monkeypatch, capsys, GOOD_TITLE, empty)
    assert code == 1
    for name in REQUIRED:
        assert f"## {name}" in out


def test_pull_request_template_passes(monkeypatch, capsys):
    template = TEMPLATE.read_text(encoding="utf-8")
    assert run(monkeypatch, capsys, GOOD_TITLE, template)[0] == 0

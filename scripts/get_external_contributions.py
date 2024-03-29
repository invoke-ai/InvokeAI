import re
from argparse import ArgumentParser, RawTextHelpFormatter
from typing import Any

import requests
from attr import dataclass
from tqdm import tqdm


def get_author(commit: dict[str, Any]) -> str:
    """Gets the author of a commit.

    If the author is not present, the committer is used instead and an asterisk appended to the name."""
    return commit["author"]["login"] if commit["author"] else f"{commit['commit']['author']['name']}*"


@dataclass
class CommitInfo:
    sha: str
    url: str
    author: str
    is_username: bool
    message: str
    data: dict[str, Any]

    def __str__(self) -> str:
        return f"{self.sha}: {self.author}{'*' if not self.is_username else ''} - {self.message} ({self.url})"

    @classmethod
    def from_data(cls, commit: dict[str, Any]) -> "CommitInfo":
        return CommitInfo(
            sha=commit["sha"],
            url=commit["url"],
            author=commit["author"]["login"] if commit["author"] else commit["commit"]["author"]["name"],
            is_username=bool(commit["author"]),
            message=commit["commit"]["message"].split("\n")[0],
            data=commit,
        )


def fetch_commits_between_tags(
    org_name: str, repo_name: str, from_ref: str, to_ref: str, token: str
) -> list[CommitInfo]:
    """Fetches all commits between two tags in a GitHub repository."""

    commit_info: list[CommitInfo] = []
    headers = {"Authorization": f"token {token}"} if token else None

    # Get the total number of pages w/ an intial request - a bit hacky but it works...
    response = requests.get(
        f"https://api.github.com/repos/{org_name}/{repo_name}/compare/{from_ref}...{to_ref}?page=1&per_page=100",
        headers=headers,
    )
    last_page_match = re.search(r'page=(\d+)&per_page=\d+>; rel="last"', response.headers["Link"])
    last_page = int(last_page_match.group(1)) if last_page_match else 1

    pbar = tqdm(range(1, last_page + 1), desc="Fetching commits", unit="page", leave=False)

    for page in pbar:
        compare_url = f"https://api.github.com/repos/{org_name}/{repo_name}/compare/{from_ref}...{to_ref}?page={page}&per_page=100"
        response = requests.get(compare_url, headers=headers)
        commits = response.json()["commits"]
        commit_info.extend([CommitInfo.from_data(c) for c in commits])

    return commit_info


def main():
    description = """Fetch external contributions between two tags in the InvokeAI GitHub repository. Useful for generating a list of contributors to include in release notes.

When the GitHub username for a commit is not available, the committer name is used instead and an asterisk appended to the name.

Example output (note the second commit has an asterisk appended to the name):
171f2aa20ddfefa23c5edbeb2849c4bd601fe104: rohinish404 - fix(ui): image not getting selected (https://api.github.com/repos/invoke-ai/InvokeAI/commits/171f2aa20ddfefa23c5edbeb2849c4bd601fe104)
0bb0e226dcec8a17e843444ad27c29b4821dad7c: Mark E. Shoulson* - Flip default ordering of workflow library; #5477 (https://api.github.com/repos/invoke-ai/InvokeAI/commits/0bb0e226dcec8a17e843444ad27c29b4821dad7c)
"""

    parser = ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument("--token", dest="token", type=str, default=None, help="The GitHub token to use")
    parser.add_argument("--from", dest="from_ref", type=str, help="The start reference (commit, tag, etc)")
    parser.add_argument("--to", dest="to_ref", type=str, help="The end reference (commit, tag, etc)")

    args = parser.parse_args()

    org_name = "invoke-ai"
    repo_name = "InvokeAI"

    # List of members of the organization, including usernames and known display names,
    # any of which may be used in the commit data. Used to filter out commits.
    org_members = [
        "blessedcoolant",
        "brandonrising",
        "chainchompa",
        "ebr",
        "Eugene Brodsky",
        "hipsterusername",
        "Kent Keirsey",
        "lstein",
        "Lincoln Stein",
        "maryhipp",
        "Mary Hipp Rogers",
        "Mary Hipp",
        "psychedelicious",
        "RyanJDick",
        "Ryan Dick",
    ]

    all_commits = fetch_commits_between_tags(
        org_name=org_name,
        repo_name=repo_name,
        from_ref=args.from_ref,
        to_ref=args.to_ref,
        token=args.token,
    )
    filtered_commits = filter(lambda x: x.author not in org_members, all_commits)

    for commit in filtered_commits:
        print(commit)


if __name__ == "__main__":
    main()

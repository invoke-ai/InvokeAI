# Pull Request Merge Policy

This document outlines the process for reviewing and merging pull requests (PRs) into the InvokeAI repository.

## Review Process

### 1. Assignment

One of the repository maintainers will assign collaborators to review a pull request. The assigned reviewer(s) will be responsible for conducting the code review.

### 2. Review and Iteration

The assignee is responsible for:
- Reviewing the PR thoroughly
- Providing constructive feedback
- Iterating with the PR author until the assignee is satisfied that the PR is fit to merge
- Ensuring the PR meets code quality standards, follows project conventions, and doesn't introduce bugs or regressions

### 3. Approval and Notification

Once the assignee is satisfied with the PR:
- The assignee approves the PR
- The assignee alerts one of the maintainers that the PR is ready for merge using the **#request-reviews Discord channel**

### 4. Final Merge

One of the maintainers is responsible for:
- Performing a final check of the PR
- Merging the PR into the appropriate branch

**Important:** Collaborators are strongly discouraged from merging PRs on their own, except in case of emergency (e.g., critical bug fix and no maintainer is available).

### 5. Release Policy

Once a feature release candidate is published, no feature PRs are to
be merged into main. Only bugfixes are allowed until the final
release.

## Best Practices

### Clean Commit History

To encourage a clean development log, PR authors are encouraged to use `git rebase -i` to suppress trivial commit messages (e.g., `ruff` and `prettier` formatting fixes) after the PR is accepted but before it is merged.

### Merge Strategy

The maintainer will perform either a **3-way merge** or **squash merge** when merging a PR into the `main` branch. This approach helps avoid rebase conflict hell and maintains a cleaner project history.

### Attribution

The PR author should reference any papers, source code or
documentation that they used while creating the code both in the PR
and as comments in the code itself. If there are any licensing
restrictions, these should be linked to and/or reproduced in the repo
root.


## Summary

This policy ensures that:
- All PRs receive proper review from assigned collaborators
- Maintainers have final oversight before code enters the main branch
- The commit history remains clean and meaningful
- Merge conflicts are minimized through appropriate merge strategies

// The community showcase: entries live in `src/content/showcase/` and arrive through a GitHub issue form.

/** The issue form for showcase submissions, `.github/ISSUE_TEMPLATE/SHOWCASE.yml`. */
export const SHOWCASE_SUBMIT_URL = 'https://github.com/invoke-ai/InvokeAI/issues/new?template=SHOWCASE.yml';

/**
 * Newest first. Entries from the same date are ordered by title, then id, so builds are stable.
 * @template {{ id: string, data: { date: Date, title: string } }} T
 * @param {readonly T[]} entries
 * @returns {T[]}
 */
export const sortShowcase = (entries) =>
  [...entries].sort(
    (a, b) =>
      b.data.date.getTime() - a.data.date.getTime() ||
      a.data.title.localeCompare(b.data.title) ||
      a.id.localeCompare(b.id),
  );

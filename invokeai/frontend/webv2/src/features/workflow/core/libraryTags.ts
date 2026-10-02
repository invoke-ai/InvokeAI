/** Normalize comma-separated tags like the backend and omit tags absent from the current category. */

export interface WorkflowTagCount {
  tag: string;
  count: number;
}

/** Splits a record's `tags` column into trimmed, non-empty tags. */
export const parseWorkflowTags = (tags: string | null | undefined): string[] => {
  if (!tags) {
    return [];
  }

  return tags
    .split(',')
    .map((tag) => tag.trim())
    .filter((tag) => tag.length > 0);
};

/**
 * Merge case variants using maximum count, not sum, because LIKE totals overlap. Choose largest-count casing,
 * breaking ties lexicographically for stable labels.
 */
export const mergeTagCountsByCase = (counts: readonly WorkflowTagCount[]): WorkflowTagCount[] => {
  const merged = new Map<string, WorkflowTagCount>();

  for (const entry of counts) {
    const key = entry.tag.toLowerCase();
    const existing = merged.get(key);

    if (!existing) {
      merged.set(key, { count: entry.count, tag: entry.tag });
      continue;
    }

    // Compared before the max is folded in, so `existing.count` is still the
    // count of the row whose casing currently wins.
    if (entry.count > existing.count || (entry.count === existing.count && entry.tag.localeCompare(existing.tag) < 0)) {
      existing.tag = entry.tag;
    }

    existing.count = Math.max(existing.count, entry.count);
  }

  return [...merged.values()];
};

/** Orders tag chips by how many workflows carry them, then alphabetically; case-duplicate and empty tags are merged away. */
export const sortTagCounts = (counts: readonly WorkflowTagCount[]): WorkflowTagCount[] =>
  mergeTagCountsByCase(counts)
    .filter((entry) => entry.count > 0)
    .sort((left, right) => right.count - left.count || left.tag.localeCompare(right.tag));

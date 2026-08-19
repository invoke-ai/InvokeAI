/**
 * Workflow library tag helpers. The backend stores a record's tags as one
 * comma-separated string and reports per-tag counts as a plain map, so both
 * shapes need the same normalization everywhere they are rendered: split and
 * trim exactly like the record service does, and drop tags no workflow in the
 * current category carries.
 */

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
 * Folds rows that differ only in casing into one chip. The backend counts tags
 * exactly as they were stored, so a library where some workflows say `sdxl` and
 * others `SDXL` reports two rows for what is, to the user, one tag — filtering
 * is a case-insensitive SQLite LIKE, so both chips return the same workflows.
 * Counts are summed and the label is the casing of the single biggest
 * contributing row (ties broken lexicographically, so the chip never depends on
 * the order the backend happened to return).
 */
export const mergeTagCountsByCase = (counts: readonly WorkflowTagCount[]): WorkflowTagCount[] => {
  const merged = new Map<string, { count: number; tag: string; tagCount: number }>();

  for (const entry of counts) {
    const key = entry.tag.toLowerCase();
    const existing = merged.get(key);

    if (!existing) {
      merged.set(key, { count: entry.count, tag: entry.tag, tagCount: entry.count });
      continue;
    }

    existing.count += entry.count;

    // Compared against the winning *row's* own count, not the running total.
    if (
      entry.count > existing.tagCount ||
      (entry.count === existing.tagCount && entry.tag.localeCompare(existing.tag) < 0)
    ) {
      existing.tag = entry.tag;
      existing.tagCount = entry.count;
    }
  }

  return [...merged.values()].map(({ count, tag }) => ({ count, tag }));
};

/** Orders tag chips by how many workflows carry them, then alphabetically; case-duplicate and empty tags are merged away. */
export const sortTagCounts = (counts: readonly WorkflowTagCount[]): WorkflowTagCount[] =>
  mergeTagCountsByCase(counts)
    .filter((entry) => entry.count > 0)
    .sort((left, right) => right.count - left.count || left.tag.localeCompare(right.tag));

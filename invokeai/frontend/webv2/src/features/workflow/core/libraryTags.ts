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
 * Folds rows that differ only in casing into one chip. The backend reports one
 * row per *stored* casing, so a library where some workflows say `sdxl` and
 * others `SDXL` gets two rows for what is, to the user, one tag.
 *
 * The merged count is the **maximum** of the variants, never their sum: the
 * backend counts each requested tag with `tags LIKE '%tag%'`, and SQLite's LIKE
 * is case-insensitive, so every variant's count is *already* the full
 * case-insensitive total — the same workflows, counted once per casing. Summing
 * them showed "sdxl 4" over two workflows. Given those semantics the variants
 * are equal anyway; taking the max just refuses to be wrong if they ever drift.
 *
 * The label is the casing of the biggest contributing row, ties broken
 * lexicographically — and since equal counts are the normal case, that tiebreak
 * is what actually decides, which keeps the chip stable whatever order the
 * backend returned.
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

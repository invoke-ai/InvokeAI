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

/** Orders tag chips by how many workflows carry them, then alphabetically; empty tags are omitted. */
export const sortTagCounts = (counts: readonly WorkflowTagCount[]): WorkflowTagCount[] =>
  counts
    .filter((entry) => entry.count > 0)
    .sort((left, right) => right.count - left.count || left.tag.localeCompare(right.tag));

/**
 * Deterministic counters for the panel's performance contract: how many times each row committed
 * and how many thumbnails were requested. Read by budget tests only; production never reads them.
 */
const rowCommits = new Map<string, number>();

export const recordLayerRowCommit = (id: string): void => {
  rowCommits.set(id, (rowCommits.get(id) ?? 0) + 1);
};

/** Commits per row since the last reset, as a plain object for assertions. */
export const getLayerRowCommits = (): Record<string, number> => Object.fromEntries(rowCommits);

export const resetLayerRowCommits = (): void => {
  rowCommits.clear();
};

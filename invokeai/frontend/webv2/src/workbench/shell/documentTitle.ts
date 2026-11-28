/** Lead tab titles with progress because background tabs hide other indicators and truncate labels from the right. */

/** Mirrors the static `<title>` in index.html, which is what we restore to. */
export const DOCUMENT_TITLE_BASE = 'Invoke Workbench';

/** Shorten the running title's product label to leave room for progress in crowded tabs. */
export const DOCUMENT_TITLE_PRODUCT = 'Invoke';

export interface DocumentTitleLabels {
  /** Fallback for "running, but nothing quantified yet", e.g. "Generating". */
  generating: string;
  /** Already-interpolated count, e.g. "4 queued". */
  queued: string;
}

/**
 * Once concurrent execution is observed, represent batch progress by settled backend slots
 * rather than unrelated percentages reported by individual GPUs.
 */
export const getBatchDocumentTitlePercent = ({
  batchCount,
  completedCount,
}: {
  batchCount: number;
  completedCount: number;
}): number | null => {
  if (completedCount <= 0) {
    return null;
  }
  const settledCount = Math.min(batchCount, completedCount);
  const roundedPercent = Math.round((settledCount / batchCount) * 100);
  // Rounding 199/200 yields 100%, but the last image is still running.
  return settledCount === batchCount ? 100 : Math.min(99, roundedPercent);
};

export const formatDocumentTitle = ({
  baseTitle = DOCUMENT_TITLE_BASE,
  current,
  labels,
  percent,
  productLabel = DOCUMENT_TITLE_PRODUCT,
  total,
}: {
  /** Shown when nothing is running; restored verbatim on unmount. */
  baseTitle?: string;
  /** One-based ordinal of the image being generated; 0 when nothing has started. */
  current: number;
  labels: DocumentTitleLabels;
  /** Whole percent for the active image, or null while indeterminate. */
  percent: number | null;
  /** Trails the progress numbers, where width is scarce. */
  productLabel?: string;
  total: number;
}): string => {
  if (total === 0) {
    return baseTitle;
  }

  if (current === 0) {
    return `${labels.queued} · ${productLabel}`;
  }

  const parts = [percent === null ? null : `${percent}%`, total > 1 ? `${current}/${total}` : null].filter(Boolean);

  return `${parts.length > 0 ? parts.join(' · ') : labels.generating} · ${productLabel}`;
};

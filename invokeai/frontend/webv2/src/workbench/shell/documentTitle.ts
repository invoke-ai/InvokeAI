/**
 * Generation progress in the browser tab, for the batch you walked away from.
 *
 * Batches run for minutes and people switch tabs while they do; every other
 * progress surface in the app is invisible the moment this document is not the
 * foreground one. Tab titles truncate hard and from the right, so the numbers
 * lead and the product name trails.
 */

/** Mirrors the static `<title>` in index.html, which is what we restore to. */
export const DOCUMENT_TITLE_BASE = 'Invoke V7 Workbench';

/**
 * The trailing label while a batch runs, deliberately shorter than the idle
 * title. The idle title can afford to name the product precisely; a running one
 * is competing for the same few characters as the numbers, and those are the
 * part worth reading. Spending the width on "V7 Workbench" is what pushes the
 * percent out of a pinned or crowded tab.
 */
export const DOCUMENT_TITLE_PRODUCT = 'Invoke';

export interface DocumentTitleLabels {
  /** Fallback for "running, but nothing quantified yet", e.g. "Generating". */
  generating: string;
  /** Already-interpolated count, e.g. "4 queued". */
  queued: string;
}

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

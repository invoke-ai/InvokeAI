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
  total,
}: {
  baseTitle?: string;
  /** One-based ordinal of the image being generated; 0 when nothing has started. */
  current: number;
  labels: DocumentTitleLabels;
  /** Whole percent for the active image, or null while indeterminate. */
  percent: number | null;
  total: number;
}): string => {
  if (total === 0) {
    return baseTitle;
  }

  if (current === 0) {
    return `${labels.queued} · ${baseTitle}`;
  }

  const parts = [percent === null ? null : `${percent}%`, total > 1 ? `${current}/${total}` : null].filter(Boolean);

  return `${parts.length > 0 ? parts.join(' · ') : labels.generating} · ${baseTitle}`;
};

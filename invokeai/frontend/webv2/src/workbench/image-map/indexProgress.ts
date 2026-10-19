/**
 * Pure progress arithmetic reports completed work and time since progress. Generation pauses make remaining-time
 * estimates unreliable.
 */

/** Embedding-index counts as the backend reports them. */
export interface ImageIndexCounts {
  total: number;
  embedded: number;
  pending: number;
  /** Given up on after repeated failures; excluded from `pending`. */
  failed: number;
}

/** Stall threshold should exceed ordinary batch intervals while revealing prolonged inactivity. */
export const STALE_AFTER_MS = 2 * 60_000;

/** Images the index is finished with, whether they embedded or were given up on. */
const getProcessed = (counts: ImageIndexCounts): number => Math.max(0, counts.total - counts.pending);

/**
 * Completion percent derives from pending so drained queues reach 100%, including abandoned items without
 * shrinking the denominator.
 */
export const getIndexPercent = (counts: ImageIndexCounts): number => {
  if (counts.total <= 0) {
    return 0;
  }

  return Math.min(100, Math.max(0, (getProcessed(counts) / counts.total) * 100));
};

/** Use one percent for visuals, text and announcements; cap below 100 while pending work remains. */
export const getDisplayPercent = (counts: ImageIndexCounts): number => {
  const percent = Math.round(getIndexPercent(counts));

  return counts.pending > 0 ? Math.min(99, percent) : percent;
};

/** Whether there is embedding work in flight worth showing progress for. */
export const isIndexing = (counts: ImageIndexCounts | null): counts is ImageIndexCounts =>
  counts !== null && counts.pending > 0;

/** Detect completed index work, ignoring total changes from gallery saves/deletions. */
export const hasProgressed = (previous: ImageIndexCounts | null, next: ImageIndexCounts): boolean =>
  previous === null || previous.embedded !== next.embedded || previous.failed !== next.failed;

/** "45s", "4m 30s", "1h 05m" - the coarser the total, the coarser the unit. */
export const formatDuration = (seconds: number): string => {
  if (!Number.isFinite(seconds) || seconds < 0) {
    return '';
  }

  // Rounded to whole seconds up front, so the unit split below can never carry
  // - rounding each part on its own renders 119.6s as "1m 60s".
  // Never "0s": a sub-second age still means some time has passed.
  const total = Math.max(1, Math.round(seconds));

  if (total < 60) {
    return `${total}s`;
  }

  if (total < 3600) {
    return `${Math.floor(total / 60)}m ${String(total % 60).padStart(2, '0')}s`;
  }

  return `${Math.floor(total / 3600)}h ${String(Math.floor((total % 3600) / 60)).padStart(2, '0')}m`;
};

const formatCount = (value: number): string => value.toLocaleString();

export interface IndexProgressDescription {
  /** Rounded, and held below 100 while anything is outstanding. Paint this too. */
  percent: number;
  /** e.g. "1,204 of 4,312 images" — done, out of the whole gallery. */
  counts: string;
  /** The same pair with no words, for the footer's single line: "1,204/4,312". */
  compact: string;
  /** Set once the counts have stood still long enough to be worth saying. */
  stale: string | null;
  /** Set only once images have been given up on. */
  skipped: string | null;
}

/**
 * Shared panel/footer progress wording. `sinceMs` reports inactivity without claiming whether the worker is paused
 * for generation or has failed.
 */
export const describeIndexProgress = (counts: ImageIndexCounts, sinceMs = 0): IndexProgressDescription => ({
  compact: `${formatCount(getProcessed(counts))}/${formatCount(counts.total)}`,
  counts: `${formatCount(getProcessed(counts))} of ${formatCount(counts.total)} images`,
  percent: getDisplayPercent(counts),
  skipped: counts.failed > 0 ? `${formatCount(counts.failed)} skipped after repeated failures` : null,
  stale: sinceMs >= STALE_AFTER_MS ? `No progress reported for ${formatDuration(sinceMs / 1000)}` : null,
});

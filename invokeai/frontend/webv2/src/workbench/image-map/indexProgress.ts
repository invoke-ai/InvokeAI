/**
 * Progress arithmetic for the embedding index, kept apart from both the store
 * that holds the counts and the components that draw them, so the wording and
 * the numbers behind it can be tested without a socket or a DOM.
 *
 * Deliberately no time estimate. The indexer stands down for the whole of
 * every generation (`_wait_for_idle_generation`) and reports nothing while it
 * waits, so the gaps between reports are mostly other people's work, not the
 * index's. Measuring across them makes the estimate pessimistic by ~200x;
 * excluding them makes it optimistic by ~10x, because the pauses still to come
 * are left out; and no threshold separates the two, since embedding a batch on
 * a slow CPU takes exactly as long as a fast generation. What the panel can
 * say honestly is how far the run has got and when it last moved, which is all
 * that was needed to tell progress from a stall.
 */

/** Embedding-index counts as the backend reports them. */
export interface ImageIndexCounts {
  total: number;
  embedded: number;
  pending: number;
  /** Given up on after repeated failures; excluded from `pending`. */
  failed: number;
}

/**
 * How long the counts may stand still before the panel says so.
 *
 * Above any single batch — the worker emits one report per batch of
 * `image_index_batch_size`, seconds apart on a GPU and up to a minute on a
 * slow CPU — so ordinary embedding never trips it, and short enough that a run
 * that has actually stopped does not keep looking busy.
 */
export const STALE_AFTER_MS = 2 * 60_000;

/** Images the index is finished with, whether they embedded or were given up on. */
const getProcessed = (counts: ImageIndexCounts): number => Math.max(0, counts.total - counts.pending);

/**
 * Share of the gallery the index is finished with, 0-100.
 *
 * Counted off `pending` so the bar is full exactly when the queue drains.
 * Images given up on are finished with too: leaving them out parks the bar
 * short of full forever, and taking them out of the denominator instead
 * shrinks the total under the user as they retire one by one.
 */
export const getIndexPercent = (counts: ImageIndexCounts): number => {
  if (counts.total <= 0) {
    return 0;
  }

  return Math.min(100, Math.max(0, (getProcessed(counts) / counts.total) * 100));
};

/**
 * What to paint, print and announce — one value for all three, so the bar
 * cannot sit visually full beside a number that says otherwise. Held below
 * full while anything is outstanding: 99.7% rounded up tells a screen reader
 * the run is done with 300 images still queued.
 */
export const getDisplayPercent = (counts: ImageIndexCounts): number => {
  const percent = Math.round(getIndexPercent(counts));

  return counts.pending > 0 ? Math.min(99, percent) : percent;
};

/** Whether there is embedding work in flight worth showing progress for. */
export const isIndexing = (counts: ImageIndexCounts | null): counts is ImageIndexCounts =>
  counts !== null && counts.pending > 0;

/**
 * Whether one report represents progress over another, rather than the same
 * work re-reported. `total` moves whenever anyone saves or deletes an image,
 * so it cannot be part of this: a generation running while the indexer waits
 * for it would otherwise look like the index moving.
 */
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
 * The strings the progress UI shows. Assembled here rather than in the
 * components so both the panel and the footer say the same thing, and so the
 * wording is covered by the same tests as the numbers behind it.
 *
 * `sinceMs` is how long the counts have been standing. The note it produces
 * states the fact and stops there: nothing the client can see distinguishes an
 * indexer waiting out a generation from one that has died, so it must not
 * claim to know which.
 */
export const describeIndexProgress = (counts: ImageIndexCounts, sinceMs = 0): IndexProgressDescription => ({
  compact: `${formatCount(getProcessed(counts))}/${formatCount(counts.total)}`,
  counts: `${formatCount(getProcessed(counts))} of ${formatCount(counts.total)} images`,
  percent: getDisplayPercent(counts),
  skipped: counts.failed > 0 ? `${formatCount(counts.failed)} skipped after repeated failures` : null,
  stale: sinceMs >= STALE_AFTER_MS ? `No progress reported for ${formatDuration(sinceMs / 1000)}` : null,
});

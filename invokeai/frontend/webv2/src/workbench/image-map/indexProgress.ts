/**
 * Progress arithmetic for the embedding index, kept apart from both the store
 * that holds the counts and the components that draw them: throughput and the
 * time estimate derived from it are the only part with real logic in it, and
 * they are worth testing without a socket or a DOM.
 */

/** Embedding-index counts as the backend reports them. */
export interface ImageIndexCounts {
  total: number;
  embedded: number;
  pending: number;
  /** Given up on after repeated failures; excluded from `pending`. */
  failed: number;
}

/** One observation of `embedded` at a point in time, used to estimate throughput. */
export interface IndexRateSample {
  at: number;
  embedded: number;
}

/**
 * Samples older than this are dropped. The rate then reflects recent
 * throughput rather than an average over the whole run, which matters because
 * the worker parks for the length of any generation (see
 * `_wait_for_idle_generation`): a run-long average would keep charging the
 * estimate for a pause that has already ended.
 */
export const RATE_WINDOW_MS = 60_000;

/** Bound on retained samples, so a long backfill cannot grow the array without limit. */
const MAX_SAMPLES = 240;

/**
 * Add an observation, dropping the ones that have aged out of the window.
 *
 * A drop in `embedded` means the counts are no longer comparable to what came
 * before - the model changed, or images were deleted - so the history is
 * discarded rather than producing a negative (and then nonsensical) rate.
 */
export const appendRateSample = (samples: IndexRateSample[], sample: IndexRateSample): IndexRateSample[] => {
  const previous = samples.at(-1);

  if (previous && (sample.embedded < previous.embedded || sample.at < previous.at)) {
    return [sample];
  }

  const appended = [...samples, sample];
  const withinWindow = appended.filter((entry) => sample.at - entry.at <= RATE_WINDOW_MS);

  // Events can arrive further apart than the window (the worker parks while a
  // generation runs). Keeping the last two regardless leaves an anchor to
  // measure against instead of silently dropping to "no estimate".
  const retained = withinWindow.length >= 2 ? withinWindow : appended.slice(-2);

  return retained.length > MAX_SAMPLES ? retained.slice(-MAX_SAMPLES) : retained;
};

/** Images embedded per second over the retained window; null until measurable. */
export const estimateRate = (samples: IndexRateSample[]): number | null => {
  const first = samples.at(0);
  const last = samples.at(-1);

  if (!first || !last) {
    return null;
  }

  const elapsedSeconds = (last.at - first.at) / 1000;
  const embedded = last.embedded - first.embedded;

  if (elapsedSeconds <= 0 || embedded <= 0) {
    return null;
  }

  return embedded / elapsedSeconds;
};

/** Percentage of eligible images that carry an embedding, clamped to 0-100. */
export const getIndexPercent = (counts: ImageIndexCounts): number => {
  if (counts.total <= 0) {
    return 0;
  }

  return Math.min(100, Math.max(0, (counts.embedded / counts.total) * 100));
};

/** Seconds until the queue drains at the measured rate; null when it cannot be estimated. */
export const getEtaSeconds = (counts: ImageIndexCounts, rate: number | null): number | null => {
  if (rate === null || rate <= 0 || counts.pending <= 0) {
    return null;
  }

  return counts.pending / rate;
};

/** Whether there is embedding work in flight worth showing progress for. */
export const isIndexing = (counts: ImageIndexCounts | null): counts is ImageIndexCounts =>
  counts !== null && counts.pending > 0;

/** "45s", "4m 30s", "1h 05m" - the coarser the total, the coarser the unit. */
export const formatDuration = (seconds: number): string => {
  if (!Number.isFinite(seconds) || seconds < 0) {
    return '';
  }

  if (seconds < 60) {
    // Never "0s": a sub-second estimate still means work is outstanding.
    return `${Math.max(1, Math.round(seconds))}s`;
  }

  if (seconds < 3600) {
    return `${Math.floor(seconds / 60)}m ${String(Math.round(seconds % 60)).padStart(2, '0')}s`;
  }

  return `${Math.floor(seconds / 3600)}h ${String(Math.floor((seconds % 3600) / 60)).padStart(2, '0')}m`;
};

const formatCount = (value: number): string => value.toLocaleString();

export interface IndexProgressDescription {
  percent: number;
  /** e.g. "1,204 of 4,312 images" */
  counts: string;
  /** e.g. "About 4m 30s remaining", or the placeholder until a rate exists. */
  eta: string;
  /** Set only once images have been given up on. */
  skipped: string | null;
}

/**
 * The strings the progress UI shows. Assembled here rather than in the
 * components so both the panel and the footer say the same thing, and so the
 * wording is covered by the same tests as the arithmetic behind it.
 */
export const describeIndexProgress = (counts: ImageIndexCounts, rate: number | null): IndexProgressDescription => {
  const etaSeconds = getEtaSeconds(counts, rate);

  return {
    counts: `${formatCount(counts.embedded)} of ${formatCount(counts.total)} images`,
    eta: etaSeconds === null ? 'Estimating time remaining…' : `About ${formatDuration(etaSeconds)} remaining`,
    percent: getIndexPercent(counts),
    skipped: counts.failed > 0 ? `${formatCount(counts.failed)} skipped after repeated failures` : null,
  };
};

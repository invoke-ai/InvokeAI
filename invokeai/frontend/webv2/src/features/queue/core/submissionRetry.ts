import type { QueueSubmissionRetry } from '@features/queue/core/historyTypes';

export interface QueueSubmissionFailure {
  message: string;
  status: number | null;
  /** `fetch` rejected, was aborted or timed out. */
  unanswered: boolean;
}

// Keyed by reason, so a new reason fails to compile until restore accepts it.
const SUBMISSION_RETRY_REASONS: Readonly<Record<QueueSubmissionRetry['reason'], true>> = {
  error: true,
  'server-error': true,
  unavailable: true,
  unreachable: true,
};

// A 5xx usually means the request itself is bad, so a run gives up after 3 consecutive 500s over at least 10 s.
const MAX_SERVER_ERROR_SUBMISSIONS = 3;
const MIN_SERVER_ERROR_WINDOW_MS = 10_000;
// Minutes, not retry delays: a hidden tab's timers fire about once a minute and must not reset a streak.
const SERVER_ERROR_STREAK_GAP_MS = 5 * 60_000;
const UNAVAILABLE_STATUSES = [408, 429, 502, 503, 504];
// Waits run from the recorded failure, so a reload keeps them; wall-clock jumps never wait longer than one delay.
const SUBMISSION_RETRY_BASE_MS = 1_000;
const SUBMISSION_RETRY_MAX_MS = 15_000;

// A 5xx without a JSON body is the whole response, and the record is persisted on every attempt.
const MAX_SUBMISSION_RETRY_MESSAGE_LENGTH = 500;

const boundMessage = (message: string): string => {
  if (message.length <= MAX_SUBMISSION_RETRY_MESSAGE_LENGTH) {
    return message;
  }
  const kept = message.slice(0, MAX_SUBMISSION_RETRY_MESSAGE_LENGTH - 1);
  return `${/[\uD800-\uDBFF]$/.test(kept) ? kept.slice(0, -1) : kept}…`;
};

const getFailureReason = ({ status, unanswered }: QueueSubmissionFailure): QueueSubmissionRetry['reason'] => {
  if (status === null) {
    return unanswered ? 'unreachable' : 'error';
  }
  if (UNAVAILABLE_STATUSES.includes(status)) {
    return 'unavailable';
  }
  return status >= 500 ? 'server-error' : 'error';
};

/** A 4xx the same submission would get again; 408 and 429 retry. */
export const isRejectedSubmissionStatus = (status: number): boolean =>
  status >= 400 && status < 500 && !UNAVAILABLE_STATUSES.includes(status);

/** The gap is measured from the attempt's start, so a slow answer cannot reset its own streak. */
export const createSubmissionRetry = (
  failure: QueueSubmissionFailure,
  previous: QueueSubmissionRetry | undefined,
  { attemptStartedAt, now }: { attemptStartedAt: number; now: number }
): QueueSubmissionRetry => {
  const base = {
    failedAt: now,
    failureCount: (previous?.failureCount ?? 0) + 1,
    message: boundMessage(failure.message),
    status: failure.status,
  };
  const reason = getFailureReason(failure);
  if (reason !== 'server-error') {
    return { ...base, reason, serverErrorCount: 0, serverErrorSince: null };
  }
  const streak =
    previous?.reason === 'server-error' && attemptStartedAt - previous.failedAt <= SERVER_ERROR_STREAK_GAP_MS
      ? previous
      : undefined;
  const since = streak?.serverErrorSince ?? null;
  return {
    ...base,
    reason,
    serverErrorCount: (streak?.serverErrorCount ?? 0) + 1,
    serverErrorSince: since !== null && since <= now ? since : now,
  };
};

export const getSubmissionRetryDelayMs = ({ failureCount }: QueueSubmissionRetry): number =>
  Math.min(SUBMISSION_RETRY_BASE_MS * 2 ** Math.max(failureCount - 1, 0), SUBMISSION_RETRY_MAX_MS);

export const getSubmissionRetryWaitMs = (retry: QueueSubmissionRetry, now: number): number => {
  const delayMs = getSubmissionRetryDelayMs(retry);
  return Math.min(Math.max(retry.failedAt + delayMs - now, 0), delayMs);
};

export const isSubmissionRetryExhausted = (retry: QueueSubmissionRetry): boolean =>
  retry.reason === 'server-error' &&
  retry.serverErrorSince !== null &&
  retry.serverErrorCount >= MAX_SERVER_ERROR_SUBMISSIONS &&
  retry.failedAt - retry.serverErrorSince >= MIN_SERVER_ERROR_WINDOW_MS;

export const isSameSubmissionRetry = (left: QueueSubmissionRetry | undefined, right: QueueSubmissionRetry): boolean =>
  left !== undefined &&
  left.reason === right.reason &&
  left.failureCount === right.failureCount &&
  left.serverErrorCount === right.serverErrorCount &&
  left.serverErrorSince === right.serverErrorSince &&
  left.status === right.status &&
  left.message === right.message &&
  left.failedAt === right.failedAt;

export const withoutSubmissionRetry = <T extends { submissionRetry?: QueueSubmissionRetry }>(
  item: T
): Omit<T, 'submissionRetry'> => {
  if (item.submissionRetry === undefined) {
    return item;
  }
  const { submissionRetry: _, ...rest } = item;
  return rest;
};

const isSubmissionRetryReason = (value: unknown): value is QueueSubmissionRetry['reason'] =>
  typeof value === 'string' && Object.hasOwn(SUBMISSION_RETRY_REASONS, value);
const isFiniteNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const isCount = (value: unknown): value is number => Number.isSafeInteger(value);

/** Drops a record with broken invariants or an unknown reason; older records may hold an unbounded message. */
export const restoreSubmissionRetry = (value: unknown): QueueSubmissionRetry | undefined => {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return undefined;
  }
  const { failedAt, failureCount, message, reason, serverErrorCount, serverErrorSince, status } = value as Record<
    string,
    unknown
  >;
  if (
    !isSubmissionRetryReason(reason) ||
    typeof message !== 'string' ||
    !isFiniteNumber(failedAt) ||
    (status !== null && !(isCount(status) && status >= 100 && status <= 599)) ||
    !isCount(serverErrorCount) ||
    serverErrorCount < 0 ||
    !isCount(failureCount) ||
    failureCount < Math.max(serverErrorCount, 1) ||
    (reason === 'server-error') !== serverErrorCount > 0 ||
    (serverErrorCount === 0
      ? serverErrorSince !== null
      : !isFiniteNumber(serverErrorSince) || serverErrorSince > failedAt)
  ) {
    return undefined;
  }
  return {
    failedAt,
    failureCount,
    message: boundMessage(message),
    reason,
    serverErrorCount,
    serverErrorSince: serverErrorSince as number | null,
    status,
  };
};

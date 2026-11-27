import type { QueueSubmissionRetry } from '@features/queue/core/historyTypes';

import { describe, expect, it } from 'vitest';

import {
  createSubmissionRetry,
  isRejectedSubmissionStatus,
  restoreSubmissionRetry,
  type QueueSubmissionFailure,
} from './submissionRetry';

const serverError: QueueSubmissionFailure = { message: 'graph validation crashed', status: 500, unanswered: false };
const unavailable: QueueSubmissionFailure = { message: 'unavailable', status: 503, unanswered: false };

const persisted: QueueSubmissionRetry = {
  failedAt: 12_000,
  failureCount: 3,
  message: 'graph validation crashed',
  reason: 'server-error',
  serverErrorCount: 2,
  serverErrorSince: 1_000,
  status: 500,
};

describe('createSubmissionRetry', () => {
  // Keyed by reason, so a new reason needs a case.
  const failures: Record<QueueSubmissionRetry['reason'], QueueSubmissionFailure> = {
    error: { message: 'Invalid queue enqueue response', status: null, unanswered: false },
    'server-error': serverError,
    unavailable,
    unreachable: { message: 'Failed to fetch', status: null, unanswered: true },
  };

  it.each(Object.entries(failures))('keeps a %s retry through persistence and restore', (reason, failure) => {
    const retry = createSubmissionRetry(failure, undefined, { attemptStartedAt: 1_000, now: 1_000 });

    expect(retry.reason).toBe(reason);
    expect(restoreSubmissionRetry(JSON.parse(JSON.stringify(retry)))).toEqual(retry);
  });
});

describe('isRejectedSubmissionStatus', () => {
  it.each([
    [400, true],
    [422, true],
    [408, false],
    [429, false],
    [500, false],
  ])('fails the run on %i: %s', (status, rejected) => {
    expect(isRejectedSubmissionStatus(status)).toBe(rejected);
  });
});

describe('restoreSubmissionRetry', () => {
  it.each([
    { name: 'a server error that is not counted', value: { ...persisted, serverErrorCount: 0 } },
    { name: 'a streak starting after its latest failure', value: { ...persisted, serverErrorSince: 13_000 } },
    { name: 'fewer failures than server errors', value: { ...persisted, failureCount: 1 } },
    // As a later build might write it.
    { name: 'a reason this build does not know', value: { ...persisted, reason: 'throttled' } },
    { name: 'no record', value: null },
  ])('drops $name', ({ value }) => {
    expect(restoreSubmissionRetry(value)).toBeUndefined();
  });

  it('bounds the message again, as records from before the bound may hold a whole response', () => {
    const page = `<html>${'x'.repeat(5_000)}</html>`;

    expect(restoreSubmissionRetry({ ...persisted, message: page })).toEqual({
      ...persisted,
      message: `${page.slice(0, 499)}…`,
    });
  });
});

import { describe, expect, it } from 'vitest';

import { isVideoMissingError } from './videoFieldErrors';

describe('isVideoMissingError', () => {
  it('is true for a 404 — the video is confirmed gone', () => {
    expect(isVideoMissingError({ status: 404, data: { detail: 'not found' } })).toBe(true);
  });

  it.each([
    // Revoking access to a shared board answers 403 for videos that are all still there.
    // Clearing on it would destroy the workflows pointing at them, and restoring the permission
    // would not bring them back.
    ['denied (403)', { status: 403, data: { detail: 'Not authorized' } }],
    ['auth (401)', { status: 401, data: {} }],
    ['server error (500)', { status: 500, data: {} }],
    ['network failure', { status: 'FETCH_ERROR', error: 'TypeError: Failed to fetch' }],
    ['timeout', { status: 'TIMEOUT_ERROR', error: 'AbortError' }],
    ['parsing failure', { status: 'PARSING_ERROR', originalStatus: 200, data: '', error: 'oops' }],
    ['no error', undefined],
  ])('is false for %s — the field value must be preserved', (_label, error) => {
    expect(isVideoMissingError(error)).toBe(false);
  });
});

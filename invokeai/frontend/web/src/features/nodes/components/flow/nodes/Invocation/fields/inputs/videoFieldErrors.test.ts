import { describe, expect, it } from 'vitest';

import { isVideoUnavailableError } from './videoFieldErrors';

describe('isVideoUnavailableError', () => {
  it('is true for a 404 — the video is confirmed gone', () => {
    expect(isVideoUnavailableError({ status: 404, data: { detail: 'not found' } })).toBe(true);
  });

  it('is true for a 403 — which is how a deleted video answers in multiuser mode', () => {
    // `_assert_video_read_access` decides on `videos.user_id`, and that row is gone, so it
    // cannot tell a deleted video from someone else's. Only an admin gets as far as the 404.
    // Requiring 404 here would leave every deleted video stuck in the workflows that use it.
    expect(isVideoUnavailableError({ status: 403, data: { detail: 'Not authorized' } })).toBe(true);
  });

  it.each([
    ['auth (401)', { status: 401, data: {} }],
    ['server error (500)', { status: 500, data: {} }],
    ['network failure', { status: 'FETCH_ERROR', error: 'TypeError: Failed to fetch' }],
    ['timeout', { status: 'TIMEOUT_ERROR', error: 'AbortError' }],
    ['parsing failure', { status: 'PARSING_ERROR', originalStatus: 200, data: '', error: 'oops' }],
    ['no error', undefined],
  ])('is false for %s — the field value must be preserved', (_label, error) => {
    expect(isVideoUnavailableError(error)).toBe(false);
  });
});

import { describe, expect, it } from 'vitest';

import { isImageUnavailableError } from './imageErrors';

describe('isImageUnavailableError', () => {
  it('is true for a 404 — the image is confirmed gone', () => {
    expect(isImageUnavailableError({ status: 404, data: { detail: 'not found' } })).toBe(true);
  });

  it('is true for a 403 — which is how a deleted image answers in multiuser mode', () => {
    // `assert_image_read_access` decides on `images.user_id`, and that row is gone, so it
    // cannot tell a deleted image from someone else's. Only an admin gets as far as the 404.
    // Requiring 404 here would leave every deleted image stuck in the workflows that use it.
    expect(isImageUnavailableError({ status: 403, data: { detail: 'Not authorized' } })).toBe(true);
  });

  it.each([
    // The one the star/unstar invalidation makes reachable: a name is in `failed_images`
    // because a storage failure interrupted its write, and the refetch that the invalidation
    // triggers hits the same unwell store.
    ['server error (500)', { status: 500, data: {} }],
    ['unauthorized (401)', { status: 401, data: {} }],
    ['network failure', { status: 'FETCH_ERROR', error: 'TypeError: Failed to fetch' }],
    ['timeout', { status: 'TIMEOUT_ERROR', error: 'AbortError' }],
    ['parsing failure', { status: 'PARSING_ERROR', originalStatus: 200, data: '', error: 'oops' }],
    ['no error', undefined],
  ])('is false for %s — the reference must be preserved', (_label, error) => {
    expect(isImageUnavailableError(error)).toBe(false);
  });
});

import { describe, expect, it } from 'vitest';

import { isImageMissingError } from './imageErrors';

describe('isImageMissingError', () => {
  it('is true for a 404 — the image is confirmed gone', () => {
    expect(isImageMissingError({ status: 404, data: { detail: 'not found' } })).toBe(true);
  });

  it.each([
    // The one that matters most: revoking access to a shared board answers 403 for images that
    // are all still there. Clearing on it would destroy the workflows pointing at them, and
    // restoring the permission would not bring them back. The server answers 404 when an image
    // is actually gone, so this arm costs nothing.
    ['denied (403)', { status: 403, data: { detail: 'Not authorized' } }],
    ['unauthorized (401)', { status: 401, data: {} }],
    // A name lands in `failed_images` because a storage failure interrupted its write, and the
    // refetch that the star/unstar invalidation triggers hits the same unwell store.
    ['server error (500)', { status: 500, data: {} }],
    ['network failure', { status: 'FETCH_ERROR', error: 'TypeError: Failed to fetch' }],
    ['timeout', { status: 'TIMEOUT_ERROR', error: 'AbortError' }],
    ['parsing failure', { status: 'PARSING_ERROR', originalStatus: 200, data: '', error: 'oops' }],
    ['no error', undefined],
  ])('is false for %s — the reference must be preserved', (_label, error) => {
    expect(isImageMissingError(error)).toBe(false);
  });
});

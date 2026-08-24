import { describe, expect, it } from 'vitest';

import { isImageMissingError } from './imageErrors';

describe('isImageMissingError', () => {
  it('is true for a 404 — the image is confirmed gone', () => {
    expect(isImageMissingError({ status: 404, data: { detail: 'not found' } })).toBe(true);
  });

  it.each([
    ['auth (401)', { status: 401, data: {} }],
    ['forbidden (403)', { status: 403, data: {} }],
    // The one the star/unstar invalidation makes reachable: the name is in `failed_images`
    // because a storage failure interrupted its write, and the refetch that reports the new
    // state hits the same unwell store.
    ['server error (500)', { status: 500, data: {} }],
    ['network failure', { status: 'FETCH_ERROR', error: 'TypeError: Failed to fetch' }],
    ['parsing failure', { status: 'PARSING_ERROR', originalStatus: 200, data: '', error: 'oops' }],
    ['no error', undefined],
  ])('is false for %s — the reference must be preserved', (_label, error) => {
    expect(isImageMissingError(error)).toBe(false);
  });
});

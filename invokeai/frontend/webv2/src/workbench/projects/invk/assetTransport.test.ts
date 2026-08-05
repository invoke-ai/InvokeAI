import { ApiError } from '@platform/transport/http';
import { describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({ apiFetchRaw: vi.fn() }));

vi.mock('@platform/transport/http', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@platform/transport/http')>()),
  apiFetchRaw: mocks.apiFetchRaw,
}));

import { findExistingVideoNames } from './assetTransport';

describe('findExistingVideoNames', () => {
  it('returns a video name when its lookup succeeds', async () => {
    mocks.apiFetchRaw.mockResolvedValue(new Response(null, { status: 200 }));

    await expect(findExistingVideoNames(['present.mp4'])).resolves.toEqual(new Set(['present.mp4']));
  });

  it.each([403, 404])('treats a %i video lookup as absent', async (status) => {
    mocks.apiFetchRaw.mockResolvedValue(new Response(null, { status }));

    await expect(findExistingVideoNames(['missing.mp4'])).resolves.toEqual(new Set());
  });

  it.each([401, 422, 500])('rejects a %i video lookup with an ApiError', async (status) => {
    mocks.apiFetchRaw.mockResolvedValue(new Response('lookup failed', { status }));

    await expect(findExistingVideoNames(['failed.mp4'])).rejects.toBeInstanceOf(ApiError);
  });

  it('propagates a network rejection unchanged', async () => {
    const networkError = new Error('network unavailable');
    mocks.apiFetchRaw.mockRejectedValue(networkError);

    await expect(findExistingVideoNames(['unreachable.mp4'])).rejects.toBe(networkError);
  });
});

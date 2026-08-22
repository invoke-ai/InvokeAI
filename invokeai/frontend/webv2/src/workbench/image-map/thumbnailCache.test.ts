import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  resolveMany: vi.fn(),
}));

vi.mock('@features/gallery', () => ({
  galleryImages: { resolveMany: mocks.resolveMany },
}));

import { accountLifecycle } from '@platform/state/accountLifecycle';

import { getThumbnailUrl } from './thumbnailCache';

describe('image map thumbnail cache', () => {
  beforeEach(() => {
    mocks.resolveMany.mockReset();
    accountLifecycle.activate('user-a');
  });

  it('caches resolved URLs per image name', async () => {
    mocks.resolveMany.mockResolvedValue([{ thumbnailUrl: '/thumbs/a.png' }]);

    await expect(getThumbnailUrl('a.png')).resolves.toBe('/thumbs/a.png');
    await expect(getThumbnailUrl('a.png')).resolves.toBe('/thumbs/a.png');
    expect(mocks.resolveMany).toHaveBeenCalledTimes(1);
  });

  it('clears settled entries and stale in-flight results on account invalidation', async () => {
    mocks.resolveMany.mockResolvedValue([{ thumbnailUrl: '/thumbs/user-a.png' }]);
    await getThumbnailUrl('a.png');

    // A request still in flight when the account switches must not seed the
    // next account's cache.
    let resolveLate: (images: { thumbnailUrl: string }[]) => void = () => {};
    mocks.resolveMany.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveLate = resolve;
        })
    );
    const late = getThumbnailUrl('b.png');

    accountLifecycle.invalidate();
    resolveLate([{ thumbnailUrl: '/thumbs/stale-b.png' }]);
    await late;

    accountLifecycle.activate('user-b');
    mocks.resolveMany.mockResolvedValue([{ thumbnailUrl: '/thumbs/user-b.png' }]);
    await expect(getThumbnailUrl('a.png')).resolves.toBe('/thumbs/user-b.png');
    await expect(getThumbnailUrl('b.png')).resolves.toBe('/thumbs/user-b.png');
  });
});

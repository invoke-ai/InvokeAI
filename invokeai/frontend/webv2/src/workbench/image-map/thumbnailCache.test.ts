import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  resolve: vi.fn(),
}));

vi.mock('@features/gallery', () => ({
  galleryItems: { resolve: mocks.resolve },
}));

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';

import { getThumbnailUrl } from './thumbnailCache';

describe('image map thumbnail cache', () => {
  beforeEach(() => {
    mocks.resolve.mockReset();
    accountLifecycle.activate('user-a');
  });

  it('caches resolved URLs per item', async () => {
    mocks.resolve.mockResolvedValue({ kind: 'image', thumbnailUrl: '/thumbs/a.png' });

    await expect(getThumbnailUrl({ kind: 'image', name: 'a.png' })).resolves.toEqual({
      durationSeconds: null,
      url: '/thumbs/a.png',
    });
    await expect(getThumbnailUrl({ kind: 'image', name: 'a.png' })).resolves.toEqual({
      durationSeconds: null,
      url: '/thumbs/a.png',
    });
    expect(mocks.resolve).toHaveBeenCalledTimes(1);
  });

  it('resolves a video through its own namespace rather than the image one', async () => {
    // Same name in both namespaces: without the kind, the video's hover card
    // would show the image's thumbnail.
    mocks.resolve.mockImplementation((ref: { kind: string; name: string }) =>
      Promise.resolve({ durationSeconds: 2.5, kind: ref.kind, thumbnailUrl: `/thumbs/${ref.kind}/${ref.name}` })
    );

    // Video duration drives the same play badge as gallery tiles; images omit it.
    await expect(getThumbnailUrl({ kind: 'video', name: 'clip' })).resolves.toEqual({
      durationSeconds: 2.5,
      url: '/thumbs/video/clip',
    });
    await expect(getThumbnailUrl({ kind: 'image', name: 'clip' })).resolves.toEqual({
      durationSeconds: null,
      url: '/thumbs/image/clip',
    });
    expect(mocks.resolve).toHaveBeenCalledTimes(2);
  });

  it('remembers a definitive miss but retries a transient failure', async () => {
    // Cache missing/deleted items despite resolver throws so pointer dwell does not repeatedly fetch them.
    mocks.resolve.mockRejectedValue(new ApiError('gone', 404));

    await expect(getThumbnailUrl({ kind: 'image', name: 'gone.png' })).resolves.toBeNull();
    await expect(getThumbnailUrl({ kind: 'image', name: 'gone.png' })).resolves.toBeNull();
    expect(mocks.resolve).toHaveBeenCalledTimes(1);

    // A backend hiccup is not an answer about this item: caching it would
    // blank that point's preview for the rest of the session.
    mocks.resolve.mockRejectedValue(new ApiError('boom', 500));

    await expect(getThumbnailUrl({ kind: 'image', name: 'flaky.png' })).resolves.toBeNull();
    await expect(getThumbnailUrl({ kind: 'image', name: 'flaky.png' })).resolves.toBeNull();
    expect(mocks.resolve).toHaveBeenCalledTimes(3);
  });

  it('clears settled entries and stale in-flight results on account invalidation', async () => {
    mocks.resolve.mockResolvedValue({ kind: 'image', thumbnailUrl: '/thumbs/user-a.png' });
    await getThumbnailUrl({ kind: 'image', name: 'a.png' });

    // A request still in flight when the account switches must not seed the
    // next account's cache.
    let resolveLate: (item: { kind: string; thumbnailUrl: string }) => void = () => {};
    mocks.resolve.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveLate = resolve;
        })
    );
    const late = getThumbnailUrl({ kind: 'image', name: 'b.png' });

    accountLifecycle.invalidate();
    resolveLate({ kind: 'image', thumbnailUrl: '/thumbs/stale-b.png' });
    await late;

    accountLifecycle.activate('user-b');
    mocks.resolve.mockResolvedValue({ kind: 'image', thumbnailUrl: '/thumbs/user-b.png' });
    await expect(getThumbnailUrl({ kind: 'image', name: 'a.png' })).resolves.toEqual({
      durationSeconds: null,
      url: '/thumbs/user-b.png',
    });
    await expect(getThumbnailUrl({ kind: 'image', name: 'b.png' })).resolves.toEqual({
      durationSeconds: null,
      url: '/thumbs/user-b.png',
    });
  });
});

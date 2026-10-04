import type * as httpModule from '@platform/transport/http';

import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  apiFetchJson: vi.fn(),
}));

vi.mock('@platform/transport/http', async (importOriginal) => ({
  ...(await importOriginal<typeof httpModule>()),
  apiFetchJson: mocks.apiFetchJson,
}));

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';

import { clearImageLabels, getImageLabels } from './imageLabelCache';
import { imageMapStore, refreshImageIndexStatus, refreshImageMapPoints } from './imageMapStore';

describe('image map image-label cache', () => {
  beforeEach(() => {
    mocks.apiFetchJson.mockReset();
    // Each activation gets account-owned cache/cooldown state, cleared on account transition.
    accountLifecycle.invalidate();
    accountLifecycle.activate('user-a');
  });

  it('caches resolved labels per item', async () => {
    mocks.apiFetchJson.mockResolvedValue({ alternates: ['boat', 'harbor'], label: 'ship' });

    await expect(getImageLabels({ kind: 'image', name: 'a.png' })).resolves.toEqual({
      alternates: ['boat', 'harbor'],
      label: 'ship',
    });
    await expect(getImageLabels({ kind: 'image', name: 'a.png' })).resolves.toEqual({
      alternates: ['boat', 'harbor'],
      label: 'ship',
    });
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(1);
  });

  it('keeps labels for a video apart from an image with the same name', async () => {
    // The fixtures ship exactly this: a video named like an image. Keying the
    // cache by name alone would serve one item's tags for the other.
    mocks.apiFetchJson.mockResolvedValueOnce({ alternates: [], label: 'photo' });
    mocks.apiFetchJson.mockResolvedValueOnce({ alternates: [], label: 'clip' });

    await expect(getImageLabels({ kind: 'image', name: 'shared.png' })).resolves.toEqual({
      alternates: [],
      label: 'photo',
    });
    await expect(getImageLabels({ kind: 'video', name: 'shared.png' })).resolves.toEqual({
      alternates: [],
      label: 'clip',
    });
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(2);
  });

  it('asks for video labels in the video namespace', async () => {
    mocks.apiFetchJson.mockResolvedValue({ alternates: [], label: 'surf' });

    await getImageLabels({ kind: 'video', name: 'clip.mp4' });

    // Without the kind the backend would look the name up among images and
    // answer 404, which the cache would then remember as "no labels".
    const [url] = mocks.apiFetchJson.mock.calls[0] as [string];
    expect(url).toContain('image_name=clip.mp4');
    expect(url).toContain('kind=video');
  });

  it('caches a definitive per-item failure but retries transient ones', async () => {
    // 404: the image is simply not indexed; asking again cannot change that.
    mocks.apiFetchJson.mockRejectedValue(new ApiError('not indexed', 404));
    await expect(getImageLabels({ kind: 'image', name: 'a.png' })).resolves.toBeNull();
    await expect(getImageLabels({ kind: 'image', name: 'a.png' })).resolves.toBeNull();
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(1);

    // A network failure is retried on the next hover.
    mocks.apiFetchJson.mockRejectedValue(new Error('offline'));
    await expect(getImageLabels({ kind: 'image', name: 'b.png' })).resolves.toBeNull();
    mocks.apiFetchJson.mockResolvedValue({ alternates: [], label: 'ship' });
    await expect(getImageLabels({ kind: 'image', name: 'b.png' })).resolves.toEqual({ alternates: [], label: 'ship' });
  });

  it('backs a server error off instead of caching it as "this image has no labels"', async () => {
    vi.useFakeTimers();

    try {
      // Temporary outages must not cache empty tags permanently for hovered items.
      mocks.apiFetchJson.mockRejectedValue(new ApiError('bad gateway', 502));
      await expect(getImageLabels({ kind: 'image', name: 'a.png' })).resolves.toBeNull();
      await expect(getImageLabels({ kind: 'image', name: 'b.png' })).resolves.toBeNull();
      // ...but a deterministic 500 must not refire on every hover either.
      expect(mocks.apiFetchJson).toHaveBeenCalledTimes(1);

      vi.setSystemTime(Date.now() + 61_000);
      mocks.apiFetchJson.mockResolvedValue({ alternates: [], label: 'ship' });
      await expect(getImageLabels({ kind: 'image', name: 'a.png' })).resolves.toEqual({
        alternates: [],
        label: 'ship',
      });
    } finally {
      vi.useRealTimers();
    }
  });

  it('backs off a 409 for a cooldown, then tries again', async () => {
    vi.useFakeTimers();

    try {
      mocks.apiFetchJson.mockRejectedValue(new ApiError('still being prepared', 409));

      await expect(getImageLabels({ kind: 'image', name: 'a.png' })).resolves.toBeNull();
      // Server-wide, so sweeping the map must not fire one request per point.
      await expect(getImageLabels({ kind: 'image', name: 'b.png' })).resolves.toBeNull();
      expect(mocks.apiFetchJson).toHaveBeenCalledTimes(1);

      // The vocabulary is built lazily by the index worker: a 409 can simply
      // mean "not ready yet", so the cooldown must expire rather than latch —
      // and quickly, since a warm build lands within seconds.
      vi.setSystemTime(Date.now() + 5_001);
      mocks.apiFetchJson.mockResolvedValue({ alternates: [], label: 'ship' });
      await expect(getImageLabels({ kind: 'image', name: 'c.png' })).resolves.toEqual({
        alternates: [],
        label: 'ship',
      });
    } finally {
      vi.useRealTimers();
    }
  });

  it('backs off a server outage for a minute', async () => {
    vi.useFakeTimers();

    try {
      mocks.apiFetchJson.mockRejectedValue(new ApiError('boom', 500));
      await expect(getImageLabels({ kind: 'image', name: 'a.png' })).resolves.toBeNull();

      vi.setSystemTime(Date.now() + 30_000);
      await expect(getImageLabels({ kind: 'image', name: 'b.png' })).resolves.toBeNull();
      expect(mocks.apiFetchJson).toHaveBeenCalledTimes(1);

      vi.setSystemTime(Date.now() + 30_001);
      mocks.apiFetchJson.mockResolvedValue({ alternates: [], label: 'ship' });
      await expect(getImageLabels({ kind: 'image', name: 'c.png' })).resolves.toEqual({
        alternates: [],
        label: 'ship',
      });
    } finally {
      vi.useRealTimers();
    }
  });

  it('keeps serving labels it already has while a 409 cooldown is active', async () => {
    mocks.apiFetchJson.mockResolvedValue({ alternates: [], label: 'ship' });
    await getImageLabels({ kind: 'image', name: 'a.png' });

    mocks.apiFetchJson.mockRejectedValue(new ApiError('still being prepared', 409));
    await expect(getImageLabels({ kind: 'image', name: 'b.png' })).resolves.toBeNull();

    // The cooldown suppresses new requests, never cached results.
    await expect(getImageLabels({ kind: 'image', name: 'a.png' })).resolves.toEqual({ alternates: [], label: 'ship' });
  });

  it('drops cached labels when a vocabulary rebuild lands', async () => {
    // Vocabulary rebuild invalidates all answers and the cooldown waiting for that rebuild.
    mocks.apiFetchJson.mockResolvedValue({ alternates: [], label: 'ship' });
    await getImageLabels({ kind: 'image', name: 'a.png' });
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(1);

    clearImageLabels();

    mocks.apiFetchJson.mockResolvedValue({ alternates: [], label: 'sailboat' });
    await expect(getImageLabels({ kind: 'image', name: 'a.png' })).resolves.toEqual({
      alternates: [],
      label: 'sailboat',
    });
  });

  it('neither answers nor caches a request that straddles a vocabulary rebuild', async () => {
    let resolveOld: (labels: { alternates: string[]; label: string }) => void = () => {};
    mocks.apiFetchJson.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveOld = resolve;
        })
    );
    const stale = getImageLabels({ kind: 'image', name: 'a.png' });

    clearImageLabels();
    // The rebuild releases the old claim, so a new reveal asks again.
    mocks.apiFetchJson.mockResolvedValueOnce({ alternates: [], label: 'sailboat' });
    const fresh = getImageLabels({ kind: 'image', name: 'a.png' });
    resolveOld({ alternates: [], label: 'ship' });

    await expect(stale).resolves.toBeNull();
    await expect(fresh).resolves.toEqual({ alternates: [], label: 'sailboat' });
    await expect(getImageLabels({ kind: 'image', name: 'a.png' })).resolves.toEqual({
      alternates: [],
      label: 'sailboat',
    });
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(2);
  });

  it.each(['points', 'status'] as const)(
    'discards cached and in-flight labels when %s reports a missing encoder',
    async (source) => {
      const cachedItem = { kind: 'image', name: 'cached.png' } as const;
      const pendingItem = { kind: 'image', name: 'pending.png' } as const;
      mocks.apiFetchJson.mockResolvedValueOnce({ alternates: [], label: 'old cached label' });
      await getImageLabels(cachedItem);

      let resolveOld: (labels: { alternates: string[]; label: string }) => void = () => {};
      mocks.apiFetchJson.mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveOld = resolve;
          })
      );
      const stale = getImageLabels(pendingItem);
      const missingResponse =
        source === 'points'
          ? { model_name: 'encoder', point_count: 0, points: [], stale: false, state: 'model_missing' }
          : { enabled: false, index: null, model_name: 'encoder', projection: { state: 'model_missing' } };
      mocks.apiFetchJson.mockResolvedValueOnce(missingResponse);

      if (source === 'points') {
        await refreshImageMapPoints();
      } else {
        refreshImageIndexStatus();
        await vi.waitFor(() => expect(imageMapStore.getSnapshot().data?.state).toBe('model_missing'));
      }

      mocks.apiFetchJson.mockResolvedValue({ alternates: [], label: 'fresh label' });
      const fresh = getImageLabels(pendingItem);
      resolveOld({ alternates: [], label: 'retired label' });
      await expect(stale).resolves.toBeNull();
      await expect(fresh).resolves.toEqual({ alternates: [], label: 'fresh label' });
      await expect(getImageLabels(cachedItem)).resolves.toEqual({ alternates: [], label: 'fresh label' });

      // Progress and repeated missing-state snapshots must not repeatedly clear the cache.
      imageMapStore.patchSnapshot({ indexUpdatedAt: 1_000 });
      await expect(getImageLabels(pendingItem)).resolves.toEqual({ alternates: [], label: 'fresh label' });
      expect(mocks.apiFetchJson).toHaveBeenCalledTimes(5);
    }
  );

  it.each(['points', 'status'] as const)(
    'invalidates old labels when %s observes a different encoder without a missing state',
    async (source) => {
      const response = {
        model_id: 'encoder-a',
        point_count: 0,
        points: [],
        stale: false,
        state: 'empty',
        updated_at: null,
      };
      mocks.apiFetchJson.mockResolvedValueOnce(response);
      await refreshImageMapPoints();

      const cachedItem = { kind: 'image', name: 'cached.png' } as const;
      const pendingItem = { kind: 'image', name: 'pending.png' } as const;
      mocks.apiFetchJson.mockResolvedValueOnce({ alternates: [], label: 'encoder A label' });
      await getImageLabels(cachedItem);
      let resolveOld: (value: { alternates: string[]; label: string }) => void = () => {};
      mocks.apiFetchJson.mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveOld = resolve;
          })
      );
      const stale = getImageLabels(pendingItem);

      // Removal and reinstallation happened while the map was closed. Neither
      // endpoint needs to return model_missing for the cached labels to be stale.
      const replacement = { ...response, model_id: 'encoder-b' };
      const labelRequest = vi.fn().mockResolvedValue({ alternates: [], label: 'encoder B label' });
      mocks.apiFetchJson.mockImplementation((url: string) => {
        if (url.startsWith('/api/v1/image_map/status')) {
          return Promise.resolve({ enabled: true, model_id: 'encoder-b', projection: { state: 'empty' } });
        }
        if (url.startsWith('/api/v1/image_map/points')) {
          return Promise.resolve(replacement);
        }
        return labelRequest();
      });
      if (source === 'points') {
        await refreshImageMapPoints();
      } else {
        refreshImageIndexStatus();
        await vi.waitFor(() => {
          expect(imageMapStore.getSnapshot().data?.modelId).toBe('encoder-b');
          expect(imageMapStore.getSnapshot().loadState).toBe('loaded');
        });
      }

      resolveOld({ alternates: [], label: 'late encoder A label' });
      await expect(stale).resolves.toBeNull();
      await expect(getImageLabels(cachedItem)).resolves.toEqual({ alternates: [], label: 'encoder B label' });
      await expect(getImageLabels(pendingItem)).resolves.toEqual({ alternates: [], label: 'encoder B label' });
      expect(labelRequest).toHaveBeenCalledTimes(2);

      // Normal progress/projection refreshes from the same encoder keep cached labels.
      await refreshImageMapPoints();
      await getImageLabels(cachedItem);
      expect(labelRequest).toHaveBeenCalledTimes(2);
    }
  );

  it('clears the cooldown on account switch', async () => {
    mocks.apiFetchJson.mockRejectedValue(new ApiError('index disabled', 409));
    await expect(getImageLabels({ kind: 'image', name: 'a.png' })).resolves.toBeNull();

    accountLifecycle.invalidate();
    accountLifecycle.activate('user-b');
    mocks.apiFetchJson.mockResolvedValue({ alternates: [], label: 'ship' });
    await expect(getImageLabels({ kind: 'image', name: 'a.png' })).resolves.toEqual({ alternates: [], label: 'ship' });
  });

  it('clears settled entries and stale in-flight results on account invalidation', async () => {
    mocks.apiFetchJson.mockResolvedValue({ alternates: [], label: 'user-a-label' });
    await getImageLabels({ kind: 'image', name: 'a.png' });

    // A request still in flight when the account switches must not seed the
    // next account's cache.
    let resolveLate: (labels: { alternates: string[]; label: string }) => void = () => {};
    mocks.apiFetchJson.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveLate = resolve;
        })
    );
    const late = getImageLabels({ kind: 'image', name: 'b.png' });

    accountLifecycle.invalidate();
    resolveLate({ alternates: [], label: 'stale-b-label' });
    await late;

    accountLifecycle.activate('user-b');
    mocks.apiFetchJson.mockResolvedValue({ alternates: [], label: 'user-b-label' });
    await expect(getImageLabels({ kind: 'image', name: 'a.png' })).resolves.toEqual({
      alternates: [],
      label: 'user-b-label',
    });
    await expect(getImageLabels({ kind: 'image', name: 'b.png' })).resolves.toEqual({
      alternates: [],
      label: 'user-b-label',
    });
  });
});

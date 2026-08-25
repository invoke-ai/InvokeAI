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

describe('image map image-label cache', () => {
  beforeEach(() => {
    mocks.apiFetchJson.mockReset();
    // Each activation owns a fresh cache: the previous test's entries (and
    // any 409 latch) are account state and go with the account.
    accountLifecycle.invalidate();
    accountLifecycle.activate('user-a');
  });

  it('caches resolved labels per image name', async () => {
    mocks.apiFetchJson.mockResolvedValue({ alternates: ['boat', 'harbor'], label: 'ship' });

    await expect(getImageLabels('a.png')).resolves.toEqual({ alternates: ['boat', 'harbor'], label: 'ship' });
    await expect(getImageLabels('a.png')).resolves.toEqual({ alternates: ['boat', 'harbor'], label: 'ship' });
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(1);
  });

  it('caches a definitive per-image failure but retries transient ones', async () => {
    // 404: the image is simply not indexed; asking again cannot change that.
    mocks.apiFetchJson.mockRejectedValue(new ApiError('not indexed', 404));
    await expect(getImageLabels('a.png')).resolves.toBeNull();
    await expect(getImageLabels('a.png')).resolves.toBeNull();
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(1);

    // A network failure is retried on the next hover.
    mocks.apiFetchJson.mockRejectedValue(new Error('offline'));
    await expect(getImageLabels('b.png')).resolves.toBeNull();
    mocks.apiFetchJson.mockResolvedValue({ alternates: [], label: 'ship' });
    await expect(getImageLabels('b.png')).resolves.toEqual({ alternates: [], label: 'ship' });
  });

  it('backs a server error off instead of caching it as "this image has no labels"', async () => {
    vi.useFakeTimers();

    try {
      // A backend or proxy restart mid-sweep must not permanently blank the
      // tags of every image the pointer crossed during the outage...
      mocks.apiFetchJson.mockRejectedValue(new ApiError('bad gateway', 502));
      await expect(getImageLabels('a.png')).resolves.toBeNull();
      await expect(getImageLabels('b.png')).resolves.toBeNull();
      // ...but a deterministic 500 must not refire on every hover either.
      expect(mocks.apiFetchJson).toHaveBeenCalledTimes(1);

      vi.setSystemTime(Date.now() + 61_000);
      mocks.apiFetchJson.mockResolvedValue({ alternates: [], label: 'ship' });
      await expect(getImageLabels('a.png')).resolves.toEqual({ alternates: [], label: 'ship' });
    } finally {
      vi.useRealTimers();
    }
  });

  it('backs off a 409 for a cooldown, then tries again', async () => {
    vi.useFakeTimers();

    try {
      mocks.apiFetchJson.mockRejectedValue(new ApiError('still being prepared', 409));

      await expect(getImageLabels('a.png')).resolves.toBeNull();
      // Server-wide, so sweeping the map must not fire one request per point.
      await expect(getImageLabels('b.png')).resolves.toBeNull();
      expect(mocks.apiFetchJson).toHaveBeenCalledTimes(1);

      // The vocabulary is built lazily by the index worker: a 409 can simply
      // mean "not ready yet", so the cooldown must expire rather than latch.
      vi.setSystemTime(Date.now() + 61_000);
      mocks.apiFetchJson.mockResolvedValue({ alternates: [], label: 'ship' });
      await expect(getImageLabels('c.png')).resolves.toEqual({ alternates: [], label: 'ship' });
    } finally {
      vi.useRealTimers();
    }
  });

  it('keeps serving labels it already has while a 409 cooldown is active', async () => {
    mocks.apiFetchJson.mockResolvedValue({ alternates: [], label: 'ship' });
    await getImageLabels('a.png');

    mocks.apiFetchJson.mockRejectedValue(new ApiError('still being prepared', 409));
    await expect(getImageLabels('b.png')).resolves.toBeNull();

    // The cooldown suppresses new requests, never cached results.
    await expect(getImageLabels('a.png')).resolves.toEqual({ alternates: [], label: 'ship' });
  });

  it('drops cached labels when a vocabulary rebuild lands', async () => {
    // The phrases these are scored against are admin-editable, so a rebuild
    // makes every cached answer stale — including a cooldown that was only
    // ever waiting for that rebuild.
    mocks.apiFetchJson.mockResolvedValue({ alternates: [], label: 'ship' });
    await getImageLabels('a.png');
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(1);

    clearImageLabels();

    mocks.apiFetchJson.mockResolvedValue({ alternates: [], label: 'sailboat' });
    await expect(getImageLabels('a.png')).resolves.toEqual({ alternates: [], label: 'sailboat' });
  });

  it('clears the cooldown on account switch', async () => {
    mocks.apiFetchJson.mockRejectedValue(new ApiError('index disabled', 409));
    await expect(getImageLabels('a.png')).resolves.toBeNull();

    accountLifecycle.invalidate();
    accountLifecycle.activate('user-b');
    mocks.apiFetchJson.mockResolvedValue({ alternates: [], label: 'ship' });
    await expect(getImageLabels('a.png')).resolves.toEqual({ alternates: [], label: 'ship' });
  });

  it('clears settled entries and stale in-flight results on account invalidation', async () => {
    mocks.apiFetchJson.mockResolvedValue({ alternates: [], label: 'user-a-label' });
    await getImageLabels('a.png');

    // A request still in flight when the account switches must not seed the
    // next account's cache.
    let resolveLate: (labels: { alternates: string[]; label: string }) => void = () => {};
    mocks.apiFetchJson.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveLate = resolve;
        })
    );
    const late = getImageLabels('b.png');

    accountLifecycle.invalidate();
    resolveLate({ alternates: [], label: 'stale-b-label' });
    await late;

    accountLifecycle.activate('user-b');
    mocks.apiFetchJson.mockResolvedValue({ alternates: [], label: 'user-b-label' });
    await expect(getImageLabels('a.png')).resolves.toEqual({ alternates: [], label: 'user-b-label' });
    await expect(getImageLabels('b.png')).resolves.toEqual({ alternates: [], label: 'user-b-label' });
  });
});

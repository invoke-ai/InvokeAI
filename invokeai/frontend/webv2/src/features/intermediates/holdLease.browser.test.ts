import type { AccountScope } from '@platform/state/accountLifecycle';

import { apiFetch } from '@platform/transport/http';
import { afterEach, expect, it, vi } from 'vitest';

import type { HeldMediaNames } from './holdLease';

import { MAX_HOLD_NAMES_PER_KIND, startIntermediatesHoldLease } from './holdLease';

vi.mock('@platform/transport/http', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  apiFetch: vi.fn().mockResolvedValue(undefined),
  getHttpAuthToken: vi.fn(() => 'lease-token'),
}));

const fetchMock = vi.mocked(apiFetch);

const ownerWith = (signal: AbortSignal): AccountScope => ({ accountId: 'alice', epoch: 1, signal, storageSuffix: '' });

const startWith = (initial: HeldMediaNames, signal = new AbortController().signal) => {
  let held = initial;
  let notify: () => void = () => undefined;
  const stop = startIntermediatesHoldLease({
    owner: ownerWith(signal),
    read: () => held,
    subscribe: (onChange) => {
      notify = onChange;
      return () => undefined;
    },
  });
  return {
    stop,
    update: (next: HeldMediaNames) => {
      held = next;
      notify();
    },
  };
};

const calls = (method: string) =>
  fetchMock.mock.calls.filter(([, init]) => init?.method === method).map(([url, init]) => ({ init, url: String(url) }));

const settle = () =>
  new Promise((resolve) => {
    setTimeout(resolve, 400);
  });

afterEach(() => {
  fetchMock.mockReset();
  fetchMock.mockResolvedValue(undefined as never);
});

it('holds up to the per-kind limit of sorted unique names under one lease', async () => {
  const images = Array.from({ length: MAX_HOLD_NAMES_PER_KIND + 1 }, (_, index) => `image-${index}`).reverse();
  const lease = startWith({ images: [...images, 'image-0'], videos: ['second.mp4', 'first.mp4'] });
  try {
    await vi.waitFor(() => expect(calls('PUT')).toHaveLength(1));
    const body = JSON.parse(String(calls('PUT')[0]!.init?.body)) as { images: string[]; videos: string[] };
    expect(body.images).toHaveLength(MAX_HOLD_NAMES_PER_KIND);
    expect(body.images).toEqual([...body.images].sort());
    expect(body.videos).toEqual(['first.mp4', 'second.mp4']);
    const leaseId = decodeURIComponent(calls('PUT')[0]!.url.split('/').at(-1)!);
    expect(leaseId).toMatch(/^[A-Za-z0-9_-]+$/);
  } finally {
    lease.stop();
  }
});

it('replaces the lease in place and releases it once nothing is held', async () => {
  const lease = startWith({ images: ['first.png'], videos: [] });
  try {
    await vi.waitFor(() =>
      expect(calls('PUT').map(({ init }) => init?.body)).toContain(
        JSON.stringify({ images: ['first.png'], videos: [] })
      )
    );
    lease.update({ images: ['second.png'], videos: [] });
    await vi.waitFor(() =>
      expect(calls('PUT').map(({ init }) => init?.body)).toContain(
        JSON.stringify({ images: ['second.png'], videos: [] })
      )
    );
    expect(new Set(calls('PUT').map(({ url }) => url)).size).toBe(1);
    expect(calls('DELETE')).toHaveLength(0);
    lease.update({ images: [], videos: [] });
    await vi.waitFor(() => expect(calls('DELETE')).toHaveLength(1));
    expect(calls('DELETE')[0]!.url).toBe(calls('PUT')[0]!.url);
  } finally {
    lease.stop();
  }
});

it('releases a lease that lands after it was disposed', async () => {
  let resolvePut!: () => void;
  fetchMock.mockImplementationOnce(
    () =>
      new Promise<never>((resolve) => {
        resolvePut = () => resolve(undefined as never);
      })
  );
  const lease = startWith({ images: ['late.png'], videos: [] });
  await vi.waitFor(() => expect(calls('PUT')).toHaveLength(1));

  lease.stop();
  expect(calls('DELETE')).toHaveLength(0);
  resolvePut();

  await vi.waitFor(() => expect(calls('DELETE')).toHaveLength(1));
  expect(calls('DELETE')[0]!.url).toBe(calls('PUT')[0]!.url);
  expect(calls('DELETE')[0]!.init).toMatchObject({ keepalive: true });
});

it('releases its lease with keepalive when disposed', async () => {
  const lease = startWith({ images: ['held.png'], videos: ['held.mp4'] });
  await vi.waitFor(() => expect(calls('PUT')).toHaveLength(1));
  const leaseUrl = calls('PUT')[0]!.url;

  lease.stop();

  expect(calls('DELETE')).toEqual([expect.objectContaining({ url: leaseUrl })]);
  expect(calls('DELETE')[0]!.init).toMatchObject({ keepalive: true });
});

it('releases as the account that took the lease after that account signed out', async () => {
  const session = new AbortController();
  const lease = startWith({ images: ['held.png'], videos: [] }, session.signal);
  await vi.waitFor(() => expect(calls('PUT')).toHaveLength(1));

  session.abort();
  lease.stop();

  const [release] = calls('DELETE');
  expect(release).toBeDefined();
  expect(release!.init?.signal).toBeUndefined();
  expect(new Headers(release!.init?.headers).get('Authorization')).toBe('Bearer lease-token');
});

it('releases on pagehide and holds again when the page is restored', async () => {
  const lease = startWith({ images: ['held.png'], videos: [] });
  try {
    await vi.waitFor(() => expect(calls('PUT')).toHaveLength(1));

    window.dispatchEvent(new PageTransitionEvent('pagehide', { persisted: true }));
    expect(calls('DELETE')).toHaveLength(1);
    expect(calls('DELETE')[0]!.init).toMatchObject({ keepalive: true });

    window.dispatchEvent(new PageTransitionEvent('pageshow', { persisted: true }));
    await vi.waitFor(() => expect(calls('PUT')).toHaveLength(2));
  } finally {
    lease.stop();
  }
});

it('refreshes an unchanged hold on becoming visible only after missing a heartbeat', async () => {
  const lease = startWith({ images: ['held.png'], videos: [] });
  const startedAt = Date.now();
  const clock = vi.spyOn(Date, 'now');
  try {
    await vi.waitFor(() => expect(calls('PUT')).toHaveLength(1));
    await settle();
    expect(document.visibilityState).toBe('visible');
    clock.mockReturnValue(startedAt + 2 * 60_000);
    document.dispatchEvent(new Event('visibilitychange'));
    document.dispatchEvent(new Event('visibilitychange'));
    await settle();
    expect(calls('PUT')).toHaveLength(1);

    clock.mockReturnValue(startedAt + 6 * 60_000);
    document.dispatchEvent(new Event('visibilitychange'));
    await vi.waitFor(() => expect(calls('PUT')).toHaveLength(2));
    expect(calls('PUT')[1]!.url).toBe(calls('PUT')[0]!.url);
  } finally {
    clock.mockRestore();
    lease.stop();
  }
});

it('skips unchanged sets and sends again after a failed send', async () => {
  const lease = startWith({ images: ['b.png', 'a.png'], videos: [] });
  try {
    await vi.waitFor(() => expect(calls('PUT')).toHaveLength(1));
    lease.update({ images: ['a.png', 'b.png'], videos: [] });
    await settle();
    expect(calls('PUT')).toHaveLength(1);

    fetchMock.mockRejectedValueOnce(new Error('temporary network failure'));
    lease.update({ images: ['a.png', 'b.png', 'c.png'], videos: [] });
    await vi.waitFor(() => expect(calls('PUT')).toHaveLength(2));
    await settle();
    // The server still holds the previous lease; the next foreground return sends the new set again.
    document.dispatchEvent(new Event('visibilitychange'));
    await vi.waitFor(() => expect(calls('PUT')).toHaveLength(3));
    expect(calls('PUT')[2]!.init?.body).toBe(JSON.stringify({ images: ['a.png', 'b.png', 'c.png'], videos: [] }));
    expect(new Set(calls('PUT').map(({ url }) => url)).size).toBe(1);
    expect(calls('DELETE')).toHaveLength(0);
  } finally {
    lease.stop();
  }
});

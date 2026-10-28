import { accountLifecycle } from '@platform/state/accountLifecycle';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  getGalleryImageRecoveryRevision,
  reportGalleryImageFailure,
  subscribeGalleryImageRecovery,
} from './imageRecovery';

const SECOND = 1_000;

class FakeDocument extends EventTarget {
  visibilityState: DocumentVisibilityState = 'visible';

  show(): void {
    this.visibilityState = 'visible';
    this.dispatchEvent(new Event('visibilitychange'));
  }
}

let fakeDocument: FakeDocument;
/** What the server answers every image read with right now. */
let serverStatus: number;
const fetchMock = vi.fn<(url: string, init?: RequestInit) => Promise<Response>>();
const requestedUrls = () => fetchMock.mock.calls.map(([url]) => url);

const answer = (status: number, contentType = 'image/webp') =>
  new Response(status === 200 ? 'image' : null, { headers: { 'content-type': contentType }, status });
const thumbnail = (index: number) => `/api/v1/images/i/image-${String(index)}.png/thumbnail`;

describe('Gallery image recovery probe', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    fakeDocument = new FakeDocument();
    serverStatus = 409;
    fetchMock.mockReset().mockImplementation(() => Promise.resolve(answer(serverStatus)));
    vi.stubGlobal('document', fakeDocument);
    vi.stubGlobal('fetch', fetchMock);
    // A fresh account lifetime per test: it also drops whatever probe the previous test left.
    accountLifecycle.activate('image-recovery-user');
  });

  afterEach(() => {
    accountLifecycle.invalidate();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('asks about one failed image per interval for as long as a move runs, then reloads every failed image once', async () => {
    const revision = getGalleryImageRecoveryRevision();
    const listener = vi.fn();
    const unsubscribe = subscribeGalleryImageRecovery(listener);

    // A screen of tiles fails at once, some of them twice.
    for (let index = 0; index < 40; index += 1) {
      reportGalleryImageFailure(thumbnail(index), 0);
    }
    reportGalleryImageFailure(thumbnail(3), 0);

    await vi.advanceTimersByTimeAsync(2 * SECOND);
    expect(requestedUrls()).toEqual([thumbnail(0)]);

    // 409: asked again 15 s later, then every 30 s, for as long as the move runs.
    await vi.advanceTimersByTimeAsync(15 * SECOND - 1);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    for (let probe = 3; probe <= 22; probe += 1) {
      await vi.advanceTimersByTimeAsync(30 * SECOND - 1);
      expect(fetchMock).toHaveBeenCalledTimes(probe - 1);
      await vi.advanceTimersByTimeAsync(1);
      expect(fetchMock).toHaveBeenCalledTimes(probe);
    }
    expect(new Set(requestedUrls())).toEqual(new Set([thumbnail(0)]));
    expect(getGalleryImageRecoveryRevision()).toBe(revision);
    // The probe reads the status itself, uncached, under the account's lifetime.
    expect(fetchMock.mock.calls[0]?.[1]).toMatchObject({ cache: 'no-store' });

    // The move ends: the next probe sees the image served and every failed image reloads, once.
    serverStatus = 200;
    await vi.advanceTimersByTimeAsync(30 * SECOND);
    expect(fetchMock).toHaveBeenCalledTimes(23);
    expect(getGalleryImageRecoveryRevision()).toBe(revision + 1);
    expect(listener).toHaveBeenCalledOnce();

    await vi.advanceTimersByTimeAsync(10 * 60 * SECOND);
    expect(fetchMock).toHaveBeenCalledTimes(23);
    expect(vi.getTimerCount()).toBe(0);
    unsubscribe();
  });

  it('stops probing an image the server says is missing', async () => {
    const revision = getGalleryImageRecoveryRevision();
    serverStatus = 404;
    reportGalleryImageFailure(thumbnail(1), 0);

    await vi.advanceTimersByTimeAsync(2 * SECOND);
    expect(requestedUrls()).toEqual([thumbnail(1)]);

    await vi.advanceTimersByTimeAsync(10 * 60 * SECOND);
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(vi.getTimerCount()).toBe(0);
    expect(getGalleryImageRecoveryRevision()).toBe(revision);
  });

  it('drops a missing image and goes on to the next failed one', async () => {
    fetchMock.mockImplementation((url) => Promise.resolve(answer(url === thumbnail(1) ? 404 : 409)));
    reportGalleryImageFailure(thumbnail(1), 0);
    reportGalleryImageFailure(thumbnail(2), 0);

    await vi.advanceTimersByTimeAsync(2 * SECOND);
    await vi.advanceTimersByTimeAsync(2 * SECOND);
    expect(requestedUrls()).toEqual([thumbnail(1), thumbnail(2)]);

    await vi.advanceTimersByTimeAsync(15 * SECOND);
    expect(requestedUrls()).toEqual([thumbnail(1), thumbnail(2), thumbnail(2)]);
  });

  it('does not take a page served in place of the image for the image', async () => {
    const revision = getGalleryImageRecoveryRevision();
    fetchMock.mockImplementation(() => Promise.resolve(answer(200, 'text/html')));
    reportGalleryImageFailure(thumbnail(1), 0);

    await vi.advanceTimersByTimeAsync(10 * 60 * SECOND);
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(getGalleryImageRecoveryRevision()).toBe(revision);
  });

  it('does not reload again an image that still fails while the server serves it', async () => {
    const revision = getGalleryImageRecoveryRevision();
    serverStatus = 200;
    reportGalleryImageFailure(thumbnail(1), 0);
    await vi.advanceTimersByTimeAsync(2 * SECOND);
    expect(getGalleryImageRecoveryRevision()).toBe(revision + 1);

    // The reloaded image fails again (a broken file, say) although the server answers 200 for it.
    reportGalleryImageFailure(thumbnail(1), revision + 1);
    await vi.advanceTimersByTimeAsync(2 * SECOND);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(getGalleryImageRecoveryRevision()).toBe(revision + 1);

    await vi.advanceTimersByTimeAsync(10 * 60 * SECOND);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('recovers a reloaded image again when a later move refuses it', async () => {
    const revision = getGalleryImageRecoveryRevision();
    serverStatus = 200;
    reportGalleryImageFailure(thumbnail(1), 0);
    await vi.advanceTimersByTimeAsync(2 * SECOND);
    expect(getGalleryImageRecoveryRevision()).toBe(revision + 1);

    // A second move refuses the reloaded image; it recovers when that move ends too.
    serverStatus = 409;
    reportGalleryImageFailure(thumbnail(1), revision + 1);
    await vi.advanceTimersByTimeAsync(2 * SECOND);
    serverStatus = 200;
    await vi.advanceTimersByTimeAsync(15 * SECOND);
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(getGalleryImageRecoveryRevision()).toBe(revision + 2);
  });

  it('keeps waiting through a server restart mid-move, then reloads every failed image once', async () => {
    const revision = getGalleryImageRecoveryRevision();
    const listener = vi.fn();
    const unsubscribe = subscribeGalleryImageRecovery(listener);
    // The server goes down mid-move (connections refused), comes back unavailable, resumes the move, then serves.
    const answers: (() => Promise<Response>)[] = [
      () => Promise.reject(new TypeError('Failed to fetch')),
      () => Promise.reject(new TypeError('Failed to fetch')),
      () => Promise.reject(new TypeError('Failed to fetch')),
      () => Promise.resolve(answer(503)),
      () => Promise.resolve(answer(502)),
      () => Promise.resolve(answer(409)),
      () => Promise.resolve(answer(200)),
    ];
    fetchMock.mockImplementation(() => answers[fetchMock.mock.calls.length - 1]!());
    for (let index = 0; index < 3; index += 1) {
      reportGalleryImageFailure(thumbnail(index), 0);
    }

    await vi.advanceTimersByTimeAsync(10 * 60 * SECOND);

    expect(fetchMock).toHaveBeenCalledTimes(answers.length);
    // No answer drained the failed images: the one that kept being asked about recovered them all.
    expect(new Set(requestedUrls())).toEqual(new Set([thumbnail(0)]));
    expect(getGalleryImageRecoveryRevision()).toBe(revision + 1);
    expect(listener).toHaveBeenCalledOnce();
    expect(vi.getTimerCount()).toBe(0);
    unsubscribe();
  });

  it('recovers an image that fails on its plain URL during a later short move', async () => {
    const revision = getGalleryImageRecoveryRevision();
    serverStatus = 200;
    reportGalleryImageFailure(thumbnail(1), 0);
    await vi.advanceTimersByTimeAsync(2 * SECOND);
    expect(getGalleryImageRecoveryRevision()).toBe(revision + 1);

    // A short move runs; a tile mounted during it fails on the plain URL, and the move is over by the first probe.
    reportGalleryImageFailure(thumbnail(1), 0);
    await vi.advanceTimersByTimeAsync(2 * SECOND);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(getGalleryImageRecoveryRevision()).toBe(revision + 2);
  });

  it('does not ask again after the account change aborts its request', async () => {
    fetchMock.mockImplementation(
      (_url, init) =>
        new Promise((_resolve, reject) => {
          init?.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
        })
    );
    reportGalleryImageFailure(thumbnail(1), 0);
    await vi.advanceTimersByTimeAsync(2 * SECOND);
    expect(fetchMock).toHaveBeenCalledOnce();

    accountLifecycle.activate('image-recovery-other-user');
    await vi.advanceTimersByTimeAsync(10 * 60 * SECOND);

    expect(fetchMock).toHaveBeenCalledOnce();
    expect(vi.getTimerCount()).toBe(0);
  });

  it('pauses while the tab is hidden and asks as soon as it is shown', async () => {
    fakeDocument.visibilityState = 'hidden';
    reportGalleryImageFailure(thumbnail(1), 0);

    await vi.advanceTimersByTimeAsync(10 * 60 * SECOND);
    expect(fetchMock).not.toHaveBeenCalled();
    expect(vi.getTimerCount()).toBe(0);

    fakeDocument.show();
    await vi.advanceTimersByTimeAsync(0);
    expect(requestedUrls()).toEqual([thumbnail(1)]);
  });

  it('stops for good when the account changes, and forgets the old account’s images', async () => {
    const revision = getGalleryImageRecoveryRevision();
    reportGalleryImageFailure(thumbnail(1), 0);
    await vi.advanceTimersByTimeAsync(2 * SECOND);
    expect(fetchMock).toHaveBeenCalledOnce();
    const firstSignal = fetchMock.mock.calls[0]?.[1]?.signal;

    accountLifecycle.activate('image-recovery-other-user');
    expect(firstSignal?.aborted).toBe(true);
    serverStatus = 200;
    await vi.advanceTimersByTimeAsync(10 * 60 * SECOND);
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(vi.getTimerCount()).toBe(0);
    expect(getGalleryImageRecoveryRevision()).toBe(revision);

    // A failure in the new account starts its own probe, which never asks about the old account's image.
    reportGalleryImageFailure(thumbnail(2), 0);
    await vi.advanceTimersByTimeAsync(2 * SECOND);
    expect(requestedUrls()).toEqual([thumbnail(1), thumbnail(2)]);
    expect(getGalleryImageRecoveryRevision()).toBe(revision + 1);
  });

  it('stops a probe whose request is out when the account changes', async () => {
    const revision = getGalleryImageRecoveryRevision();
    let resolveAnswer!: (response: Response) => void;
    fetchMock.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveAnswer = resolve;
        })
    );
    reportGalleryImageFailure(thumbnail(1), 0);
    await vi.advanceTimersByTimeAsync(2 * SECOND);

    accountLifecycle.activate('image-recovery-other-user');
    resolveAnswer(answer(200));
    await vi.advanceTimersByTimeAsync(10 * 60 * SECOND);

    expect(fetchMock).toHaveBeenCalledOnce();
    expect(getGalleryImageRecoveryRevision()).toBe(revision);
  });

  it('ignores failures reported for an account lifetime that has ended', async () => {
    const staleOwner = accountLifecycle.capture();
    accountLifecycle.activate('image-recovery-other-user');

    reportGalleryImageFailure(thumbnail(1), 0, staleOwner);
    await vi.advanceTimersByTimeAsync(10 * 60 * SECOND);

    expect(fetchMock).not.toHaveBeenCalled();
  });
});

import { describe, expect, it, vi } from 'vitest';

import { createStagedPreviewBlobCache } from './stagedPreviewBlobCache';

const deferred = <T>() => {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((nextResolve, nextReject) => {
    resolve = nextResolve;
    reject = nextReject;
  });
  return { promise, reject, resolve };
};

describe('staged preview blob cache', () => {
  it('reuses bytes fetched by a completed background preload', async () => {
    const resolveImage = vi.fn(() => Promise.resolve(new Blob(['pixels'])));
    const cache = createStagedPreviewBlobCache(resolveImage);

    cache.preload('candidate.png');
    await vi.waitFor(() => expect(resolveImage).toHaveBeenCalledOnce());
    await expect(cache.get('candidate.png')).resolves.toBeInstanceOf(Blob);

    expect(resolveImage).toHaveBeenCalledOnce();
    cache.dispose();
  });

  it('promotes a queued candidate when it is selected', async () => {
    const first = deferred<Blob>();
    const resolveImage = vi.fn((imageName: string) =>
      imageName === 'first.png' ? first.promise : Promise.resolve(new Blob([imageName]))
    );
    const cache = createStagedPreviewBlobCache(resolveImage, { maxConcurrentRequests: 1 });

    cache.preload('first.png');
    cache.preload('selected.png');
    expect(resolveImage).toHaveBeenCalledTimes(1);

    await expect(cache.get('selected.png')).resolves.toBeInstanceOf(Blob);
    expect(resolveImage).toHaveBeenNthCalledWith(2, 'selected.png', expect.any(AbortSignal));

    first.resolve(new Blob(['first']));
    await first.promise;
    cache.dispose();
  });

  it('drops failed preloads so selecting the candidate retries', async () => {
    const resolveImage = vi
      .fn<(imageName: string, signal?: AbortSignal) => Promise<Blob>>()
      .mockRejectedValueOnce(new Error('temporary failure'))
      .mockResolvedValueOnce(new Blob(['retry']));
    const cache = createStagedPreviewBlobCache(resolveImage);

    cache.preload('candidate.png');
    await vi.waitFor(() => expect(resolveImage).toHaveBeenCalledOnce());
    await vi.waitFor(() => expect(resolveImage.mock.results[0]?.type).toBe('return'));

    await expect(cache.get('candidate.png')).resolves.toBeInstanceOf(Blob);
    expect(resolveImage).toHaveBeenCalledTimes(2);
    cache.dispose();
  });

  it('retries once when an in-flight prefetch fails after the candidate is selected', async () => {
    const prefetched = deferred<Blob>();
    const resolveImage = vi
      .fn<(imageName: string, signal?: AbortSignal) => Promise<Blob>>()
      .mockReturnValueOnce(prefetched.promise)
      .mockResolvedValueOnce(new Blob(['foreground retry']));
    const cache = createStagedPreviewBlobCache(resolveImage);

    cache.preload('candidate.png');
    const selected = cache.get('candidate.png');
    prefetched.reject(new Error('prefetch failed'));

    await expect(selected).resolves.toBeInstanceOf(Blob);
    expect(resolveImage).toHaveBeenCalledTimes(2);
    cache.dispose();
  });

  it('retries a synchronously failing prefetch when the candidate is already selected', async () => {
    const resolveImage = vi
      .fn<(imageName: string, signal?: AbortSignal) => Promise<Blob>>()
      .mockImplementationOnce(() => {
        throw new Error('synchronous prefetch failure');
      })
      .mockResolvedValueOnce(new Blob(['foreground retry']));
    const cache = createStagedPreviewBlobCache(resolveImage);

    cache.preload('candidate.png');

    await expect(cache.get('candidate.png')).resolves.toBeInstanceOf(Blob);
    expect(resolveImage).toHaveBeenCalledTimes(2);
    cache.dispose();
  });

  it('retries once when the latest foreground demand fails', async () => {
    const resolveImage = vi
      .fn<(imageName: string, signal?: AbortSignal) => Promise<Blob>>()
      .mockRejectedValueOnce(new Error('temporary foreground failure'))
      .mockResolvedValueOnce(new Blob(['foreground retry']));
    const cache = createStagedPreviewBlobCache(resolveImage);

    await expect(cache.get('candidate.png')).resolves.toBeInstanceOf(Blob);

    expect(resolveImage).toHaveBeenCalledTimes(2);
    cache.dispose();
  });

  it('retries the latest demand before draining queued prefetches', async () => {
    const firstAttempt = deferred<Blob>();
    const resolveImage = vi
      .fn<(imageName: string, signal?: AbortSignal) => Promise<Blob>>()
      .mockReturnValueOnce(firstAttempt.promise)
      .mockResolvedValueOnce(new Blob(['foreground retry']))
      .mockResolvedValueOnce(new Blob(['queued prefetch']));
    const cache = createStagedPreviewBlobCache(resolveImage, { maxConcurrentRequests: 1 });

    const selected = cache.get('a.png');
    cache.preload('b.png');
    firstAttempt.reject(new Error('temporary foreground failure'));

    await expect(selected).resolves.toBeInstanceOf(Blob);
    await vi.waitFor(() => expect(resolveImage).toHaveBeenCalledTimes(3));

    expect(resolveImage.mock.calls.map(([imageName]) => imageName)).toEqual(['a.png', 'a.png', 'b.png']);
    cache.dispose();
  });

  it('retries once when the latest foreground resolver throws synchronously', async () => {
    const resolveImage = vi
      .fn<(imageName: string, signal?: AbortSignal) => Promise<Blob>>()
      .mockImplementationOnce(() => {
        throw new Error('synchronous foreground failure');
      })
      .mockResolvedValueOnce(new Blob(['foreground retry']));
    const cache = createStagedPreviewBlobCache(resolveImage);

    await expect(cache.get('candidate.png')).resolves.toBeInstanceOf(Blob);

    expect(resolveImage).toHaveBeenCalledTimes(2);
    cache.dispose();
  });

  it('keeps total requests bounded and preempts stale selections during rapid cycling', async () => {
    let activeRequests = 0;
    let maxActiveRequests = 0;
    const resolveImage = vi.fn((_imageName: string, signal?: AbortSignal) => {
      activeRequests += 1;
      maxActiveRequests = Math.max(maxActiveRequests, activeRequests);
      return new Promise<Blob>((_resolve, reject) => {
        signal?.addEventListener(
          'abort',
          () => {
            activeRequests -= 1;
            reject(new Error('aborted'));
          },
          { once: true }
        );
      });
    });
    const cache = createStagedPreviewBlobCache(resolveImage, { maxConcurrentRequests: 2 });

    cache.preload('a.png');
    cache.preload('b.png');
    cache.preload('c.png');
    const selections = ['c.png', 'd.png', 'e.png'].map((imageName) => cache.get(imageName));
    cache.clear();
    await Promise.allSettled(selections);

    expect(resolveImage).toHaveBeenCalledTimes(5);
    expect(maxActiveRequests).toBe(2);
    cache.dispose();
  });

  it('does not restart an active demand when another request slot is free', async () => {
    const signals = new Map<string, AbortSignal>();
    const resolveImage = vi.fn((imageName: string, signal?: AbortSignal) => {
      if (signal) {
        signals.set(imageName, signal);
      }
      return new Promise<Blob>((_resolve, reject) => {
        signal?.addEventListener('abort', () => reject(new Error('aborted')), { once: true });
      });
    });
    const cache = createStagedPreviewBlobCache(resolveImage, { maxConcurrentRequests: 2 });

    const first = cache.get('a.png');
    const second = cache.get('b.png');

    expect(resolveImage.mock.calls.map(([imageName]) => imageName)).toEqual(['a.png', 'b.png']);
    expect(signals.get('a.png')?.aborted).toBe(false);

    cache.clear();
    await Promise.allSettled([first, second]);
    cache.dispose();
  });

  it('does not restart active prefetches when selecting a completed cached candidate', async () => {
    const pending = deferred<Blob>();
    const signals: AbortSignal[] = [];
    const resolveImage = vi.fn((imageName: string, signal?: AbortSignal) => {
      if (imageName === 'cached.png') {
        return Promise.resolve(new Blob(['cached']));
      }
      if (signal) {
        signals.push(signal);
      }
      return pending.promise;
    });
    const cache = createStagedPreviewBlobCache(resolveImage, { maxConcurrentRequests: 2 });

    cache.preload('cached.png');
    await expect(cache.get('cached.png')).resolves.toBeInstanceOf(Blob);
    cache.preload('b.png');
    cache.preload('c.png');
    expect(resolveImage).toHaveBeenCalledTimes(3);

    await expect(cache.get('cached.png')).resolves.toBeInstanceOf(Blob);

    expect(resolveImage).toHaveBeenCalledTimes(3);
    expect(signals.every((signal) => !signal.aborted)).toBe(true);
    cache.dispose();
  });

  it('evicts least-recently-used completed blobs when the byte budget is exceeded', async () => {
    const resolveImage = vi.fn((imageName: string) => Promise.resolve(new Blob([imageName.slice(0, 2)])));
    const cache = createStagedPreviewBlobCache(resolveImage, { maxBytes: 3, maxEntries: 4 });

    cache.preload('a.png');
    await vi.waitFor(() => expect(resolveImage).toHaveBeenCalledTimes(1));
    cache.preload('b.png');
    await vi.waitFor(() => expect(resolveImage).toHaveBeenCalledTimes(2));

    await expect(cache.get('a.png')).resolves.toBeInstanceOf(Blob);
    expect(resolveImage).toHaveBeenCalledTimes(3);
    cache.dispose();
  });

  it('clear aborts active work, drops queued prefetches, and allows a later foreground refetch', async () => {
    const pending = deferred<Blob>();
    let activeSignal: AbortSignal | undefined;
    const resolveImage = vi
      .fn<(imageName: string, signal?: AbortSignal) => Promise<Blob>>()
      .mockImplementationOnce((_imageName: string, signal?: AbortSignal) => {
        activeSignal = signal;
        return pending.promise;
      })
      .mockResolvedValueOnce(new Blob(['after cooldown']));
    const cache = createStagedPreviewBlobCache(resolveImage, { maxConcurrentRequests: 1 });

    cache.preload('active.png');
    cache.preload('queued.png');
    cache.clear();

    expect(activeSignal?.aborted).toBe(true);
    expect(resolveImage).toHaveBeenCalledOnce();

    await expect(cache.get('active.png')).resolves.toBeInstanceOf(Blob);
    expect(resolveImage).toHaveBeenCalledTimes(2);
    cache.dispose();
  });

  it('does not start foreground work after disposal', async () => {
    const resolveImage = vi.fn(() => Promise.resolve(new Blob(['unexpected'])));
    const cache = createStagedPreviewBlobCache(resolveImage);

    cache.dispose();

    await expect(cache.get('after-dispose.png')).rejects.toThrow('disposed');
    expect(resolveImage).not.toHaveBeenCalled();
  });
});

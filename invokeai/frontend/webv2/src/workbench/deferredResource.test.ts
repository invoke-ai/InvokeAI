import { describe, expect, it, vi } from 'vitest';

import { createDeferredResource } from './deferredResource';

describe('deferred resource', () => {
  it('loads once no matter how many callers ask', async () => {
    const loader = vi.fn(() => Promise.resolve('value'));
    const resource = createDeferredResource(loader);

    resource.preload();
    const [first, second] = await Promise.all([resource.load(), resource.load()]);

    expect(loader).toHaveBeenCalledTimes(1);
    expect(first).toBe('value');
    expect(second).toBe('value');
  });

  it('settles the promise so React use() can read it without suspending', async () => {
    const resource = createDeferredResource(() => Promise.resolve('value'));
    const promise = resource.load() as Promise<string> & { status?: string; value?: string };

    await promise;

    expect(promise.status).toBe('fulfilled');
    expect(promise.value).toBe('value');
    expect(resource.getStatus()).toBe('loaded');
  });

  it('reports failure and lets retry start a fresh attempt', async () => {
    let attempt = 0;
    const resource = createDeferredResource(() => {
      attempt += 1;
      return attempt === 1 ? Promise.reject(new Error('cold')) : Promise.resolve('warm');
    });

    await expect(resource.load()).rejects.toThrow('cold');
    expect(resource.getStatus()).toBe('failed');

    await expect(resource.retry()).resolves.toBe('warm');
    expect(resource.getStatus()).toBe('loaded');
  });

  it('rejects when the validator throws', async () => {
    const resource = createDeferredResource(
      () => Promise.resolve('bad'),
      () => {
        throw new TypeError('invalid');
      }
    );

    await expect(resource.load()).rejects.toThrow('invalid');
  });

  it('does not reject an unhandled promise when preload fails', async () => {
    const resource = createDeferredResource(() => Promise.reject(new Error('cold')));

    expect(() => resource.preload()).not.toThrow();
    await expect(resource.load()).rejects.toThrow('cold');
  });
});

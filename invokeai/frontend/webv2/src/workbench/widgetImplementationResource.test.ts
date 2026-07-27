import { describe, expect, it, vi } from 'vitest';

import type { WidgetImplementation } from './widgetContracts';

import { createWidgetImplementationResource } from './widgetImplementationResource';

const TestView = () => null;
const implementation: WidgetImplementation = { view: TestView };

describe('widget implementation resource', () => {
  it('loads once and shares the promise and implementation instance', async () => {
    const loader = vi.fn(() => Promise.resolve(implementation));
    const resource = createWidgetImplementationResource('test', loader);

    const first = resource.load();
    const second = resource.load();

    expect(first).toBe(second);
    await expect(first).resolves.toBe(implementation);
    expect(resource.load()).toBe(first);
    expect(loader).toHaveBeenCalledOnce();
    expect(resource.getStatus()).toBe('loaded');
  });

  it('caches a rejected load and starts exactly one new attempt on retry', async () => {
    const failure = new Error('chunk unavailable');
    const loader = vi.fn().mockRejectedValueOnce(failure).mockResolvedValueOnce(implementation);
    const resource = createWidgetImplementationResource('test', loader);

    const failed = resource.load();
    await expect(failed).rejects.toBe(failure);
    expect(resource.load()).toBe(failed);
    expect(resource.getStatus()).toBe('failed');

    const retry = resource.retry();
    expect(resource.retry()).toBe(retry);
    await expect(retry).resolves.toBe(implementation);
    expect(loader).toHaveBeenCalledTimes(2);
  });

  it('rejects implementations without a view through the same failure state', async () => {
    const resource = createWidgetImplementationResource('test', () => Promise.resolve({} as WidgetImplementation));

    await expect(resource.load()).rejects.toThrow('must provide a view component');
    expect(resource.getStatus()).toBe('failed');
  });

  it('preloads without leaking a rejected promise', async () => {
    const loader = vi.fn().mockRejectedValue(new Error('offline'));
    const resource = createWidgetImplementationResource('test', loader);

    resource.preload();
    await vi.waitFor(() => expect(resource.getStatus()).toBe('failed'));
    expect(loader).toHaveBeenCalledOnce();
  });

  // React's `use()` reads these fields off the thenable before deciding whether
  // to suspend. Without them a settled promise still costs a suspension on first
  // read — `use()` cannot inspect a native promise synchronously, so it attaches
  // a callback and throws. That suspension shows a fallback, and once a fallback
  // is on screen React withholds the resolved tree for FALLBACK_THROTTLE_MS
  // (300ms) to avoid a flash. That throttle was the entire measured cost of
  // switching layout, long after the chunk had finished downloading, so these
  // three fields are load-bearing rather than decorative.
  describe('exposes its settled state on the promise for React `use()`', () => {
    it('marks the promise pending, then fulfilled with the value', async () => {
      const resource = createWidgetImplementationResource('test', () => Promise.resolve(implementation));

      const pending = resource.load() as Promise<WidgetImplementation> & {
        status?: string;
        value?: WidgetImplementation;
      };
      expect(pending.status).toBe('pending');

      await pending;

      expect(pending.status).toBe('fulfilled');
      expect(pending.value).toBe(implementation);
    });

    it('marks the promise rejected with the reason', async () => {
      const error = new Error('offline');
      const resource = createWidgetImplementationResource('test', () => Promise.reject(error));

      const pending = resource.load() as Promise<WidgetImplementation> & { reason?: unknown; status?: string };
      await expect(pending).rejects.toThrow('offline');

      expect(pending.status).toBe('rejected');
      expect(pending.reason).toBe(error);
    });

    it('keeps the settled state on the promise handed to later callers', async () => {
      const resource = createWidgetImplementationResource('test', () => Promise.resolve(implementation));

      await resource.load();
      const again = resource.load() as Promise<WidgetImplementation> & {
        status?: string;
        value?: WidgetImplementation;
      };

      expect(again.status).toBe('fulfilled');
      expect(again.value).toBe(implementation);
    });
  });
});

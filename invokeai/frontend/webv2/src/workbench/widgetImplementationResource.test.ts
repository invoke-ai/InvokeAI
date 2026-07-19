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
});

import { describe, expect, it, vi } from 'vitest';

import type { WidgetImplementation } from './widgetContracts';

import { createWidgetImplementationResource } from './widgetImplementationResource';

const TestView = () => null;
const implementation: WidgetImplementation = { view: TestView };

describe('widget implementation resource', () => {
  it('loads a valid implementation', async () => {
    const loader = vi.fn(() => Promise.resolve(implementation));
    const resource = createWidgetImplementationResource('test', loader);

    await expect(resource.load()).resolves.toBe(implementation);
    expect(resource.getStatus()).toBe('loaded');
  });

  it('rejects implementations without a view through the same failure state', async () => {
    const resource = createWidgetImplementationResource('test', () => Promise.resolve({} as WidgetImplementation));

    await expect(resource.load()).rejects.toThrow('must provide a view component');
    expect(resource.getStatus()).toBe('failed');
  });
});

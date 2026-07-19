import type { WidgetImplementationResource } from '@workbench/widgetContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { Suspense, act, use } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { WidgetFailureBoundary } from './widget-frame/WidgetFailureBoundary';
import { createWidgetImplementationResource } from './widgetImplementationResource';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const TestView = () => null;
const loadingFallback = <div data-testid="loading">Loading</div>;

const ResourceProbe = ({ resource }: { resource: WidgetImplementationResource }) => {
  use(resource.load());

  return <div data-testid="loaded">Loaded</div>;
};

const renderResource = async (resource: WidgetImplementationResource) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <WidgetFailureBoundary resetKey="test" widgetId="test" onRetry={resource.retry}>
          <Suspense fallback={loadingFallback}>
            <ResourceProbe resource={resource} />
          </Suspense>
        </WidgetFailureBoundary>
      </ChakraProvider>
    );
  });
};

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('widget implementation resource rendering', () => {
  it('suspends until the shared implementation resolves', async () => {
    let resolveLoad: (() => void) | undefined;
    const resource = createWidgetImplementationResource(
      'test',
      () =>
        new Promise((resolve) => {
          resolveLoad = () => resolve({ view: TestView });
        })
    );

    await renderResource(resource);
    expect(host?.querySelector('[data-testid="loading"]')).not.toBeNull();

    await act(() => resolveLoad?.());
    expect(host?.querySelector('[data-testid="loaded"]')).not.toBeNull();
  });

  it('isolates a rejected chunk and retries through the same resource', async () => {
    const loader = vi
      .fn()
      .mockRejectedValueOnce(new Error('chunk unavailable'))
      .mockResolvedValueOnce({ view: TestView });
    const resource = createWidgetImplementationResource('test', loader);

    await renderResource(resource);
    await expect.poll(() => host?.textContent).toContain('Widget failed: test');

    const retry = [...(host?.querySelectorAll('button') ?? [])].find((button) => button.textContent === 'Retry');
    expect(retry).toBeDefined();
    await act(() => retry?.click());

    await expect.poll(() => host?.querySelector('[data-testid="loaded"]')).not.toBeNull();
    expect(loader).toHaveBeenCalledTimes(2);
  });
});

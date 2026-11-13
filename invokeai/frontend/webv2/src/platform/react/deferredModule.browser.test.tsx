import { DEFAULT_LOGGING_CONFIG } from '@platform/logging/contracts';
import { configureLogging, getLogSnapshot, resetLogging } from '@platform/logging/logger';
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop -- each mount is a fresh render */
import { act, Component, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { createDeferredModule } from './deferredModule';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const view = { Greeting: () => <p>loaded view</p> };

class HostBoundary extends Component<{ children: ReactNode }, { message: string | null }> {
  state: { message: string | null } = { message: null };
  static getDerivedStateFromError(error: unknown) {
    return { message: error instanceof Error ? error.message : String(error) };
  }
  render() {
    return this.state.message === null ? this.props.children : <p>host caught: {this.state.message}</p>;
  }
}

const Broken = (): ReactNode => {
  throw new Error('render failed');
};

describe('createDeferredModule', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    resetLogging();
    configureLogging(DEFAULT_LOGGING_CONFIG);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    resetLogging();
  });

  it('shows its fallback while loading and after a failed load, and loads again on the next mount', async () => {
    let finishFirst: (error: Error) => void = () => {};
    const importModule = vi
      .fn<() => Promise<typeof view>>()
      .mockImplementationOnce(
        () =>
          new Promise((_resolve, reject) => {
            finishFirst = reject;
          })
      )
      .mockResolvedValue(view);
    const deferred = createDeferredModule(importModule);
    const mount = (key: string) =>
      act(() =>
        root.render(<deferred.Deferred key={key} fallback={<p>fallback</p>} render={(module) => <module.Greeting />} />)
      );
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});

    await mount('first');
    expect(host.textContent).toBe('fallback');
    await act(async () => {
      finishFirst(new Error('Failed to fetch dynamically imported module'));
      await Promise.resolve();
    });
    expect(host.textContent).toBe('fallback');
    expect(getLogSnapshot().entries).toMatchObject([
      {
        error: { message: 'Failed to fetch dynamically imported module' },
        level: 'warn',
        name: 'deferred-module.load-failed',
        source: { area: 'deferred-module', namespace: 'app' },
      },
    ]);

    await mount('second');
    await vi.waitFor(() => expect(host.textContent).toBe('loaded view'));
    expect(importModule).toHaveBeenCalledTimes(2);
    expect(getLogSnapshot().entries).toHaveLength(1);

    // Once loaded, a mount renders the module directly.
    await mount('third');
    expect(host.textContent).toBe('loaded view');
    expect(importModule).toHaveBeenCalledTimes(2);
    consoleError.mockRestore();
  });

  it('passes an error the loaded module throws while rendering to the host, loading or loaded', async () => {
    const deferred = createDeferredModule(() => Promise.resolve({ Broken }));
    const mount = (key: string) =>
      act(() =>
        root.render(
          <HostBoundary key={key}>
            <deferred.Deferred fallback={<p>fallback</p>} render={(module) => <module.Broken />} />
          </HostBoundary>
        )
      );
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});

    // The first mount renders through its lazy load.
    await mount('loading');
    await vi.waitFor(() => expect(host.textContent).toBe('host caught: render failed'));

    // Later mounts render the loaded module directly.
    await mount('loaded');
    expect(host.textContent).toBe('host caught: render failed');
    consoleError.mockRestore();
  });
});

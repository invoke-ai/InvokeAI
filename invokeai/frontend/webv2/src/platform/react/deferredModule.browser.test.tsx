/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop -- each mount is a fresh render */
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { createDeferredModule } from './deferredModule';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const view = { Greeting: () => <p>loaded view</p> };

describe('createDeferredModule', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
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

    await mount('second');
    await vi.waitFor(() => expect(host.textContent).toBe('loaded view'));
    expect(importModule).toHaveBeenCalledTimes(2);

    // Once loaded, a mount renders the module directly.
    await mount('third');
    expect(host.textContent).toBe('loaded view');
    expect(importModule).toHaveBeenCalledTimes(2);
    consoleError.mockRestore();
  });
});

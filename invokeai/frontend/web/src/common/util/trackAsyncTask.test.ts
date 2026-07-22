import { describe, expect, it, vi } from 'vitest';

import { trackAsyncTask } from './trackAsyncTask';

describe('trackAsyncTask', () => {
  it('reports loading until the tracked task settles', async () => {
    let resolveTask: (() => void) | undefined;
    const task = new Promise<void>((resolve) => {
      resolveTask = resolve;
    });
    const onLoadingChanged = vi.fn();

    const tracked = trackAsyncTask(() => task, onLoadingChanged);
    expect(onLoadingChanged).toHaveBeenCalledWith(true);
    expect(onLoadingChanged).not.toHaveBeenCalledWith(false);

    resolveTask?.();
    await tracked;
    expect(onLoadingChanged).toHaveBeenLastCalledWith(false);
  });

  it('clears loading when the task rejects', async () => {
    const onLoadingChanged = vi.fn();
    await expect(trackAsyncTask(() => Promise.reject(new Error('failed')), onLoadingChanged)).rejects.toThrow('failed');
    expect(onLoadingChanged.mock.calls).toEqual([[true], [false]]);
  });
});

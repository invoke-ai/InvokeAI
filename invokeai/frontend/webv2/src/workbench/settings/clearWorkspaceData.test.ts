import { describe, expect, it, vi } from 'vitest';

import { clearWorkspaceData, consumeWorkspaceClearFailure, rememberWorkspaceClearFailure } from './clearWorkspaceData';

describe('clearWorkspaceData', () => {
  it('waits for both cleanup paths and reports partial failures', async () => {
    const clearProjects = vi.fn(() => Promise.reject(new Error('server offline')));
    const clearSettings = vi.fn(() => Promise.resolve());

    await expect(clearWorkspaceData(clearProjects, clearSettings)).resolves.toEqual(['projects']);
    expect(clearProjects).toHaveBeenCalledOnce();
    expect(clearSettings).toHaveBeenCalledOnce();
  });

  it('reports success only when both cleanup paths succeed', async () => {
    await expect(
      clearWorkspaceData(
        () => Promise.resolve(),
        () => Promise.resolve()
      )
    ).resolves.toEqual([]);
  });

  it('carries a partial-clear error across exactly one reload', () => {
    const values = new Map<string, string>();
    const storage = {
      getItem: (key: string) => values.get(key) ?? null,
      removeItem: (key: string) => void values.delete(key),
      setItem: (key: string, value: string) => void values.set(key, value),
    };

    rememberWorkspaceClearFailure('partially cleared', storage);

    expect(consumeWorkspaceClearFailure(storage)).toBe('partially cleared');
    expect(consumeWorkspaceClearFailure(storage)).toBeNull();
  });
});

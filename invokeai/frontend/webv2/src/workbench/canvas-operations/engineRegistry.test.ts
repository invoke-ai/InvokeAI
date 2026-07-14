import type { WorkbenchState } from '@workbench/types';

import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it, vi } from 'vitest';

import type { EngineDeps, RegistryTimers } from './engineRegistry';

import { createEngineRegistry } from './engineRegistry';

const createFakeDeps = (): EngineDeps => ({
  backend: createTestStubRasterBackend(),
  imageResolver: () => Promise.resolve(new Blob()),
  store: {
    dispatch: () => {},
    getState: () => ({ projects: [] }) as unknown as WorkbenchState,
    subscribe: () => () => {},
  },
});

const createFakeTimers = (): { timers: RegistryTimers; flush: () => void; pending: () => number } => {
  let nextHandle = 0;
  const scheduled = new Map<number, () => void>();
  return {
    flush: () => {
      const callbacks = [...scheduled.values()];
      scheduled.clear();
      for (const callback of callbacks) {
        callback();
      }
    },
    pending: () => scheduled.size,
    timers: {
      clearTimeout: (handle) => {
        scheduled.delete(handle);
      },
      setTimeout: (handler) => {
        nextHandle += 1;
        scheduled.set(nextHandle, handler);
        return nextHandle;
      },
    },
  };
};

describe('createEngineRegistry', () => {
  it('returns the same instance per project id and distinct instances across ids', () => {
    const registry = createEngineRegistry();
    const deps = createFakeDeps();

    const first = registry.getOrCreateEngine('p1', deps);
    const again = registry.getOrCreateEngine('p1', deps);
    const other = registry.getOrCreateEngine('p2', deps);

    expect(again).toBe(first);
    expect(other).not.toBe(first);
    expect(registry.getEngine('p1')).toBe(first);

    first.lifecycle.dispose();
    other.lifecycle.dispose();
  });

  it('disposes after the grace period once the last reference is released', async () => {
    const fakeTimers = createFakeTimers();
    const registry = createEngineRegistry({ gracePeriodMs: 30_000, timers: fakeTimers.timers });
    const engine = registry.getOrCreateEngine('p1', createFakeDeps());
    const disposeSpy = vi.spyOn(engine.lifecycle, 'dispose');
    const cooldownSpy = vi.spyOn(engine.lifecycle, 'beginCooldown');

    registry.releaseEngine('p1');
    expect(fakeTimers.pending()).toBe(1);
    expect(cooldownSpy).toHaveBeenCalledOnce();
    expect(disposeSpy).not.toHaveBeenCalled();

    fakeTimers.flush();
    await Promise.resolve();
    await Promise.resolve();
    expect(disposeSpy).toHaveBeenCalledTimes(1);
    expect(registry.getEngine('p1')).toBeUndefined();
  });

  it('cancels the pending disposal when the engine is re-acquired', () => {
    const fakeTimers = createFakeTimers();
    const registry = createEngineRegistry({ timers: fakeTimers.timers });
    const engine = registry.getOrCreateEngine('p1', createFakeDeps());
    const disposeSpy = vi.spyOn(engine.lifecycle, 'dispose');
    const activateSpy = vi.spyOn(engine.lifecycle, 'activate');

    registry.releaseEngine('p1');
    expect(fakeTimers.pending()).toBe(1);

    const reacquired = registry.getOrCreateEngine('p1', createFakeDeps());
    expect(reacquired).toBe(engine);
    expect(fakeTimers.pending()).toBe(0);
    expect(activateSpy).toHaveBeenCalledOnce();

    fakeTimers.flush();
    expect(disposeSpy).not.toHaveBeenCalled();

    engine.lifecycle.dispose();
  });

  it('only schedules disposal when the last reference is released', async () => {
    const fakeTimers = createFakeTimers();
    const registry = createEngineRegistry({ timers: fakeTimers.timers });
    const deps = createFakeDeps();

    registry.getOrCreateEngine('p1', deps);
    registry.getOrCreateEngine('p1', deps);

    registry.releaseEngine('p1');
    expect(fakeTimers.pending()).toBe(0);

    registry.releaseEngine('p1');
    expect(fakeTimers.pending()).toBe(1);

    fakeTimers.flush();
    await Promise.resolve();
    await Promise.resolve();
    expect(registry.getEngine('p1')).toBeUndefined();
  });
});

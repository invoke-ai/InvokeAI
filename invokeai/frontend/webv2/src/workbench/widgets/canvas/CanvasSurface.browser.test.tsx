import type { CanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import type { EngineDeps, EngineRegistry } from '@workbench/canvas-operations/engineRegistry';
import type { CanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import type { CanvasStateContractV2 } from '@workbench/types';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createEngineRegistry } from '@workbench/canvas-operations/engineRegistry';
import { createEmptyCanvasStateV2 } from '@workbench/canvasMigration';
import { act, StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { CanvasSurface } from './CanvasSurface';

type Cleanup = () => void | Promise<void>;

const cleanupStack: Cleanup[] = [];
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const addCleanup = (cleanup: Cleanup): void => {
  cleanupStack.push(cleanup);
};

const delay = (ms: number): Promise<void> =>
  new Promise((resolve) => {
    globalThis.setTimeout(resolve, ms);
  });

const createTrackedRegistry = (gracePeriodMs: number): EngineRegistry => {
  const registry = createEngineRegistry({ gracePeriodMs });
  // Registered before references and roots, so reverse-order cleanup releases
  // those first and then lets the registry's grace timers finish.
  addCleanup(() => delay(gracePeriodMs + 20));
  return registry;
};

const acquireTrackedEngine = (
  registry: EngineRegistry,
  projectId: string,
  deps: EngineDeps
): { engine: CanvasEngine; release: () => void } => {
  const engine = registry.getOrCreateEngine(projectId, deps);
  let isHeld = true;
  const release = (): void => {
    if (!isHeld) {
      return;
    }
    isHeld = false;
    registry.releaseEngine(projectId);
  };
  addCleanup(release);
  return { engine, release };
};

const createTrackedRoot = (host: HTMLDivElement) => {
  const root = createRoot(host);
  let isMounted = true;
  const unmount = async (): Promise<void> => {
    if (!isMounted) {
      return;
    }
    isMounted = false;
    try {
      await act(() => {
        root.unmount();
      });
    } finally {
      host.remove();
    }
  };
  addCleanup(unmount);
  return { root, unmount };
};

const createEngineDeps = (state: CanvasStateContractV2): EngineDeps => {
  const listeners = new Set<() => void>();
  const mutationPort: CanvasProjectMutationPort = {
    dispatch: () => false,
    getCanvasState: () => state,
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
  return {
    getMainModelBase: () => null,
    imageResolver: () => Promise.resolve(new Blob()),
    mutationPort,
    reportError: () => undefined,
  };
};

const nextFrame = (): Promise<void> =>
  new Promise((resolve) => {
    requestAnimationFrame(() => resolve());
  });

const StrictCanvasHarness = ({ engine }: { engine: CanvasEngine }) => (
  <StrictMode>
    <ChakraProvider value={system}>
      <div data-testid="surface-bounds" style={{ height: 120, width: 160 }}>
        <CanvasSurface engine={engine} />
      </div>
    </ChakraProvider>
  </StrictMode>
);

afterEach(async () => {
  let firstError: unknown;
  for (const cleanup of cleanupStack.reverse()) {
    try {
      await cleanup();
    } catch (error) {
      firstError ??= error;
    }
  }
  cleanupStack.length = 0;
  vi.restoreAllMocks();
  if (firstError) {
    throw new Error('CanvasSurface browser-test cleanup failed', { cause: firstError });
  }
});

describe('CanvasSurface browser lifecycle', () => {
  it('is StrictMode-safe across resize, registry reacquisition, detachment, and project switching', async () => {
    const gracePeriodMs = 80;
    const registry = createTrackedRegistry(gracePeriodMs);
    const projectADeps = createEngineDeps(createEmptyCanvasStateV2(64, 64));
    const firstProjectA = acquireTrackedEngine(registry, 'project-a', projectADeps);
    const disposeA = vi.spyOn(firstProjectA.engine.lifecycle, 'dispose');
    const beginCooldownA = vi.spyOn(firstProjectA.engine.lifecycle, 'beginCooldown');
    const releasedAt = performance.now();
    firstProjectA.release();
    expect(beginCooldownA).toHaveBeenCalledOnce();
    expect(firstProjectA.engine.lifecycle.getLifecycleState()).toBe('cooling');

    await delay(5);
    const reacquiredProjectA = acquireTrackedEngine(registry, 'project-a', projectADeps);
    const projectA = reacquiredProjectA.engine;
    expect(projectA).toBe(firstProjectA.engine);
    expect(projectA.lifecycle.getLifecycleState()).toBe('active');

    const originalDeadline = releasedAt + gracePeriodMs;
    await delay(Math.max(0, originalDeadline - performance.now()) + 20);
    expect(performance.now()).toBeGreaterThan(originalDeadline);
    expect(registry.getEngine('project-a')).toBe(projectA);
    expect(disposeA).not.toHaveBeenCalled();
    expect(projectA.lifecycle.getLifecycleState()).toBe('active');

    const projectBReference = acquireTrackedEngine(
      registry,
      'project-b',
      createEngineDeps(createEmptyCanvasStateV2(32, 48))
    );
    const projectB = projectBReference.engine;
    const attachA = vi.spyOn(projectA.surface, 'attach');
    const resizeA = vi.spyOn(projectA.surface, 'resize');
    const detachA = vi.spyOn(projectA.surface, 'detach');
    const attachB = vi.spyOn(projectB.surface, 'attach');
    const resizeB = vi.spyOn(projectB.surface, 'resize');
    const detachB = vi.spyOn(projectB.surface, 'detach');

    const host = document.createElement('div');
    document.body.append(host);
    const { root, unmount } = createTrackedRoot(host);

    await act(async () => {
      root.render(<StrictCanvasHarness engine={projectA} />);
      await nextFrame();
    });

    expect(attachA).toHaveBeenCalled();
    expect(resizeA).toHaveBeenCalledWith(160, 120, globalThis.devicePixelRatio || 1);
    const firstScreen = host.querySelector('canvas');
    expect(firstScreen?.width).toBe(Math.round(160 * Math.min(globalThis.devicePixelRatio || 1, 2)));
    expect(firstScreen?.height).toBe(Math.round(120 * Math.min(globalThis.devicePixelRatio || 1, 2)));

    const bounds = host.querySelector<HTMLElement>('[data-testid="surface-bounds"]');
    expect(bounds).not.toBeNull();
    bounds!.style.width = '180px';
    bounds!.style.height = '90px';
    await act(async () => {
      await nextFrame();
      await nextFrame();
    });
    expect(resizeA).toHaveBeenCalledWith(180, 90, globalThis.devicePixelRatio || 1);

    bounds!.style.width = '160px';
    bounds!.style.height = '120px';

    await act(async () => {
      root.render(<StrictCanvasHarness engine={projectB} />);
      await nextFrame();
    });

    expect(detachA).toHaveBeenCalled();
    expect(attachB).toHaveBeenCalled();
    expect(resizeB).toHaveBeenCalledWith(160, 120, globalThis.devicePixelRatio || 1);

    await unmount();
    reacquiredProjectA.release();
    projectBReference.release();
    expect(detachB).toHaveBeenCalled();

    await delay(gracePeriodMs + 20);
    expect(registry.getEngine('project-a')).toBeUndefined();
    expect(registry.getEngine('project-b')).toBeUndefined();
  });
});

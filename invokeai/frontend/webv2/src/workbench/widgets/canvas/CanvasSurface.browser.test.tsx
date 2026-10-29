import type { CanvasStateContractV3 } from '@workbench/canvas-engine/contracts';
import type { CanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import type { EngineDeps, EngineRegistry } from '@workbench/canvas-operations/engineRegistry';
import type { CanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';
import type { WorkbenchInternalStore } from '@workbench/workbenchStore';
import type { ReactNode } from 'react';

import { ChakraProvider } from '@chakra-ui/react';
import { ResizeHandle } from '@platform/ui/ResizeHandle';
import { system } from '@theme/system';
import { createEngineRegistry } from '@workbench/canvas-operations/engineRegistry';
import { createEmptyCanvasState } from '@workbench/canvasMigration';
import { FocusRegionProvider, useFocusRegionProps } from '@workbench/focusRegions';
import { createTestFocusController } from '@workbench/focusRegions.testing';
import { WorkbenchFocusProvider } from '@workbench/WorkbenchRuntime';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { act, StrictMode, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

/** The surface reads the workbench only for key ownership: the active project and its widget placement. */
const workbench = vi.hoisted(() => ({ store: null as unknown as WorkbenchInternalStore }));
vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContextModule>()),
  useWorkbenchInternalStore: () => workbench.store,
  useWorkbenchQueries: () => workbench.store.queries,
  useWorkbenchSubscription: () => workbench.store.subscribe,
}));

import { CanvasSurface } from './CanvasSurface';

workbench.store = createWorkbenchStore();

/** Focus for the surface outside the workbench shell; ownership itself is covered under a real provider below. */
const TestFocus = ({ children }: { children: ReactNode }) => {
  const [controller] = useState(() => createTestFocusController());
  return <FocusRegionProvider controller={controller}>{children}</FocusRegionProvider>;
};

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

const createEngineDeps = (state: CanvasStateContractV3): EngineDeps => {
  const listeners = new Set<() => void>();
  const mutationPort: CanvasProjectMutationPort = {
    commitEdit: () => undefined,
    dispatch: () => false,
    getCanvasState: () => state,
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
  return {
    getMainModelBase: () => null,
    ensureProjectOnServer: () => Promise.resolve(),
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
      <TestFocus>
        <div data-testid="surface-bounds" style={{ height: 120, width: 160 }}>
          <CanvasSurface engine={engine} instanceId="canvas" />
        </div>
      </TestFocus>
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
    const projectADeps = createEngineDeps(createEmptyCanvasState(64, 64));
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
      createEngineDeps(createEmptyCanvasState(32, 48))
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

const ignoreCommit = (): void => undefined;

const ResizeDragHarness = ({ engine, showCanvas = true }: { engine: CanvasEngine; showCanvas?: boolean }) => {
  const paneRef = useRef<HTMLDivElement>(null);

  return (
    <ChakraProvider value={system}>
      <TestFocus>
        <div style={{ display: 'flex' }}>
          <div ref={paneRef} style={{ width: 100 }} />
          <ResizeHandle
            label="Resize pane"
            max={300}
            min={50}
            orientation="vertical"
            pane="before"
            paneRef={paneRef}
            value={100}
            onCommit={ignoreCommit}
          />
          <div data-testid="surface-bounds" style={{ height: 120, width: 160 }}>
            {showCanvas ? <CanvasSurface engine={engine} instanceId="canvas" /> : null}
          </div>
        </div>
      </TestFocus>
    </ChakraProvider>
  );
};

describe('CanvasSurface during a resize drag', () => {
  it('keeps its bitmap until the drag ends, cropping instead of stretching', async () => {
    const registry = createTrackedRegistry(20);
    const { engine } = acquireTrackedEngine(registry, 'project-a', createEngineDeps(createEmptyCanvasState(64, 64)));
    const host = document.createElement('div');
    document.body.append(host);
    const { root } = createTrackedRoot(host);

    await act(async () => {
      root.render(<ResizeDragHarness engine={engine} />);
      await nextFrame();
    });
    const resize = vi.spyOn(engine.surface, 'resize');
    const separator = host.querySelector<HTMLElement>('[role="separator"]')!;
    const bounds = host.querySelector<HTMLElement>('[data-testid="surface-bounds"]')!;
    const screen = host.querySelector('canvas')!;
    const pointer = (type: string, buttons: number) =>
      new PointerEvent(type, { bubbles: true, buttons, clientX: 0, pointerId: 1 });

    await act(() => separator.dispatchEvent(pointer('pointerdown', 1)));
    bounds.style.width = '200px';
    await act(async () => {
      await nextFrame();
      await nextFrame();
    });

    expect(resize).not.toHaveBeenCalled();
    expect(screen.style.width).toBe('160px');

    await act(() => window.dispatchEvent(pointer('pointerup', 0)));

    expect(resize).toHaveBeenCalledExactlyOnceWith(200, 120, globalThis.devicePixelRatio || 1);
    expect(screen.style.width).toBe('200px');
  });

  it('sizes a canvas that first appears during a drag without waiting for it', async () => {
    const registry = createTrackedRegistry(20);
    const { engine } = acquireTrackedEngine(registry, 'project-a', createEngineDeps(createEmptyCanvasState(64, 64)));
    const resize = vi.spyOn(engine.surface, 'resize');
    const host = document.createElement('div');
    document.body.append(host);
    const { root } = createTrackedRoot(host);

    await act(() => root.render(<ResizeDragHarness engine={engine} showCanvas={false} />));
    const separator = host.querySelector<HTMLElement>('[role="separator"]')!;
    await act(() =>
      separator.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, buttons: 1, clientX: 0, pointerId: 1 }))
    );
    await act(() => root.render(<ResizeDragHarness engine={engine} />));

    expect(resize).toHaveBeenCalledExactlyOnceWith(160, 120, globalThis.devicePixelRatio || 1);
    expect(host.querySelector('canvas')!.style.width).toBe('160px');

    await act(() =>
      window.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, buttons: 0, clientX: 0, pointerId: 1 }))
    );
  });
});

describe('CanvasSurface keyboard ownership', () => {
  it('gives hold keys to its keyboard root and leaves them to a tree row reached by keyboard', async () => {
    const registry = createTrackedRegistry(20);
    const { engine } = acquireTrackedEngine(registry, 'project-a', createEngineDeps(createEmptyCanvasState(64, 64)));
    const host = document.createElement('div');
    document.body.append(host);
    const { root } = createTrackedRoot(host);
    await act(async () => {
      root.render(
        <>
          <StrictCanvasHarness engine={engine} />
          <div role="tree">
            <div aria-selected="false" role="treeitem" tabIndex={0}>
              Layer
            </div>
          </div>
        </>
      );
      await nextFrame();
    });
    const overlay = host.querySelectorAll('canvas')[1]!;
    const surface = overlay.parentElement!;
    const row = host.querySelector<HTMLElement>('[role="treeitem"]')!;
    const tool = () => engine.interaction.get('activeTool');
    engine.tools.setTool('brush');

    // Focus that did not arrive by pointer is owned only through the keyboard root.
    await userEvent.hover(overlay);
    await userEvent.keyboard('{Tab}');
    surface.focus();
    await userEvent.keyboard('{Space>}');
    expect(tool()).toBe('view');
    await userEvent.keyboard('{/Space}');
    expect(tool()).toBe('brush');

    await userEvent.click(overlay);
    expect(document.activeElement).toBe(surface);
    await userEvent.tab();
    expect(document.activeElement).toBe(row);
    await userEvent.hover(overlay);
    await userEvent.keyboard('{Space>}');
    expect(tool()).toBe('brush');
    await userEvent.keyboard('{/Space}');
  });
});

/** The shell's layout in miniature: the canvas, the Generate panel and the Layers panel, each a focus region. */
const WorkbenchHarness = ({ engine }: { engine: CanvasEngine }) => (
  <ChakraProvider value={system}>
    <WorkbenchFocusProvider>
      <Region instanceId="canvas" region="center" typeId="canvas">
        <div style={{ height: 120, width: 160 }}>
          <CanvasSurface engine={engine} instanceId="canvas" />
        </div>
      </Region>
      <Region instanceId="generate" region="left" typeId="generate">
        <span>Model</span>
        <input aria-label="Prompt" />
      </Region>
      <Region instanceId="layers" region="right" typeId="layers">
        <div role="tree">
          <div aria-selected="true" role="treeitem" tabIndex={0}>
            Layer
          </div>
        </div>
      </Region>
    </WorkbenchFocusProvider>
  </ChakraProvider>
);

const Region = ({
  children,
  instanceId,
  region,
  typeId,
}: {
  children: ReactNode;
  instanceId: string;
  region: 'center' | 'left' | 'right';
  typeId: string;
}) => (
  <div
    {...useFocusRegionProps(region)}
    data-hotkey-widget-instance-id={instanceId}
    data-hotkey-widget-region={region}
    data-hotkey-widget-type-id={typeId}
  >
    {children}
  </div>
);

describe('CanvasSurface key owner', () => {
  const mount = async () => {
    const registry = createTrackedRegistry(20);
    const projectId = workbench.store.getSnapshot().activeProject.id;
    const { engine } = acquireTrackedEngine(registry, projectId, createEngineDeps(createEmptyCanvasState(64, 64)));
    const host = document.createElement('div');
    document.body.append(host);
    const { root } = createTrackedRoot(host);
    await act(async () => {
      root.render(<WorkbenchHarness engine={engine} />);
      await nextFrame();
    });
    const overlay = host.querySelectorAll('canvas')[1]!;
    return {
      clickCanvas: () => userEvent.click(overlay, { position: { x: 4, y: 4 } }),
      clickGenerate: () => userEvent.click(host.querySelector('span')!),
      clickLayersRow: () => userEvent.click(host.querySelector('[role="treeitem"]')!),
      engine,
      hasSelection: () => engine.interaction.get('hasSelection'),
      overlay,
      prompt: host.querySelector('input')!,
      tool: () => engine.interaction.get('activeTool'),
    };
  };

  it('keeps Escape and C from the canvas after a click in another widget', async () => {
    const view = await mount();
    view.engine.tools.setTool('view');
    view.engine.selection.selectAll();
    await view.clickGenerate();

    await userEvent.keyboard('{Escape}');
    await userEvent.keyboard('c');
    expect(view.hasSelection()).toBe(true);
    expect(view.tool()).toBe('view');
  });

  it('lets Escape and C act from the canvas and from a Layers row', async () => {
    const view = await mount();
    view.engine.tools.setTool('view');
    view.engine.selection.selectAll();
    await view.clickLayersRow();

    await userEvent.keyboard('{Escape}');
    expect(view.hasSelection()).toBe(false);
    // Away from the canvas, a C tap selects the frame tool.
    await userEvent.keyboard('c');
    expect(view.tool()).toBe('bbox');

    view.engine.tools.setTool('view');
    view.engine.selection.selectAll();
    await view.clickCanvas();
    await userEvent.keyboard('{Escape}');
    expect(view.hasSelection()).toBe(false);
    await userEvent.keyboard('{c>}');
    expect(view.tool()).toBe('bbox');
    await userEvent.keyboard('{/c}');
  });

  it('applies Enter only from the canvas itself', async () => {
    const view = await mount();
    view.engine.tools.setTool('lasso');
    view.engine.interaction.set('lassoOptions', { ...view.engine.interaction.get('lassoOptions'), shape: 'polygon' });
    for (const position of [
      { x: 20, y: 20 },
      { x: 120, y: 20 },
      { x: 120, y: 100 },
    ]) {
      await userEvent.click(view.overlay, { position });
    }

    await view.clickLayersRow();
    await userEvent.keyboard('{Enter}');
    await view.clickGenerate();
    await userEvent.keyboard('{Enter}');
    expect(view.hasSelection()).toBe(false);

    await userEvent.click(view.overlay, { position: { x: 20, y: 100 } });
    await userEvent.keyboard('{Enter}');
    expect(view.hasSelection()).toBe(true);
  });

  it('keeps the Space hold over the hovered canvas after a click elsewhere, and never takes typing', async () => {
    const view = await mount();
    view.engine.tools.setTool('brush');
    await view.clickGenerate();

    await userEvent.hover(view.overlay);
    await userEvent.keyboard('{Space>}');
    expect(view.tool()).toBe('view');
    await userEvent.keyboard('{/Space}');
    expect(view.tool()).toBe('brush');

    await userEvent.click(view.prompt);
    await userEvent.hover(view.overlay);
    await userEvent.keyboard('c');
    expect(view.prompt.value).toBe('c');
    expect(view.tool()).toBe('brush');
  });
});

import type * as DndKitCore from '@dnd-kit/core';
import type { GalleryImage } from '@features/gallery';
import type * as GalleryBackend from '@features/gallery/data/backend';
import type {
  CanvasControlLayerContract,
  CanvasLayerContract,
  CanvasNodeContract,
  ControlLayerViewOutcome,
} from '@workbench/canvas-engine/api';
import type { CanvasStagingCandidateContract } from '@workbench/canvas-engine/contracts';
import type * as CanvasOperationsApi from '@workbench/canvas-operations/api';
import type { CanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import type { ExtensionRegistry } from '@workbench/extensions/extensionRegistry';
import type { Project } from '@workbench/projectContracts';
import type * as ProjectsApi from '@workbench/projects/api';
import type { WorkbenchQueueItem as QueueItem } from '@workbench/queueHistoryContracts';
import type { WidgetHotkeyContribution, WidgetCommandContribution, WidgetViewProps } from '@workbench/widgetContracts';
import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';

import { Box, ChakraProvider } from '@chakra-ui/react';
import { auditAccessibility } from '@platform/browser/auditAccessibility.testing';
import { TINY_PNG_BASE64 } from '@platform/browser/nativeDrag.testing';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { createControlLayer, createVectorLayer } from '@workbench/canvas-engine/api';
import { layerContract, stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { getCanvasOperations } from '@workbench/canvas-operations/api';
import { createControlDetectionStatusStore } from '@workbench/canvas-operations/controlDetectionStatusStore';
import { createCanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import { createCanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import { createExtensionRegistry } from '@workbench/extensions/extensionRegistry';
import { WorkbenchHotkeyRuntime } from '@workbench/hotkeys/WorkbenchHotkeyRuntime';
import { DEFAULT_PREFERENCES, patchWorkbenchPreferences } from '@workbench/settings/store';
import { runNotificationUndo } from '@workbench/shell/notifications/undoRegistry';
import { WorkbenchFocusProvider } from '@workbench/WorkbenchRuntime';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { createInstance } from 'i18next';
import { act, useSyncExternalStore } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { commands, page, userEvent } from 'vitest/browser';

import type * as EngineStoreHooks from './engineStoreHooks';

const harness = vi.hoisted(() => ({
  dragData: null as unknown,
  extensions: null as ExtensionRegistry | null,
  engine: null as CanvasEngine | null,
  isEngineLive: true,
  operation: null as unknown,
  store: null as ReturnType<typeof createWorkbenchStore> | null,
  uploadImage: vi.fn<(file: File, boardId: string) => Promise<GalleryImage>>(),
}));
const activeProject = () => harness.store!.getState().projects[0]!;

vi.mock('@dnd-kit/core', async (importOriginal) => {
  const actual = await importOriginal<typeof DndKitCore>();
  return {
    ...actual,
    useDndContext: () => {
      const context = actual.useDndContext();
      return harness.dragData === null
        ? context
        : { ...context, active: { data: { current: harness.dragData }, id: 'gallery-drag' } };
    },
    useDndMonitor: () => undefined,
  };
});
vi.mock('@features/gallery/data/backend', async (importOriginal) => ({
  ...(await importOriginal<typeof GalleryBackend>()),
  listGalleryBoards: () => Promise.resolve([]),
  uploadGalleryImage: (file: File, boardId: string) => harness.uploadImage(file, boardId),
}));
vi.mock('@workbench/canvas-operations/api', async (importOriginal) => ({
  ...(await importOriginal<typeof CanvasOperationsApi>()),
  getCanvasEngine: (projectId: string) =>
    harness.isEngineLive && harness.engine?.projectId === projectId ? harness.engine : undefined,
}));
vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContextModule>()),
  useActiveProjectId: () => activeProject().id,
  useActiveProjectSelector: (selector: (project: Project) => unknown) => {
    const project = useSyncExternalStore(harness.store!.subscribe, activeProject, activeProject);
    return selector(project);
  },
  useWorkbenchExtensions: () => harness.extensions,
  useOptionalWorkbenchExtensions: () => harness.extensions,
  useWidgetValuesSelector: (widgetId: string, selector: (values: Record<string, unknown>) => unknown) =>
    selector(
      Object.values(activeProject().widgetInstances).find((instance) => instance.typeId === widgetId)?.state.values ??
        {}
    ),
  useOptionalWorkbenchCommands: () => harness.store!.commands,
  useWorkbenchCommands: () => harness.store!.commands,
  useWorkbenchQueries: () => harness.store!.queries,
  useWorkbenchSubscription: () => harness.store!.subscribe,
}));
vi.mock('@workbench/hotkeys/firstPartyCommands', () => ({ useRegisterFirstPartyCommands: () => {} }));
vi.mock('@workbench/projects/api', async (importOriginal) => ({
  ...(await importOriginal<typeof ProjectsApi>()),
  setClientStateValue: async () => {},
}));
vi.mock('@workbench/useCanvasProjectMutationDispatch', () => ({
  useCanvasProjectMutationDispatch: () => () => undefined,
}));
vi.mock('./engineStoreHooks', async (importOriginal) => ({
  ...(await importOriginal<typeof EngineStoreHooks>()),
  useCanvasOperation: () => harness.operation,
}));
vi.mock('@workbench/canvas-operations/react', () => ({ useCanvasEngine: () => harness.engine }));
vi.mock('./useCanvasGallerySave', () => ({
  useCanvasGallerySave: () => ({ isSaving: false, save: () => undefined }),
}));
vi.mock('./useStagedResultGallerySave', () => ({
  useStagedResultGallerySave: () => ({ isSaving: false, save: () => Promise.resolve() }),
}));
vi.mock('./useCreateFromBbox', () => ({
  useCreateFromBbox: () => ({ createFromBbox: () => undefined, isCreating: false }),
}));
vi.mock('./CanvasCreateFromBboxSubmenu', () => ({ CanvasCreateFromBboxSubmenu: () => null }));
vi.mock('./CanvasGlobalContextMenu', () => ({
  CanvasGlobalContextMenu: () => (
    <div aria-label="Canvas fallback" role="menu">
      Canvas actions
    </div>
  ),
}));
vi.mock('./CanvasSaveToGallerySubmenu', () => ({ CanvasSaveToGallerySubmenu: () => null }));
vi.mock('./CanvasSurface', () => ({ CanvasSurface: () => null }));
vi.mock('./MissingFontsDialog', () => ({ MissingFontsDialog: () => null }));
vi.mock('./ToolStrip', () => ({ ToolStrip: () => null }));
vi.mock('@workbench/widgets/layers/LayerContextMenu', () => ({ CanvasLayerContextMenu: () => null }));

import { CanvasWidgetView } from './CanvasWidgetView';

const i18n = createInstance();
let root: Root | null = null;
let host: HTMLDivElement | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeAll(async () => {
  const translation = await fetch('/locales/en.json').then((response) => response.json());
  await i18n.use(initReactI18next).init({ initAsync: false, lng: 'en', resources: { en: { translation } } });
});

afterEach(async () => {
  await act(() => root?.unmount());
  await act(() => i18n.changeLanguage('en'));
  host?.remove();
  harness.engine?.lifecycle.dispose();
  if (harness.extensions) {
    await patchWorkbenchPreferences(DEFAULT_PREFERENCES);
    accountLifecycle.invalidate();
  }
  Object.assign(harness, {
    dragData: null,
    engine: null,
    extensions: null,
    isEngineLive: true,
    operation: null,
    store: null,
  });
  harness.uploadImage.mockReset();
  root = null;
  host = null;
});

const candidate: CanvasStagingCandidateContract = {
  height: 64,
  imageName: 'left-eye.png',
  imageUrl: '/left-eye.png',
  placement: { height: 64, opacity: 1, width: 64, x: 0, y: 0 },
  queuedAt: '2026-07-16T00:00:00.000Z',
  sourceBackendItemId: 1,
  sourceQueueItemId: 'batch-eyes',
  thumbnailUrl: '/left-eye-thumb.png',
  width: 64,
};

const renderView = async (
  status: QueueItem['status'],
  {
    beforeRender,
    domRaster = false,
    hostWidth = 1000,
    hotkeys = false,
    nodes,
    selectedLayer,
    staged = true,
  }: {
    beforeRender?: (engine: CanvasEngine) => void;
    domRaster?: boolean;
    hostWidth?: number;
    hotkeys?: boolean;
    nodes?: CanvasNodeContract[];
    selectedLayer?: CanvasLayerContract;
    staged?: boolean;
  } = {}
) => {
  const initial = createInitialWorkbenchState();
  const created = initial.projects[0]!;
  const base = selectedLayer
    ? {
        ...created,
        canvas: {
          ...created.canvas,
          document: {
            ...created.canvas.document,
            selectedLayerId: selectedLayer.id,
            stacks: stacksFrom(nodes ?? [selectedLayer]),
          },
        },
      }
    : created;
  const batch = {
    backendBatchId: 'batch-eyes-backend',
    backendItemIds: [1, 2, 3],
    cancellable: true,
    completedBackendItemIds: status === 'completed' ? [1, 2, 3] : [1],
    id: 'batch-eyes',
    snapshot: {
      canvas: { document: base.canvas.document, documentRevision: base.canvas.documentRevision },
      destination: 'canvas',
      presentation: { batchCount: 3, height: 64, width: 64 },
      sourceId: 'canvas',
      submittedAt: '2026-07-16T00:00:00.000Z',
    },
    status,
  } as unknown as QueueItem;
  const store = createWorkbenchStore({ ...initial, projects: [{ ...base, queue: { items: [batch] } }] });
  const projectId = store.getState().activeProjectId;
  if (staged) {
    store.commands.canvas.appendStagingCandidate({ candidate, projectId });
    store.commands.canvas.apply(projectId, { imageIndex: 0, type: 'setStagedImageIndex' });
  }
  harness.store = store;
  harness.engine = createCanvasEngine({
    backend: domRaster ? createDomRasterBackend() : createTestStubRasterBackend(),
    ensureProjectOnServer: () => Promise.resolve(),
    imageResolver: () => Promise.resolve(new Blob()),
    mutationPort: createCanvasProjectMutationPort(store, projectId),
    projectId,
    reportError: () => undefined,
  });
  beforeRender?.(harness.engine);
  host = document.createElement('div');
  host.style.height = '600px';
  host.style.width = `${hostWidth}px`;
  document.body.append(host);
  root = createRoot(host);
  const commandHandlers = new Map<string, () => void>();
  const extensions = hotkeys ? createExtensionRegistry() : null;
  harness.extensions = extensions;
  const source = { instanceId: 'canvas', projectId, region: 'center', typeId: 'canvas' } as const;
  const createRuntime = (extensions: ExtensionRegistry | null) => ({
    commands: {
      register: (command: WidgetCommandContribution) => {
        commandHandlers.set(command.id, command.handler);
        const dispose = extensions?.commands.register({ ...command, source });
        return () => {
          commandHandlers.delete(command.id);
          dispose?.();
        };
      },
    },
    hotkeys: {
      register: (hotkey: WidgetHotkeyContribution) =>
        extensions?.hotkeys.register({ ...hotkey, source }) ?? (() => undefined),
    },
  });
  const queryClient = new QueryClient();
  const render = (extensions: ExtensionRegistry | null) => {
    const runtime = createRuntime(extensions);
    return act(() =>
      root!.render(
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <QueryClientProvider client={queryClient}>
              <WorkbenchFocusProvider>
                {hotkeys ? <WorkbenchHotkeyRuntime /> : null}
                <Box
                  h="full"
                  data-hotkey-widget-instance-id="canvas"
                  data-hotkey-widget-region="center"
                  data-hotkey-widget-type-id="canvas"
                >
                  <CanvasWidgetView
                    {...({ instance: activeProject().widgetInstances.canvas, runtime } as unknown as WidgetViewProps)}
                  />
                </Box>
              </WorkbenchFocusProvider>
            </QueryClientProvider>
          </ChakraProvider>
        </I18nextProvider>
      )
    );
  };
  await render(extensions);
  return {
    commandHandlers,
    extensions,
    projectId,
    store,
    replaceRuntime: async () => {
      const next = createExtensionRegistry();
      harness.extensions = next;
      await render(next);
      return next;
    },
  };
};

describe('mounted Canvas hotkeys', () => {
  it('creates one layer and undo step per press, suppresses held repeats, and keeps nudges repeating', async () => {
    accountLifecycle.activate('canvas-hotkeys-test');
    await patchWorkbenchPreferences({
      customHotkeys: { 'canvas.newRasterLayer': ['alt+shift+n'] },
    });
    await renderView('completed', { hotkeys: true, staged: false });
    const engine = harness.engine!;
    const target = host!.querySelector<HTMLElement>('[data-hotkey-widget-instance-id="canvas"]')!;
    const document = () => engine.document.model()!.document;
    const layerCount = () => document().stacks.raster.length;
    const initialCount = layerCount();
    const press = (repeat = false) =>
      act(() => {
        target.dispatchEvent(
          new KeyboardEvent('keydown', {
            altKey: true,
            bubbles: true,
            cancelable: true,
            code: 'KeyN',
            shiftKey: true,
            key: 'N',
            repeat,
          })
        );
      });

    await press();
    expect(layerCount()).toBe(initialCount + 1);
    expect(engine.history.getEntries().past).toEqual(['Raster layer']);
    await press(true);
    await press(true);
    expect(layerCount()).toBe(initialCount + 1);
    expect(engine.history.getEntries().past).toEqual(['Raster layer']);
    await act(() => {
      target.dispatchEvent(
        new KeyboardEvent('keyup', { altKey: true, bubbles: true, code: 'KeyN', key: 'N', shiftKey: true })
      );
    });
    await press();
    expect(layerCount()).toBe(initialCount + 2);
    expect(engine.history.getEntries().past).toEqual(['Raster layer', 'Raster layer']);

    await act(() => engine.history.undo());
    expect(layerCount()).toBe(initialCount + 1);
    await act(() => engine.history.redo());
    expect(layerCount()).toBe(initialCount + 2);
    const selected = document().stacks.raster.find((layer) => layer.id === document().selectedLayerId)!;
    if (selected.type !== 'raster') {
      throw new Error('Expected the new raster layer to be selected');
    }
    const initialX = selected.transform.x;
    for (const repeat of [false, true, true]) {
      await act(() =>
        target.dispatchEvent(
          new KeyboardEvent('keydown', {
            bubbles: true,
            cancelable: true,
            code: 'ArrowRight',
            key: 'ArrowRight',
            repeat,
          })
        )
      );
    }
    expect(document().stacks.raster.find((layer) => layer.id === selected.id)).toMatchObject({
      transform: { x: initialX + 3 },
    });
  });
});

describe('Canvas hotkey registration lifetime', () => {
  it('releases and refreshes locale and runtime registrations without discarding Canvas history', async () => {
    accountLifecycle.activate('canvas-hotkey-lifetime-test');
    const { commandHandlers, extensions, replaceRuntime } = await renderView('completed', {
      hotkeys: true,
      staged: false,
    });
    const engine = harness.engine!;
    await act(() => commandHandlers.get('canvas.newRasterLayer')!());
    expect(engine.history.getEntries().past).toEqual(['Raster layer']);
    const hotkey = (registry: ExtensionRegistry) =>
      registry.stores.hotkeys.list().find(({ id }) => id === 'canvas.newRasterLayer');
    expect(hotkey(extensions!)?.title).toBe('New raster layer');
    i18n.addResource('en-GB', 'translation', 'widgets.canvas.commands.newRasterLayer', 'Create Raster Layer');
    await act(() => i18n.changeLanguage('en-GB'));
    await expect.poll(() => hotkey(extensions!)?.title).toBe('Create Raster Layer');
    expect(extensions!.stores.hotkeys.list().filter(({ id }) => id === 'canvas.newRasterLayer')).toHaveLength(1);

    const next = await replaceRuntime();
    expect(extensions!.stores.commands.list()).toEqual([]);
    expect(extensions!.stores.hotkeys.list()).toEqual([]);
    expect(hotkey(next)).toMatchObject({ ignoreRepeat: true, title: 'Create Raster Layer' });
    expect(harness.engine).toBe(engine);
    expect(engine.history.getEntries().past).toEqual(['Raster layer']);
    await act(() => root!.unmount());
    root = null;
    expect(next.stores.commands.list()).toEqual([]);
    expect(next.stores.hotkeys.list()).toEqual([]);
  });
});

describe('Canvas spline context menu', () => {
  const renderSpline = async (selectPath = true) => {
    const layer = createVectorLayer({ id: 'vector', name: 'Guides' });
    layer.paths = [
      {
        id: 'selected-path',
        name: null,
        isClosed: false,
        points: [
          { anchor: { x: 300, y: 300 }, inHandle: null, outHandle: null, type: 'corner' },
          { anchor: { x: 400, y: 300 }, inHandle: null, outHandle: null, type: 'corner' },
        ],
      },
      {
        id: 'other-path',
        name: null,
        isClosed: false,
        points: [
          { anchor: { x: 700, y: 450 }, inHandle: null, outHandle: null, type: 'corner' },
          { anchor: { x: 800, y: 450 }, inHandle: null, outHandle: null, type: 'corner' },
        ],
      },
    ];
    await renderView('completed', {
      selectedLayer: layer,
      staged: false,
    });
    await act(() => {
      harness.engine!.tools.setTool('spline');
      if (selectPath) {
        expect(harness.engine!.layers.vectorEdit.targetContextMenu({ x: 350, y: 300 })).toBe(true);
      }
    });
    const owner = host!.querySelector<HTMLDivElement>('[data-canvas-context-menu-owner]')!;
    owner.tabIndex = -1;
    owner.focus();
    return { engine: harness.engine!, owner };
  };

  it.each(['{Shift>}{F10}{/Shift}', '{ContextMenu}'])(
    'opens the active path away from the corner with %s and executes an action with keys',
    async (keys) => {
      const { engine, owner } = await renderSpline();
      await act(() => userEvent.keyboard(keys));
      await expect.element(page.getByRole('menuitem', { name: 'Trace Path', exact: true })).toBeVisible();
      expect(engine.interaction.get('vectorEditSession')?.activePathId).toBe('selected-path');
      await act(() => userEvent.keyboard('{Escape}'));
      await vi.waitFor(() => expect(document.activeElement).toBe(owner));
      await act(() => userEvent.keyboard(keys));
      await expect.element(page.getByRole('menuitem', { name: 'Trace Path', exact: true })).toBeVisible();
      // Copy Path is immediately before Delete Path; navigate the actual menu without a pointer.
      await userEvent.keyboard('{End}{ArrowUp}');
      await expect
        .element(page.getByRole('menuitem', { name: 'Copy Path', exact: true }))
        .toHaveAttribute('data-highlighted');
      await act(() => userEvent.keyboard('{Enter}'));
      await vi.waitFor(() => expect(engine.interaction.get('vectorEditSession')?.paths).toHaveLength(3));
      expect(engine.interaction.get('vectorEditSession')?.paths[1]?.points).toEqual(
        engine.interaction.get('vectorEditSession')?.paths[0]?.points
      );
    }
  );

  it('keeps the active path when a keyboard keydown is followed by a native contextmenu event', async () => {
    const { engine, owner } = await renderSpline();
    const bounds = owner.getBoundingClientRect();
    const key = new KeyboardEvent('keydown', { key: 'ContextMenu', bubbles: true, cancelable: true });
    await act(() => owner.dispatchEvent(key));
    expect(key.defaultPrevented).toBe(true);
    await expect.element(page.getByRole('menuitem', { name: 'Trace Path', exact: true })).toBeVisible();

    // Chromium emits this after the key, with coordinates unrelated to the selected path.
    const nativeMenu = new PointerEvent('contextmenu', {
      bubbles: true,
      cancelable: true,
      button: -1,
      pointerType: 'mouse',
      clientX: bounds.left + 750,
      clientY: bounds.top + 450,
    });
    await act(() => owner.dispatchEvent(nativeMenu));

    expect(nativeMenu.defaultPrevented).toBe(true);
    expect(engine.interaction.get('vectorEditSession')?.activePathId).toBe('selected-path');
    await expect.element(page.getByRole('menuitem', { name: 'Trace Path', exact: true })).toBeVisible();
  });

  it('keeps pointer targeting and native menus in inline editors, without handling other widgets', async () => {
    const { engine, owner } = await renderSpline();
    const bounds = owner.getBoundingClientRect();
    await act(() =>
      owner.dispatchEvent(
        new MouseEvent('contextmenu', {
          bubbles: true,
          cancelable: true,
          button: 2,
          clientX: bounds.left + 750,
          clientY: bounds.top + 450,
        })
      )
    );
    await expect.element(page.getByRole('menuitem', { name: 'Trace Path', exact: true })).toBeVisible();
    expect(engine.interaction.get('vectorEditSession')?.activePathId).toBe('other-path');
    await act(() => userEvent.keyboard('{Escape}'));
    await expect.element(page.getByRole('menuitem', { name: 'Trace Path', exact: true })).not.toBeInTheDocument();

    const editor = document.createElement('textarea');
    owner.append(editor);
    editor.focus();
    const key = new KeyboardEvent('keydown', { key: 'F10', shiftKey: true, bubbles: true, cancelable: true });
    const nativeMenu = new MouseEvent('contextmenu', { bubbles: true, cancelable: true });
    await act(() => {
      editor.dispatchEvent(key);
      editor.dispatchEvent(nativeMenu);
    });
    expect(key.defaultPrevented).toBe(false);
    expect(nativeMenu.defaultPrevented).toBe(false);
    await expect.element(page.getByRole('menuitem', { name: 'Trace Path', exact: true })).not.toBeInTheDocument();

    const elsewhere = document.createElement('button');
    elsewhere.textContent = 'Other widget';
    host!.append(elsewhere);
    owner.focus();
    await act(() => userEvent.keyboard('{Shift>}{F10}{/Shift}'));
    await expect.element(page.getByRole('menuitem', { name: 'Trace Path', exact: true })).toBeVisible();
    await page.getByRole('button', { name: 'Other widget' }).click();
    await vi.waitFor(() => expect(document.activeElement).toBe(elsewhere));
    await act(() => userEvent.keyboard('{Shift>}{F10}{/Shift}'));
    await expect.element(page.getByRole('menuitem', { name: 'Trace Path', exact: true })).not.toBeInTheDocument();
  });

  it('uses the Canvas fallback when no path is active, including native keyboard contextmenu events', async () => {
    const { owner } = await renderSpline(false);
    await act(() => owner.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true, button: 0 })));
    await expect.element(page.getByRole('menu', { name: 'Canvas fallback' })).toBeVisible();
    await expect.element(page.getByRole('menuitem', { name: 'Trace Path', exact: true })).not.toBeInTheDocument();
  });
});

describe('CanvasWidgetView accept and stop', () => {
  it('accepts the result, then cancels its batch through the workbench queue command with the accept notice', async () => {
    const { store } = await renderView('running');

    await page.getByRole('button', { exact: true, name: 'Accept to layer' }).click();

    const project = store.getState().projects[0]!;
    expect(project.canvas.document.stacks.raster).toHaveLength(1);
    expect(project.queue.items[0]).toMatchObject({ cancellationPending: true, id: 'batch-eyes', status: 'cancelled' });
    expect(store.getState().notifications[0]).toMatchObject({
      message: 'Stopping the rest of its batch, which shows as cancelled in the queue.',
      projectId: project.id,
      title: 'Result accepted',
    });
  });
});

describe('CanvasWidgetView editing while results are staged', () => {
  it('lands a brush stroke on the selected layer while a result is staged and the batch runs', async () => {
    const paint = layerContract('paint', 'raster');
    await renderView('running', { domRaster: true, selectedLayer: paint });
    const engine = harness.engine!;
    const screen = document.createElement('canvas');
    const overlay = document.createElement('canvas');
    for (const canvas of [screen, overlay]) {
      Object.assign(canvas.style, { height: '300px', left: '0', position: 'fixed', top: '0', width: '300px' });
      host!.append(canvas);
    }
    await act(() => engine.surface.attach(screen, overlay));
    await act(() => engine.tools.setTool('brush'));
    const pointer = (type: string, x: number, buttons: number) =>
      new PointerEvent(type, { bubbles: true, buttons, clientX: x, clientY: 150, pointerId: 1, pointerType: 'mouse' });

    await act(() => {
      overlay.dispatchEvent(pointer('pointerdown', 100, 1));
      overlay.dispatchEvent(pointer('pointermove', 150, 1));
      overlay.dispatchEvent(pointer('pointermove', 200, 1));
      overlay.dispatchEvent(pointer('pointerup', 200, 0));
    });

    await expect.poll(() => engine.history.getEntries().past).toEqual(['Brush stroke']);
    expect(activeProject().canvas.stagingArea.pendingImages.map((image) => image.imageName)).toEqual(['left-eye.png']);
    engine.surface.detach();
  });
});

describe('CanvasWidgetView staged results can come back', () => {
  const second: CanvasStagingCandidateContract = {
    ...candidate,
    imageName: 'right-eye.png',
    imageUrl: '/right-eye.png',
    sourceBackendItemId: 2,
    thumbnailUrl: '/right-eye-thumb.png',
  };
  const undoOf = (store: ReturnType<typeof createWorkbenchStore>) => {
    const action = store.getState().notifications[0]?.actions?.find((entry) => entry.kind === 'undo');
    return action?.kind === 'undo' ? () => runNotificationUndo(action.undoId) : null;
  };
  const staged = (store: ReturnType<typeof createWorkbenchStore>) =>
    store.getState().projects[0]!.canvas.stagingArea.pendingImages.map((image) => image.imageName);

  it('keeps keyboard focus in the staging bar after Delete discards the selected result', async () => {
    const { store } = await renderView('running');
    const selected = page.getByRole('button', { name: 'Select staged candidate 1' }).element() as HTMLElement;
    await act(() => selected.focus());

    await act(() => userEvent.keyboard('{Delete}'));

    expect(staged(store)).toEqual([]);
    await expect.poll(() => document.activeElement?.getAttribute('aria-current')).toBe('true');
    expect(document.activeElement?.hasAttribute('data-staging-thumbnail')).toBe(true);
  });

  it('discards a result with an Undo that puts it back for review', async () => {
    const { store } = await renderView('completed');

    await page.getByRole('button', { exact: true, name: 'Discard' }).click();
    expect(staged(store)).toEqual([]);
    expect(store.getState().notifications[0]).toMatchObject({ title: 'Result discarded' });

    await act(() => undoOf(store)!());
    expect(staged(store)).toEqual(['left-eye.png']);
  });

  it('discards all, stops the rest of the generation, keeps a restore point and offers the results back', async () => {
    const { store } = await renderView('running', {
      beforeRender: () =>
        harness.store!.commands.canvas.appendStagingCandidate({
          candidate: second,
          projectId: harness.store!.getState().activeProjectId,
        }),
    });

    await page.getByRole('button', { exact: true, name: 'Discard all' }).click();
    const project = store.getState().projects[0]!;
    expect(staged(store)).toEqual([]);
    expect(project.queue.items[0]).toMatchObject({ status: 'cancelled' });
    expect(project.canvas.snapshots.at(-1)).toMatchObject({ kind: 'discard-results', name: 'Discarded results' });
    expect(store.getState().notifications[0]).toMatchObject({
      message: 'Also stopped the queue item still running. Undo brings back the results, not the stopped item.',
      title: 'Discarded 2 results',
    });

    await act(() => undoOf(store)!());
    expect(staged(store)).toEqual(['left-eye.png', 'right-eye.png']);
    expect(store.getState().projects[0]!.canvas.snapshots).toEqual([]);
  });
});

describe('CanvasWidgetView clipboard shortcuts', () => {
  // A plain-HTTP origin is not a secure context, so the browser exposes no async Clipboard API.
  const withoutClipboard = () => vi.spyOn(Navigator.prototype, 'clipboard', 'get').mockReturnValue(undefined!);

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('says why copy and cut fail, and keeps the pixels a failed cut would erase', async () => {
    const { commandHandlers, store } = await renderView('completed', { staged: false });
    withoutClipboard();
    vi.spyOn(harness.engine!.selection, 'exportSelectionBlob').mockResolvedValue(new Blob([], { type: 'image/png' }));
    vi.spyOn(harness.engine!.selection, 'admitSelectionErase').mockReturnValue(true);
    const erase = vi.spyOn(harness.engine!.selection, 'eraseSelection');

    await act(() => commandHandlers.get('canvas.copySelection')!());
    await vi.waitFor(() =>
      expect(store.getState().notifications[0]).toMatchObject({
        kind: 'error',
        message: 'Copying needs HTTPS or localhost',
        title: 'Could not copy the selection to the clipboard',
      })
    );

    await act(() => commandHandlers.get('canvas.cutSelection')!());
    await vi.waitFor(() => expect(store.getState().notifications[0]?.occurrenceCount).toBe(2));
    expect(erase).not.toHaveBeenCalled();
  });
});

describe('CanvasWidgetView control layer overlay', () => {
  const emptyControlLayer = createControlLayer({ id: 'control-1', name: 'Control Layer 1' });
  const hint = () => page.getByRole('note').filter({ hasText: 'Add an image' });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('points an empty control layer at its image sources until the brush or eraser takes over', async () => {
    await renderView('completed', { selectedLayer: emptyControlLayer, staged: false });

    await expect.element(hint()).toBeVisible();
    await expect
      .element(hint().getByText('Drag one onto Add as control layer, or pick one in Properties', { exact: true }))
      .toBeVisible();

    await act(() => harness.engine!.tools.setTool('brush'));
    await expect.element(hint()).not.toBeInTheDocument();

    await act(() => harness.engine!.tools.setTool('eraser'));
    await expect.element(hint()).not.toBeInTheDocument();

    await act(() => harness.engine!.tools.setTool('move'));
    await expect.element(hint()).toBeVisible();
  });

  const depthLayer: CanvasControlLayerContract = {
    ...emptyControlLayer,
    controlType: 'depth',
    detection: { settings: {}, type: 'depth_anything_depth_estimation' },
    source: { image: { height: 64, imageName: 'photo.png', width: 96 }, type: 'image' },
  };

  const renderWithStatuses = async (layer: CanvasControlLayerContract = depthLayer) => {
    const statuses = createControlDetectionStatusStore();
    await renderView('completed', {
      beforeRender: (engine) => {
        const operations = getCanvasOperations(engine);
        vi.spyOn(operations, 'getControlDetectionStatus').mockImplementation(statuses.get);
        vi.spyOn(operations, 'subscribeControlDetectionStatus').mockImplementation(statuses.subscribe);
      },
      selectedLayer: layer,
      staged: false,
    });
    return statuses;
  };

  const readyLayer: CanvasControlLayerContract = {
    ...depthLayer,
    filter: depthLayer.detection!,
    original: { image: { height: 64, imageName: 'photo.png', width: 96 }, type: 'image' },
    source: { bitmap: { height: 64, imageName: 'depth.png', width: 96 }, offset: { x: 0, y: 0 }, type: 'paint' },
  };
  const viewSwitch = () => page.getByRole('radiogroup', { name: 'Control layer view' });
  const viewOption = (name: 'Original' | 'Map' | 'Overlay') => viewSwitch().getByRole('radio', { name });

  it('switches a ready layer between its original, map and overlay, following the engine', async () => {
    await renderWithStatuses(readyLayer);
    const engine = harness.engine!;
    const setView = vi.spyOn(engine.previews, 'setControlLayerView');

    await expect.element(viewSwitch()).toBeVisible();
    await expect.element(viewOption('Map')).toBeChecked();
    expect(await auditAccessibility(viewSwitch().element())).toEqual([]);

    await act(() => userEvent.click(viewSwitch().getByText('Original', { exact: true })));
    expect(setView).toHaveBeenLastCalledWith('control-1', 'original');
    await expect.element(viewOption('Original')).toBeChecked();

    await act(() => engine.previews.setControlLayerView('control-1', 'map'));
    await expect.element(viewOption('Map')).toBeChecked();

    await act(async () => {
      (viewOption('Map').element() as HTMLElement).focus();
      await userEvent.keyboard('{ArrowRight}');
    });
    expect(setView).toHaveBeenLastCalledWith('control-1', 'overlay');
    await expect.element(viewOption('Overlay')).toBeChecked();
    await expect.element(viewOption('Overlay')).toHaveFocus();
  });

  it.each<[ControlLayerViewOutcome, string]>([
    ['failed', "Couldn't show the original image"],
    ['over-budget', 'Not enough canvas memory to show the original image'],
  ])('shows the request at once, then returns to Map with a notice when it is %s', async (outcome, title) => {
    const { store } = await renderView('completed', { selectedLayer: readyLayer, staged: false });
    let settle!: (value: ControlLayerViewOutcome) => void;
    vi.spyOn(harness.engine!.previews, 'setControlLayerView').mockReturnValueOnce(
      new Promise((resolve) => {
        settle = resolve;
      })
    );

    await act(() => userEvent.click(viewSwitch().getByText('Original', { exact: true })));
    await expect.element(viewOption('Original')).toBeChecked();

    await act(() => settle(outcome));
    await expect.element(viewOption('Map')).toBeChecked();
    expect(store.getState().notifications[0]).toMatchObject({ kind: 'error', title });
  });
});

describe('CanvasWidgetView desktop file drops', () => {
  const imageFile = (name: string) =>
    new File([Uint8Array.from(atob(TINY_PNG_BASE64), (char) => char.charCodeAt(0))], name, { type: 'image/png' });
  const stored = (file: { name: string }): GalleryImage => ({
    boardId: 'none',
    height: 2,
    imageCategory: 'user',
    imageName: `stored-${file.name}`,
    imageUrl: `/images/stored-${file.name}`,
    queuedAt: '2026-11-10T00:00:00.000Z',
    sourceQueueItemId: 'upload',
    starred: false,
    thumbnailUrl: `/thumbnails/stored-${file.name}`,
    width: 2,
  });
  const surface = () => page.getByRole('region', { exact: true, name: 'Canvas surface' }).element();
  const zone = (name: string) => page.getByText(name, { exact: true });
  const titles = (store: Awaited<ReturnType<typeof renderView>>['store']) =>
    store.getState().notifications.map((notification) => notification.title);
  const rasterLayersOf = (projectId: string) =>
    harness.store!.getState().projects.find((project) => project.id === projectId)!.canvas.document.stacks.raster;
  // The first drop loads the upload code; a cold transform can take longer than `waitFor`'s default second.
  const LANDED = { timeout: 5_000 };

  afterEach(() => {
    vi.restoreAllMocks();
  });

  const centreOf = (element: Element) => {
    const rect = element.getBoundingClientRect();
    return { clientX: rect.left + rect.width / 2, clientY: rect.top + rect.height / 2 };
  };
  const pointOn = (destination: 'control' | 'raster') => {
    const rect = surface().getBoundingClientRect();
    return {
      x: rect.left + rect.width * (destination === 'raster' ? 0.25 : 0.75),
      y: rect.top + rect.height * 0.25,
    };
  };
  const nativeDrop = (destination: 'control' | 'raster', names: string[]) =>
    commands.nativeDrag({
      files: names.map((name) => ({ base64: TINY_PNG_BASE64, name })),
      frameWidth: window.innerWidth,
      points: [pointOn(destination)],
    });
  const send = async (type: string, transfer: DataTransfer, init: DragEventInit = {}, target: Element = surface()) => {
    const event = new DragEvent(type, { bubbles: true, cancelable: true, dataTransfer: transfer, ...init });
    await act(() => target.dispatchEvent(event));
    return event;
  };
  const transferOf = (files: File[]) => {
    const transfer = new DataTransfer();
    for (const file of files) {
      transfer.items.add(file);
    }
    return transfer;
  };
  const dropOn = async (zoneName: string, files: File[]) => {
    const transfer = transferOf(files);
    await send('dragenter', transfer);
    await expect.element(zone(zoneName)).toBeVisible();
    const at = centreOf(zone(zoneName).element());
    await send('dragover', transfer, at);
    return send('drop', transfer, at);
  };

  it('uploads image files dropped from the desktop on a zone and adds them as it says, at the frame', async () => {
    harness.uploadImage.mockImplementation((file) => Promise.resolve(stored(file)));
    const { projectId, store } = await renderView('completed', { hostWidth: window.innerWidth, staged: false });
    const before = rasterLayersOf(projectId).length;

    await nativeDrop('raster', ['one.png', 'two.png']);

    await vi.waitFor(() => expect(rasterLayersOf(projectId)).toHaveLength(before + 2), LANDED);
    expect(harness.uploadImage.mock.calls.map(([file, boardId]) => [file.name, boardId])).toEqual([
      ['one.png', 'none'],
      ['two.png', 'none'],
    ]);
    const { bbox } = activeProject().canvas.document;
    expect(
      rasterLayersOf(projectId)
        .slice(-2)
        .map((layer) =>
          layer.type === 'raster' ? { source: layer.source, x: layer.transform.x, y: layer.transform.y } : null
        )
    ).toEqual([
      { source: { image: { height: 2, imageName: 'stored-one.png', width: 2 }, type: 'image' }, x: bbox.x, y: bbox.y },
      { source: { image: { height: 2, imageName: 'stored-two.png', width: 2 }, type: 'image' }, x: bbox.x, y: bbox.y },
    ]);
    await vi.waitFor(() =>
      expect(titles(store)).toEqual(['Added 2 images as canvas layers', expect.stringMatching(/^Uploaded 2 /)])
    );
    expect(zone('Add as raster layer').elements()).toHaveLength(0);
  });

  it('refuses a second drop while the first is still uploading, rather than adding everything twice', async () => {
    let finishUpload!: () => void;
    harness.uploadImage.mockImplementation(
      (file) =>
        new Promise((resolve) => {
          finishUpload = () => resolve(stored(file));
        })
    );
    const { projectId, store } = await renderView('completed', { hostWidth: window.innerWidth, staged: false });
    const before = rasterLayersOf(projectId).length;

    await dropOn('Add as raster layer', [imageFile('one.png')]);
    await vi.waitFor(() => expect(harness.uploadImage).toHaveBeenCalledTimes(1), LANDED);
    await dropOn('Add as raster layer', [imageFile('again.png')]);

    expect(store.getState().notifications[0]).toMatchObject({
      kind: 'info',
      title: 'The files you dropped are still uploading. Try again when they are done.',
    });
    await act(() => finishUpload());
    await vi.waitFor(() => expect(rasterLayersOf(projectId)).toHaveLength(before + 1));
    expect(harness.uploadImage).toHaveBeenCalledTimes(1);
  });

  it('lands an upload that finishes after a project switch on the project it was dropped on, keeping its edits', async () => {
    let finishUpload!: () => void;
    harness.uploadImage.mockImplementation(
      (file) =>
        new Promise((resolve) => {
          finishUpload = () => resolve(stored(file));
        })
    );
    const { projectId, store } = await renderView('completed', { hostWidth: window.innerWidth, staged: false });

    await dropOn('Add as raster layer', [imageFile('late.png')]);
    await vi.waitFor(() => expect(harness.uploadImage).toHaveBeenCalledTimes(1), LANDED);
    await act(() =>
      store.commands.canvas.apply(projectId, {
        document: {
          ...activeProject().canvas.document,
          stacks: stacksFrom([...rasterLayersOf(projectId), layerContract('kept', 'raster')]),
        },
        type: 'replaceCanvasDocument',
      })
    );
    const switchedTo = await act(() => store.commands.projects.create());
    await act(() => finishUpload());

    await vi.waitFor(() =>
      expect(rasterLayersOf(projectId).map((layer) => layer.type === 'raster' && layer.source)).toContainEqual({
        image: { height: 2, imageName: 'stored-late.png', width: 2 },
        type: 'image',
      })
    );
    expect(rasterLayersOf(projectId).map((layer) => layer.id)).toContain('kept');
    expect(store.getState().activeProjectId).toBe(switchedTo.id);
    expect(rasterLayersOf(switchedTo.id)).toHaveLength(0);
  });

  it('ends the drag when it leaves the canvas, leaves the window, is cancelled or is dropped elsewhere', async () => {
    await renderView('completed', { hostWidth: window.innerWidth, staged: false });
    const transfer = transferOf([imageFile('one.png')]);
    const zonesShown = () => zone('Add as inpaint mask').elements().length > 0;

    await send('dragenter', transfer);
    expect(zonesShown()).toBe(true);
    await send('dragleave', transfer, { relatedTarget: document.body });
    expect(zonesShown()).toBe(false);

    const doomed = document.createElement('div');
    surface().append(doomed);
    await send('dragenter', transfer);
    await send('dragenter', transfer, { relatedTarget: surface() }, doomed);
    doomed.remove();
    await send('dragleave', transfer, { relatedTarget: null });
    await vi.waitFor(() => expect(zonesShown()).toBe(false));

    await send('dragenter', transfer);
    await act(() => document.body.dispatchEvent(new DragEvent('drop', { bubbles: true, dataTransfer: transfer })));
    expect(zonesShown()).toBe(false);
    await send('dragenter', transfer);
    await act(() => window.dispatchEvent(new DragEvent('dragend')));
    expect(zonesShown()).toBe(false);
  });

  it('takes no drop while an operation holds the canvas, and says why while files or gallery images hover', async () => {
    harness.operation = { identity: { kind: 'filter' }, status: 'active' };
    await renderView('completed', { hostWidth: window.innerWidth, staged: false });
    const busy = () => page.getByText('The canvas is busy. Try again when the current action finishes.');

    await send('dragenter', transferOf([imageFile('one.png')]));
    await expect.element(busy()).toBeVisible();
    expect(zone('Add as raster layer').elements()).toHaveLength(0);
    await send('dragleave', transferOf([imageFile('one.png')]), { relatedTarget: document.body });

    await nativeDrop('raster', ['one.png']);
    await expect.element(busy()).not.toBeInTheDocument();
    await new Promise((resolve) => {
      setTimeout(resolve, 500);
    });
    expect(harness.uploadImage).not.toHaveBeenCalled();
  });
});

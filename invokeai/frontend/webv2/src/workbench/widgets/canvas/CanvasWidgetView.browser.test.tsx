import type * as DndKitCore from '@dnd-kit/core';
import type {
  CanvasControlLayerContract,
  CanvasLayerContract,
  CanvasNodeContract,
  ControlLayerView,
  ControlLayerViewOutcome,
} from '@workbench/canvas-engine/api';
import type { CanvasStagingCandidateContract } from '@workbench/canvas-engine/contracts';
import type { CanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import type { Project } from '@workbench/projectContracts';
import type { WorkbenchQueueItem as QueueItem } from '@workbench/queueHistoryContracts';
import type { WidgetViewProps } from '@workbench/widgetContracts';
import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';

import { ChakraProvider } from '@chakra-ui/react';
import { galleryTransfers, type GalleryImage } from '@features/gallery';
import { auditAccessibility } from '@platform/browser/auditAccessibility.testing';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { createControlLayer } from '@workbench/canvas-engine/api';
import {
  groupContract,
  layerContract,
  stacksFrom,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { getCanvasOperations, getControlSourceSnapshot } from '@workbench/canvas-operations/api';
import { createControlDetectionStatusStore } from '@workbench/canvas-operations/controlDetectionStatusStore';
import { createCanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import { createCanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import { runNotificationUndo } from '@workbench/shell/notifications/undoRegistry';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import type * as EngineStoreHooks from './engineStoreHooks';

/** The view reads the real store and drives its real commands; only unrelated chrome is stubbed. */
const harness = vi.hoisted(() => ({
  /** Drag data dnd-kit reports as active; null leaves the real (idle) context. */
  dragData: null as unknown,
  engine: null as CanvasEngine | null,
  store: null as ReturnType<typeof createWorkbenchStore> | null,
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
vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContextModule>()),
  useActiveProjectId: () => activeProject().id,
  useActiveProjectSelector: (selector: (project: Project) => unknown) => selector(activeProject()),
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
vi.mock('@workbench/useCanvasProjectMutationDispatch', () => ({
  useCanvasProjectMutationDispatch: () => () => undefined,
}));
vi.mock('./engineStoreHooks', async (importOriginal) => ({
  ...(await importOriginal<typeof EngineStoreHooks>()),
  useCanvasOperation: () => null,
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
vi.mock('./CanvasGlobalContextMenu', () => ({ CanvasGlobalContextMenu: () => null }));
vi.mock('./CanvasImageDropOverlay', () => ({ CanvasImageDropOverlay: () => null }));
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
  host?.remove();
  harness.engine?.lifecycle.dispose();
  Object.assign(harness, { dragData: null, engine: null, store: null });
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

/**
 * Stages the first image of a three-image batch whose remaining images are `status`; unstaged, the surface is live.
 * `selectedLayer` starts the document with that layer selected, alone or among `nodes`; `beforeRender` sees the
 * engine before the view subscribes to it.
 */
const renderView = async (
  status: QueueItem['status'],
  {
    beforeRender,
    domRaster = false,
    nodes,
    selectedLayer,
    staged = true,
  }: {
    beforeRender?: (engine: CanvasEngine) => void;
    /** Real Canvas2D pixels, for a test that attaches a surface and draws. */
    domRaster?: boolean;
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
  host.style.width = '1000px';
  document.body.append(host);
  root = createRoot(host);
  const commandHandlers = new Map<string, () => void>();
  const runtime = {
    commands: {
      register: ({ handler, id }: { handler: () => void; id: string }) => {
        commandHandlers.set(id, handler);
        return () => commandHandlers.delete(id);
      },
    },
    hotkeys: { register: () => () => undefined },
  };
  await act(() =>
    root!.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <QueryClientProvider client={new QueryClient()}>
            <CanvasWidgetView
              {...({ instance: activeProject().widgetInstances.canvas, runtime } as unknown as WidgetViewProps)}
            />
          </QueryClientProvider>
        </ChakraProvider>
      </I18nextProvider>
    )
  );
  return { commandHandlers, projectId, store };
};

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

  it('accepts a finished batch under the plain label and cancels nothing', async () => {
    const { store } = await renderView('completed');

    await page.getByRole('button', { exact: true, name: 'Accept to layer' }).click();

    const project = store.getState().projects[0]!;
    expect(project.canvas.document.stacks.raster).toHaveLength(1);
    expect(project.queue.items[0]?.status).toBe('completed');
  });
});

describe('CanvasWidgetView editing while results are staged', () => {
  const background = layerContract('background', 'raster');

  it('leaves the tools to the user while the batch runs: nothing is parked on View', async () => {
    await renderView('running', { selectedLayer: background });

    await act(() => harness.engine!.tools.setTool('brush'));
    expect(harness.engine!.interaction.get('activeTool')).toBe('brush');
    await act(() => harness.engine!.tools.setTool('bbox'));
    expect(harness.engine!.interaction.get('activeTool')).toBe('bbox');
  });

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
    // Staging is untouched by the edit.
    expect(activeProject().canvas.stagingArea.pendingImages.map((image) => image.imageName)).toEqual(['left-eye.png']);
    engine.surface.detach();
  });

  it('outlines where each pending result lands, and keeps those rects when the frame moves', async () => {
    let setRegions: ReturnType<typeof vi.spyOn> | null = null;
    const { projectId, store } = await renderView('running', {
      beforeRender: (engine) => {
        setRegions = vi.spyOn(engine.previews, 'setStagedRegions');
      },
      selectedLayer: background,
    });
    const submitted = store.getState().projects[0]!.queue.items[0]!.snapshot.canvas.document.bbox;
    const pinned = [
      { rect: { height: 64, width: 64, x: 0, y: 0 }, selected: true },
      { rect: submitted, selected: false },
      { rect: submitted, selected: false },
    ];
    expect(setRegions!.mock.lastCall).toEqual([pinned]);

    await act(() =>
      store.commands.canvas.apply(projectId, {
        bbox: { ...submitted, x: submitted.x + 128, y: submitted.y + 64 },
        type: 'setCanvasBbox',
      })
    );
    expect(store.getState().projects[0]!.canvas.document.bbox.x).toBe(submitted.x + 128);
    // The feed rereads only on staging, revision or queue changes: a frame move publishes nothing.
    const calls = setRegions!.mock.calls.length;
    expect(setRegions!.mock.lastCall).toEqual([pinned]);
    await act(() => store.commands.canvas.apply(projectId, { imageIndex: 1, type: 'setStagedImageIndex' }));
    expect(setRegions!.mock.calls).toHaveLength(calls + 1);
    expect(setRegions!.mock.lastCall).toEqual([pinned.map((region, index) => ({ ...region, selected: index === 1 }))]);
    await act(() => store.commands.canvas.apply(projectId, { imageIndex: 0, type: 'setStagedImageIndex' }));

    // The next Invoke uses the moved frame; this result still lands at its own rect, on top of the stack.
    await page.getByRole('button', { exact: true, name: 'Accept to layer' }).click();
    const [accepted, existing] = store.getState().projects[0]!.canvas.document.stacks.raster;
    expect(existing).toMatchObject({ id: 'background' });
    expect(accepted).toMatchObject({ source: { image: { imageName: 'left-eye.png' } }, transform: { x: 0, y: 0 } });
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

  it('announces how many results are pending while the batch runs', async () => {
    await renderView('running');
    // One staged result and the two still generating.
    await expect.element(page.getByRole('status').getByText('Generating, 3 results pending')).toBeVisible();
  });

  it('announces the results still waiting for review once the batch is done', async () => {
    await renderView('completed');
    await expect.element(page.getByRole('status').getByText('1 result pending')).toBeVisible();
  });

  it('keeps keyboard focus in the staging bar after Delete discards the selected result', async () => {
    const { store } = await renderView('running');
    const selected = page.getByRole('button', { name: 'Select staged candidate 1' }).element() as HTMLElement;
    await act(() => selected.focus());

    await act(() => userEvent.keyboard('{Delete}'));

    expect(staged(store)).toEqual([]);
    // The result now selected (a placeholder still generating) takes focus, not the page.
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
    // The Undo used up the restore point of the same results.
    expect(store.getState().projects[0]!.canvas.snapshots).toEqual([]);
  });

  it('accepts one result and offers the ones it set aside back', async () => {
    const { store } = await renderView('completed', {
      beforeRender: () =>
        harness.store!.commands.canvas.appendStagingCandidate({
          candidate: second,
          projectId: harness.store!.getState().activeProjectId,
        }),
    });

    await page.getByRole('button', { exact: true, name: 'Accept to layer' }).click();
    expect(store.getState().projects[0]!.canvas.document.stacks.raster).toHaveLength(1);
    expect(staged(store)).toEqual([]);
    expect(store.getState().notifications[0]).toMatchObject({ title: 'Result accepted' });
    expect(store.getState().notifications[0]?.message).toMatch(/^Set aside 1 other result\./);
    // The button brings back the results set aside, not the accept, and says so.
    expect(store.getState().notifications[0]?.actions).toEqual([
      expect.objectContaining({ kind: 'undo', label: 'Review the other result' }),
    ]);

    await act(() => undoOf(store)!());
    expect(staged(store)).toEqual(['right-eye.png']);
    // The accepted layer stays; undoing it is the canvas's own step.
    expect(store.getState().projects[0]!.canvas.document.stacks.raster).toHaveLength(1);
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
    const erase = vi.spyOn(harness.engine!.selection, 'eraseSelection');

    await act(() => commandHandlers.get('canvas.copySelection')!());
    await vi.waitFor(() =>
      expect(store.getState().notifications[0]).toMatchObject({
        kind: 'error',
        message: 'Copying needs HTTPS or localhost',
        title: 'Could not copy the selection to the clipboard',
      })
    );

    // Cut fails the same way, so the store coalesces it into the same notice.
    await act(() => commandHandlers.get('canvas.cutSelection')!());
    await vi.waitFor(() => expect(store.getState().notifications[0]?.occurrenceCount).toBe(2));
    expect(erase).not.toHaveBeenCalled();
  });

  it('says why paste does nothing', async () => {
    const { commandHandlers, store } = await renderView('completed', { staged: false });
    withoutClipboard();

    await act(() => commandHandlers.get('canvas.pasteImage')!());

    await vi.waitFor(() =>
      expect(store.getState().notifications[0]).toMatchObject({
        kind: 'error',
        message: 'Pasting needs HTTPS or localhost',
        title: 'Could not paste an image from the clipboard',
      })
    );
  });

  it('reports a clipboard image that cannot be retrieved without blaming the address', async () => {
    const { commandHandlers, store } = await renderView('completed', { staged: false });
    const item = {
      getType: () => Promise.reject(new DOMException('Image data unavailable', 'DataError')),
      types: ['image/png'],
    } as unknown as ClipboardItem;
    vi.spyOn(Navigator.prototype, 'clipboard', 'get').mockReturnValue({
      read: () => Promise.resolve([item]),
    } as unknown as Clipboard);

    await act(() => commandHandlers.get('canvas.pasteImage')!());

    await vi.waitFor(() =>
      expect(store.getState().notifications[0]).toMatchObject({
        kind: 'error',
        title: 'Could not paste an image from the clipboard',
      })
    );
    expect(store.getState().notifications[0]?.message).toBeUndefined();
  });
});

describe('CanvasWidgetView control layer overlay', () => {
  const emptyControlLayer = createControlLayer({ id: 'control-1', name: 'Control Layer 1' });
  const hint = () => page.getByRole('note').filter({ hasText: 'Drop an image here' });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('points an empty control layer at its image sources until the brush or eraser takes over', async () => {
    await renderView('completed', { selectedLayer: emptyControlLayer, staged: false });

    await expect.element(hint()).toBeVisible();
    await expect.element(hint().getByText('or pick one in Properties', { exact: true })).toBeVisible();

    await act(() => harness.engine!.tools.setTool('brush'));
    await expect.element(hint()).not.toBeInTheDocument();

    await act(() => harness.engine!.tools.setTool('eraser'));
    await expect.element(hint()).not.toBeInTheDocument();

    await act(() => harness.engine!.tools.setTool('move'));
    await expect.element(hint()).toBeVisible();
  });

  it('keeps pointing at the image sources while a result is staged: the canvas stays editable', async () => {
    await renderView('completed', { selectedLayer: emptyControlLayer });

    await expect.element(page.getByRole('button', { exact: true, name: 'Accept to layer' })).toBeVisible();
    await expect.element(hint()).toBeVisible();
  });

  it('gives way to the drop zones while a gallery image is dragged', async () => {
    harness.dragData = { items: [{ kind: 'image', name: 'photo.png' }], kind: 'gallery-item' };
    await renderView('completed', { selectedLayer: emptyControlLayer, staged: false });

    await expect.element(page.getByText('Drop an image here')).toBeInTheDocument();
    await expect.element(hint()).not.toBeInTheDocument();
  });

  it('shows no hint once the control layer has an image', async () => {
    const filled: CanvasControlLayerContract = {
      ...emptyControlLayer,
      source: { image: { height: 64, imageName: 'photo.png', width: 96 }, type: 'image' },
    };
    await renderView('completed', { selectedLayer: filled, staged: false });

    expect(page.getByRole('note').elements()).toHaveLength(0);
    expect(page.getByText('Drop an image here').elements()).toHaveLength(0);
  });

  const depthLayer: CanvasControlLayerContract = {
    ...emptyControlLayer,
    controlType: 'depth',
    detection: { settings: {}, type: 'depth_anything_depth_estimation' },
    source: { image: { height: 64, imageName: 'photo.png', width: 96 }, type: 'image' },
  };

  /** Renders `layer` with its detection status driven by the returned store. */
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

  it('announces a background detection run while it is queued or processing, below the header row', async () => {
    const statuses = await renderWithStatuses();
    const layer = depthLayer;
    // The shell reserves this much of the center view for its header row (zoom, fit, undo).
    host!.style.setProperty('--wb-center-chrome-inset', '3rem');
    const status = page.getByRole('status').filter({ hasText: /^(Running|Waiting to run) / });

    expect(status.elements()).toHaveLength(0);

    await act(() => statuses.set(layer.id, { status: 'queued' }));
    await expect.element(status).toHaveTextContent('Waiting to run Depth Anything…');
    await expect.element(status).toHaveAttribute('aria-live', 'polite');
    const headerBottom = host!.getBoundingClientRect().top + 48;
    expect(status.element().getBoundingClientRect().top).toBeGreaterThanOrEqual(headerBottom);

    await act(() => statuses.set(layer.id, { detector: 'normal_map', status: 'processing' }));
    await expect.element(status).toHaveTextContent('Running Normal map…');

    await act(() => statuses.set(layer.id, { status: 'idle' }));
    await expect.element(status).not.toBeInTheDocument();
  });

  it('announces a failed detection run with its reason and no spinner', async () => {
    const statuses = await renderWithStatuses();

    await act(() => statuses.set(depthLayer.id, { error: { code: 'process-failed' }, status: 'error' }));

    const status = page.getByRole('status').filter({ hasText: /failed/ });
    await expect.element(status).toHaveTextContent('Depth Anything failed: the detector could not run');
    expect(status.element().querySelector('.chakra-spinner')).toBeNull();

    await act(() => statuses.set(depthLayer.id, { status: 'idle' }));
    await expect.element(status).not.toBeInTheDocument();
  });

  /** A depth layer whose map was detected from the photo it keeps. */
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

    // The engine ends the view (an edit, another selection): the switch follows it back to Map.
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

  it('shows Map while the engine holds a view of another layer', async () => {
    await renderView('completed', { selectedLayer: readyLayer, staged: false });
    const view: ControlLayerView = { layerId: 'other-control', view: 'original' };

    const { stores } = harness.engine as unknown as {
      stores: { controlLayerView: { set(value: ControlLayerView | null): void } };
    };
    await act(() => stores.controlLayerView.set(view));

    await expect.element(viewOption('Map')).toBeChecked();
  });

  it.each([
    ['queued', { status: 'queued' as const }, /^Waiting to run /],
    ['failed', { error: { code: 'process-failed' as const }, status: 'error' as const }, /failed/],
  ])('gives the slot to the detection chip while a run is %s', async (_case, runStatus, chipText) => {
    const statuses = await renderWithStatuses(readyLayer);
    await expect.element(viewSwitch()).toBeVisible();

    await act(() => statuses.set(readyLayer.id, runStatus));
    await expect.element(page.getByRole('status').filter({ hasText: chipText })).toBeVisible();
    expect(viewSwitch().elements()).toHaveLength(0);

    await act(() => statuses.set(readyLayer.id, { status: 'idle' }));
    await expect.element(viewSwitch()).toBeVisible();
  });

  it.each([
    ['an empty untyped layer', emptyControlLayer],
    ['a typed layer with no original', depthLayer],
    ['a layer whose map awaits detection', { ...readyLayer, filter: { settings: {}, type: 'canny_edge_detection' } }],
    ['a disabled layer', { ...readyLayer, isEnabled: false }],
    ['a hidden layer', { ...readyLayer, isHidden: true }],
  ])('offers no view switch for %s', async (_case, layer) => {
    await renderWithStatuses(layer);

    await expect.element(page.getByRole('status')).toBeInTheDocument();
    expect(viewSwitch().elements()).toHaveLength(0);
  });

  it('keeps the view switch while a result is staged', async () => {
    await renderView('completed', { selectedLayer: readyLayer });

    await expect.element(page.getByRole('button', { exact: true, name: 'Accept to layer' })).toBeVisible();
    await expect.element(viewSwitch()).toBeVisible();
  });

  it('hides the view switch while an operation holds the document', async () => {
    await renderView('completed', { selectedLayer: readyLayer, staged: false });
    await expect.element(viewSwitch()).toBeVisible();

    const { stores } = harness.engine as unknown as {
      stores: { documentEditingLocked: { set(value: boolean): void } };
    };
    await act(() => stores.documentEditingLocked.set(true));

    await expect.element(viewSwitch()).not.toBeInTheDocument();
  });

  it('gives way to the drop zones while a gallery image is dragged', async () => {
    harness.dragData = { items: [{ kind: 'image', name: 'photo.png' }], kind: 'gallery-item' };
    await renderView('completed', { selectedLayer: readyLayer, staged: false });

    await expect.element(page.getByRole('status')).toBeInTheDocument();
    expect(viewSwitch().elements()).toHaveLength(0);
  });
});

describe('CanvasWidgetView desktop file drops', () => {
  const emptyControlLayer = createControlLayer({ id: 'control-1', name: 'Control Layer 1' });
  const PHOTO_FILE = new File([new Uint8Array([137, 80, 78, 71])], 'photo.png', { type: 'image/png' });
  const UPLOADED = { height: 64, imageName: 'dropped.png', width: 96 } as GalleryImage;
  const hint = () => page.getByRole('note');

  afterEach(() => {
    vi.restoreAllMocks();
  });

  /** Sends a native drag carrying `transfer` over the canvas and drops it; returns whether each event was taken. */
  const drag = async (transfer: DataTransfer) => {
    const surface = page.getByRole('region', { exact: true, name: 'Canvas surface' }).element();
    const over = new DragEvent('dragover', { bubbles: true, cancelable: true, dataTransfer: transfer });
    const drop = new DragEvent('drop', { bubbles: true, cancelable: true, dataTransfer: transfer });
    await act(() => surface.dispatchEvent(over));
    await act(() => surface.dispatchEvent(drop));
    return { dropTaken: drop.defaultPrevented, overTaken: over.defaultPrevented };
  };
  const dragFiles = (files: File[]) => {
    const transfer = new DataTransfer();
    for (const file of files) {
      transfer.items.add(file);
    }
    return drag(transfer);
  };
  const lastNotice = (store: Awaited<ReturnType<typeof renderView>>['store']) => store.getState().notifications[0];

  it('fills an empty control layer from a dropped image, says it is uploading, and refuses a second drop meanwhile', async () => {
    let finishUpload!: (image: GalleryImage) => void;
    const upload = vi.spyOn(galleryTransfers, 'upload').mockReturnValue(
      new Promise((resolve) => {
        finishUpload = resolve;
      })
    );
    const { store } = await renderView('completed', {
      beforeRender: (engine) => {
        vi.spyOn(getCanvasOperations(engine), 'applyControlLayerIntent').mockReturnValue('updated');
      },
      selectedLayer: emptyControlLayer,
      staged: false,
    });
    const apply = vi.mocked(getCanvasOperations(harness.engine!).applyControlLayerIntent);

    expect(await dragFiles([PHOTO_FILE])).toEqual({ dropTaken: true, overTaken: true });
    await expect.element(hint()).toHaveTextContent('Uploading…');
    await expect.element(hint()).toHaveAttribute('aria-busy', 'true');

    // A second drop while the first is on its way would race it: refused, with the reason.
    await dragFiles([PHOTO_FILE]);
    expect(upload).toHaveBeenCalledTimes(1);
    await vi.waitFor(() =>
      expect(lastNotice(store)).toMatchObject({ title: 'An image is still uploading to this layer.' })
    );

    await act(() => finishUpload(UPLOADED));
    await vi.waitFor(() => expect(apply).toHaveBeenCalledTimes(1));
    const { bbox } = activeProject().canvas.document;
    expect(upload).toHaveBeenCalledWith(PHOTO_FILE, 'none', expect.anything());
    expect(apply).toHaveBeenCalledWith('control-1', {
      image: { height: 64, imageName: 'dropped.png', width: 96 },
      kind: 'replace-image',
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: bbox.x, y: bbox.y },
    });
    // Properties names the image after the file, as for its own Upload.
    expect(getControlSourceSnapshot(harness.engine!).origins['dropped.png']).toEqual({
      kind: 'file',
      name: 'photo.png',
    });
    // The upload has landed: the card no longer says it is uploading.
    await vi.waitFor(() => expect(hint().element().getAttribute('aria-busy')).toBe('false'));
    expect(hint().element().textContent).toContain('Drop an image here');
  });

  it('refuses a drop on a control layer locked by its group before uploading anything', async () => {
    const upload = vi.spyOn(galleryTransfers, 'upload');
    const { store } = await renderView('completed', {
      nodes: [groupContract('group-1', [emptyControlLayer], { isLocked: true })],
      selectedLayer: emptyControlLayer,
      staged: false,
    });

    expect(await dragFiles([PHOTO_FILE])).toEqual({ dropTaken: true, overTaken: true });

    expect(lastNotice(store)).toMatchObject({ title: 'Unlock the layer before using this action.' });
    expect(upload).not.toHaveBeenCalled();
  });

  it('says what a drop needs when no empty control layer is selected, or the files do not fit', async () => {
    const upload = vi.spyOn(galleryTransfers, 'upload');
    const { store } = await renderView('completed', {
      selectedLayer: layerContract('raster-1', 'raster'),
      staged: false,
    });

    await dragFiles([PHOTO_FILE]);
    expect(lastNotice(store)).toMatchObject({ title: 'Select an empty control layer to drop an image on it.' });
    await dragFiles([new File(['notes'], 'notes.txt', { type: 'text/plain' })]);
    expect(lastNotice(store)).toMatchObject({ title: 'Only image files can be dropped on the canvas.' });
    await dragFiles([PHOTO_FILE, PHOTO_FILE]);
    expect(lastNotice(store)).toMatchObject({ title: 'Drop one image at a time.' });

    expect(upload).not.toHaveBeenCalled();
  });

  it('leaves a drag that carries no files alone', async () => {
    await renderView('completed', { selectedLayer: emptyControlLayer, staged: false });
    const transfer = new DataTransfer();
    transfer.setData('text/plain', 'not a file');

    expect(await drag(transfer)).toEqual({ dropTaken: false, overTaken: false });
  });
});

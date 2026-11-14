import type * as DndKitCore from '@dnd-kit/core';
import type { GalleryImage } from '@features/gallery';
import type * as GalleryBackend from '@features/gallery/data/backend';
import type {
  CanvasControlLayerContract,
  CanvasLayerContract,
  CanvasNodeContract,
  ControlLayerView,
  ControlLayerViewOutcome,
} from '@workbench/canvas-engine/api';
import type { CanvasStagingCandidateContract } from '@workbench/canvas-engine/contracts';
import type * as CanvasOperationsApi from '@workbench/canvas-operations/api';
import type { CanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import type { Project } from '@workbench/projectContracts';
import type { WorkbenchQueueItem as QueueItem } from '@workbench/queueHistoryContracts';
import type { WidgetViewProps } from '@workbench/widgetContracts';
import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';

import { ChakraProvider } from '@chakra-ui/react';
import { auditAccessibility } from '@platform/browser/auditAccessibility.testing';
import { TINY_PNG_BASE64 } from '@platform/browser/nativeDrag.testing';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { createControlLayer } from '@workbench/canvas-engine/api';
import { layerContract, stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { getCanvasOperations, uploadControlImageFile } from '@workbench/canvas-operations/api';
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
import { commands, page, userEvent } from 'vitest/browser';

import type * as EngineStoreHooks from './engineStoreHooks';

/** The view reads the real store and drives its real commands; only unrelated chrome is stubbed. */
const harness = vi.hoisted(() => ({
  /** Drag data dnd-kit reports as active; null leaves the real (idle) context. */
  dragData: null as unknown,
  engine: null as CanvasEngine | null,
  /** Whether the registry still holds `engine` live for its project; a closed Canvas releases it. */
  isEngineLive: true,
  /** The operation holding the document, if any; null leaves the canvas editable. */
  operation: null as unknown,
  store: null as ReturnType<typeof createWorkbenchStore> | null,
  /** The server's answer to each image upload; the transport is the only double on the upload path. */
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
  // No board listing: uploads judge their board from an empty one.
  listGalleryBoards: () => Promise.resolve([]),
  uploadGalleryImage: (file: File, boardId: string) => harness.uploadImage(file, boardId),
}));
// The engine registry's non-owning lookup, as the registry answers it for the harness engine.
vi.mock('@workbench/canvas-operations/api', async (importOriginal) => ({
  ...(await importOriginal<typeof CanvasOperationsApi>()),
  getCanvasEngine: (projectId: string) =>
    harness.isEngineLive && harness.engine?.projectId === projectId ? harness.engine : undefined,
}));
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
vi.mock('./CanvasGlobalContextMenu', () => ({ CanvasGlobalContextMenu: () => null }));
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
  Object.assign(harness, { dragData: null, engine: null, isEngineLive: true, operation: null, store: null });
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
    hostWidth = 1000,
    nodes,
    selectedLayer,
    staged = true,
  }: {
    beforeRender?: (engine: CanvasEngine) => void;
    /** Real Canvas2D pixels, for a test that attaches a surface and draws. */
    domRaster?: boolean;
    /** The surface's width; a drag through the browser's own input reaches only what fits in the test frame. */
    hostWidth?: number;
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
    // A cut the layer allows, so it reaches the clipboard write that fails.
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

  it('keeps pointing at the image sources while a result is staged: the canvas stays editable', async () => {
    await renderView('completed', { selectedLayer: emptyControlLayer });

    await expect.element(page.getByRole('button', { exact: true, name: 'Accept to layer' })).toBeVisible();
    await expect.element(hint()).toBeVisible();
  });

  it('gives way to the drop zones while a gallery image is dragged', async () => {
    harness.dragData = { items: [{ kind: 'image', name: 'photo.png' }], kind: 'gallery-item' };
    await renderView('completed', { selectedLayer: emptyControlLayer, staged: false });

    await expect.element(page.getByText('Add an image')).toBeInTheDocument();
    await expect.element(hint()).not.toBeInTheDocument();
  });

  it('shows no hint once the control layer has an image', async () => {
    const filled: CanvasControlLayerContract = {
      ...emptyControlLayer,
      source: { image: { height: 64, imageName: 'photo.png', width: 96 }, type: 'image' },
    };
    await renderView('completed', { selectedLayer: filled, staged: false });

    expect(page.getByRole('note').elements()).toHaveLength(0);
    expect(page.getByText('Add an image').elements()).toHaveLength(0);
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

  it('hides the view switch while a staged result hides the layer, and offers it again once staging ends', async () => {
    await renderView('completed', { selectedLayer: readyLayer });

    await expect.element(page.getByRole('button', { exact: true, name: 'Accept to layer' })).toBeVisible();
    await expect.poll(() => harness.engine!.interaction.get('stagingHidesOverlays')).toBe(true);
    expect(viewSwitch().elements()).toHaveLength(0);

    // The view clears the preview as accept or discard ends staging; the harness view does not rerender on its own.
    await act(() => harness.engine!.previews.setStagedPreview(null));
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
  /** The zone element dnd-kit and file drags hover, around the label `name`. */
  const zoneTarget = (name: string) => zone(name).element().closest('[data-canvas-image-drop-destination]')!;
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
  /** Where the zone labelled `name` sits on the surface, before any drag shows it: the overlay's grid places it. */
  const pointOn = (destination: 'control' | 'raster') => {
    const rect = surface().getBoundingClientRect();
    return {
      x: rect.left + rect.width * (destination === 'raster' ? 0.25 : 0.75),
      y: rect.top + rect.height * 0.25,
    };
  };
  /** A real drag from the operating system, through the browser's own drag input. */
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
  /** Dispatches a drag over the zone labelled `zoneName` and drops it there, from the test itself. */
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
    // Stored as pasted media is: in the project's auto-add board (Uncategorized here), then layered at the frame.
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
    // The zones went with the drag.
    expect(zone('Add as raster layer').elements()).toHaveLength(0);
  });

  it('fills the selected empty control layer from one image dropped on the control zone, as a gallery drop does', async () => {
    harness.uploadImage.mockImplementation((file) => Promise.resolve(stored(file)));
    const { store } = await renderView('completed', {
      beforeRender: (engine) => {
        vi.spyOn(getCanvasOperations(engine), 'applyControlLayerIntent').mockReturnValue('updated');
      },
      hostWidth: window.innerWidth,
      selectedLayer: emptyControlLayer,
      staged: false,
    });
    const apply = vi.mocked(getCanvasOperations(harness.engine!).applyControlLayerIntent);

    await nativeDrop('control', ['depth.png']);

    await vi.waitFor(() => expect(apply).toHaveBeenCalledTimes(1), LANDED);
    const { bbox } = activeProject().canvas.document;
    expect(apply).toHaveBeenCalledWith('control-1', {
      image: { height: 2, imageName: 'stored-depth.png', width: 2 },
      kind: 'replace-image',
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: bbox.x, y: bbox.y },
    });
    await vi.waitFor(() => expect(titles(store)[0]).toBe('Filled Control Layer 1'));
  });

  it('highlights the zone under the files as they move between zones', async () => {
    await renderView('completed', { hostWidth: window.innerWidth, staged: false });
    const transfer = transferOf([imageFile('one.png')]);

    await send('dragenter', transfer);
    await send('dragover', transfer, centreOf(zoneTarget('Add as inpaint mask')));
    expect(zoneTarget('Add as inpaint mask')).toHaveAttribute('data-over');
    expect(zoneTarget('Add as raster layer')).not.toHaveAttribute('data-over');

    await send('dragover', transfer, centreOf(zoneTarget('Add as raster layer')));
    expect(zoneTarget('Add as raster layer')).toHaveAttribute('data-over');
    expect(zoneTarget('Add as inpaint mask')).not.toHaveAttribute('data-over');
  });

  it('says why files that are not images are not added, and uploads only the images among them', async () => {
    harness.uploadImage.mockImplementation((file) => Promise.resolve(stored(file)));
    const { store } = await renderView('completed', { hostWidth: window.innerWidth, staged: false });
    const notes = new File(['notes'], 'notes.txt', { type: 'text/plain' });
    const clip = new File(['clip'], 'clip.mp4', { type: 'video/mp4' });

    await dropOn('Add as raster layer', [notes, clip]);
    expect(store.getState().notifications[0]).toMatchObject({
      kind: 'error',
      title: 'Only image files can be dropped on the canvas.',
    });
    expect(harness.uploadImage).not.toHaveBeenCalled();

    await dropOn('Add as raster layer', [notes, imageFile('one.png')]);
    expect(store.getState().notifications[0]).toMatchObject({
      kind: 'info',
      title: 'Only image files can be dropped on the canvas.',
    });
    await vi.waitFor(() => expect(harness.uploadImage).toHaveBeenCalledTimes(1), LANDED);
    expect(harness.uploadImage.mock.calls[0]![0].name).toBe('one.png');
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
    // An edit made while the file uploads, then a switch to another project.
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

  it('adds through the project when its Canvas was closed while the files uploaded', async () => {
    let finishUpload!: () => void;
    harness.uploadImage.mockImplementation(
      (file) =>
        new Promise((resolve) => {
          finishUpload = () => resolve(stored(file));
        })
    );
    const { projectId } = await renderView('completed', { hostWidth: window.innerWidth, staged: false });
    const commit = vi.spyOn(harness.engine!.layers, 'commitStructural');

    await dropOn('Add as raster layer', [imageFile('late.png')]);
    await vi.waitFor(() => expect(harness.uploadImage).toHaveBeenCalledTimes(1), LANDED);
    // The registry let the released engine go; the one the drop saw must not be used.
    harness.isEngineLive = false;
    await act(() => finishUpload());

    await vi.waitFor(() =>
      expect(rasterLayersOf(projectId).map((layer) => layer.type === 'raster' && layer.source)).toContainEqual({
        image: { height: 2, imageName: 'stored-late.png', width: 2 },
        type: 'image',
      })
    );
    expect(commit).not.toHaveBeenCalled();
  });

  it('ends the drag when it leaves the canvas, leaves the window, is cancelled or is dropped elsewhere', async () => {
    await renderView('completed', { hostWidth: window.innerWidth, staged: false });
    const transfer = transferOf([imageFile('one.png')]);
    const zonesShown = () => zone('Add as inpaint mask').elements().length > 0;

    // Onto the rest of the page.
    await send('dragenter', transfer);
    expect(zonesShown()).toBe(true);
    await send('dragleave', transfer, { relatedTarget: document.body });
    expect(zonesShown()).toBe(false);

    // Out of the window, after crossing onto an element the drag itself took down: no leave reports that one.
    const doomed = document.createElement('div');
    surface().append(doomed);
    await send('dragenter', transfer);
    await send('dragenter', transfer, { relatedTarget: surface() }, doomed);
    doomed.remove();
    await send('dragleave', transfer, { relatedTarget: null });
    await vi.waitFor(() => expect(zonesShown()).toBe(false));

    // A drag that finished elsewhere: dropped outside the canvas, or ended at its source.
    await send('dragenter', transfer);
    await act(() => document.body.dispatchEvent(new DragEvent('drop', { bubbles: true, dataTransfer: transfer })));
    expect(zonesShown()).toBe(false);
    await send('dragenter', transfer);
    await act(() => window.dispatchEvent(new DragEvent('dragend')));
    expect(zonesShown()).toBe(false);
  });

  it('keeps the zones while the window loses focus mid-drag, as a focus-follows-mouse window manager does', async () => {
    await renderView('completed', { hostWidth: window.innerWidth, staged: false });
    const transfer = transferOf([imageFile('one.png')]);

    await send('dragenter', transfer);
    await act(() => window.dispatchEvent(new Event('blur')));

    await expect.element(zone('Add as raster layer')).toBeVisible();
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
    // Long enough for a taken drop to have loaded the upload and sent the file.
    await new Promise((resolve) => {
      setTimeout(resolve, 500);
    });
    expect(harness.uploadImage).not.toHaveBeenCalled();
  });

  it('says the canvas is busy to a gallery image dragged over it while an operation holds it', async () => {
    harness.operation = { identity: { kind: 'filter' }, status: 'active' };
    harness.dragData = { items: [{ kind: 'image', name: 'photo.png' }], kind: 'gallery-item' };
    await renderView('completed', { hostWidth: window.innerWidth, staged: false });

    await expect
      .element(page.getByText('The canvas is busy. Try again when the current action finishes.'))
      .toBeVisible();
    expect(zone('Add as raster layer').elements()).toHaveLength(0);
  });

  it('says Uploading on the control hint while Properties uploads an image into the empty layer', async () => {
    let finishUpload!: () => void;
    await renderView('completed', { selectedLayer: emptyControlLayer, staged: false });
    const hint = () => page.getByRole('note');
    await expect.element(hint()).toHaveAttribute('aria-busy', 'false');

    const upload = uploadControlImageFile({
      engine: harness.engine!,
      file: imageFile('depth.png'),
      layerId: 'control-1',
      uploadImage: (file) =>
        new Promise((resolve) => {
          finishUpload = () => resolve(stored(file));
        }),
    });

    await expect.element(hint()).toHaveAttribute('aria-busy', 'true');
    await expect.element(hint()).toHaveTextContent('Uploading…');
    await act(() => finishUpload());
    await act(() => upload);
    // The layer has its image: the hint and its spinner are gone.
    await expect.element(page.getByText('Uploading…')).not.toBeInTheDocument();
  });

  it('leaves a drag that carries no files alone', async () => {
    await renderView('completed', { hostWidth: window.innerWidth, staged: false });
    const transfer = new DataTransfer();
    transfer.setData('text/plain', 'not a file');

    expect((await send('dragover', transfer)).defaultPrevented).toBe(false);
    expect((await send('drop', transfer)).defaultPrevented).toBe(false);
    expect(zone('Add as raster layer').elements()).toHaveLength(0);
  });
});

describe('CanvasWidgetView frame size readout', () => {
  it('follows a Frame tool resize through the engine, then shows the committed frame', async () => {
    await renderView('completed', { domRaster: true, staged: false });
    const engine = harness.engine!;
    const screen = document.createElement('canvas');
    const overlay = document.createElement('canvas');
    for (const canvas of [screen, overlay]) {
      Object.assign(canvas.style, { height: '300px', left: '0', position: 'fixed', top: '0', width: '300px' });
      host!.append(canvas);
    }
    await act(() => engine.surface.attach(screen, overlay));
    await act(() => engine.tools.setTool('bbox'));
    const readout = page.getByRole('group', { name: 'Frame and render size' });
    const frameValue = () => readout.element().querySelector('dd')?.textContent;
    const { bbox } = activeProject().canvas.document;
    await expect.element(readout).toHaveTextContent(`Frame${bbox.width}×${bbox.height}`);

    // Drag the bottom-right handle inward by 64 document pixels; Alt turns grid snapping off, so the size is exact.
    const viewport = engine.viewport.getViewport();
    const corner = viewport.documentToScreen({ x: bbox.x + bbox.width, y: bbox.y + bbox.height });
    const target = viewport.documentToScreen({ x: bbox.x + bbox.width - 64, y: bbox.y + bbox.height - 64 });
    const pointer = (type: string, { x, y }: { x: number; y: number }, buttons: number) =>
      new PointerEvent(type, { altKey: true, bubbles: true, buttons, clientX: x, clientY: y, pointerId: 1 });
    await act(() => {
      overlay.dispatchEvent(pointer('pointerdown', corner, 1));
      overlay.dispatchEvent(pointer('pointermove', target, 1));
    });
    // Mid-drag, before anything commits, the readout shows the frame being drawn.
    expect(frameValue()).toBe(`${bbox.width - 64}×${bbox.height - 64}`);
    expect(activeProject().canvas.document.bbox).toEqual(bbox);

    await act(() => overlay.dispatchEvent(pointer('pointerup', target, 0)));
    expect(activeProject().canvas.document.bbox).toMatchObject({ height: bbox.height - 64, width: bbox.width - 64 });
    expect(frameValue()).toBe(`${bbox.width - 64}×${bbox.height - 64}`);
    engine.surface.detach();
  });
});

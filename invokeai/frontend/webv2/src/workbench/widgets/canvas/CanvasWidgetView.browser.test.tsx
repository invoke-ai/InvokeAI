import type * as DndKitCore from '@dnd-kit/core';
import type {
  CanvasControlLayerContract,
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
import { auditAccessibility } from '@platform/browser/auditAccessibility.testing';
import { system } from '@theme/system';
import { createControlLayer } from '@workbench/canvas-engine/api';
import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { getCanvasOperations } from '@workbench/canvas-operations/api';
import { createControlDetectionStatusStore } from '@workbench/canvas-operations/controlDetectionStatusStore';
import { createCanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import { createCanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
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
 * `selectedLayer` starts the document with that one layer selected; `beforeRender` sees the engine before the view
 * subscribes to it.
 */
const renderView = async (
  status: QueueItem['status'],
  {
    beforeRender,
    selectedLayer,
    staged = true,
  }: {
    beforeRender?: (engine: CanvasEngine) => void;
    selectedLayer?: CanvasControlLayerContract;
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
            stacks: stacksFrom([selectedLayer]),
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
    backend: createTestStubRasterBackend(),
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
          <CanvasWidgetView
            {...({ instance: activeProject().widgetInstances.canvas, runtime } as unknown as WidgetViewProps)}
          />
        </ChakraProvider>
      </I18nextProvider>
    )
  );
  return { commandHandlers, projectId, store };
};

describe('CanvasWidgetView accept and stop', () => {
  it('accepts the result, then cancels its batch through the workbench queue command with the accept notice', async () => {
    const { store } = await renderView('running');

    await page.getByRole('button', { exact: true, name: 'Accept to Layer' }).click();

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

    await page.getByRole('button', { exact: true, name: 'Accept to Layer' }).click();

    const project = store.getState().projects[0]!;
    expect(project.canvas.document.stacks.raster).toHaveLength(1);
    expect(project.queue.items[0]?.status).toBe('completed');
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

  it('shows no hint while staging locks the surface against drops', async () => {
    await renderView('completed', { selectedLayer: emptyControlLayer });

    await expect.element(page.getByRole('button', { exact: true, name: 'Accept to Layer' })).toBeVisible();
    await expect.element(hint()).not.toBeInTheDocument();
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
    const status = page.getByRole('status').filter({ hasText: /^Running / });

    expect(status.elements()).toHaveLength(0);

    await act(() => statuses.set(layer.id, { status: 'queued' }));
    await expect.element(status).toHaveTextContent('Running Depth Anything…');
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
    ['queued', { status: 'queued' as const }, /^Running /],
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

  it('hides the view switch while staging locks the surface', async () => {
    await renderView('completed', { selectedLayer: readyLayer });

    await expect.element(page.getByRole('button', { exact: true, name: 'Accept to Layer' })).toBeVisible();
    expect(viewSwitch().elements()).toHaveLength(0);
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

import type { CanvasLayerContract, CanvasStagingCandidateContract } from '@workbench/canvas-engine/contracts';
import type * as CanvasOperationsApi from '@workbench/canvas-operations/api';
import type { CanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import type { Project } from '@workbench/projectContracts';
import type { WorkbenchQueueItem as QueueItem } from '@workbench/queueHistoryContracts';
import type { WidgetViewProps } from '@workbench/widgetContracts';
import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { createCanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import { createCanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import { runNotificationUndo } from '@workbench/shell/notifications/undoRegistry';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { createInstance } from 'i18next';
import { act, createRef, useImperativeHandle, type Ref } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import type * as EngineStoreHooks from './engineStoreHooks';

const harness = vi.hoisted(() => ({
  create: vi.fn<(destination: string) => Promise<void>>(),
  describeDestination: vi.fn<() => Promise<string | null>>(),
  editingLocked: false,
  engine: null as CanvasEngine | null,
  isCreating: false,
  isSaving: false,
  save: vi.fn<(region: 'bbox' | 'canvas') => Promise<void>>(),
  store: null as ReturnType<typeof createWorkbenchStore> | null,
}));
const activeProject = () => harness.store!.getState().projects[0]!;

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
}));
vi.mock('./engineStoreHooks', async (importOriginal) => ({
  ...(await importOriginal<typeof EngineStoreHooks>()),
  useCanvasDocumentEditingLocked: () => harness.editingLocked,
}));
vi.mock('@workbench/canvas-operations/react', () => ({ useCanvasEngine: () => harness.engine }));
vi.mock('@workbench/canvas-operations/api', async (importOriginal) => ({
  ...(await importOriginal<typeof CanvasOperationsApi>()),
  getCanvasEngine: () => harness.engine ?? undefined,
}));
vi.mock('./useCanvasGallerySave', () => ({
  useCanvasGallerySave: () => ({
    describeDestination: harness.describeDestination,
    isSaving: harness.isSaving,
    save: harness.save,
  }),
}));
vi.mock('./useCreateFromBbox', () => ({
  useCreateFromBbox: () => ({ createFromBbox: harness.create, isCreating: harness.isCreating }),
}));

import { CanvasHeaderActions } from './CanvasHeaderActions';
import { CANVAS_SURFACE_ATTR } from './surfaceFocus';
import { useRestorePoint } from './useCanvasRecovery';

const INSTANCE_ID = 'canvas-1';

const CanvasViewStandIn = () => (
  <div data-hotkey-widget-instance-id={INSTANCE_ID}>
    <div {...{ [CANVAS_SURFACE_ATTR]: '' }} data-testid="surface" tabIndex={-1} />
  </div>
);

const RestorePointProbe = ({ ref }: { ref: Ref<ReturnType<typeof useRestorePoint>> }) => {
  const restorePoint = useRestorePoint();
  useImperativeHandle(ref, () => restorePoint, [restorePoint]);
  return null;
};
const restorePointRef = createRef<ReturnType<typeof useRestorePoint>>();

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
  Object.assign(harness, { editingLocked: false, engine: null, isCreating: false, isSaving: false, store: null });
  harness.save.mockReset();
  harness.create.mockReset();
  harness.describeDestination.mockReset();
  root = null;
  host = null;
});

interface Registration {
  defaultKeys?: string[];
  handler?: () => void;
  id: string;
  title: string;
}

const paintedLayer: CanvasLayerContract = {
  blendMode: 'normal' as const,
  id: 'painted',
  isEnabled: true,
  isLocked: false,
  name: 'Painted',
  opacity: 1,
  source: { image: { height: 1, imageName: 'painted.png', width: 1 }, type: 'image' as const },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster' as const,
};

const stagedResult: CanvasStagingCandidateContract = {
  height: 64,
  imageName: 'staged.png',
  imageUrl: '/staged.png',
  placement: { height: 64, opacity: 1, width: 64, x: 0, y: 0 },
  queuedAt: '2026-07-16T00:00:00.000Z',
  sourceBackendItemId: 1,
  sourceQueueItemId: 'batch',
  thumbnailUrl: '/staged-thumb.png',
  width: 64,
};

const renderActions = async ({
  canvasValues,
  layers = [],
  running = false,
}: { canvasValues?: Record<string, unknown>; layers?: CanvasLayerContract[]; running?: boolean } = {}) => {
  const initial = createInitialWorkbenchState();
  const created = initial.projects[0]!;
  const base =
    layers.length > 0
      ? {
          ...created,
          canvas: { ...created.canvas, document: { ...created.canvas.document, stacks: stacksFrom(layers) } },
        }
      : created;
  const batch = {
    backendBatchId: 'batch-backend',
    backendItemIds: [1, 2],
    cancellable: true,
    completedBackendItemIds: [1],
    id: 'batch',
    snapshot: {
      canvas: { document: base.canvas.document, documentRevision: base.canvas.documentRevision },
      destination: 'canvas',
      presentation: { batchCount: 2, height: 64, width: 64 },
      sourceId: 'canvas',
      submittedAt: '2026-07-16T00:00:00.000Z',
    },
    status: 'running',
  } as unknown as QueueItem;
  const store = createWorkbenchStore({
    ...initial,
    projects: [running ? { ...base, queue: { items: [batch] } } : base],
  });
  if (canvasValues) {
    store.commands.widgets.patchValues('canvas', canvasValues);
  }
  const projectId = store.getState().activeProjectId;
  if (running) {
    store.commands.canvas.appendStagingCandidate({ candidate: stagedResult, projectId });
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
  const commands: Registration[] = [];
  const hotkeys: Registration[] = [];
  const runtime = {
    commands: {
      register: (command: Registration) => {
        commands.push(command);
        return () => undefined;
      },
    },
    hotkeys: {
      register: (hotkey: Registration) => {
        hotkeys.push(hotkey);
        return () => undefined;
      },
    },
  };
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  const rerender = () =>
    act(() =>
      root!.render(
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <CanvasHeaderActions {...({ instance: { id: INSTANCE_ID }, runtime } as unknown as WidgetViewProps)} />
            <CanvasViewStandIn />
            <RestorePointProbe ref={restorePointRef} />
          </ChakraProvider>
        </I18nextProvider>
      )
    );
  await rerender();
  return { commands, hotkeys, rerender, store };
};

const clearButton = () => page.getByRole('button', { exact: true, name: 'Clear canvas' });
const confirmDialog = () => page.getByRole('alertdialog');

type TestStore = ReturnType<typeof createWorkbenchStore>;
const rasterIds = (store: TestStore) =>
  store.getState().projects[0]!.canvas.document.stacks.raster.map((node) => node.id);
const undoNewest = (store: TestStore) => {
  const undo = store
    .getState()
    .notifications.flatMap((notice) => notice.actions ?? [])
    .find((action) => action.kind === 'undo');
  act(() => runNotificationUndo((undo as { undoId: string }).undoId));
};
const addLayer = (store: TestStore, id: string) =>
  store.commands.canvas.apply(store.getState().activeProjectId, {
    anchor: harness.engine!.document.captureInsertionAnchor('raster', null),
    layer: { ...paintedLayer, id },
    type: 'addCanvasLayer',
  });
const clearConfirmed = async (store: TestStore) => {
  await clearButton().click();
  await confirmDialog().getByRole('button', { exact: true, name: 'Clear canvas' }).click();
  await expect.poll(() => store.getState().notifications[0]?.title).toBe('Canvas cleared');
};

describe('CanvasHeaderActions clear canvas', () => {
  it('asks to clear the canvas on one click, without a menu, even when the canvas is empty', async () => {
    await renderActions();

    await clearButton().click();

    await expect.element(confirmDialog()).toBeVisible();
    await expect.element(confirmDialog().getByRole('heading', { name: 'Clear the canvas?' })).toBeVisible();
    await expect.element(confirmDialog().getByRole('button', { exact: true, name: 'Clear canvas' })).toBeVisible();
    await expect
      .element(confirmDialog().getByText('Removes the 1 layer and starts a fresh canvas with an empty inpaint mask.'))
      .toBeVisible();
    expect(confirmDialog().element().textContent).not.toContain('History keeps a restore point');
    expect(document.querySelector('[role="menu"]')).toBeNull();
  });

  it('names the staged results and the generation still running, and stops it while that is ticked', async () => {
    const { store } = await renderActions({ layers: [paintedLayer], running: true });

    await clearButton().click();
    const text = () => confirmDialog().element().textContent;
    await expect.element(confirmDialog().getByText('The 1 staged result goes too.')).toBeVisible();
    expect(text()).toContain('History keeps a restore point');
    expect(text()).toContain(
      'A queue item for this canvas is still queued or running. Its results would be thrown away.'
    );
    await expect.element(confirmDialog().getByRole('checkbox', { name: 'Stop it now' })).toBeChecked();
    await confirmDialog().getByRole('button', { exact: true, name: 'Clear canvas' }).click();

    await expect.poll(() => store.getState().projects[0]!.queue.items[0]?.status).toBe('cancelled');
    const canvas = store.getState().projects[0]!.canvas;
    expect(canvas.document.stacks.raster).toHaveLength(0);
    expect(canvas.snapshots[0]?.stagedImages?.map((image) => image.imageName)).toEqual(['staged.png']);
  });

  it('replaces the document once confirmed, behind a restore point that brings it back', async () => {
    const { store } = await renderActions({ layers: [paintedLayer] });
    const before = store.getState().projects[0]!.canvas.documentRevision;
    expect(store.getState().projects[0]!.canvas.document.stacks.raster).toHaveLength(1);

    await clearButton().click();
    await expect.element(confirmDialog().getByText(/Removes the 1 layer/)).toBeVisible();
    await confirmDialog().getByRole('button', { exact: true, name: 'Clear canvas' }).click();

    await expect.element(confirmDialog()).not.toBeInTheDocument();
    await expect.poll(() => store.getState().projects[0]!.canvas.document.stacks.raster).toHaveLength(0);
    const canvas = store.getState().projects[0]!.canvas;
    expect(canvas.documentRevision).toBeGreaterThan(before);
    expect(canvas.snapshots).toHaveLength(1);
    expect(canvas.snapshots[0]).toMatchObject({ kind: 'clear-canvas', name: 'Before clearing the canvas' });

    store.commands.canvas.apply(store.getState().activeProjectId, {
      snapshotId: canvas.snapshots[0]!.id,
      type: 'restoreCanvasSnapshot',
    });
    expect(store.getState().projects[0]!.canvas.document.stacks.raster.map((node) => node.id)).toEqual(['painted']);
  });

  it('brings the cleared canvas back from its Undo, even after a selection or bbox change', async () => {
    const { store } = await renderActions({ layers: [paintedLayer] });
    await clearConfirmed(store);
    store.commands.canvas.apply(store.getState().activeProjectId, {
      bbox: { height: 32, width: 32, x: 8, y: 8 },
      type: 'setCanvasBbox',
    });

    undoNewest(store);

    await expect.poll(() => rasterIds(store)).toEqual(['painted']);
  });

  it('keeps what was painted since a clear when its Undo runs late, and says the canvas was not restored', async () => {
    const { store } = await renderActions({ layers: [paintedLayer] });
    await clearConfirmed(store);
    addLayer(store, 'later');

    undoNewest(store);

    await expect.poll(() => store.getState().notifications[0]?.title).toBe('Canvas not restored');
    expect(store.getState().notifications[0]?.message).toBe(
      'The canvas changed since. Its restore point is still in History.'
    );
    expect(rasterIds(store)).toEqual(['later']);
    expect(store.getState().projects[0]!.canvas.snapshots.map((point) => point.kind)).toEqual(['clear-canvas']);
  });

  it('refuses a restore Undo when a stroke still uploading on a layer lands while the Undo saves it', async () => {
    const { store } = await renderActions({ layers: [paintedLayer] });
    await clearConfirmed(store);
    addLayer(store, 'after-clear');
    await act(() => restorePointRef.current!(store.getState().projects[0]!.canvas.snapshots[0]!));
    expect(rasterIds(store)).toEqual(['painted']);
    const projectId = store.getState().activeProjectId;
    const lifecycle = harness.engine!.lifecycle;
    const flush = lifecycle.flushPendingUploads.bind(lifecycle);
    vi.spyOn(lifecycle, 'flushPendingUploads').mockImplementationOnce(async (options) => {
      await flush(options);
      store.commands.canvas.apply(projectId, { id: 'painted', patch: { opacity: 0.5 }, type: 'updateCanvasLayer' });
    });

    undoNewest(store);

    await expect.poll(() => store.getState().notifications[0]?.title).toBe('Canvas not restored');
    expect(store.getState().notifications[0]?.message).toBe(
      'The canvas changed since. Its restore point is still in History.'
    );
    const layers = store.getState().projects[0]!.canvas.document.stacks.raster;
    expect(layers.map((node) => [node.id, 'opacity' in node ? node.opacity : null])).toEqual([['painted', 0.5]]);
  });

  it('is unavailable while editing is locked, says why, and its command opens nothing then', async () => {
    harness.editingLocked = true;
    const { commands } = await renderActions();

    await expect.element(clearButton()).toHaveAttribute('aria-disabled', 'true');
    await clearButton().click({ force: true });
    expect(document.querySelector('[role="alertdialog"]')).toBeNull();
    await act(() => commands.find((command) => command.id === 'canvas.newSession')!.handler!());
    expect(document.querySelector('[role="alertdialog"]')).toBeNull();
  });
});

describe('CanvasHeaderActions zoom menu', () => {
  const item = (name: string) => page.getByRole('menuitem', { exact: true, name });

  it('names the current zoom, leads with Fit to view, and offers every preset but the current one', async () => {
    await renderActions();
    const fitToView = vi.spyOn(harness.engine!.viewport, 'fitToView');

    await page.getByRole('button', { exact: true, name: 'Zoom 100%' }).click();
    const labels = [...document.querySelectorAll('[role="menuitem"]')].map((node) => node.textContent?.trim());
    expect(labels).toEqual([
      'Fit to view',
      '25%',
      '33%',
      '50%',
      '67%',
      '75%',
      '100%',
      '125%',
      '150%',
      '200%',
      '300%',
      '400%',
      '500%',
    ]);
    await expect.element(item('100%')).toHaveAttribute('aria-disabled', 'true');
    await item('200%').click();

    const trigger = page.getByRole('button', { exact: true, name: 'Zoom 200%' });
    await expect.element(trigger).toHaveTextContent('200%');
    await trigger.click();
    await expect.element(item('200%')).toHaveAttribute('aria-disabled', 'true');
    await item('Fit to view').click();
    expect(fitToView).toHaveBeenCalledOnce();
  });
});

describe('CanvasHeaderActions save to gallery', () => {
  const saveTrigger = () => page.getByRole('button', { exact: true, name: 'Save to Gallery' });
  const item = (name: RegExp) => page.getByRole('menuitem', { name });
  const menuItems = () => [...document.querySelectorAll('[role="menuitem"]')];

  it('leads with the frame, then the canvas, each saving its own region', async () => {
    harness.save.mockResolvedValue(undefined);
    harness.describeDestination.mockResolvedValue('Board B');
    await renderActions();

    await saveTrigger().click();
    await expect.element(item(/^Save Frame to Gallery/)).toBeVisible();
    expect(menuItems().map((node) => node.querySelector('p')?.textContent)).toEqual([
      'Save Frame to Gallery',
      'Save Canvas to Gallery',
    ]);
    await item(/^Save Canvas to Gallery/).click();
    expect(harness.save).toHaveBeenCalledExactlyOnceWith('canvas');

    await saveTrigger().click();
    await item(/^Save Frame to Gallery/).click();
    expect(harness.save).toHaveBeenLastCalledWith('bbox');
    expect(harness.save).toHaveBeenCalledTimes(2);
  });

  it('drops a destination answered for the project that was open before a switch', async () => {
    let answer: ((label: string | null) => void) | undefined;
    harness.describeDestination.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          answer = resolve;
        })
    );
    harness.describeDestination.mockImplementation(() => new Promise<string | null>(() => {}));
    const { rerender } = await renderActions();
    const { height, width } = activeProject().canvas.document.bbox;
    const frameItem = item(/^Save Frame to Gallery/);

    await saveTrigger().click();
    await expect.element(frameItem).toBeVisible();
    await userEvent.keyboard('{Escape}');
    harness.engine = { ...harness.engine!, projectId: 'project-2' } as CanvasEngine;
    await rerender();
    await act(() => answer!('Board of project 1'));

    await saveTrigger().click();
    await expect.element(frameItem).toHaveTextContent(`Save Frame to Gallery${width} × ${height}`);
  });

  it('says it is saving, and takes no second save, while one is in flight', async () => {
    harness.isSaving = true;
    await renderActions();

    const trigger = page.getByRole('button', { exact: true, name: 'Saving…' });
    await expect.element(trigger).toBeDisabled();
    await expect.element(trigger).toHaveAttribute('aria-busy', 'true');
    expect(saveTrigger().query()).toBeNull();
  });
});

const frameTrigger = () => page.getByRole('button', { exact: true, name: 'Frame' });
const menuItem = (name: RegExp) => page.getByRole('menuitem', { name });
const menuLabels = () =>
  [...document.querySelectorAll('[role="menuitem"]')].map((node) => node.querySelector('p')?.textContent);

const sizedLayer = (width: number, height: number, x = 0, y = 0): CanvasLayerContract =>
  ({
    ...paintedLayer,
    source: { ...paintedLayer.source, image: { height, imageName: 'painted.png', width } },
    transform: { ...paintedLayer.transform, x, y },
  }) as CanvasLayerContract;

describe('CanvasHeaderActions frame menu', () => {
  it('lists Edit frame, the fits with their keys, then what the frame can become', async () => {
    await renderActions();

    await frameTrigger().click();
    expect(menuLabels()).toEqual([
      'Edit frame',
      'Fit frame to layers',
      'Fit frame to masks',
      'New global reference image',
      'New regional reference image',
      'New control layer',
      'New raster layer',
    ]);
    const layerKeys = [
      ...menuItem(/^Fit frame to layers/)
        .element()
        .querySelectorAll('kbd'),
    ];
    expect(layerKeys).toHaveLength(2);
    expect(layerKeys.at(-1)?.textContent).toBe('n');
    expect(
      [
        ...menuItem(/^Fit frame to masks/)
          .element()
          .querySelectorAll('kbd'),
      ].at(-1)?.textContent
    ).toBe('b');
    for (const node of document.querySelectorAll('[role="menuitem"]')) {
      expect(node.querySelectorAll('p')).toHaveLength(1);
    }
  });
});

describe('CanvasHeaderActions use frame as', () => {
  it('groups the Create from frame items as snapshots and runs the shared create', async () => {
    harness.create.mockResolvedValue(undefined);
    await renderActions();

    await frameTrigger().click();
    const group = page.getByRole('group', { name: /^Use frame as/ });
    await expect.element(group).toMatchTextContent(/Snapshot, won’t update/);
    expect(group.getByRole('menuitem').elements()).toHaveLength(4);

    await menuItem(/^New control layer/).click();
    expect(harness.create).toHaveBeenCalledExactlyOnceWith('control');
  });
});

describe('CanvasHeaderActions undo and redo', () => {
  const historyButton = (name: string) => page.getByRole('button', { exact: true, name });
  const fitFrame = async () => {
    await frameTrigger().click();
    await menuItem(/^Fit frame to layers/).click();
    await expect.element(historyButton('Undo: Fit generation frame')).not.toHaveAttribute('aria-disabled');
  };

  it('keep keyboard focus on Undo when it steps back over the last edit, and step no further', async () => {
    await renderActions({ canvasValues: { snapToGrid: false }, layers: [sizedLayer(128, 96)] });
    await fitFrame();
    const undo = historyButton('Undo: Fit generation frame').element() as HTMLButtonElement;

    undo.focus();
    await userEvent.keyboard('{Enter}');

    await expect.element(historyButton('Undo')).toHaveAttribute('aria-disabled', 'true');
    expect(document.activeElement).toBe(undo);
    await userEvent.keyboard('{Enter}');
    await expect.element(historyButton('Redo: Fit generation frame')).toBeInTheDocument();
    expect(harness.engine!.stores.canUndo.get()).toBe(false);
  });
});

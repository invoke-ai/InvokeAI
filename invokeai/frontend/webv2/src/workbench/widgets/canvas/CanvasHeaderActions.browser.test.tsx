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
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';

import type * as EngineStoreHooks from './engineStoreHooks';

/** The actions read the real store and drive a real engine; only the editing lock and gallery save are stubbed. */
const harness = vi.hoisted(() => ({
  editingLocked: false,
  engine: null as CanvasEngine | null,
  isSaving: false,
  save: vi.fn<(region: 'bbox' | 'canvas') => Promise<void>>(),
  store: null as ReturnType<typeof createWorkbenchStore> | null,
}));
const activeProject = () => harness.store!.getState().projects[0]!;

vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContextModule>()),
  useActiveProjectId: () => activeProject().id,
  useActiveProjectSelector: (selector: (project: Project) => unknown) => selector(activeProject()),
  useOptionalWorkbenchCommands: () => harness.store!.commands,
  useWorkbenchCommands: () => harness.store!.commands,
  useWorkbenchQueries: () => harness.store!.queries,
}));
vi.mock('./engineStoreHooks', async (importOriginal) => ({
  ...(await importOriginal<typeof EngineStoreHooks>()),
  useCanvasDocumentEditingLocked: () => harness.editingLocked,
}));
vi.mock('@workbench/canvas-operations/react', () => ({ useCanvasEngine: () => harness.engine }));
// Clear canvas looks the project's engine up when it runs, as an Undo arriving later would.
vi.mock('@workbench/canvas-operations/api', async (importOriginal) => ({
  ...(await importOriginal<typeof CanvasOperationsApi>()),
  getCanvasEngine: () => harness.engine ?? undefined,
}));
vi.mock('./useCanvasGallerySave', () => ({
  useCanvasGallerySave: () => ({ isSaving: harness.isSaving, save: harness.save }),
}));

import { CanvasHeaderActions } from './CanvasHeaderActions';

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
  Object.assign(harness, { editingLocked: false, engine: null, isSaving: false, store: null });
  harness.save.mockReset();
  root = null;
  host = null;
});

interface Registration {
  defaultKeys?: string[];
  handler?: () => void;
  id: string;
  title: string;
}

/** A painted layer, so a cleared document differs from the one it replaced. */
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

/**
 * Renders the actions on a fresh project holding `layers` (none by default) and the canvas widget values
 * `canvasValues`, recording the commands and hotkeys they register. `running` queues a canvas batch still generating
 * for this canvas, whose first result is staged.
 */
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
  await act(() =>
    root!.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <CanvasHeaderActions {...({ runtime } as unknown as WidgetViewProps)} />
        </ChakraProvider>
      </I18nextProvider>
    )
  );
  return { commands, hotkeys, store };
};

const clearButton = () => page.getByRole('button', { exact: true, name: 'Clear canvas' });
const confirmDialog = () => page.getByRole('alertdialog');

describe('CanvasHeaderActions clear canvas', () => {
  it('asks to clear the canvas on one click, without a menu, even when the canvas is empty', async () => {
    await renderActions();

    await clearButton().click();

    await expect.element(confirmDialog()).toBeVisible();
    await expect.element(confirmDialog().getByRole('heading', { name: 'Clear the canvas?' })).toBeVisible();
    await expect.element(confirmDialog().getByRole('button', { exact: true, name: 'Clear canvas' })).toBeVisible();
    // The new project's empty inpaint mask is its one layer.
    await expect
      .element(confirmDialog().getByText('Removes the 1 layer and starts a fresh canvas with an empty inpaint mask.'))
      .toBeVisible();
    // A blank canvas has nothing to bring back, so the dialog promises no restore point.
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

  it('leaves the generation running once "Stop it now" is unticked', async () => {
    const { store } = await renderActions({ layers: [paintedLayer], running: true });

    await clearButton().click();
    await confirmDialog().getByText('Stop it now').click();
    await expect.element(confirmDialog().getByRole('checkbox', { name: 'Stop it now' })).not.toBeChecked();
    await confirmDialog().getByRole('button', { exact: true, name: 'Clear canvas' }).click();

    await expect.poll(() => store.getState().projects[0]!.canvas.document.stacks.raster).toHaveLength(0);
    expect(store.getState().projects[0]!.queue.items[0]?.status).toBe('running');
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

  it('is unavailable while editing is locked, says why, and its command opens nothing then', async () => {
    harness.editingLocked = true;
    const { commands } = await renderActions();

    await expect.element(clearButton()).toHaveAttribute('aria-disabled', 'true');
    await clearButton().click({ force: true });
    expect(document.querySelector('[role="alertdialog"]')).toBeNull();
    await act(() => commands.find((command) => command.id === 'canvas.newSession')!.handler!());
    expect(document.querySelector('[role="alertdialog"]')).toBeNull();
  });

  it('registers Clear canvas on Mod+Alt+N under its saved command id, opening the same confirm', async () => {
    const { commands, hotkeys } = await renderActions();

    expect(hotkeys.find((hotkey) => hotkey.id === 'canvas.newSession')).toMatchObject({
      defaultKeys: ['mod+alt+n'],
      title: 'Clear canvas',
    });
    await act(() => commands.find((command) => command.id === 'canvas.newSession')!.handler!());

    await expect.element(confirmDialog()).toBeVisible();
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

describe('CanvasHeaderActions fit frame to masks', () => {
  it("pads the masks by the project's mask blur plus 8 before snapping outward to the grid", async () => {
    const mask: CanvasLayerContract = {
      blendMode: 'normal',
      id: 'mask',
      isEnabled: true,
      isLocked: false,
      mask: { bitmap: { height: 40, imageName: 'mask.png', width: 40 }, fill: { color: '#e07575', style: 'diagonal' } },
      name: 'Mask',
      opacity: 1,
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 100, y: 100 },
      type: 'inpaint_mask',
    };
    const { store } = await renderActions({ canvasValues: { maskBlur: 40, snapToGrid: true }, layers: [mask] });

    await page.getByRole('button', { exact: true, name: 'Fit frame to masks' }).click();

    // Blur 40 pads 48: 52..188, on the default 8 px grid 48..192. The default blur 16 would give 72..168.
    await expect
      .poll(() => store.getState().projects[0]!.canvas.document.bbox)
      .toEqual({ height: 144, width: 144, x: 48, y: 48 });
  });
});

describe('CanvasHeaderActions save to gallery', () => {
  const saveTrigger = () => page.getByRole('button', { exact: true, name: 'Save to Gallery' });
  const item = (name: string) => page.getByRole('menuitem', { exact: true, name });

  it('opens one menu that saves the canvas or the frame, each to its own region', async () => {
    harness.save.mockResolvedValue(undefined);
    await renderActions();

    await saveTrigger().click();
    await expect.element(item('Save Frame to Gallery')).toBeVisible();
    await item('Save Canvas to Gallery').click();
    expect(harness.save).toHaveBeenCalledExactlyOnceWith('canvas');

    await saveTrigger().click();
    await item('Save Frame to Gallery').click();
    expect(harness.save).toHaveBeenLastCalledWith('bbox');
    expect(harness.save).toHaveBeenCalledTimes(2);
  });

  it('is disabled while editing is locked', async () => {
    harness.editingLocked = true;
    await renderActions();

    await expect.element(saveTrigger()).toBeDisabled();
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

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
import { page, userEvent } from 'vitest/browser';

import type * as EngineStoreHooks from './engineStoreHooks';

/** The actions read the real store and drive a real engine; only the editing lock and gallery save are stubbed. */
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

const INSTANCE_ID = 'canvas-1';

/** Stands in for the canvas view: the instance's focusable surface. */
const CanvasViewStandIn = () => (
  <div data-hotkey-widget-instance-id={INSTANCE_ID}>
    <div {...{ [CANVAS_SURFACE_ATTR]: '' }} data-testid="surface" tabIndex={-1} />
  </div>
);

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
  // Rendering again picks up a swapped `harness.engine`, as a project switch would.
  const rerender = () =>
    act(() =>
      root!.render(
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <CanvasHeaderActions {...({ instance: { id: INSTANCE_ID }, runtime } as unknown as WidgetViewProps)} />
            <CanvasViewStandIn />
          </ChakraProvider>
        </I18nextProvider>
      )
    );
  await rerender();
  return { commands, hotkeys, rerender, store };
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

    await page.getByRole('button', { exact: true, name: 'Frame' }).click();
    await page.getByRole('menuitem', { name: /^Fit frame to masks/ }).click();

    // Blur 40 pads 48: 52..188, on the default 8 px grid 48..192. The default blur 16 would give 72..168.
    await expect
      .poll(() => store.getState().projects[0]!.canvas.document.bbox)
      .toEqual({ height: 144, width: 144, x: 48, y: 48 });
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

  it('says what each save holds and the board it goes to, resolved as the menu opens', async () => {
    harness.describeDestination.mockResolvedValue('Board B');
    await renderActions({
      layers: [
        {
          ...paintedLayer,
          source: { ...paintedLayer.source, image: { height: 300, imageName: 'painted.png', width: 200 } },
        } as CanvasLayerContract,
      ],
    });
    const { height, width } = activeProject().canvas.document.bbox;

    await saveTrigger().click();
    await expect
      .element(item(/^Save Frame to Gallery/))
      .toHaveTextContent(`Save Frame to Gallery${width} × ${height} · Board B`);
    await expect
      .element(item(/^Save Canvas to Gallery/))
      .toHaveTextContent('Save Canvas to Gallery200 × 300 · Board B');
    expect(harness.describeDestination).toHaveBeenCalledOnce();
  });

  it('drops what it cannot name: an empty canvas has no size, an unknown board no destination', async () => {
    harness.describeDestination.mockResolvedValue(null);
    await renderActions();
    const { height, width } = activeProject().canvas.document.bbox;

    await saveTrigger().click();
    await expect.element(item(/^Save Frame to Gallery/)).toHaveTextContent(`Save Frame to Gallery${width} × ${height}`);
    await expect.element(item(/^Save Canvas to Gallery/)).toHaveTextContent('Save Canvas to Gallery');
  });

  it('takes no save while a create from the frame runs', async () => {
    harness.isCreating = true;
    await renderActions();

    await expect.element(saveTrigger()).toBeDisabled();
  });

  it('names the board from the latest opening, ignoring an earlier answer that arrives late', async () => {
    const answers: ((label: string | null) => void)[] = [];
    harness.describeDestination.mockImplementation(
      () =>
        new Promise((resolve) => {
          answers.push(resolve);
        })
    );
    await renderActions();
    const { height, width } = activeProject().canvas.document.bbox;
    const frameItem = item(/^Save Frame to Gallery/);

    await saveTrigger().click();
    await expect.element(frameItem).toBeVisible();
    await userEvent.keyboard('{Escape}');
    await saveTrigger().click();
    await expect.element(frameItem).toBeVisible();
    expect(answers).toHaveLength(2);

    await act(() => answers[1]!('Board New'));
    await expect.element(frameItem).toHaveTextContent(`Save Frame to Gallery${width} × ${height} · Board New`);
    await act(() => answers[0]!('Board Old'));
    await expect.element(frameItem).toHaveTextContent(`Save Frame to Gallery${width} × ${height} · Board New`);
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

const frameTrigger = () => page.getByRole('button', { exact: true, name: 'Frame' });
const menuItem = (name: RegExp) => page.getByRole('menuitem', { name });
const menuLabels = () =>
  [...document.querySelectorAll('[role="menuitem"]')].map((node) => node.querySelector('p')?.textContent);

/** A raster layer of `width` × `height` at (x, y). */
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
    // Single-line items: no hint under any of them.
    for (const node of document.querySelectorAll('[role="menuitem"]')) {
      expect(node.querySelectorAll('p')).toHaveLength(1);
    }
  });

  it('Edit frame selects the Frame tool and hands focus to the canvas', async () => {
    await renderActions();
    expect(harness.engine!.interaction.get('activeTool')).not.toBe('bbox');

    await frameTrigger().click();
    await menuItem(/^Edit frame/).click();

    expect(harness.engine!.interaction.get('activeTool')).toBe('bbox');
    await expect.poll(() => document.activeElement).toBe(page.getByTestId('surface').element());
  });

  it('fits the frame to the layers, and offers no mask fit without a painted mask', async () => {
    const { store } = await renderActions({
      canvasValues: { snapToGrid: false },
      layers: [sizedLayer(128, 96, 16, 32)],
    });

    await frameTrigger().click();
    // The new project's inpaint mask is empty, so there is nothing to fit the frame to.
    await expect.element(menuItem(/^Fit frame to masks/)).toHaveAttribute('aria-disabled', 'true');
    await menuItem(/^Fit frame to layers/).click();

    await expect
      .poll(() => store.getState().projects[0]!.canvas.document.bbox)
      .toEqual({ height: 96, width: 128, x: 16, y: 32 });
  });

  it('keeps the fits unavailable while an operation holds the document', async () => {
    harness.editingLocked = true;
    await renderActions({ canvasValues: { snapToGrid: false }, layers: [sizedLayer(128, 96)] });

    await frameTrigger().click();
    await expect.element(menuItem(/^Fit frame to layers/)).toHaveAttribute('aria-disabled', 'true');
    await expect.element(menuItem(/^Edit frame/)).not.toHaveAttribute('aria-disabled', 'true');
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

  it('says why it cannot run while an operation holds the document, and creates nothing', async () => {
    harness.editingLocked = true;
    await renderActions();

    await frameTrigger().click();
    const raster = menuItem(/^New raster layer/);
    await expect.element(raster).toHaveAttribute('aria-disabled', 'true');
    await expect.element(raster).toHaveAccessibleDescription('Finish or cancel the current canvas operation first.');
    await raster.click({ force: true });
    expect(harness.create).not.toHaveBeenCalled();
  });

  it('waits while another composite runs', async () => {
    harness.isCreating = true;
    await renderActions();

    await frameTrigger().click();
    await expect.element(menuItem(/^New raster layer/)).toHaveAttribute('aria-disabled', 'true');
  });
});

describe('CanvasHeaderActions undo and redo', () => {
  it('name the step they would undo or redo, and plain Undo and Redo without one', async () => {
    await renderActions({ canvasValues: { snapToGrid: false }, layers: [sizedLayer(128, 96)] });
    await expect.element(page.getByRole('button', { exact: true, name: 'Undo' })).toBeDisabled();
    await expect.element(page.getByRole('button', { exact: true, name: 'Redo' })).toBeDisabled();

    await frameTrigger().click();
    await menuItem(/^Fit frame to layers/).click();

    const undo = page.getByRole('button', { exact: true, name: 'Undo: Fit generation frame' });
    await expect.element(undo).toBeEnabled();
    await undo.click();
    await expect.element(page.getByRole('button', { exact: true, name: 'Redo: Fit generation frame' })).toBeEnabled();
    await expect.element(page.getByRole('button', { exact: true, name: 'Undo' })).toBeDisabled();
  });
});

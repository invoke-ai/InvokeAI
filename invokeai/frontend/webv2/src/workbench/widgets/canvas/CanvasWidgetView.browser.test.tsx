import type * as DndKitCore from '@dnd-kit/core';
import type { CanvasStagingCandidateContract } from '@workbench/canvas-engine/contracts';
import type { CanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import type { Project } from '@workbench/projectContracts';
import type { WorkbenchQueueItem as QueueItem } from '@workbench/queueHistoryContracts';
import type { WidgetViewProps } from '@workbench/widgetContracts';
import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
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

/** The view reads the real store and drives its real commands; only unrelated chrome is stubbed. */
const harness = vi.hoisted(() => ({
  engine: null as CanvasEngine | null,
  store: null as ReturnType<typeof createWorkbenchStore> | null,
}));
const activeProject = () => harness.store!.getState().projects[0]!;

vi.mock('@dnd-kit/core', async (importOriginal) => ({
  ...(await importOriginal<typeof DndKitCore>()),
  useDndMonitor: () => undefined,
}));
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
vi.mock('./engineStoreHooks', () => ({ useCanvasOperation: () => null }));
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
  Object.assign(harness, { engine: null, store: null });
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

/** Stages the first image of a three-image batch whose remaining images are `status`. */
const renderView = async (status: QueueItem['status']) => {
  const initial = createInitialWorkbenchState();
  const base = initial.projects[0]!;
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
  store.commands.canvas.appendStagingCandidate({ candidate, projectId });
  store.commands.canvas.apply(projectId, { imageIndex: 0, type: 'setStagedImageIndex' });
  harness.store = store;
  harness.engine = createCanvasEngine({
    backend: createTestStubRasterBackend(),
    ensureProjectOnServer: () => Promise.resolve(),
    imageResolver: () => Promise.resolve(new Blob()),
    mutationPort: createCanvasProjectMutationPort(store, projectId),
    projectId,
    reportError: () => undefined,
  });
  host = document.createElement('div');
  host.style.height = '600px';
  host.style.width = '1000px';
  document.body.append(host);
  root = createRoot(host);
  const runtime = { commands: { register: () => () => undefined }, hotkeys: { register: () => () => undefined } };
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
  return { projectId, store };
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

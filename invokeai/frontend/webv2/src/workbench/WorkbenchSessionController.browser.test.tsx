import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

import type { WorkbenchSearch } from './projects/session';

const harness = vi.hoisted(() => ({
  applyCanvas: vi.fn(),
  applyPreset: vi.fn(),
  notify: vi.fn(),
  hydrate: vi.fn(),
  navigate: vi.fn(),
  open: vi.fn(),
  patchSettings: vi.fn(),
  rename: vi.fn(),
  requestLibraryWorkflowLoad: vi.fn(),
  setSource: vi.fn(),
  switchTo: vi.fn(),
}));

vi.mock('@tanstack/react-router', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useNavigate: () => harness.navigate,
}));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));
vi.mock('@features/workflow/react', () => ({ requestLibraryWorkflowLoad: harness.requestLibraryWorkflowLoad }));
vi.mock('./WorkbenchContext', () => ({
  useWorkbenchCommands: () => ({
    canvas: { apply: harness.applyCanvas },
    generation: { patchSettings: harness.patchSettings, setSource: harness.setSource },
    layout: { applyPreset: harness.applyPreset },
    notifications: { add: harness.notify },
    projects: { open: harness.open, rename: harness.rename, switchTo: harness.switchTo },
  }),
  useWorkbenchHasHydrated: () => true,
  useWorkbenchPersistenceService: () => ({ hydrateProjectFromServer: harness.hydrate }),
  useWorkbenchSelector: (
    select: (snapshot: { activeProject: { id: string }; projects: { id: string }[] }) => unknown
  ) => select({ activeProject: { id: 'draft' }, projects: [{ id: 'previous' }, { id: 'draft' }] }),
}));

import { accountLifecycle } from '@platform/state/accountLifecycle';

import { takeCanvasStartImage } from './projects/canvasStartImage';
import { requestNewProject, takeNewProjectRequest } from './projects/newProjectRequest';
import { WorkbenchSessionController } from './WorkbenchSessionController';

const START_IMAGE = {
  boardId: 'none',
  height: 750,
  imageCategory: 'user' as const,
  imageName: 'pasted.png',
  imageUrl: '/images/pasted.png',
  queuedAt: '',
  sourceQueueItemId: '',
  starred: false,
  thumbnailUrl: '/thumbs/pasted.png',
  width: 1000,
};

let host: HTMLDivElement;
let root: Root;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeEach(() => {
  vi.resetAllMocks();
  accountLifecycle.activate('session-test', ':user:session-test');
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  takeNewProjectRequest();
  takeCanvasStartImage('draft');
  accountLifecycle.invalidate();
});

const render = (search: WorkbenchSearch) => act(() => root.render(<WorkbenchSessionController search={search} />));

it('consumes an open-project link so reload cannot override a later project selection', async () => {
  await render({ project: 'previous' });
  expect(harness.switchTo).toHaveBeenCalledWith('previous');
  expect(harness.navigate).toHaveBeenCalledWith({ replace: true, search: {}, to: '/app' });
});

it('consumes a library link only after the requested project loads', async () => {
  let resolve!: (result: unknown) => void;
  harness.hydrate.mockReturnValue(
    new Promise((done) => {
      resolve = done;
    })
  );
  await render({ project: 'library' });
  expect(harness.navigate).not.toHaveBeenCalled();
  const project = { id: 'library' };
  await act(() => resolve({ project, status: 'loaded' }));
  expect(harness.open).toHaveBeenCalledWith(project);
  expect(harness.navigate).toHaveBeenCalledWith({ replace: true, search: {}, to: '/app' });
});

it('does not consume a newer link when an older load finishes', async () => {
  let resolve!: (result: unknown) => void;
  harness.hydrate.mockReturnValue(
    new Promise((done) => {
      resolve = done;
    })
  );
  await render({ project: 'library' });
  await render({});
  await act(() => resolve({ project: { id: 'library' }, status: 'loaded' }));
  expect(harness.open).not.toHaveBeenCalled();
  expect(harness.navigate).not.toHaveBeenCalled();
});

it('applies the intent, then the staged dialog choices, to the new draft before stripping the link', async () => {
  requestNewProject({
    generateSettings: { height: 840, width: 1256 },
    intent: 'canvas',
    libraryWorkflow: null,
    name: 'Portraits',
    startImage: null,
  });

  await render({ intent: 'canvas', new: true });

  expect(harness.applyPreset).toHaveBeenCalledWith('edit');
  expect(harness.setSource).toHaveBeenCalledWith('canvas');
  expect(harness.rename).toHaveBeenCalledWith('draft', 'Portraits');
  expect(harness.patchSettings).toHaveBeenCalledWith({ height: 840, width: 1256 }, 'draft', 'system');
  expect(harness.requestLibraryWorkflowLoad).not.toHaveBeenCalled();
  // A blank Canvas start still sizes the document to the chosen generate size.
  expect(harness.applyCanvas.mock.calls).toEqual([
    ['draft', { height: 840, type: 'resizeCanvasDocument', width: 1256 }],
    ['draft', { bbox: { height: 840, width: 1256, x: 0, y: 0 }, type: 'setCanvasBbox' }],
  ]);
  expect(harness.rename.mock.invocationCallOrder[0]).toBeGreaterThan(harness.setSource.mock.invocationCallOrder[0]!);
  expect(harness.navigate).toHaveBeenCalledWith({ replace: true, search: {}, to: '/app' });
  expect(takeNewProjectRequest()).toBeNull();
});

it('asks the workflow widget to load a picked library workflow over the blank one', async () => {
  requestNewProject({
    generateSettings: null,
    intent: 'workflow',
    libraryWorkflow: { name: 'Upscale pipeline', workflowId: 'lib-1' },
    name: null,
    startImage: null,
  });

  await render({ intent: 'workflow', new: true });

  expect(harness.applyPreset).toHaveBeenCalledWith('automate');
  expect(harness.rename).not.toHaveBeenCalled();
  expect(harness.patchSettings).not.toHaveBeenCalled();
  expect(harness.requestLibraryWorkflowLoad).toHaveBeenCalledWith('lib-1', 'Upscale pipeline');
  expect(harness.navigate).toHaveBeenCalledWith({ replace: true, search: {}, to: '/app' });
});

it('opens a plain intent draft when nothing was staged', async () => {
  await render({ intent: 'generate', new: true });

  expect(harness.applyPreset).toHaveBeenCalledWith('compose');
  expect(harness.rename).not.toHaveBeenCalled();
  expect(harness.patchSettings).not.toHaveBeenCalled();
  expect(harness.navigate).toHaveBeenCalledWith({ replace: true, search: {}, to: '/app' });
});

it('leaves a staged request alone for links that do not create a project', async () => {
  requestNewProject({
    generateSettings: null,
    intent: 'workflow',
    libraryWorkflow: null,
    name: 'Kept',
    startImage: null,
  });

  await render({ project: 'previous' });

  expect(harness.rename).not.toHaveBeenCalled();
  expect(takeNewProjectRequest()?.name).toBe('Kept');
});

it('drops a request staged for another start instead of landing it on this one', async () => {
  requestNewProject({
    generateSettings: { height: 840, width: 1256 },
    intent: 'canvas',
    libraryWorkflow: null,
    name: 'Stale',
    startImage: START_IMAGE,
  });

  await render({ intent: 'generate', new: true });

  expect(harness.applyPreset).toHaveBeenCalledWith('compose');
  expect(harness.rename).not.toHaveBeenCalled();
  expect(harness.patchSettings).not.toHaveBeenCalled();
  expect(harness.applyCanvas).not.toHaveBeenCalled();
  expect(takeNewProjectRequest()).toBeNull();
  expect(takeCanvasStartImage('draft')).toBeNull();
});

it('sizes a blank Canvas start to the chosen size', async () => {
  requestNewProject({
    generateSettings: { height: 768, width: 1368 },
    intent: 'canvas',
    libraryWorkflow: null,
    name: null,
    startImage: null,
  });

  await render({ intent: 'canvas', new: true });

  expect(harness.applyCanvas.mock.calls).toEqual([
    ['draft', { height: 768, type: 'resizeCanvasDocument', width: 1368 }],
    ['draft', { bbox: { height: 768, width: 1368, x: 0, y: 0 }, type: 'setCanvasBbox' }],
  ]);
  expect(takeCanvasStartImage('draft')).toBeNull();
});

it('sizes the canvas to a start image and stages it for the Canvas widget', async () => {
  requestNewProject({
    generateSettings: { height: 752, width: 1000 },
    intent: 'canvas',
    libraryWorkflow: null,
    name: null,
    startImage: START_IMAGE,
  });

  await render({ intent: 'canvas', new: true });

  expect(harness.applyCanvas.mock.calls).toEqual([
    ['draft', { height: 750, type: 'resizeCanvasDocument', width: 1000 }],
    ['draft', { bbox: { height: 750, width: 1000, x: 0, y: 0 }, type: 'setCanvasBbox' }],
  ]);
  expect(takeCanvasStartImage('draft')).toBe(START_IMAGE);
  expect(harness.navigate).toHaveBeenCalledWith({ replace: true, search: {}, to: '/app' });
});

it('ignores a start image for starts other than Canvas', async () => {
  requestNewProject({
    generateSettings: null,
    intent: 'generate',
    libraryWorkflow: null,
    name: null,
    startImage: START_IMAGE,
  });

  await render({ intent: 'generate', new: true });

  expect(harness.applyCanvas).not.toHaveBeenCalled();
  expect(takeCanvasStartImage('draft')).toBeNull();
});

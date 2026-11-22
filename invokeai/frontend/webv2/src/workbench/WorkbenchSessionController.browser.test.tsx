import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

import type { WorkbenchSearch } from './projects/session';

import { getLayoutPreset } from './layoutPresets';

const harness = vi.hoisted(() => ({
  applyPreset: vi.fn(),
  hydrate: vi.fn(),
  navigate: vi.fn(),
  open: vi.fn(),
  openInWorkspace: vi.fn(),
  setSource: vi.fn(),
  switchTo: vi.fn(),
}));

vi.mock('@tanstack/react-router', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useNavigate: () => harness.navigate,
}));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));
vi.mock('./WorkbenchContext', () => ({
  useWorkbenchCommands: () => ({
    generation: { setSource: harness.setSource },
    layout: { applyPreset: harness.applyPreset },
    notifications: { add: vi.fn() },
    projects: { open: harness.open, switchTo: harness.switchTo },
  }),
  useWorkbenchHasHydrated: () => true,
  useWorkbenchPersistenceService: () => ({ hydrateProjectFromServer: harness.hydrate }),
  // Compose as entered, its route locked to Upscale: the job is still Generate.
  useWorkbenchQueries: () => ({
    getSnapshot: () => ({
      account: {},
      activeProject: {
        ...getLayoutPreset('compose').snapshot,
        invocation: { sourceId: 'upscale', sourceLocked: true },
      },
    }),
  }),
  useWorkbenchSelector: () => ['previous'],
}));
vi.mock('@workbench/workspace/useWorkspaceOpen', () => ({ useWorkspaceOpen: () => harness.openInWorkspace }));

import { WorkbenchSessionController } from './WorkbenchSessionController';

let host: HTMLDivElement;
let root: Root;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeEach(() => {
  vi.resetAllMocks();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

const render = (search: WorkbenchSearch) => act(() => root.render(<WorkbenchSessionController search={search} />));

it('enters an intent’s workspace and leaves a source that is its job to the entry', async () => {
  await render({ intent: 'generate', new: true });

  expect(harness.applyPreset).toHaveBeenCalledWith('compose');
  expect(harness.openInWorkspace).not.toHaveBeenCalled();
});

it('enters an intent’s workspace, then shows and aims at a source that is not its job', async () => {
  await render({ intent: 'upscale', new: true });

  expect(harness.applyPreset).toHaveBeenCalledWith('compose');
  expect(harness.openInWorkspace).toHaveBeenCalledWith(
    { typeId: 'upscale' },
    expect.objectContaining({ focus: false, mode: 'stay' })
  );
  harness.openInWorkspace.mock.calls[0]![1].afterShown();
  expect(harness.setSource).toHaveBeenCalledWith('upscale');
});

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

import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

import type { WorkbenchSearch } from './projects/session';

const harness = vi.hoisted(() => ({
  hydrate: vi.fn(),
  navigate: vi.fn(),
  open: vi.fn(),
  switchTo: vi.fn(),
}));

vi.mock('@features/generation/react', () => ({ flushGenerateDrafts: vi.fn() }));
vi.mock('@tanstack/react-router', () => ({ useNavigate: () => harness.navigate }));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));
vi.mock('./WorkbenchContext', () => ({
  useWorkbenchCommands: () => ({
    notifications: { add: vi.fn() },
    projects: { open: harness.open, switchTo: harness.switchTo },
  }),
  useWorkbenchHasHydrated: () => true,
  useWorkbenchPersistenceService: () => ({ hydrateProjectFromServer: harness.hydrate }),
  useWorkbenchSelector: () => ['previous'],
}));

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

import type * as apiModule from '@workbench/projects/api';
import type { ProjectSummaryDTO } from '@workbench/projects/api';

import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle, captureAccountScope } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { MAX_SUPPORTED_CANVAS_SCHEMA_VERSION } from '@workbench/canvasSchemaVersion';
import { seedProjectLibrary } from '@workbench/projects/library';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

const harness = vi.hoisted(() => ({
  hydrate: vi.fn(),
  listProjects: vi.fn(),
  notifyError: vi.fn(),
  open: vi.fn(),
  openProjectIds: [] as string[],
}));

// The project library is real; only its HTTP listing (and the cover index it reads alongside) is replaced.
vi.mock('@workbench/projects/api', async (importOriginal) => ({
  ...(await importOriginal<typeof apiModule>()),
  getClientStateValue: () => Promise.resolve(null),
  listProjects: harness.listProjects,
}));
vi.mock('@workbench/WorkbenchContext', () => ({
  useWorkbenchCommands: () => ({ projects: { open: harness.open } }),
  useWorkbenchPersistenceService: () => ({ adoptProjectRecord: vi.fn(), hydrateProjectFromServer: harness.hydrate }),
  useWorkbenchSelector: <T,>(selector: (snapshot: { projects: { id: string }[] }) => T) =>
    selector({ projects: harness.openProjectIds.map((id) => ({ id })) }),
}));
vi.mock('@workbench/useNotify', () => ({ useNotify: () => ({ error: harness.notifyError }) }));
vi.mock('@workbench/projects/useProjectFileActions', () => ({ useImportProjectFile: () => vi.fn() }));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

import { OpenProjectDialog } from './OpenProjectDialog';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const dto = (id: string, name: string, updatedAt: string, minimumCanvasSchemaVersion = 3): ProjectSummaryDTO => ({
  board_id: `board-${id}`,
  created_at: '2026-10-01T08:00:00.000Z',
  minimum_canvas_schema_version: minimumCanvasSchemaVersion,
  minimum_document_schema_version: 3,
  name,
  project_id: id,
  revision: 1,
  updated_at: updatedAt,
  workspace_updated_at: null,
});

// Listed most recently edited first.
const harbor = dto('harbor', 'Harbor Tower', '2026-10-30T08:00:00.000Z');
const mahogany = dto('mahogany', 'Mahogany House', '2026-10-29T08:00:00.000Z');
const site = dto('site', 'Site plan', '2026-10-28T08:00:00.000Z');
const future = dto('future', 'From the future', '2026-10-31T08:00:00.000Z', MAX_SUPPORTED_CANVAS_SCHEMA_VERSION + 1);

const onClose = vi.fn();
let host: HTMLDivElement;
let root: Root;

/** Opens the dialog, whose mount refreshes the library from `listProjects`, and lets that listing settle. */
const render = async () => {
  await act(async () => {
    root.render(
      <ChakraProvider value={system}>
        <OpenProjectDialog isOpen onClose={onClose} />
      </ChakraProvider>
    );
    await new Promise<void>((resolve) => {
      setTimeout(resolve, 0);
    });
  });
};

const dialog = () => document.querySelector<HTMLElement>('[role="dialog"]')!;
const list = () => dialog().querySelector<HTMLElement>('[role="list"]');
const buttonNamed = (label: string) =>
  [...dialog().querySelectorAll<HTMLButtonElement>('button')].find((button) => button.textContent === label)!;
const rowNamed = (name: string) =>
  [...dialog().querySelectorAll<HTMLElement>('[role="listitem"]')].find((row) => row.textContent?.includes(name))!;
const primaryOf = (name: string) => rowNamed(name).querySelector<HTMLButtonElement>('[data-list-primary]')!;

beforeEach(() => {
  accountLifecycle.activate('open-project-dialog-browser-test');
  harness.hydrate.mockReset();
  harness.hydrate.mockImplementation((id: string) => Promise.resolve({ project: { id }, status: 'loaded' }));
  harness.listProjects.mockReset();
  harness.notifyError.mockReset();
  harness.open.mockReset();
  harness.openProjectIds = [];
  onClose.mockReset();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  accountLifecycle.invalidate();
  host.remove();
});

describe('OpenProjectDialog', () => {
  it('lists the projects not already open as list items with a divider between each pair', async () => {
    harness.openProjectIds = ['mahogany'];
    harness.listProjects.mockResolvedValue([harbor, mahogany, future, site]);
    await render();

    expect(list()?.getAttribute('aria-label')).toBe('projects.openProject');
    const rows = [...list()!.querySelectorAll('[role="listitem"]')];

    expect(rows.map((row) => row.querySelector('[data-list-primary]')?.textContent)).toEqual([
      expect.stringContaining('From the future'),
      expect.stringContaining('Harbor Tower'),
      expect.stringContaining('Site plan'),
    ]);
    expect(rows.every((row) => row.textContent?.includes('projects.editedRelative'))).toBe(true);
    expect(list()!.querySelectorAll('[data-list-divider]')).toHaveLength(rows.length - 1);
  });

  it('opens a project by click and closes once it is loaded', async () => {
    harness.listProjects.mockResolvedValue([harbor, mahogany]);
    await render();

    await userEvent.click(primaryOf('Mahogany House'));

    expect(harness.hydrate).toHaveBeenCalledWith('mahogany', 'Mahogany House');
    expect(harness.open).toHaveBeenCalledWith({ id: 'mahogany' });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('keeps one tab stop, moves between projects with the arrow keys, and opens with Enter', async () => {
    harness.listProjects.mockResolvedValue([harbor, mahogany]);
    await render();

    expect(primaryOf('Harbor Tower').tabIndex).toBe(0);
    expect(primaryOf('Mahogany House').tabIndex).toBe(-1);

    await act(() => primaryOf('Harbor Tower').focus());
    await userEvent.keyboard('{ArrowDown}');
    expect(document.activeElement).toBe(primaryOf('Mahogany House'));

    await userEvent.keyboard('{Enter}');
    expect(harness.hydrate).toHaveBeenCalledWith('mahogany', 'Mahogany House');
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('starts keyboard users on the first openable project, keeping an incompatible one reachable but inert', async () => {
    harness.listProjects.mockResolvedValue([harbor, future]);
    await render();

    const incompatible = primaryOf('From the future');

    // The incompatible project is the most recent, so it is listed first; the tab stop skips it.
    expect(list()!.querySelector('[data-list-primary]')).toBe(incompatible);
    expect(primaryOf('Harbor Tower').tabIndex).toBe(0);
    expect(incompatible.tabIndex).toBe(-1);

    expect(incompatible.getAttribute('aria-disabled')).toBe('true');
    expect(incompatible.textContent).toContain('projects.requiresNewerInvoke');
    expect(incompatible.querySelector('svg')).toBeNull();
    expect(primaryOf('Harbor Tower').hasAttribute('aria-disabled')).toBe(false);
    expect(primaryOf('Harbor Tower').textContent).not.toContain('projects.requiresNewerInvoke');

    await act(() => primaryOf('Harbor Tower').focus());
    await userEvent.keyboard('{ArrowUp}');
    expect(document.activeElement).toBe(incompatible);

    await userEvent.keyboard('{Enter}');
    // Force past Playwright's actionability check: aria-disabled is exactly what is under test.
    await userEvent.click(incompatible, { force: true });
    expect(harness.hydrate).not.toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
  });

  it('holds every row while one project opens, then reports an unavailable project without closing', async () => {
    let finish: (result: unknown) => void = () => undefined;
    harness.hydrate.mockImplementation(
      () =>
        new Promise((resolve) => {
          finish = resolve;
        })
    );
    harness.listProjects.mockResolvedValue([harbor, mahogany]);
    await render();

    await userEvent.click(primaryOf('Harbor Tower'));

    expect(list()!.getAttribute('aria-busy')).toBe('true');
    expect(primaryOf('Harbor Tower').querySelector('.chakra-spinner')).not.toBeNull();
    expect(primaryOf('Mahogany House').querySelector('.chakra-spinner')).toBeNull();
    expect(primaryOf('Mahogany House').getAttribute('aria-disabled')).toBe('true');
    await userEvent.click(primaryOf('Mahogany House'), { force: true });
    expect(harness.hydrate).toHaveBeenCalledTimes(1);

    await act(async () => {
      finish({ status: 'unavailable' });
      await Promise.resolve();
    });

    expect(harness.notifyError).toHaveBeenCalledWith('projects.couldNotOpen', 'projects.couldNotOpenDescription');
    expect(onClose).not.toHaveBeenCalled();
    expect(list()!.hasAttribute('aria-busy')).toBe(false);
    expect(primaryOf('Harbor Tower').querySelector('.chakra-spinner')).toBeNull();
  });

  it('says why there is nothing to open: no saved projects, or all of them already open', async () => {
    harness.listProjects.mockResolvedValue([]);
    await render();

    expect(list()).toBeNull();
    expect(dialog().textContent).toContain('projects.noSavedProjects');

    harness.openProjectIds = ['harbor'];
    harness.listProjects.mockResolvedValue([harbor]);
    await act(() => root.unmount());
    root = createRoot(host);
    await render();

    expect(dialog().textContent).toContain('projects.allSavedAlreadyOpen');
  });

  it('shows loading before the first listing, then a failure whose retry lands focus on Import when nothing returns', async () => {
    let fail: (error: Error) => void = () => undefined;
    harness.listProjects.mockImplementation(
      () =>
        new Promise((_resolve, reject) => {
          fail = reject;
        })
    );
    await render();

    expect(dialog().querySelector('[role="status"]')?.getAttribute('aria-label')).toBe('common.loading');
    expect(dialog().textContent).not.toContain('projects.noSavedProjects');

    await act(async () => {
      fail(new Error('Server unavailable'));
      await new Promise<void>((resolve) => {
        setTimeout(resolve, 0);
      });
    });

    expect(dialog().textContent).toContain('projects.failedToLoad');
    expect(dialog().textContent).not.toContain('projects.noSavedProjects');

    harness.listProjects.mockResolvedValue([]);
    const retry = dialog().querySelector<HTMLButtonElement>('[role="alert"] button')!;
    await userEvent.click(retry);

    await expect.poll(() => dialog().textContent).toContain('projects.noSavedProjects');
    expect(harness.listProjects).toHaveBeenCalledTimes(2);
    await expect.poll(() => document.activeElement).toBe(buttonNamed('projects.importWithEllipsis'));
  });

  it('lands focus on the first openable project when a retry recovers', async () => {
    harness.listProjects.mockRejectedValue(new Error('Server unavailable'));
    await render();

    harness.listProjects.mockResolvedValue([future, harbor]);
    await userEvent.click(dialog().querySelector<HTMLButtonElement>('[role="alert"] button')!);

    await expect.poll(() => document.activeElement).toBe(primaryOf('Harbor Tower'));
  });

  it('keeps the projects listed earlier usable when a later refresh fails', async () => {
    seedProjectLibrary([harbor, mahogany], captureAccountScope());
    harness.listProjects.mockRejectedValue(new Error('Server unavailable'));
    await render();

    expect(harness.listProjects).toHaveBeenCalledTimes(1);
    expect(dialog().textContent).not.toContain('projects.failedToLoad');

    await userEvent.click(primaryOf('Mahogany House'));
    expect(harness.hydrate).toHaveBeenCalledWith('mahogany', 'Mahogany House');
  });
});

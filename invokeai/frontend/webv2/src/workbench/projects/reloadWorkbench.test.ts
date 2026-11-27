import { accountLifecycle } from '@platform/state/accountLifecycle';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

import type { WorkbenchSaveResult } from './syncedPersistence';

import { reloadWorkbench, WorkbenchReloadError, describeWorkbenchReloadError } from './reloadWorkbench';

beforeEach(() => {
  accountLifecycle.activate('reload-test', ':user:reload-test');
});
afterEach(() => {
  accountLifecycle.invalidate();
});

const fixture = () => {
  const store = createWorkbenchStore();
  const result: WorkbenchSaveResult = {
    awaitsDecision: false,
    conflicts: [],
    error: null,
    hasPendingChanges: false,
    hasRetryablePendingChanges: false,
    localDraftStatus: 'ok',
    projectBoardAssignments: [],
    shouldRetry: false,
    snapshot: { refusedProjects: [], savedAt: '2026-11-05T00:00:00Z', state: store.getState(), version: 1 },
  };
  let canvasListener = () => {};
  return {
    notifyCanvas: () => canvasListener(),
    owner: accountLifecycle.capture(),
    store,
    persistence: {
      hasClosedSession: vi.fn(() => false),
      saveWorkbench: vi.fn((_state: ReturnType<typeof store.getState>) => Promise.resolve(result)),
    },
    engines: {
      flushPendingPixels: vi.fn((_projectId: string) => Promise.resolve()),
      subscribe: (listener: () => void) => {
        canvasListener = listener;
        return () => {
          canvasListener = () => {};
        };
      },
    },
    reload: vi.fn(),
    isCurrent: () => true,
    result,
  };
};

it('flushes pixels for every open project before saving the committed edits and reloading', async () => {
  const f = fixture();
  f.store.commands.projects.create();
  f.store.commands.projects.switchTo(f.store.getSnapshot().projects[0]!.id);
  const active = f.store.getSnapshot().activeProject.id;
  f.store.commands.projects.rename(active, 'Committed edit');
  await reloadWorkbench(f);
  const ids = f.store.getSnapshot().projects.map((project) => project.id);
  expect(f.engines.flushPendingPixels.mock.calls.map(([id]) => id)).toEqual([
    ...ids,
    ...ids.filter((id) => id !== active),
    active,
  ]);
  expect(f.persistence.saveWorkbench.mock.calls[0]?.[0].projects.find((p) => p.id === active)?.name).toBe(
    'Committed edit'
  );
  expect(f.reload).toHaveBeenCalledOnce();
});

it.each(['error', 'pending', 'conflict', 'pixel', 'edit', 'account', 'removed', 'closed', 'pixel-edit'] as const)(
  'refuses document reload after %s while retaining the usable editor',
  async (reason) => {
    const f = fixture();
    if (reason === 'closed') {
      f.persistence.hasClosedSession.mockReturnValue(true);
    }
    if (reason === 'pixel') {
      f.engines.flushPendingPixels.mockRejectedValueOnce(new Error('Pixel upload failed'));
    }
    if (reason === 'error') {
      f.result.error = 'Storage unavailable';
    }
    if (reason === 'pending') {
      f.result.hasPendingChanges = true;
    }
    if (reason === 'conflict') {
      f.result.awaitsDecision = true;
    }
    f.persistence.saveWorkbench.mockImplementationOnce(() => {
      if (reason === 'pixel-edit') {
        f.notifyCanvas();
      }
      if (reason === 'edit') {
        f.store.commands.projects.rename(f.store.getSnapshot().activeProject.id, 'Newer edit');
      }
      if (reason === 'account') {
        accountLifecycle.activate('next-account', ':user:next-account');
      }
      if (reason === 'removed') {
        f.isCurrent = () => false;
      }
      return Promise.resolve(f.result);
    });
    await expect(reloadWorkbench({ ...f, isCurrent: () => f.isCurrent() })).rejects.toThrow();
    expect(f.reload).not.toHaveBeenCalled();
  }
);

it.each(['edit', 'pixel', 'view', 'partial-save'] as const)(
  'keeps the first-save board assignment after refusing reload for %s, then saves it on retry',
  async (reason) => {
    const f = fixture();
    const projectId = f.store.getSnapshot().activeProject.id;
    const boardId = 'first-save-board';
    f.result.projectBoardAssignments = [{ projectId, boardId }];
    let current = true;
    f.persistence.saveWorkbench.mockImplementationOnce(() => {
      if (reason === 'edit') {
        f.store.commands.projects.rename(projectId, 'Edited during first save');
      }
      if (reason === 'view') {
        current = false;
      }
      if (reason === 'partial-save') {
        f.result.error = 'Another project could not be saved';
      }
      return Promise.resolve(f.result);
    });
    if (reason === 'pixel') {
      f.engines.flushPendingPixels
        .mockResolvedValueOnce(undefined)
        .mockRejectedValueOnce(new Error('Final pixels failed'));
    }
    await expect(reloadWorkbench({ ...f, isCurrent: () => current })).rejects.toThrow();
    expect(f.reload).not.toHaveBeenCalled();
    expect(getProjectWidgetValues(f.store.getSnapshot().activeProject, 'gallery').projectBoardId).toBe(boardId);
    if (reason === 'edit') {
      expect(f.store.getSnapshot().activeProject.name).toBe('Edited during first save');
    }

    // Persistence drains assignments when it returns them; subsequent saves cannot repair a discarded result.
    f.result.projectBoardAssignments = [];
    f.result.error = null;
    current = true;
    await reloadWorkbench({ ...f, isCurrent: () => current });
    const saved = f.persistence.saveWorkbench.mock.lastCall![0].projects.find((project) => project.id === projectId)!;
    expect(getProjectWidgetValues(saved, 'gallery').projectBoardId).toBe(boardId);
    expect(f.reload).toHaveBeenCalledOnce();
  }
);

it('saves an expected first board assignment on the next attempt without bypassing the revision barrier', async () => {
  const f = fixture();
  const projectId = f.store.getSnapshot().activeProject.id;
  f.result.projectBoardAssignments = [{ projectId, boardId: 'new-inbox' }];
  await expect(reloadWorkbench(f)).rejects.toMatchObject({ reason: 'changed' });
  expect(f.reload).not.toHaveBeenCalled();
  expect(getProjectWidgetValues(f.store.getSnapshot().activeProject, 'gallery').projectBoardId).toBe('new-inbox');
  f.result.projectBoardAssignments = [];
  await reloadWorkbench(f);
  expect(f.reload).toHaveBeenCalledOnce();
});

it('does not apply a first-save board assignment after the captured account has ended', async () => {
  const f = fixture();
  const projectId = f.store.getSnapshot().activeProject.id;
  const before = f.store.getState();
  f.result.projectBoardAssignments = [{ projectId, boardId: 'old-account-board' }];
  f.persistence.saveWorkbench.mockImplementationOnce(() => {
    accountLifecycle.activate('replacement-account', ':user:replacement-account');
    return Promise.resolve(f.result);
  });
  await expect(reloadWorkbench(f)).rejects.toMatchObject({ reason: 'inactive' });
  expect(f.store.getState()).toBe(before);
  expect(f.reload).not.toHaveBeenCalled();
});

it('stops applying assignments if publishing one ends the captured account', async () => {
  const f = fixture();
  f.store.commands.projects.create();
  const [first, second] = f.store.getSnapshot().projects;
  f.result.projectBoardAssignments = [
    { projectId: first!.id, boardId: 'first-board' },
    { projectId: second!.id, boardId: 'second-board' },
  ];
  const unsubscribe = f.store.subscribe(() => {
    const project = f.store.getSnapshot().projects.find((p) => p.id === first!.id)!;
    if (getProjectWidgetValues(project, 'gallery').projectBoardId === 'first-board') {
      accountLifecycle.activate('replacement-account', ':user:replacement-account');
    }
  });
  try {
    await expect(reloadWorkbench(f)).rejects.toMatchObject({ reason: 'inactive' });
    const projects = f.store.getSnapshot().projects;
    expect(getProjectWidgetValues(projects[0]!, 'gallery').projectBoardId).toBe('first-board');
    expect(getProjectWidgetValues(projects[1]!, 'gallery').projectBoardId).toBeUndefined();
    expect(f.reload).not.toHaveBeenCalled();
  } finally {
    unsubscribe();
  }
});

it('localizes refusal reasons while retaining pixel failure diagnostics', async () => {
  const translate = (key: string) => `localized:${key}`;
  for (const reason of ['inactive', 'unsaved', 'changed', 'unavailable'] as const) {
    expect(describeWorkbenchReloadError(new WorkbenchReloadError(reason), translate)).toBe(
      `localized:widgets.failure.reloadReasons.${reason}`
    );
  }
  const f = fixture();
  const failure = new Error('Pixel encoder diagnostic');
  f.engines.flushPendingPixels.mockRejectedValueOnce(failure);
  let caught: unknown;
  try {
    await reloadWorkbench(f);
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(AggregateError);
  expect((caught as AggregateError).errors).toEqual([failure]);
  expect(describeWorkbenchReloadError(caught, translate)).toBe('localized:widgets.failure.reloadReasons.pixels');
  expect(describeWorkbenchReloadError(new Error('Untranslated backend detail'), translate)).toBe(
    'localized:widgets.failure.reloadReasons.failed'
  );
  expect(f.reload).not.toHaveBeenCalled();
});

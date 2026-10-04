import type { BrowserLockResult } from '@platform/browser/webLocks';
import type * as accountLifecycleModule from '@platform/state/accountLifecycle';

import { beforeEach, describe, expect, it, vi } from 'vitest';

import type * as libraryModule from './library';
import type { ProjectPushOutcome } from './projectFlush';
import type * as syncStoreModule from './syncStore';

const api = vi.hoisted(() => ({
  createProject: vi.fn(),
  deleteProject: vi.fn(),
  getClientStateValue: vi.fn(() => Promise.resolve<string | null>(null)),
  getProject: vi.fn(),
  getProjectBoardSnapshot: vi.fn(),
  listProjects: vi.fn(),
  setClientStateValue: vi.fn(() => Promise.resolve()),
  updateProject: vi.fn(),
}));

/** The copying itself is `duplicateProject`'s own test; this file owns the library's part of it. */
const duplication = vi.hoisted(() => ({ duplicateProjectRecord: vi.fn() }));
const projectLocks = vi.hoisted(() => ({
  acquireProjectMutationLock: vi.fn<() => Promise<BrowserLockResult>>(() =>
    Promise.resolve({ kind: 'acquired', release: () => Promise.resolve() })
  ),
}));
const queueJournal = vi.hoisted(() => ({
  createAccountOwnedQueueRunJournal: vi.fn(() =>
    Promise.resolve({
      close: vi.fn(),
      deleteForProject: vi.fn(() => Promise.resolve({ kind: 'removed' as const })),
      listForProject: vi.fn(() => Promise.resolve({ entries: [], kind: 'available' as const, removedCorrupt: 0 })),
    })
  ),
}));

vi.mock('./api', () => api);
vi.mock('./invk/duplicateProject', () => duplication);
vi.mock('./projectLifecycleLocks', () => projectLocks);
vi.mock('@workbench/queue-integration/queueRunJournal', () => queueJournal);

let library: typeof libraryModule;
let syncStore: typeof syncStoreModule;
let account: typeof accountLifecycleModule;

const summaryDto = (id: string, name: string, updatedAt: string) => ({
  board_id: `board-for-${id}`,
  created_at: '2026-06-01 08:00:00.000',
  minimum_canvas_schema_version: 3,
  name,
  project_id: id,
  revision: 1,
  updated_at: updatedAt,
});

beforeEach(async () => {
  vi.resetModules();
  api.createProject.mockReset();
  api.deleteProject.mockReset();
  api.getProject.mockReset();
  api.listProjects.mockReset();
  api.updateProject.mockReset();
  api.getClientStateValue.mockReset();
  api.getClientStateValue.mockResolvedValue(null);
  api.setClientStateValue.mockReset();
  api.setClientStateValue.mockResolvedValue(undefined);
  api.getProjectBoardSnapshot.mockReset();
  api.getProjectBoardSnapshot.mockResolvedValue({ items: [] });
  duplication.duplicateProjectRecord.mockReset();
  projectLocks.acquireProjectMutationLock.mockReset();
  projectLocks.acquireProjectMutationLock.mockResolvedValue({ kind: 'acquired', release: () => Promise.resolve() });
  queueJournal.createAccountOwnedQueueRunJournal.mockReset();
  queueJournal.createAccountOwnedQueueRunJournal.mockResolvedValue({
    close: vi.fn(),
    deleteForProject: vi.fn(() => Promise.resolve({ kind: 'removed' })),
    listForProject: vi.fn(() => Promise.resolve({ entries: [], kind: 'available', removedCorrupt: 0 })),
  });

  library = await import('./library');
  syncStore = await import('./syncStore');
  account = await import('@platform/state/accountLifecycle');
});

/** A stand-in for the editor holding a project, with every call recorded in order. */
const openProject = (projectId: string, calls: string[] = []) => {
  const handle = {
    close: vi.fn(() => {
      calls.push('close');
    }),
    deleteOnServer: vi.fn(() => {
      calls.push('deleteOnServer');

      return Promise.resolve();
    }),
    flush: vi.fn(() => {
      calls.push('flush');

      return Promise.resolve<ProjectPushOutcome>({ documentJson: '{}', kind: 'acknowledged' });
    }),
    current: vi.fn(() => undefined),
    flushPixels: vi.fn(() => {
      calls.push('flushPixels');

      return Promise.resolve();
    }),
    markDeleted: vi.fn(() => {
      calls.push('markDeleted');
    }),
    rename: vi.fn((name: string) => {
      calls.push(`rename:${name}`);

      return Promise.resolve();
    }),
    unmarkDeleted: vi.fn(() => {
      calls.push('unmarkDeleted');
    }),
  };

  syncStore.registerOpenProject(projectId, handle);

  return { calls, handle };
};

describe('refreshProjectLibrary', () => {
  it('normalizes SQLite timestamps to ISO and sorts newest first', async () => {
    api.listProjects.mockResolvedValue([
      summaryDto('older', 'Older', '2026-06-09 10:00:00.000'),
      summaryDto('newer', 'Newer', '2026-06-10 10:00:00.000'),
    ]);

    await library.refreshProjectLibrary();

    const { status, summaries } = library.getProjectLibrary();

    expect(status).toBe('ready');
    expect(summaries.map((summary) => summary.id)).toEqual(['newer', 'older']);
    expect(summaries[0].updatedAt).toBe('2026-06-10T10:00:00.000Z');
    expect(summaries[0].minimumCanvasSchemaVersion).toBe(3);
  });

  it('identifies summaries that require a newer canvas reader without fetching the document', async () => {
    api.listProjects.mockResolvedValue([
      { ...summaryDto('future', 'Future', '2026-06-10 10:00:00.000'), minimum_canvas_schema_version: 5 },
    ]);

    await library.refreshProjectLibrary();

    expect(library.isProjectSummaryCompatible(library.getProjectLibrary().summaries[0]!)).toBe(false);
    expect(api.getProject).not.toHaveBeenCalled();
  });

  it('keeps the previous summaries and reports the failure on error', async () => {
    api.listProjects.mockResolvedValue([summaryDto('kept', 'Kept', '2026-06-10 10:00:00.000')]);
    await library.refreshProjectLibrary();

    api.listProjects.mockRejectedValue(new Error('offline'));
    await library.refreshProjectLibrary();

    const { error, status, summaries } = library.getProjectLibrary();

    expect(status).toBe('error');
    expect(error).toBe('offline');
    expect(summaries.map((summary) => summary.id)).toEqual(['kept']);
  });

  it('cannot commit a delayed response into a later account epoch', async () => {
    account.accountLifecycle.activate('user-a');
    let resolveUserA: ((value: ReturnType<typeof summaryDto>[]) => void) | undefined;
    api.listProjects.mockReturnValueOnce(
      new Promise((resolve) => {
        resolveUserA = resolve;
      })
    );
    const userARefresh = library.refreshProjectLibrary();

    account.accountLifecycle.invalidate();
    account.accountLifecycle.activate('user-b');
    api.listProjects.mockResolvedValueOnce([summaryDto('b', 'User B', '2026-06-11 10:00:00.000')]);
    await library.refreshProjectLibrary();

    resolveUserA?.([summaryDto('a', 'User A', '2026-06-12 10:00:00.000')]);
    await userARefresh;

    expect(library.getProjectLibrary().summaries.map((summary) => summary.name)).toEqual(['User B']);
  });
});

describe('upsertProjectSummary', () => {
  it('inserts new entries and moves updated ones to the front', async () => {
    api.listProjects.mockResolvedValue([
      summaryDto('a', 'A', '2026-06-09 10:00:00.000'),
      summaryDto('b', 'B', '2026-06-10 10:00:00.000'),
    ]);
    await library.refreshProjectLibrary();

    library.upsertProjectSummary({ id: 'a', name: 'A renamed', revision: 2 }, account.accountLifecycle.capture());

    const { summaries } = library.getProjectLibrary();

    expect(summaries[0].id).toBe('a');
    expect(summaries[0].name).toBe('A renamed');
    expect(summaries[0].revision).toBe(2);
  });
});

describe('library mutations', () => {
  it('deleteLibraryProject removes from server and store', async () => {
    api.listProjects.mockResolvedValue([summaryDto('doomed', 'Doomed', '2026-06-10 10:00:00.000')]);
    await library.refreshProjectLibrary();
    api.deleteProject.mockResolvedValue(undefined);

    await library.deleteLibraryProject('doomed');

    expect(api.deleteProject).toHaveBeenCalledWith('doomed', expect.any(AbortSignal));
    expect(library.getProjectLibrary().summaries).toHaveLength(0);
  });

  it('renameLibraryProject updates name in both the record and its document', async () => {
    api.getProject.mockResolvedValue({
      ...summaryDto('p1', 'Old name', '2026-06-10 10:00:00.000'),
      data: { id: 'p1', layout: {}, name: 'Old name' },
    });
    api.updateProject.mockResolvedValue({
      ...summaryDto('p1', 'New name', '2026-06-10 11:00:00.000'),
      data: { id: 'p1', layout: {}, name: 'New name' },
      revision: 2,
    });

    await library.renameLibraryProject('p1', 'New name');

    expect(api.updateProject).toHaveBeenCalledWith(
      'p1',
      {
        data: { id: 'p1', layout: {}, name: 'New name' },
        expected_revision: 1,
        name: 'New name',
      },
      expect.any(AbortSignal)
    );
    expect(library.getProjectLibrary().summaries[0]?.name).toBe('New name');
  });

  /** Route open-project mutations through sync ownership; use HTTP for closed projects. */
  it('renames an open project through the editor rather than over HTTP', async () => {
    api.listProjects.mockResolvedValue([summaryDto('open', 'Old name', '2026-06-10 10:00:00.000')]);
    await library.refreshProjectLibrary();

    const { handle } = openProject('open');

    await library.renameLibraryProject('open', 'New name');

    expect(handle.rename).toHaveBeenCalledWith('New name');
    expect(api.getProject).not.toHaveBeenCalled();
    expect(api.updateProject).not.toHaveBeenCalled();
    expect(library.getProjectLibrary().summaries[0]?.name).toBe('New name');
  });

  /** Serialize deletion after in-flight saves; marking cannot stop an already-sent PUT. */
  it('deletes an open project through the sync engine, then closes the tab', async () => {
    const { calls } = openProject('open');

    await library.deleteLibraryProject('open');

    expect(calls).toEqual(['deleteOnServer', 'close']);
    // Never beside the engine: a bare HTTP delete is what could overtake an in-flight push.
    expect(api.deleteProject).not.toHaveBeenCalled();
  });

  it('deletes a closed project over HTTP, because no engine holds it', async () => {
    api.deleteProject.mockResolvedValue(undefined);

    await library.deleteLibraryProject('closed');

    expect(api.deleteProject).toHaveBeenCalledWith('closed', expect.any(AbortSignal));
  });

  it('does not delete while another tab owns an active project run', async () => {
    projectLocks.acquireProjectMutationLock.mockResolvedValueOnce({ kind: 'contended' });

    await expect(library.deleteLibraryProject('closed')).rejects.toThrow('active queue runs');

    expect(api.deleteProject).not.toHaveBeenCalled();
    expect(queueJournal.createAccountOwnedQueueRunJournal).not.toHaveBeenCalled();
  });

  it('fails closed when durable queue runs cannot be verified', async () => {
    const release = vi.fn().mockResolvedValue(undefined);
    projectLocks.acquireProjectMutationLock.mockResolvedValueOnce({ kind: 'acquired', release });
    queueJournal.createAccountOwnedQueueRunJournal.mockResolvedValueOnce({
      close: vi.fn(),
      deleteForProject: vi.fn(),
      listForProject: vi.fn().mockResolvedValue({ kind: 'unavailable' }),
    });

    await expect(library.deleteLibraryProject('closed')).rejects.toThrow('could not be verified');

    expect(api.deleteProject).not.toHaveBeenCalled();
    expect(release).toHaveBeenCalledOnce();
  });

  /** Unmark failed deletes only with mark ownership so autosave resumes safely. */
  it('surfaces a failed deletion of an open project to its caller', async () => {
    const { calls, handle } = openProject('open');

    handle.deleteOnServer.mockImplementation(() => {
      calls.push('deleteOnServer');

      return Promise.reject(new Error('offline'));
    });

    await expect(library.deleteLibraryProject('open')).rejects.toThrow('offline');

    expect(calls).toEqual(['deleteOnServer']);
  });

  it('takes a deleted project out of the saved session', async () => {
    api.deleteProject.mockResolvedValue(undefined);
    api.getClientStateValue.mockResolvedValue(
      JSON.stringify({
        account: { userId: 'user-a' },
        activeProjectId: 'doomed',
        openProjectIds: ['doomed', 'kept'],
      })
    );

    await library.deleteLibraryProject('doomed');

    const [, blob] = api.setClientStateValue.mock.calls.at(-1) as unknown as [string, string];

    // The next boot would otherwise try to hydrate a project the server no longer has.
    expect(JSON.parse(blob)).toMatchObject({ activeProjectId: 'kept', openProjectIds: ['kept'] });
  });

  it('leaves the session alone when the deleted project was not open', async () => {
    api.deleteProject.mockResolvedValue(undefined);
    api.getClientStateValue.mockResolvedValue(
      JSON.stringify({ account: {}, activeProjectId: 'kept', openProjectIds: ['kept'] })
    );

    await library.deleteLibraryProject('closed');

    expect(api.setClientStateValue).not.toHaveBeenCalled();
  });

  /** A copy of what the server last acknowledged would silently drop what is on screen. */
  it('flushes an open project before reading it for duplication', async () => {
    const { calls } = openProject('source');

    api.getProject.mockImplementation(() => {
      calls.push('get');

      return Promise.resolve({
        ...summaryDto('source', 'Source', '2026-06-10 10:00:00.000'),
        data: { id: 'source', layout: {}, name: 'Source' },
      });
    });

    await library.readAcknowledgedProject('source', account.accountLifecycle.capture());

    expect(calls).toEqual(['flushPixels', 'flush', 'get']);
  });

  /** An unacknowledged flush must abort duplication rather than copy stale server bytes. */
  it('refuses to duplicate a project whose flush never reached the server', async () => {
    const { calls, handle } = openProject('source');

    handle.flush.mockImplementation(() => {
      calls.push('flush');

      return Promise.resolve<ProjectPushOutcome>({ documentJson: '{}', kind: 'unsynced' });
    });

    await expect(library.readAcknowledgedProject('source', account.accountLifecycle.capture())).rejects.toMatchObject({
      name: 'ProjectFlushError',
      reason: 'unsynced',
    });

    expect(calls).toEqual(['flushPixels', 'flush']);
    expect(api.getProject).not.toHaveBeenCalled();
  });

  it('refuses to duplicate a project whose id was taken over by another device', async () => {
    const { handle } = openProject('source');

    handle.flush.mockResolvedValue({ documentJson: '{}', kind: 'superseded' });

    await expect(library.readAcknowledgedProject('source', account.accountLifecycle.capture())).rejects.toMatchObject({
      reason: 'superseded',
    });
  });

  it('duplicates through the shared restore engine and adopts the copy', async () => {
    api.getProject.mockResolvedValue({
      ...summaryDto('source', 'Source', '2026-06-10 10:00:00.000'),
      data: { id: 'source', layout: {}, name: 'Source' },
    });
    api.getProjectBoardSnapshot.mockResolvedValue({
      items: [{ category: 'general', kind: 'image', name: 'on-board.png', starred: false }],
    });
    duplication.duplicateProjectRecord.mockResolvedValue({
      boardItemIssues: [{ kind: 'image', name: 'lost.png', reason: 'upload-failed' }],
      coverImageName: null,
      documentReferenceIssues: [],
      record: { ...summaryDto('copy-id', 'Source copy', '2026-06-10 11:00:00.000'), data: {} },
    });

    const duplicated = await library.duplicateLibraryProject('source');

    // Enumerate the board successfully before creating copy resources.
    expect(api.getProjectBoardSnapshot).toHaveBeenCalledWith('source', expect.any(AbortSignal));
    expect(duplication.duplicateProjectRecord.mock.calls[0]?.[0]).toMatchObject({
      boardItems: [{ name: 'on-board.png' }],
      record: { project_id: 'source' },
    });
    expect(duplicated.summary.id).toBe('copy-id');
    expect(duplicated.boardItemIssues).toHaveLength(1);
    expect(library.getProjectLibrary().summaries[0]?.name).toBe('Source copy');
  });

  it('does not create anything when the board cannot be enumerated', async () => {
    api.getProject.mockResolvedValue({
      ...summaryDto('source', 'Source', '2026-06-10 10:00:00.000'),
      data: { id: 'source', layout: {}, name: 'Source' },
    });
    api.getProjectBoardSnapshot.mockRejectedValue(new Error('snapshot unavailable'));

    await expect(library.duplicateLibraryProject('source')).rejects.toThrow('snapshot unavailable');
    expect(duplication.duplicateProjectRecord).not.toHaveBeenCalled();
  });
});

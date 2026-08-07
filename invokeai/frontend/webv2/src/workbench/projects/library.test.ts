import type * as accountLifecycleModule from '@platform/state/accountLifecycle';

import { beforeEach, describe, expect, it, vi } from 'vitest';

import type * as libraryModule from './library';
import type * as syncStoreModule from './syncStore';

/**
 * The project library store: summaries normalized at the boundary, sorted by
 * recency, and the explicit mutations (delete, rename, duplicate) that are
 * the only paths off the server.
 */

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

vi.mock('./api', () => api);
vi.mock('./invk/duplicateProject', () => duplication);

let library: typeof libraryModule;
let syncStore: typeof syncStoreModule;
let account: typeof accountLifecycleModule;

const summaryDto = (id: string, name: string, updatedAt: string) => ({
  created_at: '2026-06-01 08:00:00.000',
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
    flush: vi.fn(() => {
      calls.push('flush');

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

  /**
   * The invariant: a project the workbench holds is mutated through the sync engine, everything
   * else over HTTP. A library PUT beside an open project's revision chain used to fork it into a
   * conflict copy, and would now rename its board from outside the transaction that owns both.
   */
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

  it('deletes an open project without letting its autosave recreate it, then closes the tab', async () => {
    api.deleteProject.mockResolvedValue(undefined);

    const { calls } = openProject('open');

    await library.deleteLibraryProject('open');

    expect(calls).toEqual(['markDeleted', 'close']);
    expect(api.deleteProject).toHaveBeenCalledWith('open', expect.any(AbortSignal));
  });

  /**
   * A project left marked deleted never autosaves again for the rest of the session, and nothing
   * says so. Unmarking belongs here, beside the mark, rather than in each caller's catch — one of
   * the two that had to remember it did not.
   */
  it('lets an open project save again when its deletion fails', async () => {
    api.deleteProject.mockRejectedValue(new Error('offline'));

    const { calls } = openProject('open');

    await expect(library.deleteLibraryProject('open')).rejects.toThrow('offline');

    expect(calls).toEqual(['markDeleted', 'unmarkDeleted']);
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

    await library.readProjectForDuplication('source', account.accountLifecycle.capture());

    expect(calls).toEqual(['flush', 'get']);
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

    // The board is enumerated before anything is created: a copy whose board came back silently
    // empty would look like a faithful duplication of a project that had produced nothing.
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

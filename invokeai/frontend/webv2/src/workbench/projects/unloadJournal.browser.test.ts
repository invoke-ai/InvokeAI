import type { Project, WorkbenchState } from '@workbench/projectContracts';
import type * as projectsApi from '@workbench/projects/api';

import { acquireExclusiveLock } from '@platform/browser/webLocks';
import { accountLifecycle, captureAccountScope } from '@platform/state/accountLifecycle';
import { createDraftProject } from '@workbench/workbenchState';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { ProjectDraftStore } from './draftStore';

import { createDurableSyncedWorkbenchPersistence, type DurableProjectPersistenceApi } from './durableSyncedPersistence';
import { EDITOR_SESSION_STORAGE_KEY } from './editorSession';
import { createAccountOwnedProjectDraftStore } from './indexedDbDraftStore';
import { serializeProjectDocumentV3 } from './projectDocument';
import { peekOpenProjectIds } from './session';
import { createUnloadJournalServer, stateWith } from './unloadJournal.scenarios';
import { getUnloadJournalDatabaseName, getWorkbenchDatabaseName } from './workbenchDatabase';

/** The route guard's session read; persistence itself uses the injected server. */
const guardSession = vi.hoisted(() => ({ json: null as string | null }));
vi.mock('./api', async (importOriginal) => ({
  ...(await importOriginal<typeof projectsApi>()),
  getClientStateValue: () => Promise.resolve(guardSession.json),
}));

/**
 * Persistence with real IndexedDB, Web Locks and editor sessions per simulated tab; only the server is in memory. A
 * "reload" is a new persistence for the same editor session that never ran the old page's pending saves.
 */
const now = '2026-10-05T00:00:00.000Z';

let api: DurableProjectPersistenceApi;
let server: ReturnType<typeof createUnloadJournalServer>['server'];
let project: Project;
const releases: (() => Promise<void>)[] = [];

/** One tab's editor lifetime: `editorSessionId` survives its reloads, `writer` does not. */
const openTab = (editorSessionId: string, writer: string, draftStore?: Promise<ProjectDraftStore>) => {
  const service = createDurableSyncedWorkbenchPersistence(captureAccountScope(), {
    api,
    ...(draftStore ? { draftStore } : {}),
    editorSession: Promise.resolve({ id: editorSessionId, release: () => Promise.resolve() }),
    now: () => now,
    writerToken: writer,
  });
  releases.push(service.retain());
  return service;
};

/** Every draft lineage's document for the project, read straight from browser storage. */
const storedDocuments = async (projectId: string): Promise<Record<string, string>> => {
  const store = await createAccountOwnedProjectDraftStore(captureAccountScope());
  try {
    const listed = await store.listForProject(projectId);
    const documents: Record<string, string> = {};
    for (const item of listed.kind === 'available' ? listed.items : []) {
      const found = await store.get(projectId, item.editorSessionId);
      if (found.kind === 'found') {
        documents[item.editorSessionId] = (JSON.parse(found.draft.documentJson) as { name: string }).name;
      }
    }
    return documents;
  } finally {
    store.close();
  }
};

const renamed = (state: WorkbenchState, name: string): WorkbenchState =>
  stateWith(state.projects.map((candidate) => (candidate.id === project.id ? { ...candidate, name } : candidate)));

beforeEach(() => {
  accountLifecycle.activate('unload-journal-test', `:unload-journal-test:${crypto.randomUUID()}`);
  ({ api, server } = createUnloadJournalServer());
  guardSession.json = null;
  project = createDraftProject([]);
  server.records.set(project.id, {
    board_id: `board-${project.id}`,
    created_at: now,
    data: serializeProjectDocumentV3(project),
    minimum_canvas_schema_version: 3,
    name: project.name,
    project_id: project.id,
    revision: 1,
    updated_at: now,
  });
  void api.saveSession(stateWith([project]), 'tab-a', {});
});

afterEach(async () => {
  await Promise.all(releases.splice(0).map((release) => release()));
  window.sessionStorage.removeItem(EDITOR_SESSION_STORAGE_KEY);
  // Clearing the account lifetime deletes its database, as signing out does.
  accountLifecycle.invalidate();
});

describe('unload journal across tabs', () => {
  it("surfaces a reloaded tab's journal as a conflict instead of overwriting another tab's newer work", async () => {
    const tabA = openTab('tab-a', 'writer-a1');
    const tabB = openTab('tab-b', 'writer-b');
    const loadedA = await tabA.loadWorkbench();
    const loadedB = await tabB.loadWorkbench();

    await tabB.saveWorkbench(renamed(loadedB.state, 'B saved'));
    server.offline = true;
    await tabB.saveWorkbench(renamed(loadedB.state, 'B newer, staged only'));
    // Tab A, still on revision 1, unloads right after typing: only its journal is written.
    expect(tabA.journalBeforeUnload(renamed(loadedA.state, 'A typed'))).toMatchObject({ kind: 'journaled' });
    server.offline = false;

    const reloadedA = openTab('tab-a', 'writer-a2');
    const recovered = await reloadedA.loadWorkbench();

    expect(recovered.state.projects).toMatchObject([{ id: project.id, name: 'A typed' }]);
    expect(recovered.conflicts).toEqual([
      expect.objectContaining({ kind: 'revision', projectId: project.id, serverRevision: 2 }),
    ]);
    expect(server.records.get(project.id)).toMatchObject({ name: 'B saved', revision: 2 });
    await expect(storedDocuments(project.id)).resolves.toEqual({
      'tab-a': 'A typed',
      'tab-b': 'B newer, staged only',
    });
    // The recovered tab keeps the conflict rather than pushing over the server.
    await reloadedA.saveWorkbench(recovered.state);
    expect(server.records.get(project.id)).toMatchObject({ name: 'B saved', revision: 2 });
  });

  it('discards the journal of a tab whose draft another tab adopted', async () => {
    const tabA = openTab('tab-a', 'writer-a1');
    const loadedA = await tabA.loadWorkbench();
    server.offline = true;
    await tabA.saveWorkbench(renamed(loadedA.state, 'A staged'));
    server.offline = false;

    // Tab B opens the project while tab A's draft is unacknowledged and adopts that lineage.
    const tabB = openTab('tab-b', 'writer-b');
    const loadedB = await tabB.loadWorkbench();
    expect(loadedB.state.projects).toMatchObject([{ name: 'A staged' }]);
    // Tab A, no longer the lineage's writer, unloads with a newer edit.
    tabA.journalBeforeUnload(renamed(loadedA.state, 'A after losing the lineage'));

    const recovered = await openTab('tab-a', 'writer-a2').loadWorkbench();

    expect(recovered.state.projects).toMatchObject([{ id: project.id, name: 'A staged' }]);
    expect(Object.values(await storedDocuments(project.id))).toEqual(['A staged']);
  });
});

describe('unload journal and a closed session', () => {
  it('neither journals after the last tab closed nor reopens the project from an earlier journal', async () => {
    const tab = openTab('tab-a', 'writer-a1');
    const loaded = await tab.loadWorkbench();
    const closing = renamed(loaded.state, 'Edited, then closed');

    // Hidden while typing (the journal is written), then shown again and the last tab closed: the close flow pushes.
    expect(tab.journalBeforeUnload(closing)).toMatchObject({ kind: 'journaled' });
    await expect(tab.flushProjectToServer(closing.projects[0]!)).resolves.toMatchObject({ kind: 'acknowledged' });
    await tab.persistEmptySession(closing);
    expect(tab.journalBeforeUnload(renamed(closing, 'After closing'))).toEqual({ kind: 'nothing-to-journal' });

    const reopened = await openTab('tab-a', 'writer-a2').loadWorkbench();

    expect(reopened.state.projects.map(({ id }) => id)).not.toContain(project.id);
    expect(server.records.get(project.id)).toMatchObject({ name: 'Edited, then closed' });
  });
});

describe('unload journal and account cleanup', () => {
  it('is deleted with the account database when the account lifetime ends', async () => {
    const { storageSuffix } = captureAccountScope();
    const tab = openTab('tab-a', 'writer-a1');
    const loaded = await tab.loadWorkbench();
    expect(tab.journalBeforeUnload(renamed(loaded.state, 'Before signing out'))).toMatchObject({ kind: 'journaled' });
    const names = async () => (await indexedDB.databases()).map(({ name }) => name);
    await expect(names()).resolves.toContain(getUnloadJournalDatabaseName(storageSuffix));

    accountLifecycle.invalidate();

    await vi.waitFor(
      async () => {
        const remaining = await names();
        expect(remaining).not.toContain(getUnloadJournalDatabaseName(storageSuffix));
        expect(remaining).not.toContain(getWorkbenchDatabaseName(storageSuffix));
      },
      { timeout: 5_000 }
    );
  });
});

describe('unload journal without its database', () => {
  it('loads, saves and recovers staged drafts with no recovery warning', async () => {
    const failingJournal = () =>
      createAccountOwnedProjectDraftStore(captureAccountScope(), {
        openJournalDatabase: () => Promise.reject(new DOMException('The journal could not open.', 'TimeoutError')),
      });
    const tab = openTab('tab-a', 'writer-a1', failingJournal());
    const loaded = await tab.loadWorkbench();
    expect(loaded.localDraftStatus).toBe('ok');
    const edited = renamed(loaded.state, 'Staged, not pushed');
    expect(tab.journalBeforeUnload(edited)).toEqual({ kind: 'unavailable' });
    server.offline = true;
    await expect(tab.saveWorkbench(edited)).resolves.toMatchObject({ localDraftStatus: 'ok' });
    server.offline = false;

    const reloaded = await openTab('tab-a', 'writer-a2', failingJournal()).loadWorkbench();

    expect(reloaded.localDraftStatus).toBe('ok');
    expect(reloaded.state.projects).toMatchObject([{ id: project.id, name: 'Staged, not pushed' }]);
  });
});

describe('unload journal and the editor route', () => {
  it("does not count another running tab's journal as one that would open", async () => {
    // Tab A still runs, holding its editor session, and journaled an edit while hidden.
    const lock = await acquireExclusiveLock('invokeai:v7:webv2:editor-session:tab-a');
    expect(lock.kind).toBe('acquired');
    const tabA = openTab('tab-a', 'writer-a');
    const loaded = await tabA.loadWorkbench();
    expect(tabA.journalBeforeUnload(renamed(loaded.state, 'Typed in A'))).toMatchObject({ kind: 'journaled' });
    guardSession.json = JSON.stringify({ account: loaded.state.account, activeProjectId: '', openProjectIds: [] });
    window.sessionStorage.setItem(EDITOR_SESSION_STORAGE_KEY, 'tab-b');

    await expect(peekOpenProjectIds()).resolves.toEqual([]);
    // The tab that wrote it would reconcile it on its own reload.
    window.sessionStorage.setItem(EDITOR_SESSION_STORAGE_KEY, 'tab-a');
    await expect(peekOpenProjectIds()).resolves.toEqual([project.id]);
    // Tab A is gone: any tab would recover it now.
    window.sessionStorage.setItem(EDITOR_SESSION_STORAGE_KEY, 'tab-b');
    if (lock.kind === 'acquired') {
      await lock.release();
    }
    await expect(peekOpenProjectIds()).resolves.toEqual([project.id]);
  });

  it('counts a project known only from its journal as one that would open', async () => {
    const tab = openTab('tab-a', 'writer-a1');
    const loaded = await tab.loadWorkbench();
    // A new project typed into and reloaded before its first save: no server record, draft, or session entry.
    const fresh = createDraftProject(loaded.state.projects);
    guardSession.json = JSON.stringify({ account: loaded.state.account, activeProjectId: '', openProjectIds: [] });
    expect(tab.journalBeforeUnload(stateWith([...loaded.state.projects, fresh]))).toMatchObject({ kind: 'journaled' });

    await expect(peekOpenProjectIds()).resolves.toEqual([fresh.id]);
  });
});

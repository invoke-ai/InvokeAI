import type { Project, WorkbenchState } from '@workbench/projectContracts';

import { createUuid } from '@platform/browser/randomUuid';
import { accountLifecycle, captureAccountScope } from '@platform/state/accountLifecycle';
import { createDraftProject, createInitialWorkbenchState } from '@workbench/workbenchState';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { ProjectRecordDTO, ProjectSummaryDTO } from './api';
import type { ProjectDraftStore, ProjectUnloadJournalEntry } from './draftStore';
import type { WorkbenchSessionBlob } from './session';

import { createDurableSyncedWorkbenchPersistence, type DurableProjectPersistenceApi } from './durableSyncedPersistence';
import { serializeProjectDocumentV3 } from './projectDocument';
import { serializeSessionBlob } from './session';
import { getProjectSyncSnapshot } from './syncStore';

const now = '2026-10-05T00:00:00.000Z';

/** An in-memory project server; `offline` fails writes as an unreachable server does, and `gate` holds updates. */
export const createUnloadJournalServer = () => {
  const records = new Map<string, ProjectRecordDTO>();
  let session: WorkbenchSessionBlob | null = null;
  const server = {
    /** Applies an update but never answers it, as when the response is lost. */
    dropNextUpdateResponse: false,
    gate: null as Promise<void> | null,
    offline: false,
    records,
    get session() {
      return session;
    },
    updates: 0,
  };
  const write = <T>(apply: () => T): Promise<T> =>
    server.offline ? Promise.reject(new TypeError('Failed to fetch')) : Promise.resolve(apply());
  const notFound = () => Object.assign(new Error('not found'), { status: 404 });
  const api: DurableProjectPersistenceApi = {
    createProject: (request) =>
      write(() => {
        const record: ProjectRecordDTO = {
          board_id: `board-${request.project_id!}`,
          created_at: now,
          data: structuredClone(request.data),
          minimum_canvas_schema_version: request.minimum_canvas_schema_version ?? 3,
          name: request.name,
          project_id: request.project_id!,
          revision: 1,
          updated_at: now,
        };
        records.set(record.project_id, record);
        return structuredClone(record);
      }),
    deleteProject: (projectId) => write(() => void records.delete(projectId)),
    deleteSession: () => write(() => void (session = null)),
    getProject: (projectId) => {
      const record = records.get(projectId);
      return record ? Promise.resolve(structuredClone(record)) : Promise.reject(notFound());
    },
    listProjects: () =>
      Promise.resolve(
        [...records.values()].map(({ data: _data, ...summary }): ProjectSummaryDTO => structuredClone(summary))
      ),
    loadSession: () => Promise.resolve(session && structuredClone(session)),
    saveSession: (state, editorSessionId, draftEditorSessionIds) =>
      write(() => {
        session = JSON.parse(
          serializeSessionBlob(state, editorSessionId, draftEditorSessionIds)
        ) as WorkbenchSessionBlob;
      }),
    updateProject: async (projectId, request) => {
      server.updates += 1;
      await server.gate;
      if (server.offline) {
        throw new TypeError('Failed to fetch');
      }
      const current = records.get(projectId);
      if (!current) {
        throw notFound();
      }
      if (current.revision !== request.expected_revision) {
        throw Object.assign(new Error('conflict'), { status: 409 });
      }
      const record = {
        ...current,
        data: structuredClone(request.data),
        name: request.name,
        revision: current.revision + 1,
      };
      records.set(projectId, record);
      if (server.dropNextUpdateResponse) {
        server.dropNextUpdateResponse = false;
        throw new TypeError('Failed to fetch');
      }
      return structuredClone(record);
    },
  };
  return { api, server };
};

export const stateWith = (projects: Project[]): WorkbenchState => ({
  ...createInitialWorkbenchState(),
  activeProjectId: projects[0]?.id ?? '',
  projects,
});

const emptyQueueRunJournal = (() =>
  Promise.resolve({
    close: () => undefined,
    listForProject: () => Promise.resolve({ entries: [], kind: 'available', removedCorrupt: 0 }),
    listProjectIds: () => Promise.resolve({ kind: 'available', projectIds: [] }),
  })) as unknown as NonNullable<Parameters<typeof createDurableSyncedWorkbenchPersistence>[1]>['queueRunJournal'];

/**
 * Unload journal journeys through the persistence service, against a draft store the caller provides (in memory or
 * real IndexedDB). A "reload" is a new service for the same editor session and store, which never ran the old page's
 * pending work; every write the old page could still make is made explicitly.
 */
export const testUnloadJournalScenarios = (createStore: () => Promise<ProjectDraftStore>): void => {
  let api: DurableProjectPersistenceApi;
  let server: ReturnType<typeof createUnloadJournalServer>['server'];
  let store: ProjectDraftStore;
  let project: Project;
  /** Every journal write, so a test can write one again as if its deletion had failed. */
  let written: ProjectUnloadJournalEntry[];
  let writer = 0;
  /** Holds the writer's settlement of its journal, and the server's creation of a save-as-new copy. */
  let settleGate: Promise<void> | null;
  let settlesStarted: number;
  let copyGate: Promise<void> | null;
  let copiesStarted: number;

  /** The editor sessions of pages that are still running, as the Web Lock each page holds would show. */
  let liveEditorSessionIds: Set<string>;

  /** A page of the tab `editorSessionId`; it stays live until `unloadTab`, and a reload reuses the same id. */
  const openTab = (editorSessionId = 'tab-a') => {
    writer += 1;
    liveEditorSessionIds.add(editorSessionId);
    return createDurableSyncedWorkbenchPersistence(captureAccountScope(), {
      api,
      deleteDatabase: () => Promise.resolve({ kind: 'deleted' }),
      draftStore: Promise.resolve(store),
      editorSession: Promise.resolve({ id: editorSessionId, release: () => Promise.resolve() }),
      isEditorSessionLive: (candidate) => Promise.resolve(liveEditorSessionIds.has(candidate)),
      now: () => now,
      projectMutationLock: () => Promise.resolve({ kind: 'acquired', release: () => Promise.resolve() }),
      queueRunJournal: emptyQueueRunJournal,
      saveDraftAsNew: async (input) => {
        copiesStarted += 1;
        await copyGate;
        return api.createProject(
          {
            data: input.document,
            minimum_canvas_schema_version: input.minimumCanvasSchemaVersion,
            name: input.name,
            project_id: input.copyProjectId,
          },
          input.owner
        );
      },
      writerToken: `writer-${writer}`,
    });
  };
  const unloadTab = (editorSessionId: string) => {
    liveEditorSessionIds.delete(editorSessionId);
  };
  const named = (state: WorkbenchState, name: string): WorkbenchState =>
    stateWith(state.projects.map((candidate) => (candidate.id === project.id ? { ...candidate, name } : candidate)));
  const serverName = () => server.records.get(project.id)?.name;
  /** Another tab's save, as the server sees it. */
  const saveElsewhere = (name: string, revision: number) => {
    const current = server.records.get(project.id)!;
    server.records.set(project.id, { ...current, data: { ...current.data, name }, name, revision });
  };
  /** What the deletion of each journal entry would have removed, written back. */
  const failEveryJournalDeletion = () => {
    store.journalBeforeUnload({ entries: written, retired: [] });
  };

  beforeEach(async () => {
    accountLifecycle.activate('unload-journal-scenarios', `:unload-journal-scenarios:${createUuid()}`);
    ({ api, server } = createUnloadJournalServer());
    const backing = await createStore();
    written = [];
    liveEditorSessionIds = new Set();
    settleGate = null;
    settlesStarted = 0;
    copyGate = null;
    copiesStarted = 0;
    store = {
      ...backing,
      get availability() {
        return backing.availability;
      },
      journalBeforeUnload: (write) => {
        written.push(...write.entries);
        return backing.journalBeforeUnload(write);
      },
      settleUnloadJournal: async (...args) => {
        settlesStarted += 1;
        await settleGate;
        return backing.settleUnloadJournal(...args);
      },
    };
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
    await api.saveSession(stateWith([project]), 'tab-a', {});
  });

  afterEach(() => {
    store.close();
    accountLifecycle.invalidate();
  });

  it('does not bring back an edit that was undone before the tab closed', async () => {
    const tab = openTab();
    const loaded = await tab.loadWorkbench();
    expect(tab.journalBeforeUnload(named(loaded.state, 'Typed, then undone'))).toMatchObject({ kind: 'journaled' });
    // Shown again and undone: the save finds nothing to stage or push.
    await tab.saveWorkbench(loaded.state);
    failEveryJournalDeletion();

    const reloaded = openTab();
    const recovered = await reloaded.loadWorkbench();
    await reloaded.saveWorkbench(recovered.state);

    expect(recovered.state.projects).toMatchObject([{ id: project.id, name: project.name }]);
    expect(recovered.conflicts).toEqual([]);
    expect(server.updates).toBe(0);
    expect(server.records.get(project.id)?.revision).toBe(1);
    await expect(store.listForProject(project.id)).resolves.toMatchObject({ items: [] });
  });

  describe('a journal of a tab that is still running', () => {
    it('is left to that tab: another tab neither adopts it nor brings back the edit once it is undone', async () => {
      const tab = openTab();
      const loaded = await tab.loadWorkbench();
      expect(tab.journalBeforeUnload(named(loaded.state, 'Typed, then undone'))).toMatchObject({ kind: 'journaled' });

      const other = await openTab('tab-b').loadWorkbench();

      expect(other.state.projects).toMatchObject([{ id: project.id, name: project.name }]);
      expect(other.conflicts).toEqual([]);
      // Shown again and undone: the save settles the journal, and nothing holds the undone edit any more.
      await tab.saveWorkbench(loaded.state);
      const recovered = await openTab().loadWorkbench();

      expect(recovered.state.projects).toMatchObject([{ id: project.id, name: project.name }]);
      expect(recovered.conflicts).toEqual([]);
      expect(server.updates).toBe(0);
      await expect(store.listForProject(project.id)).resolves.toMatchObject({ items: [] });
    });

    it('does not come back through another tab when it was written during a save and then undone', async () => {
      const tab = openTab();
      const loaded = await tab.loadWorkbench();
      let release!: () => void;
      server.gate = new Promise((resolve) => {
        release = resolve;
      });
      const saved = named(loaded.state, 'Saved');
      const saving = tab.saveWorkbench(saved);
      await vi.waitFor(() => expect(server.updates).toBe(1));
      tab.journalBeforeUnload(named(loaded.state, 'Typed while it saved, then undone'));
      const other = await openTab('tab-b').loadWorkbench();
      server.gate = null;
      release();
      await saving;
      await tab.saveWorkbench(saved);

      expect(JSON.stringify(other.state)).not.toContain('then undone');
      const recovered = await openTab().loadWorkbench();
      expect(JSON.stringify(recovered.state)).not.toContain('then undone');
      expect(server.records.get(project.id)).toMatchObject({ name: 'Saved', revision: 2 });
    });

    it('is recovered by another tab once the page that wrote it is gone', async () => {
      const tab = openTab();
      const loaded = await tab.loadWorkbench();
      tab.journalBeforeUnload(named(loaded.state, 'Typed before the crash'));
      unloadTab('tab-a');

      const recovered = await openTab('tab-b').loadWorkbench();

      expect(recovered.state.projects).toMatchObject([{ id: project.id, name: 'Typed before the crash' }]);
      expect(recovered.conflicts).toEqual([]);
    });

    it('is still recovered by a reload of that tab, which holds the same editor session', async () => {
      const tab = openTab();
      const loaded = await tab.loadWorkbench();
      tab.journalBeforeUnload(named(loaded.state, 'Typed before the reload'));

      const recovered = await openTab().loadWorkbench();

      expect(recovered.state.projects).toMatchObject([{ id: project.id, name: 'Typed before the reload' }]);
    });

    describe('under a lineage named after another tab, which the session blob handed on', () => {
      // Tab A isolated the project's lineage (an unopenable draft, say); every tab that loads the session inherits it.
      const inheritedLineage = `tab-a:writer:writer-1`;
      beforeEach(async () => {
        await api.saveSession(stateWith([project]), 'tab-a', { [project.id]: inheritedLineage });
      });

      it("is recovered by a reload of the tab that wrote it while the lineage's tab lives on", async () => {
        liveEditorSessionIds.add('tab-a');
        const tab = openTab('tab-b');
        const loaded = await tab.loadWorkbench();
        expect(tab.journalBeforeUnload(named(loaded.state, 'Typed in the other tab'))).toMatchObject({
          kind: 'journaled',
        });

        const recovered = await openTab('tab-b').loadWorkbench();

        expect(recovered.state.projects).toMatchObject([{ id: project.id, name: 'Typed in the other tab' }]);
        expect(recovered.conflicts).toEqual([]);
        await expect(store.get(project.id, inheritedLineage)).resolves.toMatchObject({ kind: 'found' });
      });

      it('is left to the tab that wrote it while that tab lives, whichever tab the lineage is named after', async () => {
        const tab = openTab();
        const loaded = await tab.loadWorkbench();
        tab.journalBeforeUnload(named(loaded.state, "Typed in the lineage's own tab"));

        const other = await openTab('tab-b').loadWorkbench();

        expect(other.state.projects).toMatchObject([{ id: project.id, name: project.name }]);
        await expect(store.peekUnloadJournalProjectIds(1)).resolves.toEqual({
          kind: 'available',
          projectIds: [project.id],
        });
        // Gone now: the next load takes the entry.
        unloadTab('tab-a');
        const recovered = await openTab('tab-b').loadWorkbench();
        expect(recovered.state.projects).toMatchObject([{ id: project.id, name: "Typed in the lineage's own tab" }]);
      });
    });
  });

  describe('a staged draft of a tab that is still running', () => {
    /** Tab A stages an edit the server has not acknowledged (it is unreachable), then keeps running. */
    const stageInTab = async (name: string) => {
      const tab = openTab();
      const loaded = await tab.loadWorkbench();
      server.offline = true;
      await tab.saveWorkbench(named(loaded.state, name));
      server.offline = false;
      return { loaded, tab };
    };

    it('is left to that tab, whose undo then retires it', async () => {
      const { loaded, tab } = await stageInTab('Staged in tab A');

      const other = openTab('tab-b');
      const loadedOther = await other.loadWorkbench();

      expect(loadedOther.state.projects).toMatchObject([{ id: project.id, name: project.name }]);
      expect(loadedOther.conflicts).toEqual([]);
      await expect(other.hydrateProjectFromServer(project.id)).resolves.toMatchObject({
        project: { name: project.name },
      });
      expect(getProjectSyncSnapshot().recoverableDrafts).toEqual([]);
      await expect(store.get(project.id, 'tab-a')).resolves.toMatchObject({
        draft: { documentJson: expect.stringContaining('Staged in tab A') as unknown },
        kind: 'found',
      });
      // Undone in tab A: its own save retires the edit, which no other tab holds.
      await tab.saveWorkbench(loaded.state);
      await expect(store.listForProject(project.id)).resolves.toMatchObject({ items: [] });
    });

    it('is still taken by a reload of that tab, which holds the same editor session', async () => {
      const tab = openTab();
      const loaded = await tab.loadWorkbench();
      // Never reached the server or the session: only its draft makes the reload open it.
      const fresh = { ...createDraftProject(loaded.state.projects), name: 'Created offline' };
      server.offline = true;
      await tab.saveWorkbench(stateWith([...loaded.state.projects, fresh]));
      server.offline = false;

      const reloaded = await openTab().loadWorkbench();

      expect(reloaded.state.projects).toContainEqual(
        expect.objectContaining({ id: fresh.id, name: 'Created offline' })
      );
    });

    it('is recovered by another tab once that tab is gone', async () => {
      await stageInTab('Staged before the crash');
      unloadTab('tab-a');

      const recovered = await openTab('tab-b').loadWorkbench();

      expect(recovered.state.projects).toMatchObject([{ id: project.id, name: 'Staged before the crash' }]);
      expect(recovered.conflicts).toEqual([]);
    });
  });

  it('does not bring back an edit undone just before the page unloaded, with no save in between', async () => {
    const tab = openTab();
    const loaded = await tab.loadWorkbench();
    tab.journalBeforeUnload(named(loaded.state, 'Typed, then undone'));
    // Undone and hidden again at once: the journal of the undone state is retired in the same blind write.
    expect(tab.journalBeforeUnload(loaded.state)).toMatchObject({ kind: 'journaled', projectIds: [] });

    const recovered = await openTab().loadWorkbench();

    expect(recovered.state.projects).toMatchObject([{ id: project.id, name: project.name }]);
    expect(recovered.conflicts).toEqual([]);
  });

  it('settles a journal retired at a hidden event on the next save, should that retirement be lost', async () => {
    const tab = openTab();
    const loaded = await tab.loadWorkbench();
    tab.journalBeforeUnload(named(loaded.state, 'Typed, then undone'));
    tab.journalBeforeUnload(loaded.state);
    // The blind retirement never reached storage, but the page lived on and saved.
    failEveryJournalDeletion();
    await tab.saveWorkbench(loaded.state);
    failEveryJournalDeletion();

    const recovered = await openTab().loadWorkbench();

    expect(recovered.state.projects).toMatchObject([{ id: project.id, name: project.name }]);
    expect(server.updates).toBe(0);
  });

  it('journals an edit redone after its journal was retired', async () => {
    const tab = openTab();
    const loaded = await tab.loadWorkbench();
    const redone = named(loaded.state, 'Typed, undone, redone');
    tab.journalBeforeUnload(redone);
    tab.journalBeforeUnload(loaded.state);

    expect(tab.journalBeforeUnload(redone)).toMatchObject({ kind: 'journaled', projectIds: [project.id] });
    const recovered = await openTab().loadWorkbench();

    expect(recovered.state.projects).toMatchObject([{ id: project.id, name: 'Typed, undone, redone' }]);
  });

  it('journals an edit redone while its earlier journal was being settled', async () => {
    const tab = openTab();
    const loaded = await tab.loadWorkbench();
    const redone = named(loaded.state, 'Typed, undone, redone');
    tab.journalBeforeUnload(redone);
    let release!: () => void;
    settleGate = new Promise((resolve) => {
      release = resolve;
    });
    const undoing = tab.saveWorkbench(loaded.state);
    await vi.waitFor(() => expect(settlesStarted).toBe(1));

    expect(tab.journalBeforeUnload(redone)).toMatchObject({ kind: 'journaled', projectIds: [project.id] });
    settleGate = null;
    release();
    await undoing;
    const recovered = await openTab().loadWorkbench();

    expect(recovered.state.projects).toMatchObject([{ id: project.id, name: 'Typed, undone, redone' }]);
  });

  describe('a hidden event during save-as-new', () => {
    const startCopy = async () => {
      const tab = openTab();
      const loaded = await tab.loadWorkbench();
      saveElsewhere('Remote', 2);
      await expect(tab.saveWorkbench(named(loaded.state, 'Local'))).resolves.toMatchObject({
        conflicts: [{ kind: 'revision' }],
      });
      let release!: () => void;
      copyGate = new Promise((resolve) => {
        release = resolve;
      });
      const copying = tab.resolveConflictSaveAsNew(named(loaded.state, 'Local').projects[0]!);
      await vi.waitFor(() => expect(copiesStarted).toBe(1));
      expect(tab.journalBeforeUnload(named(loaded.state, 'Typed during the copy'))).toMatchObject({
        kind: 'journaled',
        projectIds: [project.id],
      });
      return { copying, release };
    };

    it('recovers the edit into the conflicted source when the page unloads before the copy exists', async () => {
      await startCopy();

      const recovered = await openTab().loadWorkbench();

      expect(recovered.state.projects).toMatchObject([{ id: project.id, name: 'Typed during the copy' }]);
      expect(recovered.conflicts).toEqual([expect.objectContaining({ kind: 'revision', projectId: project.id })]);
      expect([...server.records.keys()]).toEqual([project.id]);
    });

    it('creates the copy from its reserved document and never writes the edit into the source', async () => {
      const { copying, release } = await startCopy();
      copyGate = null;
      release();
      const copy = await copying;

      const copyRecord = server.records.get(copy.targetProjectId);
      expect(copyRecord?.data.name).toBe('Local (copy)');
      expect(JSON.stringify(copyRecord?.data)).not.toContain('Typed during the copy');
      // The source lineage moved to the copy, so the entry written for the source is discarded rather than revived.
      await openTab().loadWorkbench();
      expect(serverName()).toBe('Remote');
      await expect(store.listForProject(project.id)).resolves.toMatchObject({ items: [] });
    });
  });

  it('recovers an edit journaled after an earlier journal of the same project was settled', async () => {
    const tab = openTab();
    const loaded = await tab.loadWorkbench();
    tab.journalBeforeUnload(named(loaded.state, 'Typed, then undone'));
    await tab.saveWorkbench(loaded.state);
    expect(tab.journalBeforeUnload(named(loaded.state, 'Typed again'))).toMatchObject({ kind: 'journaled' });

    const recovered = await openTab().loadWorkbench();

    expect(recovered.state.projects).toMatchObject([{ id: project.id, name: 'Typed again' }]);
    expect(recovered.conflicts).toEqual([]);
  });

  it('does not reopen a project closed after its journaled edit was undone', async () => {
    const tab = openTab();
    const loaded = await tab.loadWorkbench();
    tab.journalBeforeUnload(named(loaded.state, 'Typed, then undone'));
    await expect(tab.flushProjectToServer(loaded.state.projects[0]!)).resolves.toMatchObject({ kind: 'acknowledged' });
    await tab.persistEmptySession(loaded.state);
    failEveryJournalDeletion();

    const reopened = await openTab().loadWorkbench();

    expect(reopened.state.projects.map(({ id }) => id)).not.toContain(project.id);
    expect(serverName()).toBe(project.name);
  });

  it('does not raise a conflict for an edit undone back to a save that was still in flight', async () => {
    const tab = openTab();
    const loaded = await tab.loadWorkbench();
    let release!: () => void;
    server.gate = new Promise((resolve) => {
      release = resolve;
    });
    const saved = named(loaded.state, 'Saved');
    const saving = tab.saveWorkbench(saved);
    await vi.waitFor(() => expect(server.updates).toBe(1));
    tab.journalBeforeUnload(named(loaded.state, 'Typed, then undone'));
    const undoing = tab.saveWorkbench(saved);
    server.gate = null;
    release();
    await Promise.all([saving, undoing]);
    failEveryJournalDeletion();

    const recovered = await openTab().loadWorkbench();

    expect(recovered.conflicts).toEqual([]);
    expect(recovered.state.projects).toMatchObject([{ id: project.id, name: 'Saved' }]);
    expect(server.records.get(project.id)).toMatchObject({ name: 'Saved', revision: 2 });
  });

  it('keeps a newer journal across an older acknowledgement, rebased onto it', async () => {
    const tab = openTab();
    const loaded = await tab.loadWorkbench();
    let release!: () => void;
    server.gate = new Promise((resolve) => {
      release = resolve;
    });
    const saving = tab.saveWorkbench(named(loaded.state, 'Saved'));
    await vi.waitFor(() => expect(server.updates).toBe(1));
    tab.journalBeforeUnload(named(loaded.state, 'Typed while it saved'));
    server.gate = null;
    release();
    await saving;
    // The page unloads before another save.

    const reloaded = openTab();
    const recovered = await reloaded.loadWorkbench();

    expect(recovered.conflicts).toEqual([]);
    expect(recovered.state.projects).toMatchObject([{ id: project.id, name: 'Typed while it saved' }]);
    await reloaded.saveWorkbench(recovered.state);
    expect(server.records.get(project.id)).toMatchObject({ name: 'Typed while it saved', revision: 3 });
  });

  it('does not bring back an acknowledged journal whose deletion failed', async () => {
    const tab = openTab();
    const loaded = await tab.loadWorkbench();
    const hidden = named(loaded.state, 'While hidden');
    tab.journalBeforeUnload(hidden);
    await tab.saveWorkbench(hidden);
    await tab.saveWorkbench(named(loaded.state, 'Newer'));
    failEveryJournalDeletion();

    const recovered = await openTab().loadWorkbench();

    expect(recovered.conflicts).toEqual([]);
    expect(recovered.state.projects).toMatchObject([{ id: project.id, name: 'Newer' }]);
  });

  it('does not bring back a journaled edit whose draft the user discarded', async () => {
    const tab = openTab();
    const loaded = await tab.loadWorkbench();
    const local = named(loaded.state, 'Local');
    saveElsewhere('Remote', 2);
    await expect(tab.saveWorkbench(local)).resolves.toMatchObject({ conflicts: [{ kind: 'revision' }] });
    tab.journalBeforeUnload(named(loaded.state, 'Typed into the conflict'));

    await tab.resolveConflictUseServer(project.id);
    failEveryJournalDeletion();
    const recovered = await openTab().loadWorkbench();

    expect(recovered.conflicts).toEqual([]);
    expect(recovered.state.projects).toMatchObject([{ id: project.id, name: 'Remote' }]);
  });

  it('rebases a journal onto its own staged save that reached the server unanswered, without a conflict', async () => {
    const tab = openTab();
    const loaded = await tab.loadWorkbench();
    server.dropNextUpdateResponse = true;
    await tab.saveWorkbench(named(loaded.state, 'Saved, answer lost'));
    expect(server.records.get(project.id)).toMatchObject({ name: 'Saved, answer lost', revision: 2 });
    tab.journalBeforeUnload(named(loaded.state, 'Typed after it'));

    const reloaded = openTab();
    const recovered = await reloaded.loadWorkbench();

    expect(recovered.conflicts).toEqual([]);
    expect(recovered.state.projects).toMatchObject([{ id: project.id, name: 'Typed after it' }]);
    await reloaded.saveWorkbench(recovered.state);
    expect(server.records.get(project.id)).toMatchObject({ name: 'Typed after it', revision: 3 });
  });

  it('still raises a conflict when the server moved past an unanswered save to something else', async () => {
    const tab = openTab();
    const loaded = await tab.loadWorkbench();
    server.dropNextUpdateResponse = true;
    await tab.saveWorkbench(named(loaded.state, 'Saved, answer lost'));
    saveElsewhere('Another tab', 3);
    tab.journalBeforeUnload(named(loaded.state, 'Typed after it'));

    const recovered = await openTab().loadWorkbench();

    expect(recovered.conflicts).toEqual([expect.objectContaining({ kind: 'revision', serverRevision: 3 })]);
    expect(serverName()).toBe('Another tab');
  });
};

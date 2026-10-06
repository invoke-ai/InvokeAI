import { expect, it } from 'vitest';

import {
  getUtf8ByteSize,
  type ProjectDraft,
  type ProjectDraftInput,
  type ProjectDraftStore,
  type ProjectUnloadJournalEntry,
} from './draftStore';

const DEFAULT_DOCUMENT = '{"id":"project-1","name":"Project"}';
export const createCopyReservation = (copyProjectId: string) => ({
  copyDocumentByteSize: `{"id":"${copyProjectId}"}`.length,
  copyDocumentJson: `{"id":"${copyProjectId}"}`,
  copyProjectGeneration: 1,
  copyProjectId,
  copyProjectMinimumCanvasSchemaVersion: 3,
  copyProjectName: `Copy ${copyProjectId}`,
  copySourceProjectName: 'Project',
});

export const createProjectDraftInput = (overrides: Partial<ProjectDraftInput> = {}): ProjectDraftInput => ({
  baseRevision: 3,
  documentJson: DEFAULT_DOCUMENT,
  documentSchemaVersion: 2,
  editorSessionId: 'session-a',
  generation: 1,
  projectId: 'project-1',
  updatedAt: 100,
  writerToken: 'writer-a',
  ...overrides,
});

export const createProjectDraft = (overrides: Partial<ProjectDraft> = {}): ProjectDraft => {
  const documentJson = overrides.documentJson ?? DEFAULT_DOCUMENT;
  return {
    ...createProjectDraftInput({ ...overrides, documentJson }),
    documentByteSize: getUtf8ByteSize(documentJson),
    state: 'dirty',
    ...overrides,
  } as ProjectDraft;
};

export const createUnloadJournalEntry = (
  overrides: Partial<ProjectUnloadJournalEntry> = {}
): ProjectUnloadJournalEntry => {
  const documentJson = overrides.documentJson ?? '{"id":"project-1","name":"Journaled"}';
  return {
    accountId: 'account-a',
    baseMinimumCanvasSchemaVersion: 3,
    baseRevision: 3,
    documentByteSize: getUtf8ByteSize(documentJson),
    documentJson,
    documentSchemaVersion: 2,
    editorSessionId: 'session-a',
    generation: 2,
    journaledAt: 500,
    ownerEditorSessionId: 'session-a',
    projectId: 'project-1',
    recordType: 'unload-journal',
    schemaVersion: 1,
    writerToken: 'writer-a',
    ...overrides,
  };
};

export const testProjectDraftStoreContract = (createStore: () => Promise<ProjectDraftStore>): void => {
  it('stages monotonic generations and recognizes reconstructed idempotent retries', async () => {
    const store = await createStore();
    await expect(store.stage(createProjectDraftInput())).resolves.toEqual({ kind: 'stored' });
    await expect(store.stage(createProjectDraftInput({ baseRevision: 1, updatedAt: 200 }))).resolves.toEqual({
      kind: 'replayed',
    });
    await expect(store.stage(createProjectDraftInput({ documentJson: '{"different":true}' }))).resolves.toEqual({
      kind: 'generation-conflict',
    });
    await expect(store.stage(createProjectDraftInput({ generation: 0 }))).resolves.toEqual({ kind: 'stale' });
    await expect(store.stage(createProjectDraftInput({ generation: 2, updatedAt: 200 }))).resolves.toEqual({
      kind: 'stored',
    });
    await expect(store.get('project-1', 'session-a')).resolves.toEqual({
      draft: createProjectDraft({ generation: 2, updatedAt: 200 }),
      kind: 'found',
    });
    store.close();
  });

  it('preserves an acknowledged rebase when a newer generation arrives', async () => {
    const store = await createStore();
    await store.stage(createProjectDraftInput({ generation: 2 }));
    await expect(store.settleAcknowledgement('project-1', 'session-a', 'writer-a', 1, 7)).resolves.toMatchObject({
      draft: { baseRevision: 7, generation: 2 },
      kind: 'rebased',
    });

    await store.stage(createProjectDraftInput({ baseRevision: 3, documentJson: '{"newer":true}', generation: 3 }));

    await expect(store.get('project-1', 'session-a')).resolves.toMatchObject({
      draft: { baseRevision: 7, documentJson: '{"newer":true}', generation: 3 },
      kind: 'found',
    });
    await expect(store.settleAcknowledgement('project-1', 'session-a', 'writer-a', 3, 8)).resolves.toEqual({
      kind: 'deleted',
    });
    store.close();
  });

  it('removes an older durable generation after a newer volatile generation is acknowledged', async () => {
    const store = await createStore();
    await store.stage(createProjectDraftInput());

    await expect(store.settleAcknowledgement('project-1', 'session-a', 'writer-a', 2, 7)).resolves.toEqual({
      kind: 'deleted',
    });
    await expect(store.get('project-1', 'session-a')).resolves.toMatchObject({ kind: 'empty' });
    store.close();
  });

  it('keeps conflict metadata sticky while newer edits replace only authored fields', async () => {
    const store = await createStore();
    await store.stage(createProjectDraftInput());
    await store.settleConflict('project-1', 'session-a', 'writer-a', 1, {
      kind: 'revision',
      serverRevision: 9,
    });

    await store.stage(createProjectDraftInput({ documentJson: '{"newer":true}', generation: 2 }));

    await expect(store.get('project-1', 'session-a')).resolves.toMatchObject({
      draft: {
        conflict: { kind: 'revision', serverRevision: 9 },
        documentJson: '{"newer":true}',
        generation: 2,
        state: 'conflict',
      },
      kind: 'found',
    });
    store.close();
  });

  it('keeps schema refusal metadata sticky while newer edits are staged', async () => {
    const store = await createStore();
    await store.stage(createProjectDraftInput());
    await store.settleSchemaRefusal('project-1', 'session-a', 'writer-a', 1, {
      kind: 'canvas',
      maxCanvasSchemaVersion: 3,
      minimumCanvasSchemaVersion: 4,
    });

    await store.stage(createProjectDraftInput({ generation: 2 }));

    await expect(store.get('project-1', 'session-a')).resolves.toMatchObject({
      draft: {
        generation: 2,
        refusal: { kind: 'canvas', maxCanvasSchemaVersion: 3, minimumCanvasSchemaVersion: 4 },
        state: 'schema-refused',
      },
      kind: 'found',
    });
    store.close();
  });

  it('resumes a schema-refused lineage without changing its document', async () => {
    const store = await createStore();
    await store.stage(createProjectDraftInput());
    await store.settleSchemaRefusal('project-1', 'session-a', 'writer-a', 1, {
      kind: 'canvas',
      maxCanvasSchemaVersion: 3,
      minimumCanvasSchemaVersion: 4,
    });

    await expect(store.resumeSchemaRefused('project-1', 'session-a', 'writer-a', 1)).resolves.toMatchObject({
      draft: { documentJson: DEFAULT_DOCUMENT, state: 'dirty' },
      kind: 'marked',
    });
    store.close();
  });

  it('reserves one copy identity across edits and retargets the lineage atomically', async () => {
    const store = await createStore();
    await store.stage(createProjectDraftInput({ documentJson: '{"id":"project-1"}' }));
    await store.settleConflict('project-1', 'session-a', 'writer-a', 1, { kind: 'deleted' });
    await expect(
      store.reserveCopyIdentity('project-1', 'session-a', 'writer-a', createCopyReservation('copy-1'))
    ).resolves.toEqual({ ...createCopyReservation('copy-1'), kind: 'reserved' });
    await store.stage(createProjectDraftInput({ documentJson: '{"id":"project-1","edited":true}', generation: 2 }));
    await expect(
      store.reserveCopyIdentity('project-1', 'session-a', 'writer-a', createCopyReservation('copy-2'))
    ).resolves.toEqual({ ...createCopyReservation('copy-1'), kind: 'reserved' });
    await expect(
      store.reserveCopyIdentity('project-1', 'session-a', 'writer-a', createCopyReservation('copy-2'), 'copy-1')
    ).resolves.toEqual({ ...createCopyReservation('copy-2'), kind: 'reserved' });
    await expect(
      store.reserveCopyIdentity('project-1', 'session-a', 'writer-a', createCopyReservation('copy-3'), 'copy-1')
    ).resolves.toEqual({
      kind: 'stale',
    });
    await store.reserveCopyIdentity('project-1', 'session-a', 'writer-a', createCopyReservation('copy-1'), 'copy-2');

    await expect(
      store.retargetAcknowledgedCopy({
        acknowledgedRevision: 1,
        copyProjectId: 'copy-1',
        editorSessionId: 'session-a',
        projectId: 'project-1',
        retargetDocument: () => '{"id":"copy-1","edited":true}',
        sentGeneration: 1,
        writerToken: 'writer-a',
      })
    ).resolves.toMatchObject({
      draft: {
        baseRevision: 1,
        documentJson: '{"id":"copy-1","edited":true}',
        generation: 2,
        projectId: 'copy-1',
        state: 'dirty',
      },
      kind: 'retargeted',
    });
    await expect(store.get('project-1', 'session-a')).resolves.toEqual({
      kind: 'retargeted',
      projectId: 'copy-1',
      revision: 1,
      writerToken: 'writer-a',
    });
    store.close();
  });

  it('does not retain an exact-generation copy after the server acknowledges it', async () => {
    const store = await createStore();
    await store.stage(createProjectDraftInput({ documentJson: '{"id":"project-1"}' }));
    await store.reserveCopyIdentity('project-1', 'session-a', 'writer-a', createCopyReservation('copy-1'));
    const retarget = {
      acknowledgedRevision: 1,
      copyProjectId: 'copy-1',
      editorSessionId: 'session-a',
      projectId: 'project-1',
      retargetDocument: () => '{"id":"copy-1"}',
      sentGeneration: 1,
      writerToken: 'writer-a',
    };

    await expect(store.retargetAcknowledgedCopy(retarget)).resolves.toEqual({ draft: null, kind: 'retargeted' });
    await expect(store.retargetAcknowledgedCopy(retarget)).resolves.toEqual({ draft: null, kind: 'retargeted' });
    await expect(store.get('project-1', 'session-a')).resolves.toEqual({
      kind: 'retargeted',
      projectId: 'copy-1',
      revision: 1,
      writerToken: 'writer-a',
    });
    await expect(store.get('copy-1', 'session-a')).resolves.toEqual({
      kind: 'empty',
      writerState: 'active',
      writerToken: 'writer-a',
    });
    await expect(
      store.stage(
        createProjectDraftInput({
          documentJson: '{"id":"copy-1","edited":true}',
          generation: 2,
          projectId: 'copy-1',
        })
      )
    ).resolves.toEqual({ kind: 'stored' });
    await expect(store.retargetAcknowledgedCopy(retarget)).resolves.toMatchObject({
      draft: {
        documentJson: '{"id":"copy-1","edited":true}',
        generation: 2,
        projectId: 'copy-1',
        writerToken: 'writer-a',
      },
      kind: 'retargeted',
    });
    store.close();
  });

  it('pages and acknowledges durable retarget handoffs', async () => {
    const store = await createStore();
    for (const [projectId, copyProjectId] of [
      ['project-1', 'copy-1'],
      ['project-2', 'copy-2'],
    ] as const) {
      await store.stage(createProjectDraftInput({ projectId }));
      await store.reserveCopyIdentity(projectId, 'session-a', 'writer-a', createCopyReservation(copyProjectId));
      await store.retargetAcknowledgedCopy({
        acknowledgedRevision: 1,
        copyProjectId,
        editorSessionId: 'session-a',
        projectId,
        retargetDocument: () => `{"id":"${copyProjectId}"}`,
        sentGeneration: 1,
        writerToken: 'writer-a',
      });
    }

    const first = await store.listRetargets({ limit: 1 });
    expect(first).toMatchObject({
      items: [{ editorSessionId: 'session-a', projectId: 'project-1', targetProjectId: 'copy-1' }],
      kind: 'available',
      nextCursor: ['project-1', 'session-a', 'copy-1'],
    });
    if (first.kind !== 'available' || !first.nextCursor) {
      throw new Error('Expected a second retarget page.');
    }
    await expect(store.listRetargets({ after: first.nextCursor, limit: 1 })).resolves.toMatchObject({
      items: [{ projectId: 'project-2', targetProjectId: 'copy-2' }],
      kind: 'available',
      nextCursor: null,
    });
    await expect(store.acknowledgeRetarget('project-1', 'session-a', 'wrong-copy')).resolves.toEqual({ kind: 'stale' });
    await expect(store.acknowledgeRetarget('project-1', 'session-a', 'copy-1')).resolves.toEqual({ kind: 'deleted' });
    await expect(store.listRetargets()).resolves.toMatchObject({
      items: [{ projectId: 'project-2' }],
      kind: 'available',
    });
    store.close();
  });

  it('fences a retarget replay after target ownership rotates', async () => {
    const store = await createStore();
    await store.stage(createProjectDraftInput());
    await store.reserveCopyIdentity('project-1', 'session-a', 'writer-a', createCopyReservation('copy-1'));
    const retarget = {
      acknowledgedRevision: 1,
      copyProjectId: 'copy-1',
      editorSessionId: 'session-a',
      projectId: 'project-1',
      retargetDocument: () => '{"id":"copy-1"}',
      sentGeneration: 1,
      writerToken: 'writer-a',
    };
    await store.retargetAcknowledgedCopy(retarget);
    await store.claimWriter('copy-1', 'session-a', 'writer-a', 'writer-b');
    await store.stage(createProjectDraftInput({ generation: 2, projectId: 'copy-1', writerToken: 'writer-b' }));

    await expect(store.retargetAcknowledgedCopy(retarget)).resolves.toEqual({ kind: 'fenced' });
    store.close();
  });

  it('reports a failed copy transform without mutating the source', async () => {
    const store = await createStore();
    await store.stage(createProjectDraftInput());
    await store.reserveCopyIdentity('project-1', 'session-a', 'writer-a', createCopyReservation('copy-1'));
    await store.stage(createProjectDraftInput({ generation: 2 }));

    await expect(
      store.retargetAcknowledgedCopy({
        acknowledgedRevision: 1,
        copyProjectId: 'copy-1',
        editorSessionId: 'session-a',
        projectId: 'project-1',
        retargetDocument: () => {
          throw new Error('invalid document');
        },
        sentGeneration: 1,
        writerToken: 'writer-a',
      })
    ).resolves.toEqual({ kind: 'corrupt' });
    await expect(store.get('project-1', 'session-a')).resolves.toMatchObject({ kind: 'found' });
    store.close();
  });

  it('fences stale writers while allowing adoption and legitimate reopening', async () => {
    const store = await createStore();
    await store.stage(createProjectDraftInput());
    await store.startFreshWriter('project-1', 'session-b', null, 'writer-b');

    await expect(store.adopt('project-1', 'session-a', 'session-b', 'writer-b')).resolves.toEqual({ kind: 'adopted' });
    await expect(store.claimWriter('project-1', 'session-a', 'writer-a', 'writer-stale')).resolves.toEqual({
      kind: 'fenced',
    });
    await expect(store.stage(createProjectDraftInput({ generation: 2 }))).resolves.toEqual({ kind: 'fenced' });
    await expect(store.adopt('project-1', 'session-b', 'session-a', 'writer-a2')).resolves.toEqual({ kind: 'adopted' });
    await expect(
      store.stage(createProjectDraftInput({ editorSessionId: 'session-b', generation: 2, writerToken: 'writer-b' }))
    ).resolves.toEqual({
      kind: 'fenced',
    });

    await expect(store.claimWriter('project-1', 'session-a', 'writer-a2', 'writer-a3')).resolves.toEqual({
      kind: 'claimed',
    });
    await expect(store.settleAcknowledgement('project-1', 'session-a', 'writer-a2', 1, 7)).resolves.toEqual({
      kind: 'fenced',
    });
    await expect(
      store.settleConflict('project-1', 'session-a', 'writer-a2', 1, {
        kind: 'revision',
        serverRevision: 8,
      })
    ).resolves.toEqual({ kind: 'fenced' });
    await expect(
      store.reserveCopyIdentity('project-1', 'session-a', 'writer-a2', createCopyReservation('copy-1'))
    ).resolves.toEqual({
      kind: 'fenced',
    });
    await expect(
      store.retargetAcknowledgedCopy({
        acknowledgedRevision: 1,
        copyProjectId: 'copy-1',
        editorSessionId: 'session-a',
        projectId: 'project-1',
        retargetDocument: () => {
          throw new Error('A fenced writer must not transform the current draft.');
        },
        sentGeneration: 1,
        writerToken: 'writer-a2',
      })
    ).resolves.toEqual({ kind: 'fenced' });
    await expect(store.stage(createProjectDraftInput({ generation: 2, writerToken: 'writer-a2' }))).resolves.toEqual({
      kind: 'fenced',
    });
    await expect(store.stage(createProjectDraftInput({ generation: 2, writerToken: 'writer-a3' }))).resolves.toEqual({
      kind: 'stored',
    });
    store.close();
  });

  it('keeps writer ownership after acknowledgement and explicit deletion', async () => {
    const store = await createStore();
    await store.stage(createProjectDraftInput());
    await store.settleAcknowledgement('project-1', 'session-a', 'writer-a', 1, 7);
    await expect(store.claimWriter('project-1', 'session-a', 'writer-a', 'writer-a2')).resolves.toEqual({
      kind: 'claimed',
    });
    await expect(store.stage(createProjectDraftInput({ generation: 2 }))).resolves.toEqual({ kind: 'fenced' });
    await expect(store.stage(createProjectDraftInput({ generation: 2, writerToken: 'writer-a2' }))).resolves.toEqual({
      kind: 'stored',
    });
    await store.delete('project-1', 'session-a', 'writer-a2');
    await expect(store.claimWriter('project-1', 'session-a', 'writer-a2', 'writer-a3')).resolves.toEqual({
      kind: 'claimed',
    });
    await expect(store.stage(createProjectDraftInput({ generation: 3, writerToken: 'writer-a2' }))).resolves.toEqual({
      kind: 'fenced',
    });
    store.close();
  });

  it('starts an empty lineage with compare-and-swap ownership', async () => {
    const store = await createStore();
    await store.stage(createProjectDraftInput());
    await store.settleAcknowledgement('project-1', 'session-a', 'writer-a', 1, 7);

    const outcomes = await Promise.all([
      store.startFreshWriter('project-1', 'session-a', 'writer-a', 'writer-b'),
      store.startFreshWriter('project-1', 'session-a', 'writer-a', 'writer-c'),
    ]);

    expect(outcomes.map((outcome) => outcome.kind).sort()).toEqual(['fenced', 'started']);
    store.close();
  });

  it('requires an explicit fresh-lineage transition before reusing an adopted source', async () => {
    const store = await createStore();
    await store.stage(createProjectDraftInput());
    await store.adopt('project-1', 'session-a', 'session-b', 'writer-b');

    await expect(store.stage(createProjectDraftInput({ generation: 2, writerToken: 'writer-new' }))).resolves.toEqual({
      kind: 'fenced',
    });
    await expect(store.startFreshWriter('project-1', 'session-a', 'writer-a', 'writer-new')).resolves.toEqual({
      kind: 'started',
    });
    await expect(store.stage(createProjectDraftInput({ generation: 2, writerToken: 'writer-new' }))).resolves.toEqual({
      kind: 'stored',
    });
    store.close();
  });

  it('deletes only the owned writer lineage', async () => {
    const store = await createStore();
    await store.stage(createProjectDraftInput());
    await store.stage(createProjectDraftInput({ editorSessionId: 'session-b', writerToken: 'writer-b' }));

    await expect(store.delete('project-1', 'session-a', 'wrong-writer')).resolves.toEqual({ kind: 'fenced' });
    await expect(store.delete('project-1', 'session-a', 'writer-a')).resolves.toEqual({ kind: 'deleted' });
    await expect(store.listForProject('project-1')).resolves.toMatchObject({
      items: [{ editorSessionId: 'session-b' }],
      kind: 'available',
    });
    store.close();
  });

  it('refuses corrupt-record cleanup for a valid lineage', async () => {
    const store = await createStore();
    await store.stage(createProjectDraftInput());

    await expect(store.deleteCorrupt('project-1', 'session-a')).resolves.toEqual({ kind: 'not-corrupt' });
    await expect(store.get('project-1', 'session-a')).resolves.toMatchObject({ kind: 'found' });
    store.close();
  });

  it('paginates bounded metadata without returning document bodies', async () => {
    const store = await createStore();
    await store.stage(createProjectDraftInput({ projectId: 'project-1' }));
    await store.stage(createProjectDraftInput({ projectId: 'project-2' }));
    await store.stage(createProjectDraftInput({ projectId: 'project-3' }));

    const first = await store.list({ limit: 2 });
    expect(first).toMatchObject({ kind: 'available', nextCursor: ['project-2', 'session-a'] });
    if (first.kind !== 'available') {
      throw new Error('Expected draft metadata.');
    }
    expect(first.items.map((item) => item.projectId)).toEqual(['project-1', 'project-2']);
    await expect(store.list({ after: first.nextCursor!, limit: 2 })).resolves.toMatchObject({
      items: [{ projectId: 'project-3' }],
      kind: 'available',
      nextCursor: null,
    });
    store.close();
  });

  it('enforces list limits in the store instead of trusting callers', async () => {
    const store = await createStore();
    for (let index = 0; index < 101; index += 1) {
      await store.stage(createProjectDraftInput({ projectId: `project-${index.toString().padStart(3, '0')}` }));
    }
    for (let index = 1; index < 34; index += 1) {
      await store.stage(
        createProjectDraftInput({
          editorSessionId: `session-${index.toString().padStart(3, '0')}`,
          writerToken: `writer-${index}`,
        })
      );
    }

    const page = await store.list({ limit: Number.MAX_SAFE_INTEGER });
    const projectRows = await store.listForProject('project-1', { limit: Number.MAX_SAFE_INTEGER });
    expect(page.kind).toBe('available');
    expect(projectRows.kind).toBe('available');
    if (page.kind === 'available' && projectRows.kind === 'available') {
      expect(page.items).toHaveLength(100);
      expect(projectRows.items).toHaveLength(32);
      expect(projectRows.nextCursor).toBe('session-032');
      await expect(
        store.listForProject('project-1', { after: projectRows.nextCursor!, limit: Number.MAX_SAFE_INTEGER })
      ).resolves.toMatchObject({
        items: [{ editorSessionId: 'session-033' }],
        kind: 'available',
        nextCursor: null,
      });
    }
    store.close();
  });

  const journal = (store: ProjectDraftStore, ...entries: ProjectUnloadJournalEntry[]) =>
    store.journalBeforeUnload({ entries, retired: [] });
  const reconcile = (store: ProjectDraftStore) => store.reconcileUnloadJournal('account-a', 1_000);
  const outcomesOf = async (store: ProjectDraftStore) => {
    const result = await reconcile(store);
    return result.kind === 'available' ? result.outcomes.map(({ outcome }) => outcome) : result.kind;
  };

  it('reconciles an unload journal into its lineage as the next staged generation, once', async () => {
    const store = await createStore();
    await store.stage(createProjectDraftInput());
    expect(journal(store, createUnloadJournalEntry({ baseRevision: 1 }))).toMatchObject({ kind: 'started' });

    await expect(reconcile(store)).resolves.toEqual({
      kind: 'available',
      outcomes: [
        {
          editorSessionId: 'session-a',
          outcome: 'applied',
          projectId: 'project-1',
          replacedDraft: { documentJson: createProjectDraftInput().documentJson, generation: 1 },
        },
      ],
    });
    // The staged base revision is kept, as staging a newer generation keeps it.
    await expect(store.get('project-1', 'session-a')).resolves.toEqual({
      draft: createProjectDraft({
        documentJson: createUnloadJournalEntry().documentJson,
        generation: 2,
        updatedAt: 500,
      }),
      kind: 'found',
    });
    await expect(reconcile(store)).resolves.toEqual({ kind: 'available', outcomes: [] });
    // The lineage is consistent: the next load claims it and stages on top.
    await expect(store.claimWriter('project-1', 'session-a', 'writer-a', 'writer-b')).resolves.toEqual({
      kind: 'claimed',
    });
    await expect(store.stage(createProjectDraftInput({ generation: 3, writerToken: 'writer-b' }))).resolves.toEqual({
      kind: 'stored',
    });
    store.close();
  });

  it('finds an entry it already applied superseded, leaving the draft as it was', async () => {
    const store = await createStore();
    journal(store, createUnloadJournalEntry());
    await reconcile(store);
    journal(store, createUnloadJournalEntry());

    await expect(outcomesOf(store)).resolves.toEqual(['superseded']);
    await expect(store.get('project-1', 'session-a')).resolves.toMatchObject({
      draft: { documentJson: createUnloadJournalEntry().documentJson, generation: 2 },
      kind: 'found',
    });
    store.close();
  });

  it("keeps one entry per lineage and writer: a write replaces the writer's lower generations", async () => {
    const store = await createStore();
    journal(store, createUnloadJournalEntry({ baseRevision: null, generation: 2 }));
    journal(
      store,
      createUnloadJournalEntry({
        baseRevision: null,
        documentJson: '{"id":"project-1","name":"Later"}',
        generation: 3,
      }),
      createUnloadJournalEntry({ writerToken: 'writer-other' })
    );

    await expect(outcomesOf(store)).resolves.toEqual(['applied', 'fenced']);
    await expect(store.get('project-1', 'session-a')).resolves.toMatchObject({
      draft: { baseRevision: null, documentJson: '{"id":"project-1","name":"Later"}', generation: 3, state: 'dirty' },
      kind: 'found',
    });
    store.close();
  });

  it("retires a writer's entries for a lineage in the same blind write", async () => {
    const store = await createStore();
    journal(store, createUnloadJournalEntry({ generation: 2 }), createUnloadJournalEntry({ projectId: 'project-2' }));
    store.journalBeforeUnload({ entries: [], retired: [['project-1', 'session-a', 'writer-a', 2]] });

    await expect(reconcile(store)).resolves.toMatchObject({ outcomes: [{ projectId: 'project-2' }] });
    store.close();
  });

  it('discards a journal that staging already superseded', async () => {
    const store = await createStore();
    await store.stage(createProjectDraftInput());
    journal(store, createUnloadJournalEntry({ generation: 2 }));
    await store.stage(createProjectDraftInput({ documentJson: '{"newer":true}', generation: 3 }));

    await expect(outcomesOf(store)).resolves.toEqual(['superseded']);
    await expect(store.get('project-1', 'session-a')).resolves.toMatchObject({
      draft: { documentJson: '{"newer":true}', generation: 3 },
      kind: 'found',
    });
    store.close();
  });

  it('settles the journal with an acknowledgement, even when no draft is left to say so', async () => {
    const store = await createStore();
    await store.stage(createProjectDraftInput({ generation: 2 }));
    journal(store, createUnloadJournalEntry({ generation: 2 }));
    await expect(store.settleAcknowledgement('project-1', 'session-a', 'writer-a', 2, 4)).resolves.toEqual({
      kind: 'deleted',
    });
    // Another tab reconciling after the acknowledgement must not bring the acknowledged document back.
    await expect(outcomesOf(store)).resolves.toEqual(['settled']);
    await expect(store.get('project-1', 'session-a')).resolves.toMatchObject({ kind: 'empty' });

    // With nothing staged at all, an acknowledgement still settles what it covers.
    journal(store, createUnloadJournalEntry({ generation: 3 }));
    await expect(store.settleAcknowledgement('project-1', 'session-a', 'writer-a', 3, 5)).resolves.toEqual({
      kind: 'missing',
    });
    await expect(outcomesOf(store)).resolves.toEqual(['settled']);
    store.close();
  });

  it('keeps a journal newer than the acknowledged generation', async () => {
    const store = await createStore();
    await store.stage(createProjectDraftInput());
    journal(store, createUnloadJournalEntry({ generation: 3 }));
    await store.settleAcknowledgement('project-1', 'session-a', 'writer-a', 1, 4);

    await expect(outcomesOf(store)).resolves.toEqual(['applied']);
    await expect(store.get('project-1', 'session-a')).resolves.toMatchObject({
      draft: { baseRevision: 3, generation: 3 },
      kind: 'found',
    });
    store.close();
  });

  it('settles a journal in the claim, so an entry whose deletion failed stays settled', async () => {
    const store = await createStore();
    const entry = createUnloadJournalEntry({ baseRevision: null });
    journal(store, entry);
    // The lineage was never staged: settling claims it for the writer.
    await expect(store.settleUnloadJournal('project-1', 'session-a', 'writer-a', 2)).resolves.toEqual({
      kind: 'settled',
    });
    await expect(reconcile(store)).resolves.toEqual({ kind: 'available', outcomes: [] });
    journal(store, entry);
    await expect(outcomesOf(store)).resolves.toEqual(['settled']);
    await expect(store.get('project-1', 'session-a')).resolves.toMatchObject({ kind: 'empty' });

    // Deleting a draft settles through the generation it is given, in the same transaction.
    await store.stage(createProjectDraftInput({ generation: 3 }));
    journal(store, createUnloadJournalEntry({ generation: 4 }));
    await store.delete('project-1', 'session-a', 'writer-a', 4);
    journal(store, createUnloadJournalEntry({ generation: 4 }));
    await expect(outcomesOf(store)).resolves.toEqual(['settled']);
    // A later generation of the same writer is not affected.
    journal(store, createUnloadJournalEntry({ generation: 5 }));
    await expect(outcomesOf(store)).resolves.toEqual(['applied']);
    store.close();
  });

  it("ignores an earlier writer's settlement once the lineage changed hands", async () => {
    const store = await createStore();
    await store.stage(createProjectDraftInput());
    await store.settleUnloadJournal('project-1', 'session-a', 'writer-a', 9);
    await store.claimWriter('project-1', 'session-a', 'writer-a', 'writer-b');
    journal(store, createUnloadJournalEntry({ generation: 2, writerToken: 'writer-b' }));

    await expect(outcomesOf(store)).resolves.toEqual(['applied']);
    await expect(store.settleUnloadJournal('project-1', 'session-a', 'writer-a', 9)).resolves.toEqual({
      kind: 'fenced',
    });
    store.close();
  });

  it('never applies a journal to a lineage another editor adopted', async () => {
    const store = await createStore();
    await store.stage(createProjectDraftInput());
    journal(store, createUnloadJournalEntry({ documentJson: '{"stale":true}' }));
    await store.adopt('project-1', 'session-a', 'session-b', 'writer-b');
    await store.stage(
      createProjectDraftInput({
        documentJson: '{"adopter":true}',
        editorSessionId: 'session-b',
        generation: 2,
        writerToken: 'writer-b',
      })
    );

    await expect(outcomesOf(store)).resolves.toEqual(['fenced']);
    await expect(store.get('project-1', 'session-b')).resolves.toMatchObject({
      draft: { documentJson: '{"adopter":true}', generation: 2 },
      kind: 'found',
    });
    await expect(store.get('project-1', 'session-a')).resolves.toMatchObject({ kind: 'empty', writerState: 'fenced' });
    await expect(reconcile(store)).resolves.toEqual({ kind: 'available', outcomes: [] });
    store.close();
  });

  it('leaves the entries written by a live page in place, and reconciles them once it is gone', async () => {
    const store = await createStore();
    journal(
      store,
      // Written by page A under its own lineage.
      createUnloadJournalEntry(),
      // Written by page B under a lineage named after page A, which the session blob handed on.
      createUnloadJournalEntry({ editorSessionId: 'session-a:writer:w', ownerEditorSessionId: 'session-b' }),
      // Written by page A under a lineage named after page B.
      createUnloadJournalEntry({ editorSessionId: 'session-b', ownerEditorSessionId: 'session-a' }),
      // Written before entries named their page.
      createUnloadJournalEntry({
        editorSessionId: 'session-c',
        ownerEditorSessionId: undefined,
        projectId: 'project-2',
      })
    );
    const asked: string[] = [];
    const isEditorSessionLive = (editorSessionId: string) => {
      asked.push(editorSessionId);
      return Promise.resolve(editorSessionId === 'session-a');
    };

    await expect(store.reconcileUnloadJournal('account-a', 1_000, { isEditorSessionLive })).resolves.toEqual({
      kind: 'available',
      outcomes: [
        { editorSessionId: 'session-a', outcome: 'live', projectId: 'project-1' },
        { editorSessionId: 'session-a:writer:w', outcome: 'applied', projectId: 'project-1' },
        { editorSessionId: 'session-b', outcome: 'live', projectId: 'project-1' },
        { editorSessionId: 'session-c', outcome: 'applied', projectId: 'project-2' },
      ],
    });
    // Asked once per page, never about a lineage.
    expect(asked).toEqual(['session-a', 'session-b']);
    await expect(store.get('project-1', 'session-a')).resolves.toEqual({ kind: 'missing' });
    await expect(store.get('project-1', 'session-a:writer:w')).resolves.toMatchObject({ kind: 'found' });
    // The peek skips the live page's entries the same way, and keeps looking past them.
    await expect(store.peekUnloadJournalProjectIds(1)).resolves.toEqual({
      kind: 'available',
      projectIds: ['project-1'],
    });
    journal(
      store,
      createUnloadJournalEntry({
        editorSessionId: 'session-c',
        ownerEditorSessionId: 'session-c',
        projectId: 'project-3',
      })
    );
    await expect(store.peekUnloadJournalProjectIds(2, { isEditorSessionLive })).resolves.toEqual({
      kind: 'available',
      projectIds: ['project-3'],
    });
    // The page is gone: its entries are recovery material now.
    await expect(outcomesOf(store)).resolves.toEqual(['applied', 'applied', 'applied']);
    await expect(store.get('project-1', 'session-a')).resolves.toMatchObject({
      draft: { documentJson: createUnloadJournalEntry().documentJson, generation: 2 },
      kind: 'found',
    });
    store.close();
  });

  it("discards one writer's journal of a lineage through a generation, blind", async () => {
    const store = await createStore();
    journal(
      store,
      createUnloadJournalEntry({ generation: 2 }),
      createUnloadJournalEntry({ projectId: 'project-2' }),
      createUnloadJournalEntry({ writerToken: 'writer-b' })
    );
    journal(store, createUnloadJournalEntry({ documentJson: '{"id":"project-1","name":"Newer"}', generation: 4 }));

    await expect(store.discardUnloadJournal('project-1', 'session-a', 'writer-a', 3)).resolves.toEqual({
      kind: 'deleted',
    });
    await expect(store.discardUnloadJournal('project-2', 'session-a', 'writer-a')).resolves.toEqual({
      kind: 'deleted',
    });

    await expect(outcomesOf(store)).resolves.toEqual(['applied', 'fenced']);
    await expect(store.get('project-1', 'session-a')).resolves.toMatchObject({
      draft: { documentJson: '{"id":"project-1","name":"Newer"}', generation: 4, writerToken: 'writer-a' },
      kind: 'found',
    });
    store.close();
  });

  it('discards journals of another account or record schema without touching drafts', async () => {
    const store = await createStore();
    await store.stage(createProjectDraftInput());
    journal(store, createUnloadJournalEntry({ accountId: 'account-b' }), {
      ...createUnloadJournalEntry({ projectId: 'project-2' }),
      schemaVersion: 0,
    } as unknown as ProjectUnloadJournalEntry);

    await expect(reconcile(store)).resolves.toEqual({
      kind: 'available',
      outcomes: [
        { editorSessionId: 'session-a', outcome: 'foreign-account', projectId: 'project-1' },
        { editorSessionId: 'session-a', outcome: 'invalid', projectId: 'project-2' },
      ],
    });
    await expect(store.get('project-1', 'session-a')).resolves.toEqual({ draft: createProjectDraft(), kind: 'found' });
    await expect(store.get('project-2', 'session-a')).resolves.toEqual({ kind: 'missing' });
    await expect(reconcile(store)).resolves.toEqual({ kind: 'available', outcomes: [] });
    store.close();
    expect(journal(store, createUnloadJournalEntry())).toEqual({ kind: 'unavailable' });
  });

  it('reconciles every entry past one page, and peeks at which projects have them', async () => {
    const store = await createStore();
    const projectIds = Array.from({ length: 105 }, (_, index) => `project-${index.toString().padStart(3, '0')}`);
    journal(store, ...projectIds.map((projectId) => createUnloadJournalEntry({ projectId })));

    await expect(store.peekUnloadJournalProjectIds(1)).resolves.toEqual({
      kind: 'available',
      projectIds: ['project-000'],
    });
    const outcomes = await outcomesOf(store);
    expect(outcomes).toHaveLength(105);
    expect(new Set(outcomes)).toEqual(new Set(['applied']));
    await expect(store.get('project-104', 'session-a')).resolves.toMatchObject({ kind: 'found' });
    await expect(store.peekUnloadJournalProjectIds(1)).resolves.toEqual({ kind: 'available', projectIds: [] });
    store.close();
  });

  it('returns detached bodies and explicit unavailability after close', async () => {
    const store = await createStore();
    await store.stage(createProjectDraftInput());
    const loaded = await store.get('project-1', 'session-a');
    if (loaded.kind === 'found') {
      loaded.draft.generation = 99;
    }
    await expect(store.get('project-1', 'session-a')).resolves.toEqual({
      draft: createProjectDraft(),
      kind: 'found',
    });

    store.close();
    await expect(store.stage(createProjectDraftInput())).resolves.toEqual({ kind: 'unavailable' });
    await expect(store.list()).resolves.toEqual({ kind: 'unavailable' });
  });
};

import { expect, it } from 'vitest';

import { getUtf8ByteSize, type ProjectDraft, type ProjectDraftInput, type ProjectDraftStore } from './draftStore';

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

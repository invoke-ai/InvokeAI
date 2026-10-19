import { describe, expect, it } from 'vitest';

import type { ProjectDraftInput, ProjectDraftWriterClaim } from './draftStore';

import {
  createMemoryProjectDraftStore,
  decideUnloadJournalEntry,
  getUtf8ByteSize,
  isProjectDraftWriterClaim,
  PROJECT_UNLOAD_JOURNAL_RETENTION_MS,
} from './draftStore';
import {
  createProjectDraft,
  createProjectDraftInput,
  createUnloadJournalEntry,
  testProjectDraftStoreContract,
} from './draftStore.contract';

describe('memory project draft store contract', () => {
  testProjectDraftStoreContract(() => Promise.resolve(createMemoryProjectDraftStore()));

  it('derives the exact UTF-8 byte count instead of trusting caller metadata', async () => {
    const store = createMemoryProjectDraftStore({ maxDraftBytes: 4 });

    await expect(
      store.stage({ ...createProjectDraftInput({ documentJson: 'éé' }), documentByteSize: 999 } as ProjectDraftInput)
    ).resolves.toEqual({
      kind: 'stored',
    });
    await expect(
      store.stage({
        ...createProjectDraftInput({ documentJson: 'ééx', generation: 2 }),
        documentByteSize: 1,
      } as ProjectDraftInput)
    ).resolves.toEqual({ kind: 'too-large' });
    await expect(store.get('project-1', 'session-a')).resolves.toMatchObject({
      draft: { documentByteSize: 4 },
      kind: 'found',
    });
  });

  it.each(['plain ascii', 'café', '😀', '\ud800', 'a😀é\udfff'])(
    'counts UTF-8 bytes like TextEncoder for %j',
    (value) => {
      expect(getUtf8ByteSize(value)).toBe(new TextEncoder().encode(value).byteLength);
    }
  );
});

describe('unload journal reconciliation rules', () => {
  const claim = (overrides: Partial<ProjectDraftWriterClaim> = {}): ProjectDraftWriterClaim =>
    ({
      editorSessionId: 'session-a',
      metadataRevision: 1,
      projectId: 'project-1',
      state: 'active',
      updatedAt: 1,
      writerToken: 'writer-a',
      ...overrides,
    }) as ProjectDraftWriterClaim;
  const decide = (
    entry: unknown,
    lineage: {
      claim?: ProjectDraftWriterClaim | 'corrupt';
      current?: ReturnType<typeof createProjectDraft> | 'corrupt';
    }
  ) =>
    decideUnloadJournalEntry({
      accountId: 'account-a',
      claim: lineage.claim,
      current: lineage.current ?? null,
      entry,
      maxDocumentBytes: 1024,
      now: 1_000,
    });

  it('applies a journal newer than the staged draft and keeps what staging keeps', () => {
    const current = createProjectDraft({
      baseRevision: 7,
      conflict: { kind: 'revision', serverRevision: 9 },
      generation: 2,
      state: 'conflict',
    });
    const entry = createUnloadJournalEntry({ baseRevision: 3, documentJson: '{"newest":true}', generation: 3 });

    expect(decide(entry, { claim: claim(), current })).toEqual({
      draft: {
        ...current,
        documentByteSize: entry.documentByteSize,
        documentJson: '{"newest":true}',
        generation: 3,
        updatedAt: entry.journaledAt,
      },
      kind: 'apply',
    });
  });

  it('starts a lineage the writer never staged, or one whose draft was acknowledged, from the journal', () => {
    const entry = createUnloadJournalEntry({ baseRevision: 4, generation: 2 });
    const expected = {
      draft: {
        baseMinimumCanvasSchemaVersion: 3,
        baseRevision: 4,
        documentByteSize: entry.documentByteSize,
        documentJson: entry.documentJson,
        documentSchemaVersion: entry.documentSchemaVersion,
        editorSessionId: 'session-a',
        generation: 2,
        projectId: 'project-1',
        state: 'dirty',
        updatedAt: entry.journaledAt,
        writerToken: 'writer-a',
      },
      kind: 'apply',
    };

    expect(decide(entry, {})).toEqual(expected);
    expect(decide(entry, { claim: claim() })).toEqual(expected);
  });

  it('discards a journal the lineage already holds at or above its generation', () => {
    const entry = createUnloadJournalEntry({ generation: 3 });

    expect(decide(entry, { claim: claim(), current: createProjectDraft({ generation: 3 }) })).toEqual({
      kind: 'discard',
      reason: 'superseded',
    });
    expect(decide(entry, { claim: claim(), current: createProjectDraft({ generation: 4 }) })).toEqual({
      kind: 'discard',
      reason: 'superseded',
    });
  });

  it('never writes into a lineage another writer holds or that moved away', () => {
    const entry = createUnloadJournalEntry();

    expect(decide(entry, { claim: claim({ writerToken: 'writer-b' }) })).toEqual({
      kind: 'discard',
      reason: 'fenced',
    });
    expect(
      decide(entry, {
        claim: claim({ adoptedByEditorSessionId: 'session-b', fenceReason: 'moved', state: 'fenced' }),
      })
    ).toEqual({ kind: 'discard', reason: 'fenced' });
  });

  it('keeps a journal while its lineage is damaged, for cleanup to settle', () => {
    const entry = createUnloadJournalEntry({ generation: 5 });

    expect(decide(entry, { claim: 'corrupt' })).toEqual({ kind: 'retain', reason: 'corrupt' });
    expect(decide(entry, { claim: claim(), current: 'corrupt' })).toEqual({ kind: 'retain', reason: 'corrupt' });
  });

  it('stops waiting for a damaged lineage after the retention period', () => {
    const entry = createUnloadJournalEntry({ journaledAt: 0 });
    const later = (lineage: Pick<Parameters<typeof decideUnloadJournalEntry>[0], 'claim' | 'current'>, now: number) =>
      decideUnloadJournalEntry({ accountId: 'account-a', entry, now, ...lineage });

    expect(later({ claim: 'corrupt', current: null }, PROJECT_UNLOAD_JOURNAL_RETENTION_MS)).toEqual({
      kind: 'retain',
      reason: 'corrupt',
    });
    expect(later({ claim: 'corrupt', current: null }, PROJECT_UNLOAD_JOURNAL_RETENTION_MS + 1)).toEqual({
      kind: 'discard',
      reason: 'expired',
    });
    expect(later({ claim: claim(), current: 'corrupt' }, PROJECT_UNLOAD_JOURNAL_RETENTION_MS + 1)).toEqual({
      kind: 'discard',
      reason: 'expired',
    });
  });

  it("discards what the writer settled, even with no draft left, and nothing of another writer's mark", () => {
    const entry = createUnloadJournalEntry({ generation: 4 });
    const settled = (generation: number, writerToken = 'writer-a') =>
      claim({ unloadJournalSettled: { generation, writerToken } });

    expect(decide(entry, { claim: settled(4) })).toEqual({ kind: 'discard', reason: 'settled' });
    expect(decide(entry, { claim: settled(9) })).toEqual({ kind: 'discard', reason: 'settled' });
    expect(decide(entry, { claim: settled(3) })).toMatchObject({ kind: 'apply' });
    expect(decide(entry, { claim: settled(9, 'writer-earlier') })).toMatchObject({ kind: 'apply' });
  });

  it('reads claims written before the settlement mark existed, and refuses a damaged mark', () => {
    expect(isProjectDraftWriterClaim(claim())).toBe(true);
    expect(isProjectDraftWriterClaim(claim({ unloadJournalSettled: { generation: 3, writerToken: 'writer-a' } }))).toBe(
      true
    );
    expect(
      isProjectDraftWriterClaim(claim({ unloadJournalSettled: { generation: -1, writerToken: 'writer-a' } }))
    ).toBe(false);
  });

  it.each([
    ['another account', { accountId: 'account-b' }, 'foreign-account'],
    ['an older record schema', { schemaVersion: 0 }, 'invalid'],
    ['a newer record schema', { schemaVersion: 2 }, 'invalid'],
    ['a byte count that does not match its document', { documentByteSize: 3 }, 'invalid'],
    ['a document over the size limit', { documentByteSize: 1025, documentJson: 'x'.repeat(1025) }, 'invalid'],
    ['no generation', { generation: 0 }, 'invalid'],
  ])('discards a journal from %s', (_label, overrides, reason) => {
    expect(decide({ ...createUnloadJournalEntry(), ...overrides }, { claim: claim() })).toEqual({
      kind: 'discard',
      reason,
    });
  });

  it('discards a damaged record before looking at its lineage', () => {
    expect(decide(null, { claim: 'corrupt' })).toEqual({ kind: 'discard', reason: 'invalid' });
    expect(decide({ recordType: 'unload-journal' }, {})).toEqual({ kind: 'discard', reason: 'invalid' });
  });
});

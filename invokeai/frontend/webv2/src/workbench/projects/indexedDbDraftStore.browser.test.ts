import { accountLifecycle } from '@platform/state/accountLifecycle';
import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  toProjectDraftBody,
  toProjectDraftMetadata,
  type ProjectDraftMetadata,
  type ProjectDraftStore,
} from './draftStore';
import {
  createCopyReservation,
  createProjectDraft,
  createProjectDraftInput,
  testProjectDraftStoreContract,
} from './draftStore.contract';
import { createAccountOwnedProjectDraftStore, createIndexedDbProjectDraftStore } from './indexedDbDraftStore';
import {
  deleteWorkbenchDatabase,
  getWorkbenchDatabaseName,
  openWorkbenchDatabase,
  WORKBENCH_DATABASE_VERSION,
  type WorkbenchDatabase,
} from './workbenchDatabase';

const suffixes = new Set<string>();
const stores: { close(): void }[] = [];

const createSuffix = (): string => {
  const suffix = `:browser-test:${crypto.randomUUID()}`;
  suffixes.add(suffix);
  return suffix;
};

const createStore = async (): Promise<ProjectDraftStore> => {
  const store = createIndexedDbProjectDraftStore(await openWorkbenchDatabase(createSuffix()));
  stores.push(store);
  return store;
};

const interceptFirstWrite = (database: WorkbenchDatabase, beforeWrite: () => void): WorkbenchDatabase => {
  let isArmed = true;
  return new Proxy(database, {
    get(target, property) {
      if (property === 'transaction') {
        return (...args: unknown[]) => {
          if (isArmed && args[1] === 'readwrite') {
            isArmed = false;
            beforeWrite();
          }
          return Reflect.apply(target.transaction, target, args);
        };
      }
      const value = Reflect.get(target, property, target) as unknown;
      return typeof value === 'function' ? value.bind(target) : value;
    },
  }) as WorkbenchDatabase;
};

const abortFirstBodyWriteWithQuota = (database: WorkbenchDatabase): WorkbenchDatabase => {
  let isArmed = true;
  return new Proxy(database, {
    get(target, property) {
      if (property === 'transaction') {
        return (...args: unknown[]) => {
          const transaction = Reflect.apply(target.transaction, target, args);
          if (!isArmed || args[1] !== 'readwrite') {
            return transaction;
          }
          isArmed = false;
          return new Proxy(transaction, {
            get(transactionTarget, transactionProperty) {
              if (transactionProperty === 'objectStore') {
                return (storeName: string) => {
                  const store = transactionTarget.objectStore(storeName);
                  if (storeName !== 'draftBodies') {
                    return store;
                  }
                  return new Proxy(store, {
                    get(storeTarget, storeProperty) {
                      if (storeProperty === 'put') {
                        return () => {
                          transactionTarget.abort();
                          throw new DOMException('quota', 'QuotaExceededError');
                        };
                      }
                      const value = Reflect.get(storeTarget, storeProperty, storeTarget) as unknown;
                      return typeof value === 'function' ? value.bind(storeTarget) : value;
                    },
                  });
                };
              }
              const value = Reflect.get(transactionTarget, transactionProperty, transactionTarget) as unknown;
              return typeof value === 'function' ? value.bind(transactionTarget) : value;
            },
          });
        };
      }
      const value = Reflect.get(target, property, target) as unknown;
      return typeof value === 'function' ? value.bind(target) : value;
    },
  }) as WorkbenchDatabase;
};

const reserveCopyInRawTransaction = async (database: WorkbenchDatabase): Promise<void> => {
  const transaction = database.transaction(['drafts', 'draftBodies', 'draftWriters'], 'readwrite');
  const metadataStore = transaction.objectStore('drafts');
  const bodyStore = transaction.objectStore('draftBodies');
  const writerStore = transaction.objectStore('draftWriters');
  const key: [string, string] = ['project-1', 'session-a'];
  const [metadata, body, claim] = await Promise.all([metadataStore.get(key), bodyStore.get(key), writerStore.get(key)]);
  if (!metadata || !body || !claim) {
    throw new Error('Expected draft metadata, body and writer claim.');
  }
  const { copyDocumentJson, ...reservationMetadata } = createCopyReservation('copy-1');
  const metadataRevision = metadata.metadataRevision + 1;
  await metadataStore.put({ ...metadata, ...reservationMetadata, metadataRevision });
  await bodyStore.put({ ...body, copyDocumentJson });
  await writerStore.put({ ...claim, metadataRevision });
  await transaction.done;
};

const createVersion2Database = (suffix: string, claims: unknown[]): Promise<void> =>
  new Promise((resolve, reject) => {
    const request = indexedDB.open(getWorkbenchDatabaseName(suffix), 2);
    request.addEventListener('upgradeneeded', () => {
      const database = request.result;
      const transaction = request.transaction!;
      const drafts = database.createObjectStore('drafts', { keyPath: ['projectId', 'editorSessionId'] });
      drafts.createIndex('byProject', 'projectId');
      const bodies = database.createObjectStore('draftBodies', { keyPath: ['projectId', 'editorSessionId'] });
      bodies.createIndex('byIntegrity', ['projectId', 'editorSessionId', 'generation', 'documentByteSize'], {
        unique: true,
      });
      const writers = database.createObjectStore('draftWriters', {
        keyPath: ['projectId', 'editorSessionId'],
      });
      const draft = createProjectDraft({
        editorSessionId: 'session-live',
        projectId: 'project-live',
        writerToken: 'writer-live',
      });
      drafts.put(toProjectDraftMetadata(draft, 1));
      bodies.put(toProjectDraftBody(draft));
      writers.put({
        editorSessionId: draft.editorSessionId,
        metadataRevision: 1,
        projectId: draft.projectId,
        state: 'active',
        updatedAt: draft.updatedAt,
        writerToken: draft.writerToken,
      });
      for (const claim of claims) {
        writers.put(claim);
      }
      const queueRuns = database.createObjectStore('queueRuns', { keyPath: 'key' });
      queueRuns.createIndex('byProject', 'projectId');
      const recallCache = database.createObjectStore('recallCache', { keyPath: 'queueItemId' });
      recallCache.createIndex('byLastAccessOrder', 'lastAccessOrder');
      database.createObjectStore('recallBodies', { keyPath: 'queueItemId' });
      database.createObjectStore('metadata', { keyPath: 'key' });
      transaction.addEventListener('abort', () => reject(transaction.error));
    });
    request.addEventListener('error', () => reject(request.error));
    request.addEventListener('success', () => {
      request.result.close();
      resolve();
    });
  });

afterEach(async () => {
  accountLifecycle.invalidate();
  for (const store of stores.splice(0)) {
    store.close();
  }
  for (const suffix of suffixes) {
    await deleteWorkbenchDatabase(suffix);
  }
  suffixes.clear();
});

describe('IndexedDB project drafts', () => {
  testProjectDraftStoreContract(createStore);

  it('creates the complete workbench database schema in one version', async () => {
    const suffix = createSuffix();
    const database = await openWorkbenchDatabase(suffix);
    stores.push(database);

    expect(database.name).toBe(getWorkbenchDatabaseName(suffix));
    expect(database.version).toBe(4);
    expect([...database.objectStoreNames]).toEqual([
      'draftBodies',
      'draftWriters',
      'drafts',
      'metadata',
      'queueReceiptAcks',
      'queueRuns',
      'recallBodies',
      'recallCache',
    ]);
    expect([...database.transaction('draftWriters').store.indexNames]).toEqual(['byRetarget']);

    const store = createIndexedDbProjectDraftStore(database);
    stores.push(store);
    await store.stage(createProjectDraftInput());
    const metadata = await database.get('drafts', ['project-1', 'session-a']);
    const body = await database.get('draftBodies', ['project-1', 'session-a']);
    expect(metadata).not.toHaveProperty('documentJson');
    expect(metadata?.metadataRevision).toBe(1);
    expect(body).toEqual({
      documentByteSize: createProjectDraft().documentByteSize,
      documentJson: createProjectDraft().documentJson,
      editorSessionId: 'session-a',
      generation: 1,
      projectId: 'project-1',
      recordType: 'draft-body',
    });
  });

  it('upgrades version-one storage without retaining legacy recall payloads', async () => {
    const suffix = createSuffix();
    const name = getWorkbenchDatabaseName(suffix);
    const legacy = await new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDB.open(name, 1);
      request.addEventListener('upgradeneeded', () => {
        const database = request.result;
        const drafts = database.createObjectStore('drafts', { keyPath: ['projectId', 'editorSessionId'] });
        drafts.createIndex('byProject', 'projectId');
        const bodies = database.createObjectStore('draftBodies', { keyPath: ['projectId', 'editorSessionId'] });
        bodies.createIndex('byIntegrity', ['projectId', 'editorSessionId', 'generation', 'documentByteSize'], {
          unique: true,
        });
        database.createObjectStore('draftWriters', { keyPath: ['projectId', 'editorSessionId'] });
        const runs = database.createObjectStore('queueRuns', { keyPath: 'key' });
        runs.createIndex('byProject', 'projectId');
        const recall = database.createObjectStore('recallCache', { keyPath: 'queueItemId' });
        recall.createIndex('byLastAccessAt', 'lastAccessAt');
      });
      request.addEventListener('error', () => reject(request.error));
      request.addEventListener('success', () => resolve(request.result));
    });
    const write = legacy.transaction('recallCache', 'readwrite');
    write.objectStore('recallCache').put({
      byteSize: 2,
      lastAccessAt: 1,
      payloadJson: '{}',
      projectId: 'project-1',
      queueItemId: 'queue-1',
    });
    await new Promise<void>((resolve, reject) => {
      write.addEventListener('complete', () => resolve());
      write.addEventListener('error', () => reject(write.error));
    });
    legacy.close();

    const upgraded = await openWorkbenchDatabase(suffix);
    stores.push(upgraded);

    expect(upgraded.version).toBe(4);
    expect(await upgraded.count('recallCache')).toBe(0);
    expect([...upgraded.objectStoreNames]).toContain('recallBodies');
    expect([...upgraded.transaction('draftWriters').store.indexNames]).toEqual(['byRetarget']);
  });

  it('adopts at most once across connections and fences the old writer', async () => {
    const suffix = createSuffix();
    const first = createIndexedDbProjectDraftStore(await openWorkbenchDatabase(suffix));
    const second = createIndexedDbProjectDraftStore(await openWorkbenchDatabase(suffix));
    stores.push(first, second);
    await first.stage(createProjectDraftInput());

    const outcomes = await Promise.all([
      first.adopt('project-1', 'session-a', 'session-b', 'writer-b'),
      second.adopt('project-1', 'session-a', 'session-c', 'writer-c'),
    ]);

    expect(outcomes.map((outcome) => outcome.kind).sort()).toEqual(['adopted', 'missing']);
    await expect(first.stage(createProjectDraftInput({ generation: 2 }))).resolves.toEqual({ kind: 'fenced' });
  });

  it('refuses an oversized draft before opening a write transaction', async () => {
    const suffix = createSuffix();
    const database = await openWorkbenchDatabase(suffix);
    stores.push(database);
    const store = createIndexedDbProjectDraftStore(database, { maxDraftBytes: 4 });
    stores.push(store);

    await expect(store.stage(createProjectDraftInput({ documentJson: '12345' }))).resolves.toEqual({
      kind: 'too-large',
    });
    await expect(store.list()).resolves.toEqual({ items: [], kind: 'available', nextCursor: null });
  });

  it('closes every account-owned instance and eventually deletes its database on rotation', async () => {
    const suffix = createSuffix();
    const owner = accountLifecycle.activate('draft-browser-test', suffix);
    const first = await createAccountOwnedProjectDraftStore(owner);
    const second = await createAccountOwnedProjectDraftStore(owner);
    stores.push(first, second);
    await first.stage(createProjectDraftInput());
    const inFlight = first.stage(createProjectDraftInput({ documentJson: 'x'.repeat(1024 * 1024), generation: 2 }));

    accountLifecycle.invalidate();
    await inFlight;

    await expect(first.stage(createProjectDraftInput({ generation: 2 }))).resolves.toEqual({ kind: 'unavailable' });
    await expect(second.stage(createProjectDraftInput({ editorSessionId: 'session-b' }))).resolves.toEqual({
      kind: 'unavailable',
    });
    await vi.waitFor(async () => {
      const databases = await indexedDB.databases();
      expect(databases.some((database) => database.name === getWorkbenchDatabaseName(suffix))).toBe(false);
    });
    const reopened = createIndexedDbProjectDraftStore(await openWorkbenchDatabase(suffix));
    stores.push(reopened);
    await expect(reopened.get('project-1', 'session-a')).resolves.toEqual({ kind: 'missing' });
  });

  it('degrades to an unavailable store when IndexedDB cannot open', async () => {
    const suffix = createSuffix();
    const owner = accountLifecycle.activate('draft-browser-test', suffix);
    const openDatabase = vi.fn<() => Promise<WorkbenchDatabase>>().mockRejectedValue(new DOMException('denied'));
    const deleteDatabase = vi.fn<typeof deleteWorkbenchDatabase>(() => Promise.resolve({ kind: 'deleted' }));

    const store = await createAccountOwnedProjectDraftStore(owner, { deleteDatabase, openDatabase });
    stores.push(store);

    await expect(store.stage(createProjectDraftInput())).resolves.toEqual({ kind: 'unavailable' });
    await expect(store.list()).resolves.toEqual({ kind: 'unavailable' });

    accountLifecycle.invalidate();
    expect(deleteDatabase).toHaveBeenCalledWith(suffix);
  });

  it('rejects the unauthenticated boot scope before opening account storage', async () => {
    const owner = accountLifecycle.invalidate();
    const openDatabase = vi.fn<() => Promise<WorkbenchDatabase>>();

    await expect(createAccountOwnedProjectDraftStore(owner, { openDatabase })).rejects.toThrow(
      'requires an active account'
    );
    expect(openDatabase).not.toHaveBeenCalled();
  });

  it('deletes account storage when rotation races database opening', async () => {
    const suffix = createSuffix();
    const owner = accountLifecycle.activate('draft-browser-test', suffix);
    let resolveOpen: (database: WorkbenchDatabase) => void = () => undefined;
    const opening = new Promise<WorkbenchDatabase>((resolve) => {
      resolveOpen = resolve;
    });
    const database = { close: vi.fn() } as unknown as WorkbenchDatabase;
    const deleteDatabase = vi.fn<typeof deleteWorkbenchDatabase>(() => Promise.resolve({ kind: 'deleted' }));
    const pending = createAccountOwnedProjectDraftStore(owner, { deleteDatabase, openDatabase: () => opening });

    accountLifecycle.invalidate();
    resolveOpen(database);

    await expect(pending).rejects.toMatchObject({ name: 'AccountScopeExpiredError' });
    expect(database.close).toHaveBeenCalledOnce();
    expect(deleteDatabase).toHaveBeenCalledWith(suffix);
  });

  it('surfaces corrupt rows without loading every document into the result page', async () => {
    const suffix = createSuffix();
    const database = await openWorkbenchDatabase(suffix);
    stores.push(database);
    const malformed = toProjectDraftMetadata(createProjectDraft()) as unknown as Record<string, unknown>;
    delete malformed.baseRevision;
    await database.put('drafts', malformed as unknown as ProjectDraftMetadata);

    const store = createIndexedDbProjectDraftStore(database);
    stores.push(store);
    await expect(store.list()).resolves.toEqual({
      items: [
        {
          documentByteSize: null,
          documentSchemaVersion: null,
          editorSessionId: 'session-a',
          generation: null,
          projectId: 'project-1',
          state: 'corrupt',
          updatedAt: null,
        },
      ],
      kind: 'available',
      nextCursor: null,
    });
    await expect(store.stage(createProjectDraftInput({ generation: 2 }))).resolves.toEqual({ kind: 'corrupt' });
    await expect(store.deleteCorrupt('project-1', 'session-a')).resolves.toEqual({ kind: 'deleted' });
    await expect(store.get('project-1', 'session-a')).resolves.toMatchObject({
      kind: 'empty',
      writerState: 'fenced',
    });
  });

  it('keeps a stale writer fenced after corrupt cleanup', async () => {
    const suffix = createSuffix();
    const firstDatabase = await openWorkbenchDatabase(suffix);
    const secondDatabase = await openWorkbenchDatabase(suffix);
    const firstStore = createIndexedDbProjectDraftStore(firstDatabase);
    const secondStore = createIndexedDbProjectDraftStore(secondDatabase);
    stores.push(firstStore, secondStore);
    await firstStore.stage(createProjectDraftInput());
    const body = await firstDatabase.get('draftBodies', ['project-1', 'session-a']);
    if (!body) {
      throw new Error('Expected a draft body.');
    }
    await firstDatabase.put('draftBodies', { ...body, documentByteSize: body.documentByteSize + 1 });

    await expect(secondStore.deleteCorrupt('project-1', 'session-a')).resolves.toEqual({ kind: 'deleted' });
    await expect(firstStore.stage(createProjectDraftInput({ generation: 2 }))).resolves.toEqual({ kind: 'fenced' });
    await expect(secondStore.get('project-1', 'session-a')).resolves.toEqual({
      kind: 'empty',
      writerState: 'fenced',
      writerToken: 'writer-a',
    });
    await expect(secondStore.startFreshWriter('project-1', 'session-a', 'writer-a', 'writer-b')).resolves.toEqual({
      kind: 'started',
    });
    await expect(
      secondStore.stage(createProjectDraftInput({ generation: 2, writerToken: 'writer-b' }))
    ).resolves.toEqual({ kind: 'stored' });
  });

  it('does not overwrite an orphaned target body during adoption or retargeting', async () => {
    const suffix = createSuffix();
    const database = await openWorkbenchDatabase(suffix);
    stores.push(database);
    const store = createIndexedDbProjectDraftStore(database);
    stores.push(store);
    await store.stage(createProjectDraftInput());
    const orphan = toProjectDraftBody(
      createProjectDraft({ editorSessionId: 'session-b', projectId: 'project-1', writerToken: 'writer-b' })
    );
    await database.put('draftBodies', orphan);

    await expect(store.adopt('project-1', 'session-a', 'session-b', 'writer-b')).resolves.toEqual({ kind: 'corrupt' });
    await expect(database.get('draftBodies', ['project-1', 'session-b'])).resolves.toEqual(orphan);

    await database.delete('draftBodies', ['project-1', 'session-b']);
    await store.reserveCopyIdentity('project-1', 'session-a', 'writer-a', createCopyReservation('copy-1'));
    const copyOrphan = { ...orphan, editorSessionId: 'session-a', projectId: 'copy-1' };
    await database.put('draftBodies', copyOrphan);
    await expect(
      store.retargetAcknowledgedCopy({
        acknowledgedRevision: 1,
        copyProjectId: 'copy-1',
        editorSessionId: 'session-a',
        projectId: 'project-1',
        retargetDocument: () => '{"id":"copy-1"}',
        sentGeneration: 1,
        writerToken: 'writer-a',
      })
    ).resolves.toEqual({ kind: 'corrupt' });
    await expect(database.get('draftBodies', ['copy-1', 'session-a'])).resolves.toEqual(copyOrphan);
  });

  it('refuses ownership and reservation changes when a split body is corrupt', async () => {
    const suffix = createSuffix();
    const database = await openWorkbenchDatabase(suffix);
    stores.push(database);
    const store = createIndexedDbProjectDraftStore(database);
    stores.push(store);
    await store.stage(createProjectDraftInput());
    const body = await database.get('draftBodies', ['project-1', 'session-a']);
    if (!body) {
      throw new Error('Expected a staged draft body.');
    }
    await database.put('draftBodies', { ...body, documentByteSize: body.documentByteSize + 1 });

    await expect(store.claimWriter('project-1', 'session-a', 'writer-a', 'writer-a2')).resolves.toEqual({
      kind: 'corrupt',
    });
    await expect(
      store.reserveCopyIdentity('project-1', 'session-a', 'writer-a', createCopyReservation('copy-1'))
    ).resolves.toEqual({
      kind: 'corrupt',
    });
    await expect(store.stage(createProjectDraftInput({ generation: 2 }))).resolves.toEqual({ kind: 'corrupt' });
    await expect(database.get('draftBodies', ['project-1', 'session-a'])).resolves.toEqual({
      ...body,
      documentByteSize: body.documentByteSize + 1,
    });
  });

  it('rejects a copy reservation whose durable body no longer matches its byte binding', async () => {
    const suffix = createSuffix();
    const database = await openWorkbenchDatabase(suffix);
    stores.push(database);
    const store = createIndexedDbProjectDraftStore(database);
    stores.push(store);
    await store.stage(createProjectDraftInput());
    await store.reserveCopyIdentity('project-1', 'session-a', 'writer-a', createCopyReservation('copy-1'));
    const body = await database.get('draftBodies', ['project-1', 'session-a']);
    if (!body) {
      throw new Error('Expected a staged draft body.');
    }
    await database.put('draftBodies', { ...body, copyDocumentJson: '{"id":"tampered"}' });

    await expect(store.get('project-1', 'session-a')).resolves.toEqual({ kind: 'corrupt' });
    await expect(
      store.reserveCopyIdentity('project-1', 'session-a', 'writer-a', createCopyReservation('copy-2'))
    ).resolves.toEqual({ kind: 'corrupt' });
  });

  it('does not delete a mismatched body while settling an acknowledgement', async () => {
    const suffix = createSuffix();
    const database = await openWorkbenchDatabase(suffix);
    stores.push(database);
    const store = createIndexedDbProjectDraftStore(database);
    stores.push(store);
    await store.stage(createProjectDraftInput());
    const body = await database.get('draftBodies', ['project-1', 'session-a']);
    if (!body) {
      throw new Error('Expected a staged draft body.');
    }
    const newerBody = { ...body, generation: 2 };
    await database.put('draftBodies', newerBody);

    await expect(store.settleAcknowledgement('project-1', 'session-a', 'writer-a', 1, 7)).resolves.toEqual({
      kind: 'corrupt',
    });
    await expect(database.get('draftBodies', ['project-1', 'session-a'])).resolves.toEqual(newerBody);
  });

  it('does not read the previous body when staging a normal newer generation', async () => {
    const suffix = createSuffix();
    const database = await openWorkbenchDatabase(suffix);
    stores.push(database);
    const store = createIndexedDbProjectDraftStore(database);
    stores.push(store);
    await store.stage(createProjectDraftInput());
    const get = vi.spyOn(IDBObjectStore.prototype, 'get');

    await expect(store.stage(createProjectDraftInput({ generation: 2 }))).resolves.toEqual({ kind: 'stored' });

    expect(
      get.mock.instances.filter(
        (objectStore): objectStore is IDBObjectStore =>
          objectStore instanceof IDBObjectStore && objectStore.name === 'draftBodies'
      )
    ).toHaveLength(0);
    get.mockRestore();
  });

  it('retries a retarget when a newer generation wins the transaction race', async () => {
    const suffix = createSuffix();
    const first = createIndexedDbProjectDraftStore(await openWorkbenchDatabase(suffix));
    stores.push(first);
    await first.stage(createProjectDraftInput());
    await first.reserveCopyIdentity('project-1', 'session-a', 'writer-a', createCopyReservation('copy-1'));
    await first.stage(createProjectDraftInput({ generation: 2 }));
    let concurrentStage: Promise<unknown> | null = null;

    const outcome = await first.retargetAcknowledgedCopy({
      acknowledgedRevision: 1,
      copyProjectId: 'copy-1',
      editorSessionId: 'session-a',
      projectId: 'project-1',
      retargetDocument: (documentJson) => {
        concurrentStage ??= first.stage(
          createProjectDraftInput({ documentJson: '{"id":"project-1","newer":true}', generation: 3 })
        );
        return documentJson.replace('project-1', 'copy-1');
      },
      sentGeneration: 1,
      writerToken: 'writer-a',
    });

    await concurrentStage;
    expect(outcome).toMatchObject({ draft: { generation: 3, projectId: 'copy-1' }, kind: 'retargeted' });
  });

  it('does not let adoption erase a same-generation copy reservation', async () => {
    const suffix = createSuffix();
    const database = await openWorkbenchDatabase(suffix);
    stores.push(database);
    const source = createIndexedDbProjectDraftStore(database);
    stores.push(source);
    await source.stage(createProjectDraftInput());
    let reservation: Promise<unknown> = Promise.resolve();
    const intercepted = interceptFirstWrite(database, () => {
      reservation = reserveCopyInRawTransaction(database);
    });
    const adopter = createIndexedDbProjectDraftStore(intercepted);
    stores.push(adopter);

    await expect(adopter.adopt('project-1', 'session-a', 'session-b', 'writer-b')).resolves.toEqual({
      kind: 'occupied',
    });
    await reservation;
    await expect(source.get('project-1', 'session-a')).resolves.toMatchObject({
      draft: { copyProjectId: 'copy-1' },
      kind: 'found',
    });
  });

  it('does not let adoption move a stale snapshot across acknowledgement and recreation', async () => {
    const suffix = createSuffix();
    const database = await openWorkbenchDatabase(suffix);
    stores.push(database);
    const source = createIndexedDbProjectDraftStore(database);
    stores.push(source);
    await source.stage(createProjectDraftInput());
    let recreation: Promise<unknown> = Promise.resolve();
    const intercepted = interceptFirstWrite(database, () => {
      const transaction = database.transaction(['drafts', 'draftBodies', 'draftWriters'], 'readwrite');
      recreation = (async () => {
        const key: [string, string] = ['project-1', 'session-a'];
        const claim = await transaction.objectStore('draftWriters').get(key);
        if (!claim) {
          throw new Error('Expected writer claim.');
        }
        const recreated = createProjectDraft({ documentJson: '{"recreated":true}', generation: 2 });
        const metadataRevision = claim.metadataRevision + 2;
        await transaction.objectStore('drafts').put(toProjectDraftMetadata(recreated, metadataRevision));
        await transaction.objectStore('draftBodies').put(toProjectDraftBody(recreated));
        await transaction.objectStore('draftWriters').put({ ...claim, metadataRevision });
        await transaction.done;
      })();
    });
    const adopter = createIndexedDbProjectDraftStore(intercepted);
    stores.push(adopter);

    await expect(adopter.adopt('project-1', 'session-a', 'session-b', 'writer-b')).resolves.toEqual({
      kind: 'occupied',
    });
    await recreation;
    await expect(source.get('project-1', 'session-a')).resolves.toMatchObject({
      draft: { documentJson: '{"recreated":true}', generation: 2 },
      kind: 'found',
    });
  });

  it('keeps the metadata watermark monotonic across acknowledgement and recreation', async () => {
    const suffix = createSuffix();
    const database = await openWorkbenchDatabase(suffix);
    stores.push(database);
    const store = createIndexedDbProjectDraftStore(database);
    stores.push(store);
    await store.stage(createProjectDraftInput());
    await store.settleAcknowledgement('project-1', 'session-a', 'writer-a', 1, 7);
    await store.stage(createProjectDraftInput({ baseRevision: 7, generation: 2 }));

    const metadata = await database.get('drafts', ['project-1', 'session-a']);
    const claim = await database.get('draftWriters', ['project-1', 'session-a']);
    expect(metadata?.metadataRevision).toBe(3);
    expect(claim?.metadataRevision).toBe(3);
  });

  it('reports a re-key quota failure without disabling the store or losing the source', async () => {
    const suffix = createSuffix();
    const database = await openWorkbenchDatabase(suffix);
    stores.push(database);
    const source = createIndexedDbProjectDraftStore(database);
    stores.push(source);
    await source.stage(createProjectDraftInput());
    const failing = createIndexedDbProjectDraftStore(abortFirstBodyWriteWithQuota(database));
    stores.push(failing);

    await expect(failing.adopt('project-1', 'session-a', 'session-b', 'writer-b')).resolves.toEqual({ kind: 'quota' });
    expect(failing.availability).toBe('available');
    await expect(source.get('project-1', 'session-a')).resolves.toMatchObject({ kind: 'found' });
  });

  it('retries settlement without erasing a same-generation copy reservation', async () => {
    const suffix = createSuffix();
    const database = await openWorkbenchDatabase(suffix);
    stores.push(database);
    const source = createIndexedDbProjectDraftStore(database);
    stores.push(source);
    await source.stage(createProjectDraftInput());
    let reservation: Promise<unknown> = Promise.resolve();
    const intercepted = interceptFirstWrite(database, () => {
      reservation = reserveCopyInRawTransaction(database);
    });
    const settling = createIndexedDbProjectDraftStore(intercepted);
    stores.push(settling);

    await expect(
      settling.settleConflict('project-1', 'session-a', 'writer-a', 1, {
        kind: 'revision',
        serverRevision: 7,
      })
    ).resolves.toMatchObject({ draft: { copyProjectId: 'copy-1', state: 'conflict' }, kind: 'marked' });
    await reservation;
  });

  it('reports a blocked database deletion without waiting indefinitely', async () => {
    const suffix = createSuffix();
    const opened = await openWorkbenchDatabase(suffix);
    opened.close();
    const foreign = await new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDB.open(getWorkbenchDatabaseName(suffix));
      request.addEventListener('error', () => reject(request.error));
      request.addEventListener('success', () => resolve(request.result));
    });
    foreign.onversionchange = () => undefined;

    const deletion = await deleteWorkbenchDatabase(suffix);
    expect(deletion.kind).toBe('blocked');
    const startedAt = performance.now();
    await expect(openWorkbenchDatabase(suffix, { timeoutMs: 25 })).rejects.toMatchObject({ name: 'TimeoutError' });
    expect(performance.now() - startedAt).toBeLessThan(500);
    foreign.close();
    if (deletion.kind === 'blocked') {
      await expect(deletion.completion).resolves.toEqual({ kind: 'deleted' });
    }
    const reopened = await openWorkbenchDatabase(suffix);
    stores.push(reopened);
    expect([...reopened.objectStoreNames]).toContain('drafts');
  });

  it('marks a store unavailable when another connection upgrades the database', async () => {
    const suffix = createSuffix();
    const store = createIndexedDbProjectDraftStore(await openWorkbenchDatabase(suffix));
    stores.push(store);

    const upgraded = await new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDB.open(getWorkbenchDatabaseName(suffix), WORKBENCH_DATABASE_VERSION + 1);
      request.addEventListener('error', () => reject(request.error));
      request.addEventListener('success', () => resolve(request.result));
    });
    stores.push(upgraded);

    expect(store.availability).toBe('unavailable');
    await expect(store.list()).resolves.toEqual({ kind: 'unavailable' });
  });

  it('upgrades populated v2 writer claims and indexes only retarget handoffs', async () => {
    const suffix = createSuffix();
    const claim = (projectId: string, targetProjectId?: string) => ({
      editorSessionId: 'session-a',
      fenceReason: 'moved',
      metadataRevision: 1,
      projectId,
      ...(targetProjectId
        ? {
            adoptedByEditorSessionId: 'session-a',
            retargetedToProjectId: targetProjectId,
            retargetedToRevision: 1,
          }
        : { adoptedByEditorSessionId: 'session-b' }),
      state: 'fenced',
      updatedAt: 1,
      writerToken: 'writer-a',
    });
    const retargetClaims = Array.from({ length: 205 }, (_, index) => {
      const suffix = index.toString().padStart(3, '0');
      return claim(`project-${suffix}`, `copy-${suffix}`);
    });
    await createVersion2Database(suffix, [...retargetClaims, claim('project-no-retarget')]);
    const database = await openWorkbenchDatabase(suffix);
    stores.push(database);
    const store = createIndexedDbProjectDraftStore(database);
    stores.push(store);

    await expect(store.get('project-live', 'session-live')).resolves.toMatchObject({
      draft: { projectId: 'project-live', writerToken: 'writer-live' },
      kind: 'found',
    });
    const first = await store.listRetargets({ limit: 100 });
    expect(first).toMatchObject({
      kind: 'available',
      nextCursor: ['project-099', 'session-a', 'copy-099'],
    });
    expect(first.kind === 'available' ? first.items : []).toHaveLength(100);
    if (first.kind !== 'available' || !first.nextCursor) {
      throw new Error('Expected a second handoff page.');
    }
    const second = await store.listRetargets({ after: first.nextCursor, limit: 100 });
    expect(second).toMatchObject({ kind: 'available', nextCursor: ['project-199', 'session-a', 'copy-199'] });
    if (second.kind !== 'available' || !second.nextCursor) {
      throw new Error('Expected a third handoff page.');
    }
    expect(second.items).toHaveLength(100);
    await expect(store.listRetargets({ after: second.nextCursor, limit: 100 })).resolves.toMatchObject({
      items: [
        { projectId: 'project-200' },
        { projectId: 'project-201' },
        { projectId: 'project-202' },
        { projectId: 'project-203' },
        { projectId: 'project-204' },
      ],
      kind: 'available',
      nextCursor: null,
    });
  });
});

import { accountLifecycle } from '@platform/state/accountLifecycle';
import {
  deleteWorkbenchDatabase,
  getWorkbenchDatabaseName,
  openWorkbenchDatabase,
  type WorkbenchDatabase,
} from '@workbench/projects/workbenchDatabase';
import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  createAccountOwnedQueueRecallCache,
  createIndexedDbQueueRecallCache,
  type QueueRecallSnapshotInput,
} from './queueRecallCache';
import { createIndexedDbQueueRunJournal } from './queueRunJournal';

const databases: WorkbenchDatabase[] = [];
const suffixes = new Set<string>();

const createCache = async (options?: { maxBytes?: number; maxEntries?: number }) => {
  const suffix = `:queue-recall-test:${crypto.randomUUID()}`;
  suffixes.add(suffix);
  const database = await openWorkbenchDatabase(suffix);
  databases.push(database);
  return { cache: createIndexedDbQueueRecallCache(database, options), database };
};

const createSnapshot = (queueItemId: string, projectId = 'project-1'): QueueRecallSnapshotInput => ({
  generateValues: { marker: queueItemId } as never,
  projectId,
  queueItemId,
  sourceId: 'generate',
  submittedAt: '2026-09-03T00:00:00.000Z',
});

const abortFirstRecallWriteWithQuota = (database: WorkbenchDatabase): WorkbenchDatabase => {
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
              const wrapStore = (store: ReturnType<typeof transactionTarget.objectStore>) =>
                new Proxy(store, {
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
              if (transactionProperty === 'store') {
                return wrapStore(transactionTarget.store);
              }
              if (transactionProperty === 'objectStore') {
                return (storeName: string) => {
                  const store = transactionTarget.objectStore(storeName);
                  if (storeName !== 'recallBodies') {
                    return store;
                  }
                  return wrapStore(store);
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

afterEach(async () => {
  for (const database of databases.splice(0)) {
    database.close();
  }
  for (const suffix of suffixes) {
    await deleteWorkbenchDatabase(suffix);
  }
  suffixes.clear();
});

describe('IndexedDB queue recall cache', () => {
  it('round-trips snapshots and updates the LRU on read', async () => {
    const { cache } = await createCache({ maxEntries: 2 });
    await cache.put(createSnapshot('queue-1'));
    await cache.put(createSnapshot('queue-2'));
    await expect(cache.get('queue-1')).resolves.toMatchObject({
      kind: 'found',
      snapshot: { queueItemId: 'queue-1' },
    });
    await cache.put(createSnapshot('queue-3'));

    await expect(cache.get('queue-2')).resolves.toEqual({ kind: 'missing' });
    await expect(cache.get('queue-1')).resolves.toMatchObject({ kind: 'found' });
    await expect(cache.get('queue-3')).resolves.toMatchObject({ kind: 'found' });
  });

  it('enforces the byte budget atomically without evicting for a rejected entry', async () => {
    const baseline = createSnapshot('queue-1');
    const baselineBytes = new TextEncoder().encode(JSON.stringify(baseline)).byteLength;
    const { cache } = await createCache({ maxBytes: baselineBytes + 8 });
    await expect(cache.put(baseline)).resolves.toEqual({ kind: 'stored' });
    await expect(
      cache.put({ ...createSnapshot('queue-2'), generateValues: { marker: 'x'.repeat(128) } as never })
    ).resolves.toEqual({ kind: 'too-large' });
    await expect(cache.get('queue-1')).resolves.toMatchObject({ kind: 'found' });
  });

  it('coalesces entries by queue item id', async () => {
    const { cache } = await createCache({ maxEntries: 1 });
    await cache.put(createSnapshot('queue-1', 'project-1'));
    await cache.put(createSnapshot('queue-1', 'project-2'));

    await expect(cache.get('queue-1')).resolves.toMatchObject({
      kind: 'found',
      snapshot: { projectId: 'project-2', queueItemId: 'queue-1' },
    });
  });

  it('removes corrupt entries without poisoning the cache', async () => {
    const { cache, database } = await createCache();
    await database.put('recallCache', {
      byteSize: 1,
      lastAccessOrder: 1,
      projectId: 'project-1',
      queueItemId: 'queue-1',
    });
    await database.put('recallBodies', { payloadJson: '{broken', queueItemId: 'queue-1' });

    await expect(cache.get('queue-1')).resolves.toEqual({ kind: 'corrupt' });
    await expect(cache.get('queue-1')).resolves.toEqual({ kind: 'missing' });
    await expect(cache.put(createSnapshot('queue-1'))).resolves.toEqual({ kind: 'stored' });
  });

  it('keeps payload bodies unchanged when reads update recency', async () => {
    const { cache, database } = await createCache();
    await cache.put(createSnapshot('queue-1'));
    const before = await database.get('recallBodies', 'queue-1');

    await cache.get('queue-1');

    expect(await database.get('recallBodies', 'queue-1')).toEqual(before);
  });

  it('does not parse unrelated payload bodies while enforcing LRU budgets', async () => {
    const { cache, database } = await createCache({ maxEntries: 2 });
    await cache.put(createSnapshot('queue-1'));
    await database.put('recallBodies', { payloadJson: '{unrelated-corruption', queueItemId: 'queue-1' });

    await expect(cache.put(createSnapshot('queue-2'))).resolves.toEqual({ kind: 'stored' });
    await expect(cache.get('queue-2')).resolves.toMatchObject({ kind: 'found' });
  });

  it('rejects malformed snapshots', async () => {
    const { cache } = await createCache();
    await expect(cache.put({ ...createSnapshot('queue-1'), projectId: '' })).resolves.toEqual({ kind: 'invalid' });
    await expect(cache.get('')).resolves.toEqual({ kind: 'missing' });
  });

  it('reports quota without poisoning later writes', async () => {
    const { database } = await createCache();
    const cache = createIndexedDbQueueRecallCache(abortFirstRecallWriteWithQuota(database));

    await expect(cache.put(createSnapshot('queue-1'))).resolves.toEqual({ kind: 'quota' });
    expect(cache.availability).toBe('available');
    await expect(cache.put(createSnapshot('queue-1'))).resolves.toEqual({ kind: 'stored' });
  });

  it('does not close a shared borrowed database when a leaf adapter closes', async () => {
    const { database } = await createCache();
    const cache = createIndexedDbQueueRecallCache(database);
    const journal = createIndexedDbQueueRunJournal(database);

    journal.close();

    await expect(cache.put(createSnapshot('queue-1'))).resolves.toEqual({ kind: 'stored' });
  });

  it('isolates a leaf adapter failure from other adapters sharing the connection', async () => {
    const { database } = await createCache();
    const failingDatabase = new Proxy(database, {
      get(target, property) {
        if (property === 'transaction') {
          return (...args: unknown[]) => {
            const transaction = Reflect.apply(target.transaction, target, args);
            return new Proxy(transaction, {
              get(transactionTarget, transactionProperty) {
                if (transactionProperty === 'objectStore') {
                  return (storeName: string) => {
                    const store = transactionTarget.objectStore(storeName);
                    if (storeName !== 'queueRuns') {
                      return store;
                    }
                    return new Proxy(store, {
                      get(storeTarget, storeProperty) {
                        if (storeProperty === 'put') {
                          return () => {
                            transactionTarget.abort();
                            throw new DOMException('failed', 'InvalidStateError');
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
    const journal = createIndexedDbQueueRunJournal(failingDatabase);
    const cache = createIndexedDbQueueRecallCache(database);

    await expect(
      journal.record('project-1', {
        cancellable: true,
        id: 'queue-1',
        snapshot: {
          backendSubmission: { batchCount: 1, graph: { edges: [], id: 'graph', nodes: {} }, kind: 'workflow' },
          canvas: {
            document: { bbox: { height: 512, width: 512, x: 0, y: 0 }, height: 512, width: 512 },
            documentRevision: 1,
          },
          destination: 'gallery',
          filterIntermediateResults: true,
          galleryBoardId: null,
          graph: { id: 'graph', label: 'Graph' },
          presentation: { batchCount: 1, height: 512, width: 512 },
          sourceId: 'workflow',
          submittedAt: '2026-09-03T00:00:00.000Z',
        },
        status: 'pending',
      })
    ).resolves.toEqual({ kind: 'unavailable' });

    await expect(cache.put(createSnapshot('queue-1'))).resolves.toEqual({ kind: 'stored' });
  });

  it('closes account-owned queue connections and deletes their database on rotation', async () => {
    const suffix = `:queue-account-test:${crypto.randomUUID()}`;
    suffixes.add(suffix);
    const owner = accountLifecycle.activate('queue-browser-test', suffix);
    const cache = await createAccountOwnedQueueRecallCache(owner);
    await cache.put(createSnapshot('queue-1'));

    accountLifecycle.invalidate();

    await expect(cache.get('queue-1')).resolves.toEqual({ kind: 'unavailable' });
    await vi.waitFor(async () => {
      const openDatabases = await indexedDB.databases();
      expect(openDatabases.some((database) => database.name === getWorkbenchDatabaseName(suffix))).toBe(false);
    });
  });
});

import type { WorkbenchQueueItem } from '@workbench/queueHistoryContracts';

import {
  deleteWorkbenchDatabase,
  openWorkbenchDatabase,
  type WorkbenchDatabase,
} from '@workbench/projects/workbenchDatabase';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { afterEach, describe, expect, it } from 'vitest';

import { createIndexedDbQueueRunJournal } from './queueRunJournal';

const databases: WorkbenchDatabase[] = [];
const suffixes = new Set<string>();

const createJournal = async (options?: Parameters<typeof createIndexedDbQueueRunJournal>[1]) => {
  const suffix = `:queue-journal-test:${crypto.randomUUID()}`;
  suffixes.add(suffix);
  const database = await openWorkbenchDatabase(suffix);
  databases.push(database);
  return { database, journal: createIndexedDbQueueRunJournal(database, options) };
};

const createItem = (overrides: Partial<WorkbenchQueueItem> = {}): WorkbenchQueueItem =>
  ({
    cancellable: true,
    id: 'queue-item-1',
    snapshot: {
      backendSubmission: {
        batchCount: 1,
        graph: { edges: [], id: 'backend-graph-1', nodes: {} },
        kind: 'workflow',
      },
      canvas: {
        document: { bbox: { height: 512, width: 512, x: 0, y: 0 }, height: 512, width: 512 },
        documentRevision: 1,
      },
      destination: 'gallery',
      filterIntermediateResults: true,
      galleryBoardId: null,
      graph: { id: 'graph-1', label: 'Graph' },
      presentation: { batchCount: 1, height: 512, width: 512 },
      sourceId: 'workflow',
      submittedAt: '2026-09-03T00:00:00.000Z',
    },
    status: 'pending',
    ...overrides,
  }) as WorkbenchQueueItem;

const rejectFirstPutWithQuota = (database: WorkbenchDatabase, failingStore = 'queueRuns'): WorkbenchDatabase => {
  let isArmed = true;
  return new Proxy(database, {
    get(target, property) {
      if (property === 'transaction') {
        return (...args: unknown[]) => {
          const transaction = Reflect.apply(target.transaction, target, args);
          return new Proxy(transaction, {
            get(transactionTarget, transactionProperty) {
              if (transactionProperty === 'objectStore') {
                return (storeName: string) => {
                  const store = transactionTarget.objectStore(storeName);
                  if (storeName !== failingStore) {
                    return store;
                  }
                  return new Proxy(store, {
                    get(storeTarget, storeProperty) {
                      if (storeProperty === 'put' && isArmed) {
                        return () => {
                          isArmed = false;
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

afterEach(async () => {
  for (const database of databases.splice(0)) {
    database.close();
  }
  for (const suffix of suffixes) {
    await deleteWorkbenchDatabase(suffix);
  }
  suffixes.clear();
});

describe('IndexedDB queue run journal', () => {
  it('queues receipt cleanup when explicitly discarding a project journal', async () => {
    const { journal } = await createJournal();
    await journal.record('project-1', createItem());
    await journal.record('project-2', createItem({ id: 'other-run' }));
    expect(await journal.deleteForProject('project-1')).toEqual({ kind: 'removed' });
    expect(await journal.listPendingReceipts()).toMatchObject({
      kind: 'available',
      entries: [{ projectId: 'project-1', queueItemId: 'queue-item-1' }],
    });
    expect(await journal.listProjectIds()).toEqual({ kind: 'available', projectIds: ['project-2'] });
  });

  it('skips malformed receipt cleanup keys without blocking later acknowledgements', async () => {
    const { database, journal } = await createJournal();
    await database.put('queueReceiptAcks', { key: '0-corrupt', projectId: '', queueItemId: 'bad' });
    await journal.record('project-1', createItem({ backendItemIds: [11], status: 'running' }));
    expect(await journal.listPendingReceipts()).toEqual({
      kind: 'available',
      entries: [
        {
          key: JSON.stringify(['project-1', 'queue-item-1']),
          projectId: 'project-1',
          queueItemId: 'queue-item-1',
        },
      ],
    });
  });

  it('lists distinct projects without decoding their run payloads', async () => {
    const { database, journal } = await createJournal();
    await journal.record('project-1', createItem());
    await journal.record('project-1', createItem({ id: 'run-2' }));
    await journal.record('project-2', createItem());
    const records = await database.getAll('queueRuns');
    for (const record of records) {
      await database.put('queueRuns', { ...record, itemJson: 'invalid-json' });
    }
    expect(await journal.listProjectIds()).toEqual({
      kind: 'available',
      projectIds: ['project-1', 'project-2'],
    });
    journal.close();
    expect(await journal.listProjectIds()).toEqual({ kind: 'unavailable' });
  });

  it('retains the run when receipt cleanup cannot be stored and permits retry after quota recovers', async () => {
    const { database, journal } = await createJournal();
    await journal.record('project-1', createItem());
    const recoveringJournal = createIndexedDbQueueRunJournal(rejectFirstPutWithQuota(database, 'queueReceiptAcks'));
    expect(await recoveringJournal.settle('project-1', 'queue-item-1')).toEqual({ kind: 'unavailable' });
    expect((await journal.listAll()).kind).toBe('available');
    expect(await journal.listAll()).toMatchObject({ entries: [{ queueItemId: 'queue-item-1' }] });
    expect(await recoveringJournal.settle('project-1', 'queue-item-1')).toEqual({ kind: 'removed' });
    expect(await journal.listAll()).toMatchObject({ entries: [] });
    expect(await journal.listPendingReceipts()).toMatchObject({ entries: [{ queueItemId: 'queue-item-1' }] });
  });

  it('retains receipt cleanup after settlement and retries it across adapter reloads', async () => {
    const { database, journal } = await createJournal();
    await journal.record('project-1', createItem({ backendItemIds: [11], status: 'running' }));

    await expect(journal.listPendingReceipts()).resolves.toMatchObject({
      kind: 'available',
      entries: [{ projectId: 'project-1', queueItemId: 'queue-item-1' }],
    });
    await journal.settle('project-1', 'queue-item-1');
    const reloaded = createIndexedDbQueueRunJournal(database);
    await expect(reloaded.listPendingReceipts()).resolves.toMatchObject({
      kind: 'available',
      entries: [{ projectId: 'project-1', queueItemId: 'queue-item-1' }],
    });
    await reloaded.acknowledgeReceipt('project-1', 'queue-item-1');
    await expect(reloaded.listPendingReceipts()).resolves.toEqual({ kind: 'available', entries: [] });
  });

  it('does not repeatedly enqueue receipt cleanup for already acknowledged active runs', async () => {
    const { journal } = await createJournal();
    const item = createItem({ backendItemIds: [11, 12], status: 'running' });
    await journal.record('project-1', item);
    await journal.acknowledgeReceipt('project-1', item.id);
    await journal.record('project-1', { ...item, completedBackendItemIds: [11] });
    await journal.settle('project-1', item.id);

    await expect(journal.listPendingReceipts()).resolves.toEqual({ kind: 'available', entries: [] });
  });
  it('round-trips active runs and coalesces status updates', async () => {
    const { journal } = await createJournal();
    await expect(journal.record('project-1', createItem())).resolves.toEqual({ kind: 'stored' });
    await expect(
      journal.record(
        'project-1',
        createItem({ backendBatchId: 'batch-1', backendItemIds: [11, 12], status: 'running' })
      )
    ).resolves.toEqual({ kind: 'stored' });

    await expect(journal.listForProject('project-1')).resolves.toMatchObject({
      entries: [
        {
          item: { backendBatchId: 'batch-1', backendItemIds: [11, 12], id: 'queue-item-1', status: 'running' },
          projectId: 'project-1',
          queueItemId: 'queue-item-1',
        },
      ],
      kind: 'available',
      removedCorrupt: 0,
    });
  });

  it('round-trips a durable cancellation intent but rejects settled cancellation history', async () => {
    const { journal } = await createJournal();

    await expect(
      journal.record('project-1', createItem({ backendItemIds: [11], cancellationPending: true, status: 'cancelled' }))
    ).resolves.toEqual({ kind: 'stored' });
    await expect(journal.listForProject('project-1')).resolves.toMatchObject({
      entries: [{ item: { backendItemIds: [11], cancellationPending: true, status: 'cancelled' } }],
      kind: 'available',
    });
    await expect(
      journal.record('project-1', createItem({ backendItemIds: [11], status: 'cancelled' }))
    ).resolves.toEqual({ kind: 'invalid' });
  });

  it('restores a round-tripped run through the workbench trust boundary', async () => {
    const { journal } = await createJournal();
    const store = createWorkbenchStore();
    const projectId = store.getState().activeProjectId;
    const item = createItem({ backendBatchId: 'batch-1', backendItemIds: [11], status: 'running' });
    await journal.record(projectId, item);

    const loaded = await journal.listForProject(projectId);
    expect(loaded.kind).toBe('available');
    store.commands.queue.restoreFromJournal({
      items: loaded.kind === 'available' ? loaded.entries.map((entry) => entry.item) : [],
      projectId,
    });

    expect(store.getState().projects[0]?.queue.items).toEqual([
      expect.objectContaining({
        backendBatchId: 'batch-1',
        backendItemIds: [11],
        id: item.id,
        snapshot: expect.objectContaining({
          canvas: item.snapshot.canvas,
          graph: { id: 'graph-1', label: 'Graph' },
        }),
        status: 'running',
      }),
    ]);
  });

  it('restores newest-first submission order without moving a run when its status changes', async () => {
    const { journal } = await createJournal();
    await journal.record('project-1', createItem({ id: 'queue-item-1' }));
    await journal.record('project-1', createItem({ id: 'queue-item-2' }));
    await journal.record('project-1', createItem({ id: 'queue-item-1', status: 'running' }));

    const loaded = await journal.listForProject('project-1');

    expect(loaded.kind === 'available' ? loaded.entries.map((entry) => entry.queueItemId) : []).toEqual([
      'queue-item-2',
      'queue-item-1',
    ]);
  });

  it('keeps project and item identities collision-free', async () => {
    const { journal } = await createJournal();
    await journal.record('a:b', createItem({ id: 'c' }));
    await journal.record('a', createItem({ id: 'b:c' }));

    await expect(journal.listAll()).resolves.toMatchObject({
      entries: [
        { item: { id: 'b:c' }, projectId: 'a' },
        { item: { id: 'c' }, projectId: 'a:b' },
      ],
      kind: 'available',
    });
  });

  it('settles idempotently and never stores terminal items', async () => {
    const { journal } = await createJournal();
    await expect(journal.record('project-1', createItem({ status: 'completed' }))).resolves.toEqual({
      kind: 'invalid',
    });
    await journal.record('project-1', createItem());
    await expect(journal.settle('project-1', 'queue-item-1')).resolves.toEqual({ kind: 'removed' });
    await expect(journal.settle('project-1', 'queue-item-1')).resolves.toEqual({ kind: 'removed' });
    await expect(journal.listAll()).resolves.toMatchObject({ entries: [], kind: 'available' });
  });

  it('deletes all runs for one project without touching another', async () => {
    const { journal } = await createJournal();
    await journal.record('project-1', createItem({ id: 'one' }));
    await journal.record('project-1', createItem({ id: 'two' }));
    await journal.record('project-2', createItem({ id: 'three' }));

    await expect(journal.deleteForProject('project-1')).resolves.toEqual({ kind: 'removed' });
    await expect(journal.listAll()).resolves.toMatchObject({
      entries: [{ item: { id: 'three' }, projectId: 'project-2' }],
      kind: 'available',
    });
  });

  it('skips corrupt rows without hiding valid runs', async () => {
    const { database, journal } = await createJournal();
    await journal.record('project-1', createItem());
    await database.put('queueRuns', {
      itemJson: '{broken',
      key: '["project-2","queue-item-2"]',
      projectId: 'project-2',
      queueItemId: 'queue-item-2',
      schemaVersion: 1,
      submissionOrder: 2,
      updatedAt: 1,
    });

    await expect(journal.listAll()).resolves.toMatchObject({
      entries: [{ item: { id: 'queue-item-1' }, projectId: 'project-1' }],
      kind: 'available',
      removedCorrupt: 1,
    });
    expect(await database.get('queueRuns', '["project-2","queue-item-2"]')).toBeUndefined();
  });

  it('removes deeply invalid payloads at the storage boundary', async () => {
    const { database, journal } = await createJournal();
    const item = createItem();
    await database.put('queueRuns', {
      itemJson: JSON.stringify({
        cancellable: true,
        id: item.id,
        snapshot: { backendSubmission: {}, sourceId: 'workflow', submittedAt: 'now' },
        status: 'pending',
      }),
      key: '["project-1","queue-item-1"]',
      projectId: 'project-1',
      queueItemId: 'queue-item-1',
      schemaVersion: 1,
      submissionOrder: 1,
      updatedAt: 1,
    });

    const loaded = await journal.listAll();
    expect(loaded).toEqual({ entries: [], kind: 'available', removedCorrupt: 1 });
    expect(await database.count('queueRuns')).toBe(0);
  });

  it('removes malformed rows without hiding valid runs', async () => {
    const { database, journal } = await createJournal();
    await journal.record('project-1', createItem({ id: 'valid' }));
    const malformed = createItem({ id: 'malformed' }) as unknown as Record<string, unknown>;
    malformed.backendItemIds = 7;
    await database.put('queueRuns', {
      itemJson: JSON.stringify(malformed),
      key: '["project-1","malformed"]',
      projectId: 'project-1',
      queueItemId: 'malformed',
      schemaVersion: 1,
      submissionOrder: 2,
      updatedAt: 2,
    });

    await expect(journal.listForProject('project-1')).resolves.toMatchObject({
      entries: [{ item: { id: 'valid' } }],
      kind: 'available',
      removedCorrupt: 1,
    });
    expect(await database.get('queueRuns', '["project-1","malformed"]')).toBeUndefined();
  });

  it('rejects malformed and oversized records without poisoning the journal', async () => {
    const baselineBytes = new TextEncoder().encode(JSON.stringify(createItem())).byteLength;
    const { journal } = await createJournal({ maxRunBytes: baselineBytes + 8 });
    await expect(journal.record('', createItem())).resolves.toEqual({ kind: 'invalid' });
    await expect(journal.record('project-1', createItem({ id: '' }))).resolves.toEqual({ kind: 'invalid' });
    const cyclic = createItem();
    if (cyclic.snapshot.backendSubmission.kind === 'invalid') {
      throw new Error('Expected a compiled submission');
    }
    (cyclic.snapshot.backendSubmission.graph as unknown as Record<string, unknown>).cycle =
      cyclic.snapshot.backendSubmission;
    await expect(journal.record('project-1', cyclic)).resolves.toEqual({ kind: 'invalid' });
    const oversized = createItem();
    if (oversized.snapshot.backendSubmission.kind === 'invalid') {
      throw new Error('Expected a compiled submission');
    }
    (oversized.snapshot.backendSubmission.graph as unknown as Record<string, unknown>).padding = 'x'.repeat(256);
    await expect(journal.record('project-1', oversized)).resolves.toEqual({
      kind: 'too-large',
    });
    await expect(journal.record('project-1', createItem())).resolves.toEqual({ kind: 'stored' });
  });

  it('prunes oversized rows written by an older or less restrictive client', async () => {
    const item = createItem({ error: 'x'.repeat(256) });
    const baselineBytes = new TextEncoder().encode(JSON.stringify(createItem())).byteLength;
    const { database, journal } = await createJournal({ maxRunBytes: baselineBytes + 8 });
    await database.put('queueRuns', {
      itemJson: JSON.stringify(item),
      key: '["project-1","queue-item-1"]',
      projectId: 'project-1',
      queueItemId: 'queue-item-1',
      schemaVersion: 1,
      submissionOrder: 1,
      updatedAt: 1,
    });

    await expect(journal.listAll()).resolves.toEqual({ entries: [], kind: 'available', removedCorrupt: 1 });
    expect(await database.count('queueRuns')).toBe(0);
  });

  it('reports quota without poisoning later writes', async () => {
    const { database } = await createJournal();
    const journal = createIndexedDbQueueRunJournal(rejectFirstPutWithQuota(database));

    await expect(journal.record('project-1', createItem())).resolves.toEqual({ kind: 'quota' });
    expect(journal.availability).toBe('available');
    await expect(journal.record('project-1', createItem())).resolves.toEqual({ kind: 'stored' });
  });

  it('bounds aggregate entry count without evicting active runs', async () => {
    const { journal } = await createJournal({ maxEntries: 1 });

    await expect(journal.record('project-1', createItem({ id: 'first' }))).resolves.toEqual({ kind: 'stored' });
    await expect(journal.record('project-1', createItem({ id: 'second' }))).resolves.toEqual({ kind: 'quota' });
    await expect(journal.record('project-1', createItem({ id: 'first', status: 'running' }))).resolves.toEqual({
      kind: 'stored',
    });
    await expect(journal.listAll()).resolves.toMatchObject({
      entries: [{ item: { id: 'first', status: 'running' } }],
      kind: 'available',
    });
  });

  it('bounds aggregate bytes without evicting active runs', async () => {
    const itemBytes = new TextEncoder().encode(JSON.stringify(createItem())).byteLength;
    const { journal } = await createJournal({ totalMaxBytes: itemBytes + 32 });

    await expect(journal.record('project-1', createItem({ id: 'first' }))).resolves.toEqual({ kind: 'stored' });
    await expect(journal.record('project-1', createItem({ id: 'second' }))).resolves.toEqual({ kind: 'quota' });
    await expect(journal.listAll()).resolves.toMatchObject({
      entries: [{ item: { id: 'first' } }],
      kind: 'available',
    });
  });

  it('never expires active rows to admit newer work', async () => {
    let currentTime = 100;
    const { journal } = await createJournal({ maxEntries: 1, now: () => currentTime });
    await journal.record('closed-project', createItem({ id: 'old-active-run' }));

    currentTime = 30 * 24 * 60 * 60 * 1_000;
    await expect(journal.record('open-project', createItem({ id: 'new-run' }))).resolves.toEqual({ kind: 'quota' });
    await expect(journal.listAll()).resolves.toMatchObject({
      entries: [{ item: { id: 'old-active-run' }, projectId: 'closed-project' }],
      kind: 'available',
      removedCorrupt: 0,
    });
  });

  it('repairs negative aggregate metadata before enforcing budgets', async () => {
    const { database, journal } = await createJournal({ maxEntries: 1 });
    await database.put('metadata', { key: 'queueRunCount', value: -100 });
    await database.put('metadata', { key: 'queueRunTotalBytes', value: -100 });

    await expect(journal.record('project-1', createItem({ id: 'first' }))).resolves.toEqual({ kind: 'stored' });
    await expect(journal.record('project-1', createItem({ id: 'second' }))).resolves.toEqual({ kind: 'quota' });
    await expect(database.get('metadata', 'queueRunCount')).resolves.toEqual({ key: 'queueRunCount', value: 1 });
  });
});

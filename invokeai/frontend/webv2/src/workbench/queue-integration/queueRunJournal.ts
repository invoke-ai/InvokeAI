import type { AccountScope } from '@platform/state/accountLifecycle';
import type { IDBPObjectStore } from 'idb';

import { acquireAccountOwnedWorkbenchDatabase } from '@workbench/projects/accountOwnedWorkbenchDatabase';
import {
  isWorkbenchDatabaseAvailable,
  WORKBENCH_DATABASE_METADATA_STORE,
  WORKBENCH_QUEUE_RUN_STORE,
  WORKBENCH_QUEUE_RECEIPT_STORE,
  type QueueReceiptAcknowledgement,
  type QueueRunDatabaseRecord,
  type WorkbenchDatabase,
  type WorkbenchDatabaseSchema,
} from '@workbench/projects/workbenchDatabase';
import { normalizeRestoredQueueItem } from '@workbench/queue-integration/queueRunRestoration';

export const QUEUE_RUN_MAX_BYTES = 32 * 1024 * 1024;
export const QUEUE_RUN_TOTAL_MAX_BYTES = 128 * 1024 * 1024;
export const QUEUE_RUN_MAX_ENTRIES = 10_000;

export interface QueueRunJournalEntry {
  /** Untrusted serialized candidate. Normalize it against the current project before restoration. */
  item: unknown;
  projectId: string;
  queueItemId: string;
  submissionOrder: number;
}

export type QueueRunJournalLoadResult =
  | { entries: QueueRunJournalEntry[]; kind: 'available'; removedCorrupt: number }
  | { kind: 'unavailable' };
export type QueueRunJournalWriteResult = { kind: 'invalid' | 'quota' | 'stored' | 'too-large' | 'unavailable' };
export type QueueRunJournalSettleResult = { kind: 'removed' | 'unavailable' };

export interface QueueRunJournal {
  readonly availability: 'available' | 'unavailable';
  close(): void;
  acknowledgeReceipt(projectId: string, queueItemId: string): Promise<QueueRunJournalSettleResult>;
  listPendingReceipts(): Promise<
    { entries: QueueReceiptAcknowledgement[]; kind: 'available' } | { kind: 'unavailable' }
  >;
  deleteForProject(projectId: string): Promise<QueueRunJournalSettleResult>;
  listAll(): Promise<QueueRunJournalLoadResult>;
  listProjectIds(): Promise<{ kind: 'available'; projectIds: string[] } | { kind: 'unavailable' }>;
  listForProject(projectId: string): Promise<QueueRunJournalLoadResult>;
  record(projectId: string, item: unknown): Promise<QueueRunJournalWriteResult>;
  settle(projectId: string, queueItemId: string): Promise<QueueRunJournalSettleResult>;
}

const getByteSize = (value: string): number => new TextEncoder().encode(value).byteLength;
const QUEUE_RUN_SUBMISSION_SEQUENCE_KEY = 'queueRunSubmissionSequence';
const QUEUE_RUN_TOTAL_BYTES_KEY = 'queueRunTotalBytes';
const QUEUE_RUN_COUNT_KEY = 'queueRunCount';
const QUEUE_RUN_STORES = [
  WORKBENCH_DATABASE_METADATA_STORE,
  WORKBENCH_QUEUE_RUN_STORE,
  WORKBENCH_QUEUE_RECEIPT_STORE,
] as const;
type ReceiptStore = IDBPObjectStore<
  WorkbenchDatabaseSchema,
  typeof QUEUE_RUN_STORES,
  typeof WORKBENCH_QUEUE_RECEIPT_STORE,
  'readwrite'
>;
const queueReceiptAcknowledgement = async (store: ReceiptStore, record: QueueReceiptAcknowledgement): Promise<void> => {
  if (!(await store.getKey(record.key)) && (await store.count()) >= QUEUE_RUN_MAX_ENTRIES) {
    throw new DOMException('Queue receipt cleanup capacity exceeded.', 'QuotaExceededError');
  }
  await store.put(record);
};
type QueueRunStore = IDBPObjectStore<
  WorkbenchDatabaseSchema,
  typeof QUEUE_RUN_STORES,
  typeof WORKBENCH_QUEUE_RUN_STORE,
  'readwrite'
>;
type MetadataStore = IDBPObjectStore<
  WorkbenchDatabaseSchema,
  typeof QUEUE_RUN_STORES,
  typeof WORKBENCH_DATABASE_METADATA_STORE,
  'readwrite'
>;
const isNonEmptyString = (value: unknown): value is string => typeof value === 'string' && value.length > 0;
interface ActiveQueueItemEnvelope {
  cancellationPending?: boolean;
  cancellable: boolean;
  id: string;
  snapshot: { backendSubmission: object; sourceId: string; submittedAt: string };
  status: 'cancelled' | 'pending' | 'running';
}
const isActiveQueueItemEnvelope = (value: unknown): value is ActiveQueueItemEnvelope => {
  if (!value || typeof value !== 'object') {
    return false;
  }
  const item = value as Partial<ActiveQueueItemEnvelope>;
  const snapshot = item.snapshot as Partial<ActiveQueueItemEnvelope['snapshot']> | undefined;
  return (
    isNonEmptyString(item.id) &&
    (item.status === 'pending' ||
      item.status === 'running' ||
      (item.status === 'cancelled' && item.cancellationPending === true)) &&
    typeof item.cancellable === 'boolean' &&
    Boolean(snapshot && typeof snapshot === 'object') &&
    isNonEmptyString(snapshot?.sourceId) &&
    isNonEmptyString(snapshot?.submittedAt) &&
    Boolean(snapshot?.backendSubmission && typeof snapshot.backendSubmission === 'object')
  );
};
const getKey = (projectId: string, queueItemId: string): string => JSON.stringify([projectId, queueItemId]);
const isQuotaError = (error: unknown): boolean =>
  typeof error === 'object' && error !== null && 'name' in error && error.name === 'QuotaExceededError';
const observeTransaction = <T extends { done: Promise<unknown> }>(transaction: T): T => {
  void transaction.done.catch(() => undefined);
  return transaction;
};
const abortTransaction = (transaction: { abort(): void }): void => {
  try {
    transaction.abort();
  } catch {
    // A failed write may already have aborted the transaction.
  }
};

const decodeRecord = (record: QueueRunDatabaseRecord, maxRunBytes: number): QueueRunJournalEntry | null => {
  if (
    !isNonEmptyString(record.projectId) ||
    !isNonEmptyString(record.queueItemId) ||
    record.schemaVersion !== 1 ||
    record.key !== getKey(record.projectId, record.queueItemId) ||
    typeof record.itemJson !== 'string' ||
    getByteSize(record.itemJson) > maxRunBytes ||
    !Number.isSafeInteger(record.submissionOrder) ||
    record.submissionOrder <= 0 ||
    typeof record.updatedAt !== 'number' ||
    !Number.isFinite(record.updatedAt) ||
    record.updatedAt < 0
  ) {
    return null;
  }
  try {
    const item = normalizeRestoredQueueItem(JSON.parse(record.itemJson));
    return item?.id === record.queueItemId
      ? {
          item,
          projectId: record.projectId,
          queueItemId: record.queueItemId,
          submissionOrder: record.submissionOrder,
        }
      : null;
  } catch {
    return null;
  }
};

export const createIndexedDbQueueRunJournal = (
  database: WorkbenchDatabase,
  {
    maxEntries = QUEUE_RUN_MAX_ENTRIES,
    maxRunBytes = QUEUE_RUN_MAX_BYTES,
    now = Date.now,
    totalMaxBytes = QUEUE_RUN_TOTAL_MAX_BYTES,
  }: {
    maxEntries?: number;
    maxRunBytes?: number;
    now?: () => number;
    totalMaxBytes?: number;
  } = {}
): QueueRunJournal => {
  const runByteBudget =
    Number.isFinite(maxRunBytes) && maxRunBytes >= 0 ? Math.floor(maxRunBytes) : QUEUE_RUN_MAX_BYTES;
  const totalByteBudget =
    Number.isFinite(totalMaxBytes) && totalMaxBytes >= 0 ? Math.floor(totalMaxBytes) : QUEUE_RUN_TOTAL_MAX_BYTES;
  const entryBudget = Number.isSafeInteger(maxEntries) && maxEntries >= 0 ? maxEntries : QUEUE_RUN_MAX_ENTRIES;
  let isClosed = false;
  let isUnavailable = false;
  const isAvailable = (): boolean => !isClosed && !isUnavailable && isWorkbenchDatabaseAvailable(database);
  const markUnavailable = (): void => {
    isUnavailable = true;
  };
  const readCounters = async (
    runStore: QueueRunStore,
    metadataStore: MetadataStore
  ): Promise<{ count: number; totalBytes: number }> => {
    const storedTotal = await metadataStore.get(QUEUE_RUN_TOTAL_BYTES_KEY);
    const storedCount = await metadataStore.get(QUEUE_RUN_COUNT_KEY);
    if (
      Number.isSafeInteger(storedTotal?.value) &&
      storedTotal!.value >= 0 &&
      Number.isSafeInteger(storedCount?.value) &&
      storedCount!.value >= 0
    ) {
      return { count: storedCount!.value, totalBytes: storedTotal!.value };
    }
    let count = 0;
    let totalBytes = 0;
    let cursor = await runStore.openCursor();
    while (cursor) {
      count += 1;
      totalBytes += getByteSize(cursor.value.itemJson);
      cursor = await cursor.continue();
    }
    return { count, totalBytes };
  };
  const writeCounters = async (metadataStore: MetadataStore, count: number, totalBytes: number): Promise<void> => {
    await metadataStore.put({ key: QUEUE_RUN_TOTAL_BYTES_KEY, value: totalBytes });
    await metadataStore.put({ key: QUEUE_RUN_COUNT_KEY, value: count });
  };
  const load = async (projectId?: string): Promise<QueueRunJournalLoadResult> => {
    if (!isAvailable()) {
      return { kind: 'unavailable' };
    }
    try {
      const transaction = observeTransaction(database.transaction(QUEUE_RUN_STORES, 'readwrite'));
      const runStore = transaction.objectStore(WORKBENCH_QUEUE_RUN_STORE);
      const metadataStore = transaction.objectStore(WORKBENCH_DATABASE_METADATA_STORE);
      const countersBefore = await readCounters(runStore, metadataStore);
      const entries: QueueRunJournalEntry[] = [];
      let removedCorrupt = 0;
      let removedBytes = 0;
      const source = projectId ? runStore.index('byProject') : runStore;
      let cursor = await source.openCursor(projectId);
      while (cursor) {
        const record = cursor.value;
        const entry = decodeRecord(record, runByteBudget);
        if (entry) {
          entries.push(entry);
        } else {
          removedCorrupt += 1;
          removedBytes += getByteSize(record.itemJson);
          await cursor.delete();
        }
        cursor = await cursor.continue();
      }
      if (removedCorrupt > 0) {
        await writeCounters(
          metadataStore,
          Math.max(0, countersBefore.count - removedCorrupt),
          Math.max(0, countersBefore.totalBytes - removedBytes)
        );
      }
      await transaction.done;
      entries.sort((left, right) => {
        const projectOrder = left.projectId.localeCompare(right.projectId);
        return projectOrder || right.submissionOrder - left.submissionOrder;
      });
      return { entries, kind: 'available', removedCorrupt };
    } catch {
      markUnavailable();
      return { kind: 'unavailable' };
    }
  };

  return {
    get availability() {
      return isAvailable() ? 'available' : 'unavailable';
    },
    close() {
      isClosed = true;
    },
    async listProjectIds() {
      if (!isAvailable()) {
        return { kind: 'unavailable' };
      }
      try {
        const transaction = observeTransaction(database.transaction(WORKBENCH_QUEUE_RUN_STORE));
        const projectIds: string[] = [];
        let cursor = await transaction.store.index('byProject').openKeyCursor(undefined, 'nextunique');
        while (cursor) {
          if (isNonEmptyString(cursor.key)) {
            projectIds.push(cursor.key);
          }
          cursor = await cursor.continue();
        }
        await transaction.done;
        return { kind: 'available', projectIds };
      } catch {
        return { kind: 'unavailable' };
      }
    },
    async listPendingReceipts() {
      if (!isAvailable()) {
        return { kind: 'unavailable' };
      }
      try {
        const transaction = observeTransaction(database.transaction(WORKBENCH_QUEUE_RECEIPT_STORE, 'readwrite'));
        const entries: QueueReceiptAcknowledgement[] = [];
        let cursor = await transaction.store.openCursor();
        while (cursor && entries.length < 100) {
          const entry = cursor.value;
          if (
            entry &&
            isNonEmptyString(entry.projectId) &&
            isNonEmptyString(entry.queueItemId) &&
            entry.key === getKey(entry.projectId, entry.queueItemId)
          ) {
            entries.push(entry);
          } else {
            await cursor.delete();
          }
          cursor = await cursor.continue();
        }
        await transaction.done;
        return { entries, kind: 'available' };
      } catch {
        return { kind: 'unavailable' };
      }
    },
    async acknowledgeReceipt(projectId, queueItemId) {
      if (!isAvailable()) {
        return { kind: 'unavailable' };
      }
      try {
        const transaction = observeTransaction(database.transaction(QUEUE_RUN_STORES, 'readwrite'));
        const key = getKey(projectId, queueItemId);
        const runStore = transaction.objectStore(WORKBENCH_QUEUE_RUN_STORE);
        const existing = await runStore.get(key);
        if (existing) {
          await runStore.put({ ...existing, receiptAcknowledged: true });
        }
        await transaction.objectStore(WORKBENCH_QUEUE_RECEIPT_STORE).delete(key);
        await transaction.done;
        return { kind: 'removed' };
      } catch {
        return { kind: 'unavailable' };
      }
    },
    async deleteForProject(projectId) {
      if (!isAvailable()) {
        return { kind: 'unavailable' };
      }
      if (!isNonEmptyString(projectId)) {
        return { kind: 'removed' };
      }
      try {
        const transaction = observeTransaction(database.transaction(QUEUE_RUN_STORES, 'readwrite'));
        const runStore = transaction.objectStore(WORKBENCH_QUEUE_RUN_STORE);
        const metadataStore = transaction.objectStore(WORKBENCH_DATABASE_METADATA_STORE);
        const countersBefore = await readCounters(runStore, metadataStore);
        let removedCount = 0;
        let removedBytes = 0;
        let cursor = await runStore.index('byProject').openCursor(projectId);
        while (cursor) {
          const record = cursor.value;
          if (!record.receiptAcknowledged && isNonEmptyString(record.queueItemId)) {
            try {
              await queueReceiptAcknowledgement(transaction.objectStore(WORKBENCH_QUEUE_RECEIPT_STORE), {
                key: getKey(projectId, record.queueItemId),
                projectId,
                queueItemId: record.queueItemId,
              });
            } catch (error) {
              abortTransaction(transaction);
              throw error;
            }
          }
          removedCount += 1;
          removedBytes += getByteSize(cursor.value.itemJson);
          await cursor.delete();
          cursor = await cursor.continue();
        }
        await writeCounters(
          metadataStore,
          Math.max(0, countersBefore.count - removedCount),
          Math.max(0, countersBefore.totalBytes - removedBytes)
        );
        await transaction.done;
        return { kind: 'removed' };
      } catch (error) {
        if (!isQuotaError(error)) {
          markUnavailable();
        }
        return { kind: 'unavailable' };
      }
    },
    listAll: () => load(),
    listForProject(projectId) {
      return isNonEmptyString(projectId)
        ? load(projectId)
        : Promise.resolve({ entries: [], kind: 'available', removedCorrupt: 0 });
    },
    async record(projectId, item) {
      if (!isAvailable()) {
        return { kind: 'unavailable' };
      }
      if (!isNonEmptyString(projectId)) {
        return { kind: 'invalid' };
      }
      const normalized = normalizeRestoredQueueItem(item);
      if (!normalized) {
        return { kind: 'invalid' };
      }
      let itemJson: string;
      try {
        itemJson = JSON.stringify(normalized);
      } catch {
        return { kind: 'invalid' };
      }
      if (getByteSize(itemJson) > runByteBudget) {
        return { kind: 'too-large' };
      }
      try {
        if (!isActiveQueueItemEnvelope(JSON.parse(itemJson))) {
          return { kind: 'invalid' };
        }
      } catch {
        return { kind: 'invalid' };
      }
      try {
        const transaction = observeTransaction(database.transaction(QUEUE_RUN_STORES, 'readwrite'));
        const runStore = transaction.objectStore(WORKBENCH_QUEUE_RUN_STORE);
        const metadataStore = transaction.objectStore(WORKBENCH_DATABASE_METADATA_STORE);
        const key = getKey(projectId, normalized.id);
        const existing = await runStore.get(key);
        const { count, totalBytes } = await readCounters(runStore, metadataStore);
        const existingBytes = existing ? getByteSize(existing.itemJson) : 0;
        const nextTotalBytes = totalBytes - existingBytes + getByteSize(itemJson);
        const nextCount = count + (existing ? 0 : 1);
        if (nextTotalBytes > totalByteBudget || nextCount > entryBudget) {
          await writeCounters(metadataStore, count, totalBytes);
          await transaction.done;
          return { kind: 'quota' };
        }
        let submissionOrder = existing?.submissionOrder;
        if (!Number.isSafeInteger(submissionOrder) || (submissionOrder as number) <= 0) {
          const sequence = await metadataStore.get(QUEUE_RUN_SUBMISSION_SEQUENCE_KEY);
          const current = Number.isSafeInteger(sequence?.value) && sequence!.value >= 0 ? sequence!.value : 0;
          submissionOrder = current >= Number.MAX_SAFE_INTEGER ? 1 : current + 1;
          await metadataStore.put({ key: QUEUE_RUN_SUBMISSION_SEQUENCE_KEY, value: submissionOrder });
        }
        const stableSubmissionOrder = submissionOrder as number;
        const record: QueueRunDatabaseRecord = {
          receiptAcknowledged: existing?.receiptAcknowledged,
          byteSize: getByteSize(itemJson),
          itemJson,
          key,
          projectId,
          queueItemId: normalized.id,
          schemaVersion: 1,
          submissionOrder: stableSubmissionOrder,
          updatedAt: now(),
        };
        await runStore.put(record);
        if (!existing?.receiptAcknowledged && normalized.backendItemIds?.length) {
          try {
            await queueReceiptAcknowledgement(transaction.objectStore(WORKBENCH_QUEUE_RECEIPT_STORE), {
              key,
              projectId,
              queueItemId: normalized.id,
            });
          } catch (error) {
            abortTransaction(transaction);
            throw error;
          }
        }
        await writeCounters(metadataStore, nextCount, nextTotalBytes);
        await transaction.done;
        return { kind: 'stored' };
      } catch (error) {
        if (isQuotaError(error)) {
          return { kind: 'quota' };
        }
        markUnavailable();
        return { kind: 'unavailable' };
      }
    },
    async settle(projectId, queueItemId) {
      if (!isAvailable()) {
        return { kind: 'unavailable' };
      }
      if (!isNonEmptyString(projectId) || !isNonEmptyString(queueItemId)) {
        return { kind: 'removed' };
      }
      try {
        const transaction = observeTransaction(database.transaction(QUEUE_RUN_STORES, 'readwrite'));
        const runStore = transaction.objectStore(WORKBENCH_QUEUE_RUN_STORE);
        const metadataStore = transaction.objectStore(WORKBENCH_DATABASE_METADATA_STORE);
        const key = getKey(projectId, queueItemId);
        const existing = await runStore.get(key);
        if (existing && !existing.receiptAcknowledged) {
          try {
            await queueReceiptAcknowledgement(transaction.objectStore(WORKBENCH_QUEUE_RECEIPT_STORE), {
              key,
              projectId,
              queueItemId,
            });
          } catch (error) {
            abortTransaction(transaction);
            throw error;
          }
        }
        await runStore.delete(key);
        if (existing) {
          const storedTotal = await metadataStore.get(QUEUE_RUN_TOTAL_BYTES_KEY);
          const storedCount = await metadataStore.get(QUEUE_RUN_COUNT_KEY);
          if (Number.isSafeInteger(storedTotal?.value)) {
            await metadataStore.put({
              key: QUEUE_RUN_TOTAL_BYTES_KEY,
              value: Math.max(0, storedTotal!.value - getByteSize(existing.itemJson)),
            });
          }
          if (Number.isSafeInteger(storedCount?.value)) {
            await metadataStore.put({ key: QUEUE_RUN_COUNT_KEY, value: Math.max(0, storedCount!.value - 1) });
          }
        }
        await transaction.done;
        return { kind: 'removed' };
      } catch (error) {
        if (!isQuotaError(error)) {
          markUnavailable();
        }
        return { kind: 'unavailable' };
      }
    },
  };
};

export const createAccountOwnedQueueRunJournal = async (
  owner: AccountScope,
  dependencies: Parameters<typeof acquireAccountOwnedWorkbenchDatabase>[1] = {}
): Promise<QueueRunJournal> => {
  const lease = await acquireAccountOwnedWorkbenchDatabase(owner, dependencies);
  if (!lease) {
    return {
      availability: 'unavailable',
      acknowledgeReceipt: () => Promise.resolve({ kind: 'unavailable' }),
      listPendingReceipts: () => Promise.resolve({ kind: 'unavailable' }),
      close: () => undefined,
      deleteForProject: () => Promise.resolve({ kind: 'unavailable' }),
      listAll: () => Promise.resolve({ kind: 'unavailable' }),
      listProjectIds: () => Promise.resolve({ kind: 'unavailable' }),
      listForProject: () => Promise.resolve({ kind: 'unavailable' }),
      record: () => Promise.resolve({ kind: 'unavailable' }),
      settle: () => Promise.resolve({ kind: 'unavailable' }),
    };
  }
  const journal = createIndexedDbQueueRunJournal(lease.database);
  let isReleased = false;
  return {
    ...journal,
    get availability() {
      return journal.availability;
    },
    close() {
      if (isReleased) {
        return;
      }
      isReleased = true;
      journal.close();
      lease.release();
    },
  };
};

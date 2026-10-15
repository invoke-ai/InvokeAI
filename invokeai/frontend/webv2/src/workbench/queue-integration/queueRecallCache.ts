import type { GenerateWidgetValues } from '@features/generation/contracts';
import type { QueueSourceId } from '@features/queue/contracts';
import type { VideoWidgetValues } from '@features/video';
import type { AccountScope } from '@platform/state/accountLifecycle';
import type { IDBPTransaction } from 'idb';

import { acquireAccountOwnedWorkbenchDatabase } from '@workbench/projects/accountOwnedWorkbenchDatabase';
import {
  isWorkbenchDatabaseAvailable,
  WORKBENCH_DATABASE_METADATA_STORE,
  WORKBENCH_RECALL_CACHE_BODY_STORE,
  WORKBENCH_RECALL_CACHE_STORE,
  type RecallCacheBodyDatabaseRecord,
  type RecallCacheDatabaseRecord,
  type WorkbenchDatabase,
  type WorkbenchDatabaseSchema,
} from '@workbench/projects/workbenchDatabase';

export const QUEUE_RECALL_CACHE_MAX_BYTES = 32 * 1024 * 1024;
export const QUEUE_RECALL_CACHE_MAX_ENTRIES = 500;

export interface QueueRecallSnapshotInput {
  generateValues?: GenerateWidgetValues;
  projectId: string;
  queueItemId: string;
  sourceId: QueueSourceId;
  submittedAt: string;
  videoValues?: VideoWidgetValues;
}

/** Values crossing the persistence boundary stay untrusted until their feature owner normalizes them. */
export interface QueueRecallSnapshot {
  generateValues?: unknown;
  projectId: string;
  queueItemId: string;
  sourceId: QueueSourceId;
  submittedAt: string;
  videoValues?: unknown;
}

export type QueueRecallCacheGetResult =
  | { kind: 'found'; snapshot: QueueRecallSnapshot }
  | { kind: 'corrupt' | 'missing' | 'unavailable' };
export type QueueRecallCachePutResult = { kind: 'invalid' | 'quota' | 'stored' | 'too-large' | 'unavailable' };

export interface QueueRecallCache {
  readonly availability: 'available' | 'unavailable';
  close(): void;
  get(queueItemId: string): Promise<QueueRecallCacheGetResult>;
  put(snapshot: QueueRecallSnapshotInput): Promise<QueueRecallCachePutResult>;
}

const RECALL_ACCESS_SEQUENCE_KEY = 'recallAccessSequence';
const RECALL_STORES = [
  WORKBENCH_DATABASE_METADATA_STORE,
  WORKBENCH_RECALL_CACHE_STORE,
  WORKBENCH_RECALL_CACHE_BODY_STORE,
] as const;
const QUEUE_SOURCE_IDS = new Set<QueueSourceId>(['canvas', 'generate', 'upscale', 'video', 'workflow']);

const getByteSize = (value: string): number => new TextEncoder().encode(value).byteLength;
const isNonEmptyString = (value: unknown): value is string => typeof value === 'string' && value.length > 0;
const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);
const isQuotaError = (error: unknown): boolean =>
  typeof error === 'object' && error !== null && 'name' in error && error.name === 'QuotaExceededError';
const observeTransaction = <T extends { done: Promise<unknown> }>(transaction: T): T => {
  void transaction.done.catch(() => undefined);
  return transaction;
};
const isQueueSourceId = (value: unknown): value is QueueSourceId =>
  typeof value === 'string' && QUEUE_SOURCE_IDS.has(value as QueueSourceId);
const isQueueRecallSnapshot = (value: unknown): value is QueueRecallSnapshot => {
  if (!isRecord(value)) {
    return false;
  }
  return (
    isNonEmptyString(value.projectId) &&
    isNonEmptyString(value.queueItemId) &&
    isQueueSourceId(value.sourceId) &&
    isNonEmptyString(value.submittedAt) &&
    (isRecord(value.generateValues) || isRecord(value.videoValues)) &&
    (value.generateValues === undefined || isRecord(value.generateValues)) &&
    (value.videoValues === undefined || isRecord(value.videoValues))
  );
};
const isRecallMetadata = (record: unknown): record is RecallCacheDatabaseRecord =>
  isRecord(record) &&
  isNonEmptyString(record.queueItemId) &&
  isNonEmptyString(record.projectId) &&
  Number.isSafeInteger(record.byteSize) &&
  (record.byteSize as number) >= 0 &&
  Number.isSafeInteger(record.lastAccessOrder) &&
  (record.lastAccessOrder as number) > 0;

const decodeRecord = (
  metadata: RecallCacheDatabaseRecord,
  body: RecallCacheBodyDatabaseRecord
): QueueRecallSnapshot | null => {
  if (
    !isRecallMetadata(metadata) ||
    !isRecord(body) ||
    body.queueItemId !== metadata.queueItemId ||
    typeof body.payloadJson !== 'string' ||
    getByteSize(body.payloadJson) !== metadata.byteSize
  ) {
    return null;
  }
  try {
    const snapshot: unknown = JSON.parse(body.payloadJson);
    return isQueueRecallSnapshot(snapshot) &&
      snapshot.queueItemId === metadata.queueItemId &&
      snapshot.projectId === metadata.projectId
      ? snapshot
      : null;
  } catch {
    return null;
  }
};

export const createIndexedDbQueueRecallCache = (
  database: WorkbenchDatabase,
  {
    maxBytes = QUEUE_RECALL_CACHE_MAX_BYTES,
    maxEntries = QUEUE_RECALL_CACHE_MAX_ENTRIES,
  }: { maxBytes?: number; maxEntries?: number } = {}
): QueueRecallCache => {
  let isClosed = false;
  let isUnavailable = false;
  const byteBudget = Number.isFinite(maxBytes) && maxBytes >= 0 ? Math.floor(maxBytes) : QUEUE_RECALL_CACHE_MAX_BYTES;
  const entryBudget =
    Number.isFinite(maxEntries) && maxEntries >= 0 ? Math.floor(maxEntries) : QUEUE_RECALL_CACHE_MAX_ENTRIES;
  const isAvailable = (): boolean => !isClosed && !isUnavailable && isWorkbenchDatabaseAvailable(database);
  const markUnavailable = (): void => {
    isUnavailable = true;
  };
  const nextAccessOrder = async (
    transaction: IDBPTransaction<WorkbenchDatabaseSchema, typeof RECALL_STORES, 'readwrite'>
  ): Promise<number> => {
    const store = transaction.objectStore(WORKBENCH_DATABASE_METADATA_STORE);
    const current = await store.get(RECALL_ACCESS_SEQUENCE_KEY);
    let value = current?.value;
    if (!Number.isSafeInteger(value) || (value as number) < 0) {
      value = 0;
    }
    if ((value as number) >= Number.MAX_SAFE_INTEGER) {
      const recallStore = transaction.objectStore(WORKBENCH_RECALL_CACHE_STORE);
      const records = (await recallStore.getAll()).filter(isRecallMetadata);
      records.sort(
        (left, right) =>
          left.lastAccessOrder - right.lastAccessOrder || left.queueItemId.localeCompare(right.queueItemId)
      );
      for (const [index, record] of records.entries()) {
        await recallStore.put({ ...record, lastAccessOrder: index + 1 });
      }
      value = records.length;
    }
    const next = (value as number) + 1;
    await store.put({ key: RECALL_ACCESS_SEQUENCE_KEY, value: next });
    return next;
  };

  return {
    get availability() {
      return isAvailable() ? 'available' : 'unavailable';
    },
    close() {
      isClosed = true;
    },
    async get(queueItemId) {
      if (!isAvailable()) {
        return { kind: 'unavailable' };
      }
      if (!isNonEmptyString(queueItemId)) {
        return { kind: 'missing' };
      }
      let decodedSnapshot: QueueRecallSnapshot | null = null;
      try {
        const transaction = observeTransaction(database.transaction(RECALL_STORES, 'readwrite'));
        const metadataStore = transaction.objectStore(WORKBENCH_RECALL_CACHE_STORE);
        const bodyStore = transaction.objectStore(WORKBENCH_RECALL_CACHE_BODY_STORE);
        const [metadata, body] = await Promise.all([metadataStore.get(queueItemId), bodyStore.get(queueItemId)]);
        if (!metadata && !body) {
          await transaction.done;
          return { kind: 'missing' };
        }
        if (!metadata || !body) {
          await Promise.all([metadataStore.delete(queueItemId), bodyStore.delete(queueItemId)]);
          await transaction.done;
          return { kind: 'corrupt' };
        }
        const snapshot = decodeRecord(metadata, body);
        if (!snapshot) {
          await Promise.all([metadataStore.delete(queueItemId), bodyStore.delete(queueItemId)]);
          await transaction.done;
          return { kind: 'corrupt' };
        }
        decodedSnapshot = snapshot;
        await metadataStore.put({ ...metadata, lastAccessOrder: await nextAccessOrder(transaction) });
        await transaction.done;
        return { kind: 'found', snapshot };
      } catch (error) {
        if (decodedSnapshot && isQuotaError(error)) {
          return { kind: 'found', snapshot: decodedSnapshot };
        }
        markUnavailable();
        return { kind: 'unavailable' };
      }
    },
    async put(snapshot) {
      if (!isAvailable()) {
        return { kind: 'unavailable' };
      }
      if (!isQueueRecallSnapshot(snapshot)) {
        return { kind: 'invalid' };
      }
      let payloadJson: string;
      try {
        payloadJson = JSON.stringify(snapshot);
      } catch {
        return { kind: 'invalid' };
      }
      const byteSize = getByteSize(payloadJson);
      if (byteSize > byteBudget || entryBudget === 0) {
        return { kind: 'too-large' };
      }
      try {
        if (!isQueueRecallSnapshot(JSON.parse(payloadJson))) {
          return { kind: 'invalid' };
        }
      } catch {
        return { kind: 'invalid' };
      }
      try {
        const transaction = observeTransaction(database.transaction(RECALL_STORES, 'readwrite'));
        const metadataStore = transaction.objectStore(WORKBENCH_RECALL_CACHE_STORE);
        const bodyStore = transaction.objectStore(WORKBENCH_RECALL_CACHE_BODY_STORE);
        const [records, keys] = await Promise.all([
          metadataStore.getAll() as Promise<unknown[]>,
          metadataStore.getAllKeys(),
        ]);
        const candidates: RecallCacheDatabaseRecord[] = [];
        let totalBytes = 0;
        for (const [index, record] of records.entries()) {
          const storedKey = keys[index];
          if (!isRecallMetadata(record) || storedKey !== record.queueItemId) {
            if (typeof storedKey === 'string') {
              await Promise.all([metadataStore.delete(storedKey), bodyStore.delete(storedKey)]);
            }
            continue;
          }
          if (record.queueItemId === snapshot.queueItemId) {
            continue;
          }
          candidates.push(record);
          totalBytes += record.byteSize;
        }
        candidates.sort(
          (left, right) =>
            left.lastAccessOrder - right.lastAccessOrder || left.queueItemId.localeCompare(right.queueItemId)
        );
        while (candidates.length + 1 > entryBudget || totalBytes + byteSize > byteBudget) {
          const evicted = candidates.shift();
          if (!evicted) {
            break;
          }
          totalBytes -= evicted.byteSize;
          await Promise.all([metadataStore.delete(evicted.queueItemId), bodyStore.delete(evicted.queueItemId)]);
        }
        const lastAccessOrder = await nextAccessOrder(transaction);
        await bodyStore.put({ payloadJson, queueItemId: snapshot.queueItemId });
        await metadataStore.put({
          byteSize,
          lastAccessOrder,
          projectId: snapshot.projectId,
          queueItemId: snapshot.queueItemId,
        });
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
  };
};

export const createAccountOwnedQueueRecallCache = async (
  owner: AccountScope,
  dependencies: Parameters<typeof acquireAccountOwnedWorkbenchDatabase>[1] = {}
): Promise<QueueRecallCache> => {
  const lease = await acquireAccountOwnedWorkbenchDatabase(owner, dependencies);
  if (!lease) {
    return {
      availability: 'unavailable',
      close: () => undefined,
      get: () => Promise.resolve({ kind: 'unavailable' }),
      put: () => Promise.resolve({ kind: 'unavailable' }),
    };
  }
  const cache = createIndexedDbQueueRecallCache(lease.database);
  let isReleased = false;
  return {
    ...cache,
    get availability() {
      return cache.availability;
    },
    close() {
      if (isReleased) {
        return;
      }
      isReleased = true;
      cache.close();
      lease.release();
    },
  };
};

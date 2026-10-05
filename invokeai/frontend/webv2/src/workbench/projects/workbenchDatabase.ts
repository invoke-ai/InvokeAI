import { openDB, type DBSchema, type IDBPDatabase } from 'idb';

import type { ProjectDraftBody, ProjectDraftMetadata, ProjectDraftWriterClaim } from './draftStore';

export const WORKBENCH_DATABASE_VERSION = 4;
export const WORKBENCH_DRAFT_STORE = 'drafts';
export const WORKBENCH_DRAFT_BODY_STORE = 'draftBodies';
export const WORKBENCH_DRAFT_WRITER_STORE = 'draftWriters';
export const WORKBENCH_QUEUE_RUN_STORE = 'queueRuns';
export const WORKBENCH_QUEUE_RECEIPT_STORE = 'queueReceiptAcks';
export const WORKBENCH_RECALL_CACHE_STORE = 'recallCache';
export const WORKBENCH_RECALL_CACHE_BODY_STORE = 'recallBodies';
export const WORKBENCH_DATABASE_METADATA_STORE = 'metadata';

export interface QueueRunDatabaseRecord {
  receiptAcknowledged?: boolean;
  byteSize?: number;
  itemJson: string;
  key: string;
  schemaVersion: 1;
  submissionOrder: number;
  projectId: string;
  queueItemId: string;
  updatedAt: number;
}

export interface QueueReceiptAcknowledgement {
  key: string;
  projectId: string;
  queueItemId: string;
}

export interface RecallCacheDatabaseRecord {
  byteSize: number;
  lastAccessOrder: number;
  projectId: string;
  queueItemId: string;
}

export interface RecallCacheBodyDatabaseRecord {
  payloadJson: string;
  queueItemId: string;
}

export interface WorkbenchDatabaseMetadataRecord {
  key: string;
  value: number;
}

export interface WorkbenchDatabaseSchema extends DBSchema {
  drafts: {
    indexes: { byProject: string };
    key: [string, string];
    value: ProjectDraftMetadata;
  };
  draftBodies: {
    indexes: { byIntegrity: [string, string, number, number] };
    key: [string, string];
    value: ProjectDraftBody;
  };
  draftWriters: {
    indexes: { byRetarget: [string, string, string] };
    key: [string, string];
    value: ProjectDraftWriterClaim;
  };
  queueRuns: {
    indexes: { byProject: string };
    key: string;
    value: QueueRunDatabaseRecord;
  };
  queueReceiptAcks: {
    key: string;
    value: QueueReceiptAcknowledgement;
  };
  recallCache: {
    indexes: { byLastAccessOrder: number };
    key: string;
    value: RecallCacheDatabaseRecord;
  };
  recallBodies: {
    key: string;
    value: RecallCacheBodyDatabaseRecord;
  };
  metadata: {
    key: string;
    value: WorkbenchDatabaseMetadataRecord;
  };
}

export type WorkbenchDatabase = IDBPDatabase<WorkbenchDatabaseSchema>;

const WORKBENCH_DATABASE_NAME_BASE = 'invokeai:v7:webv2:workbench';

export const getWorkbenchDatabaseName = (storageSuffix: string): string =>
  `${WORKBENCH_DATABASE_NAME_BASE}${storageSuffix}`;

const unavailableDatabases = new WeakSet<WorkbenchDatabase>();

export const isWorkbenchDatabaseAvailable = (database: WorkbenchDatabase): boolean =>
  !unavailableDatabases.has(database);

export const openWorkbenchDatabase = (
  storageSuffix: string,
  { timeoutMs = 1_000 }: { timeoutMs?: number } = {}
): Promise<WorkbenchDatabase> => {
  let connection: WorkbenchDatabase | undefined;
  let didSettle = false;
  let rejectOpen: (reason: unknown) => void = () => undefined;
  const opening = openDB<WorkbenchDatabaseSchema>(getWorkbenchDatabaseName(storageSuffix), WORKBENCH_DATABASE_VERSION, {
    blocked: () => rejectOpen(new DOMException('Opening the workbench database was blocked.', 'InvalidStateError')),
    blocking: () => {
      if (connection) {
        unavailableDatabases.add(connection);
        connection.close();
      }
    },
    terminated: () => {
      if (connection) {
        unavailableDatabases.add(connection);
      }
    },
    upgrade(database, oldVersion, _newVersion, transaction) {
      if (oldVersion < 1) {
        const drafts = database.createObjectStore(WORKBENCH_DRAFT_STORE, {
          keyPath: ['projectId', 'editorSessionId'],
        });
        drafts.createIndex('byProject', 'projectId');

        const draftBodies = database.createObjectStore(WORKBENCH_DRAFT_BODY_STORE, {
          keyPath: ['projectId', 'editorSessionId'],
        });
        draftBodies.createIndex('byIntegrity', ['projectId', 'editorSessionId', 'generation', 'documentByteSize'], {
          unique: true,
        });

        database.createObjectStore(WORKBENCH_DRAFT_WRITER_STORE, {
          keyPath: ['projectId', 'editorSessionId'],
        });

        const queueRuns = database.createObjectStore(WORKBENCH_QUEUE_RUN_STORE, { keyPath: 'key' });
        queueRuns.createIndex('byProject', 'projectId');
      }

      if (oldVersion === 1) {
        database.deleteObjectStore(WORKBENCH_RECALL_CACHE_STORE);
      }
      if (oldVersion < 2) {
        const recallCache = database.createObjectStore(WORKBENCH_RECALL_CACHE_STORE, { keyPath: 'queueItemId' });
        recallCache.createIndex('byLastAccessOrder', 'lastAccessOrder');
        database.createObjectStore(WORKBENCH_RECALL_CACHE_BODY_STORE, { keyPath: 'queueItemId' });
        database.createObjectStore(WORKBENCH_DATABASE_METADATA_STORE, { keyPath: 'key' });
      }
      if (oldVersion < 3) {
        transaction
          .objectStore(WORKBENCH_DRAFT_WRITER_STORE)
          .createIndex('byRetarget', ['projectId', 'editorSessionId', 'retargetedToProjectId']);
      }
      if (oldVersion < 4) {
        database.createObjectStore(WORKBENCH_QUEUE_RECEIPT_STORE, { keyPath: 'key' });
      }
    },
  });
  return new Promise((resolve, reject) => {
    rejectOpen = (reason) => {
      if (!didSettle) {
        didSettle = true;
        reject(reason);
      }
    };
    const timeout = globalThis.setTimeout(
      () => rejectOpen(new DOMException('Opening the workbench database timed out.', 'TimeoutError')),
      timeoutMs
    );
    void opening.then(
      (database) => {
        connection = database;
        globalThis.clearTimeout(timeout);
        if (didSettle) {
          unavailableDatabases.add(database);
          database.close();
          return;
        }
        didSettle = true;
        resolve(database);
      },
      (error) => {
        globalThis.clearTimeout(timeout);
        rejectOpen(error);
      }
    );
  });
};

export type DeleteWorkbenchDatabaseFinalResult = { kind: 'deleted' | 'unavailable' };
export type DeleteWorkbenchDatabaseResult =
  | DeleteWorkbenchDatabaseFinalResult
  | { completion: Promise<DeleteWorkbenchDatabaseFinalResult>; kind: 'blocked' };

const pendingDeletions = new Map<string, Promise<DeleteWorkbenchDatabaseResult>>();

export const deleteWorkbenchDatabase = (storageSuffix: string): Promise<DeleteWorkbenchDatabaseResult> => {
  const name = getWorkbenchDatabaseName(storageSuffix);
  const pending = pendingDeletions.get(name);
  if (pending) {
    return pending;
  }

  const deletion = new Promise<DeleteWorkbenchDatabaseResult>((resolve) => {
    let didReport = false;
    let resolveCompletion: (result: DeleteWorkbenchDatabaseFinalResult) => void = () => undefined;
    const completion = new Promise<DeleteWorkbenchDatabaseFinalResult>((resolveFinal) => {
      resolveCompletion = resolveFinal;
    });
    const finish = (result: DeleteWorkbenchDatabaseFinalResult): void => {
      resolveCompletion(result);
      queueMicrotask(() => {
        if (pendingDeletions.get(name) === deletion) {
          pendingDeletions.delete(name);
        }
      });
      if (!didReport) {
        didReport = true;
        resolve(result);
      }
    };

    let request: IDBOpenDBRequest;
    try {
      request = indexedDB.deleteDatabase(name);
    } catch {
      finish({ kind: 'unavailable' });
      return;
    }
    request.addEventListener('blocked', () => {
      if (!didReport) {
        didReport = true;
        resolve({ completion, kind: 'blocked' });
      }
    });
    request.addEventListener('error', () => finish({ kind: 'unavailable' }));
    request.addEventListener('success', () => finish({ kind: 'deleted' }));
  });
  pendingDeletions.set(name, deletion);
  return deletion;
};

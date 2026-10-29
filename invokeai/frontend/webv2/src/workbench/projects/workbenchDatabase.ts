import { openDB, type DBSchema, type IDBPDatabase } from 'idb';

import type {
  ProjectDraftBody,
  ProjectDraftMetadata,
  ProjectDraftWriterClaim,
  ProjectUnloadJournalEntry,
  ProjectUnloadJournalKey,
} from './draftStore';

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

export const UNLOAD_JOURNAL_DATABASE_VERSION = 1;
export const UNLOAD_JOURNAL_STORE = 'entries';

/**
 * Unload journal entries live in their own database: Chromium drops a committed write at unload while a read-write
 * transaction of the same database is still open, which a stage in flight in the workbench database would be.
 */
export interface UnloadJournalDatabaseSchema extends DBSchema {
  entries: {
    key: ProjectUnloadJournalKey;
    value: ProjectUnloadJournalEntry;
  };
}

export type UnloadJournalDatabase = IDBPDatabase<UnloadJournalDatabaseSchema>;

const WORKBENCH_DATABASE_NAME_BASE = 'invokeai:v7:webv2:workbench';
const UNLOAD_JOURNAL_DATABASE_NAME_BASE = 'invokeai:v7:webv2:unload-journal';

export const getWorkbenchDatabaseName = (storageSuffix: string): string =>
  `${WORKBENCH_DATABASE_NAME_BASE}${storageSuffix}`;

export const getUnloadJournalDatabaseName = (storageSuffix: string): string =>
  `${UNLOAD_JOURNAL_DATABASE_NAME_BASE}${storageSuffix}`;

const unavailableDatabases = new WeakSet<object>();

export const isWorkbenchDatabaseAvailable = (database: object): boolean => !unavailableDatabases.has(database);

const openDatabaseWithin = <Schema extends DBSchema>(
  name: string,
  version: number,
  label: string,
  timeoutMs: number,
  upgrade: NonNullable<Parameters<typeof openDB<Schema>>[2]>['upgrade']
): Promise<IDBPDatabase<Schema>> => {
  let connection: IDBPDatabase<Schema> | undefined;
  let didSettle = false;
  let rejectOpen: (reason: unknown) => void = () => undefined;
  const markUnavailable = (database: IDBPDatabase<Schema>): void => {
    unavailableDatabases.add(database);
  };
  const opening = openDB<Schema>(name, version, {
    blocked: () => rejectOpen(new DOMException(`Opening the ${label} database was blocked.`, 'InvalidStateError')),
    blocking: () => {
      if (connection) {
        markUnavailable(connection);
        connection.close();
      }
    },
    terminated: () => {
      if (connection) {
        markUnavailable(connection);
      }
    },
    upgrade,
  });
  return new Promise((resolve, reject) => {
    rejectOpen = (reason) => {
      if (!didSettle) {
        didSettle = true;
        reject(reason);
      }
    };
    const timeout = globalThis.setTimeout(
      () => rejectOpen(new DOMException(`Opening the ${label} database timed out.`, 'TimeoutError')),
      timeoutMs
    );
    void opening.then(
      (database) => {
        connection = database;
        globalThis.clearTimeout(timeout);
        if (didSettle) {
          markUnavailable(database);
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

export const openUnloadJournalDatabase = (
  storageSuffix: string,
  { timeoutMs = 1_000 }: { timeoutMs?: number } = {}
): Promise<UnloadJournalDatabase> =>
  openDatabaseWithin<UnloadJournalDatabaseSchema>(
    getUnloadJournalDatabaseName(storageSuffix),
    UNLOAD_JOURNAL_DATABASE_VERSION,
    'unload journal',
    timeoutMs,
    (database, oldVersion) => {
      if (oldVersion < 1) {
        database.createObjectStore(UNLOAD_JOURNAL_STORE, {
          keyPath: ['projectId', 'editorSessionId', 'writerToken', 'generation'],
        });
      }
    }
  );

export const openWorkbenchDatabase = (
  storageSuffix: string,
  { timeoutMs = 1_000 }: { timeoutMs?: number } = {}
): Promise<WorkbenchDatabase> =>
  openDatabaseWithin<WorkbenchDatabaseSchema>(
    getWorkbenchDatabaseName(storageSuffix),
    WORKBENCH_DATABASE_VERSION,
    'workbench',
    timeoutMs,
    (database, oldVersion, _newVersion, transaction) => {
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
    }
  );

export type DeleteWorkbenchDatabaseFinalResult = { kind: 'deleted' | 'unavailable' };
export type DeleteWorkbenchDatabaseResult =
  | DeleteWorkbenchDatabaseFinalResult
  | { completion: Promise<DeleteWorkbenchDatabaseFinalResult>; kind: 'blocked' };

const pendingDeletions = new Map<string, Promise<DeleteWorkbenchDatabaseResult>>();

const combineDeletions = (results: DeleteWorkbenchDatabaseFinalResult[]): DeleteWorkbenchDatabaseFinalResult => ({
  kind: results.every((result) => result.kind === 'deleted') ? 'deleted' : 'unavailable',
});

/** Deletes the account's browser recovery: the workbench database and its unload journal database. */
export const deleteWorkbenchDatabase = async (storageSuffix: string): Promise<DeleteWorkbenchDatabaseResult> => {
  const deletions = await Promise.all([
    deleteDatabaseNamed(getWorkbenchDatabaseName(storageSuffix)),
    deleteDatabaseNamed(getUnloadJournalDatabaseName(storageSuffix)),
  ]);
  if (deletions.every((deletion) => deletion.kind !== 'blocked')) {
    return combineDeletions(deletions as DeleteWorkbenchDatabaseFinalResult[]);
  }
  return {
    completion: Promise.all(
      deletions.map((deletion) => (deletion.kind === 'blocked' ? deletion.completion : deletion))
    ).then(combineDeletions),
    kind: 'blocked',
  };
};

const deleteDatabaseNamed = (name: string): Promise<DeleteWorkbenchDatabaseResult> => {
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

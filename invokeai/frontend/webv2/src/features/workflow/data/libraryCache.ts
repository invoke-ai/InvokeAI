import { queryClient } from '@platform/query/client';
import {
  type AccountScope,
  assertAccountScopeCurrent,
  captureAccountScope,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';

import { getLibraryWorkflowRecord, type WorkflowRecordDTO } from './api';
import { savedWorkflowPickerQueryKeyPrefix } from './savedWorkflowQueries';

/**
 * Library records behind open, replace, fork, download and browse enrichment. The server keeps only a record's
 * current revision, so a cached record is served only to a caller naming the revision its list shows, and only while
 * it is at least that revision; a caller without one always fetches. A record newer than the caller's list means the
 * list is behind, so listeners hear about it once. Every invalidation advances a generation: a read in flight across
 * one stores nothing and retries once rather than answer with what the library held before the change.
 *
 * A matching revision guarantees the authored workflow document only: a visibility change rewrites `is_public` and
 * the `shared` tag without advancing it.
 */

type WorkflowLibraryCacheInvalidationListener = (workflowId?: string) => void;

const invalidationListeners = new Set<WorkflowLibraryCacheInvalidationListener>();

/**
 * Registers a listener fired when the library changed: after every `invalidateWorkflowLibraryCache()` call, and when
 * a read finds a record newer than the list its caller showed.
 */
export const onWorkflowLibraryCacheInvalidated = (listener: WorkflowLibraryCacheInvalidationListener): (() => void) => {
  invalidationListeners.add(listener);
  return () => invalidationListeners.delete(listener);
};

const notifyLibraryChanged = (workflowId?: string): void => {
  void queryClient.invalidateQueries({ queryKey: savedWorkflowPickerQueryKeyPrefix });

  for (const listener of invalidationListeners) {
    listener(workflowId);
  }
};

/** Invalidations kept landing while a record was read; nothing is wrong with the record, so reading again can succeed. */
export class WorkflowLibraryChangedDuringReadError extends Error {
  constructor() {
    super('The workflow library changed while this workflow was loading. Try again.');
    this.name = 'WorkflowLibraryChangedDuringReadError';
  }
}

/** Entries are whole workflow documents; enough for a few pages of browse enrichment. */
const RECORD_CACHE_LIMIT = 50;

interface CachedRecord {
  record: WorkflowRecordDTO;
  /** Whether listeners were already told a list showed an older revision than this record. */
  hasReportedNewerRevision: boolean;
}

/** Insertion order is recency order, so the first key is the least recently used. */
const records = new Map<string, CachedRecord>();
let generation = 0;

const remember = (workflowId: string, entry: CachedRecord): void => {
  records.delete(workflowId);
  records.set(workflowId, entry);

  if (records.size > RECORD_CACHE_LIMIT) {
    const oldest = records.keys().next().value;

    if (oldest !== undefined) {
      records.delete(oldest);
    }
  }
};

const fetchRecord = async (workflowId: string, signal: AbortSignal, owner: AccountScope): Promise<CachedRecord> => {
  for (let attempt = 0; attempt < 2; attempt += 1) {
    const startedIn = generation;
    let record: WorkflowRecordDTO;

    try {
      record = await getLibraryWorkflowRecord(workflowId, signal);
    } catch (error) {
      // A deleted or forbidden record must not keep answering from the cache; an abort or outage says nothing about it.
      if (error instanceof ApiError && (error.status === 403 || error.status === 404)) {
        records.delete(workflowId);
      }
      throw error;
    }

    assertAccountScopeCurrent(owner);
    signal.throwIfAborted();

    if (generation === startedIn) {
      const current = records.get(workflowId);
      // Concurrent reads can finish out of order; the cache never moves back to an older revision.
      const entry =
        current && current.record.revision > record.revision ? current : { hasReportedNewerRevision: false, record };

      remember(workflowId, entry);

      return entry;
    }
  }

  throw new WorkflowLibraryChangedDuringReadError();
};

export interface LibraryWorkflowReadOptions {
  /** The revision the caller's library list shows for this workflow. */
  expectedRevision?: number;
  signal?: AbortSignal;
}

export const getLibraryWorkflowRecordCached = async (
  workflowId: string,
  { expectedRevision, signal: externalSignal }: LibraryWorkflowReadOptions = {}
): Promise<WorkflowRecordDTO> => {
  const owner = captureAccountScope();
  const signal = externalSignal ? AbortSignal.any([externalSignal, owner.signal]) : owner.signal;

  signal.throwIfAborted();

  const cached = records.get(workflowId);
  let entry: CachedRecord;

  if (cached && expectedRevision !== undefined && cached.record.revision >= expectedRevision) {
    entry = cached;
    remember(workflowId, entry);
  } else {
    entry = await fetchRecord(workflowId, signal, owner);
  }

  if (expectedRevision !== undefined && entry.record.revision > expectedRevision && !entry.hasReportedNewerRevision) {
    entry.hasReportedNewerRevision = true;
    notifyLibraryChanged(workflowId);
  }

  return entry.record;
};

/** The stored workflow JSON with the record id stamped in. */
export const getLibraryWorkflowCached = async (
  workflowId: string,
  options?: LibraryWorkflowReadOptions
): Promise<Record<string, unknown>> => {
  const record = await getLibraryWorkflowRecordCached(workflowId, options);

  return { ...record.workflow, id: record.workflow_id };
};

export const invalidateWorkflowLibraryCache = (workflowId?: string): void => {
  generation += 1;
  records.clear();
  notifyLibraryChanged(workflowId);
};

registerAccountOwnedResource({
  clear: invalidateWorkflowLibraryCache,
  name: 'workflow-library',
});

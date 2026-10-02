import { queryClient } from '@platform/query/client';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';

import {
  getLibraryWorkflowRecord,
  listLibraryWorkflows,
  type ListWorkflowsParams,
  type WorkflowLibraryPage,
  type WorkflowRecordDTO,
} from './api';
import { savedWorkflowPickerQueryKeyPrefix } from './savedWorkflowQueries';

/**
 * Serve cached library pages immediately and revalidate; local mutations invalidate ordering and pagination
 * together.
 */

const pageCache = new Map<string, WorkflowLibraryPage>();
const recordCache = new Map<string, WorkflowRecordDTO>();

/** `JSON.stringify` on a sorted copy avoids delimiter collisions between tag values. */
const getTagsKey = (tags: string[] | undefined): string => JSON.stringify([...(tags ?? [])].sort());

const getPageKey = (params: ListWorkflowsParams): string =>
  `${params.category}|${params.page}|${params.perPage ?? 20}|${params.query?.trim() ?? ''}|${getTagsKey(params.tags)}`;

export const getCachedWorkflowPage = (params: ListWorkflowsParams): WorkflowLibraryPage | null =>
  pageCache.get(getPageKey(params)) ?? null;

/** Fetches a page and stores it; callers show `getCachedWorkflowPage` while this resolves. */
export const listLibraryWorkflowsCached = async (params: ListWorkflowsParams): Promise<WorkflowLibraryPage> => {
  const owner = captureAccountScope();
  const signal = params.signal ? AbortSignal.any([params.signal, owner.signal]) : owner.signal;
  const result = await listLibraryWorkflows({ ...params, signal });

  assertAccountScopeCurrent(owner);
  signal.throwIfAborted();
  pageCache.set(getPageKey(params), result);

  return result;
};

/** A record is immutable per revision; a cache hit skips the fetch until a write invalidates it. */
export const getLibraryWorkflowRecordCached = async (
  workflowId: string,
  externalSignal?: AbortSignal
): Promise<WorkflowRecordDTO> => {
  const owner = captureAccountScope();
  const signal = externalSignal ? AbortSignal.any([externalSignal, owner.signal]) : owner.signal;

  signal.throwIfAborted();
  const cached = recordCache.get(workflowId);

  if (cached) {
    assertAccountScopeCurrent(owner);
    return cached;
  }

  const result = await getLibraryWorkflowRecord(workflowId, signal);

  assertAccountScopeCurrent(owner);
  signal.throwIfAborted();
  recordCache.set(workflowId, result);

  return result;
};

/** The stored workflow JSON with the record id stamped in, from the record cache. */
export const getLibraryWorkflowCached = async (
  workflowId: string,
  externalSignal?: AbortSignal
): Promise<Record<string, unknown>> => {
  const record = await getLibraryWorkflowRecordCached(workflowId, externalSignal);

  return { ...record.workflow, id: record.workflow_id };
};

type WorkflowLibraryCacheInvalidationListener = (workflowId?: string) => void;

const invalidationListeners = new Set<WorkflowLibraryCacheInvalidationListener>();

/** Registers a listener fired at the end of every `invalidateWorkflowLibraryCache()` call. */
export const onWorkflowLibraryCacheInvalidated = (listener: WorkflowLibraryCacheInvalidationListener): (() => void) => {
  invalidationListeners.add(listener);
  return () => invalidationListeners.delete(listener);
};

export const invalidateWorkflowLibraryCache = (workflowId?: string): void => {
  pageCache.clear();
  recordCache.clear();
  void queryClient.invalidateQueries({ queryKey: savedWorkflowPickerQueryKeyPrefix });

  for (const listener of invalidationListeners) {
    listener(workflowId);
  }
};

registerAccountOwnedResource({
  clear: invalidateWorkflowLibraryCache,
  name: 'workflow-library',
});

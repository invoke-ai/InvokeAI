import {
  assertAccountScopeCurrent,
  captureAccountScope,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';

import { getLibraryWorkflow, listLibraryWorkflows, type ListWorkflowsParams, type WorkflowLibraryPage } from './api';

/**
 * Session-lived cache in front of the workflow library API. The library
 * dialog opens instantly on cached pages and revalidates in the background;
 * any local mutation (save/delete) invalidates everything since it shifts
 * ordering and pagination.
 */

const pageCache = new Map<string, WorkflowLibraryPage>();
const workflowCache = new Map<string, Record<string, unknown>>();

const getPageKey = (params: ListWorkflowsParams): string =>
  `${params.category}|${params.page}|${params.perPage ?? 20}|${params.query?.trim() ?? ''}`;

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

/** Workflow payloads are immutable per save; cache hits skip the fetch entirely. */
export const getLibraryWorkflowCached = async (
  workflowId: string,
  externalSignal?: AbortSignal
): Promise<Record<string, unknown>> => {
  const owner = captureAccountScope();
  const signal = externalSignal ? AbortSignal.any([externalSignal, owner.signal]) : owner.signal;

  signal.throwIfAborted();
  const cached = workflowCache.get(workflowId);

  if (cached) {
    assertAccountScopeCurrent(owner);
    return cached;
  }

  const result = await getLibraryWorkflow(workflowId, signal);

  assertAccountScopeCurrent(owner);
  signal.throwIfAborted();
  workflowCache.set(workflowId, result);

  return result;
};

const invalidationListeners = new Set<() => void>();

/** Registers a listener fired at the end of every `invalidateWorkflowLibraryCache()` call. */
export const onWorkflowLibraryCacheInvalidated = (listener: () => void): (() => void) => {
  invalidationListeners.add(listener);
  return () => invalidationListeners.delete(listener);
};

export const invalidateWorkflowLibraryCache = (): void => {
  pageCache.clear();
  workflowCache.clear();

  for (const listener of invalidationListeners) {
    listener();
  }
};

registerAccountOwnedResource({
  clear: invalidateWorkflowLibraryCache,
  name: 'workflow-library',
});

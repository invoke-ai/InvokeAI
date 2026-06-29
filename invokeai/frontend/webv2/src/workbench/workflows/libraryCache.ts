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
  const result = await listLibraryWorkflows(params);

  pageCache.set(getPageKey(params), result);

  return result;
};

/** Workflow payloads are immutable per save; cache hits skip the fetch entirely. */
export const getLibraryWorkflowCached = async (workflowId: string): Promise<Record<string, unknown>> => {
  const cached = workflowCache.get(workflowId);

  if (cached) {
    return cached;
  }

  const result = await getLibraryWorkflow(workflowId);

  workflowCache.set(workflowId, result);

  return result;
};

export const invalidateWorkflowLibraryCache = (): void => {
  pageCache.clear();
  workflowCache.clear();
};

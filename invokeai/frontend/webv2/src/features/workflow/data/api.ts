import { apiFetch, apiFetchJson } from '@platform/transport/http';

/**
 * Client for the backend workflow library (`/api/v1/workflows`). The library
 * stores legacy-format WorkflowV3 JSON, so payloads round-trip through
 * `workflowJson.ts` on their way in and out of the project graph document.
 */

export type WorkflowLibraryCategory = 'user' | 'default';

export interface WorkflowLibraryListItem {
  workflow_id: string;
  name: string;
  description: string;
  category: WorkflowLibraryCategory;
  tags?: string | null;
  created_at?: string;
  updated_at?: string;
  opened_at?: string | null;
  thumbnail_url?: string | null;
}

export interface WorkflowLibraryPage {
  items: WorkflowLibraryListItem[];
  page: number;
  pages: number;
  total: number;
}

export interface ListWorkflowsParams {
  category: WorkflowLibraryCategory;
  page: number;
  perPage?: number;
  query?: string;
  signal?: AbortSignal;
}

export const listLibraryWorkflows = ({
  category,
  page,
  perPage = 20,
  query,
  signal,
}: ListWorkflowsParams): Promise<WorkflowLibraryPage> => {
  const params = new URLSearchParams({
    direction: 'DESC',
    order_by: 'updated_at',
    page: String(page),
    per_page: String(perPage),
  });

  params.append('categories', category);

  if (query?.trim()) {
    params.set('query', query.trim());
  }

  return apiFetchJson<WorkflowLibraryPage>(`/api/v1/workflows/?${params.toString()}`, { signal });
};

interface WorkflowRecordDTO {
  workflow_id: string;
  name: string;
  workflow: Record<string, unknown>;
}

/** Returns the stored workflow JSON, with the record id stamped in. */
export const getLibraryWorkflow = async (
  workflowId: string,
  signal?: AbortSignal
): Promise<Record<string, unknown>> => {
  const record = await apiFetchJson<WorkflowRecordDTO>(`/api/v1/workflows/i/${encodeURIComponent(workflowId)}`, {
    signal,
  });

  return { ...record.workflow, id: record.workflow_id };
};

export const createLibraryWorkflow = async (
  workflow: Record<string, unknown>,
  signal?: AbortSignal
): Promise<string> => {
  const { id: _id, ...workflowWithoutId } = workflow;
  const record = await apiFetchJson<WorkflowRecordDTO>('/api/v1/workflows/', {
    body: JSON.stringify({ workflow: workflowWithoutId }),
    method: 'POST',
    signal,
  });

  return record.workflow_id;
};

export const updateLibraryWorkflow = async (
  workflowId: string,
  workflow: Record<string, unknown>,
  signal?: AbortSignal
): Promise<void> => {
  await apiFetchJson(`/api/v1/workflows/i/${encodeURIComponent(workflowId)}`, {
    body: JSON.stringify({ workflow: { ...workflow, id: workflowId } }),
    method: 'PATCH',
    signal,
  });
};

export const deleteLibraryWorkflow = async (workflowId: string, signal?: AbortSignal): Promise<void> => {
  await apiFetch(`/api/v1/workflows/i/${encodeURIComponent(workflowId)}`, { method: 'DELETE', signal });
};

export const touchLibraryWorkflowOpenedAt = async (workflowId: string, signal?: AbortSignal): Promise<void> => {
  await apiFetch(`/api/v1/workflows/i/${encodeURIComponent(workflowId)}/opened_at`, { method: 'PUT', signal });
};

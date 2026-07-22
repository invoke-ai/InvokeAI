import { apiFetchJson } from '@platform/transport/http';

interface WorkflowLibraryListItemDTO {
  workflow_id: string;
  name: string;
  description: string;
  category: 'user' | 'default';
  tags?: string | null;
}

interface WorkflowLibraryPageDTO {
  items: WorkflowLibraryListItemDTO[];
  page: number;
  pages: number;
  total: number;
}

export interface WorkflowLibraryItem {
  workflowId: string;
  name: string;
  description: string;
  category: 'user' | 'default';
  tags?: string;
}

export interface WorkflowLibraryPage {
  items: WorkflowLibraryItem[];
  page: number;
  pages: number;
  total: number;
}

const mapItem = (item: WorkflowLibraryListItemDTO): WorkflowLibraryItem => ({
  category: item.category,
  description: item.description,
  name: item.name,
  tags: item.tags ?? undefined,
  workflowId: item.workflow_id,
});

/** Cancellable workflow-library read used by deferred cross-feature search surfaces. */
export const listLibraryWorkflows = async ({
  category,
  page,
  perPage = 20,
  query,
  signal,
}: {
  category: 'user' | 'default';
  page: number;
  perPage?: number;
  query?: string;
  signal?: AbortSignal;
}): Promise<WorkflowLibraryPage> => {
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

  const body = await apiFetchJson<WorkflowLibraryPageDTO>(`/api/v1/workflows/?${params.toString()}`, { signal });

  return { items: body.items.map(mapItem), page: body.page, pages: body.pages, total: body.total };
};

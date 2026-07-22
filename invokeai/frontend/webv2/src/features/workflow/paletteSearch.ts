import { apiFetchJson } from '@platform/transport/http';

interface WorkflowLibraryListItem {
  workflow_id: string;
  name: string;
  description: string;
  category: 'user' | 'default';
  tags?: string | null;
}

interface WorkflowLibraryPage {
  items: WorkflowLibraryListItem[];
  page: number;
  pages: number;
  total: number;
}

/** Cancellable workflow-library read used by deferred cross-feature search surfaces. */
export const listLibraryWorkflows = ({
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

  return apiFetchJson<WorkflowLibraryPage>(`/api/v1/workflows/?${params.toString()}`, { signal });
};

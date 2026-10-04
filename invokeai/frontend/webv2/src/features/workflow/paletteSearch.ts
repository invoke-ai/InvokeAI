import { listLibraryWorkflows as listLibraryWorkflowsApi, type ListWorkflowsParams } from './data/api';

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

/** Cancellable workflow-library read used by deferred cross-feature search surfaces. */
export const listLibraryWorkflows = async (params: ListWorkflowsParams): Promise<WorkflowLibraryPage> => {
  const body = await listLibraryWorkflowsApi(params);

  return {
    items: body.items.map((item) => ({
      category: item.category,
      description: item.description,
      name: item.name,
      tags: item.tags ?? undefined,
      workflowId: item.workflow_id,
    })),
    page: body.page,
    pages: body.pages,
    total: body.total,
  };
};

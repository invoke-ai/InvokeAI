import type { ComboboxOption } from '@platform/ui';

import type { ListWorkflowsParams, WorkflowCallCompatibility, WorkflowLibraryListItem, WorkflowRecordDTO } from './api';

export const MISSING_WORKFLOW_OPTION_VALUE = '__missing_workflow__';

export type SavedWorkflowSelectionState =
  | { status: 'unselected' }
  | { status: 'selected'; workflow: WorkflowLibraryListItem }
  | { status: 'missing'; workflowId: string };

export const buildSavedWorkflowOptions = (workflows: WorkflowLibraryListItem[]): ComboboxOption[] =>
  workflows.map((workflow) => {
    const searchText = [workflow.description, workflow.tags]
      .filter((value): value is string => Boolean(value))
      .join(' ');

    return {
      disabled: workflow.call_saved_workflow_compatibility?.is_callable === false,
      label: workflow.name,
      ...(searchText ? { searchText } : {}),
      value: workflow.workflow_id,
    };
  });

const baseSavedWorkflowPickerQuery = {
  page: 0,
  perPage: undefined,
  orderBy: 'name' as const,
  direction: 'ASC' as const,
  callable: true,
  tags: [],
} satisfies Partial<ListWorkflowsParams>;

export const getSavedWorkflowPickerOwnedQuery = (query = ''): ListWorkflowsParams => ({
  ...baseSavedWorkflowPickerQuery,
  categories: ['user', 'default'],
  isPublic: undefined,
  query,
});

export const getSavedWorkflowPickerSharedQuery = (query = ''): ListWorkflowsParams => ({
  ...baseSavedWorkflowPickerQuery,
  categories: ['user'],
  isPublic: true,
  query,
});

export const mergeSavedWorkflowPickerItems = (
  ...workflowLists: WorkflowLibraryListItem[][]
): WorkflowLibraryListItem[] => {
  const workflowsById = new Map<string, WorkflowLibraryListItem>();

  for (const workflows of workflowLists) {
    for (const workflow of workflows) {
      if (!workflowsById.has(workflow.workflow_id)) {
        workflowsById.set(workflow.workflow_id, workflow);
      }
    }
  }

  return [...workflowsById.values()];
};

export const shouldFetchNextSavedWorkflowPickerPage = ({
  hasNextPage,
  isFetching,
}: {
  hasNextPage: boolean;
  isFetching: boolean;
}): boolean => hasNextPage && !isFetching;

export const getSavedWorkflowSelectionState = (
  workflows: WorkflowLibraryListItem[],
  workflowId: string,
  selectedWorkflow?: WorkflowLibraryListItem
): SavedWorkflowSelectionState => {
  if (!workflowId) {
    return { status: 'unselected' };
  }

  const workflow = workflows.find((candidate) => candidate.workflow_id === workflowId);

  if (workflow) {
    return { status: 'selected', workflow };
  }

  if (selectedWorkflow?.workflow_id === workflowId) {
    return { status: 'selected', workflow: selectedWorkflow };
  }

  return { status: 'missing', workflowId };
};

export const getSavedWorkflowSelectionOption = (selectionState: SavedWorkflowSelectionState): ComboboxOption | null => {
  if (selectionState.status === 'unselected') {
    return null;
  }

  if (selectionState.status === 'selected') {
    return { label: selectionState.workflow.name, value: selectionState.workflow.workflow_id };
  }

  return { label: MISSING_WORKFLOW_OPTION_VALUE, value: MISSING_WORKFLOW_OPTION_VALUE };
};

export interface SavedWorkflowDisplayState {
  selection: 'unselected' | 'selected' | 'missing';
  statusLabel: 'choose' | 'missing' | null;
  badges: Array<'unsupported' | 'default' | 'shared'>;
  compatibility: WorkflowCallCompatibility | null;
}

export const getSavedWorkflowDisplayState = (
  selectionState: SavedWorkflowSelectionState
): SavedWorkflowDisplayState => {
  if (selectionState.status === 'unselected') {
    return { badges: [], compatibility: null, selection: 'unselected', statusLabel: 'choose' };
  }

  if (selectionState.status === 'missing') {
    return { badges: [], compatibility: null, selection: 'missing', statusLabel: 'missing' };
  }

  const { workflow } = selectionState;
  const badges: SavedWorkflowDisplayState['badges'] = [];

  if (workflow.call_saved_workflow_compatibility?.is_callable === false) {
    badges.push('unsupported');
  }

  if (workflow.category === 'default') {
    badges.push('default');
  } else if (workflow.is_public === true) {
    badges.push('shared');
  }

  return {
    badges,
    compatibility: workflow.call_saved_workflow_compatibility ?? null,
    selection: 'selected',
    statusLabel: null,
  };
};

/** Maps the detail endpoint's full record into the same shape as list results. */
export const getSavedWorkflowListItemFromRecord = (record: WorkflowRecordDTO): WorkflowLibraryListItem => ({
  call_saved_workflow_compatibility: record.call_saved_workflow_compatibility,
  category:
    record.category === 'default' ||
    (typeof record.workflow.meta === 'object' &&
      record.workflow.meta !== null &&
      (record.workflow.meta as { category?: unknown }).category === 'default')
      ? 'default'
      : 'user',
  created_at: record.created_at,
  description: typeof record.workflow.description === 'string' ? record.workflow.description : record.description,
  is_public: record.is_public,
  name: record.name,
  opened_at: record.opened_at,
  revision: record.revision,
  tags: typeof record.workflow.tags === 'string' ? record.workflow.tags : record.tags,
  thumbnail_url: record.thumbnail_url,
  updated_at: record.updated_at,
  user_id: record.user_id,
  workflow_id: record.workflow_id,
});

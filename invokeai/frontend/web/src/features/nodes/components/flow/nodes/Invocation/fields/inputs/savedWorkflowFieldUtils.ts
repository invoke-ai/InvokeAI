import type { ComboboxOption } from '@invoke-ai/ui-library';
import type { WorkflowRecordListItemWithThumbnailDTO } from 'services/api/types';

export const MISSING_WORKFLOW_OPTION_VALUE = '__missing_workflow__';
export const MISSING_SELECTION_LABEL = 'Missing or inaccessible workflow';
export const EMPTY_SELECTION_LABEL = 'Choose a workflow';

type SavedWorkflowSelectionState =
  | { status: 'unselected' }
  | { status: 'selected'; workflow: WorkflowRecordListItemWithThumbnailDTO }
  | { status: 'missing'; workflowId: string };

export const buildSavedWorkflowOptions = (workflows: WorkflowRecordListItemWithThumbnailDTO[]): ComboboxOption[] => {
  return workflows.map((workflow) => ({
    label: workflow.name,
    value: workflow.workflow_id,
  }));
};

export const getSavedWorkflowSelectionState = (
  workflows: WorkflowRecordListItemWithThumbnailDTO[],
  workflowId: string
): SavedWorkflowSelectionState => {
  if (!workflowId) {
    return { status: 'unselected' };
  }

  const workflow = workflows.find((workflow) => workflow.workflow_id === workflowId);
  if (workflow) {
    return { status: 'selected', workflow };
  }

  return { status: 'missing', workflowId };
};

export const getSavedWorkflowSelectionOption = (selectionState: SavedWorkflowSelectionState): ComboboxOption | null => {
  if (selectionState.status === 'unselected') {
    return null;
  }

  if (selectionState.status === 'selected') {
    return {
      label: selectionState.workflow.name,
      value: selectionState.workflow.workflow_id,
    };
  }

  return {
    label: MISSING_SELECTION_LABEL,
    value: MISSING_WORKFLOW_OPTION_VALUE,
  };
};

export const getSavedWorkflowSelectionStatusLabel = (selectionState: SavedWorkflowSelectionState): string | null => {
  if (selectionState.status === 'selected') {
    return null;
  }

  return selectionState.status === 'missing' ? MISSING_SELECTION_LABEL : EMPTY_SELECTION_LABEL;
};

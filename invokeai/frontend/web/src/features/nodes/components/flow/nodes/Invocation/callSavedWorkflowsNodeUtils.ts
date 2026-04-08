import type { ComboboxOption } from '@invoke-ai/ui-library';
import type { WorkflowRecordListItemWithThumbnailDTO } from 'services/api/types';

export const MISSING_WORKFLOW_OPTION_VALUE = '__missing_workflow__';

export type SavedWorkflowSelectionState =
  | { status: 'unselected' }
  | { status: 'selected'; workflow: WorkflowRecordListItemWithThumbnailDTO }
  | { status: 'missing'; workflowId: string };

export const buildSavedWorkflowOptions = (
  workflows: WorkflowRecordListItemWithThumbnailDTO[]
): ComboboxOption[] => {
  return workflows.map((workflow) => ({
    label: workflow.name,
    value: workflow.workflow_id,
  }));
};

export const getSelectedWorkflow = (
  workflows: WorkflowRecordListItemWithThumbnailDTO[],
  workflowId: string
): WorkflowRecordListItemWithThumbnailDTO | null => {
  const selectionState = getSavedWorkflowSelectionState(workflows, workflowId);

  if (selectionState.status !== 'selected') {
    return null;
  }

  return selectionState.workflow;
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

export const getSelectedWorkflowOption = (
  workflows: WorkflowRecordListItemWithThumbnailDTO[],
  workflowId: string,
  missingLabel: string
): ComboboxOption | null => {
  const selectionState = getSavedWorkflowSelectionState(workflows, workflowId);

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
    label: missingLabel,
    value: MISSING_WORKFLOW_OPTION_VALUE,
  };
};

import type { ComboboxOption } from '@invoke-ai/ui-library';
import { getWorkflowCallCompatibilityState } from 'features/workflowLibrary/util/workflowCallCompatibility';
import type { WorkflowRecordListItemWithThumbnailDTO } from 'services/api/types';

export const MISSING_WORKFLOW_OPTION_VALUE = '__missing_workflow__';
export type SavedWorkflowBadge = 'unsupported' | 'default' | 'shared';

type SavedWorkflowSelectionState =
  | { status: 'unselected' }
  | { status: 'selected'; workflow: WorkflowRecordListItemWithThumbnailDTO }
  | { status: 'missing'; workflowId: string };

export const buildSavedWorkflowOptions = (workflows: WorkflowRecordListItemWithThumbnailDTO[]): ComboboxOption[] => {
  return workflows.map((workflow) => ({
    label: workflow.name,
    value: workflow.workflow_id,
    isDisabled: workflow.call_saved_workflow_compatibility?.is_callable === false,
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
    label: MISSING_WORKFLOW_OPTION_VALUE,
    value: MISSING_WORKFLOW_OPTION_VALUE,
  };
};

export type SavedWorkflowDisplayState =
  | {
      selection: 'unselected' | 'missing';
      statusLabelKey: 'nodes.savedWorkflowChoose' | 'nodes.savedWorkflowMissing';
      badges: SavedWorkflowBadge[];
      compatibilityMessage: null;
    }
  | {
      selection: 'selected';
      statusLabelKey: null;
      badges: SavedWorkflowBadge[];
      compatibilityMessage: string | null;
    };

export const getSavedWorkflowDisplayState = (
  selectionState: SavedWorkflowSelectionState
): SavedWorkflowDisplayState => {
  if (selectionState.status === 'unselected') {
    return {
      selection: 'unselected',
      statusLabelKey: 'nodes.savedWorkflowChoose',
      badges: [],
      compatibilityMessage: null,
    };
  }

  if (selectionState.status === 'missing') {
    return {
      selection: 'missing',
      statusLabelKey: 'nodes.savedWorkflowMissing',
      badges: [],
      compatibilityMessage: null,
    };
  }

  const compatibilityState = getWorkflowCallCompatibilityState(selectionState.workflow);
  const badges: SavedWorkflowBadge[] = [];
  if (compatibilityState.isUnsupported) {
    badges.push('unsupported');
  }
  if (selectionState.workflow.category === 'default') {
    badges.push('default');
  } else if (selectionState.workflow.is_public) {
    badges.push('shared');
  }

  return {
    selection: 'selected',
    statusLabelKey: null,
    badges,
    compatibilityMessage: compatibilityState.message,
  };
};

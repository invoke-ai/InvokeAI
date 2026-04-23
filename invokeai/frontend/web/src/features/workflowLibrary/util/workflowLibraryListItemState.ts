import { getWorkflowCallCompatibilityState } from 'features/workflowLibrary/util/workflowCallCompatibility';
import type { WorkflowRecordListItemWithThumbnailDTO } from 'services/api/types';

export type WorkflowLibraryListItemState = {
  showUnsupportedBadge: boolean;
  unsupportedMessage: string | null;
  showSharedBadge: boolean;
  showDefaultIcon: boolean;
};

export const getWorkflowLibraryListItemState = (
  workflow: Pick<WorkflowRecordListItemWithThumbnailDTO, 'category' | 'is_public' | 'call_saved_workflow_compatibility'>
): WorkflowLibraryListItemState => {
  const compatibilityState = getWorkflowCallCompatibilityState(workflow);

  return {
    showUnsupportedBadge: compatibilityState.isUnsupported,
    unsupportedMessage: compatibilityState.message,
    showSharedBadge: workflow.is_public && workflow.category !== 'default',
    showDefaultIcon: workflow.category === 'default',
  };
};

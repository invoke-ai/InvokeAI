import {
  getWorkflowCallCompatibilityState,
  type WorkflowCallCompatibilityMessageKey,
} from 'features/workflowLibrary/util/workflowCallCompatibility';
import type { WorkflowRecordListItemWithThumbnailDTO } from 'services/api/types';

type WorkflowLibraryListItemState = {
  showUnsupportedBadge: boolean;
  unsupportedMessageKey: WorkflowCallCompatibilityMessageKey | null;
  showSharedBadge: boolean;
  showDefaultIcon: boolean;
};

export const getWorkflowLibraryListItemState = (
  workflow: Pick<WorkflowRecordListItemWithThumbnailDTO, 'category' | 'is_public' | 'call_saved_workflow_compatibility'>
): WorkflowLibraryListItemState => {
  const compatibilityState = getWorkflowCallCompatibilityState(workflow);

  return {
    showUnsupportedBadge: compatibilityState.isUnsupported,
    unsupportedMessageKey: compatibilityState.messageKey,
    showSharedBadge: workflow.is_public && workflow.category !== 'default',
    showDefaultIcon: workflow.category === 'default',
  };
};

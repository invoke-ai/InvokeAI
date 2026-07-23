import type { WorkflowRecordListItemWithThumbnailDTO } from 'services/api/types';

type WorkflowLibraryListItemState = {
  showCallableBadge: boolean;
  showSharedBadge: boolean;
  showDefaultIcon: boolean;
};

export const getWorkflowLibraryListItemState = (
  workflow: Pick<WorkflowRecordListItemWithThumbnailDTO, 'category' | 'is_public' | 'call_saved_workflow_compatibility'>
): WorkflowLibraryListItemState => {
  return {
    showCallableBadge: workflow.call_saved_workflow_compatibility?.is_callable === true,
    showSharedBadge: workflow.is_public && workflow.category !== 'default',
    showDefaultIcon: workflow.category === 'default',
  };
};

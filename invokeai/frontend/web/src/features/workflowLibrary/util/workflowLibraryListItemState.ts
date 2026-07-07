import type { WorkflowRecordListItemWithThumbnailDTO } from 'services/api/types';

type WorkflowLibraryListItemState = {
  showUnsupportedBadge: boolean;
  unsupportedMessageKey: null;
  showSharedBadge: boolean;
  showDefaultIcon: boolean;
};

export const getWorkflowLibraryListItemState = (
  workflow: Pick<WorkflowRecordListItemWithThumbnailDTO, 'category' | 'is_public' | 'call_saved_workflow_compatibility'>
): WorkflowLibraryListItemState => {
  return {
    showUnsupportedBadge: false,
    unsupportedMessageKey: null,
    showSharedBadge: workflow.is_public && workflow.category !== 'default',
    showDefaultIcon: workflow.category === 'default',
  };
};

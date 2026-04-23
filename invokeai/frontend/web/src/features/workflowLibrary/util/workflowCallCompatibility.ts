import type { WorkflowRecordListItemWithThumbnailDTO } from 'services/api/types';

export type WorkflowCallCompatibilityState =
  | {
      isUnsupported: false;
      message: null;
    }
  | {
      isUnsupported: true;
      message: string | null;
    };

export const getWorkflowCallCompatibilityState = (
  workflow: Pick<WorkflowRecordListItemWithThumbnailDTO, 'call_saved_workflow_compatibility'>
): WorkflowCallCompatibilityState => {
  const compatibility = workflow.call_saved_workflow_compatibility;
  if (!compatibility || compatibility.is_callable) {
    return {
      isUnsupported: false,
      message: null,
    };
  }

  return {
    isUnsupported: true,
    message: compatibility.message ?? null,
  };
};

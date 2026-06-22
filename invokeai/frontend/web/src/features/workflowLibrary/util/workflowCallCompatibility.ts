import type { S, WorkflowRecordListItemWithThumbnailDTO } from 'services/api/types';

const compatibilityMessageKeys = {
  ok: 'workflows.savedWorkflowCompatibility.unknown',
  missing_workflow_return: 'workflows.savedWorkflowCompatibility.missingWorkflowReturn',
  multiple_workflow_return: 'workflows.savedWorkflowCompatibility.multipleWorkflowReturn',
  unsupported_node: 'workflows.savedWorkflowCompatibility.unsupportedNode',
  unsupported_batch_input: 'workflows.savedWorkflowCompatibility.unsupportedBatchInput',
  invalid_graph: 'workflows.savedWorkflowCompatibility.invalidGraph',
  invalid_inputs: 'workflows.savedWorkflowCompatibility.invalidInputs',
  exceeds_capacity: 'workflows.savedWorkflowCompatibility.exceedsCapacity',
  unknown: 'workflows.savedWorkflowCompatibility.unknown',
} as const satisfies Record<S['WorkflowCallCompatibilityReason'], string>;

export type WorkflowCallCompatibilityMessageKey =
  (typeof compatibilityMessageKeys)[keyof typeof compatibilityMessageKeys];

type WorkflowCallCompatibilityState =
  | {
      isUnsupported: false;
      messageKey: null;
    }
  | {
      isUnsupported: true;
      messageKey: WorkflowCallCompatibilityMessageKey;
    };

export const getWorkflowCallCompatibilityState = (
  workflow: Pick<WorkflowRecordListItemWithThumbnailDTO, 'call_saved_workflow_compatibility'>
): WorkflowCallCompatibilityState => {
  const compatibility = workflow.call_saved_workflow_compatibility;
  if (!compatibility || compatibility.is_callable) {
    return {
      isUnsupported: false,
      messageKey: null,
    };
  }

  return {
    isUnsupported: true,
    messageKey: compatibilityMessageKeys[compatibility.reason],
  };
};

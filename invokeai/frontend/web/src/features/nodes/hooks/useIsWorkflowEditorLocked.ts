import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { $isInPublishFlow, useIsValidationRunInProgress } from 'features/nodes/components/sidePanel/workflow/publish';
import { selectWorkflowIsPublished } from 'features/nodes/store/workflowSlice';

export const useIsWorkflowEditorLocked = () => {
  const isInPublishFlow = useStore($isInPublishFlow);
  const isPublished = useAppSelector(selectWorkflowIsPublished);
  const isValidationRunInProgress = useIsValidationRunInProgress();

  const isLocked = isInPublishFlow || isPublished || isValidationRunInProgress;
  return isLocked;
};

import { useStore } from '@nanostores/react';
import {
  $isInPublishFlow,
  useIsValidationRunInProgress,
  useIsWorkflowPublished,
} from 'features/nodes/components/sidePanel/workflow/publish';

export const useIsWorkflowEditorLocked = () => {
  const isInPublishFlow = useStore($isInPublishFlow);
  const isPublished = useIsWorkflowPublished();
  const isValidationRunInProgress = useIsValidationRunInProgress();

  const isLocked = isInPublishFlow || isPublished || isValidationRunInProgress;
  return isLocked;
};

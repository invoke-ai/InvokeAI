import { useStore } from '@nanostores/react';
import { $hasTemplates } from 'features/nodes/store/nodesSlice';
import { useLoadWorkflowWithDialog } from 'features/workflowLibrary/components/LoadWorkflowConfirmationAlertDialog';
import { useCallback, useMemo } from 'react';
import type { ImageDTO } from 'services/api/types';

export const useLoadWorkflow = (imageDTO: ImageDTO) => {
  const hasTemplates = useStore($hasTemplates);

  const loadWorkflowWithDialog = useLoadWorkflowWithDialog();

  const isEnabled = useMemo(() => {
    if (!imageDTO.has_workflow) {
      return false;
    }
    if (!hasTemplates) {
      return false;
    }
    return true;
  }, [hasTemplates, imageDTO]);

  const load = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    if (!isEnabled) {
      return;
    }

    loadWorkflowWithDialog({ type: 'image', data: imageDTO.image_name });
  }, [imageDTO, isEnabled, loadWorkflowWithDialog]);

  return { load, isEnabled };
};

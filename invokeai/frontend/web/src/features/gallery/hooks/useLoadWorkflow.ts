import { useStore } from '@nanostores/react';
import { useIsRegionFocused } from 'common/hooks/focus';
import { $hasTemplates } from 'features/nodes/store/nodesSlice';
import { useLoadWorkflowWithDialog } from 'features/workflowLibrary/components/LoadWorkflowConfirmationAlertDialog';
import { useCallback, useMemo } from 'react';
import type { ImageDTO } from 'services/api/types';

export const useLoadWorkflow = (imageDTO?: ImageDTO | null) => {
  const hasTemplates = useStore($hasTemplates);
  const isGalleryFocused = useIsRegionFocused('gallery');
  const isViewerFocused = useIsRegionFocused('viewer');

  const loadWorkflowWithDialog = useLoadWorkflowWithDialog();

  const isEnabled = useMemo(() => {
    if (!imageDTO) {
      return false;
    }
    if (!imageDTO.has_workflow) {
      return false;
    }
    if (!hasTemplates) {
      return false;
    }
    if (!isGalleryFocused && !isViewerFocused) {
      return false;
    }
    return true;
  }, [hasTemplates, imageDTO, isGalleryFocused, isViewerFocused]);

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

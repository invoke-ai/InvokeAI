import { useAppDispatch } from 'app/store/storeHooks';
import { settingsSendToCanvasChanged } from 'features/controlLayers/store/canvasSettingsSlice';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { $isMenuOpen } from 'features/stylePresets/store/isMenuOpen';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { useWorkflowLibraryModal } from 'features/workflowLibrary/store/isWorkflowLibraryModalOpen';
import { useCallback, useState } from 'react';

export type StudioDestination =
  | 'generation'
  | 'canvas'
  | 'workflows'
  | 'upscaling'
  | 'viewAllWorkflows'
  | 'viewAllStylePresets';

export const useHandleStudioDestination = () => {
  const dispatch = useAppDispatch();
  const { open: imageViewerOpen, close: imageViewerClose } = useImageViewer();
  const [initialized, setInitialized] = useState(false);

  const workflowLibraryModal = useWorkflowLibraryModal();

  const handleStudioDestination = useCallback(
    (destination: StudioDestination) => {
      if (initialized) {
        return;
      }
      switch (destination) {
        case 'generation':
          dispatch(setActiveTab('canvas'));
          dispatch(settingsSendToCanvasChanged(false));
          imageViewerOpen();
          break;
        case 'canvas':
          dispatch(setActiveTab('canvas'));
          dispatch(settingsSendToCanvasChanged(true));
          imageViewerClose();
          break;
        case 'workflows':
          dispatch(setActiveTab('workflows'));
          break;
        case 'upscaling':
          dispatch(setActiveTab('upscaling'));
          break;
        case 'viewAllWorkflows':
          dispatch(setActiveTab('workflows'));
          workflowLibraryModal.setTrue();
          break;
        case 'viewAllStylePresets':
          dispatch(setActiveTab('canvas'));
          $isMenuOpen.set(true);
          break;
        default:
          dispatch(setActiveTab('canvas'));
          break;
      }
      setInitialized(true);
    },
    [dispatch, imageViewerOpen, imageViewerClose, workflowLibraryModal, initialized]
  );

  return handleStudioDestination;
};

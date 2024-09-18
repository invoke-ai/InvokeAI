import { useAppDispatch } from 'app/store/storeHooks';
import { settingsSendToCanvasChanged } from 'features/controlLayers/store/canvasSettingsSlice';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { $isMenuOpen } from 'features/stylePresets/store/isMenuOpen';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { $isWorkflowLibraryModalOpen } from 'features/workflowLibrary/store/isWorkflowLibraryModalOpen';
import { useCallback } from 'react';

export type StudioDestination =
  | 'generation'
  | 'canvas'
  | 'workflows'
  | 'upscaling'
  | 'viewAllWorkflows'
  | 'viewAllStylePresets';

export const useHandleStudioDestination = () => {
  const dispatch = useAppDispatch();
  const imageViewer = useImageViewer();

  const handleStudioDestination = useCallback(
    (destination: StudioDestination) => {
      switch (destination) {
        case 'generation':
          dispatch(setActiveTab('canvas'));
          dispatch(settingsSendToCanvasChanged(false));
          imageViewer.open();
          break;
        case 'canvas':
          dispatch(setActiveTab('canvas'));
          dispatch(settingsSendToCanvasChanged(true));
          imageViewer.close();
          break;
        case 'workflows':
          dispatch(setActiveTab('workflows'));
          break;
        case 'upscaling':
          dispatch(setActiveTab('upscaling'));
          break;
        case 'viewAllWorkflows':
          dispatch(setActiveTab('workflows'));
          $isWorkflowLibraryModalOpen.set(true);
          break;
        case 'viewAllStylePresets':
          dispatch(setActiveTab('canvas'));
          $isMenuOpen.set(true);
          break;
        default:
          dispatch(setActiveTab('canvas'));
          break;
      }
    },
    [dispatch, imageViewer]
  );

  return handleStudioDestination;
};

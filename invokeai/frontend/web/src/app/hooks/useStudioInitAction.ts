import { useAppStore } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { withResultAsync } from 'common/util/result';
import { canvasReset } from 'features/controlLayers/store/actions';
import { settingsSendToCanvasChanged } from 'features/controlLayers/store/canvasSettingsSlice';
import { rasterLayerAdded } from 'features/controlLayers/store/canvasSlice';
import type { CanvasRasterLayerState } from 'features/controlLayers/store/types';
import { imageDTOToImageObject } from 'features/controlLayers/store/util';
import { $imageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { sentImageToCanvas } from 'features/gallery/store/actions';
import { parseAndRecallAllMetadata } from 'features/metadata/util/handlers';
import { $isStylePresetsMenuOpen, activeStylePresetIdChanged } from 'features/stylePresets/store/stylePresetSlice';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { useGetAndLoadLibraryWorkflow } from 'features/workflowLibrary/hooks/useGetAndLoadLibraryWorkflow';
import { $workflowLibraryModal } from 'features/workflowLibrary/store/isWorkflowLibraryModalOpen';
import { useCallback, useEffect, useRef } from 'react';
import { getImageDTO, getImageMetadata } from 'services/api/endpoints/images';

type _StudioInitAction<T extends string, U> = { type: T; data: U };

type LoadWorkflowAction = _StudioInitAction<'loadWorkflow', { workflowId: string }>;
type SelectStylePresetAction = _StudioInitAction<'selectStylePreset', { stylePresetId: string }>;
type SendToCanvasAction = _StudioInitAction<'sendToCanvas', { imageName: string }>;
type UseAllParametersAction = _StudioInitAction<'useAllParameters', { imageName: string }>;
type StudioDestinationAction = _StudioInitAction<
  'goToDestination',
  { destination: 'generation' | 'canvas' | 'workflows' | 'upscaling' | 'viewAllWorkflows' | 'viewAllStylePresets' }
>;

export type StudioInitAction =
  | LoadWorkflowAction
  | SelectStylePresetAction
  | SendToCanvasAction
  | UseAllParametersAction
  | StudioDestinationAction;

export const useStudioInitAction = (action?: StudioInitAction) => {
  useAssertSingleton('useStudioInitAction');
  const didUseRef = useRef(false);
  const store = useAppStore();
  const { getAndLoadWorkflow } = useGetAndLoadLibraryWorkflow();

  /**
   * Sends the image to the canvas.
   */
  const handleSendToCanvas = useCallback(
    async (imageName: string) => {
      // Try to the image DTO
      const getImageDTOResult = await withResultAsync(() => getImageDTO(imageName));
      if (getImageDTOResult.isErr()) {
        // TODO(psyche): show a toast
        return;
      }
      const imageDTO = getImageDTOResult.value;
      const imageObject = imageDTOToImageObject(imageDTO);
      const overrides: Partial<CanvasRasterLayerState> = {
        objects: [imageObject],
      };
      store.dispatch(canvasReset());
      store.dispatch(rasterLayerAdded({ overrides, isSelected: true }));
      store.dispatch(settingsSendToCanvasChanged(true));
      store.dispatch(setActiveTab('canvas'));
      store.dispatch(sentImageToCanvas());
      $imageViewer.set(false);
      // TODO(psyche): show a toast
    },
    [store]
  );

  const handleUseAllMetadata = useCallback(
    async (imageName: string) => {
      const getImageMetadataResult = await withResultAsync(() => getImageMetadata(imageName));
      if (getImageMetadataResult.isErr()) {
        // TODO(psyche): show a toast
        return;
      }
      const metadata = getImageMetadataResult.value;
      parseAndRecallAllMetadata(metadata, true); // this shows a toast
      store.dispatch(setActiveTab('canvas'));
    },
    [store]
  );

  const handleLoadWorkflow = useCallback(
    (workflowId: string) => {
      getAndLoadWorkflow(workflowId);
      store.dispatch(setActiveTab('workflows'));
      // TODO(psyche): show a toast
    },
    [getAndLoadWorkflow, store]
  );

  const handleSelectStylePreset = useCallback(
    (stylePresetId: string) => {
      store.dispatch(activeStylePresetIdChanged(stylePresetId));
      store.dispatch(setActiveTab('canvas'));
      // TODO(psyche): show a toast
    },
    [store]
  );

  const handleGoToDestination = useCallback(
    (destination: StudioDestinationAction['data']['destination']) => {
      switch (destination) {
        case 'generation':
          store.dispatch(setActiveTab('canvas'));
          store.dispatch(settingsSendToCanvasChanged(false));
          $imageViewer.set(true);
          break;
        case 'canvas':
          store.dispatch(setActiveTab('canvas'));
          store.dispatch(settingsSendToCanvasChanged(true));
          $imageViewer.set(false);
          break;
        case 'workflows':
          store.dispatch(setActiveTab('workflows'));
          break;
        case 'upscaling':
          store.dispatch(setActiveTab('upscaling'));
          break;
        case 'viewAllWorkflows':
          store.dispatch(setActiveTab('workflows'));
          $workflowLibraryModal.set(true);
          break;
        case 'viewAllStylePresets':
          store.dispatch(setActiveTab('canvas'));
          $isStylePresetsMenuOpen.set(true);
          break;
      }
      // TODO(psyche): show a toast?
    },
    [store]
  );

  useEffect(() => {
    if (didUseRef.current || !action) {
      return;
    }

    didUseRef.current = true;

    switch (action.type) {
      case 'loadWorkflow':
        handleLoadWorkflow(action.data.workflowId);
        break;
      case 'selectStylePreset':
        handleSelectStylePreset(action.data.stylePresetId);
        break;
      case 'sendToCanvas':
        handleSendToCanvas(action.data.imageName);
        break;
      case 'useAllParameters':
        handleUseAllMetadata(action.data.imageName);
        break;
      case 'goToDestination':
        handleGoToDestination(action.data.destination);
        break;
    }
  }, [
    handleSendToCanvas,
    handleUseAllMetadata,
    action,
    handleLoadWorkflow,
    handleSelectStylePreset,
    handleGoToDestination,
  ]);
};

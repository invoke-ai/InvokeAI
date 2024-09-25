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
import { toast } from 'features/toast/toast';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { useGetAndLoadLibraryWorkflow } from 'features/workflowLibrary/hooks/useGetAndLoadLibraryWorkflow';
import { $workflowLibraryModal } from 'features/workflowLibrary/store/isWorkflowLibraryModalOpen';
import { useCallback, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { getImageDTO, getImageMetadata } from 'services/api/endpoints/images';
import { getStylePreset } from 'services/api/endpoints/stylePresets';

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
  const { t } = useTranslation();
  const didUseRef = useRef(false);
  const store = useAppStore();
  const { getAndLoadWorkflow } = useGetAndLoadLibraryWorkflow();

  const handleSendToCanvas = useCallback(
    async (imageName: string) => {
      // Try to the image DTO
      const getImageDTOResult = await withResultAsync(() => getImageDTO(imageName));
      if (getImageDTOResult.isErr()) {
        toast({
          title: t('toast.unableToLoadImage'),
          status: 'error',
        });
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
      toast({
        title: t('toast.sentToCanvas'),
        status: 'info',
      });
    },
    [store, t]
  );

  const handleUseAllMetadata = useCallback(
    async (imageName: string) => {
      const getImageMetadataResult = await withResultAsync(() => getImageMetadata(imageName));
      if (getImageMetadataResult.isErr()) {
        toast({
          title: t('toast.unableToLoadImageMetadata'),
          status: 'error',
        });
        return;
      }
      const metadata = getImageMetadataResult.value;
      // This shows a toast
      parseAndRecallAllMetadata(metadata, true);
      store.dispatch(setActiveTab('canvas'));
    },
    [store, t]
  );

  const handleLoadWorkflow = useCallback(
    (workflowId: string) => {
      // This shows a toast
      getAndLoadWorkflow(workflowId);
      store.dispatch(setActiveTab('workflows'));
    },
    [getAndLoadWorkflow, store]
  );

  const handleSelectStylePreset = useCallback(
    async (stylePresetId: string) => {
      const getStylePresetResult = await withResultAsync(() => getStylePreset(stylePresetId));
      if (getStylePresetResult.isErr()) {
        toast({
          title: t('toast.unableToLoadStylePreset'),
          status: 'error',
        });
        return;
      }
      store.dispatch(activeStylePresetIdChanged(stylePresetId));
      store.dispatch(setActiveTab('canvas'));
      toast({
        title: t('toast.stylePresetLoaded'),
        status: 'info',
      });
    },
    [store, t]
  );

  const handleGoToDestination = useCallback(
    (destination: StudioDestinationAction['data']['destination']) => {
      switch (destination) {
        case 'generation':
          // Go to the canvas tab, open the image viewer, and enable send-to-gallery mode
          store.dispatch(setActiveTab('canvas'));
          store.dispatch(settingsSendToCanvasChanged(false));
          $imageViewer.set(true);
          break;
        case 'canvas':
          // Go to the canvas tab, close the image viewer, and disable send-to-gallery mode
          store.dispatch(setActiveTab('canvas'));
          store.dispatch(settingsSendToCanvasChanged(true));
          $imageViewer.set(false);
          break;
        case 'workflows':
          // Go to the workflows tab
          store.dispatch(setActiveTab('workflows'));
          break;
        case 'upscaling':
          // Go to the upscaling tab
          store.dispatch(setActiveTab('upscaling'));
          break;
        case 'viewAllWorkflows':
          // Go to the workflows tab and open the workflow library modal
          store.dispatch(setActiveTab('workflows'));
          $workflowLibraryModal.set(true);
          break;
        case 'viewAllStylePresets':
          // Go to the canvas tab and open the style presets menu
          store.dispatch(setActiveTab('canvas'));
          $isStylePresetsMenuOpen.set(true);
          break;
      }
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

import { useStore } from '@nanostores/react';
import { useAppStore } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { withResultAsync } from 'common/util/result';
import { canvasReset } from 'features/controlLayers/store/actions';
import { rasterLayerAdded } from 'features/controlLayers/store/canvasSlice';
import type { CanvasRasterLayerState } from 'features/controlLayers/store/types';
import { imageDTOToImageObject } from 'features/controlLayers/store/util';
import { sentImageToCanvas } from 'features/gallery/store/actions';
import { MetadataUtils } from 'features/metadata/parsing';
import { $hasTemplates } from 'features/nodes/store/nodesSlice';
import { $isWorkflowLibraryModalOpen } from 'features/nodes/store/workflowLibraryModal';
import {
  $workflowLibraryTagOptions,
  workflowLibraryTagsReset,
  workflowLibraryTagToggled,
  workflowLibraryViewChanged,
} from 'features/nodes/store/workflowLibrarySlice';
import { $isStylePresetsMenuOpen, activeStylePresetIdChanged } from 'features/stylePresets/store/stylePresetSlice';
import { toast } from 'features/toast/toast';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { LAUNCHPAD_PANEL_ID, WORKSPACE_PANEL_ID } from 'features/ui/layouts/shared';
import { useLoadWorkflowWithDialog } from 'features/workflowLibrary/components/LoadWorkflowConfirmationAlertDialog';
import { atom } from 'nanostores';
import { useCallback, useEffect } from 'react';
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
  {
    destination:
      | 'generation'
      | 'canvas'
      | 'workflows'
      | 'upscaling'
      | 'video'
      | 'viewAllWorkflows'
      | 'viewAllWorkflowsRecommended'
      | 'viewAllStylePresets';
  }
>;
// Use global state to show loader until we are ready to render the studio.
export const $didStudioInit = atom(false);

export type StudioInitAction =
  | LoadWorkflowAction
  | SelectStylePresetAction
  | SendToCanvasAction
  | UseAllParametersAction
  | StudioDestinationAction;

/**
 * A hook that performs an action when the studio is initialized. This is useful for deep linking into the studio.
 *
 * The action is performed only once, when the hook is first run.
 *
 * In this hook, we prefer to use imperative APIs over hooks to avoid re-rendering the parent component. For example:
 * - Use `getImageDTO` helper instead of `useGetImageDTO`
 * - Usee the `$imageViewer` atom instead of `useImageViewer`
 */
export const useStudioInitAction = (action?: StudioInitAction) => {
  useAssertSingleton('useStudioInitAction');
  const { t } = useTranslation();
  const didParseOpenAPISchema = useStore($hasTemplates);
  const store = useAppStore();
  const loadWorkflowWithDialog = useLoadWorkflowWithDialog();
  const workflowLibraryTagOptions = useStore($workflowLibraryTagOptions);

  const handleSendToCanvas = useCallback(
    async (imageName: string) => {
      // Try to the image DTO - use an imperative helper, rather than `useGetImageDTO`, so that we aren't re-rendering
      // the parent of this hook whenever the image name changes
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
      await navigationApi.focusPanel('canvas', WORKSPACE_PANEL_ID);
      store.dispatch(canvasReset());
      store.dispatch(rasterLayerAdded({ overrides, isSelected: true }));
      store.dispatch(sentImageToCanvas());
      toast({
        title: t('toast.sentToCanvas'),
        status: 'info',
      });
    },
    [store, t]
  );

  const handleUseAllMetadata = useCallback(
    async (imageName: string) => {
      // Try to the image metadata - use an imperative helper, rather than `useGetImageMetadata`, so that we aren't
      // re-rendering the parent of this hook whenever the image name changes
      const getImageMetadataResult = await withResultAsync(() => getImageMetadata(imageName));
      if (getImageMetadataResult.isErr()) {
        toast({
          title: t('toast.unableToLoadImageMetadata'),
          status: 'error',
        });
        return;
      }
      const metadata = getImageMetadataResult.value;
      store.dispatch(canvasReset());
      // This shows a toast
      await MetadataUtils.recallAllImageMetadata(metadata, store);
    },
    [store, t]
  );

  const handleLoadWorkflow = useCallback(
    (workflowId: string) => {
      // This shows a toast
      loadWorkflowWithDialog({
        type: 'library',
        data: workflowId,
        onSuccess: () => {
          navigationApi.switchToTab('workflows');
        },
      });
    },
    [loadWorkflowWithDialog]
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
      navigationApi.switchToTab('canvas');
      toast({
        title: t('toast.stylePresetLoaded'),
        status: 'info',
      });
    },
    [store, t]
  );

  const handleGoToDestination = useCallback(
    async (destination: StudioDestinationAction['data']['destination']) => {
      switch (destination) {
        case 'generation':
          // Go to the generate tab, open the launchpad
          await navigationApi.focusPanel('generate', LAUNCHPAD_PANEL_ID);
          break;
        case 'canvas':
          // Go to the canvas tab, open the launchpad
          await navigationApi.focusPanel('canvas', WORKSPACE_PANEL_ID);
          break;
        case 'workflows':
          // Go to the workflows tab
          navigationApi.switchToTab('workflows');
          break;
        case 'upscaling':
          // Go to the upscaling tab
          navigationApi.switchToTab('upscaling');
          break;
        case 'video':
          // Go to the video tab
          await navigationApi.focusPanel('video', LAUNCHPAD_PANEL_ID);
          break;
        case 'viewAllWorkflows':
          // Go to the workflows tab and open the workflow library modal
          navigationApi.switchToTab('workflows');
          $isWorkflowLibraryModalOpen.set(true);
          break;
        case 'viewAllWorkflowsRecommended':
          // Go to the workflows tab and open the workflow library modal with the recommended workflows view
          navigationApi.switchToTab('workflows');
          $isWorkflowLibraryModalOpen.set(true);
          store.dispatch(workflowLibraryViewChanged('defaults'));
          store.dispatch(workflowLibraryTagsReset());
          for (const tag of workflowLibraryTagOptions) {
            if (tag.recommended) {
              store.dispatch(workflowLibraryTagToggled(tag.label));
            }
          }
          break;
        case 'viewAllStylePresets':
          // Go to the canvas tab and open the style presets menu
          navigationApi.switchToTab('canvas');
          $isStylePresetsMenuOpen.set(true);
          break;
      }
    },
    [store, workflowLibraryTagOptions]
  );

  const handleStudioInitAction = useCallback(
    async (action: StudioInitAction) => {
      // This cannot be in the useEffect below because we need to await some of the actions before setting didStudioInit.
      switch (action.type) {
        case 'loadWorkflow':
          await handleLoadWorkflow(action.data.workflowId);
          break;
        case 'selectStylePreset':
          await handleSelectStylePreset(action.data.stylePresetId);
          break;

        case 'sendToCanvas':
          await handleSendToCanvas(action.data.imageName);
          break;

        case 'useAllParameters':
          await handleUseAllMetadata(action.data.imageName);
          break;

        case 'goToDestination':
          handleGoToDestination(action.data.destination);
          break;

        default:
          break;
      }
      $didStudioInit.set(true);
    },
    [handleGoToDestination, handleLoadWorkflow, handleSelectStylePreset, handleSendToCanvas, handleUseAllMetadata]
  );

  useEffect(() => {
    if ($didStudioInit.get() || !didParseOpenAPISchema) {
      return;
    }

    if (!action) {
      $didStudioInit.set(true);
      return;
    }

    handleStudioInitAction(action);
  }, [
    handleSendToCanvas,
    handleUseAllMetadata,
    action,
    handleSelectStylePreset,
    handleGoToDestination,
    handleLoadWorkflow,
    didParseOpenAPISchema,
    handleStudioInitAction,
  ]);
};

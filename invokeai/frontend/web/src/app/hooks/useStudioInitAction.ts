import { logger } from 'app/logging/logger';
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
import { boardIdSelected } from 'features/gallery/store/gallerySlice';
import { parseAndRecallAllMetadata } from 'features/metadata/util/handlers';
import { $isWorkflowListMenuIsOpen } from 'features/nodes/store/workflowListMenu';
import { $isStylePresetsMenuOpen, activeStylePresetIdChanged } from 'features/stylePresets/store/stylePresetSlice';
import { toast } from 'features/toast/toast';
import { activeTabCanvasRightPanelChanged, setActiveTab } from 'features/ui/store/uiSlice';
import { useGetAndLoadLibraryWorkflow } from 'features/workflowLibrary/hooks/useGetAndLoadLibraryWorkflow';
import { useCallback, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { serializeError } from 'serialize-error';
import { getImageDTO, getImageMetadata } from 'services/api/endpoints/images';
import { getStylePreset } from 'services/api/endpoints/stylePresets';
import { z } from 'zod';

const zLoadWorkflowAction = z.object({
  type: z.literal('loadWorkflow'),
  data: z.object({ workflowId: z.string() }),
});
// type LoadWorkflowAction = z.infer<typeof zLoadWorkflowAction>;

const zSelectBoardAction = z.object({
  type: z.literal('selectBoard'),
  data: z.object({ boardId: z.string() }),
});
// type SelectBoardAction = z.infer<typeof zSelectBoardAction>;

const zSelectImageAction = z.object({
  type: z.literal('selectImage'),
  data: z.object({ imageName: z.string() }),
});
// type SelectImageAction = z.infer<typeof zSelectImageAction>;

const zSelectStylePresetAction = z.object({
  type: z.literal('selectStylePreset'),
  data: z.object({ stylePresetId: z.string() }),
});
// type SelectStylePresetAction = z.infer<typeof zSelectStylePresetAction>;

const zSendToCanvasAction = z.object({
  type: z.literal('sendToCanvas'),
  data: z.object({ imageName: z.string() }),
});
// type SendToCanvasAction = z.infer<typeof zSendToCanvasAction>;

const zUseAllParametersAction = z.object({
  type: z.literal('useAllParameters'),
  data: z.object({ imageName: z.string() }),
});
// type UseAllParametersAction = z.infer<typeof zUseAllParametersAction>;

const zStudioDestinationAction = z.object({
  type: z.literal('goToDestination'),
  data: z.object({
    destination: z.enum(['generation', 'canvas', 'workflows', 'upscaling', 'viewAllWorkflows', 'viewAllStylePresets']),
  }),
});
type StudioDestinationAction = z.infer<typeof zStudioDestinationAction>;

export const zStudioInitAction = z.discriminatedUnion('type', [
  zLoadWorkflowAction,
  zSelectBoardAction,
  zSelectImageAction,
  zSelectStylePresetAction,
  zSendToCanvasAction,
  zUseAllParametersAction,
  zStudioDestinationAction,
]);

export type StudioInitAction = z.infer<typeof zStudioInitAction>;

/**
 * Converts a given hashbang string to a valid StudioInitAction
 * @see fillStudioInitAction
 * @param {string} hashBang
 * @returns {StudioInitAction}
 * @throws {z.ZodError | Error} If there is a validation error.
 */
export const genHashBangStudioInitAction = (hashBang: string): StudioInitAction => {
  if (!hashBang.startsWith('#!')) {
    throw new Error("The given string isn't a valid hashbang action");
  }
  const parts = hashBang.substring(2).split('&');
  return zStudioInitAction.parse({
    type: parts.shift(),
    data: Object.fromEntries(new URLSearchParams(parts.join('&'))),
  });
};

/**
 * Uses the HashBang fragment to populate an unset StudioInitAction in case the user tries to execute a StudioInitAction on startup via a location.hash fragment
 * If any studioInitAction is given, it will early bail with it.
 * this will interpret and validate the hashbang as an studioInitAction
 * @returns {StudioInitAction | undefined} undefined if nothing can be resolved
 */
export const fillStudioInitAction = (
  studioInitAction?: StudioInitAction,
  clearHashBang: boolean = false
): StudioInitAction | undefined => {
  if (studioInitAction !== undefined) {
    return studioInitAction;
  }
  if (!location.hash.startsWith('#!')) {
    return undefined;
  }

  try {
    studioInitAction = genHashBangStudioInitAction(location.hash);
    if (clearHashBang) {
      location.hash = '';  //reset the hash to "acknowledge" the initAction (and push the history forward)
    }
  } catch (err) {
    const log = logger('system');
    if (err instanceof z.ZodError) {
      log.error({ error: serializeError(err) }, 'Problem persisting the studioInitAction from the given hashbang');
    } else if (err instanceof Error) {
      log.error({ error: serializeError(err) }, 'Problem interpreting the hashbang');
    } else {
      log.error({ error: serializeError(err) }, 'Problem while filling StudioInitAction');
    }
  }
  return studioInitAction;
};

/**
 * A hook that performs an action when the studio is initialized. This is useful for deep linking into the studio.
 *
 * The action is performed only once, when the hook is first run.
 *
 * In this hook, we prefer to use imperative APIs over hooks to avoid re-rendering the parent component. For example:
 * - Use `getImageDTO` helper instead of `useGetImageDTO`
 * - Use the `$imageViewer` atom instead of `useImageViewer`
 */
export const useStudioInitAction = (action?: StudioInitAction) => {
  useAssertSingleton('useStudioInitAction');
  const { t } = useTranslation();
  // Use a ref to ensure that we only perform the action once
  const didInit = useRef(false);
  const store = useAppStore();
  const { getAndLoadWorkflow } = useGetAndLoadLibraryWorkflow();

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

  const handleSelectBoard = useCallback(
    (boardId: string) => {
      //TODO: validate given boardID
      store.dispatch(boardIdSelected({ boardId: boardId }));
      //TODO: scroll into view
    },
    [store]
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
          store.dispatch(activeTabCanvasRightPanelChanged('gallery'));
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
          $isWorkflowListMenuIsOpen.set(true);
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
    if (didInit.current || !action) {
      return;
    }

    didInit.current = true;

    switch (action.type) {
      case 'loadWorkflow':
        handleLoadWorkflow(action.data.workflowId);
        break;
      case 'selectBoard':
        handleSelectBoard(action.data.boardId);
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
    handleSelectBoard,
    handleSelectStylePreset,
    handleGoToDestination,
  ]);
};

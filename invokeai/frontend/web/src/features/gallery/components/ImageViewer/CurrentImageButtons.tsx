import { Button, Divider, IconButton, Menu, MenuButton, MenuList } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { useCanvasManagerSafe } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { selectIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { DeleteImageButton } from 'features/deleteImageModal/components/DeleteImageButton';
import SingleSelectionMenuItems from 'features/gallery/components/ImageContextMenu/SingleSelectionMenuItems';
import { useImageActions } from 'features/gallery/hooks/useImageActions';
import { selectLastSelectedImage } from 'features/gallery/store/gallerySelectors';
import { newCanvasFromImage } from 'features/imageActions/actions';
import { $hasTemplates } from 'features/nodes/store/nodesSlice';
import { PostProcessingPopover } from 'features/parameters/components/PostProcessing/PostProcessingPopover';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { toast } from 'features/toast/toast';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { WORKSPACE_PANEL_ID } from 'features/ui/layouts/shared';
import { selectActiveTab, selectShouldShowProgressInViewer } from 'features/ui/store/uiSelectors';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiArrowsCounterClockwiseBold,
  PiAsteriskBold,
  PiDotsThreeOutlineFill,
  PiFlowArrowBold,
  PiPencilBold,
  PiPlantBold,
  PiQuotesBold,
  PiRulerBold,
} from 'react-icons/pi';
import { useImageDTO } from 'services/api/endpoints/images';

import { useImageViewerContext } from './context';

export const CurrentImageButtons = memo(() => {
  const { t } = useTranslation();
  const ctx = useImageViewerContext();
  const hasProgressImage = useStore(ctx.$hasProgressImage);
  const shouldShowProgressInViewer = useAppSelector(selectShouldShowProgressInViewer);
  const isDisabledOverride = hasProgressImage && shouldShowProgressInViewer;

  const imageName = useAppSelector(selectLastSelectedImage);
  const imageDTO = useImageDTO(imageName);
  const hasTemplates = useStore($hasTemplates);
  const imageActions = useImageActions(imageDTO);
  const isStaging = useAppSelector(selectIsStaging);
  const activeTab = useAppSelector(selectActiveTab);
  const isUpscalingEnabled = useFeatureStatus('upscaling');
  const { getState, dispatch } = useAppStore();
  const canvasManager = useCanvasManagerSafe();

  const isCanvasTabAndStaging = useMemo(() => activeTab === 'canvas' && isStaging, [activeTab, isStaging]);

  const handleEdit = useCallback(async () => {
    if (!imageDTO) {
      return;
    }

    await newCanvasFromImage({
      imageDTO,
      type: 'raster_layer',
      withInpaintMask: true,
      getState,
      dispatch,
    });
    navigationApi.focusPanel('canvas', WORKSPACE_PANEL_ID);

    // Automatically select the brush tool when editing an image
    if (canvasManager) {
      canvasManager.tool.$tool.set('brush');
    }

    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToCanvas'),
      status: 'success',
    });
  }, [imageDTO, getState, dispatch, t, canvasManager]);

  return (
    <>
      <Menu isLazy>
        <MenuButton
          as={IconButton}
          aria-label={t('parameters.imageActions')}
          tooltip={t('parameters.imageActions')}
          isDisabled={isDisabledOverride || !imageDTO}
          variant="link"
          alignSelf="stretch"
          icon={<PiDotsThreeOutlineFill />}
        />
        <MenuList>{imageDTO && <SingleSelectionMenuItems imageDTO={imageDTO} />}</MenuList>
      </Menu>

      <Divider orientation="vertical" h={8} mx={2} />

      <Button
        leftIcon={<PiPencilBold />}
        onClick={handleEdit}
        isDisabled={isDisabledOverride || !imageDTO}
        variant="link"
        size="sm"
        alignSelf="stretch"
        px={2}
      >
        {t('common.edit')}
      </Button>

      <Divider orientation="vertical" h={8} mx={2} />

      <IconButton
        icon={<PiFlowArrowBold />}
        tooltip={`${t('nodes.loadWorkflow')} (W)`}
        aria-label={`${t('nodes.loadWorkflow')} (W)`}
        isDisabled={isDisabledOverride || !imageDTO || !imageActions.hasWorkflow || !hasTemplates}
        variant="link"
        alignSelf="stretch"
        onClick={imageActions.loadWorkflow}
      />
      <IconButton
        icon={<PiArrowsCounterClockwiseBold />}
        tooltip={`${t('parameters.remixImage')} (R)`}
        aria-label={`${t('parameters.remixImage')} (R)`}
        isDisabled={isDisabledOverride || !imageDTO || !imageActions.hasMetadata}
        variant="link"
        alignSelf="stretch"
        onClick={imageActions.remix}
      />
      <IconButton
        icon={<PiQuotesBold />}
        tooltip={`${t('parameters.usePrompt')} (P)`}
        aria-label={`${t('parameters.usePrompt')} (P)`}
        isDisabled={isDisabledOverride || !imageDTO || !imageActions.hasPrompts}
        variant="link"
        alignSelf="stretch"
        onClick={imageActions.recallPrompts}
      />
      <IconButton
        icon={<PiPlantBold />}
        tooltip={`${t('parameters.useSeed')} (S)`}
        aria-label={`${t('parameters.useSeed')} (S)`}
        isDisabled={isDisabledOverride || !imageDTO || !imageActions.hasSeed}
        variant="link"
        alignSelf="stretch"
        onClick={imageActions.recallSeed}
      />
      <IconButton
        icon={<PiRulerBold />}
        tooltip={`${t('parameters.useSize')} (D)`}
        aria-label={`${t('parameters.useSize')} (D)`}
        variant="link"
        alignSelf="stretch"
        onClick={imageActions.recallSize}
        isDisabled={isDisabledOverride || !imageDTO || isCanvasTabAndStaging}
      />
      <IconButton
        icon={<PiAsteriskBold />}
        tooltip={`${t('parameters.useAll')} (A)`}
        aria-label={`${t('parameters.useAll')} (A)`}
        isDisabled={isDisabledOverride || !imageDTO || !imageActions.hasMetadata}
        variant="link"
        alignSelf="stretch"
        onClick={imageActions.recallAll}
      />

      {isUpscalingEnabled && <PostProcessingPopover imageDTO={imageDTO} isDisabled={isDisabledOverride} />}

      <Divider orientation="vertical" h={8} mx={2} />

      <DeleteImageButton onClick={imageActions.delete} isDisabled={isDisabledOverride || !imageDTO} />
    </>
  );
});

CurrentImageButtons.displayName = 'CurrentImageButtons';

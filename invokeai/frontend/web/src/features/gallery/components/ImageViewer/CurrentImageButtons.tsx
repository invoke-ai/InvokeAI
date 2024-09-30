import { ButtonGroup, IconButton, Menu, MenuButton, MenuList } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { skipToken } from '@reduxjs/toolkit/query';
import { adHocPostProcessingRequested } from 'app/store/middleware/listenerMiddleware/listeners/addAdHocPostProcessingRequestedListener';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useIsRegionFocused } from 'common/hooks/interactionScopes';
import { selectIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { DeleteImageButton } from 'features/deleteImageModal/components/DeleteImageButton';
import { imagesToDeleteSelected } from 'features/deleteImageModal/store/slice';
import SingleSelectionMenuItems from 'features/gallery/components/ImageContextMenu/SingleSelectionMenuItems';
import { useImageActions } from 'features/gallery/hooks/useImageActions';
import { selectLastSelectedImage } from 'features/gallery/store/gallerySelectors';
import { parseAndRecallImageDimensions } from 'features/metadata/util/handlers';
import { $templates } from 'features/nodes/store/nodesSlice';
import { PostProcessingPopover } from 'features/parameters/components/PostProcessing/PostProcessingPopover';
import { useIsQueueMutationInProgress } from 'features/queue/hooks/useIsQueueMutationInProgress';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { useGetAndLoadEmbeddedWorkflow } from 'features/workflowLibrary/hooks/useGetAndLoadEmbeddedWorkflow';
import { size } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiArrowsCounterClockwiseBold,
  PiAsteriskBold,
  PiDotsThreeOutlineFill,
  PiFlowArrowBold,
  PiPlantBold,
  PiQuotesBold,
  PiRulerBold,
} from 'react-icons/pi';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import { $isConnected, $progressImage } from 'services/events/stores';

const CurrentImageButtons = () => {
  const dispatch = useAppDispatch();
  const isConnected = useStore($isConnected);
  const isStaging = useAppSelector(selectIsStaging);
  const lastSelectedImage = useAppSelector(selectLastSelectedImage);
  const progressImage = useStore($progressImage);
  const shouldDisableToolbarButtons = useMemo(() => {
    return Boolean(progressImage) || !lastSelectedImage;
  }, [lastSelectedImage, progressImage]);
  const templates = useStore($templates);
  const isUpscalingEnabled = useFeatureStatus('upscaling');
  const isQueueMutationInProgress = useIsQueueMutationInProgress();
  const { t } = useTranslation();
  const isGalleryFocused = useIsRegionFocused('gallery');
  const isViewerFocused = useIsRegionFocused('viewer');
  const { currentData: imageDTO } = useGetImageDTOQuery(lastSelectedImage?.image_name ?? skipToken);

  const { recallAll, remix, recallSeed, recallPrompts, hasMetadata, hasSeed, hasPrompts, isLoadingMetadata } =
    useImageActions(lastSelectedImage?.image_name);

  const { getAndLoadEmbeddedWorkflow, getAndLoadEmbeddedWorkflowResult } = useGetAndLoadEmbeddedWorkflow({});

  const handleLoadWorkflow = useCallback(() => {
    if (!lastSelectedImage || !lastSelectedImage.has_workflow) {
      return;
    }
    getAndLoadEmbeddedWorkflow(lastSelectedImage.image_name);
  }, [getAndLoadEmbeddedWorkflow, lastSelectedImage]);

  const handleUseSize = useCallback(() => {
    if (isStaging) {
      return;
    }
    parseAndRecallImageDimensions(lastSelectedImage);
  }, [isStaging, lastSelectedImage]);
  const handleClickUpscale = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    dispatch(adHocPostProcessingRequested({ imageDTO }));
  }, [dispatch, imageDTO]);

  const handleDelete = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    dispatch(imagesToDeleteSelected([imageDTO]));
  }, [dispatch, imageDTO]);

  useRegisteredHotkeys({
    id: 'loadWorkflow',
    category: 'viewer',
    callback: handleLoadWorkflow,
    options: { enabled: isGalleryFocused || isViewerFocused },
    dependencies: [handleLoadWorkflow, isGalleryFocused, isViewerFocused],
  });
  useRegisteredHotkeys({
    id: 'recallAll',
    category: 'viewer',
    callback: recallAll,
    options: { enabled: isGalleryFocused || isViewerFocused },
    dependencies: [recallAll, isGalleryFocused, isViewerFocused],
  });
  useRegisteredHotkeys({
    id: 'recallSeed',
    category: 'viewer',
    callback: recallSeed,
    options: { enabled: isGalleryFocused || isViewerFocused },
    dependencies: [recallSeed, isGalleryFocused, isViewerFocused],
  });
  useRegisteredHotkeys({
    id: 'recallPrompts',
    category: 'viewer',
    callback: recallPrompts,
    options: { enabled: isGalleryFocused || isViewerFocused },
    dependencies: [recallPrompts, isGalleryFocused, isViewerFocused],
  });
  useRegisteredHotkeys({
    id: 'remix',
    category: 'viewer',
    callback: remix,
    options: { enabled: isGalleryFocused || isViewerFocused },
    dependencies: [remix, isGalleryFocused, isViewerFocused],
  });
  useRegisteredHotkeys({
    id: 'useSize',
    category: 'viewer',
    callback: handleUseSize,
    options: { enabled: isGalleryFocused || isViewerFocused },
    dependencies: [handleUseSize, isGalleryFocused, isViewerFocused],
  });
  useRegisteredHotkeys({
    id: 'runPostprocessing',
    category: 'viewer',
    callback: handleClickUpscale,
    options: { enabled: Boolean(isUpscalingEnabled && isViewerFocused && isConnected) },
    dependencies: [isUpscalingEnabled, imageDTO, shouldDisableToolbarButtons, isConnected, isViewerFocused],
  });

  return (
    <>
      <ButtonGroup isDisabled={shouldDisableToolbarButtons}>
        <Menu isLazy>
          <MenuButton
            as={IconButton}
            aria-label={t('parameters.imageActions')}
            tooltip={t('parameters.imageActions')}
            isDisabled={!imageDTO}
            icon={<PiDotsThreeOutlineFill />}
          />
          <MenuList>{imageDTO && <SingleSelectionMenuItems imageDTO={imageDTO} />}</MenuList>
        </Menu>
      </ButtonGroup>

      <ButtonGroup isDisabled={shouldDisableToolbarButtons}>
        <IconButton
          icon={<PiFlowArrowBold />}
          tooltip={`${t('nodes.loadWorkflow')} (W)`}
          aria-label={`${t('nodes.loadWorkflow')} (W)`}
          isDisabled={!imageDTO?.has_workflow || !size(templates)}
          onClick={handleLoadWorkflow}
          isLoading={getAndLoadEmbeddedWorkflowResult.isLoading}
        />
        <IconButton
          isLoading={isLoadingMetadata}
          icon={<PiArrowsCounterClockwiseBold />}
          tooltip={`${t('parameters.remixImage')} (R)`}
          aria-label={`${t('parameters.remixImage')} (R)`}
          isDisabled={!hasMetadata}
          onClick={remix}
        />
        <IconButton
          isLoading={isLoadingMetadata}
          icon={<PiQuotesBold />}
          tooltip={`${t('parameters.usePrompt')} (P)`}
          aria-label={`${t('parameters.usePrompt')} (P)`}
          isDisabled={!hasPrompts}
          onClick={recallPrompts}
        />
        <IconButton
          isLoading={isLoadingMetadata}
          icon={<PiPlantBold />}
          tooltip={`${t('parameters.useSeed')} (S)`}
          aria-label={`${t('parameters.useSeed')} (S)`}
          isDisabled={!hasSeed}
          onClick={recallSeed}
        />
        <IconButton
          isLoading={isLoadingMetadata}
          icon={<PiRulerBold />}
          tooltip={`${t('parameters.useSize')} (D)`}
          aria-label={`${t('parameters.useSize')} (D)`}
          onClick={handleUseSize}
          isDisabled={isStaging}
        />
        <IconButton
          isLoading={isLoadingMetadata}
          icon={<PiAsteriskBold />}
          tooltip={`${t('parameters.useAll')} (A)`}
          aria-label={`${t('parameters.useAll')} (A)`}
          isDisabled={!hasMetadata}
          onClick={recallAll}
        />
      </ButtonGroup>

      {isUpscalingEnabled && (
        <ButtonGroup isDisabled={isQueueMutationInProgress}>
          {isUpscalingEnabled && <PostProcessingPopover imageDTO={imageDTO} />}
        </ButtonGroup>
      )}

      <ButtonGroup>
        <DeleteImageButton onClick={handleDelete} />
      </ButtonGroup>
    </>
  );
};

export default memo(CurrentImageButtons);

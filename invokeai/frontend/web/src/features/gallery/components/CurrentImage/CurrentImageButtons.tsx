import { ButtonGroup, Flex, IconButton, Menu, MenuButton, MenuList } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppToaster } from 'app/components/Toaster';
import { upscaleRequested } from 'app/store/middleware/listenerMiddleware/listeners/upscaleRequested';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { DeleteImageButton } from 'features/deleteImageModal/components/DeleteImageButton';
import { imagesToDeleteSelected } from 'features/deleteImageModal/store/slice';
import SingleSelectionMenuItems from 'features/gallery/components/ImageContextMenu/SingleSelectionMenuItems';
import { sentImageToImg2Img } from 'features/gallery/store/actions';
import { selectLastSelectedImage } from 'features/gallery/store/gallerySelectors';
import { selectGallerySlice } from 'features/gallery/store/gallerySlice';
import ParamUpscalePopover from 'features/parameters/components/Upscale/ParamUpscaleSettings';
import { useRecallParameters } from 'features/parameters/hooks/useRecallParameters';
import { initialImageSelected } from 'features/parameters/store/actions';
import { useIsQueueMutationInProgress } from 'features/queue/hooks/useIsQueueMutationInProgress';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { selectSystemSlice } from 'features/system/store/systemSlice';
import {
  setShouldShowImageDetails,
  setShouldShowProgressInViewer,
  setShouldShowShowcase,
} from 'features/ui/store/uiSlice';
import { useGetAndLoadEmbeddedWorkflow } from 'features/workflowLibrary/hooks/useGetAndLoadEmbeddedWorkflow';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import {
  PiArrowsCounterClockwiseBold,
  PiAsteriskBold,
  PiDotsThreeOutlineFill,
  PiFlowArrowBold,
  PiHourglassHighBold,
  PiInfoBold,
  PiPlantBold,
  PiProjectorScreen,
  PiQuotesBold,
  PiRulerBold,
} from 'react-icons/pi';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import { useDebouncedMetadata } from 'services/api/hooks/useDebouncedMetadata';

const selectShouldDisableToolbarButtons = createSelector(
  selectSystemSlice,
  selectGallerySlice,
  selectLastSelectedImage,
  (system, gallery, lastSelectedImage) => {
    const hasProgressImage = Boolean(system.denoiseProgress?.progress_image);
    return hasProgressImage || !lastSelectedImage;
  }
);

const CurrentImageButtons = () => {
  const dispatch = useAppDispatch();
  const isConnected = useAppSelector((s) => s.system.isConnected);
  const shouldShowImageDetails = useAppSelector((s) => s.ui.shouldShowImageDetails);
  const shouldShowProgressInViewer = useAppSelector((s) => s.ui.shouldShowProgressInViewer);
  const showShowcase = useAppSelector((s) => s.ui.showShowcase);
  const lastSelectedImage = useAppSelector(selectLastSelectedImage);
  const shouldDisableToolbarButtons = useAppSelector(selectShouldDisableToolbarButtons);

  const isUpscalingEnabled = useFeatureStatus('upscaling').isFeatureEnabled;
  const isQueueMutationInProgress = useIsQueueMutationInProgress();
  const toaster = useAppToaster();
  const { t } = useTranslation();

  const { recallBothPrompts, recallSeed, recallWidthAndHeight, recallAllParameters } = useRecallParameters();

  const { currentData: imageDTO } = useGetImageDTOQuery(lastSelectedImage?.image_name ?? skipToken);

  const { metadata, isLoading: isLoadingMetadata } = useDebouncedMetadata(lastSelectedImage?.image_name);

  const { getAndLoadEmbeddedWorkflow, getAndLoadEmbeddedWorkflowResult } = useGetAndLoadEmbeddedWorkflow({});

  const toggleShowcase = useCallback(() => {
    if (imageDTO && !showShowcase) {
      dispatch(setShouldShowShowcase(imageDTO));
    } else {
      dispatch(setShouldShowShowcase(null));
    }
  }, [dispatch, showShowcase, imageDTO]);

  useHotkeys('shift+s', toggleShowcase);

  const handleLoadWorkflow = useCallback(() => {
    if (!lastSelectedImage || !lastSelectedImage.has_workflow) {
      return;
    }
    getAndLoadEmbeddedWorkflow(lastSelectedImage.image_name);
  }, [getAndLoadEmbeddedWorkflow, lastSelectedImage]);

  useHotkeys('w', handleLoadWorkflow, [lastSelectedImage]);

  const handleClickUseAllParameters = useCallback(() => {
    recallAllParameters(metadata);
  }, [metadata, recallAllParameters]);

  useHotkeys('a', handleClickUseAllParameters, [metadata]);

  const handleUseSeed = useCallback(() => {
    recallSeed(metadata?.seed);
  }, [metadata?.seed, recallSeed]);

  useHotkeys('s', handleUseSeed, [metadata]);

  const handleUsePrompt = useCallback(() => {
    recallBothPrompts(
      metadata?.positive_prompt,
      metadata?.negative_prompt,
      metadata?.positive_style_prompt,
      metadata?.negative_style_prompt
    );
  }, [
    metadata?.negative_prompt,
    metadata?.positive_prompt,
    metadata?.positive_style_prompt,
    metadata?.negative_style_prompt,
    recallBothPrompts,
  ]);

  useHotkeys('p', handleUsePrompt, [metadata]);

  const handleRemixImage = useCallback(() => {
    // Recalls all metadata parameters except seed
    recallAllParameters({
      ...metadata,
      seed: undefined,
    });
  }, [metadata, recallAllParameters]);

  useHotkeys('r', handleRemixImage, [metadata]);

  const handleUseSize = useCallback(() => {
    recallWidthAndHeight(metadata?.width, metadata?.height);
  }, [metadata?.width, metadata?.height, recallWidthAndHeight]);

  useHotkeys('d', handleUseSize, [metadata]);

  const handleSendToImageToImage = useCallback(() => {
    dispatch(sentImageToImg2Img());
    dispatch(initialImageSelected(imageDTO));
  }, [dispatch, imageDTO]);

  useHotkeys('shift+i', handleSendToImageToImage, [imageDTO]);

  const handleClickUpscale = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    dispatch(upscaleRequested({ imageDTO }));
  }, [dispatch, imageDTO]);

  const handleDelete = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    dispatch(imagesToDeleteSelected([imageDTO]));
  }, [dispatch, imageDTO]);

  useHotkeys(
    'Shift+U',
    () => {
      handleClickUpscale();
    },
    {
      enabled: () => Boolean(isUpscalingEnabled && !shouldDisableToolbarButtons && isConnected),
    },
    [isUpscalingEnabled, imageDTO, shouldDisableToolbarButtons, isConnected]
  );

  const handleClickShowImageDetails = useCallback(
    () => dispatch(setShouldShowImageDetails(!shouldShowImageDetails)),
    [dispatch, shouldShowImageDetails]
  );

  useHotkeys(
    'i',
    () => {
      if (imageDTO) {
        handleClickShowImageDetails();
      } else {
        toaster({
          title: t('toast.metadataLoadFailed'),
          status: 'error',
          duration: 2500,
          isClosable: true,
        });
      }
    },
    [imageDTO, shouldShowImageDetails, toaster]
  );

  useHotkeys(
    'delete',
    () => {
      handleDelete();
    },
    [dispatch, imageDTO]
  );

  const handleClickProgressImagesToggle = useCallback(() => {
    dispatch(setShouldShowProgressInViewer(!shouldShowProgressInViewer));
  }, [dispatch, shouldShowProgressInViewer]);

  return (
    <>
      <Flex flexWrap="wrap" justifyContent="center" alignItems="center" gap={2}>
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

        <IconButton
          icon={<PiProjectorScreen />}
          tooltip={`${!showShowcase ? t('common.openShowcase') : t('common.closeShowcase')} (Shift+S)`}
          aria-label={`${!showShowcase ? t('common.openShowcase') : t('common.closeShowcase')} (Shift+S)`}
          isDisabled={!imageDTO}
          onClick={toggleShowcase}
          isChecked={showShowcase !== null}
        />

        <ButtonGroup isDisabled={shouldDisableToolbarButtons}>
          <IconButton
            icon={<PiFlowArrowBold />}
            tooltip={`${t('nodes.loadWorkflow')} (W)`}
            aria-label={`${t('nodes.loadWorkflow')} (W)`}
            isDisabled={!imageDTO?.has_workflow}
            onClick={handleLoadWorkflow}
            isLoading={getAndLoadEmbeddedWorkflowResult.isLoading}
          />
          <IconButton
            isLoading={isLoadingMetadata}
            icon={<PiArrowsCounterClockwiseBold />}
            tooltip={`${t('parameters.remixImage')} (R)`}
            aria-label={`${t('parameters.remixImage')} (R)`}
            isDisabled={!metadata?.positive_prompt}
            onClick={handleRemixImage}
          />
          <IconButton
            isLoading={isLoadingMetadata}
            icon={<PiQuotesBold />}
            tooltip={`${t('parameters.usePrompt')} (P)`}
            aria-label={`${t('parameters.usePrompt')} (P)`}
            isDisabled={!metadata?.positive_prompt}
            onClick={handleUsePrompt}
          />
          <IconButton
            isLoading={isLoadingMetadata}
            icon={<PiPlantBold />}
            tooltip={`${t('parameters.useSeed')} (S)`}
            aria-label={`${t('parameters.useSeed')} (S)`}
            isDisabled={metadata?.seed === null || metadata?.seed === undefined}
            onClick={handleUseSeed}
          />
          <IconButton
            isLoading={isLoadingMetadata}
            icon={<PiRulerBold />}
            tooltip={`${t('parameters.useSize')} (D)`}
            aria-label={`${t('parameters.useSize')} (D)`}
            isDisabled={
              metadata?.height === null ||
              metadata?.height === undefined ||
              metadata?.width === null ||
              metadata?.width === undefined
            }
            onClick={handleUseSize}
          />
          <IconButton
            isLoading={isLoadingMetadata}
            icon={<PiAsteriskBold />}
            tooltip={`${t('parameters.useAll')} (A)`}
            aria-label={`${t('parameters.useAll')} (A)`}
            isDisabled={!metadata}
            onClick={handleClickUseAllParameters}
          />
        </ButtonGroup>

        {isUpscalingEnabled && (
          <ButtonGroup isDisabled={isQueueMutationInProgress}>
            {isUpscalingEnabled && <ParamUpscalePopover imageDTO={imageDTO} />}
          </ButtonGroup>
        )}

        <ButtonGroup>
          <IconButton
            icon={<PiInfoBold />}
            tooltip={`${t('parameters.info')} (I)`}
            aria-label={`${t('parameters.info')} (I)`}
            isChecked={shouldShowImageDetails}
            onClick={handleClickShowImageDetails}
          />
        </ButtonGroup>

        <ButtonGroup>
          <IconButton
            aria-label={t('settings.displayInProgress')}
            tooltip={t('settings.displayInProgress')}
            icon={<PiHourglassHighBold />}
            isChecked={shouldShowProgressInViewer}
            onClick={handleClickProgressImagesToggle}
          />
        </ButtonGroup>

        <ButtonGroup>
          <DeleteImageButton onClick={handleDelete} />
        </ButtonGroup>
      </Flex>
    </>
  );
};

export default memo(CurrentImageButtons);

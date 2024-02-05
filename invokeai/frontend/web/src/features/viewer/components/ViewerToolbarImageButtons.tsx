import { ButtonGroup, IconButton } from '@invoke-ai/ui-library';
import { upscaleRequested } from 'app/store/middleware/listenerMiddleware/listeners/upscaleRequested';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { DeleteImageButton } from 'features/deleteImageModal/components/DeleteImageButton';
import { imagesToDeleteSelected } from 'features/deleteImageModal/store/slice';
import { sentImageToImg2Img } from 'features/gallery/store/actions';
import ParamUpscalePopover from 'features/parameters/components/Upscale/ParamUpscaleSettings';
import { useRecallParameters } from 'features/parameters/hooks/useRecallParameters';
import { initialImageSelected } from 'features/parameters/store/actions';
import { useIsQueueMutationInProgress } from 'features/queue/hooks/useIsQueueMutationInProgress';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { useCurrentImageDTO } from 'features/viewer/hooks/useCurrentImageDTO';
import { useGetAndLoadEmbeddedWorkflow } from 'features/workflowLibrary/hooks/useGetAndLoadEmbeddedWorkflow';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import {
  PiArrowsCounterClockwiseBold,
  PiAsteriskBold,
  PiFlowArrowBold,
  PiPlantBold,
  PiQuotesBold,
  PiRulerBold,
} from 'react-icons/pi';
import { useDebouncedMetadata } from 'services/api/hooks/useDebouncedMetadata';

export const ViewerToolbarImageButtons = memo(() => {
  const dispatch = useAppDispatch();
  const isConnected = useAppSelector((s) => s.system.isConnected);
  const isUpscalingEnabled = useFeatureStatus('upscaling').isFeatureEnabled;
  const isQueueMutationInProgress = useIsQueueMutationInProgress();
  const { t } = useTranslation();
  const imageDTO = useCurrentImageDTO();

  const { recallBothPrompts, recallSeed, recallWidthAndHeight, recallAllParameters } = useRecallParameters();

  const { metadata, isLoading: isLoadingMetadata } = useDebouncedMetadata(imageDTO?.image_name);

  const { getAndLoadEmbeddedWorkflow, getAndLoadEmbeddedWorkflowResult } = useGetAndLoadEmbeddedWorkflow({});

  const handleLoadWorkflow = useCallback(() => {
    if (!imageDTO || !imageDTO.has_workflow) {
      return;
    }
    getAndLoadEmbeddedWorkflow(imageDTO.image_name);
  }, [getAndLoadEmbeddedWorkflow, imageDTO]);

  useHotkeys('w', handleLoadWorkflow, [imageDTO]);

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
      enabled: () => Boolean(isUpscalingEnabled && imageDTO && isConnected),
    },
    [isUpscalingEnabled, imageDTO, imageDTO, isConnected]
  );

  useHotkeys(
    'delete',
    () => {
      handleDelete();
    },
    [dispatch, imageDTO]
  );

  return (
    <>
      <ButtonGroup>
        <IconButton
          icon={<PiFlowArrowBold />}
          tooltip={`${t('nodes.loadWorkflow')} (W)`}
          aria-label={`${t('nodes.loadWorkflow')} (W)`}
          isDisabled={!imageDTO || !imageDTO?.has_workflow}
          onClick={handleLoadWorkflow}
          isLoading={getAndLoadEmbeddedWorkflowResult.isLoading}
        />
        <IconButton
          isLoading={isLoadingMetadata}
          icon={<PiArrowsCounterClockwiseBold />}
          tooltip={`${t('parameters.remixImage')} (R)`}
          aria-label={`${t('parameters.remixImage')} (R)`}
          isDisabled={!imageDTO || !metadata?.positive_prompt}
          onClick={handleRemixImage}
        />
        <IconButton
          isLoading={isLoadingMetadata}
          icon={<PiQuotesBold />}
          tooltip={`${t('parameters.usePrompt')} (P)`}
          aria-label={`${t('parameters.usePrompt')} (P)`}
          isDisabled={!imageDTO || !metadata?.positive_prompt}
          onClick={handleUsePrompt}
        />
        <IconButton
          isLoading={isLoadingMetadata}
          icon={<PiPlantBold />}
          tooltip={`${t('parameters.useSeed')} (S)`}
          aria-label={`${t('parameters.useSeed')} (S)`}
          isDisabled={!imageDTO || metadata?.seed === null || metadata?.seed === undefined}
          onClick={handleUseSeed}
        />
        <IconButton
          isLoading={isLoadingMetadata}
          icon={<PiRulerBold />}
          tooltip={`${t('parameters.useSize')} (D)`}
          aria-label={`${t('parameters.useSize')} (D)`}
          isDisabled={
            !imageDTO ||
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
          isDisabled={!imageDTO || !metadata}
          onClick={handleClickUseAllParameters}
        />
      </ButtonGroup>

      {isUpscalingEnabled && (
        <ButtonGroup isDisabled={isQueueMutationInProgress || !imageDTO}>
          {isUpscalingEnabled && <ParamUpscalePopover imageDTO={imageDTO} />}
        </ButtonGroup>
      )}
      <DeleteImageButton onClick={handleDelete} isDisabled={!imageDTO} />
    </>
  );
});

ViewerToolbarImageButtons.displayName = 'ViewerToolbarImageButtons';

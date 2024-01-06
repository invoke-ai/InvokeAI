import { Flex, Spinner } from '@chakra-ui/react';
import { useStore } from '@nanostores/react';
import { useAppToaster } from 'app/components/Toaster';
import { $customStarUI } from 'app/store/nanostores/customStarUI';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvMenuItem } from 'common/components/InvMenu/InvMenuItem';
import { InvMenuDivider } from 'common/components/InvMenu/wrapper'
import { useCopyImageToClipboard } from 'common/hooks/useCopyImageToClipboard';
import { setInitialCanvasImage } from 'features/canvas/store/canvasSlice';
import {
  imagesToChangeSelected,
  isModalOpenChanged,
} from 'features/changeBoardModal/store/slice';
import { imagesToDeleteSelected } from 'features/deleteImageModal/store/slice';
import {
  sentImageToCanvas,
  sentImageToImg2Img,
} from 'features/gallery/store/actions';
import { useRecallParameters } from 'features/parameters/hooks/useRecallParameters';
import { initialImageSelected } from 'features/parameters/store/actions';
import { selectOptimalDimension } from 'features/parameters/store/generationSlice';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { useGetAndLoadEmbeddedWorkflow } from 'features/workflowLibrary/hooks/useGetAndLoadEmbeddedWorkflow';
import { memo, useCallback } from 'react';
import { flushSync } from 'react-dom';
import { useTranslation } from 'react-i18next';
import { PiAsteriskBold, PiCopyBold, PiDownloadSimpleBold, PiFlowArrowBold, PiFoldersBold, PiPlantBold, PiQuotesBold, PiShareFatBold, PiStarBold, PiStarFill, PiTrashSimpleBold } from 'react-icons/pi'
import {
  useStarImagesMutation,
  useUnstarImagesMutation,
} from 'services/api/endpoints/images';
import { useDebouncedMetadata } from 'services/api/hooks/useDebouncedMetadata';
import type { ImageDTO } from 'services/api/types';

type SingleSelectionMenuItemsProps = {
  imageDTO: ImageDTO;
};

const SingleSelectionMenuItems = (props: SingleSelectionMenuItemsProps) => {
  const { imageDTO } = props;
  const optimalDimension = useAppSelector(selectOptimalDimension);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const toaster = useAppToaster();
  const isCanvasEnabled = useFeatureStatus('unifiedCanvas').isFeatureEnabled;
  const customStarUi = useStore($customStarUI);

  const { metadata, isLoading: isLoadingMetadata } = useDebouncedMetadata(
    imageDTO?.image_name
  );

  const { getAndLoadEmbeddedWorkflow, getAndLoadEmbeddedWorkflowResult } =
    useGetAndLoadEmbeddedWorkflow({});

  const handleLoadWorkflow = useCallback(() => {
    getAndLoadEmbeddedWorkflow(imageDTO.image_name);
  }, [getAndLoadEmbeddedWorkflow, imageDTO.image_name]);

  const [starImages] = useStarImagesMutation();
  const [unstarImages] = useUnstarImagesMutation();

  const { isClipboardAPIAvailable, copyImageToClipboard } =
    useCopyImageToClipboard();

  const handleDelete = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    dispatch(imagesToDeleteSelected([imageDTO]));
  }, [dispatch, imageDTO]);

  const { recallBothPrompts, recallSeed, recallAllParameters } =
    useRecallParameters();

  // Recall parameters handlers
  const handleRecallPrompt = useCallback(() => {
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

  const handleRecallSeed = useCallback(() => {
    recallSeed(metadata?.seed);
  }, [metadata?.seed, recallSeed]);

  const handleSendToImageToImage = useCallback(() => {
    dispatch(sentImageToImg2Img());
    dispatch(initialImageSelected(imageDTO));
  }, [dispatch, imageDTO]);

  const handleSendToCanvas = useCallback(() => {
    dispatch(sentImageToCanvas());
    flushSync(() => {
      dispatch(setActiveTab('unifiedCanvas'));
    });
    dispatch(setInitialCanvasImage(imageDTO, optimalDimension));

    toaster({
      title: t('toast.sentToUnifiedCanvas'),
      status: 'success',
      duration: 2500,
      isClosable: true,
    });
  }, [dispatch, imageDTO, t, toaster, optimalDimension]);

  const handleUseAllParameters = useCallback(() => {
    recallAllParameters(metadata);
  }, [metadata, recallAllParameters]);

  const handleChangeBoard = useCallback(() => {
    dispatch(imagesToChangeSelected([imageDTO]));
    dispatch(isModalOpenChanged(true));
  }, [dispatch, imageDTO]);

  const handleCopyImage = useCallback(() => {
    copyImageToClipboard(imageDTO.image_url);
  }, [copyImageToClipboard, imageDTO.image_url]);

  const handleStarImage = useCallback(() => {
    if (imageDTO) {
      starImages({ imageDTOs: [imageDTO] });
    }
  }, [starImages, imageDTO]);

  const handleUnstarImage = useCallback(() => {
    if (imageDTO) {
      unstarImages({ imageDTOs: [imageDTO] });
    }
  }, [unstarImages, imageDTO]);

  return (
    <>
      <InvMenuItem
        as="a"
        href={imageDTO.image_url}
        target="_blank"
        icon={<PiShareFatBold />}
      >
        {t('common.openInNewTab')}
      </InvMenuItem>
      {isClipboardAPIAvailable && (
        <InvMenuItem icon={<PiCopyBold />} onClickCapture={handleCopyImage}>
          {t('parameters.copyImage')}
        </InvMenuItem>
      )}
      <InvMenuItem
        as="a"
        download={true}
        href={imageDTO.image_url}
        target="_blank"
        icon={<PiDownloadSimpleBold />}
        w="100%"
      >
        {t('parameters.downloadImage')}
      </InvMenuItem>
      <InvMenuDivider />
      <InvMenuItem
        icon={
          getAndLoadEmbeddedWorkflowResult.isLoading ? (
            <SpinnerIcon />
          ) : (
            <PiFlowArrowBold />
          )
        }
        onClickCapture={handleLoadWorkflow}
        isDisabled={!imageDTO.has_workflow}
      >
        {t('nodes.loadWorkflow')}
      </InvMenuItem>
      <InvMenuItem
        icon={isLoadingMetadata ? <SpinnerIcon /> : <PiQuotesBold />}
        onClickCapture={handleRecallPrompt}
        isDisabled={
          isLoadingMetadata ||
          (metadata?.positive_prompt === undefined &&
            metadata?.negative_prompt === undefined)
        }
      >
        {t('parameters.usePrompt')}
      </InvMenuItem>
      <InvMenuItem
        icon={isLoadingMetadata ? <SpinnerIcon /> : <PiPlantBold />}
        onClickCapture={handleRecallSeed}
        isDisabled={isLoadingMetadata || metadata?.seed === undefined}
      >
        {t('parameters.useSeed')}
      </InvMenuItem>
      <InvMenuItem
        icon={isLoadingMetadata ? <SpinnerIcon /> : <PiAsteriskBold />}
        onClickCapture={handleUseAllParameters}
        isDisabled={isLoadingMetadata || !metadata}
      >
        {t('parameters.useAll')}
      </InvMenuItem>
      <InvMenuDivider />
      <InvMenuItem
        icon={<PiShareFatBold />}
        onClickCapture={handleSendToImageToImage}
        id="send-to-img2img"
      >
        {t('parameters.sendToImg2Img')}
      </InvMenuItem>
      {isCanvasEnabled && (
        <InvMenuItem
          icon={<PiShareFatBold />}
          onClickCapture={handleSendToCanvas}
          id="send-to-canvas"
        >
          {t('parameters.sendToUnifiedCanvas')}
        </InvMenuItem>
      )}
      <InvMenuDivider />
      <InvMenuItem icon={<PiFoldersBold />} onClickCapture={handleChangeBoard}>
        {t('boards.changeBoard')}
      </InvMenuItem>
      {imageDTO.starred ? (
        <InvMenuItem
          icon={customStarUi ? customStarUi.off.icon : <PiStarFill />}
          onClickCapture={handleUnstarImage}
        >
          {customStarUi ? customStarUi.off.text : t('gallery.unstarImage')}
        </InvMenuItem>
      ) : (
        <InvMenuItem
          icon={customStarUi ? customStarUi.on.icon : <PiStarBold />}
          onClickCapture={handleStarImage}
        >
          {customStarUi ? customStarUi.on.text : t('gallery.starImage')}
        </InvMenuItem>
      )}
      <InvMenuDivider />
      <InvMenuItem
        color="error.300"
        icon={<PiTrashSimpleBold />}
        onClickCapture={handleDelete}
      >
        {t('gallery.deleteImage')}
      </InvMenuItem>
    </>
  );
};

export default memo(SingleSelectionMenuItems);

const SpinnerIcon = () => (
  <Flex w="14px" alignItems="center" justifyContent="center">
    <Spinner size="xs" />
  </Flex>
);

import { Flex, MenuDivider, MenuItem, Spinner } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppToaster } from 'app/components/Toaster';
import { $customStarUI } from 'app/store/nanostores/customStarUI';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useCopyImageToClipboard } from 'common/hooks/useCopyImageToClipboard';
import { useDownloadImage } from 'common/hooks/useDownloadImage';
import { setInitialCanvasImage } from 'features/canvas/store/canvasSlice';
import { imagesToChangeSelected, isModalOpenChanged } from 'features/changeBoardModal/store/slice';
import { imagesToDeleteSelected } from 'features/deleteImageModal/store/slice';
import { useImageActions } from 'features/gallery/hooks/useImageActions';
import { sentImageToCanvas, sentImageToImg2Img } from 'features/gallery/store/actions';
import { initialImageSelected } from 'features/parameters/store/actions';
import { selectOptimalDimension } from 'features/parameters/store/generationSlice';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { useGetAndLoadEmbeddedWorkflow } from 'features/workflowLibrary/hooks/useGetAndLoadEmbeddedWorkflow';
import { memo, useCallback } from 'react';
import { flushSync } from 'react-dom';
import { useTranslation } from 'react-i18next';
import {
  PiArrowsCounterClockwiseBold,
  PiAsteriskBold,
  PiCopyBold,
  PiDownloadSimpleBold,
  PiFlowArrowBold,
  PiFoldersBold,
  PiPlantBold,
  PiQuotesBold,
  PiShareFatBold,
  PiStarBold,
  PiStarFill,
  PiTrashSimpleBold,
} from 'react-icons/pi';
import { useStarImagesMutation, useUnstarImagesMutation } from 'services/api/endpoints/images';
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
  const { downloadImage } = useDownloadImage();

  const { recallAll, remix, recallSeed, recallPrompts, hasMetadata, hasSeed, hasPrompts, isLoadingMetadata } =
    useImageActions(imageDTO?.image_name);

  const { getAndLoadEmbeddedWorkflow, getAndLoadEmbeddedWorkflowResult } = useGetAndLoadEmbeddedWorkflow({});

  const handleLoadWorkflow = useCallback(() => {
    getAndLoadEmbeddedWorkflow(imageDTO.image_name);
  }, [getAndLoadEmbeddedWorkflow, imageDTO.image_name]);

  const [starImages] = useStarImagesMutation();
  const [unstarImages] = useUnstarImagesMutation();

  const { isClipboardAPIAvailable, copyImageToClipboard } = useCopyImageToClipboard();

  const handleDelete = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    dispatch(imagesToDeleteSelected([imageDTO]));
  }, [dispatch, imageDTO]);

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

  const handleDownloadImage = useCallback(() => {
    downloadImage(imageDTO.image_url, imageDTO.image_name);
  }, [downloadImage, imageDTO.image_name, imageDTO.image_url]);

  return (
    <>
      <MenuItem as="a" href={imageDTO.image_url} target="_blank" icon={<PiShareFatBold />}>
        {t('common.openInNewTab')}
      </MenuItem>
      {isClipboardAPIAvailable && (
        <MenuItem icon={<PiCopyBold />} onClickCapture={handleCopyImage}>
          {t('parameters.copyImage')}
        </MenuItem>
      )}
      <MenuItem icon={<PiDownloadSimpleBold />} onClickCapture={handleDownloadImage}>
        {t('parameters.downloadImage')}
      </MenuItem>
      <MenuDivider />
      <MenuItem
        icon={getAndLoadEmbeddedWorkflowResult.isLoading ? <SpinnerIcon /> : <PiFlowArrowBold />}
        onClickCapture={handleLoadWorkflow}
        isDisabled={!imageDTO.has_workflow}
      >
        {t('nodes.loadWorkflow')}
      </MenuItem>
      <MenuItem
        icon={isLoadingMetadata ? <SpinnerIcon /> : <PiArrowsCounterClockwiseBold />}
        onClickCapture={remix}
        isDisabled={isLoadingMetadata || !hasMetadata}
      >
        {t('parameters.remixImage')}
      </MenuItem>
      <MenuItem
        icon={isLoadingMetadata ? <SpinnerIcon /> : <PiQuotesBold />}
        onClickCapture={recallPrompts}
        isDisabled={isLoadingMetadata || !hasPrompts}
      >
        {t('parameters.usePrompt')}
      </MenuItem>
      <MenuItem
        icon={isLoadingMetadata ? <SpinnerIcon /> : <PiPlantBold />}
        onClickCapture={recallSeed}
        isDisabled={isLoadingMetadata || !hasSeed}
      >
        {t('parameters.useSeed')}
      </MenuItem>
      <MenuItem
        icon={isLoadingMetadata ? <SpinnerIcon /> : <PiAsteriskBold />}
        onClickCapture={recallAll}
        isDisabled={isLoadingMetadata || !hasMetadata}
      >
        {t('parameters.useAll')}
      </MenuItem>
      <MenuDivider />
      <MenuItem icon={<PiShareFatBold />} onClickCapture={handleSendToImageToImage} id="send-to-img2img">
        {t('parameters.sendToImg2Img')}
      </MenuItem>
      {isCanvasEnabled && (
        <MenuItem icon={<PiShareFatBold />} onClickCapture={handleSendToCanvas} id="send-to-canvas">
          {t('parameters.sendToUnifiedCanvas')}
        </MenuItem>
      )}
      <MenuDivider />
      <MenuItem icon={<PiFoldersBold />} onClickCapture={handleChangeBoard}>
        {t('boards.changeBoard')}
      </MenuItem>
      {imageDTO.starred ? (
        <MenuItem icon={customStarUi ? customStarUi.off.icon : <PiStarFill />} onClickCapture={handleUnstarImage}>
          {customStarUi ? customStarUi.off.text : t('gallery.unstarImage')}
        </MenuItem>
      ) : (
        <MenuItem icon={customStarUi ? customStarUi.on.icon : <PiStarBold />} onClickCapture={handleStarImage}>
          {customStarUi ? customStarUi.on.text : t('gallery.starImage')}
        </MenuItem>
      )}
      <MenuDivider />
      <MenuItem color="error.300" icon={<PiTrashSimpleBold />} onClickCapture={handleDelete}>
        {t('gallery.deleteImage')}
      </MenuItem>
    </>
  );
};

export default memo(SingleSelectionMenuItems);

const SpinnerIcon = () => (
  <Flex w="14px" alignItems="center" justifyContent="center">
    <Spinner size="xs" />
  </Flex>
);

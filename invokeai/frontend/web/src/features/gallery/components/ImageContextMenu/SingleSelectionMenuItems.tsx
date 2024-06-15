import { Flex, MenuDivider, MenuItem, Spinner } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $customStarUI } from 'app/store/nanostores/customStarUI';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useCopyImageToClipboard } from 'common/hooks/useCopyImageToClipboard';
import { useDownloadImage } from 'common/hooks/useDownloadImage';
import { setInitialCanvasImage } from 'features/canvas/store/canvasSlice';
import { imagesToChangeSelected, isModalOpenChanged } from 'features/changeBoardModal/store/slice';
import { iiLayerAdded } from 'features/controlLayers/store/canvasV2Slice';
import { selectOptimalDimension } from 'features/controlLayers/store/selectors';
import { imagesToDeleteSelected } from 'features/deleteImageModal/store/slice';
import { useImageActions } from 'features/gallery/hooks/useImageActions';
import { sentImageToCanvas, sentImageToImg2Img } from 'features/gallery/store/actions';
import { imageToCompareChanged } from 'features/gallery/store/gallerySlice';
import { $templates } from 'features/nodes/store/nodesSlice';
import { selectOptimalDimension } from 'features/parameters/store/generationSlice';
import { upscaleInitialImageChanged } from 'features/parameters/store/upscaleSlice';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { toast } from 'features/toast/toast';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { useGetAndLoadEmbeddedWorkflow } from 'features/workflowLibrary/hooks/useGetAndLoadEmbeddedWorkflow';
import { size } from 'lodash-es';
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
  PiImagesBold,
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
  const maySelectForCompare = useAppSelector((s) => s.gallery.imageToCompare?.image_name !== imageDTO.image_name);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const isCanvasEnabled = useFeatureStatus('canvas');
  const customStarUi = useStore($customStarUI);
  const { downloadImage } = useDownloadImage();
  const templates = useStore($templates);

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
    dispatch(iiLayerAdded(imageDTO));
    dispatch(setActiveTab('generation'));
  }, [dispatch, imageDTO]);

  const handleSendToCanvas = useCallback(() => {
    dispatch(sentImageToCanvas());
    flushSync(() => {
      dispatch(setActiveTab('canvas'));
    });
    dispatch(setInitialCanvasImage(imageDTO, optimalDimension));

    toast({
      id: 'SENT_TO_CANVAS',
      title: t('toast.sentToUnifiedCanvas'),
      status: 'success',
    });
  }, [dispatch, imageDTO, t, optimalDimension]);

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

  const handleSelectImageForCompare = useCallback(() => {
    dispatch(imageToCompareChanged(imageDTO));
  }, [dispatch, imageDTO]);

  const handleSendToUpscale = useCallback(() => {
    dispatch(upscaleInitialImageChanged(imageDTO));
    dispatch(setActiveTab('upscaling'));
  }, [dispatch, imageDTO]);

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
      <MenuItem icon={<PiImagesBold />} isDisabled={!maySelectForCompare} onClick={handleSelectImageForCompare}>
        {t('gallery.selectForCompare')}
      </MenuItem>
      <MenuDivider />
      <MenuItem
        icon={getAndLoadEmbeddedWorkflowResult.isLoading ? <SpinnerIcon /> : <PiFlowArrowBold />}
        onClickCapture={handleLoadWorkflow}
        isDisabled={!imageDTO.has_workflow || !size(templates)}
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
      <MenuItem icon={<PiShareFatBold />} onClickCapture={handleSendToUpscale} id="send-to-upscale">
        {t('parameters.sendToUpscale')}
      </MenuItem>
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
        {t('gallery.deleteImage', { count: 1 })}
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

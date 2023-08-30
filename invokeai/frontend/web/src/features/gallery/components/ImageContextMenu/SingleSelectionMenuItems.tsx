import { Flex, MenuItem, Text } from '@chakra-ui/react';
import { skipToken } from '@reduxjs/toolkit/dist/query';
import { useAppToaster } from 'app/components/Toaster';
import { useAppDispatch } from 'app/store/storeHooks';
import { setInitialCanvasImage } from 'features/canvas/store/canvasSlice';
import {
  imagesToChangeSelected,
  isModalOpenChanged,
} from 'features/changeBoardModal/store/slice';
import { imagesToDeleteSelected } from 'features/deleteImageModal/store/slice';
import { useRecallParameters } from 'features/parameters/hooks/useRecallParameters';
import { initialImageSelected } from 'features/parameters/store/actions';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { useCopyImageToClipboard } from 'features/ui/hooks/useCopyImageToClipboard';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import {
  FaAsterisk,
  FaCopy,
  FaDownload,
  FaExternalLinkAlt,
  FaFolder,
  FaQuoteRight,
  FaSeedling,
  FaShare,
  FaTrash,
} from 'react-icons/fa';
import { MdStar, MdStarBorder } from 'react-icons/md';
import {
  useGetImageMetadataQuery,
  useStarImagesMutation,
  useUnstarImagesMutation,
} from 'services/api/endpoints/images';
import { ImageDTO } from 'services/api/types';
import { useDebounce } from 'use-debounce';
import { sentImageToCanvas, sentImageToImg2Img } from '../../store/actions';

type SingleSelectionMenuItemsProps = {
  imageDTO: ImageDTO;
};

const SingleSelectionMenuItems = (props: SingleSelectionMenuItemsProps) => {
  const { imageDTO } = props;

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const toaster = useAppToaster();

  const isCanvasEnabled = useFeatureStatus('unifiedCanvas').isFeatureEnabled;

  const [debouncedMetadataQueryArg, debounceState] = useDebounce(
    imageDTO.image_name,
    500
  );

  const { currentData } = useGetImageMetadataQuery(
    debounceState.isPending()
      ? skipToken
      : debouncedMetadataQueryArg ?? skipToken
  );

  const [starImages] = useStarImagesMutation();
  const [unstarImages] = useUnstarImagesMutation();

  const { isClipboardAPIAvailable, copyImageToClipboard } =
    useCopyImageToClipboard();

  const metadata = currentData?.metadata;

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
    dispatch(setInitialCanvasImage(imageDTO));
    dispatch(setActiveTab('unifiedCanvas'));

    toaster({
      title: t('toast.sentToUnifiedCanvas'),
      status: 'success',
      duration: 2500,
      isClosable: true,
    });
  }, [dispatch, imageDTO, t, toaster]);

  const handleUseAllParameters = useCallback(() => {
    console.log(metadata);
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
      <MenuItem
        as="a"
        href={imageDTO.image_url}
        target="_blank"
        icon={<FaExternalLinkAlt />}
      >
        {t('common.openInNewTab')}
      </MenuItem>
      {isClipboardAPIAvailable && (
        <MenuItem icon={<FaCopy />} onClickCapture={handleCopyImage}>
          {t('parameters.copyImage')}
        </MenuItem>
      )}
      <MenuItem
        as="a"
        download={true}
        href={imageDTO.image_url}
        target="_blank"
        icon={<FaDownload />}
        w="100%"
      >
        {t('parameters.downloadImage')}
      </MenuItem>
      <MenuItem
        icon={<FaQuoteRight />}
        onClickCapture={handleRecallPrompt}
        isDisabled={
          metadata?.positive_prompt === undefined &&
          metadata?.negative_prompt === undefined
        }
      >
        {t('parameters.usePrompt')}
      </MenuItem>

      <MenuItem
        icon={<FaSeedling />}
        onClickCapture={handleRecallSeed}
        isDisabled={metadata?.seed === undefined}
      >
        {t('parameters.useSeed')}
      </MenuItem>
      <MenuItem
        icon={<FaAsterisk />}
        onClickCapture={handleUseAllParameters}
        isDisabled={!metadata}
      >
        {t('parameters.useAll')}
      </MenuItem>
      <MenuItem
        icon={<FaShare />}
        onClickCapture={handleSendToImageToImage}
        id="send-to-img2img"
      >
        {t('parameters.sendToImg2Img')}
      </MenuItem>
      {isCanvasEnabled && (
        <MenuItem
          icon={<FaShare />}
          onClickCapture={handleSendToCanvas}
          id="send-to-canvas"
        >
          {t('parameters.sendToUnifiedCanvas')}
        </MenuItem>
      )}
      <MenuItem icon={<FaFolder />} onClickCapture={handleChangeBoard}>
        Change Board
      </MenuItem>
      {imageDTO.starred ? (
        <MenuItem icon={<MdStar />} onClickCapture={handleUnstarImage}>
          Unstar Image
        </MenuItem>
      ) : (
        <MenuItem icon={<MdStarBorder />} onClickCapture={handleStarImage}>
          Star Image
        </MenuItem>
      )}
      <MenuItem
        sx={{ color: 'error.600', _dark: { color: 'error.300' } }}
        icon={<FaTrash />}
        onClickCapture={handleDelete}
      >
        {t('gallery.deleteImage')}
      </MenuItem>
      {metadata?.created_by && (
        <Flex
          sx={{
            padding: '5px 10px',
            marginTop: '5px',
          }}
        >
          <Text fontSize="xs" fontWeight="bold">
            Created by {metadata?.created_by}
          </Text>
        </Flex>
      )}
    </>
  );
};

export default memo(SingleSelectionMenuItems);

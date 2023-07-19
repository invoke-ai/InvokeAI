import { Link, MenuItem } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppToaster } from 'app/components/Toaster';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import {
  resizeAndScaleCanvas,
  setInitialCanvasImage,
} from 'features/canvas/store/canvasSlice';
import { imagesAddedToBatch } from 'features/gallery/store/gallerySlice';
import { imageToDeleteSelected } from 'features/imageDeletion/store/imageDeletionSlice';
import { useRecallParameters } from 'features/parameters/hooks/useRecallParameters';
import { initialImageSelected } from 'features/parameters/store/actions';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { useCopyImageToClipboard } from 'features/ui/hooks/useCopyImageToClipboard';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { memo, useCallback, useContext, useMemo } from 'react';
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
import {
  useGetImageMetadataQuery,
  useRemoveImageFromBoardMutation,
} from 'services/api/endpoints/images';
import { ImageDTO } from 'services/api/types';
import { AddImageToBoardContext } from '../../../../app/contexts/AddImageToBoardContext';
import { sentImageToCanvas, sentImageToImg2Img } from '../../store/actions';

type SingleSelectionMenuItemsProps = {
  imageDTO: ImageDTO;
};

const SingleSelectionMenuItems = (props: SingleSelectionMenuItemsProps) => {
  const { imageDTO } = props;

  const selector = useMemo(
    () =>
      createSelector(
        [stateSelector],
        ({ gallery }) => {
          const isInBatch = gallery.batchImageNames.includes(
            imageDTO.image_name
          );

          return { isInBatch };
        },
        defaultSelectorOptions
      ),
    [imageDTO.image_name]
  );

  const { isInBatch } = useAppSelector(selector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const toaster = useAppToaster();

  const isCanvasEnabled = useFeatureStatus('unifiedCanvas').isFeatureEnabled;
  const isBatchEnabled = useFeatureStatus('batches').isFeatureEnabled;

  const { onClickAddToBoard } = useContext(AddImageToBoardContext);

  const { currentData } = useGetImageMetadataQuery(imageDTO.image_name);

  const { isClipboardAPIAvailable, copyImageToClipboard } =
    useCopyImageToClipboard();

  const metadata = currentData?.metadata;

  const handleDelete = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    dispatch(imageToDeleteSelected(imageDTO));
  }, [dispatch, imageDTO]);

  const { recallBothPrompts, recallSeed, recallAllParameters } =
    useRecallParameters();

  const [removeFromBoard] = useRemoveImageFromBoardMutation();

  // Recall parameters handlers
  const handleRecallPrompt = useCallback(() => {
    recallBothPrompts(metadata?.positive_prompt, metadata?.negative_prompt);
  }, [metadata?.negative_prompt, metadata?.positive_prompt, recallBothPrompts]);

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
    dispatch(resizeAndScaleCanvas());
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

  const handleAddToBoard = useCallback(() => {
    onClickAddToBoard(imageDTO);
  }, [imageDTO, onClickAddToBoard]);

  const handleRemoveFromBoard = useCallback(() => {
    if (!imageDTO.board_id) {
      return;
    }
    removeFromBoard({ imageDTO });
  }, [imageDTO, removeFromBoard]);

  const handleAddToBatch = useCallback(() => {
    dispatch(imagesAddedToBatch([imageDTO.image_name]));
  }, [dispatch, imageDTO.image_name]);

  const handleCopyImage = useCallback(() => {
    copyImageToClipboard(imageDTO.image_url);
  }, [copyImageToClipboard, imageDTO.image_url]);

  return (
    <>
      <Link href={imageDTO.image_url} target="_blank">
        <MenuItem icon={<FaExternalLinkAlt />}>
          {t('common.openInNewTab')}
        </MenuItem>
      </Link>
      {isClipboardAPIAvailable && (
        <MenuItem icon={<FaCopy />} onClickCapture={handleCopyImage}>
          {t('parameters.copyImage')}
        </MenuItem>
      )}
      <Link download={true} href={imageDTO.image_url} target="_blank">
        <MenuItem icon={<FaDownload />} w="100%">
          {t('parameters.downloadImage')}
        </MenuItem>
      </Link>
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
      {isBatchEnabled && (
        <MenuItem
          icon={<FaFolder />}
          isDisabled={isInBatch}
          onClickCapture={handleAddToBatch}
        >
          Add to Batch
        </MenuItem>
      )}
      <MenuItem icon={<FaFolder />} onClickCapture={handleAddToBoard}>
        {imageDTO.board_id ? 'Change Board' : 'Add to Board'}
      </MenuItem>
      {imageDTO.board_id && (
        <MenuItem icon={<FaFolder />} onClickCapture={handleRemoveFromBoard}>
          Remove from Board
        </MenuItem>
      )}
      <MenuItem
        sx={{ color: 'error.600', _dark: { color: 'error.300' } }}
        icon={<FaTrash />}
        onClickCapture={handleDelete}
      >
        {t('gallery.deleteImage')}
      </MenuItem>
    </>
  );
};

export default memo(SingleSelectionMenuItems);

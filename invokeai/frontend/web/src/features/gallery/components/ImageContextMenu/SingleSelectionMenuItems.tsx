import { Flex, MenuItem, Spinner } from '@chakra-ui/react';
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
import { MdDeviceHub, MdStar, MdStarBorder } from 'react-icons/md';
import {
  useGetImageMetadataFromFileQuery,
  useStarImagesMutation,
  useUnstarImagesMutation,
} from 'services/api/endpoints/images';
import { ImageDTO } from 'services/api/types';
import { sentImageToCanvas, sentImageToImg2Img } from '../../store/actions';
import { workflowLoadRequested } from 'features/nodes/store/actions';

type SingleSelectionMenuItemsProps = {
  imageDTO: ImageDTO;
};

const SingleSelectionMenuItems = (props: SingleSelectionMenuItemsProps) => {
  const { imageDTO } = props;

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const toaster = useAppToaster();

  const isCanvasEnabled = useFeatureStatus('unifiedCanvas').isFeatureEnabled;

  const { metadata, workflow, isLoading } = useGetImageMetadataFromFileQuery(
    imageDTO,
    {
      selectFromResult: (res) => ({
        isLoading: res.isFetching,
        metadata: res?.currentData?.metadata,
        workflow: res?.currentData?.workflow,
      }),
    }
  );

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

  const handleLoadWorkflow = useCallback(() => {
    if (!workflow) {
      return;
    }
    dispatch(workflowLoadRequested(workflow));
  }, [dispatch, workflow]);

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
        icon={isLoading ? <SpinnerIcon /> : <MdDeviceHub />}
        onClickCapture={handleLoadWorkflow}
        isDisabled={isLoading || !workflow}
      >
        {t('nodes.loadWorkflow')}
      </MenuItem>
      <MenuItem
        icon={isLoading ? <SpinnerIcon /> : <FaQuoteRight />}
        onClickCapture={handleRecallPrompt}
        isDisabled={
          isLoading ||
          (metadata?.positive_prompt === undefined &&
            metadata?.negative_prompt === undefined)
        }
      >
        {t('parameters.usePrompt')}
      </MenuItem>
      <MenuItem
        icon={isLoading ? <SpinnerIcon /> : <FaSeedling />}
        onClickCapture={handleRecallSeed}
        isDisabled={isLoading || metadata?.seed === undefined}
      >
        {t('parameters.useSeed')}
      </MenuItem>
      <MenuItem
        icon={isLoading ? <SpinnerIcon /> : <FaAsterisk />}
        onClickCapture={handleUseAllParameters}
        isDisabled={isLoading || !metadata}
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
    </>
  );
};

export default memo(SingleSelectionMenuItems);

const SpinnerIcon = () => (
  <Flex w="14px" alignItems="center" justifyContent="center">
    <Spinner size="xs" />
  </Flex>
);

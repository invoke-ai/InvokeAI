import { Flex, MenuItem, Spinner } from '@chakra-ui/react';
import { useStore } from '@nanostores/react';
import { useAppToaster } from 'app/components/Toaster';
import { $customStarUI } from 'app/store/nanostores/customStarUI';
import { useAppDispatch } from 'app/store/storeHooks';
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
import { workflowLoadRequested } from 'features/nodes/store/actions';
import { useRecallParameters } from 'features/parameters/hooks/useRecallParameters';
import { initialImageSelected } from 'features/parameters/store/actions';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { memo, useCallback } from 'react';
import { flushSync } from 'react-dom';
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
import { FaCircleNodes } from 'react-icons/fa6';
import { MdStar, MdStarBorder } from 'react-icons/md';
import {
  useLazyGetImageWorkflowQuery,
  useStarImagesMutation,
  useUnstarImagesMutation,
} from 'services/api/endpoints/images';
import { useDebouncedMetadata } from 'services/api/hooks/useDebouncedMetadata';
import { ImageDTO } from 'services/api/types';

type SingleSelectionMenuItemsProps = {
  imageDTO: ImageDTO;
};

const SingleSelectionMenuItems = (props: SingleSelectionMenuItemsProps) => {
  const { imageDTO } = props;

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const toaster = useAppToaster();

  const isCanvasEnabled = useFeatureStatus('unifiedCanvas').isFeatureEnabled;
  const customStarUi = useStore($customStarUI);

  const { metadata, isLoading: isLoadingMetadata } = useDebouncedMetadata(
    imageDTO?.image_name
  );

  const [getWorkflow, getWorkflowResult] = useLazyGetImageWorkflowQuery();
  const handleLoadWorkflow = useCallback(() => {
    getWorkflow(imageDTO.image_name).then((workflow) => {
      dispatch(workflowLoadRequested(workflow.data));
    });
  }, [dispatch, getWorkflow, imageDTO]);

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
    dispatch(setInitialCanvasImage(imageDTO));

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
        icon={getWorkflowResult.isLoading ? <SpinnerIcon /> : <FaCircleNodes />}
        onClickCapture={handleLoadWorkflow}
        isDisabled={!imageDTO.has_workflow}
      >
        {t('nodes.loadWorkflow')}
      </MenuItem>
      <MenuItem
        icon={isLoadingMetadata ? <SpinnerIcon /> : <FaQuoteRight />}
        onClickCapture={handleRecallPrompt}
        isDisabled={
          isLoadingMetadata ||
          (metadata?.positive_prompt === undefined &&
            metadata?.negative_prompt === undefined)
        }
      >
        {t('parameters.usePrompt')}
      </MenuItem>
      <MenuItem
        icon={isLoadingMetadata ? <SpinnerIcon /> : <FaSeedling />}
        onClickCapture={handleRecallSeed}
        isDisabled={isLoadingMetadata || metadata?.seed === undefined}
      >
        {t('parameters.useSeed')}
      </MenuItem>
      <MenuItem
        icon={isLoadingMetadata ? <SpinnerIcon /> : <FaAsterisk />}
        onClickCapture={handleUseAllParameters}
        isDisabled={isLoadingMetadata || !metadata}
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
        {t('boards.changeBoard')}
      </MenuItem>
      {imageDTO.starred ? (
        <MenuItem
          icon={customStarUi ? customStarUi.off.icon : <MdStar />}
          onClickCapture={handleUnstarImage}
        >
          {customStarUi ? customStarUi.off.text : t('controlnet.unstarImage')}
        </MenuItem>
      ) : (
        <MenuItem
          icon={customStarUi ? customStarUi.on.icon : <MdStarBorder />}
          onClickCapture={handleStarImage}
        >
          {customStarUi ? customStarUi.on.text : `Star Image`}
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

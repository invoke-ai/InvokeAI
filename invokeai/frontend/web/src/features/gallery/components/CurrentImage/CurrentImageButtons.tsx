import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash-es';

import {
  ButtonGroup,
  Flex,
  FlexProps,
  Link,
  Menu,
  MenuButton,
  MenuItem,
  MenuList,
} from '@chakra-ui/react';
// import { runESRGAN, runFacetool } from 'app/socketio/actions';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAIIconButton from 'common/components/IAIIconButton';
import IAIPopover from 'common/components/IAIPopover';

import { skipToken } from '@reduxjs/toolkit/dist/query';
import { useAppToaster } from 'app/components/Toaster';
import { stateSelector } from 'app/store/store';
import { setInitialCanvasImage } from 'features/canvas/store/canvasSlice';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import { DeleteImageButton } from 'features/imageDeletion/components/DeleteImageButton';
import { imageToDeleteSelected } from 'features/imageDeletion/store/imageDeletionSlice';
import FaceRestoreSettings from 'features/parameters/components/Parameters/FaceRestore/FaceRestoreSettings';
import UpscaleSettings from 'features/parameters/components/Parameters/Upscale/UpscaleSettings';
import { useRecallParameters } from 'features/parameters/hooks/useRecallParameters';
import { initialImageSelected } from 'features/parameters/store/actions';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { useCopyImageToClipboard } from 'features/ui/hooks/useCopyImageToClipboard';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import {
  setActiveTab,
  setShouldShowImageDetails,
  setShouldShowProgressInViewer,
} from 'features/ui/store/uiSlice';
import { useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import {
  FaAsterisk,
  FaCode,
  FaCopy,
  FaDownload,
  FaExpandArrowsAlt,
  FaGrinStars,
  FaHourglassHalf,
  FaQuoteRight,
  FaSeedling,
  FaShare,
  FaShareAlt,
} from 'react-icons/fa';
import {
  useGetImageDTOQuery,
  useGetImageMetadataQuery,
} from 'services/api/endpoints/images';
import { useDebounce } from 'use-debounce';
import { sentImageToCanvas, sentImageToImg2Img } from '../../store/actions';
import { menuListMotionProps } from 'theme/components/menu';
import SingleSelectionMenuItems from '../ImageContextMenu/SingleSelectionMenuItems';

const currentImageButtonsSelector = createSelector(
  [stateSelector, activeTabNameSelector],
  ({ gallery, system, postprocessing, ui }, activeTabName) => {
    const {
      isProcessing,
      isConnected,
      isGFPGANAvailable,
      isESRGANAvailable,
      shouldConfirmOnDelete,
      progressImage,
    } = system;

    const { upscalingLevel, facetoolStrength } = postprocessing;

    const {
      shouldShowImageDetails,
      shouldHidePreview,
      shouldShowProgressInViewer,
    } = ui;

    const lastSelectedImage = gallery.selection[gallery.selection.length - 1];

    return {
      canDeleteImage: isConnected && !isProcessing,
      shouldConfirmOnDelete,
      isProcessing,
      isConnected,
      isGFPGANAvailable,
      isESRGANAvailable,
      upscalingLevel,
      facetoolStrength,
      shouldDisableToolbarButtons: Boolean(progressImage) || !lastSelectedImage,
      shouldShowImageDetails,
      activeTabName,
      shouldHidePreview,
      shouldShowProgressInViewer,
      lastSelectedImage,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

type CurrentImageButtonsProps = FlexProps;

const CurrentImageButtons = (props: CurrentImageButtonsProps) => {
  const dispatch = useAppDispatch();
  const {
    isProcessing,
    isConnected,
    isGFPGANAvailable,
    isESRGANAvailable,
    upscalingLevel,
    facetoolStrength,
    shouldDisableToolbarButtons,
    shouldShowImageDetails,
    activeTabName,
    lastSelectedImage,
    shouldShowProgressInViewer,
  } = useAppSelector(currentImageButtonsSelector);

  const isCanvasEnabled = useFeatureStatus('unifiedCanvas').isFeatureEnabled;
  const isUpscalingEnabled = useFeatureStatus('upscaling').isFeatureEnabled;
  const isFaceRestoreEnabled = useFeatureStatus('faceRestore').isFeatureEnabled;

  const toaster = useAppToaster();
  const { t } = useTranslation();

  const { isClipboardAPIAvailable, copyImageToClipboard } =
    useCopyImageToClipboard();

  const { recallBothPrompts, recallSeed, recallAllParameters } =
    useRecallParameters();

  const [debouncedMetadataQueryArg, debounceState] = useDebounce(
    lastSelectedImage,
    500
  );

  const { currentData: imageDTO, isFetching } = useGetImageDTOQuery(
    lastSelectedImage ?? skipToken
  );

  const { currentData: metadataData } = useGetImageMetadataQuery(
    debounceState.isPending()
      ? skipToken
      : debouncedMetadataQueryArg ?? skipToken
  );

  const metadata = metadataData?.metadata;

  const handleCopyImageLink = useCallback(() => {
    const getImageUrl = () => {
      if (!imageDTO) {
        return;
      }

      if (imageDTO.image_url.startsWith('http')) {
        return imageDTO.image_url;
      }

      return window.location.toString() + imageDTO.image_url;
    };

    const url = getImageUrl();

    if (!url) {
      toaster({
        title: t('toast.problemCopyingImageLink'),
        status: 'error',
        duration: 2500,
        isClosable: true,
      });

      return;
    }

    navigator.clipboard.writeText(url).then(() => {
      toaster({
        title: t('toast.imageLinkCopied'),
        status: 'success',
        duration: 2500,
        isClosable: true,
      });
    });
  }, [toaster, t, imageDTO]);

  const handleClickUseAllParameters = useCallback(() => {
    recallAllParameters(metadata);
  }, [metadata, recallAllParameters]);

  useHotkeys(
    'a',
    () => {
      handleClickUseAllParameters;
    },
    [metadata, recallAllParameters]
  );

  const handleUseSeed = useCallback(() => {
    recallSeed(metadata?.seed);
  }, [metadata?.seed, recallSeed]);

  useHotkeys('s', handleUseSeed, [imageDTO]);

  const handleUsePrompt = useCallback(() => {
    recallBothPrompts(metadata?.positive_prompt, metadata?.negative_prompt);
  }, [metadata?.negative_prompt, metadata?.positive_prompt, recallBothPrompts]);

  useHotkeys('p', handleUsePrompt, [imageDTO]);

  const handleSendToImageToImage = useCallback(() => {
    dispatch(sentImageToImg2Img());
    dispatch(initialImageSelected(imageDTO));
  }, [dispatch, imageDTO]);

  useHotkeys('shift+i', handleSendToImageToImage, [imageDTO]);

  const handleClickUpscale = useCallback(() => {
    // selectedImage && dispatch(runESRGAN(selectedImage));
  }, []);

  const handleDelete = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    dispatch(imageToDeleteSelected(imageDTO));
  }, [dispatch, imageDTO]);

  useHotkeys(
    'Shift+U',
    () => {
      handleClickUpscale();
    },
    {
      enabled: () =>
        Boolean(
          isUpscalingEnabled &&
            isESRGANAvailable &&
            !shouldDisableToolbarButtons &&
            isConnected &&
            !isProcessing &&
            upscalingLevel
        ),
    },
    [
      isUpscalingEnabled,
      imageDTO,
      isESRGANAvailable,
      shouldDisableToolbarButtons,
      isConnected,
      isProcessing,
      upscalingLevel,
    ]
  );

  const handleClickFixFaces = useCallback(() => {
    // selectedImage && dispatch(runFacetool(selectedImage));
  }, []);

  useHotkeys(
    'Shift+R',
    () => {
      handleClickFixFaces();
    },
    {
      enabled: () =>
        Boolean(
          isFaceRestoreEnabled &&
            isGFPGANAvailable &&
            !shouldDisableToolbarButtons &&
            isConnected &&
            !isProcessing &&
            facetoolStrength
        ),
    },

    [
      isFaceRestoreEnabled,
      imageDTO,
      isGFPGANAvailable,
      shouldDisableToolbarButtons,
      isConnected,
      isProcessing,
      facetoolStrength,
    ]
  );

  const handleClickShowImageDetails = useCallback(
    () => dispatch(setShouldShowImageDetails(!shouldShowImageDetails)),
    [dispatch, shouldShowImageDetails]
  );

  const handleSendToCanvas = useCallback(() => {
    if (!imageDTO) return;
    dispatch(sentImageToCanvas());

    dispatch(setInitialCanvasImage(imageDTO));
    dispatch(requestCanvasRescale());

    if (activeTabName !== 'unifiedCanvas') {
      dispatch(setActiveTab('unifiedCanvas'));
    }

    toaster({
      title: t('toast.sentToUnifiedCanvas'),
      status: 'success',
      duration: 2500,
      isClosable: true,
    });
  }, [imageDTO, dispatch, activeTabName, toaster, t]);

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

  const handleClickProgressImagesToggle = useCallback(() => {
    dispatch(setShouldShowProgressInViewer(!shouldShowProgressInViewer));
  }, [dispatch, shouldShowProgressInViewer]);

  const handleCopyImage = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    copyImageToClipboard(imageDTO.image_url);
  }, [copyImageToClipboard, imageDTO]);

  return (
    <>
      <Flex
        sx={{
          flexWrap: 'wrap',
          justifyContent: 'center',
          alignItems: 'center',
          gap: 2,
        }}
        {...props}
      >
        <ButtonGroup isAttached={true} isDisabled={shouldDisableToolbarButtons}>
          <Menu>
            <MenuButton
              as={IAIIconButton}
              aria-label={`${t('parameters.sendTo')}...`}
              tooltip={`${t('parameters.sendTo')}...`}
              isDisabled={!imageDTO}
              icon={<FaShareAlt />}
            />
            <MenuList motionProps={menuListMotionProps}>
              {imageDTO && <SingleSelectionMenuItems imageDTO={imageDTO} />}
            </MenuList>
          </Menu>
        </ButtonGroup>

        <ButtonGroup isAttached={true} isDisabled={shouldDisableToolbarButtons}>
          <IAIIconButton
            icon={<FaQuoteRight />}
            tooltip={`${t('parameters.usePrompt')} (P)`}
            aria-label={`${t('parameters.usePrompt')} (P)`}
            isDisabled={!metadata?.positive_prompt}
            onClick={handleUsePrompt}
          />

          <IAIIconButton
            icon={<FaSeedling />}
            tooltip={`${t('parameters.useSeed')} (S)`}
            aria-label={`${t('parameters.useSeed')} (S)`}
            isDisabled={!metadata?.seed}
            onClick={handleUseSeed}
          />

          <IAIIconButton
            icon={<FaAsterisk />}
            tooltip={`${t('parameters.useAll')} (A)`}
            aria-label={`${t('parameters.useAll')} (A)`}
            isDisabled={!metadata}
            onClick={handleClickUseAllParameters}
          />
        </ButtonGroup>

        {(isUpscalingEnabled || isFaceRestoreEnabled) && (
          <ButtonGroup
            isAttached={true}
            isDisabled={shouldDisableToolbarButtons}
          >
            {isFaceRestoreEnabled && (
              <IAIPopover
                triggerComponent={
                  <IAIIconButton
                    icon={<FaGrinStars />}
                    aria-label={t('parameters.restoreFaces')}
                  />
                }
              >
                <Flex
                  sx={{
                    flexDirection: 'column',
                    rowGap: 4,
                  }}
                >
                  <FaceRestoreSettings />
                  <IAIButton
                    isDisabled={
                      !isGFPGANAvailable ||
                      !imageDTO ||
                      !(isConnected && !isProcessing) ||
                      !facetoolStrength
                    }
                    onClick={handleClickFixFaces}
                  >
                    {t('parameters.restoreFaces')}
                  </IAIButton>
                </Flex>
              </IAIPopover>
            )}

            {isUpscalingEnabled && (
              <IAIPopover
                triggerComponent={
                  <IAIIconButton
                    icon={<FaExpandArrowsAlt />}
                    aria-label={t('parameters.upscale')}
                  />
                }
              >
                <Flex
                  sx={{
                    flexDirection: 'column',
                    gap: 4,
                  }}
                >
                  <UpscaleSettings />
                  <IAIButton
                    isDisabled={
                      !isESRGANAvailable ||
                      !imageDTO ||
                      !(isConnected && !isProcessing) ||
                      !upscalingLevel
                    }
                    onClick={handleClickUpscale}
                  >
                    {t('parameters.upscaleImage')}
                  </IAIButton>
                </Flex>
              </IAIPopover>
            )}
          </ButtonGroup>
        )}

        <ButtonGroup isAttached={true} isDisabled={shouldDisableToolbarButtons}>
          <IAIIconButton
            icon={<FaCode />}
            tooltip={`${t('parameters.info')} (I)`}
            aria-label={`${t('parameters.info')} (I)`}
            isChecked={shouldShowImageDetails}
            onClick={handleClickShowImageDetails}
          />
        </ButtonGroup>

        <ButtonGroup isAttached={true}>
          <IAIIconButton
            aria-label={t('settings.displayInProgress')}
            tooltip={t('settings.displayInProgress')}
            icon={<FaHourglassHalf />}
            isChecked={shouldShowProgressInViewer}
            onClick={handleClickProgressImagesToggle}
          />
        </ButtonGroup>

        <ButtonGroup isAttached={true}>
          <DeleteImageButton
            onClick={handleDelete}
            isDisabled={shouldDisableToolbarButtons}
          />
        </ButtonGroup>
      </Flex>
    </>
  );
};

export default CurrentImageButtons;

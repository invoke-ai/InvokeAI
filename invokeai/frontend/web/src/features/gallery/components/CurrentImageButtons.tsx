import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash-es';

import {
  ButtonGroup,
  Flex,
  FlexProps,
  Link,
  useDisclosure,
} from '@chakra-ui/react';
// import { runESRGAN, runFacetool } from 'app/socketio/actions';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAIIconButton from 'common/components/IAIIconButton';
import IAIPopover from 'common/components/IAIPopover';

import { lightboxSelector } from 'features/lightbox/store/lightboxSelectors';
import { setIsLightboxOpen } from 'features/lightbox/store/lightboxSlice';
import { postprocessingSelector } from 'features/parameters/store/postprocessingSelectors';
import { systemSelector } from 'features/system/store/systemSelectors';

import {
  activeTabNameSelector,
  uiSelector,
} from 'features/ui/store/uiSelectors';
import {
  setActiveTab,
  setShouldShowImageDetails,
  setShouldShowProgressInViewer,
} from 'features/ui/store/uiSlice';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import {
  FaAsterisk,
  FaCode,
  FaCopy,
  FaDownload,
  FaExpand,
  FaExpandArrowsAlt,
  FaGrinStars,
  FaHourglassHalf,
  FaQuoteRight,
  FaSeedling,
  FaShare,
  FaShareAlt,
} from 'react-icons/fa';
import { gallerySelector } from '../store/gallerySelectors';
import { useCallback } from 'react';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import { useGetUrl } from 'common/util/getUrl';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { useRecallParameters } from 'features/parameters/hooks/useRecallParameters';
import { initialImageSelected } from 'features/parameters/store/actions';
import {
  requestedImageDeletion,
  sentImageToCanvas,
  sentImageToImg2Img,
} from '../store/actions';
import FaceRestoreSettings from 'features/parameters/components/Parameters/FaceRestore/FaceRestoreSettings';
import UpscaleSettings from 'features/parameters/components/Parameters/Upscale/UpscaleSettings';
import DeleteImageButton from './ImageActionButtons/DeleteImageButton';
import { useAppToaster } from 'app/components/Toaster';
import { setInitialCanvasImage } from 'features/canvas/store/canvasSlice';

const currentImageButtonsSelector = createSelector(
  [
    systemSelector,
    gallerySelector,
    postprocessingSelector,
    uiSelector,
    lightboxSelector,
    activeTabNameSelector,
  ],
  (system, gallery, postprocessing, ui, lightbox, activeTabName) => {
    const {
      isProcessing,
      isConnected,
      isGFPGANAvailable,
      isESRGANAvailable,
      shouldConfirmOnDelete,
      progressImage,
    } = system;

    const { upscalingLevel, facetoolStrength } = postprocessing;

    const { isLightboxOpen } = lightbox;

    const {
      shouldShowImageDetails,
      shouldHidePreview,
      shouldShowProgressInViewer,
    } = ui;

    const { selectedImage } = gallery;

    return {
      canDeleteImage: isConnected && !isProcessing,
      shouldConfirmOnDelete,
      isProcessing,
      isConnected,
      isGFPGANAvailable,
      isESRGANAvailable,
      upscalingLevel,
      facetoolStrength,
      shouldDisableToolbarButtons: Boolean(progressImage) || !selectedImage,
      shouldShowImageDetails,
      activeTabName,
      isLightboxOpen,
      shouldHidePreview,
      image: selectedImage,
      seed: selectedImage?.metadata?.seed,
      prompt: selectedImage?.metadata?.positive_conditioning,
      negativePrompt: selectedImage?.metadata?.negative_conditioning,
      shouldShowProgressInViewer,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

type CurrentImageButtonsProps = FlexProps;

/**
 * Row of buttons for common actions:
 * Use as init image, use all params, use seed, upscale, fix faces, details, delete.
 */
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
    // currentImage,
    isLightboxOpen,
    activeTabName,
    shouldHidePreview,
    image,
    canDeleteImage,
    shouldConfirmOnDelete,
    shouldShowProgressInViewer,
  } = useAppSelector(currentImageButtonsSelector);

  const isLightboxEnabled = useFeatureStatus('lightbox').isFeatureEnabled;
  const isCanvasEnabled = useFeatureStatus('unifiedCanvas').isFeatureEnabled;
  const isUpscalingEnabled = useFeatureStatus('upscaling').isFeatureEnabled;
  const isFaceRestoreEnabled = useFeatureStatus('faceRestore').isFeatureEnabled;

  const { getUrl, shouldTransformUrls } = useGetUrl();

  const {
    isOpen: isDeleteDialogOpen,
    onOpen: onDeleteDialogOpen,
    onClose: onDeleteDialogClose,
  } = useDisclosure();

  const toaster = useAppToaster();
  const { t } = useTranslation();

  const { recallBothPrompts, recallSeed, recallAllParameters } =
    useRecallParameters();

  // const handleCopyImage = useCallback(async () => {
  //   if (!image?.url) {
  //     return;
  //   }

  //   const url = getUrl(image.url);

  //   if (!url) {
  //     return;
  //   }

  //   const blob = await fetch(url).then((res) => res.blob());
  //   const data = [new ClipboardItem({ [blob.type]: blob })];

  //   await navigator.clipboard.write(data);

  //   toast({
  //     title: t('toast.imageCopied'),
  //     status: 'success',
  //     duration: 2500,
  //     isClosable: true,
  //   });
  // }, [getUrl, t, image?.url, toast]);

  const handleCopyImageLink = useCallback(() => {
    const getImageUrl = () => {
      if (!image) {
        return;
      }

      if (shouldTransformUrls) {
        return getUrl(image.image_url);
      }

      if (image.image_url.startsWith('http')) {
        return image.image_url;
      }

      return window.location.toString() + image.image_url;
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
  }, [toaster, shouldTransformUrls, getUrl, t, image]);

  const handleClickUseAllParameters = useCallback(() => {
    recallAllParameters(image);
  }, [image, recallAllParameters]);

  useHotkeys(
    'a',
    () => {
      handleClickUseAllParameters;
    },
    [image, recallAllParameters]
  );

  const handleUseSeed = useCallback(() => {
    recallSeed(image?.metadata?.seed);
  }, [image, recallSeed]);

  useHotkeys('s', handleUseSeed, [image]);

  const handleUsePrompt = useCallback(() => {
    recallBothPrompts(
      image?.metadata?.positive_conditioning,
      image?.metadata?.negative_conditioning
    );
  }, [image, recallBothPrompts]);

  useHotkeys('p', handleUsePrompt, [image]);

  const handleSendToImageToImage = useCallback(() => {
    dispatch(sentImageToImg2Img());
    dispatch(initialImageSelected(image));
  }, [dispatch, image]);

  useHotkeys('shift+i', handleSendToImageToImage, [image]);

  const handleClickUpscale = useCallback(() => {
    // selectedImage && dispatch(runESRGAN(selectedImage));
  }, []);

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
      image,
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
      image,
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
    if (!image) return;
    dispatch(sentImageToCanvas());
    if (isLightboxOpen) dispatch(setIsLightboxOpen(false));

    dispatch(setInitialCanvasImage(image));
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
  }, [image, isLightboxOpen, dispatch, activeTabName, toaster, t]);

  useHotkeys(
    'i',
    () => {
      if (image) {
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
    [image, shouldShowImageDetails, toaster]
  );

  const handleDelete = useCallback(() => {
    if (canDeleteImage && image) {
      dispatch(requestedImageDeletion(image));
    }
  }, [image, canDeleteImage, dispatch]);

  const handleInitiateDelete = useCallback(() => {
    if (shouldConfirmOnDelete) {
      onDeleteDialogOpen();
    } else {
      handleDelete();
    }
  }, [shouldConfirmOnDelete, onDeleteDialogOpen, handleDelete]);

  const handleClickProgressImagesToggle = useCallback(() => {
    dispatch(setShouldShowProgressInViewer(!shouldShowProgressInViewer));
  }, [dispatch, shouldShowProgressInViewer]);

  useHotkeys('delete', handleInitiateDelete, [
    image,
    shouldConfirmOnDelete,
    isConnected,
    isProcessing,
  ]);

  const handleLightBox = useCallback(() => {
    dispatch(setIsLightboxOpen(!isLightboxOpen));
  }, [dispatch, isLightboxOpen]);

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
        <ButtonGroup isAttached={true}>
          <IAIPopover
            triggerComponent={
              <IAIIconButton
                aria-label={`${t('parameters.sendTo')}...`}
                tooltip={`${t('parameters.sendTo')}...`}
                isDisabled={!image}
                icon={<FaShareAlt />}
              />
            }
          >
            <Flex
              sx={{
                flexDirection: 'column',
                rowGap: 2,
              }}
            >
              <IAIButton
                size="sm"
                onClick={handleSendToImageToImage}
                leftIcon={<FaShare />}
                id="send-to-img2img"
              >
                {t('parameters.sendToImg2Img')}
              </IAIButton>
              {isCanvasEnabled && (
                <IAIButton
                  size="sm"
                  onClick={handleSendToCanvas}
                  leftIcon={<FaShare />}
                  id="send-to-canvas"
                >
                  {t('parameters.sendToUnifiedCanvas')}
                </IAIButton>
              )}

              {/* <IAIButton
                size="sm"
                onClick={handleCopyImage}
                leftIcon={<FaCopy />}
              >
                {t('parameters.copyImage')}
              </IAIButton> */}
              <IAIButton
                size="sm"
                onClick={handleCopyImageLink}
                leftIcon={<FaCopy />}
              >
                {t('parameters.copyImageToLink')}
              </IAIButton>

              <Link
                download={true}
                href={getUrl(image?.image_url ?? '')}
                target="_blank"
              >
                <IAIButton leftIcon={<FaDownload />} size="sm" w="100%">
                  {t('parameters.downloadImage')}
                </IAIButton>
              </Link>
            </Flex>
          </IAIPopover>
          {isLightboxEnabled && (
            <IAIIconButton
              icon={<FaExpand />}
              tooltip={
                !isLightboxOpen
                  ? `${t('parameters.openInViewer')} (Z)`
                  : `${t('parameters.closeViewer')} (Z)`
              }
              aria-label={
                !isLightboxOpen
                  ? `${t('parameters.openInViewer')} (Z)`
                  : `${t('parameters.closeViewer')} (Z)`
              }
              isChecked={isLightboxOpen}
              onClick={handleLightBox}
            />
          )}
        </ButtonGroup>

        <ButtonGroup isAttached={true}>
          <IAIIconButton
            icon={<FaQuoteRight />}
            tooltip={`${t('parameters.usePrompt')} (P)`}
            aria-label={`${t('parameters.usePrompt')} (P)`}
            isDisabled={!image?.metadata?.positive_conditioning}
            onClick={handleUsePrompt}
          />

          <IAIIconButton
            icon={<FaSeedling />}
            tooltip={`${t('parameters.useSeed')} (S)`}
            aria-label={`${t('parameters.useSeed')} (S)`}
            isDisabled={!image?.metadata?.seed}
            onClick={handleUseSeed}
          />

          <IAIIconButton
            icon={<FaAsterisk />}
            tooltip={`${t('parameters.useAll')} (A)`}
            aria-label={`${t('parameters.useAll')} (A)`}
            isDisabled={
              // not sure what this list should be
              !['t2l', 'l2l', 'inpaint'].includes(String(image?.metadata?.type))
            }
            onClick={handleClickUseAllParameters}
          />
        </ButtonGroup>

        {(isUpscalingEnabled || isFaceRestoreEnabled) && (
          <ButtonGroup isAttached={true}>
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
                      !image ||
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
                      !image ||
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

        <ButtonGroup isAttached={true}>
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
          <DeleteImageButton image={image} />
        </ButtonGroup>
      </Flex>
    </>
  );
};

export default CurrentImageButtons;

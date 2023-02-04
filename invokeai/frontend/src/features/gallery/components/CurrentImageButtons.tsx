import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';

import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import {
  setAllParameters,
  setInitialImage,
  setPrompt,
  setSeed,
} from 'features/parameters/store/generationSlice';
import { setShouldShowImageDetails } from 'features/ui/store/uiSlice';
import { setIsLightboxOpen } from 'features/lightbox/store/lightboxSlice';
import DeleteImageModal from './DeleteImageModal';
import { SystemState } from 'features/system/store/systemSlice';
import IAIButton from 'common/components/IAIButton';
import { runESRGAN, runFacetool } from 'app/socketio/actions';
import IAIIconButton from 'common/components/IAIIconButton';
import UpscaleSettings from 'features/parameters/components/AdvancedParameters/Upscale/UpscaleSettings';
import FaceRestoreSettings from 'features/parameters/components/AdvancedParameters/FaceRestore/FaceRestoreSettings';
import { useHotkeys } from 'react-hotkeys-hook';
import { ButtonGroup, Link, useToast } from '@chakra-ui/react';
import {
  FaAsterisk,
  FaCode,
  FaCopy,
  FaDownload,
  FaExpand,
  FaExpandArrowsAlt,
  FaGrinStars,
  FaQuoteRight,
  FaSeedling,
  FaShare,
  FaShareAlt,
  FaTrash,
} from 'react-icons/fa';
import {
  setDoesCanvasNeedScaling,
  setInitialCanvasImage,
} from 'features/canvas/store/canvasSlice';
import { GalleryState } from 'features/gallery/store/gallerySlice';
import {
  activeTabNameSelector,
  uiSelector,
} from 'features/ui/store/uiSelectors';
import IAIPopover from 'common/components/IAIPopover';
import { useTranslation } from 'react-i18next';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { lightboxSelector } from 'features/lightbox/store/lightboxSelectors';
import { postprocessingSelector } from 'features/parameters/store/postprocessingSelectors';
import { systemSelector } from 'features/system/store/systemSelectors';
import { gallerySelector } from '../store/gallerySelectors';

const currentImageButtonsSelector = createSelector(
  [
    systemSelector,
    gallerySelector,
    postprocessingSelector,
    uiSelector,
    lightboxSelector,
    activeTabNameSelector,
  ],
  (
    system: SystemState,
    gallery: GalleryState,
    postprocessing,
    ui,
    lightbox,
    activeTabName
  ) => {
    const { isProcessing, isConnected, isGFPGANAvailable, isESRGANAvailable } =
      system;

    const { upscalingLevel, facetoolStrength } = postprocessing;

    const { isLightboxOpen } = lightbox;

    const { shouldShowImageDetails } = ui;

    const { intermediateImage, currentImage } = gallery;

    return {
      isProcessing,
      isConnected,
      isGFPGANAvailable,
      isESRGANAvailable,
      upscalingLevel,
      facetoolStrength,
      shouldDisableToolbarButtons: Boolean(intermediateImage) || !currentImage,
      currentImage,
      shouldShowImageDetails,
      activeTabName,
      isLightboxOpen,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

/**
 * Row of buttons for common actions:
 * Use as init image, use all params, use seed, upscale, fix faces, details, delete.
 */
const CurrentImageButtons = () => {
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
    currentImage,
    isLightboxOpen,
    activeTabName,
  } = useAppSelector(currentImageButtonsSelector);

  const toast = useToast();
  const { t } = useTranslation();

  const handleClickUseAsInitialImage = () => {
    if (!currentImage) return;
    if (isLightboxOpen) dispatch(setIsLightboxOpen(false));
    dispatch(setInitialImage(currentImage));
    dispatch(setActiveTab('img2img'));
  };

  const handleCopyImageLink = () => {
    navigator.clipboard
      .writeText(
        currentImage ? window.location.toString() + currentImage.url : ''
      )
      .then(() => {
        toast({
          title: t('toast:imageLinkCopied'),
          status: 'success',
          duration: 2500,
          isClosable: true,
        });
      });
  };

  useHotkeys(
    'shift+i',
    () => {
      if (currentImage) {
        handleClickUseAsInitialImage();
        toast({
          title: t('toast:sentToImageToImage'),
          status: 'success',
          duration: 2500,
          isClosable: true,
        });
      } else {
        toast({
          title: t('toast:imageNotLoaded'),
          description: t('toast:imageNotLoadedDesc'),
          status: 'error',
          duration: 2500,
          isClosable: true,
        });
      }
    },
    [currentImage]
  );

  const handleClickUseAllParameters = () => {
    if (!currentImage) return;
    currentImage.metadata && dispatch(setAllParameters(currentImage.metadata));
    if (currentImage.metadata?.image.type === 'img2img') {
      dispatch(setActiveTab('img2img'));
    } else if (currentImage.metadata?.image.type === 'txt2img') {
      dispatch(setActiveTab('txt2img'));
    }
  };

  useHotkeys(
    'a',
    () => {
      if (
        ['txt2img', 'img2img'].includes(currentImage?.metadata?.image?.type)
      ) {
        handleClickUseAllParameters();
        toast({
          title: t('toast:parametersSet'),
          status: 'success',
          duration: 2500,
          isClosable: true,
        });
      } else {
        toast({
          title: t('toast:parametersNotSet'),
          description: t('toast:parametersNotSetDesc'),
          status: 'error',
          duration: 2500,
          isClosable: true,
        });
      }
    },
    [currentImage]
  );

  const handleClickUseSeed = () => {
    currentImage?.metadata &&
      dispatch(setSeed(currentImage.metadata.image.seed));
  };

  useHotkeys(
    's',
    () => {
      if (currentImage?.metadata?.image?.seed) {
        handleClickUseSeed();
        toast({
          title: t('toast:seedSet'),
          status: 'success',
          duration: 2500,
          isClosable: true,
        });
      } else {
        toast({
          title: t('toast:seedNotSet'),
          description: t('toast:seedNotSetDesc'),
          status: 'error',
          duration: 2500,
          isClosable: true,
        });
      }
    },
    [currentImage]
  );

  const handleClickUsePrompt = () =>
    currentImage?.metadata?.image?.prompt &&
    dispatch(setPrompt(currentImage.metadata.image.prompt));

  useHotkeys(
    'p',
    () => {
      if (currentImage?.metadata?.image?.prompt) {
        handleClickUsePrompt();
        toast({
          title: t('toast:promptSet'),
          status: 'success',
          duration: 2500,
          isClosable: true,
        });
      } else {
        toast({
          title: t('toast:promptNotSet'),
          description: t('toast:promptNotSetDesc'),
          status: 'error',
          duration: 2500,
          isClosable: true,
        });
      }
    },
    [currentImage]
  );

  const handleClickUpscale = () => {
    currentImage && dispatch(runESRGAN(currentImage));
  };

  useHotkeys(
    'Shift+U',
    () => {
      if (
        isESRGANAvailable &&
        !shouldDisableToolbarButtons &&
        isConnected &&
        !isProcessing &&
        upscalingLevel
      ) {
        handleClickUpscale();
      } else {
        toast({
          title: t('toast:upscalingFailed'),
          status: 'error',
          duration: 2500,
          isClosable: true,
        });
      }
    },
    [
      currentImage,
      isESRGANAvailable,
      shouldDisableToolbarButtons,
      isConnected,
      isProcessing,
      upscalingLevel,
    ]
  );

  const handleClickFixFaces = () => {
    currentImage && dispatch(runFacetool(currentImage));
  };

  useHotkeys(
    'Shift+R',
    () => {
      if (
        isGFPGANAvailable &&
        !shouldDisableToolbarButtons &&
        isConnected &&
        !isProcessing &&
        facetoolStrength
      ) {
        handleClickFixFaces();
      } else {
        toast({
          title: t('toast:faceRestoreFailed'),
          status: 'error',
          duration: 2500,
          isClosable: true,
        });
      }
    },
    [
      currentImage,
      isGFPGANAvailable,
      shouldDisableToolbarButtons,
      isConnected,
      isProcessing,
      facetoolStrength,
    ]
  );

  const handleClickShowImageDetails = () =>
    dispatch(setShouldShowImageDetails(!shouldShowImageDetails));

  const handleSendToCanvas = () => {
    if (!currentImage) return;
    if (isLightboxOpen) dispatch(setIsLightboxOpen(false));

    dispatch(setInitialCanvasImage(currentImage));
    dispatch(setDoesCanvasNeedScaling(true));

    if (activeTabName !== 'unifiedCanvas') {
      dispatch(setActiveTab('unifiedCanvas'));
    }

    toast({
      title: t('toast:sentToUnifiedCanvas'),
      status: 'success',
      duration: 2500,
      isClosable: true,
    });
  };

  useHotkeys(
    'i',
    () => {
      if (currentImage) {
        handleClickShowImageDetails();
      } else {
        toast({
          title: t('toast:metadataLoadFailed'),
          status: 'error',
          duration: 2500,
          isClosable: true,
        });
      }
    },
    [currentImage, shouldShowImageDetails]
  );

  const handleLightBox = () => {
    dispatch(setIsLightboxOpen(!isLightboxOpen));
  };

  return (
    <div className="current-image-options">
      <ButtonGroup isAttached={true}>
        <IAIPopover
          trigger="hover"
          triggerComponent={
            <IAIIconButton
              aria-label={`${t('parameters:sendTo')}...`}
              icon={<FaShareAlt />}
            />
          }
        >
          <div className="current-image-send-to-popover">
            <IAIButton
              size={'sm'}
              onClick={handleClickUseAsInitialImage}
              leftIcon={<FaShare />}
            >
              {t('parameters:sendToImg2Img')}
            </IAIButton>
            <IAIButton
              size={'sm'}
              onClick={handleSendToCanvas}
              leftIcon={<FaShare />}
            >
              {t('parameters:sendToUnifiedCanvas')}
            </IAIButton>
            <IAIButton
              size={'sm'}
              onClick={handleCopyImageLink}
              leftIcon={<FaCopy />}
            >
              {t('parameters:copyImageToLink')}
            </IAIButton>

            <Link download={true} href={currentImage?.url}>
              <IAIButton leftIcon={<FaDownload />} size={'sm'} w="100%">
                {t('parameters:downloadImage')}
              </IAIButton>
            </Link>
          </div>
        </IAIPopover>
        <IAIIconButton
          icon={<FaExpand />}
          tooltip={
            !isLightboxOpen
              ? `${t('parameters:openInViewer')} (Z)`
              : `${t('parameters:closeViewer')} (Z)`
          }
          aria-label={
            !isLightboxOpen
              ? `${t('parameters:openInViewer')} (Z)`
              : `${t('parameters:closeViewer')} (Z)`
          }
          data-selected={isLightboxOpen}
          onClick={handleLightBox}
        />
      </ButtonGroup>

      <ButtonGroup isAttached={true}>
        <IAIIconButton
          icon={<FaQuoteRight />}
          tooltip={`${t('parameters:usePrompt')} (P)`}
          aria-label={`${t('parameters:usePrompt')} (P)`}
          isDisabled={!currentImage?.metadata?.image?.prompt}
          onClick={handleClickUsePrompt}
        />

        <IAIIconButton
          icon={<FaSeedling />}
          tooltip={`${t('parameters:useSeed')} (S)`}
          aria-label={`${t('parameters:useSeed')} (S)`}
          isDisabled={!currentImage?.metadata?.image?.seed}
          onClick={handleClickUseSeed}
        />

        <IAIIconButton
          icon={<FaAsterisk />}
          tooltip={`${t('parameters:useAll')} (A)`}
          aria-label={`${t('parameters:useAll')} (A)`}
          isDisabled={
            !['txt2img', 'img2img'].includes(
              currentImage?.metadata?.image?.type
            )
          }
          onClick={handleClickUseAllParameters}
        />
      </ButtonGroup>

      <ButtonGroup isAttached={true}>
        <IAIPopover
          trigger="hover"
          triggerComponent={
            <IAIIconButton
              icon={<FaGrinStars />}
              aria-label={t('parameters:restoreFaces')}
            />
          }
        >
          <div className="current-image-postprocessing-popover">
            <FaceRestoreSettings />
            <IAIButton
              isDisabled={
                !isGFPGANAvailable ||
                !currentImage ||
                !(isConnected && !isProcessing) ||
                !facetoolStrength
              }
              onClick={handleClickFixFaces}
            >
              {t('parameters:restoreFaces')}
            </IAIButton>
          </div>
        </IAIPopover>

        <IAIPopover
          trigger="hover"
          triggerComponent={
            <IAIIconButton
              icon={<FaExpandArrowsAlt />}
              aria-label={t('parameters:upscale')}
            />
          }
        >
          <div className="current-image-postprocessing-popover">
            <UpscaleSettings />
            <IAIButton
              isDisabled={
                !isESRGANAvailable ||
                !currentImage ||
                !(isConnected && !isProcessing) ||
                !upscalingLevel
              }
              onClick={handleClickUpscale}
            >
              {t('parameters:upscaleImage')}
            </IAIButton>
          </div>
        </IAIPopover>
      </ButtonGroup>

      <ButtonGroup isAttached={true}>
        <IAIIconButton
          icon={<FaCode />}
          tooltip={`${t('parameters:info')} (I)`}
          aria-label={`${t('parameters:info')} (I)`}
          data-selected={shouldShowImageDetails}
          onClick={handleClickShowImageDetails}
        />
      </ButtonGroup>

      <DeleteImageModal image={currentImage}>
        <IAIIconButton
          icon={<FaTrash />}
          tooltip={`${t('parameters:deleteImage')} (Del)`}
          aria-label={`${t('parameters:deleteImage')} (Del)`}
          isDisabled={!currentImage || !isConnected || isProcessing}
          style={{ backgroundColor: 'var(--btn-delete-image)' }}
        />
      </DeleteImageModal>
    </div>
  );
};

export default CurrentImageButtons;

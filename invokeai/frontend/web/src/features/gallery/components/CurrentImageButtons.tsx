import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';

import { ButtonGroup, Flex, FlexProps, Link, useToast } from '@chakra-ui/react';
import { runESRGAN, runFacetool } from 'app/socketio/actions';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAIIconButton from 'common/components/IAIIconButton';
import IAIPopover from 'common/components/IAIPopover';
import { setInitialCanvasImage } from 'features/canvas/store/canvasSlice';
import { GalleryState } from 'features/gallery/store/gallerySlice';
import { lightboxSelector } from 'features/lightbox/store/lightboxSelectors';
import { setIsLightboxOpen } from 'features/lightbox/store/lightboxSlice';
import FaceRestoreSettings from 'features/parameters/components/AdvancedParameters/FaceRestore/FaceRestoreSettings';
import UpscaleSettings from 'features/parameters/components/AdvancedParameters/Upscale/UpscaleSettings';
import {
  setAllParameters,
  setInitialImage,
  setSeed,
} from 'features/parameters/store/generationSlice';
import { postprocessingSelector } from 'features/parameters/store/postprocessingSelectors';
import { systemSelector } from 'features/system/store/systemSelectors';
import { SystemState } from 'features/system/store/systemSlice';
import {
  activeTabNameSelector,
  uiSelector,
} from 'features/ui/store/uiSelectors';
import {
  setActiveTab,
  setShouldShowImageDetails,
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
  FaQuoteRight,
  FaSeedling,
  FaShare,
  FaShareAlt,
  FaTrash,
} from 'react-icons/fa';
import { gallerySelector } from '../store/gallerySelectors';
import DeleteImageModal from './DeleteImageModal';
import { useCallback } from 'react';
import useSetBothPrompts from 'features/parameters/hooks/usePrompt';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';

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
    currentImage,
    isLightboxOpen,
    activeTabName,
  } = useAppSelector(currentImageButtonsSelector);

  const toast = useToast();
  const { t } = useTranslation();
  const setBothPrompts = useSetBothPrompts();

  const handleClickUseAsInitialImage = () => {
    if (!currentImage) return;
    if (isLightboxOpen) dispatch(setIsLightboxOpen(false));
    dispatch(setInitialImage(currentImage));
    dispatch(setActiveTab('img2img'));
  };

  const handleCopyImage = async () => {
    if (!currentImage) return;

    const blob = await fetch(currentImage.url).then((res) => res.blob());
    const data = [new ClipboardItem({ [blob.type]: blob })];

    await navigator.clipboard.write(data);

    toast({
      title: t('toast.imageCopied'),
      status: 'success',
      duration: 2500,
      isClosable: true,
    });
  };

  const handleCopyImageLink = () => {
    navigator.clipboard
      .writeText(
        currentImage ? window.location.toString() + currentImage.url : ''
      )
      .then(() => {
        toast({
          title: t('toast.imageLinkCopied'),
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
          title: t('toast.sentToImageToImage'),
          status: 'success',
          duration: 2500,
          isClosable: true,
        });
      } else {
        toast({
          title: t('toast.imageNotLoaded'),
          description: t('toast.imageNotLoadedDesc'),
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
          title: t('toast.parametersSet'),
          status: 'success',
          duration: 2500,
          isClosable: true,
        });
      } else {
        toast({
          title: t('toast.parametersNotSet'),
          description: t('toast.parametersNotSetDesc'),
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
          title: t('toast.seedSet'),
          status: 'success',
          duration: 2500,
          isClosable: true,
        });
      } else {
        toast({
          title: t('toast.seedNotSet'),
          description: t('toast.seedNotSetDesc'),
          status: 'error',
          duration: 2500,
          isClosable: true,
        });
      }
    },
    [currentImage]
  );

  const handleClickUsePrompt = useCallback(() => {
    if (currentImage?.metadata?.image?.prompt) {
      setBothPrompts(currentImage?.metadata?.image?.prompt);
    }
  }, [currentImage?.metadata?.image?.prompt, setBothPrompts]);

  useHotkeys(
    'p',
    () => {
      if (currentImage?.metadata?.image?.prompt) {
        handleClickUsePrompt();
        toast({
          title: t('toast.promptSet'),
          status: 'success',
          duration: 2500,
          isClosable: true,
        });
      } else {
        toast({
          title: t('toast.promptNotSet'),
          description: t('toast.promptNotSetDesc'),
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
          title: t('toast.upscalingFailed'),
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
          title: t('toast.faceRestoreFailed'),
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
    dispatch(requestCanvasRescale());

    if (activeTabName !== 'unifiedCanvas') {
      dispatch(setActiveTab('unifiedCanvas'));
    }

    toast({
      title: t('toast.sentToUnifiedCanvas'),
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
          title: t('toast.metadataLoadFailed'),
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
    <Flex
      sx={{
        justifyContent: 'center',
        alignItems: 'center',
        columnGap: '0.5em',
      }}
      {...props}
    >
      <ButtonGroup isAttached={true}>
        <IAIPopover
          triggerComponent={
            <IAIIconButton
              aria-label={`${t('parameters.sendTo')}...`}
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
              onClick={handleClickUseAsInitialImage}
              leftIcon={<FaShare />}
            >
              {t('parameters.sendToImg2Img')}
            </IAIButton>
            <IAIButton
              size="sm"
              onClick={handleSendToCanvas}
              leftIcon={<FaShare />}
            >
              {t('parameters.sendToUnifiedCanvas')}
            </IAIButton>

            <IAIButton
              size="sm"
              onClick={handleCopyImage}
              leftIcon={<FaCopy />}
            >
              {t('parameters.copyImage')}
            </IAIButton>
            <IAIButton
              size="sm"
              onClick={handleCopyImageLink}
              leftIcon={<FaCopy />}
            >
              {t('parameters.copyImageToLink')}
            </IAIButton>

            <Link download={true} href={currentImage?.url}>
              <IAIButton leftIcon={<FaDownload />} size="sm" w="100%">
                {t('parameters.downloadImage')}
              </IAIButton>
            </Link>
          </Flex>
        </IAIPopover>
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
      </ButtonGroup>

      <ButtonGroup isAttached={true}>
        <IAIIconButton
          icon={<FaQuoteRight />}
          tooltip={`${t('parameters.usePrompt')} (P)`}
          aria-label={`${t('parameters.usePrompt')} (P)`}
          isDisabled={!currentImage?.metadata?.image?.prompt}
          onClick={handleClickUsePrompt}
        />

        <IAIIconButton
          icon={<FaSeedling />}
          tooltip={`${t('parameters.useSeed')} (S)`}
          aria-label={`${t('parameters.useSeed')} (S)`}
          isDisabled={!currentImage?.metadata?.image?.seed}
          onClick={handleClickUseSeed}
        />

        <IAIIconButton
          icon={<FaAsterisk />}
          tooltip={`${t('parameters.useAll')} (A)`}
          aria-label={`${t('parameters.useAll')} (A)`}
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
                !currentImage ||
                !(isConnected && !isProcessing) ||
                !facetoolStrength
              }
              onClick={handleClickFixFaces}
            >
              {t('parameters.restoreFaces')}
            </IAIButton>
          </Flex>
        </IAIPopover>

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
                !currentImage ||
                !(isConnected && !isProcessing) ||
                !upscalingLevel
              }
              onClick={handleClickUpscale}
            >
              {t('parameters.upscaleImage')}
            </IAIButton>
          </Flex>
        </IAIPopover>
      </ButtonGroup>

      <ButtonGroup isAttached={true}>
        <IAIIconButton
          icon={<FaCode />}
          tooltip={`${t('parameters.info')} (I)`}
          aria-label={`${t('parameters.info')} (I)`}
          isChecked={shouldShowImageDetails}
          onClick={handleClickShowImageDetails}
        />
      </ButtonGroup>

      <DeleteImageModal image={currentImage}>
        <IAIIconButton
          icon={<FaTrash />}
          tooltip={`${t('parameters.deleteImage')} (Del)`}
          aria-label={`${t('parameters.deleteImage')} (Del)`}
          isDisabled={!currentImage || !isConnected || isProcessing}
          colorScheme="error"
        />
      </DeleteImageModal>
    </Flex>
  );
};

export default CurrentImageButtons;

import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';

import { useAppDispatch, useAppSelector } from 'app/store';
import { RootState } from 'app/store';
import {
  OptionsState,
  setActiveTab,
  setAllParameters,
  setInitialImage,
  setIsLightBoxOpen,
  setPrompt,
  setSeed,
  setShouldShowImageDetails,
} from 'features/options/store/optionsSlice';
import DeleteImageModal from './DeleteImageModal';
import { SystemState } from 'features/system/store/systemSlice';
import IAIButton from 'common/components/IAIButton';
import { runESRGAN, runFacetool } from 'app/socketio/actions';
import IAIIconButton from 'common/components/IAIIconButton';
import UpscaleOptions from 'features/options/components/AdvancedOptions/Upscale/UpscaleOptions';
import FaceRestoreOptions from 'features/options/components/AdvancedOptions/FaceRestore/FaceRestoreOptions';
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
import { activeTabNameSelector } from 'features/options/store/optionsSelectors';
import IAIPopover from 'common/components/IAIPopover';

const systemSelector = createSelector(
  [
    (state: RootState) => state.system,
    (state: RootState) => state.options,
    (state: RootState) => state.gallery,
    activeTabNameSelector,
  ],
  (
    system: SystemState,
    options: OptionsState,
    gallery: GalleryState,
    activeTabName
  ) => {
    const { isProcessing, isConnected, isGFPGANAvailable, isESRGANAvailable } =
      system;

    const {
      upscalingLevel,
      facetoolStrength,
      shouldShowImageDetails,
      isLightBoxOpen,
    } = options;

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
      isLightBoxOpen,
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
    isLightBoxOpen,
    activeTabName,
  } = useAppSelector(systemSelector);

  const toast = useToast();

  const handleClickUseAsInitialImage = () => {
    if (!currentImage) return;
    if (isLightBoxOpen) dispatch(setIsLightBoxOpen(false));
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
          title: 'Image Link Copied',
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
          title: 'Sent To Image To Image',
          status: 'success',
          duration: 2500,
          isClosable: true,
        });
      } else {
        toast({
          title: 'No Image Loaded',
          description: 'No image found to send to image to image module.',
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
          title: 'Parameters Set',
          status: 'success',
          duration: 2500,
          isClosable: true,
        });
      } else {
        toast({
          title: 'Parameters Not Set',
          description: 'No metadata found for this image.',
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
          title: 'Seed Set',
          status: 'success',
          duration: 2500,
          isClosable: true,
        });
      } else {
        toast({
          title: 'Seed Not Set',
          description: 'Could not find seed for this image.',
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
          title: 'Prompt Set',
          status: 'success',
          duration: 2500,
          isClosable: true,
        });
      } else {
        toast({
          title: 'Prompt Not Set',
          description: 'Could not find prompt for this image.',
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
    'u',
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
          title: 'Upscaling Failed',
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
    'r',
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
          title: 'Face Restoration Failed',
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
    if (isLightBoxOpen) dispatch(setIsLightBoxOpen(false));

    dispatch(setInitialCanvasImage(currentImage));
    dispatch(setDoesCanvasNeedScaling(true));

    if (activeTabName !== 'unifiedCanvas') {
      dispatch(setActiveTab('unifiedCanvas'));
    }

    toast({
      title: 'Sent to Unified Canvas',
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
          title: 'Failed to load metadata',
          status: 'error',
          duration: 2500,
          isClosable: true,
        });
      }
    },
    [currentImage, shouldShowImageDetails]
  );

  const handleLightBox = () => {
    dispatch(setIsLightBoxOpen(!isLightBoxOpen));
  };

  return (
    <div className="current-image-options">
      <ButtonGroup isAttached={true}>
        <IAIPopover
          trigger="hover"
          triggerComponent={
            <IAIIconButton aria-label="Send to..." icon={<FaShareAlt />} />
          }
        >
          <div className="current-image-send-to-popover">
            <IAIButton
              size={'sm'}
              onClick={handleClickUseAsInitialImage}
              leftIcon={<FaShare />}
            >
              Send to Image to Image
            </IAIButton>
            <IAIButton
              size={'sm'}
              onClick={handleSendToCanvas}
              leftIcon={<FaShare />}
            >
              Send to Unified Canvas
            </IAIButton>
            <IAIButton
              size={'sm'}
              onClick={handleCopyImageLink}
              leftIcon={<FaCopy />}
            >
              Copy Link to Image
            </IAIButton>

            <IAIButton leftIcon={<FaDownload />} size={'sm'}>
              <Link download={true} href={currentImage?.url}>
                Download Image
              </Link>
            </IAIButton>
          </div>
        </IAIPopover>
        <IAIIconButton
          icon={<FaExpand />}
          tooltip={!isLightBoxOpen ? 'Open In Viewer (Z)' : 'Close Viewer (Z)'}
          aria-label={
            !isLightBoxOpen ? 'Open In Viewer (Z)' : 'Close Viewer (Z)'
          }
          data-selected={isLightBoxOpen}
          onClick={handleLightBox}
        />
      </ButtonGroup>

      <ButtonGroup isAttached={true}>
        <IAIIconButton
          icon={<FaQuoteRight />}
          tooltip="Use Prompt (P)"
          aria-label="Use Prompt (P)"
          isDisabled={!currentImage?.metadata?.image?.prompt}
          onClick={handleClickUsePrompt}
        />

        <IAIIconButton
          icon={<FaSeedling />}
          tooltip="Use Seed (S)"
          aria-label="Use Seed (S)"
          isDisabled={!currentImage?.metadata?.image?.seed}
          onClick={handleClickUseSeed}
        />

        <IAIIconButton
          icon={<FaAsterisk />}
          tooltip="Use All (A)"
          aria-label="Use All (A)"
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
            <IAIIconButton icon={<FaGrinStars />} aria-label="Restore Faces" />
          }
        >
          <div className="current-image-postprocessing-popover">
            <FaceRestoreOptions />
            <IAIButton
              isDisabled={
                !isGFPGANAvailable ||
                !currentImage ||
                !(isConnected && !isProcessing) ||
                !facetoolStrength
              }
              onClick={handleClickFixFaces}
            >
              Restore Faces
            </IAIButton>
          </div>
        </IAIPopover>

        <IAIPopover
          trigger="hover"
          triggerComponent={
            <IAIIconButton icon={<FaExpandArrowsAlt />} aria-label="Upscale" />
          }
        >
          <div className="current-image-postprocessing-popover">
            <UpscaleOptions />
            <IAIButton
              isDisabled={
                !isESRGANAvailable ||
                !currentImage ||
                !(isConnected && !isProcessing) ||
                !upscalingLevel
              }
              onClick={handleClickUpscale}
            >
              Upscale Image
            </IAIButton>
          </div>
        </IAIPopover>
      </ButtonGroup>

      <ButtonGroup isAttached={true}>
        <IAIIconButton
          icon={<FaCode />}
          tooltip="Info (I)"
          aria-label="Info (I)"
          data-selected={shouldShowImageDetails}
          onClick={handleClickShowImageDetails}
        />
      </ButtonGroup>

      <DeleteImageModal image={currentImage}>
        <IAIIconButton
          icon={<FaTrash />}
          tooltip="Delete Image"
          aria-label="Delete Image"
          isDisabled={!currentImage || !isConnected || isProcessing}
          style={{ backgroundColor: 'var(--btn-delete-image)' }}
        />
      </DeleteImageModal>
    </div>
  );
};

export default CurrentImageButtons;

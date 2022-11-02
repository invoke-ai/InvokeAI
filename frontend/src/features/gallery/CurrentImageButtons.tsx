import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';

import { useAppDispatch, useAppSelector } from '../../app/store';
import { RootState } from '../../app/store';
import {
  OptionsState,
  setActiveTab,
  setAllParameters,
  setInitialImage,
  setPrompt,
  setSeed,
  setShouldShowImageDetails,
} from '../options/optionsSlice';
import DeleteImageModal from './DeleteImageModal';
import { SystemState } from '../system/systemSlice';
import IAIButton from '../../common/components/IAIButton';
import { runESRGAN, runFacetool } from '../../app/socketio/actions';
import IAIIconButton from '../../common/components/IAIIconButton';
import UpscaleOptions from '../options/AdvancedOptions/Upscale/UpscaleOptions';
import FaceRestoreOptions from '../options/AdvancedOptions/FaceRestore/FaceRestoreOptions';
import { useHotkeys } from 'react-hotkeys-hook';
import { ButtonGroup, Link, useClipboard, useToast } from '@chakra-ui/react';
import {
  FaAsterisk,
  FaCode,
  FaCopy,
  FaDownload,
  FaExpandArrowsAlt,
  FaGrinStars,
  FaQuoteRight,
  FaSeedling,
  FaShare,
  FaShareAlt,
  FaTrash,
} from 'react-icons/fa';
import {
  setImageToInpaint,
  setNeedsCache,
} from '../tabs/Inpainting/inpaintingSlice';
import { GalleryState } from './gallerySlice';
import { activeTabNameSelector } from '../options/optionsSelectors';
import IAIPopover from '../../common/components/IAIPopover';

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

    const { upscalingLevel, facetoolStrength, shouldShowImageDetails } =
      options;

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
  } = useAppSelector(systemSelector);

  const { onCopy } = useClipboard(
    currentImage ? window.location.toString() + currentImage.url : ''
  );

  const toast = useToast();

  const handleClickUseAsInitialImage = () => {
    if (!currentImage) return;
    dispatch(setInitialImage(currentImage));
    dispatch(setActiveTab('img2img'));
  };

  const handleCopyImageLink = () => {
    onCopy();
    toast({
      title: 'Image Link Copied',
      status: 'success',
      duration: 2500,
      isClosable: true,
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

  const handleSendToInpainting = () => {
    if (!currentImage) return;

    dispatch(setImageToInpaint(currentImage));

    dispatch(setActiveTab('inpainting'));
    dispatch(setNeedsCache(true));

    toast({
      title: 'Sent to Inpainting',
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
              onClick={handleSendToInpainting}
              leftIcon={<FaShare />}
            >
              Send to Inpainting
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
      </ButtonGroup>

      <ButtonGroup isAttached={true}>
        <IAIIconButton
          icon={<FaQuoteRight />}
          tooltip="Use Prompt"
          aria-label="Use Prompt"
          isDisabled={!currentImage?.metadata?.image?.prompt}
          onClick={handleClickUsePrompt}
        />

        <IAIIconButton
          icon={<FaSeedling />}
          tooltip="Use Seed"
          aria-label="Use Seed"
          isDisabled={!currentImage?.metadata?.image?.seed}
          onClick={handleClickUseSeed}
        />

        <IAIIconButton
          icon={<FaAsterisk />}
          tooltip="Use All"
          aria-label="Use All"
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

      <IAIIconButton
        icon={<FaCode />}
        tooltip="Details"
        aria-label="Details"
        data-selected={shouldShowImageDetails}
        onClick={handleClickShowImageDetails}
      />

      <DeleteImageModal image={currentImage}>
        <IAIIconButton
          icon={<FaTrash />}
          tooltip="Delete Image"
          aria-label="Delete Image"
          isDisabled={!currentImage || !isConnected || isProcessing}
          className="delete-image-btn"
        />
      </DeleteImageModal>
    </div>
  );
};

export default CurrentImageButtons;

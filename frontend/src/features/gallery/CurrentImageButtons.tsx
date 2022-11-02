import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';

import * as InvokeAI from '../../app/invokeai';

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
import { setImageToInpaint } from '../tabs/Inpainting/inpaintingSlice';
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

    const { intermediateImage } = gallery;

    return {
      isProcessing,
      isConnected,
      isGFPGANAvailable,
      isESRGANAvailable,
      upscalingLevel,
      facetoolStrength,
      intermediateImage,
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

type CurrentImageButtonsProps = {
  image: InvokeAI.Image;
};

/**
 * Row of buttons for common actions:
 * Use as init image, use all params, use seed, upscale, fix faces, details, delete.
 */
const CurrentImageButtons = ({ image }: CurrentImageButtonsProps) => {
  const dispatch = useAppDispatch();
  const {
    isProcessing,
    isConnected,
    isGFPGANAvailable,
    isESRGANAvailable,
    upscalingLevel,
    facetoolStrength,
    intermediateImage,
    shouldShowImageDetails,
    activeTabName,
  } = useAppSelector(systemSelector);

  const { onCopy } = useClipboard(window.location.toString() + image.url);

  const toast = useToast();

  const handleClickUseAsInitialImage = () => {
    dispatch(setInitialImage(image));
    dispatch(setActiveTab(1));
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
      if (image) {
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
    [image]
  );

  const handleClickUseAllParameters = () =>
    image.metadata && dispatch(setAllParameters(image.metadata));

  useHotkeys(
    'a',
    () => {
      if (['txt2img', 'img2img'].includes(image?.metadata?.image?.type)) {
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
    [image]
  );

  const handleClickUseSeed = () =>
    image.metadata && dispatch(setSeed(image.metadata.image.seed));
  useHotkeys(
    's',
    () => {
      if (image?.metadata?.image?.seed) {
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
    [image]
  );

  const handleClickUsePrompt = () =>
    image?.metadata?.image?.prompt &&
    dispatch(setPrompt(image.metadata.image.prompt));

  useHotkeys(
    'p',
    () => {
      if (image?.metadata?.image?.prompt) {
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
    [image]
  );

  const handleClickUpscale = () => dispatch(runESRGAN(image));
  useHotkeys(
    'u',
    () => {
      if (
        isESRGANAvailable &&
        Boolean(!intermediateImage) &&
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
      image,
      isESRGANAvailable,
      intermediateImage,
      isConnected,
      isProcessing,
      upscalingLevel,
    ]
  );

  const handleClickFixFaces = () => dispatch(runFacetool(image));

  useHotkeys(
    'r',
    () => {
      if (
        isGFPGANAvailable &&
        Boolean(!intermediateImage) &&
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
      image,
      isGFPGANAvailable,
      intermediateImage,
      isConnected,
      isProcessing,
      facetoolStrength,
    ]
  );

  const handleClickShowImageDetails = () =>
    dispatch(setShouldShowImageDetails(!shouldShowImageDetails));

  const handleSendToInpainting = () => {
    dispatch(setImageToInpaint(image));
    if (activeTabName !== 'inpainting') {
      dispatch(setActiveTab('inpainting'));
    }
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
      if (image) {
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
    [image, shouldShowImageDetails]
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
              <Link download={true} href={image.url}>
                Download Image
              </Link>
            </IAIButton>
          </div>
        </IAIPopover>

        <IAIIconButton
          icon={<FaQuoteRight />}
          tooltip="Use Prompt"
          aria-label="Use Prompt"
          isDisabled={!image?.metadata?.image?.prompt}
          onClick={handleClickUsePrompt}
        />

        <IAIIconButton
          icon={<FaSeedling />}
          tooltip="Use Seed"
          aria-label="Use Seed"
          isDisabled={!image?.metadata?.image?.seed}
          onClick={handleClickUseSeed}
        />

        <IAIIconButton
          icon={<FaAsterisk />}
          tooltip="Use All"
          aria-label="Use All"
          isDisabled={
            !['txt2img', 'img2img'].includes(image?.metadata?.image?.type)
          }
          onClick={handleClickUseAllParameters}
        />

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
                Boolean(intermediateImage) ||
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
                Boolean(intermediateImage) ||
                !(isConnected && !isProcessing) ||
                !upscalingLevel
              }
              onClick={handleClickUpscale}
            >
              Upscale Image
            </IAIButton>
          </div>
        </IAIPopover>

        <IAIIconButton
          icon={<FaCode />}
          tooltip="Details"
          aria-label="Details"
          data-selected={shouldShowImageDetails}
          onClick={handleClickShowImageDetails}
        />

        <DeleteImageModal image={image}>
          <IAIIconButton
            icon={<FaTrash />}
            tooltip="Delete Image"
            aria-label="Delete Image"
            isDisabled={
              Boolean(intermediateImage) || !isConnected || isProcessing
            }
          />
        </DeleteImageModal>
      </ButtonGroup>
    </div>
  );
};

export default CurrentImageButtons;

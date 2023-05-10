import { createSelector } from '@reduxjs/toolkit';
import { get, isEqual, isNumber, isString } from 'lodash-es';

import {
  ButtonGroup,
  Flex,
  FlexProps,
  FormControl,
  Link,
  useDisclosure,
  useToast,
} from '@chakra-ui/react';
// import { runESRGAN, runFacetool } from 'app/socketio/actions';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
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
  initialImageSelected,
  setAllParameters,
  // setInitialImage,
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
  setShouldHidePreview,
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
  FaEye,
  FaEyeSlash,
  FaGrinStars,
  FaQuoteRight,
  FaSeedling,
  FaShare,
  FaShareAlt,
  FaTrash,
} from 'react-icons/fa';
import {
  gallerySelector,
  selectedImageSelector,
} from '../store/gallerySelectors';
import DeleteImageModal from './DeleteImageModal';
import { useCallback } from 'react';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import { useGetUrl } from 'common/util/getUrl';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { imageDeleted } from 'services/thunks/image';
import { useParameters } from 'features/parameters/hooks/useParameters';

const currentImageButtonsSelector = createSelector(
  [
    systemSelector,
    gallerySelector,
    postprocessingSelector,
    uiSelector,
    lightboxSelector,
    activeTabNameSelector,
    selectedImageSelector,
  ],
  (system, gallery, postprocessing, ui, lightbox, activeTabName, image) => {
    const {
      isProcessing,
      isConnected,
      isGFPGANAvailable,
      isESRGANAvailable,
      shouldConfirmOnDelete,
    } = system;

    const { upscalingLevel, facetoolStrength } = postprocessing;

    const { isLightboxOpen } = lightbox;

    const { shouldShowImageDetails, shouldHidePreview } = ui;

    const { intermediateImage, currentImage } = gallery;

    return {
      canDeleteImage: isConnected && !isProcessing,
      shouldConfirmOnDelete,
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
      shouldHidePreview,
      image,
      seed: image?.metadata?.invokeai?.node?.seed,
      prompt: image?.metadata?.invokeai?.node?.prompt,
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
  } = useAppSelector(currentImageButtonsSelector);

  const isLightboxEnabled = useFeatureStatus('lightbox').isFeatureEnabled;
  const isUpscalingEnabled = useFeatureStatus('upscaling').isFeatureEnabled;
  const isFaceRestoreEnabled = useFeatureStatus('faceRestore').isFeatureEnabled;

  const { getUrl, shouldTransformUrls } = useGetUrl();

  const {
    isOpen: isDeleteDialogOpen,
    onOpen: onDeleteDialogOpen,
    onClose: onDeleteDialogClose,
  } = useDisclosure();

  const toast = useToast();
  const { t } = useTranslation();

  const { recallPrompt, recallSeed, sendToImageToImage } = useParameters();

  const handleCopyImage = useCallback(async () => {
    if (!image?.url) {
      return;
    }

    const url = getUrl(image.url);

    if (!url) {
      return;
    }

    const blob = await fetch(url).then((res) => res.blob());
    const data = [new ClipboardItem({ [blob.type]: blob })];

    await navigator.clipboard.write(data);

    toast({
      title: t('toast.imageCopied'),
      status: 'success',
      duration: 2500,
      isClosable: true,
    });
  }, [getUrl, t, image?.url, toast]);

  const handleCopyImageLink = useCallback(() => {
    const url = image
      ? shouldTransformUrls
        ? getUrl(image.url)
        : window.location.toString() + image.url
      : '';

    if (!url) {
      return;
    }

    navigator.clipboard.writeText(url).then(() => {
      toast({
        title: t('toast.imageLinkCopied'),
        status: 'success',
        duration: 2500,
        isClosable: true,
      });
    });
  }, [toast, shouldTransformUrls, getUrl, t, image]);

  const handlePreviewVisibility = useCallback(() => {
    dispatch(setShouldHidePreview(!shouldHidePreview));
  }, [dispatch, shouldHidePreview]);

  const handleClickUseAllParameters = useCallback(() => {
    if (!image) return;
    // selectedImage.metadata &&
    //   dispatch(setAllParameters(selectedImage.metadata));
    // if (selectedImage.metadata?.image.type === 'img2img') {
    //   dispatch(setActiveTab('img2img'));
    // } else if (selectedImage.metadata?.image.type === 'txt2img') {
    //   dispatch(setActiveTab('txt2img'));
    // }
  }, [image]);

  useHotkeys(
    'a',
    () => {
      const type = image?.metadata?.invokeai?.node?.types;
      if (isString(type) && ['txt2img', 'img2img'].includes(type)) {
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
    [image]
  );

  const handleUseSeed = useCallback(() => {
    recallSeed(image?.metadata?.invokeai?.node?.seed);
  }, [image, recallSeed]);

  useHotkeys('s', handleUseSeed, [image]);

  const handleUsePrompt = useCallback(() => {
    recallPrompt(image?.metadata?.invokeai?.node?.prompt);
  }, [image, recallPrompt]);

  useHotkeys('p', handleUsePrompt, [image]);

  const handleSendToImageToImage = useCallback(() => {
    sendToImageToImage(image);
  }, [image, sendToImageToImage]);

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
    if (isLightboxOpen) dispatch(setIsLightboxOpen(false));

    // dispatch(setInitialCanvasImage(selectedImage));
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
  }, [image, isLightboxOpen, dispatch, activeTabName, toast, t]);

  useHotkeys(
    'i',
    () => {
      if (image) {
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
    [image, shouldShowImageDetails]
  );

  const handleDelete = useCallback(() => {
    if (canDeleteImage && image) {
      dispatch(imageDeleted({ imageType: image.type, imageName: image.name }));
    }
  }, [image, canDeleteImage, dispatch]);

  const handleInitiateDelete = useCallback(() => {
    if (shouldConfirmOnDelete) {
      onDeleteDialogOpen();
    } else {
      handleDelete();
    }
  }, [shouldConfirmOnDelete, onDeleteDialogOpen, handleDelete]);

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
                isDisabled={!image}
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
                onClick={handleSendToImageToImage}
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

              <Link download={true} href={getUrl(image?.url ?? '')}>
                <IAIButton leftIcon={<FaDownload />} size="sm" w="100%">
                  {t('parameters.downloadImage')}
                </IAIButton>
              </Link>
            </Flex>
          </IAIPopover>
          <IAIIconButton
            icon={shouldHidePreview ? <FaEyeSlash /> : <FaEye />}
            tooltip={
              !shouldHidePreview
                ? t('parameters.hidePreview')
                : t('parameters.showPreview')
            }
            aria-label={
              !shouldHidePreview
                ? t('parameters.hidePreview')
                : t('parameters.showPreview')
            }
            isChecked={shouldHidePreview}
            onClick={handlePreviewVisibility}
          />
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
            isDisabled={!image?.metadata?.invokeai?.node?.prompt}
            onClick={handleUsePrompt}
          />

          <IAIIconButton
            icon={<FaSeedling />}
            tooltip={`${t('parameters.useSeed')} (S)`}
            aria-label={`${t('parameters.useSeed')} (S)`}
            isDisabled={!image?.metadata?.invokeai?.node?.seed}
            onClick={handleUseSeed}
          />

          <IAIIconButton
            icon={<FaAsterisk />}
            tooltip={`${t('parameters.useAll')} (A)`}
            aria-label={`${t('parameters.useAll')} (A)`}
            isDisabled={
              !['txt2img', 'img2img'].includes(
                image?.metadata?.sd_metadata?.type
              )
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

        <IAIIconButton
          onClick={handleInitiateDelete}
          icon={<FaTrash />}
          tooltip={`${t('gallery.deleteImage')} (Del)`}
          aria-label={`${t('gallery.deleteImage')} (Del)`}
          isDisabled={!image || !isConnected}
          colorScheme="error"
        />
      </Flex>
      {image && (
        <DeleteImageModal
          isOpen={isDeleteDialogOpen}
          onClose={onDeleteDialogClose}
          handleDelete={handleDelete}
        />
      )}
    </>
  );
};

export default CurrentImageButtons;

import {
  Box,
  Flex,
  Icon,
  Image,
  MenuItem,
  MenuList,
  Skeleton,
  useDisclosure,
  useTheme,
  useToast,
} from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import { DragEvent, memo, useCallback, useState } from 'react';
import { FaCheck, FaExpand, FaImage, FaShare, FaTrash } from 'react-icons/fa';
import DeleteImageModal from './DeleteImageModal';
import { ContextMenu } from 'chakra-ui-contextmenu';
import * as InvokeAI from 'app/types/invokeai';
import { resizeAndScaleCanvas } from 'features/canvas/store/canvasSlice';
import { gallerySelector } from 'features/gallery/store/gallerySelectors';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { useTranslation } from 'react-i18next';
import IAIIconButton from 'common/components/IAIIconButton';
import { useGetUrl } from 'common/util/getUrl';
import { ExternalLinkIcon } from '@chakra-ui/icons';
import { IoArrowUndoCircleOutline } from 'react-icons/io5';
import { imageDeleted } from 'services/thunks/image';
import { createSelector } from '@reduxjs/toolkit';
import { systemSelector } from 'features/system/store/systemSelectors';
import { lightboxSelector } from 'features/lightbox/store/lightboxSelectors';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { isEqual } from 'lodash-es';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { useParameters } from 'features/parameters/hooks/useParameters';

export const selector = createSelector(
  [gallerySelector, systemSelector, lightboxSelector, activeTabNameSelector],
  (gallery, system, lightbox, activeTabName) => {
    const {
      galleryImageObjectFit,
      galleryImageMinimumWidth,
      shouldUseSingleGalleryColumn,
    } = gallery;

    const { isLightboxOpen } = lightbox;
    const { isConnected, isProcessing, shouldConfirmOnDelete } = system;

    return {
      canDeleteImage: isConnected && !isProcessing,
      shouldConfirmOnDelete,
      galleryImageObjectFit,
      galleryImageMinimumWidth,
      shouldUseSingleGalleryColumn,
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

interface HoverableImageProps {
  image: InvokeAI.Image;
  isSelected: boolean;
}

const memoEqualityCheck = (
  prev: HoverableImageProps,
  next: HoverableImageProps
) => prev.image.name === next.image.name && prev.isSelected === next.isSelected;

/**
 * Gallery image component with delete/use all/use seed buttons on hover.
 */
const HoverableImage = memo((props: HoverableImageProps) => {
  const dispatch = useAppDispatch();
  const {
    activeTabName,
    galleryImageObjectFit,
    galleryImageMinimumWidth,
    canDeleteImage,
    shouldUseSingleGalleryColumn,
    shouldConfirmOnDelete,
  } = useAppSelector(selector);

  const {
    isOpen: isDeleteDialogOpen,
    onOpen: onDeleteDialogOpen,
    onClose: onDeleteDialogClose,
  } = useDisclosure();

  const { image, isSelected } = props;
  const { url, thumbnail, name, metadata } = image;
  const { getUrl } = useGetUrl();

  const [isHovered, setIsHovered] = useState<boolean>(false);

  const toast = useToast();
  const { direction } = useTheme();
  const { t } = useTranslation();
  const { isFeatureEnabled: isLightboxEnabled } = useFeatureStatus('lightbox');
  const { recallSeed, recallPrompt, sendToImageToImage, recallInitialImage } =
    useParameters();

  const handleMouseOver = () => setIsHovered(true);
  const handleMouseOut = () => setIsHovered(false);

  // Immediately deletes an image
  const handleDelete = useCallback(() => {
    if (canDeleteImage && image) {
      dispatch(imageDeleted({ imageType: image.type, imageName: image.name }));
    }
  }, [dispatch, image, canDeleteImage]);

  // Opens the alert dialog to check if user is sure they want to delete
  const handleInitiateDelete = useCallback(() => {
    if (shouldConfirmOnDelete) {
      onDeleteDialogOpen();
    } else {
      handleDelete();
    }
  }, [handleDelete, onDeleteDialogOpen, shouldConfirmOnDelete]);

  const handleSelectImage = useCallback(() => {
    dispatch(imageSelected(image));
  }, [image, dispatch]);

  const handleDragStart = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.dataTransfer.setData('invokeai/imageName', image.name);
      e.dataTransfer.setData('invokeai/imageType', image.type);
      e.dataTransfer.effectAllowed = 'move';
    },
    [image]
  );

  // Recall parameters handlers
  const handleRecallPrompt = useCallback(() => {
    recallPrompt(image.metadata?.invokeai?.node?.prompt);
  }, [image, recallPrompt]);

  const handleRecallSeed = useCallback(() => {
    recallSeed(image.metadata.invokeai?.node?.seed);
  }, [image, recallSeed]);

  const handleSendToImageToImage = useCallback(() => {
    sendToImageToImage(image);
  }, [image, sendToImageToImage]);

  const handleRecallInitialImage = useCallback(() => {
    recallInitialImage(image.metadata.invokeai?.node?.image);
  }, [image, recallInitialImage]);

  /**
   * TODO: the rest of these
   */
  const handleSendToCanvas = () => {
    // dispatch(setInitialCanvasImage(image));

    dispatch(resizeAndScaleCanvas());

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

  const handleUseAllParameters = () => {
    // metadata.invokeai?.node &&
    //   dispatch(setAllParameters(metadata.invokeai?.node));
    // toast({
    //   title: t('toast.parametersSet'),
    //   status: 'success',
    //   duration: 2500,
    //   isClosable: true,
    // });
  };

  const handleLightBox = () => {
    // dispatch(setCurrentImage(image));
    // dispatch(setIsLightboxOpen(true));
  };

  const handleOpenInNewTab = () => {
    window.open(getUrl(image.url), '_blank');
  };

  return (
    <>
      <ContextMenu<HTMLDivElement>
        menuProps={{ size: 'sm', isLazy: true }}
        renderMenu={() => (
          <MenuList sx={{ visibility: 'visible !important' }}>
            <MenuItem
              icon={<ExternalLinkIcon />}
              onClickCapture={handleOpenInNewTab}
            >
              {t('common.openInNewTab')}
            </MenuItem>
            {isLightboxEnabled && (
              <MenuItem icon={<FaExpand />} onClickCapture={handleLightBox}>
                {t('parameters.openInViewer')}
              </MenuItem>
            )}
            <MenuItem
              icon={<IoArrowUndoCircleOutline />}
              onClickCapture={handleRecallPrompt}
              isDisabled={image?.metadata?.invokeai?.node?.prompt === undefined}
            >
              {t('parameters.usePrompt')}
            </MenuItem>

            <MenuItem
              icon={<IoArrowUndoCircleOutline />}
              onClickCapture={handleRecallSeed}
              isDisabled={image?.metadata?.invokeai?.node?.seed === undefined}
            >
              {t('parameters.useSeed')}
            </MenuItem>
            <MenuItem
              icon={<IoArrowUndoCircleOutline />}
              onClickCapture={handleRecallInitialImage}
              isDisabled={image?.metadata?.invokeai?.node?.type !== 'img2img'}
            >
              {t('parameters.useInitImg')}
            </MenuItem>
            <MenuItem
              icon={<IoArrowUndoCircleOutline />}
              onClickCapture={handleUseAllParameters}
              isDisabled={
                !['txt2img', 'img2img'].includes(
                  String(image?.metadata?.invokeai?.node?.type)
                )
              }
            >
              {t('parameters.useAll')}
            </MenuItem>
            <MenuItem
              icon={<FaShare />}
              onClickCapture={handleSendToImageToImage}
            >
              {t('parameters.sendToImg2Img')}
            </MenuItem>
            <MenuItem icon={<FaShare />} onClickCapture={handleSendToCanvas}>
              {t('parameters.sendToUnifiedCanvas')}
            </MenuItem>
            <MenuItem icon={<FaTrash />} onClickCapture={onDeleteDialogOpen}>
              {t('gallery.deleteImage')}
            </MenuItem>
          </MenuList>
        )}
      >
        {(ref) => (
          <Box
            position="relative"
            key={name}
            onMouseOver={handleMouseOver}
            onMouseOut={handleMouseOut}
            userSelect="none"
            draggable={true}
            onDragStart={handleDragStart}
            onClick={handleSelectImage}
            ref={ref}
            sx={{
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              w: 'full',
              h: 'full',
              transition: 'transform 0.2s ease-out',
              aspectRatio: '1/1',
            }}
          >
            <Image
              loading="lazy"
              objectFit={
                shouldUseSingleGalleryColumn ? 'contain' : galleryImageObjectFit
              }
              rounded="md"
              src={getUrl(thumbnail || url)}
              fallback={<FaImage />}
              sx={{
                width: '100%',
                height: '100%',
                maxWidth: '100%',
                maxHeight: '100%',
              }}
            />
            {isSelected && (
              <Flex
                sx={{
                  position: 'absolute',
                  top: '0',
                  insetInlineStart: '0',
                  width: '100%',
                  height: '100%',
                  alignItems: 'center',
                  justifyContent: 'center',
                  pointerEvents: 'none',
                }}
              >
                <Icon
                  filter={'drop-shadow(0px 0px 1rem black)'}
                  as={FaCheck}
                  sx={{
                    width: '50%',
                    height: '50%',
                    fill: 'ok.500',
                  }}
                />
              </Flex>
            )}
            {isHovered && galleryImageMinimumWidth >= 100 && (
              <Box
                sx={{
                  position: 'absolute',
                  top: 1,
                  insetInlineEnd: 1,
                }}
              >
                <IAIIconButton
                  onClickCapture={handleInitiateDelete}
                  aria-label={t('gallery.deleteImage')}
                  icon={<FaTrash />}
                  size="xs"
                  fontSize={14}
                  isDisabled={!canDeleteImage}
                />
              </Box>
            )}
          </Box>
        )}
      </ContextMenu>
      <DeleteImageModal
        isOpen={isDeleteDialogOpen}
        onClose={onDeleteDialogClose}
        handleDelete={handleDelete}
      />
    </>
  );
}, memoEqualityCheck);

HoverableImage.displayName = 'HoverableImage';

export default HoverableImage;

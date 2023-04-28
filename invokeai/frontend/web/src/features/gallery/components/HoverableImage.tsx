import {
  Box,
  Flex,
  Icon,
  Image,
  MenuItem,
  MenuList,
  Text,
  useDisclosure,
  useTheme,
  useToast,
} from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import {
  imageSelected,
  setCurrentImage,
} from 'features/gallery/store/gallerySlice';
import {
  initialImageSelected,
  setAllImageToImageParameters,
  setAllParameters,
  setSeed,
} from 'features/parameters/store/generationSlice';
import { DragEvent, memo, useState } from 'react';
import {
  FaCheck,
  FaExpand,
  FaLink,
  FaShare,
  FaTrash,
  FaTrashAlt,
} from 'react-icons/fa';
import DeleteImageModal from './DeleteImageModal';
import { ContextMenu } from 'chakra-ui-contextmenu';
import * as InvokeAI from 'app/invokeai';
import {
  resizeAndScaleCanvas,
  setInitialCanvasImage,
} from 'features/canvas/store/canvasSlice';
import { gallerySelector } from 'features/gallery/store/gallerySelectors';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { useTranslation } from 'react-i18next';
import useSetBothPrompts from 'features/parameters/hooks/usePrompt';
import { setIsLightboxOpen } from 'features/lightbox/store/lightboxSlice';
import IAIIconButton from 'common/components/IAIIconButton';
import { useGetUrl } from 'common/util/getUrl';
import { ExternalLinkIcon } from '@chakra-ui/icons';
import { BiZoomIn } from 'react-icons/bi';
import { IoArrowUndoCircleOutline } from 'react-icons/io5';
import { imageDeleted } from 'services/thunks/image';
import { createSelector } from '@reduxjs/toolkit';
import { systemSelector } from 'features/system/store/systemSelectors';
import { configSelector } from 'features/system/store/configSelectors';
import { lightboxSelector } from 'features/lightbox/store/lightboxSelectors';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { isEqual } from 'lodash';

export const selector = createSelector(
  [
    gallerySelector,
    systemSelector,
    configSelector,
    lightboxSelector,
    activeTabNameSelector,
  ],
  (gallery, system, config, lightbox, activeTabName) => {
    const {
      galleryImageObjectFit,
      galleryImageMinimumWidth,
      shouldUseSingleGalleryColumn,
    } = gallery;

    const { isLightboxOpen } = lightbox;
    const { disabledFeatures } = config;
    const { isConnected, isProcessing, shouldConfirmOnDelete } = system;

    return {
      canDeleteImage: isConnected && !isProcessing,
      shouldConfirmOnDelete,
      galleryImageObjectFit,
      galleryImageMinimumWidth,
      shouldUseSingleGalleryColumn,
      activeTabName,
      isLightboxOpen,
      disabledFeatures,
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
    disabledFeatures,
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
  const setBothPrompts = useSetBothPrompts();

  const handleMouseOver = () => setIsHovered(true);

  const handleMouseOut = () => setIsHovered(false);

  const handleInitiateDelete = () => {
    if (shouldConfirmOnDelete) {
      onDeleteDialogOpen();
    } else {
      handleDelete();
    }
  };

  const handleDelete = () => {
    if (canDeleteImage && image) {
      dispatch(imageDeleted({ imageType: image.type, imageName: image.name }));
    }
  };

  const handleUsePrompt = () => {
    if (image.metadata?.sd_metadata?.prompt) {
      setBothPrompts(image.metadata?.sd_metadata?.prompt);
    }
    toast({
      title: t('toast.promptSet'),
      status: 'success',
      duration: 2500,
      isClosable: true,
    });
  };

  const handleUseSeed = () => {
    image.metadata.sd_metadata &&
      dispatch(setSeed(image.metadata.sd_metadata.image.seed));
    toast({
      title: t('toast.seedSet'),
      status: 'success',
      duration: 2500,
      isClosable: true,
    });
  };

  const handleSendToImageToImage = () => {
    dispatch(initialImageSelected(image.name));
  };

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
    metadata.sd_metadata && dispatch(setAllParameters(metadata.sd_metadata));
    toast({
      title: t('toast.parametersSet'),
      status: 'success',
      duration: 2500,
      isClosable: true,
    });
  };

  const handleUseInitialImage = async () => {
    if (metadata.sd_metadata?.image?.init_image_path) {
      const response = await fetch(
        metadata.sd_metadata?.image?.init_image_path
      );
      if (response.ok) {
        dispatch(setAllImageToImageParameters(metadata?.sd_metadata));
        toast({
          title: t('toast.initialImageSet'),
          status: 'success',
          duration: 2500,
          isClosable: true,
        });
        return;
      }
    }
    toast({
      title: t('toast.initialImageNotSet'),
      description: t('toast.initialImageNotSetDesc'),
      status: 'error',
      duration: 2500,
      isClosable: true,
    });
  };

  const handleSelectImage = () => {
    dispatch(imageSelected(image.name));
  };

  const handleDragStart = (e: DragEvent<HTMLDivElement>) => {
    e.dataTransfer.setData('invokeai/imageName', image.name);
    e.dataTransfer.setData('invokeai/imageType', image.type);
    e.dataTransfer.effectAllowed = 'move';
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
          <MenuList>
            <MenuItem
              icon={<ExternalLinkIcon />}
              onClickCapture={handleOpenInNewTab}
            >
              {t('common.openInNewTab')}
            </MenuItem>
            {!disabledFeatures.includes('lightbox') && (
              <MenuItem icon={<FaExpand />} onClickCapture={handleLightBox}>
                {t('parameters.openInViewer')}
              </MenuItem>
            )}
            <MenuItem
              icon={<IoArrowUndoCircleOutline />}
              onClickCapture={handleUsePrompt}
              isDisabled={image?.metadata?.sd_metadata?.prompt === undefined}
            >
              {t('parameters.usePrompt')}
            </MenuItem>

            <MenuItem
              icon={<IoArrowUndoCircleOutline />}
              onClickCapture={handleUseSeed}
              isDisabled={image?.metadata?.sd_metadata?.seed === undefined}
            >
              {t('parameters.useSeed')}
            </MenuItem>
            <MenuItem
              icon={<IoArrowUndoCircleOutline />}
              onClickCapture={handleUseInitialImage}
              isDisabled={image?.metadata?.sd_metadata?.type !== 'img2img'}
            >
              {t('parameters.useInitImg')}
            </MenuItem>
            <MenuItem
              icon={<IoArrowUndoCircleOutline />}
              onClickCapture={handleUseAllParameters}
              isDisabled={
                !['txt2img', 'img2img'].includes(
                  image?.metadata?.sd_metadata?.type
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
            ref={ref}
            sx={{
              padding: 2,
              display: 'flex',
              justifyContent: 'center',
              transition: 'transform 0.2s ease-out',
              _hover: {
                cursor: 'pointer',

                zIndex: 2,
              },
              _before: {
                content: '""',
                display: 'block',
                paddingBottom: '100%',
              },
            }}
          >
            <Image
              objectFit={
                shouldUseSingleGalleryColumn ? 'contain' : galleryImageObjectFit
              }
              rounded="md"
              src={getUrl(thumbnail || url)}
              loading="lazy"
              sx={{
                position: 'absolute',
                width: '100%',
                height: '100%',
                maxWidth: '100%',
                maxHeight: '100%',
                top: '50%',
                transform: 'translate(-50%,-50%)',
                ...(direction === 'rtl'
                  ? { insetInlineEnd: '50%' }
                  : { insetInlineStart: '50%' }),
              }}
            />
            <Flex
              onClick={handleSelectImage}
              sx={{
                position: 'absolute',
                top: '0',
                insetInlineStart: '0',
                width: '100%',
                height: '100%',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              {isSelected && (
                <Icon
                  as={FaCheck}
                  sx={{
                    width: '50%',
                    height: '50%',
                    fill: 'ok.500',
                  }}
                />
              )}
            </Flex>
            {isHovered && galleryImageMinimumWidth >= 64 && (
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

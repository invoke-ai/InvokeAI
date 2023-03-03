import { Box, Icon, IconButton, Image, useToast } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import {
  setCurrentImage,
  setShouldHoldGalleryOpen,
} from 'features/gallery/store/gallerySlice';
import {
  setAllImageToImageParameters,
  setAllParameters,
  setInitialImage,
  setSeed,
} from 'features/parameters/store/generationSlice';
import { DragEvent, memo, useState } from 'react';
import { FaCheck, FaTrashAlt } from 'react-icons/fa';
import DeleteImageModal from './DeleteImageModal';

import * as ContextMenu from '@radix-ui/react-context-menu';
import * as InvokeAI from 'app/invokeai';
import {
  resizeAndScaleCanvas,
  setInitialCanvasImage,
} from 'features/canvas/store/canvasSlice';
import { hoverableImageSelector } from 'features/gallery/store/gallerySelectors';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { useTranslation } from 'react-i18next';
import useSetBothPrompts from 'features/parameters/hooks/usePrompt';

interface HoverableImageProps {
  image: InvokeAI.Image;
  isSelected: boolean;
}

const memoEqualityCheck = (
  prev: HoverableImageProps,
  next: HoverableImageProps
) => prev.image.uuid === next.image.uuid && prev.isSelected === next.isSelected;

/**
 * Gallery image component with delete/use all/use seed buttons on hover.
 */
const HoverableImage = memo((props: HoverableImageProps) => {
  const dispatch = useAppDispatch();
  const {
    activeTabName,
    galleryImageObjectFit,
    galleryImageMinimumWidth,
    mayDeleteImage,
    shouldUseSingleGalleryColumn,
  } = useAppSelector(hoverableImageSelector);
  const { image, isSelected } = props;
  const { url, thumbnail, uuid, metadata } = image;

  const [isHovered, setIsHovered] = useState<boolean>(false);

  const toast = useToast();
  const { t } = useTranslation();
  const setBothPrompts = useSetBothPrompts();

  const handleMouseOver = () => setIsHovered(true);

  const handleMouseOut = () => setIsHovered(false);

  const handleUsePrompt = () => {
    if (image.metadata?.image?.prompt) {
      setBothPrompts(image.metadata?.image?.prompt);
    }

    toast({
      title: t('toast.promptSet'),
      status: 'success',
      duration: 2500,
      isClosable: true,
    });
  };

  const handleUseSeed = () => {
    image.metadata && dispatch(setSeed(image.metadata.image.seed));
    toast({
      title: t('toast.seedSet'),
      status: 'success',
      duration: 2500,
      isClosable: true,
    });
  };

  const handleSendToImageToImage = () => {
    dispatch(setInitialImage(image));
    if (activeTabName !== 'img2img') {
      dispatch(setActiveTab('img2img'));
    }
    toast({
      title: t('toast.sentToImageToImage'),
      status: 'success',
      duration: 2500,
      isClosable: true,
    });
  };

  const handleSendToCanvas = () => {
    dispatch(setInitialCanvasImage(image));

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
    metadata && dispatch(setAllParameters(metadata));
    toast({
      title: t('toast.parametersSet'),
      status: 'success',
      duration: 2500,
      isClosable: true,
    });
  };

  const handleUseInitialImage = async () => {
    if (metadata?.image?.init_image_path) {
      const response = await fetch(metadata.image.init_image_path);
      if (response.ok) {
        dispatch(setActiveTab('img2img'));
        dispatch(setAllImageToImageParameters(metadata));
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

  const handleSelectImage = () => dispatch(setCurrentImage(image));

  const handleDragStart = (e: DragEvent<HTMLDivElement>) => {
    e.dataTransfer.setData('invokeai/imageUuid', uuid);
    e.dataTransfer.effectAllowed = 'move';
  };

  const handleLightBox = () => {
    dispatch(setCurrentImage(image));
  };

  return (
    <ContextMenu.Root
      onOpenChange={(open: boolean) => {
        dispatch(setShouldHoldGalleryOpen(open));
      }}
    >
      <ContextMenu.Trigger>
        <Box
          position="relative"
          key={uuid}
          className="hoverable-image"
          onMouseOver={handleMouseOver}
          onMouseOut={handleMouseOut}
          userSelect="none"
          draggable={true}
          onDragStart={handleDragStart}
        >
          <Image
            className="hoverable-image-image"
            objectFit={
              shouldUseSingleGalleryColumn ? 'contain' : galleryImageObjectFit
            }
            rounded="md"
            src={thumbnail || url}
            loading="lazy"
          />
          <div className="hoverable-image-content" onClick={handleSelectImage}>
            {isSelected && (
              <Icon
                width="50%"
                height="50%"
                as={FaCheck}
                className="hoverable-image-check"
              />
            )}
          </div>
          {isHovered && galleryImageMinimumWidth >= 64 && (
            <div className="hoverable-image-delete-button">
              <DeleteImageModal image={image}>
                <IconButton
                  aria-label={t('parameters.deleteImage')}
                  icon={<FaTrashAlt />}
                  size="xs"
                  variant="imageHoverIconButton"
                  fontSize={14}
                  isDisabled={!mayDeleteImage}
                />
              </DeleteImageModal>
            </div>
          )}
        </Box>
      </ContextMenu.Trigger>
      <ContextMenu.Content
        className="hoverable-image-context-menu"
        sticky="always"
        onInteractOutside={(e) => {
          e.detail.originalEvent.preventDefault();
        }}
      >
        <ContextMenu.Item onClickCapture={handleLightBox}>
          {t('parameters.openInViewer')}
        </ContextMenu.Item>
        <ContextMenu.Item
          onClickCapture={handleUsePrompt}
          disabled={image?.metadata?.image?.prompt === undefined}
        >
          {t('parameters.usePrompt')}
        </ContextMenu.Item>

        <ContextMenu.Item
          onClickCapture={handleUseSeed}
          disabled={image?.metadata?.image?.seed === undefined}
        >
          {t('parameters.useSeed')}
        </ContextMenu.Item>
        <ContextMenu.Item
          onClickCapture={handleUseAllParameters}
          disabled={
            !['txt2img', 'img2img'].includes(image?.metadata?.image?.type)
          }
        >
          {t('parameters.useAll')}
        </ContextMenu.Item>
        <ContextMenu.Item
          onClickCapture={handleUseInitialImage}
          disabled={image?.metadata?.image?.type !== 'img2img'}
        >
          {t('parameters.useInitImg')}
        </ContextMenu.Item>
        <ContextMenu.Item onClickCapture={handleSendToImageToImage}>
          {t('parameters.sendToImg2Img')}
        </ContextMenu.Item>
        <ContextMenu.Item onClickCapture={handleSendToCanvas}>
          {t('parameters.sendToUnifiedCanvas')}
        </ContextMenu.Item>
        <ContextMenu.Item data-warning>
          <DeleteImageModal image={image}>
            <p>{t('parameters.deleteImage')}</p>
          </DeleteImageModal>
        </ContextMenu.Item>
      </ContextMenu.Content>
    </ContextMenu.Root>
  );
}, memoEqualityCheck);

HoverableImage.displayName = 'HoverableImage';

export default HoverableImage;

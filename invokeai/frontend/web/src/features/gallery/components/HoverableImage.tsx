import {
  Box,
  Flex,
  Icon,
  Image,
  MenuItem,
  MenuList,
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
import { FaCheck, FaTrashAlt } from 'react-icons/fa';
import DeleteImageModal from './DeleteImageModal';
import { ContextMenu } from 'chakra-ui-contextmenu';
import * as InvokeAI from 'app/invokeai';
import {
  resizeAndScaleCanvas,
  setInitialCanvasImage,
} from 'features/canvas/store/canvasSlice';
import { hoverableImageSelector } from 'features/gallery/store/gallerySelectors';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { useTranslation } from 'react-i18next';
import useSetBothPrompts from 'features/parameters/hooks/usePrompt';
import { setIsLightboxOpen } from 'features/lightbox/store/lightboxSlice';
import IAIIconButton from 'common/components/IAIIconButton';

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
    mayDeleteImage,
    shouldUseSingleGalleryColumn,
  } = useAppSelector(hoverableImageSelector);
  const { image, isSelected } = props;
  const { url, thumbnail, name, metadata } = image;

  const [isHovered, setIsHovered] = useState<boolean>(false);

  const toast = useToast();
  const { direction } = useTheme();
  const { t } = useTranslation();
  const setBothPrompts = useSetBothPrompts();

  const handleMouseOver = () => setIsHovered(true);

  const handleMouseOut = () => setIsHovered(false);

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
        dispatch(setActiveTab('img2img'));
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
    // e.dataTransfer.setData('invokeai/imageUuid', uuid);
    // e.dataTransfer.effectAllowed = 'move';
  };

  const handleLightBox = () => {
    // dispatch(setCurrentImage(image));
    // dispatch(setIsLightboxOpen(true));
  };

  return (
    <ContextMenu<HTMLDivElement>
      menuProps={{ size: 'sm', isLazy: true }}
      renderMenu={() => (
        <MenuList>
          <MenuItem onClickCapture={handleLightBox}>
            {t('parameters.openInViewer')}
          </MenuItem>
          <MenuItem
            onClickCapture={handleUsePrompt}
            isDisabled={image?.metadata?.sd_metadata?.prompt === undefined}
          >
            {t('parameters.usePrompt')}
          </MenuItem>

          <MenuItem
            onClickCapture={handleUseSeed}
            isDisabled={image?.metadata?.sd_metadata?.seed === undefined}
          >
            {t('parameters.useSeed')}
          </MenuItem>
          <MenuItem
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
            onClickCapture={handleUseInitialImage}
            isDisabled={image?.metadata?.sd_metadata?.type !== 'img2img'}
          >
            {t('parameters.useInitImg')}
          </MenuItem>
          <MenuItem onClickCapture={handleSendToImageToImage}>
            {t('parameters.sendToImg2Img')}
          </MenuItem>
          <MenuItem onClickCapture={handleSendToCanvas}>
            {t('parameters.sendToUnifiedCanvas')}
          </MenuItem>
          <MenuItem data-warning>
            {/* <DeleteImageModal image={image}>
              <p>{t('parameters.deleteImage')}</p>
            </DeleteImageModal> */}
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
            _before: { content: '""', display: 'block', paddingBottom: '100%' },
          }}
        >
          <Image
            objectFit={
              shouldUseSingleGalleryColumn ? 'contain' : galleryImageObjectFit
            }
            rounded="md"
            src={thumbnail || url}
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
              {/* <DeleteImageModal image={image}>
                <IAIIconButton
                  aria-label={t('parameters.deleteImage')}
                  icon={<FaTrashAlt />}
                  size="xs"
                  fontSize={14}
                  isDisabled={!mayDeleteImage}
                />
              </DeleteImageModal> */}
            </Box>
          )}
        </Box>
      )}
    </ContextMenu>
  );
}, memoEqualityCheck);

HoverableImage.displayName = 'HoverableImage';

export default HoverableImage;

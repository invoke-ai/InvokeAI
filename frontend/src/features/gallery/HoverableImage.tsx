import {
  Box,
  Icon,
  IconButton,
  Image,
  Tooltip,
  useToast,
} from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from '../../app/store';
import { setCurrentImage } from './gallerySlice';
import { FaCheck, FaTrashAlt } from 'react-icons/fa';
import DeleteImageModal from './DeleteImageModal';
import { memo, useState } from 'react';
import {
  setActiveTab,
  setAllImageToImageParameters,
  setAllTextToImageParameters,
  setInitialImage,
  setPrompt,
  setSeed,
} from '../options/optionsSlice';
import * as InvokeAI from '../../app/invokeai';
import * as ContextMenu from '@radix-ui/react-context-menu';
import { setImageToInpaint } from '../tabs/Inpainting/inpaintingSlice';
import { hoverableImageSelector } from './gallerySliceSelectors';

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
  } = useAppSelector(hoverableImageSelector);
  const { image, isSelected } = props;
  const { url, uuid, metadata } = image;

  const [isHovered, setIsHovered] = useState<boolean>(false);

  const toast = useToast();

  const handleMouseOver = () => setIsHovered(true);

  const handleMouseOut = () => setIsHovered(false);

  const handleUsePrompt = () => {
    image.metadata && dispatch(setPrompt(image.metadata.image.prompt));
    toast({
      title: 'Prompt Set',
      status: 'success',
      duration: 2500,
      isClosable: true,
    });
  };

  const handleUseSeed = () => {
    image.metadata && dispatch(setSeed(image.metadata.image.seed));
    toast({
      title: 'Seed Set',
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
      title: 'Sent to Image To Image',
      status: 'success',
      duration: 2500,
      isClosable: true,
    });
  };

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

  const handleUseAllParameters = () => {
    metadata && dispatch(setAllTextToImageParameters(metadata));
    toast({
      title: 'Parameters Set',
      status: 'success',
      duration: 2500,
      isClosable: true,
    });
  };

  const handleUseInitialImage = async () => {
    // check if the image exists before setting it as initial image
    if (metadata?.image?.init_image_path) {
      const response = await fetch(metadata.image.init_image_path);
      if (response.ok) {
        dispatch(setActiveTab('img2img'));
        dispatch(setAllImageToImageParameters(metadata));
        toast({
          title: 'Initial Image Set',
          status: 'success',
          duration: 2500,
          isClosable: true,
        });
        return;
      }
    }
    toast({
      title: 'Initial Image Not Set',
      description: 'Could not load initial image.',
      status: 'error',
      duration: 2500,
      isClosable: true,
    });
  };

  const handleSelectImage = () => dispatch(setCurrentImage(image));

  return (
    <ContextMenu.Root
    // onOpenChange={(open: boolean) => {
    //   dispatch(setShouldHoldGalleryOpen(open));
    //   dispatch(setShouldShowGallery(true));
    // }}
    >
      <ContextMenu.Trigger>
        <Box
          position={'relative'}
          key={uuid}
          className="hoverable-image"
          onMouseOver={handleMouseOver}
          onMouseOut={handleMouseOut}
        >
          <Image
            className="hoverable-image-image"
            objectFit={galleryImageObjectFit}
            rounded={'md'}
            src={url}
            loading={'lazy'}
          />
          <div className="hoverable-image-content" onClick={handleSelectImage}>
            {isSelected && (
              <Icon
                width={'50%'}
                height={'50%'}
                as={FaCheck}
                className="hoverable-image-check"
              />
            )}
          </div>
          {isHovered && galleryImageMinimumWidth >= 64 && (
            <div className="hoverable-image-delete-button">
              <Tooltip label={'Delete image'} hasArrow>
                <DeleteImageModal image={image}>
                  <IconButton
                    aria-label="Delete image"
                    icon={<FaTrashAlt />}
                    size="xs"
                    variant={'imageHoverIconButton'}
                    fontSize={14}
                    isDisabled={!mayDeleteImage}
                  />
                </DeleteImageModal>
              </Tooltip>
            </div>
          )}
        </Box>
      </ContextMenu.Trigger>
      <ContextMenu.Content
        className="hoverable-image-context-menu"
        sticky={'always'}
      >
        <ContextMenu.Item
          onClickCapture={handleUsePrompt}
          disabled={image?.metadata?.image?.prompt === undefined}
        >
          Use Prompt
        </ContextMenu.Item>

        <ContextMenu.Item
          onClickCapture={handleUseSeed}
          disabled={image?.metadata?.image?.seed === undefined}
        >
          Use Seed
        </ContextMenu.Item>
        <ContextMenu.Item
          onClickCapture={handleUseAllParameters}
          disabled={
            !['txt2img', 'img2img'].includes(image?.metadata?.image?.type)
          }
        >
          Use All Parameters
        </ContextMenu.Item>
        <Tooltip label="Load initial image used for this generation">
          <ContextMenu.Item
            onClickCapture={handleUseInitialImage}
            disabled={image?.metadata?.image?.type !== 'img2img'}
          >
            Use Initial Image
          </ContextMenu.Item>
        </Tooltip>
        <ContextMenu.Item onClickCapture={handleSendToImageToImage}>
          Send to Image To Image
        </ContextMenu.Item>
        <ContextMenu.Item onClickCapture={handleSendToInpainting}>
          Send to Inpainting
        </ContextMenu.Item>
        <DeleteImageModal image={image}>
          <ContextMenu.Item data-warning>Delete Image</ContextMenu.Item>
        </DeleteImageModal>
      </ContextMenu.Content>
    </ContextMenu.Root>
  );
}, memoEqualityCheck);

export default HoverableImage;

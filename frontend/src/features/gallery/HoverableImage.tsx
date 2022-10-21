import {
  Box,
  Icon,
  IconButton,
  Image,
  Tooltip,
  useToast,
} from '@chakra-ui/react';
import { RootState, useAppDispatch, useAppSelector } from '../../app/store';
import { setCurrentImage } from './gallerySlice';
import { FaCheck, FaTrashAlt } from 'react-icons/fa';
import DeleteImageModal from './DeleteImageModal';
import { memo, useState } from 'react';
import {
  setActiveTab,
  setAllImageToImageParameters,
  setAllTextToImageParameters,
  setInitialImagePath,
  setPrompt,
  setSeed,
} from '../options/optionsSlice';
import * as InvokeAI from '../../app/invokeai';
import * as ContextMenu from '@radix-ui/react-context-menu';
import { tabMap } from '../tabs/InvokeTabs';

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
  const activeTab = useAppSelector(
    (state: RootState) => state.options.activeTab
  );

  const [isHovered, setIsHovered] = useState<boolean>(false);

  const toast = useToast();

  const { image, isSelected } = props;
  const { url, uuid, metadata } = image;

  const handleMouseOver = () => setIsHovered(true);

  const handleMouseOut = () => setIsHovered(false);

  const handleUsePrompt = () => {
    dispatch(setPrompt(image.metadata.image.prompt));
    toast({
      title: 'Prompt Set',
      status: 'success',
      duration: 2500,
      isClosable: true,
    });
  };

  const handleUseSeed = () => {
    dispatch(setSeed(image.metadata.image.seed));
    toast({
      title: 'Seed Set',
      status: 'success',
      duration: 2500,
      isClosable: true,
    });
  };

  const handleSendToImageToImage = () => {
    dispatch(setInitialImagePath(image.url));
    if (activeTab !== 1) {
      dispatch(setActiveTab(1));
    }
    toast({
      title: 'Sent to Image To Image',
      status: 'success',
      duration: 2500,
      isClosable: true,
    });
  };

  const handleUseAllParameters = () => {
    dispatch(setAllTextToImageParameters(metadata));
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
        dispatch(setActiveTab(tabMap.indexOf('img2img')));
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
    <ContextMenu.Root>
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
            objectFit="cover"
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
          {isHovered && (
            <div className="hoverable-image-delete-button">
              <Tooltip label={'Delete image'} hasArrow>
                <DeleteImageModal image={image}>
                  <IconButton
                    aria-label="Delete image"
                    icon={<FaTrashAlt />}
                    size="xs"
                    variant={'imageHoverIconButton'}
                    fontSize={14}
                  />
                </DeleteImageModal>
              </Tooltip>
            </div>
          )}
        </Box>
      </ContextMenu.Trigger>
      <ContextMenu.Content className="hoverable-image-context-menu">
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
        <DeleteImageModal image={image}>
          <ContextMenu.Item data-warning>Delete Image</ContextMenu.Item>
        </DeleteImageModal>
      </ContextMenu.Content>
    </ContextMenu.Root>
  );
}, memoEqualityCheck);

export default HoverableImage;

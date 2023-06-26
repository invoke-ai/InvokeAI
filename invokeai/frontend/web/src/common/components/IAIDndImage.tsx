import {
  Box,
  ChakraProps,
  Flex,
  Icon,
  IconButtonProps,
  Image,
} from '@chakra-ui/react';
import { useDraggable, useDroppable } from '@dnd-kit/core';
import { useCombinedRefs } from '@dnd-kit/utilities';
import IAIIconButton from 'common/components/IAIIconButton';
import { IAIImageLoadingFallback } from 'common/components/IAIImageFallback';
import ImageMetadataOverlay from 'common/components/ImageMetadataOverlay';
import { AnimatePresence } from 'framer-motion';
import { ReactElement, SyntheticEvent } from 'react';
import { memo, useRef } from 'react';
import { FaImage, FaUndo, FaUpload } from 'react-icons/fa';
import { ImageDTO } from 'services/api/types';
import { v4 as uuidv4 } from 'uuid';
import IAIDropOverlay from './IAIDropOverlay';
import { PostUploadAction } from 'services/api/thunks/image';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';

type IAIDndImageProps = {
  image: ImageDTO | null | undefined;
  onDrop: (droppedImage: ImageDTO) => void;
  onReset?: () => void;
  onError?: (event: SyntheticEvent<HTMLImageElement>) => void;
  onLoad?: (event: SyntheticEvent<HTMLImageElement>) => void;
  resetIconSize?: IconButtonProps['size'];
  withResetIcon?: boolean;
  withMetadataOverlay?: boolean;
  isDragDisabled?: boolean;
  isDropDisabled?: boolean;
  isUploadDisabled?: boolean;
  fallback?: ReactElement;
  payloadImage?: ImageDTO | null | undefined;
  minSize?: number;
  postUploadAction?: PostUploadAction;
  imageSx?: ChakraProps['sx'];
  fitContainer?: boolean;
};

const IAIDndImage = (props: IAIDndImageProps) => {
  const {
    image,
    onDrop,
    onReset,
    onError,
    resetIconSize = 'md',
    withResetIcon = false,
    withMetadataOverlay = false,
    isDropDisabled = false,
    isDragDisabled = false,
    isUploadDisabled = false,
    fallback = <IAIImageLoadingFallback />,
    payloadImage,
    minSize = 24,
    postUploadAction,
    imageSx,
    fitContainer = false,
  } = props;

  const dndId = useRef(uuidv4());

  const {
    isOver,
    setNodeRef: setDroppableRef,
    active: isDropActive,
  } = useDroppable({
    id: dndId.current,
    disabled: isDropDisabled,
    data: {
      handleDrop: onDrop,
    },
  });

  const {
    attributes,
    listeners,
    setNodeRef: setDraggableRef,
    isDragging,
  } = useDraggable({
    id: dndId.current,
    data: {
      image: payloadImage ? payloadImage : image,
    },
    disabled: isDragDisabled || !image,
  });

  const { getUploadButtonProps, getUploadInputProps } = useImageUploadButton({
    postUploadAction,
    isDisabled: isUploadDisabled,
  });

  const setNodeRef = useCombinedRefs(setDroppableRef, setDraggableRef);

  const uploadButtonStyles = isUploadDisabled
    ? {}
    : {
        cursor: 'pointer',
        bg: 'base.800',
        _hover: {
          bg: 'base.750',
          color: 'base.300',
        },
      };

  return (
    <Flex
      sx={{
        width: 'full',
        height: 'full',
        alignItems: 'center',
        justifyContent: 'center',
        position: 'relative',
        minW: minSize,
        minH: minSize,
        userSelect: 'none',
        cursor: isDragDisabled || !image ? 'auto' : 'grab',
      }}
      {...attributes}
      {...listeners}
      ref={setNodeRef}
    >
      {image && (
        <Flex
          sx={{
            w: 'full',
            h: 'full',
            position: fitContainer ? 'absolute' : 'relative',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Image
            src={image.image_url}
            fallback={fallback}
            onError={onError}
            objectFit="contain"
            draggable={false}
            sx={{
              maxW: 'full',
              maxH: 'full',
              borderRadius: 'base',
              ...imageSx,
            }}
          />
          {withMetadataOverlay && <ImageMetadataOverlay image={image} />}
          {onReset && withResetIcon && (
            <Box
              sx={{
                position: 'absolute',
                top: 0,
                right: 0,
              }}
            >
              <IAIIconButton
                size={resetIconSize}
                tooltip="Reset Image"
                aria-label="Reset Image"
                icon={<FaUndo />}
                onClick={onReset}
              />
            </Box>
          )}
          <AnimatePresence>
            {isDropActive && <IAIDropOverlay isOver={isOver} />}
          </AnimatePresence>
        </Flex>
      )}
      {!image && (
        <>
          <Flex
            sx={{
              minH: minSize,
              w: 'full',
              h: 'full',
              alignItems: 'center',
              justifyContent: 'center',
              borderRadius: 'base',
              transitionProperty: 'common',
              transitionDuration: '0.1s',
              color: 'base.500',
              ...uploadButtonStyles,
            }}
            {...getUploadButtonProps()}
          >
            <input {...getUploadInputProps()} />
            <Icon
              as={isUploadDisabled ? FaImage : FaUpload}
              sx={{
                boxSize: 12,
              }}
            />
          </Flex>
          <AnimatePresence>
            {isDropActive && <IAIDropOverlay isOver={isOver} />}
          </AnimatePresence>
        </>
      )}
    </Flex>
  );
};

export default memo(IAIDndImage);

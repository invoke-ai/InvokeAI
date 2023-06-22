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
import { ReactElement, SyntheticEvent, useCallback } from 'react';
import { memo, useRef } from 'react';
import { FaImage, FaTimes, FaUndo, FaUpload } from 'react-icons/fa';
import { ImageDTO } from 'services/api';
import { v4 as uuidv4 } from 'uuid';
import IAIDropOverlay from './IAIDropOverlay';
import { PostUploadAction, imageUploaded } from 'services/thunks/image';
import { useDropzone } from 'react-dropzone';
import { useAppDispatch } from 'app/store/storeHooks';

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
  } = props;
  const dispatch = useAppDispatch();
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

  const handleOnDropAccepted = useCallback(
    (files: Array<File>) => {
      const file = files[0];
      if (!file) {
        return;
      }

      dispatch(
        imageUploaded({
          formData: { file },
          imageCategory: 'user',
          isIntermediate: false,
          postUploadAction,
        })
      );
    },
    [dispatch, postUploadAction]
  );

  const { getRootProps, getInputProps } = useDropzone({
    accept: { 'image/png': ['.png'], 'image/jpeg': ['.jpg', '.jpeg', '.png'] },
    onDropAccepted: handleOnDropAccepted,
    noDrag: true,
    multiple: false,
    disabled: isUploadDisabled,
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
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Image
            src={image.image_url}
            fallbackStrategy="beforeLoadOrError"
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
            {...getRootProps()}
          >
            <input {...getInputProps()} />
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

import { Box, Flex, Icon, IconButtonProps, Image } from '@chakra-ui/react';
import { useDraggable, useDroppable } from '@dnd-kit/core';
import { useCombinedRefs } from '@dnd-kit/utilities';
import IAIIconButton from 'common/components/IAIIconButton';
import { IAIImageFallback } from 'common/components/IAIImageFallback';
import ImageMetadataOverlay from 'common/components/ImageMetadataOverlay';
import { AnimatePresence } from 'framer-motion';
import { ReactElement, SyntheticEvent } from 'react';
import { memo, useRef } from 'react';
import { FaImage, FaTimes } from 'react-icons/fa';
import { ImageDTO } from 'services/api';
import { v4 as uuidv4 } from 'uuid';
import IAIDropOverlay from './IAIDropOverlay';

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
  fallback?: ReactElement;
  payloadImage?: ImageDTO | null | undefined;
  minSize?: number;
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
    fallback = <IAIImageFallback />,
    payloadImage,
    minSize = 24,
  } = props;
  const dndId = useRef(uuidv4());
  const {
    isOver,
    setNodeRef: setDroppableRef,
    active,
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
  } = useDraggable({
    id: dndId.current,
    data: {
      image: payloadImage ? payloadImage : image,
    },
    disabled: isDragDisabled,
  });

  const setNodeRef = useCombinedRefs(setDroppableRef, setDraggableRef);

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
        cursor: 'grab',
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
            position: 'relative',
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
            }}
          />
          {withMetadataOverlay && <ImageMetadataOverlay image={image} />}
          {onReset && withResetIcon && (
            <Box
              sx={{
                position: 'absolute',
                top: 0,
                right: 0,
                p: 2,
              }}
            >
              <IAIIconButton
                size={resetIconSize}
                tooltip="Reset Image"
                aria-label="Reset Image"
                icon={<FaTimes />}
                onClick={onReset}
              />
            </Box>
          )}
          <AnimatePresence>
            {active && <IAIDropOverlay isOver={isOver} />}
          </AnimatePresence>
        </Flex>
      )}
      {!image && (
        <>
          <Flex
            sx={{
              minH: minSize,
              bg: 'base.850',
              w: 'full',
              h: 'full',
              alignItems: 'center',
              justifyContent: 'center',
              borderRadius: 'base',
            }}
          >
            <Icon
              as={FaImage}
              sx={{
                boxSize: 24,
                color: 'base.500',
              }}
            />
          </Flex>
          <AnimatePresence>
            {active && <IAIDropOverlay isOver={isOver} />}
          </AnimatePresence>
        </>
      )}
    </Flex>
  );
};

export default memo(IAIDndImage);

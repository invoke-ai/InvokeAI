import {
  Box,
  Flex,
  Icon,
  IconButtonProps,
  Image,
  Text,
} from '@chakra-ui/react';
import { useDraggable, useDroppable } from '@dnd-kit/core';
import { useCombinedRefs } from '@dnd-kit/utilities';
import IAIIconButton from 'common/components/IAIIconButton';
import { IAIImageFallback } from 'common/components/IAIImageFallback';
import ImageMetadataOverlay from 'common/components/ImageMetadataOverlay';
import { useGetUrl } from 'common/util/getUrl';
import { AnimatePresence, motion } from 'framer-motion';
import { ReactElement, SyntheticEvent } from 'react';
import { memo, useRef } from 'react';
import { FaImage, FaTimes } from 'react-icons/fa';
import { ImageDTO } from 'services/api';
import { v4 as uuidv4 } from 'uuid';

type IAIDndImageProps = {
  image: ImageDTO | null | undefined;
  onDrop: (image: ImageDTO) => void;
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
  const { getUrl } = useGetUrl();
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
            src={getUrl(image.image_url)}
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
            {active && <DropOverlay isOver={isOver} />}
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
            {active && <DropOverlay isOver={isOver} />}
          </AnimatePresence>
        </>
      )}
    </Flex>
  );
};

export default memo(IAIDndImage);

type DropOverlayProps = {
  isOver: boolean;
};

const DropOverlay = (props: DropOverlayProps) => {
  const { isOver } = props;
  return (
    <motion.div
      key="statusText"
      initial={{
        opacity: 0,
      }}
      animate={{
        opacity: 1,
        transition: { duration: 0.1 },
      }}
      exit={{
        opacity: 0,
        transition: { duration: 0.1 },
      }}
    >
      <Flex
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          w: 'full',
          h: 'full',
        }}
      >
        <Flex
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            w: 'full',
            h: 'full',
            bg: 'base.900',
            opacity: 0.7,
            borderRadius: 'base',
            alignItems: 'center',
            justifyContent: 'center',
            transitionProperty: 'common',
            transitionDuration: '0.1s',
          }}
        />

        <Flex
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            w: 'full',
            h: 'full',
            opacity: 1,
            borderWidth: 2,
            borderColor: isOver ? 'base.200' : 'base.500',
            borderRadius: 'base',
            borderStyle: 'dashed',
            transitionProperty: 'common',
            transitionDuration: '0.1s',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Text
            sx={{
              fontSize: '2xl',
              fontWeight: 600,
              transform: isOver ? 'scale(1.1)' : 'scale(1)',
              color: isOver ? 'base.100' : 'base.500',
              transitionProperty: 'common',
              transitionDuration: '0.1s',
            }}
          >
            Drop
          </Text>
        </Flex>
      </Flex>
    </motion.div>
  );
};

import {
  ChakraProps,
  Flex,
  FlexProps,
  Icon,
  Image,
  useColorMode,
} from '@chakra-ui/react';
import {
  IAILoadingImageFallback,
  IAINoContentFallback,
} from 'common/components/IAIImageFallback';
import ImageMetadataOverlay from 'common/components/ImageMetadataOverlay';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import ImageContextMenu from 'features/gallery/components/ImageContextMenu/ImageContextMenu';
import {
  MouseEvent,
  ReactElement,
  ReactNode,
  SyntheticEvent,
  memo,
  useCallback,
  useState,
} from 'react';
import { FaImage, FaUpload } from 'react-icons/fa';
import { ImageDTO, PostUploadAction } from 'services/api/types';
import { mode } from 'theme/util/mode';
import IAIDraggable from './IAIDraggable';
import IAIDroppable from './IAIDroppable';
import SelectionOverlay from './SelectionOverlay';
import {
  TypesafeDraggableData,
  TypesafeDroppableData,
} from 'features/dnd/types';

const defaultUploadElement = (
  <Icon
    as={FaUpload}
    sx={{
      boxSize: 16,
    }}
  />
);

const defaultNoContentFallback = <IAINoContentFallback icon={FaImage} />;

type IAIDndImageProps = FlexProps & {
  imageDTO: ImageDTO | undefined;
  onError?: (event: SyntheticEvent<HTMLImageElement>) => void;
  onLoad?: (event: SyntheticEvent<HTMLImageElement>) => void;
  onClick?: (event: MouseEvent<HTMLDivElement>) => void;
  withMetadataOverlay?: boolean;
  isDragDisabled?: boolean;
  isDropDisabled?: boolean;
  isUploadDisabled?: boolean;
  minSize?: number;
  postUploadAction?: PostUploadAction;
  imageSx?: ChakraProps['sx'];
  fitContainer?: boolean;
  droppableData?: TypesafeDroppableData;
  draggableData?: TypesafeDraggableData;
  dropLabel?: ReactNode;
  isSelected?: boolean;
  thumbnail?: boolean;
  noContentFallback?: ReactElement;
  useThumbailFallback?: boolean;
  withHoverOverlay?: boolean;
  children?: JSX.Element;
  uploadElement?: ReactNode;
};

const IAIDndImage = (props: IAIDndImageProps) => {
  const {
    imageDTO,
    onError,
    onClick,
    withMetadataOverlay = false,
    isDropDisabled = false,
    isDragDisabled = false,
    isUploadDisabled = false,
    minSize = 24,
    postUploadAction,
    imageSx,
    fitContainer = false,
    droppableData,
    draggableData,
    dropLabel,
    isSelected = false,
    thumbnail = false,
    noContentFallback = defaultNoContentFallback,
    uploadElement = defaultUploadElement,
    useThumbailFallback,
    withHoverOverlay = false,
    children,
    onMouseOver,
    onMouseOut,
  } = props;

  const { colorMode } = useColorMode();
  const [isHovered, setIsHovered] = useState(false);
  const handleMouseOver = useCallback(
    (e: MouseEvent<HTMLDivElement>) => {
      if (onMouseOver) {
        onMouseOver(e);
      }
      setIsHovered(true);
    },
    [onMouseOver]
  );
  const handleMouseOut = useCallback(
    (e: MouseEvent<HTMLDivElement>) => {
      if (onMouseOut) {
        onMouseOut(e);
      }
      setIsHovered(false);
    },
    [onMouseOut]
  );

  const { getUploadButtonProps, getUploadInputProps } = useImageUploadButton({
    postUploadAction,
    isDisabled: isUploadDisabled,
  });

  const uploadButtonStyles = isUploadDisabled
    ? {}
    : {
        cursor: 'pointer',
        bg: mode('base.200', 'base.700')(colorMode),
        _hover: {
          bg: mode('base.300', 'base.650')(colorMode),
          color: mode('base.500', 'base.300')(colorMode),
        },
      };

  return (
    <ImageContextMenu imageDTO={imageDTO}>
      {(ref) => (
        <Flex
          ref={ref}
          onMouseOver={handleMouseOver}
          onMouseOut={handleMouseOut}
          sx={{
            width: 'full',
            height: 'full',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative',
            minW: minSize ? minSize : undefined,
            minH: minSize ? minSize : undefined,
            userSelect: 'none',
            cursor: isDragDisabled || !imageDTO ? 'default' : 'pointer',
          }}
        >
          {imageDTO && (
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
                src={thumbnail ? imageDTO.thumbnail_url : imageDTO.image_url}
                fallbackStrategy="beforeLoadOrError"
                fallbackSrc={
                  useThumbailFallback ? imageDTO.thumbnail_url : undefined
                }
                fallback={
                  useThumbailFallback ? undefined : (
                    <IAILoadingImageFallback image={imageDTO} />
                  )
                }
                onError={onError}
                draggable={false}
                sx={{
                  w: imageDTO.width,
                  objectFit: 'contain',
                  maxW: 'full',
                  maxH: 'full',
                  borderRadius: 'base',
                  ...imageSx,
                }}
              />
              {withMetadataOverlay && (
                <ImageMetadataOverlay imageDTO={imageDTO} />
              )}
              <SelectionOverlay
                isSelected={isSelected}
                isHovered={withHoverOverlay ? isHovered : false}
              />
            </Flex>
          )}
          {!imageDTO && !isUploadDisabled && (
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
                  color: mode('base.500', 'base.500')(colorMode),
                  ...uploadButtonStyles,
                }}
                {...getUploadButtonProps()}
              >
                <input {...getUploadInputProps()} />
                {uploadElement}
              </Flex>
            </>
          )}
          {!imageDTO && isUploadDisabled && noContentFallback}
          {imageDTO && !isDragDisabled && (
            <IAIDraggable
              data={draggableData}
              disabled={isDragDisabled || !imageDTO}
              onClick={onClick}
            />
          )}
          {children}
          {!isDropDisabled && (
            <IAIDroppable
              data={droppableData}
              disabled={isDropDisabled}
              dropLabel={dropLabel}
            />
          )}
        </Flex>
      )}
    </ImageContextMenu>
  );
};

export default memo(IAIDndImage);

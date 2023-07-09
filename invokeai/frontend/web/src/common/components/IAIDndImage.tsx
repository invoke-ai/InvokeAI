import {
  ChakraProps,
  Flex,
  Icon,
  Image,
  useColorMode,
  useColorModeValue,
} from '@chakra-ui/react';
import {
  TypesafeDraggableData,
  TypesafeDroppableData,
} from 'app/components/ImageDnd/typesafeDnd';
import IAIIconButton from 'common/components/IAIIconButton';
import {
  IAILoadingImageFallback,
  IAINoContentFallback,
} from 'common/components/IAIImageFallback';
import ImageMetadataOverlay from 'common/components/ImageMetadataOverlay';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import { MouseEvent, ReactElement, SyntheticEvent, memo } from 'react';
import { FaImage, FaUndo, FaUpload } from 'react-icons/fa';
import { PostUploadAction } from 'services/api/thunks/image';
import { ImageDTO } from 'services/api/types';
import { mode } from 'theme/util/mode';
import IAIDraggable from './IAIDraggable';
import IAIDroppable from './IAIDroppable';

type IAIDndImageProps = {
  imageDTO: ImageDTO | undefined;
  onError?: (event: SyntheticEvent<HTMLImageElement>) => void;
  onLoad?: (event: SyntheticEvent<HTMLImageElement>) => void;
  onClick?: (event: MouseEvent<HTMLDivElement>) => void;
  onClickReset?: (event: MouseEvent<HTMLButtonElement>) => void;
  withResetIcon?: boolean;
  resetIcon?: ReactElement;
  resetTooltip?: string;
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
  dropLabel?: string;
  isSelected?: boolean;
  thumbnail?: boolean;
  noContentFallback?: ReactElement;
};

const IAIDndImage = (props: IAIDndImageProps) => {
  const {
    imageDTO,
    onClickReset,
    onError,
    onClick,
    withResetIcon = false,
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
    resetTooltip = 'Reset',
    resetIcon = <FaUndo />,
    noContentFallback = <IAINoContentFallback icon={FaImage} />,
  } = props;

  const { colorMode } = useColorMode();

  const { getUploadButtonProps, getUploadInputProps } = useImageUploadButton({
    postUploadAction,
    isDisabled: isUploadDisabled,
  });

  const resetIconShadow = useColorModeValue(
    `drop-shadow(0px 0px 0.1rem var(--invokeai-colors-base-600))`,
    `drop-shadow(0px 0px 0.1rem var(--invokeai-colors-base-800))`
  );

  const uploadButtonStyles = isUploadDisabled
    ? {}
    : {
        cursor: 'pointer',
        bg: mode('base.200', 'base.800')(colorMode),
        _hover: {
          bg: mode('base.300', 'base.650')(colorMode),
          color: mode('base.500', 'base.300')(colorMode),
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
            fallback={<IAILoadingImageFallback image={imageDTO} />}
            onError={onError}
            draggable={false}
            sx={{
              objectFit: 'contain',
              maxW: 'full',
              maxH: 'full',
              borderRadius: 'base',
              shadow: isSelected ? 'selected.light' : undefined,
              _dark: { shadow: isSelected ? 'selected.dark' : undefined },
              ...imageSx,
            }}
          />
          {withMetadataOverlay && <ImageMetadataOverlay image={imageDTO} />}
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
            <Icon
              as={FaUpload}
              sx={{
                boxSize: 16,
              }}
            />
          </Flex>
        </>
      )}
      {!imageDTO && isUploadDisabled && noContentFallback}
      <IAIDroppable
        data={droppableData}
        disabled={isDropDisabled}
        dropLabel={dropLabel}
      />
      {imageDTO && (
        <IAIDraggable
          data={draggableData}
          disabled={isDragDisabled || !imageDTO}
          onClick={onClick}
        />
      )}
      {onClickReset && withResetIcon && imageDTO && (
        <IAIIconButton
          onClick={onClickReset}
          aria-label={resetTooltip}
          tooltip={resetTooltip}
          icon={resetIcon}
          size="sm"
          variant="link"
          sx={{
            position: 'absolute',
            top: 1,
            insetInlineEnd: 1,
            p: 0,
            minW: 0,
            svg: {
              transitionProperty: 'common',
              transitionDuration: 'normal',
              fill: 'base.100',
              _hover: { fill: 'base.50' },
              filter: resetIconShadow,
            },
          }}
        />
      )}
    </Flex>
  );
};

export default memo(IAIDndImage);

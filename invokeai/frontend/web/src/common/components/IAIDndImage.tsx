import type { ChakraProps, FlexProps, SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, Icon, Image } from '@invoke-ai/ui-library';
import { IAILoadingImageFallback, IAINoContentFallback } from 'common/components/IAIImageFallback';
import ImageMetadataOverlay from 'common/components/ImageMetadataOverlay';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import type { TypesafeDraggableData, TypesafeDroppableData } from 'features/dnd/types';
import { useImageContextMenu } from 'features/gallery/components/ImageContextMenu/ImageContextMenu';
import type { MouseEvent, ReactElement, ReactNode, SyntheticEvent } from 'react';
import { memo, useCallback, useMemo, useRef } from 'react';
import { PiImageBold, PiUploadSimpleBold } from 'react-icons/pi';
import type { ImageDTO, PostUploadAction } from 'services/api/types';

import IAIDraggable from './IAIDraggable';
import IAIDroppable from './IAIDroppable';

const defaultUploadElement = <Icon as={PiUploadSimpleBold} boxSize={16} />;

const defaultNoContentFallback = <IAINoContentFallback icon={PiImageBold} />;

const baseStyles: SystemStyleObject = {
  touchAction: 'none',
  userSelect: 'none',
  webkitUserSelect: 'none',
};

const sx: SystemStyleObject = {
  ...baseStyles,
  '.gallery-image-container::before': {
    content: '""',
    display: 'inline-block',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    pointerEvents: 'none',
    borderRadius: 'base',
  },
  '&[data-selected="selected"]>.gallery-image-container::before': {
    boxShadow:
      'inset 0px 0px 0px 3px var(--invoke-colors-invokeBlue-500), inset 0px 0px 0px 4px var(--invoke-colors-invokeBlue-800)',
  },
  '&[data-selected="selectedForCompare"]>.gallery-image-container::before': {
    boxShadow:
      'inset 0px 0px 0px 3px var(--invoke-colors-invokeGreen-300), inset 0px 0px 0px 4px var(--invoke-colors-invokeGreen-800)',
  },
  '&:hover>.gallery-image-container::before': {
    boxShadow:
      'inset 0px 0px 0px 2px var(--invoke-colors-invokeBlue-300), inset 0px 0px 0px 3px var(--invoke-colors-invokeBlue-800)',
  },
  '&:hover[data-selected="selected"]>.gallery-image-container::before': {
    boxShadow:
      'inset 0px 0px 0px 3px var(--invoke-colors-invokeBlue-400), inset 0px 0px 0px 4px var(--invoke-colors-invokeBlue-800)',
  },
  '&:hover[data-selected="selectedForCompare"]>.gallery-image-container::before': {
    boxShadow:
      'inset 0px 0px 0px 3px var(--invoke-colors-invokeGreen-200), inset 0px 0px 0px 4px var(--invoke-colors-invokeGreen-800)',
  },
};

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
  dropLabel?: string;
  isSelected?: boolean;
  isSelectedForCompare?: boolean;
  thumbnail?: boolean;
  noContentFallback?: ReactElement;
  useThumbailFallback?: boolean;
  withHoverOverlay?: boolean;
  children?: JSX.Element;
  uploadElement?: ReactNode;
  dataTestId?: string;
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
    isSelectedForCompare = false,
    thumbnail = false,
    noContentFallback = defaultNoContentFallback,
    uploadElement = defaultUploadElement,
    useThumbailFallback,
    withHoverOverlay = false,
    children,
    dataTestId,
    ...rest
  } = props;

  const openInNewTab = useCallback(
    (e: MouseEvent) => {
      if (!imageDTO) {
        return;
      }
      if (e.button !== 1) {
        return;
      }
      window.open(imageDTO.image_url, '_blank');
    },
    [imageDTO]
  );

  const ref = useRef<HTMLDivElement>(null);
  useImageContextMenu(imageDTO, ref);

  return (
    <Flex
      ref={ref}
      width="full"
      height="full"
      alignItems="center"
      justifyContent="center"
      position="relative"
      minW={minSize ? minSize : undefined}
      minH={minSize ? minSize : undefined}
      userSelect="none"
      cursor={isDragDisabled || !imageDTO ? 'default' : 'pointer'}
      sx={withHoverOverlay ? sx : baseStyles}
      data-selected={isSelectedForCompare ? 'selectedForCompare' : isSelected ? 'selected' : undefined}
      {...rest}
    >
      {imageDTO && (
        <Flex
          className="gallery-image-container"
          w="full"
          h="full"
          position={fitContainer ? 'absolute' : 'relative'}
          alignItems="center"
          justifyContent="center"
        >
          <Image
            src={thumbnail ? imageDTO.thumbnail_url : imageDTO.image_url}
            fallbackStrategy="beforeLoadOrError"
            fallbackSrc={useThumbailFallback ? imageDTO.thumbnail_url : undefined}
            fallback={useThumbailFallback ? undefined : <IAILoadingImageFallback image={imageDTO} />}
            onError={onError}
            draggable={false}
            w={imageDTO.width}
            objectFit="contain"
            maxW="full"
            maxH="full"
            borderRadius="base"
            sx={imageSx}
            data-testid={dataTestId}
          />
          {withMetadataOverlay && <ImageMetadataOverlay imageDTO={imageDTO} />}
        </Flex>
      )}
      {!imageDTO && !isUploadDisabled && (
        <UploadButton
          isUploadDisabled={isUploadDisabled}
          postUploadAction={postUploadAction}
          uploadElement={uploadElement}
          minSize={minSize}
        />
      )}
      {!imageDTO && isUploadDisabled && noContentFallback}
      {imageDTO && !isDragDisabled && (
        <IAIDraggable
          data={draggableData}
          disabled={isDragDisabled || !imageDTO}
          onClick={onClick}
          onAuxClick={openInNewTab}
        />
      )}
      {children}
      {!isDropDisabled && <IAIDroppable data={droppableData} disabled={isDropDisabled} dropLabel={dropLabel} />}
    </Flex>
  );
};

export default memo(IAIDndImage);

const UploadButton = memo(
  ({
    isUploadDisabled,
    postUploadAction,
    uploadElement,
    minSize,
  }: {
    isUploadDisabled: boolean;
    postUploadAction?: PostUploadAction;
    uploadElement: ReactNode;
    minSize: number;
  }) => {
    const { getUploadButtonProps, getUploadInputProps } = useImageUploadButton({
      postUploadAction,
      isDisabled: isUploadDisabled,
    });

    const uploadButtonStyles = useMemo<SystemStyleObject>(() => {
      const styles: SystemStyleObject = {
        minH: minSize,
        w: 'full',
        h: 'full',
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: 'base',
        transitionProperty: 'common',
        transitionDuration: '0.1s',
        color: 'base.500',
      };
      if (!isUploadDisabled) {
        Object.assign(styles, {
          cursor: 'pointer',
          bg: 'base.700',
          _hover: {
            bg: 'base.650',
            color: 'base.300',
          },
        });
      }
      return styles;
    }, [isUploadDisabled, minSize]);

    return (
      <Flex sx={uploadButtonStyles} {...getUploadButtonProps()}>
        <input {...getUploadInputProps()} />
        {uploadElement}
      </Flex>
    );
  }
);

UploadButton.displayName = 'UploadButton';

import type { ChakraProps, FlexProps, SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, Icon, Image } from '@invoke-ai/ui-library';
import { IAILoadingImageFallback, IAINoContentFallback } from 'common/components/IAIImageFallback';
import ImageMetadataOverlay from 'common/components/ImageMetadataOverlay';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import type { TypesafeDraggableData, TypesafeDroppableData } from 'features/dnd/types';
import ImageContextMenu from 'features/gallery/components/ImageContextMenu/ImageContextMenu';
import type { MouseEvent, ReactElement, ReactNode, SyntheticEvent } from 'react';
import { memo, useCallback, useMemo, useState } from 'react';
import { PiImageBold, PiUploadSimpleBold } from 'react-icons/pi';
import type { ImageDTO, PostUploadAction } from 'services/api/types';

import IAIDraggable from './IAIDraggable';
import IAIDroppable from './IAIDroppable';
import SelectionOverlay from './SelectionOverlay';

const defaultUploadElement = <Icon as={PiUploadSimpleBold} boxSize={16} />;

const defaultNoContentFallback = <IAINoContentFallback icon={PiImageBold} />;

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
    thumbnail = false,
    noContentFallback = defaultNoContentFallback,
    uploadElement = defaultUploadElement,
    useThumbailFallback,
    withHoverOverlay = false,
    children,
    onMouseOver,
    onMouseOut,
    dataTestId,
  } = props;

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
    <ImageContextMenu imageDTO={imageDTO}>
      {(ref) => (
        <Flex
          ref={ref}
          onMouseOver={handleMouseOver}
          onMouseOut={handleMouseOut}
          width="full"
          height="full"
          alignItems="center"
          justifyContent="center"
          position="relative"
          minW={minSize ? minSize : undefined}
          minH={minSize ? minSize : undefined}
          userSelect="none"
          cursor={isDragDisabled || !imageDTO ? 'default' : 'pointer'}
        >
          {imageDTO && (
            <Flex
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
              <SelectionOverlay isSelected={isSelected} isHovered={withHoverOverlay ? isHovered : false} />
            </Flex>
          )}
          {!imageDTO && !isUploadDisabled && (
            <>
              <Flex sx={uploadButtonStyles} {...getUploadButtonProps()}>
                <input {...getUploadInputProps()} />
                {uploadElement}
              </Flex>
            </>
          )}
          {!imageDTO && isUploadDisabled && noContentFallback}
          {imageDTO && !isDragDisabled && (
            <IAIDraggable data={draggableData} disabled={isDragDisabled || !imageDTO} onClick={onClick} />
          )}
          {children}
          {!isDropDisabled && <IAIDroppable data={droppableData} disabled={isDropDisabled} dropLabel={dropLabel} />}
        </Flex>
      )}
    </ImageContextMenu>
  );
};

export default memo(IAIDndImage);

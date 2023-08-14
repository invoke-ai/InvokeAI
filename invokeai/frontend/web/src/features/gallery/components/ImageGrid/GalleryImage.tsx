import { Box, Flex, useColorModeValue } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIDndImage from 'common/components/IAIDndImage';
import IAIFillSkeleton from 'common/components/IAIFillSkeleton';
import { imagesToDeleteSelected } from 'features/deleteImageModal/store/slice';
import {
  ImageDTOsDraggableData,
  ImageDraggableData,
  TypesafeDraggableData,
} from 'features/dnd/types';
import { useMultiselect } from 'features/gallery/hooks/useMultiselect.ts';
import { MouseEvent, memo, useCallback, useMemo, useState } from 'react';
import { FaTrash } from 'react-icons/fa';
import { MdStar, MdStarBorder } from 'react-icons/md';
import {
  useChangeImagePinnedMutation,
  useGetImageDTOQuery,
} from 'services/api/endpoints/images';
import IAIDndImageIcon from '../../../../common/components/IAIDndImageIcon';

interface HoverableImageProps {
  imageName: string;
}

const GalleryImage = (props: HoverableImageProps) => {
  const dispatch = useAppDispatch();
  const { imageName } = props;
  const { currentData: imageDTO } = useGetImageDTOQuery(imageName);
  const shouldShowDeleteButton = useAppSelector(
    (state) => state.gallery.shouldShowDeleteButton
  );

  const { handleClick, isSelected, selection, selectionCount } =
    useMultiselect(imageDTO);

  const handleDelete = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      if (!imageDTO) {
        return;
      }
      dispatch(imagesToDeleteSelected([imageDTO]));
    },
    [dispatch, imageDTO]
  );

  const draggableData = useMemo<TypesafeDraggableData | undefined>(() => {
    if (selectionCount > 1) {
      const data: ImageDTOsDraggableData = {
        id: 'gallery-image',
        payloadType: 'IMAGE_DTOS',
        payload: { imageDTOs: selection },
      };
      return data;
    }

    if (imageDTO) {
      const data: ImageDraggableData = {
        id: 'gallery-image',
        payloadType: 'IMAGE_DTO',
        payload: { imageDTO },
      };
      return data;
    }
  }, [imageDTO, selection, selectionCount]);

  const [togglePin] = useChangeImagePinnedMutation();

  const togglePinnedState = useCallback(() => {
    if (imageDTO) {
      togglePin({ imageDTO, pinned: !imageDTO.pinned });
    }
  }, [togglePin, imageDTO]);

  const [isHovered, setIsHovered] = useState(false);

  const pinIcon = useMemo(() => {
    if (imageDTO?.pinned) return <MdStar size="20" />;
    if (!imageDTO?.pinned && isHovered) return <MdStarBorder size="20" />;
  }, [imageDTO?.pinned, isHovered]);

  const resetIconShadow = useColorModeValue(
    `drop-shadow(0px 0px 0.1rem var(--invokeai-colors-base-600))`,
    `drop-shadow(0px 0px 0.1rem var(--invokeai-colors-base-800))`
  );

  const iconButtonStyles = {
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
  };

  if (!imageDTO) {
    return <IAIFillSkeleton />;
  }

  return (
    <Box sx={{ w: 'full', h: 'full', touchAction: 'none' }}>
      <Flex
        userSelect="none"
        sx={{
          position: 'relative',
          justifyContent: 'center',
          alignItems: 'center',
          aspectRatio: '1/1',
        }}
      >
        <IAIDndImage
          onClick={handleClick}
          imageDTO={imageDTO}
          draggableData={draggableData}
          isSelected={isSelected}
          minSize={0}
          imageSx={{ w: 'full', h: 'full' }}
          isDropDisabled={true}
          isUploadDisabled={true}
          thumbnail={true}
          withHoverOverlay
          onMouseOver={() => setIsHovered(true)}
          onMouseOut={() => setIsHovered(false)}
        >
          <>
            <IAIDndImageIcon
              onClick={togglePinnedState}
              icon={pinIcon}
              tooltip={imageDTO.pinned ? 'Unstar' : 'Star'}
            />

            {isHovered && shouldShowDeleteButton && (
              <IAIDndImageIcon
                onClick={handleDelete}
                icon={<FaTrash />}
                tooltip={'Delete'}
                styleOverrides={{
                  bottom: 1,
                  top: 'auto',
                }}
              />
            )}
          </>
        </IAIDndImage>
      </Flex>
    </Box>
  );
};

export default memo(GalleryImage);

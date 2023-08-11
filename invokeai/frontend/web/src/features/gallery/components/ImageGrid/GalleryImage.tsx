import { Box, Flex } from '@chakra-ui/react';
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
import { MouseEvent, memo, useCallback, useMemo } from 'react';
import { BsBookmarkStar, BsFillBookmarkStarFill } from 'react-icons/bs';
import {
  useChangeImagePinnedMutation,
  useGetImageDTOQuery,
} from 'services/api/endpoints/images';

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
          onClickReset={togglePinnedState}
          imageSx={{ w: 'full', h: 'full' }}
          isDropDisabled={true}
          isUploadDisabled={true}
          thumbnail={true}
          withHoverOverlay
          resetIcon={
            imageDTO.pinned ? <BsFillBookmarkStarFill /> : <BsBookmarkStar />
          }
          resetTooltip="Pin image"
          withResetIcon={true}
        />
      </Flex>
    </Box>
  );
};

export default memo(GalleryImage);

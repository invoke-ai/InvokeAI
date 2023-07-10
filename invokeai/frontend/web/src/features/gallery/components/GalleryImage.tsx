import { Box } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { TypesafeDraggableData } from 'app/components/ImageDnd/typesafeDnd';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIDndImage from 'common/components/IAIDndImage';
import { imageToDeleteSelected } from 'features/imageDeletion/store/imageDeletionSlice';
import { MouseEvent, memo, useCallback, useMemo } from 'react';
import { ImageDTO } from 'services/api/types';
import { imageSelected } from '../store/gallerySlice';
import ImageContextMenu from './ImageContextMenu';

export const makeSelector = (image_name: string) =>
  createSelector(
    [stateSelector],
    ({ gallery }) => ({
      isSelected: gallery.selection.includes(image_name),
      selectionCount: gallery.selection.length,
    }),
    defaultSelectorOptions
  );

interface HoverableImageProps {
  imageDTO: ImageDTO;
}

const GalleryImage = (props: HoverableImageProps) => {
  const dispatch = useAppDispatch();

  const { imageDTO } = props;
  const { image_name } = imageDTO;

  const localSelector = useMemo(() => makeSelector(image_name), [image_name]);

  const { isSelected, selectionCount } = useAppSelector(localSelector);

  const handleClick = useCallback(
    (e: MouseEvent<HTMLDivElement>) => {
      // multiselect disabled for now
      // if (e.shiftKey) {
      //   dispatch(imageRangeEndSelected(props.imageDTO.image_name));
      // } else if (e.ctrlKey || e.metaKey) {
      //   dispatch(imageSelectionToggled(props.imageDTO.image_name));
      // } else {
      //   dispatch(imageSelected(props.imageDTO.image_name));
      // }
      dispatch(imageSelected(props.imageDTO.image_name));
    },
    [dispatch, props.imageDTO.image_name]
  );

  const handleDelete = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      if (!imageDTO) {
        return;
      }
      dispatch(imageToDeleteSelected(imageDTO));
    },
    [dispatch, imageDTO]
  );

  const draggableData = useMemo<TypesafeDraggableData | undefined>(() => {
    if (selectionCount > 1) {
      return {
        id: 'gallery-image',
        payloadType: 'GALLERY_SELECTION',
      };
    }

    if (imageDTO) {
      return {
        id: 'gallery-image',
        payloadType: 'IMAGE_DTO',
        payload: { imageDTO },
      };
    }
  }, [imageDTO, selectionCount]);

  return (
    <Box sx={{ w: 'full', h: 'full', touchAction: 'none' }}>
      <ImageContextMenu image={imageDTO}>
        {(ref) => (
          <Box
            position="relative"
            key={image_name}
            userSelect="none"
            ref={ref}
            sx={{
              display: 'flex',
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
              onClickReset={handleDelete}
              imageSx={{ w: 'full', h: 'full' }}
              isDropDisabled={true}
              isUploadDisabled={true}
              thumbnail={true}
              // resetIcon={<FaTrash />}
              // resetTooltip="Delete image"
              // withResetIcon // removed bc it's too easy to accidentally delete images
            />
          </Box>
        )}
      </ImageContextMenu>
    </Box>
  );
};

export default memo(GalleryImage);

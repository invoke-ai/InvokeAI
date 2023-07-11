import { Box } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { TypesafeDraggableData } from 'app/components/ImageDnd/typesafeDnd';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIDndImage from 'common/components/IAIDndImage';
import { imageToDeleteSelected } from 'features/imageDeletion/store/imageDeletionSlice';
import { MouseEvent, memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaTrash } from 'react-icons/fa';
import { ImageDTO } from 'services/api/types';
import {
  imageRangeEndSelected,
  imageSelected,
  imageSelectionToggled,
} from '../store/gallerySlice';
import ImageContextMenu from './ImageContextMenu';

export const makeSelector = (image_name: string) =>
  createSelector(
    [stateSelector],
    ({ gallery }) => {
      const isSelected = gallery.selection.includes(image_name);
      const selectionCount = gallery.selection.length;

      return {
        isSelected,
        selectionCount,
      };
    },
    defaultSelectorOptions
  );

interface HoverableImageProps {
  imageDTO: ImageDTO;
}

/**
 * Gallery image component with delete/use all/use seed buttons on hover.
 */
const GalleryImage = (props: HoverableImageProps) => {
  const { imageDTO } = props;
  const { image_url, thumbnail_url, image_name } = imageDTO;

  const localSelector = useMemo(() => makeSelector(image_name), [image_name]);

  const { isSelected, selectionCount } = useAppSelector(localSelector);

  const dispatch = useAppDispatch();

  const { t } = useTranslation();

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
              resetIcon={<FaTrash />}
              resetTooltip="Delete image"
              imageSx={{ w: 'full', h: 'full' }}
              // withResetIcon // removed bc it's too easy to accidentally delete images
              isDropDisabled={true}
              isUploadDisabled={true}
              thumbnail={true}
            />
          </Box>
        )}
      </ImageContextMenu>
    </Box>
  );
};

export default memo(GalleryImage);

import { Box } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { MouseEvent, memo, useCallback, useMemo } from 'react';
import { FaTrash } from 'react-icons/fa';
import { useTranslation } from 'react-i18next';
import { createSelector } from '@reduxjs/toolkit';
import { ImageDTO } from 'services/api/types';
import { TypesafeDraggableData } from 'app/components/ImageDnd/typesafeDnd';
import { stateSelector } from 'app/store/store';
import ImageContextMenu from './ImageContextMenu';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIDndImage from 'common/components/IAIDndImage';
import {
  imageRangeEndSelected,
  imageSelected,
  imageSelectionToggled,
} from '../store/gallerySlice';
import { imageToDeleteSelected } from 'features/imageDeletion/store/imageDeletionSlice';

export const selector = createSelector(
  [stateSelector, (state, { image_name }: ImageDTO) => image_name],
  ({ gallery }, image_name) => {
    const isSelected = gallery.selection.includes(image_name);
    const selection = gallery.selection;
    return {
      isSelected,
      selection,
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
  const { isSelected, selection } = useAppSelector((state) =>
    selector(state, props.imageDTO)
  );

  const { imageDTO } = props;
  const { image_url, thumbnail_url, image_name } = imageDTO;

  const dispatch = useAppDispatch();

  const { t } = useTranslation();

  const handleClick = useCallback(
    (e: MouseEvent<HTMLDivElement>) => {
      if (e.shiftKey) {
        dispatch(imageRangeEndSelected(props.imageDTO.image_name));
      } else if (e.ctrlKey || e.metaKey) {
        dispatch(imageSelectionToggled(props.imageDTO.image_name));
      } else {
        dispatch(imageSelected(props.imageDTO.image_name));
      }
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
    if (selection.length > 1) {
      return {
        id: 'gallery-image',
        payloadType: 'IMAGE_NAMES',
        payload: { imageNames: selection },
      };
    }

    if (imageDTO) {
      return {
        id: 'gallery-image',
        payloadType: 'IMAGE_DTO',
        payload: { imageDTO },
      };
    }
  }, [imageDTO, selection]);

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
              withResetIcon
              isDropDisabled={true}
              isUploadDisabled={true}
            />
          </Box>
        )}
      </ImageContextMenu>
    </Box>
  );
};

export default memo(GalleryImage);

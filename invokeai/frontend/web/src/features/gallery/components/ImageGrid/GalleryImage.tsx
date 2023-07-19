import { Box } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { TypesafeDraggableData } from 'app/components/ImageDnd/typesafeDnd';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIDndImage from 'common/components/IAIDndImage';
import IAIFillSkeleton from 'common/components/IAIFillSkeleton';
import ImageContextMenu from 'features/gallery/components/ImageContextMenu/ImageContextMenu';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import { imageToDeleteSelected } from 'features/imageDeletion/store/imageDeletionSlice';
import { MouseEvent, memo, useCallback, useMemo } from 'react';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';

export const makeSelector = (image_name: string) =>
  createSelector(
    [stateSelector],
    ({ gallery }) => ({
      isSelected: gallery.selection.includes(image_name),
      selectionCount: gallery.selection.length,
      selection: gallery.selection,
    }),
    defaultSelectorOptions
  );

interface HoverableImageProps {
  imageName: string;
}

const GalleryImage = (props: HoverableImageProps) => {
  const dispatch = useAppDispatch();
  const { imageName } = props;
  const { currentData: imageDTO } = useGetImageDTOQuery(imageName);
  const localSelector = useMemo(() => makeSelector(imageName), [imageName]);

  const { isSelected, selectionCount, selection } =
    useAppSelector(localSelector);

  const handleClick = useCallback(
    (e: MouseEvent<HTMLDivElement>) => {
      // disable multiselect for now
      // if (e.shiftKey) {
      //   dispatch(imageRangeEndSelected(imageName));
      // } else if (e.ctrlKey || e.metaKey) {
      //   dispatch(imageSelectionToggled(imageName));
      // } else {
      //   dispatch(imageSelected(imageName));
      // }
      dispatch(imageSelected(imageName));
    },
    [dispatch, imageName]
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
        payloadType: 'IMAGE_NAMES',
        payload: { image_names: selection },
      };
    }

    if (imageDTO) {
      return {
        id: 'gallery-image',
        payloadType: 'IMAGE_DTO',
        payload: { imageDTO },
      };
    }
  }, [imageDTO, selection, selectionCount]);

  if (!imageDTO) {
    return <IAIFillSkeleton />;
  }

  return (
    <Box sx={{ w: 'full', h: 'full', touchAction: 'none' }}>
      <ImageContextMenu imageDTO={imageDTO}>
        {(ref) => (
          <Box
            position="relative"
            key={imageName}
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

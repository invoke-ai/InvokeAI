import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { draggable, monitorForElements } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Flex, Image } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { galleryImageClicked } from 'app/store/middleware/listenerMiddleware/listeners/galleryImageClicked';
import { useAppStore } from 'app/store/nanostores/store';
import { useAppSelector } from 'app/store/storeHooks';
import { multipleImageDndSource, singleImageDndSource } from 'features/dnd/dnd';
import type { DndDragPreviewMultipleImageState } from 'features/dnd/DndDragPreviewMultipleImage';
import { createMultipleImageDragPreview, setMultipleImageDragPreview } from 'features/dnd/DndDragPreviewMultipleImage';
import type { DndDragPreviewSingleImageState } from 'features/dnd/DndDragPreviewSingleImage';
import { createSingleImageDragPreview, setSingleImageDragPreview } from 'features/dnd/DndDragPreviewSingleImage';
import { firefoxDndFix } from 'features/dnd/util';
import { useImageContextMenu } from 'features/gallery/components/ImageContextMenu/ImageContextMenu';
import { GalleryImageHoverIcons } from 'features/gallery/components/ImageGrid/GalleryImageHoverIcons';
import { getGalleryImageDataTestId } from 'features/gallery/components/ImageGrid/getGalleryImageDataTestId';
import { SizedSkeletonLoader } from 'features/gallery/components/ImageGrid/SizedSkeletonLoader';
import { $imageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { imageToCompareChanged, selectGallerySlice } from 'features/gallery/store/gallerySlice';
import type { MouseEventHandler } from 'react';
import { memo, useCallback, useEffect, useId, useMemo, useState } from 'react';
import type { ImageDTO } from 'services/api/types';

// This class name is used to calculate the number of images that fit in the gallery
export const GALLERY_IMAGE_CONTAINER_CLASS_NAME = 'gallery-image-container';

const galleryImageContainerSX = {
  containerType: 'inline-size',
  w: 'full',
  h: 'full',
  '.gallery-image-size-badge': {
    '@container (max-width: 80px)': {
      '&': { display: 'none' },
    },
  },
  '&[data-is-dragging=true]': {
    opacity: 0.3,
  },
  '.gallery-image': {
    touchAction: 'none',
    userSelect: 'none',
    webkitUserSelect: 'none',
    position: 'relative',
    justifyContent: 'center',
    alignItems: 'center',
    aspectRatio: '1/1',
    '::before': {
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
    '&[data-selected=true]::before': {
      boxShadow:
        'inset 0px 0px 0px 3px var(--invoke-colors-invokeBlue-500), inset 0px 0px 0px 4px var(--invoke-colors-invokeBlue-800)',
    },
    '&[data-selected-for-compare=true]::before': {
      boxShadow:
        'inset 0px 0px 0px 3px var(--invoke-colors-invokeGreen-300), inset 0px 0px 0px 4px var(--invoke-colors-invokeGreen-800)',
    },
    '&:hover::before': {
      boxShadow:
        'inset 0px 0px 0px 1px var(--invoke-colors-invokeBlue-300), inset 0px 0px 0px 2px var(--invoke-colors-invokeBlue-800)',
    },
    '&:hover[data-selected=true]::before': {
      boxShadow:
        'inset 0px 0px 0px 3px var(--invoke-colors-invokeBlue-400), inset 0px 0px 0px 4px var(--invoke-colors-invokeBlue-800)',
    },
    '&:hover[data-selected-for-compare=true]::before': {
      boxShadow:
        'inset 0px 0px 0px 3px var(--invoke-colors-invokeGreen-200), inset 0px 0px 0px 4px var(--invoke-colors-invokeGreen-800)',
    },
  },
} satisfies SystemStyleObject;

interface Props {
  imageDTO: ImageDTO;
}

export const GalleryImage = memo(({ imageDTO }: Props) => {
  const store = useAppStore();
  const [isDragging, setIsDragging] = useState(false);
  const [dragPreviewState, setDragPreviewState] = useState<
    DndDragPreviewSingleImageState | DndDragPreviewMultipleImageState | null
  >(null);
  const [element, ref] = useState<HTMLImageElement | null>(null);
  const dndId = useId();
  const selectIsSelectedForCompare = useMemo(
    () => createSelector(selectGallerySlice, (gallery) => gallery.imageToCompare?.image_name === imageDTO.image_name),
    [imageDTO.image_name]
  );
  const isSelectedForCompare = useAppSelector(selectIsSelectedForCompare);
  const selectIsSelected = useMemo(
    () =>
      createSelector(selectGallerySlice, (gallery) => {
        for (const selectedImage of gallery.selection) {
          if (selectedImage.image_name === imageDTO.image_name) {
            return true;
          }
        }
        return false;
      }),
    [imageDTO.image_name]
  );
  const isSelected = useAppSelector(selectIsSelected);

  useEffect(() => {
    if (!element) {
      return;
    }
    return combine(
      firefoxDndFix(element),
      draggable({
        element,
        getInitialData: () => {
          const { gallery } = store.getState();
          // When we have multiple images selected, and the dragged image is part of the selection, initiate a
          // multi-image drag.
          if (
            gallery.selection.length > 1 &&
            gallery.selection.find(({ image_name }) => image_name === imageDTO.image_name) !== undefined
          ) {
            return multipleImageDndSource.getData({
              imageDTOs: gallery.selection,
              boardId: gallery.selectedBoardId,
            });
          }

          // Otherwise, initiate a single-image drag
          return singleImageDndSource.getData({ imageDTO }, imageDTO.image_name);
        },
        // This is a "local" drag start event, meaning that it is only called when this specific image is dragged.
        onDragStart: ({ source }) => {
          // When we start dragging a single image, set the dragging state to true. This is only called when this
          // specific image is dragged.
          if (singleImageDndSource.typeGuard(source.data)) {
            setIsDragging(true);
            return;
          }
        },
        onGenerateDragPreview: (args) => {
          if (multipleImageDndSource.typeGuard(args.source.data)) {
            setMultipleImageDragPreview({
              multipleImageDndData: args.source.data,
              onGenerateDragPreviewArgs: args,
              setDragPreviewState,
            });
          } else if (singleImageDndSource.typeGuard(args.source.data)) {
            setSingleImageDragPreview({
              singleImageDndData: args.source.data,
              onGenerateDragPreviewArgs: args,
              setDragPreviewState,
            });
          }
        },
      }),
      monitorForElements({
        // This is a "global" drag start event, meaning that it is called for all drag events.
        onDragStart: ({ source }) => {
          // When we start dragging multiple images, set the dragging state to true if the dragged image is part of the
          // selection. This is called for all drag events.
          if (multipleImageDndSource.typeGuard(source.data) && source.data.payload.imageDTOs.includes(imageDTO)) {
            setIsDragging(true);
          }
        },
        onDrop: () => {
          // Always set the dragging state to false when a drop event occurs.
          setIsDragging(false);
        },
      })
    );
  }, [imageDTO, element, store, dndId]);

  const [isHovered, setIsHovered] = useState(false);

  const onMouseOver = useCallback(() => {
    setIsHovered(true);
  }, []);

  const onMouseOut = useCallback(() => {
    setIsHovered(false);
  }, []);

  const onClick = useCallback<MouseEventHandler<HTMLDivElement>>(
    (e) => {
      store.dispatch(
        galleryImageClicked({
          imageDTO,
          shiftKey: e.shiftKey,
          ctrlKey: e.ctrlKey,
          metaKey: e.metaKey,
          altKey: e.altKey,
        })
      );
    },
    [imageDTO, store]
  );

  const onDoubleClick = useCallback<MouseEventHandler<HTMLDivElement>>(() => {
    // Use the atom here directly instead of the `useImageViewer` to avoid re-rendering the gallery when the viewer
    // opened state changes.
    $imageViewer.set(true);
    store.dispatch(imageToCompareChanged(null));
  }, [store]);

  const dataTestId = useMemo(() => getGalleryImageDataTestId(imageDTO.image_name), [imageDTO.image_name]);

  useImageContextMenu(imageDTO, element);

  return (
    <>
      <Box
        className={GALLERY_IMAGE_CONTAINER_CLASS_NAME}
        sx={galleryImageContainerSX}
        data-testid={dataTestId}
        data-is-dragging={isDragging}
      >
        <Flex
          role="button"
          className="gallery-image"
          onMouseOver={onMouseOver}
          onMouseOut={onMouseOut}
          onClick={onClick}
          onDoubleClick={onDoubleClick}
          data-selected={isSelected}
          data-selected-for-compare={isSelectedForCompare}
        >
          <Image
            ref={ref}
            src={imageDTO.thumbnail_url}
            fallback={<SizedSkeletonLoader width={imageDTO.width} height={imageDTO.height} />}
            w={imageDTO.width}
            objectFit="contain"
            maxW="full"
            maxH="full"
            borderRadius="base"
          />
          <GalleryImageHoverIcons imageDTO={imageDTO} isHovered={isHovered} />
        </Flex>
      </Box>
      {dragPreviewState?.type === 'multiple-image' ? createMultipleImageDragPreview(dragPreviewState) : null}
      {dragPreviewState?.type === 'single-image' ? createSingleImageDragPreview(dragPreviewState) : null}
    </>
  );
});

GalleryImage.displayName = 'GalleryImage';

import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { draggable, monitorForElements } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import type { FlexProps, SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, Icon, Image } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import type { AppDispatch, AppGetState } from 'app/store/store';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { uniq } from 'es-toolkit';
import { multipleImageDndSource, singleImageDndSource } from 'features/dnd/dnd';
import type { DndDragPreviewMultipleImageState } from 'features/dnd/DndDragPreviewMultipleImage';
import { createMultipleImageDragPreview, setMultipleImageDragPreview } from 'features/dnd/DndDragPreviewMultipleImage';
import type { DndDragPreviewSingleImageState } from 'features/dnd/DndDragPreviewSingleImage';
import { createSingleImageDragPreview, setSingleImageDragPreview } from 'features/dnd/DndDragPreviewSingleImage';
import { firefoxDndFix } from 'features/dnd/util';
import { useImageContextMenu } from 'features/gallery/components/ImageContextMenu/ImageContextMenu';
import { GalleryImageHoverIcons } from 'features/gallery/components/ImageGrid/GalleryImageHoverIcons';
import {
  selectGetImageNamesQueryArgs,
  selectSelectedBoardId,
  selectSelection,
} from 'features/gallery/store/gallerySelectors';
import { imageToCompareChanged, selectGallerySlice, selectionChanged } from 'features/gallery/store/gallerySlice';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { VIEWER_PANEL_ID } from 'features/ui/layouts/shared';
import type { MouseEvent, MouseEventHandler } from 'react';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { PiImageBold } from 'react-icons/pi';
import { imagesApi } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';

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
} satisfies SystemStyleObject;

interface Props {
  imageDTO: ImageDTO;
}

const buildOnClick =
  (imageName: string, dispatch: AppDispatch, getState: AppGetState) => (e: MouseEvent<HTMLDivElement>) => {
    const { shiftKey, ctrlKey, metaKey, altKey } = e;
    const state = getState();
    const queryArgs = selectGetImageNamesQueryArgs(state);
    const imageNames = imagesApi.endpoints.getImageNames.select(queryArgs)(state).data?.image_names ?? [];

    // If we don't have the image names cached, we can't perform selection operations
    // This can happen if the user clicks on an image before the names are loaded
    if (imageNames.length === 0) {
      // For basic click without modifiers, we can still set selection
      if (!shiftKey && !ctrlKey && !metaKey && !altKey) {
        dispatch(selectionChanged([imageName]));
      }
      return;
    }

    const selection = state.gallery.selection;

    if (altKey) {
      if (state.gallery.imageToCompare === imageName) {
        dispatch(imageToCompareChanged(null));
      } else {
        dispatch(imageToCompareChanged(imageName));
      }
    } else if (shiftKey) {
      const rangeEndImageName = imageName;
      const lastSelectedImage = selection.at(-1);
      const lastClickedIndex = imageNames.findIndex((name) => name === lastSelectedImage);
      const currentClickedIndex = imageNames.findIndex((name) => name === rangeEndImageName);
      if (lastClickedIndex > -1 && currentClickedIndex > -1) {
        // We have a valid range!
        const start = Math.min(lastClickedIndex, currentClickedIndex);
        const end = Math.max(lastClickedIndex, currentClickedIndex);
        const imagesToSelect = imageNames.slice(start, end + 1);
        if (currentClickedIndex < lastClickedIndex) {
          imagesToSelect.reverse();
        }
        dispatch(selectionChanged(uniq(selection.concat(imagesToSelect))));
      }
    } else if (ctrlKey || metaKey) {
      if (selection.some((n) => n === imageName) && selection.length > 1) {
        dispatch(selectionChanged(uniq(selection.filter((n) => n !== imageName))));
      } else {
        dispatch(selectionChanged(uniq(selection.concat(imageName))));
      }
    } else {
      dispatch(selectionChanged([imageName]));
    }
  };

export const GalleryImage = memo(({ imageDTO }: Props) => {
  const store = useAppStore();
  const [isDragging, setIsDragging] = useState(false);
  const [dragPreviewState, setDragPreviewState] = useState<
    DndDragPreviewSingleImageState | DndDragPreviewMultipleImageState | null
  >(null);
  const ref = useRef<HTMLDivElement>(null);
  const selectIsSelectedForCompare = useMemo(
    () => createSelector(selectGallerySlice, (gallery) => gallery.imageToCompare === imageDTO.image_name),
    [imageDTO.image_name]
  );
  const isSelectedForCompare = useAppSelector(selectIsSelectedForCompare);
  const selectIsSelected = useMemo(
    () => createSelector(selectGallerySlice, (gallery) => gallery.selection.includes(imageDTO.image_name)),
    [imageDTO.image_name]
  );
  const isSelected = useAppSelector(selectIsSelected);

  useEffect(() => {
    const element = ref.current;
    if (!element) {
      return;
    }
    return combine(
      firefoxDndFix(element),
      draggable({
        element,
        getInitialData: () => {
          const selection = selectSelection(store.getState());
          const boardId = selectSelectedBoardId(store.getState());
          // When we have multiple images selected, and the dragged image is part of the selection, initiate a
          // multi-image drag.
          if (selection.length > 1 && selection.includes(imageDTO.image_name)) {
            return multipleImageDndSource.getData({
              image_names: selection,
              board_id: boardId,
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
          if (
            multipleImageDndSource.typeGuard(source.data) &&
            source.data.payload.image_names.includes(imageDTO.image_name)
          ) {
            setIsDragging(true);
          }
        },
        onDrop: () => {
          // Always set the dragging state to false when a drop event occurs.
          setIsDragging(false);
        },
      })
    );
  }, [imageDTO, store]);

  const [isHovered, setIsHovered] = useState(false);

  const onMouseOver = useCallback(() => {
    setIsHovered(true);
  }, []);

  const onMouseOut = useCallback(() => {
    setIsHovered(false);
  }, []);

  const onClick = useMemo(() => buildOnClick(imageDTO.image_name, store.dispatch, store.getState), [imageDTO, store]);

  const onDoubleClick = useCallback<MouseEventHandler<HTMLDivElement>>(() => {
    store.dispatch(imageToCompareChanged(null));
    navigationApi.focusPanelInActiveTab(VIEWER_PANEL_ID);
  }, [store]);

  useImageContextMenu(imageDTO, ref);

  return (
    <>
      <Flex
        ref={ref}
        sx={galleryImageContainerSX}
        data-is-dragging={isDragging}
        data-image-name={imageDTO.image_name}
        role="button"
        onMouseOver={onMouseOver}
        onMouseOut={onMouseOut}
        onClick={onClick}
        onDoubleClick={onDoubleClick}
        data-selected={isSelected}
        data-selected-for-compare={isSelectedForCompare}
      >
        <Image
          pointerEvents="none"
          src={imageDTO.thumbnail_url}
          w={imageDTO.width}
          fallback={<GalleryImagePlaceholder />}
          objectFit="contain"
          maxW="full"
          maxH="full"
          borderRadius="base"
        />
        <GalleryImageHoverIcons imageDTO={imageDTO} isHovered={isHovered} />
      </Flex>
      {dragPreviewState?.type === 'multiple-image' ? createMultipleImageDragPreview(dragPreviewState) : null}
      {dragPreviewState?.type === 'single-image' ? createSingleImageDragPreview(dragPreviewState) : null}
    </>
  );
});

GalleryImage.displayName = 'GalleryImage';

export const GalleryImagePlaceholder = memo((props: FlexProps) => (
  <Flex w="full" h="full" bg="base.850" borderRadius="base" alignItems="center" justifyContent="center" {...props}>
    <Icon as={PiImageBold} boxSize={16} color="base.800" />
  </Flex>
));

GalleryImagePlaceholder.displayName = 'GalleryImagePlaceholder';

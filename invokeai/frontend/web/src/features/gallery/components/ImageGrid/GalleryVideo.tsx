import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { draggable, monitorForElements } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import type { FlexProps } from '@invoke-ai/ui-library';
import { Flex, Icon, Image } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import type { AppDispatch, AppGetState } from 'app/store/store';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { uniq } from 'es-toolkit';
import { multipleVideoDndSource, singleVideoDndSource } from 'features/dnd/dnd';
import type { DndDragPreviewMultipleVideoState } from 'features/dnd/DndDragPreviewMultipleVideo';
import { createMultipleVideoDragPreview, setMultipleVideoDragPreview } from 'features/dnd/DndDragPreviewMultipleVideo';
import type { DndDragPreviewSingleVideoState } from 'features/dnd/DndDragPreviewSingleVideo';
import { createSingleVideoDragPreview, setSingleVideoDragPreview } from 'features/dnd/DndDragPreviewSingleVideo';
import { firefoxDndFix } from 'features/dnd/util';
import { useVideoContextMenu } from 'features/gallery/components/ContextMenu/VideoContextMenu';
import { selectGetImageNamesQueryArgs, selectSelectedBoardId, selectSelection } from 'features/gallery/store/gallerySelectors';
import { imageToCompareChanged, selectGallerySlice, selectionChanged } from 'features/gallery/store/gallerySlice';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { VIEWER_PANEL_ID } from 'features/ui/layouts/shared';
import type { MouseEvent, MouseEventHandler } from 'react';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { PiVideoBold } from 'react-icons/pi';
import { imagesApi } from 'services/api/endpoints/images';
import type { VideoDTO } from 'services/api/types';

import { GalleryItemHoverIcons } from './GalleryItemHoverIcons';
import { galleryItemContainerSX } from './galleryItemContainerSX';

interface Props {
  videoDTO: VideoDTO;
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

export const GalleryVideo = memo(({ videoDTO }: Props) => {
  const store = useAppStore();
  const [isDragging, setIsDragging] = useState(false);
  const [dragPreviewState, setDragPreviewState] = useState<
    DndDragPreviewSingleVideoState | DndDragPreviewMultipleVideoState | null
  >(null);
  const ref = useRef<HTMLDivElement>(null);
  const selectIsSelected = useMemo(
    () => createSelector(selectGallerySlice, (gallery) => gallery.selection.includes(videoDTO.video_id)),
    [videoDTO.video_id]
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
          if (selection.length > 1 && selection.includes(videoDTO.video_id)) {
            return multipleVideoDndSource.getData({
              ids: selection,
              board_id: boardId,
            });
          } // Otherwise, initiate a single-image drag

          return singleVideoDndSource.getData({ videoDTO }, videoDTO.video_id);
        },
        // This is a "local" drag start event, meaning that it is only called when this specific image is dragged.
        onDragStart: ({ source }) => {
          // When we start dragging a single image, set the dragging state to true. This is only called when this
          // specific image is dragged.
          if (singleVideoDndSource.typeGuard(source.data)) {
            setIsDragging(true);
            return;
          }
        },
        onGenerateDragPreview: (args) => {
          if (multipleVideoDndSource.typeGuard(args.source.data)) {
            setMultipleVideoDragPreview({
              multipleVideoDndData: args.source.data,
              onGenerateDragPreviewArgs: args,
              setDragPreviewState,
            });
          } else if (singleVideoDndSource.typeGuard(args.source.data)) {
            setSingleVideoDragPreview({
              singleVideoDndData: args.source.data,
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
          if (multipleVideoDndSource.typeGuard(source.data) && source.data.payload.ids.includes(videoDTO.video_id)) {
            setIsDragging(true);
          }
        },
        onDrop: () => {
          // Always set the dragging state to false when a drop event occurs.
          setIsDragging(false);
        },
      })
    );
  }, [videoDTO, store]);

  const [isHovered, setIsHovered] = useState(false);

  const onMouseOver = useCallback(() => {
    setIsHovered(true);
  }, []);

  const onMouseOut = useCallback(() => {
    setIsHovered(false);
  }, []);

  const onClick = useMemo(() => buildOnClick(videoDTO.video_id, store.dispatch, store.getState), [videoDTO, store]);

  const onDoubleClick = useCallback<MouseEventHandler<HTMLDivElement>>(() => {
    store.dispatch(imageToCompareChanged(null));
    navigationApi.focusPanelInActiveTab(VIEWER_PANEL_ID);
  }, [store]);

  useVideoContextMenu(videoDTO, ref);

  return (
    <>
      <Flex
        ref={ref}
        sx={galleryItemContainerSX}
        data-is-dragging={isDragging}
        data-item-id={videoDTO.video_id}
        role="button"
        onMouseOver={onMouseOver}
        onMouseOut={onMouseOut}
        onClick={onClick}
        onDoubleClick={onDoubleClick}
        data-selected={isSelected}
        data-selected-for-compare={false}
      >
        <Image
          pointerEvents="none"
          src={videoDTO.thumbnail_url}
          w={videoDTO.width}
          fallback={<GalleryVideoPlaceholder />}
          objectFit="contain"
          maxW="full"
          maxH="full"
          borderRadius="base"
        />
        <GalleryItemHoverIcons itemDTO={videoDTO} isHovered={isHovered} />
      </Flex>
      {dragPreviewState?.type === 'multiple-video' ? createMultipleVideoDragPreview(dragPreviewState) : null}
      {dragPreviewState?.type === 'single-video' ? createSingleVideoDragPreview(dragPreviewState) : null}
    </>
  );
});

GalleryVideo.displayName = 'GalleryVideo';

export const GalleryVideoPlaceholder = memo((props: FlexProps) => (
  <Flex w="full" h="full" bg="base.850" borderRadius="base" alignItems="center" justifyContent="center" {...props}>
    <Icon as={PiVideoBold} boxSize={16} color="base.800" />
  </Flex>
));

GalleryVideoPlaceholder.displayName = 'GalleryVideoPlaceholder';

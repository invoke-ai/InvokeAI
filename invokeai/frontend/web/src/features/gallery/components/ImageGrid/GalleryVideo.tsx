import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { draggable, monitorForElements } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { Flex, Image } from '@invoke-ai/ui-library';
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
import {
  selectGetVideoIdsQueryArgs,
  selectSelectedBoardId,
  selectSelection,
} from 'features/gallery/store/gallerySelectors';
import { imageToCompareChanged, selectGallerySlice, selectionChanged } from 'features/gallery/store/gallerySlice';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { VIEWER_PANEL_ID } from 'features/ui/layouts/shared';
import type { MouseEvent, MouseEventHandler } from 'react';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { videosApi } from 'services/api/endpoints/videos';
import type { VideoDTO } from 'services/api/types';

import { galleryItemContainerSX } from './galleryItemContainerSX';
import { GalleryItemHoverIcons } from './GalleryItemHoverIcons';
import { GalleryVideoPlaceholder } from './GalleryVideoPlaceholder';

interface Props {
  videoDTO: VideoDTO;
}

const buildOnClick =
  (videoId: string, dispatch: AppDispatch, getState: AppGetState) => (e: MouseEvent<HTMLDivElement>) => {
    const { shiftKey, ctrlKey, metaKey, altKey } = e;
    const state = getState();
    const queryArgs = selectGetVideoIdsQueryArgs(state);
    const videoIds = videosApi.endpoints.getVideoIds.select(queryArgs)(state).data?.video_ids ?? [];

    // If we don't have the video ids cached, we can't perform selection operations
    // This can happen if the user clicks on a video before the ids are loaded
    if (videoIds.length === 0) {
      // For basic click without modifiers, we can still set selection
      if (!shiftKey && !ctrlKey && !metaKey && !altKey) {
        dispatch(selectionChanged([{ type: 'video', id: videoId }]));
      }
      return;
    }

    const selection = state.gallery.selection;

    if (altKey) {
      if (state.gallery.imageToCompare === videoId) {
        dispatch(imageToCompareChanged(null));
      } else {
        dispatch(imageToCompareChanged(videoId));
      }
    } else if (shiftKey) {
      const rangeEndVideoId = videoId;
      const lastSelectedVideo = selection.at(-1)?.id;
      const lastClickedIndex = videoIds.findIndex((id) => id === lastSelectedVideo);
      const currentClickedIndex = videoIds.findIndex((id) => id === rangeEndVideoId);
      if (lastClickedIndex > -1 && currentClickedIndex > -1) {
        // We have a valid range!
        const start = Math.min(lastClickedIndex, currentClickedIndex);
        const end = Math.max(lastClickedIndex, currentClickedIndex);
        const videosToSelect = videoIds.slice(start, end + 1);
        dispatch(selectionChanged(uniq(selection.concat(videosToSelect.map((id) => ({ type: 'video', id }))))));
      }
    } else if (ctrlKey || metaKey) {
      if (selection.some((n) => n.id === videoId) && selection.length > 1) {
        dispatch(selectionChanged(uniq(selection.filter((n) => n.id !== videoId))));
      } else {
        dispatch(selectionChanged(uniq(selection.concat({ type: 'video', id: videoId }))));
      }
    } else {
      dispatch(selectionChanged([{ type: 'video', id: videoId }]));
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
    () => createSelector(selectGallerySlice, (gallery) => gallery.selection.some((s) => s.id === videoDTO.video_id)),
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
          if (selection.length > 1 && selection.some((s) => s.id === videoDTO.video_id)) {
            return multipleVideoDndSource.getData({
              video_ids: selection.map((s) => s.id),
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
          if (
            multipleVideoDndSource.typeGuard(source.data) &&
            source.data.payload.video_ids.includes(videoDTO.video_id)
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

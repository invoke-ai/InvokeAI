import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { draggable } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { Flex } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import type { AppDispatch, AppGetState } from 'app/store/store';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { uniq } from 'es-toolkit';
import { multipleVideoDndSource, singleVideoDndSource } from 'features/dnd/dnd';
import { firefoxDndFix } from 'features/dnd/util';
import { useVideoContextMenu } from 'features/gallery/components/ContextMenu/VideoContextMenu';
import {
  selectAlwaysShouldImageSizeBadge,
  selectSelectedBoardId,
  selectSelection,
} from 'features/gallery/store/gallerySelectors';
import { selectGallerySlice, selectionChanged } from 'features/gallery/store/gallerySlice';
import { selectCachedGalleryItemNames } from 'features/gallery/store/selectCachedGalleryItemNames';
import { isVideoName } from 'features/gallery/store/types';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { VIEWER_PANEL_ID } from 'features/ui/layouts/shared';
import type { MouseEvent, MouseEventHandler } from 'react';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { VideoDTO } from 'services/api/types';

import { galleryItemContainerSX } from './galleryItemContainerSX';
import { GalleryItemPlayBadge } from './GalleryItemPlayBadge';
import { GalleryItemSizeBadge } from './GalleryItemSizeBadge';
import { GalleryItemVideoStarIconButton } from './GalleryItemVideoStarIconButton';
import { GalleryVideoThumbnail } from './GalleryVideoThumbnail';

interface Props {
  videoDTO: VideoDTO;
}

/**
 * Click handler for selection. Mirrors the image grid's logic but reads the polymorphic
 * /gallery/items/names cache to know the full ordered list (since a shift-range across a
 * mixed image+video gallery has to include both kinds).
 *
 * Video items do not participate in alt-click comparison (comparison is image-only).
 */
const buildOnClick =
  (videoName: string, dispatch: AppDispatch, getState: AppGetState) => (e: MouseEvent<HTMLDivElement>) => {
    const { shiftKey, ctrlKey, metaKey, altKey } = e;
    // We need the same query args the gallery grid used to fetch its name list. The grid
    // calls `useGalleryItemNames` which forwards the args to the polymorphic gallery endpoint.
    // Pull the most recent cached entry to recover the ordering.
    const state = getState();
    const itemNames = selectCachedGalleryItemNames(state);

    if (itemNames.length === 0) {
      // Without an ordered list, only basic single-click selection is possible.
      if (!shiftKey && !ctrlKey && !metaKey && !altKey) {
        dispatch(selectionChanged([videoName]));
      }
      return;
    }

    const selection = state.gallery.selection;

    if (altKey) {
      // Alt-click is image-only (comparison view). Quietly treat as a normal click for videos.
      dispatch(selectionChanged([videoName]));
    } else if (shiftKey) {
      const lastSelectedItem = selection.at(-1);
      const lastClickedIndex = itemNames.findIndex((name) => name === lastSelectedItem);
      const currentClickedIndex = itemNames.findIndex((name) => name === videoName);
      if (lastClickedIndex > -1 && currentClickedIndex > -1) {
        const start = Math.min(lastClickedIndex, currentClickedIndex);
        const end = Math.max(lastClickedIndex, currentClickedIndex);
        const itemsToSelect = itemNames.slice(start, end + 1);
        if (currentClickedIndex < lastClickedIndex) {
          itemsToSelect.reverse();
        }
        dispatch(selectionChanged(uniq(selection.concat(itemsToSelect))));
      }
    } else if (ctrlKey || metaKey) {
      if (selection.some((n) => n === videoName) && selection.length > 1) {
        dispatch(selectionChanged(uniq(selection.filter((n) => n !== videoName))));
      } else {
        dispatch(selectionChanged(uniq(selection.concat(videoName))));
      }
    } else {
      dispatch(selectionChanged([videoName]));
    }
  };

export const GalleryVideoItem = memo(({ videoDTO }: Props) => {
  const store = useAppStore();
  const ref = useRef<HTMLDivElement>(null);
  const [isHovered, setIsHovered] = useState(false);
  const alwaysShowSizeBadge = useAppSelector(selectAlwaysShouldImageSizeBadge);

  const selectIsSelected = useMemo(
    () => createSelector(selectGallerySlice, (gallery) => gallery.selection.some((n) => n === videoDTO.video_name)),
    [videoDTO.video_name]
  );
  const isSelected = useAppSelector(selectIsSelected);

  const onMouseOver = useCallback(() => setIsHovered(true), []);
  const onMouseOut = useCallback(() => setIsHovered(false), []);

  const onClick = useMemo(() => buildOnClick(videoDTO.video_name, store.dispatch, store.getState), [videoDTO, store]);

  const onDoubleClick = useCallback<MouseEventHandler<HTMLDivElement>>(() => {
    navigationApi.focusPanelInActiveTab(VIEWER_PANEL_ID);
  }, []);

  // Reuse the image item's size-badge component — its only inputs are width/height.
  const sizeBadgeImageStandIn = useMemo(
    () => ({ width: videoDTO.width, height: videoDTO.height }),
    [videoDTO.width, videoDTO.height]
  );

  // Right-click / long-press context menu (delete, change board, download).
  useVideoContextMenu(videoDTO, ref);

  // Register the item as a drag source so users can drop videos onto node fields,
  // ref-image inputs, etc. — mirrors DndImage for image gallery items.
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
          // When the dragged video is part of a multi-selection, send the whole selection so a
          // bulk move-to-board fires for every selected item. Mixed selections (videos + images)
          // ride along in the same payload: the board drop handler splits them and dispatches
          // both mutations. Without this, only the single dragged video would move.
          const state = store.getState();
          const selection = selectSelection(state);
          const boardId = selectSelectedBoardId(state);
          if (selection.length > 1 && selection.includes(videoDTO.video_name)) {
            const video_names = selection.filter(isVideoName);
            const image_names = selection.filter((n) => !isVideoName(n));
            return multipleVideoDndSource.getData({
              video_names,
              image_names,
              board_id: boardId,
            });
          }
          return singleVideoDndSource.getData({ videoDTO }, videoDTO.video_name);
        },
      })
    );
  }, [videoDTO, store]);

  return (
    <Flex
      ref={ref}
      sx={galleryItemContainerSX}
      data-item-id={videoDTO.video_name}
      role="button"
      onMouseOver={onMouseOver}
      onMouseOut={onMouseOut}
      onClick={onClick}
      onDoubleClick={onDoubleClick}
      data-selected={isSelected}
    >
      <GalleryVideoThumbnail thumbnailUrl={videoDTO.thumbnail_url} videoUrl={videoDTO.video_url} />
      <GalleryItemPlayBadge />
      {(isHovered || alwaysShowSizeBadge) && (
        <GalleryItemSizeBadge
          imageDTO={sizeBadgeImageStandIn as Parameters<typeof GalleryItemSizeBadge>[0]['imageDTO']}
        />
      )}
      {(isHovered || videoDTO.starred) && <GalleryItemVideoStarIconButton videoDTO={videoDTO} />}
    </Flex>
  );
});

GalleryVideoItem.displayName = 'GalleryVideoItem';

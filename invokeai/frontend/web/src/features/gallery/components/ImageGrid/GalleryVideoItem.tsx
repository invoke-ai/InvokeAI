import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { draggable } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { Flex, Image } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import type { AppDispatch, AppGetState } from 'app/store/store';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { uniq } from 'es-toolkit';
import { singleVideoDndSource } from 'features/dnd/dnd';
import { firefoxDndFix } from 'features/dnd/util';
import { useVideoContextMenu } from 'features/gallery/components/ContextMenu/VideoContextMenu';
import { selectAlwaysShouldImageSizeBadge } from 'features/gallery/store/gallerySelectors';
import { selectGallerySlice, selectionChanged } from 'features/gallery/store/gallerySlice';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { VIEWER_PANEL_ID } from 'features/ui/layouts/shared';
import type { MouseEvent, MouseEventHandler } from 'react';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { galleryApi } from 'services/api/endpoints/gallery';
import type { VideoDTO } from 'services/api/types';

import { galleryItemContainerSX } from './galleryItemContainerSX';
import { GalleryItemPlayBadge } from './GalleryItemPlayBadge';
import { GalleryItemSizeBadge } from './GalleryItemSizeBadge';
import { GalleryItemVideoStarIconButton } from './GalleryItemVideoStarIconButton';

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

/**
 * Returns the names of the currently-cached gallery item list (whichever query args were last
 * used). For most sessions there is exactly one active list, so iterating cache entries is fine.
 */
const selectCachedGalleryItemNames = (state: ReturnType<AppGetState>): string[] => {
  const entries = galleryApi.util.selectInvalidatedBy(state, ['GalleryItemNameList']);
  // selectInvalidatedBy returns subscription entries; for the polymorphic names list we just need
  // any one match. Fall back to scanning the API state directly for robustness.
  for (const entry of entries) {
    if (entry.endpointName !== 'getGalleryItemNames') {
      continue;
    }
    const data = galleryApi.endpoints.getGalleryItemNames.select(entry.originalArgs)(state).data;
    if (data) {
      return data.items.map((ref) => ref.name);
    }
  }
  return [];
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
        getInitialData: () => singleVideoDndSource.getData({ videoDTO }, videoDTO.video_name),
      })
    );
  }, [videoDTO]);

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
      <Image
        pointerEvents="none"
        src={videoDTO.thumbnail_url}
        objectFit="contain"
        maxW="full"
        maxH="full"
        borderRadius="base"
      />
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

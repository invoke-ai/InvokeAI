import { Box, Flex, forwardRef, Grid, GridItem, Spinner, Text } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { getFocusedRegion } from 'common/hooks/focus';
import { useRangeBasedVideoFetching } from 'features/gallery/hooks/useRangeBasedVideoFetching';
import type { selectGetVideoIdsQueryArgs } from 'features/gallery/store/gallerySelectors';
import { selectGalleryImageMinimumWidth, selectLastSelectedItem } from 'features/gallery/store/gallerySelectors';
import { selectionChanged } from 'features/gallery/store/gallerySlice';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import type { MutableRefObject } from 'react';
import React, { memo, useCallback, useEffect, useMemo, useRef } from 'react';
import type {
  GridComponents,
  GridComputeItemKey,
  GridItemContent,
  ListRange,
  ScrollSeekConfiguration,
  VirtuosoGridHandle,
} from 'react-virtuoso';
import { VirtuosoGrid } from 'react-virtuoso';
import { videosApi } from 'services/api/endpoints/videos';
import { useDebounce } from 'use-debounce';

import { getItemsPerRow } from '../../../../../../../getItemsPerRow';
import { getItemIndex } from './getItemIndex';
import { GalleryImagePlaceholder } from './ImageGrid/GalleryImage';
import { GallerySelectionCountTag } from './ImageGrid/GallerySelectionCountTag';
import { GalleryVideo } from './ImageGrid/GalleryVideo';
import { GalleryVideoPlaceholder } from './ImageGrid/GalleryVideoPlaceholder';
import { scrollIntoView } from './scrollIntoView';
import { useGalleryVideoIds } from './use-gallery-video-ids';
import { useScrollableGallery } from './useScrollableGallery';

export const log = logger('gallery');

type ListVideoIdsQueryArgs = ReturnType<typeof selectGetVideoIdsQueryArgs>;

type GridContext = {
  queryArgs: ListVideoIdsQueryArgs;
  videoIds: string[];
};

const VideoAtPosition = memo(({ videoId }: { index: number; videoId: string }) => {
  /*
   * We rely on the useRangeBasedImageFetching to fetch all image DTOs, caching them with RTK Query.
   *
   * In this component, we just want to consume that cache. Unforutnately, RTK Query does not provide a way to
   * subscribe to a query without triggering a new fetch.
   *
   * There is a hack, though:
   * - https://github.com/reduxjs/redux-toolkit/discussions/4213
   *
   * This essentially means "subscribe to the query once it has some data".
   */

  // Use `currentData` instead of `data` to prevent a flash of previous image rendered at this index
  const { currentData: videoDTO, isUninitialized } = videosApi.endpoints.getVideoDTO.useQueryState(videoId);
  videosApi.endpoints.getVideoDTO.useQuerySubscription(videoId, { skip: isUninitialized });

  if (!videoDTO) {
    return <GalleryImagePlaceholder data-video-id={videoId} />;
  }

  return <GalleryVideo videoDTO={videoDTO} />;
});
VideoAtPosition.displayName = 'VideoAtPosition';

const computeItemKey: GridComputeItemKey<string, GridContext> = (index, itemId, { queryArgs }) => {
  return `${JSON.stringify(queryArgs)}-${itemId ?? index}`;
};

/**
 * Handles keyboard navigation for the gallery.
 */
const useKeyboardNavigation = (
  itemIds: string[],
  virtuosoRef: React.RefObject<VirtuosoGridHandle>,
  rootRef: React.RefObject<HTMLDivElement>
) => {
  const { dispatch, getState } = useAppStore();

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (getFocusedRegion() !== 'gallery') {
        // Only handle keyboard navigation when the gallery is focused
        return;
      }
      // Only handle arrow keys
      if (!['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(event.key)) {
        return;
      }
      // Don't interfere if user is typing in an input
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
        return;
      }

      const rootEl = rootRef.current;
      const virtuosoGridHandle = virtuosoRef.current;

      if (!rootEl || !virtuosoGridHandle) {
        return;
      }

      if (itemIds.length === 0) {
        return;
      }

      const itemsPerRow = getItemsPerRow(rootEl);

      if (itemsPerRow === 0) {
        // This can happen if the grid is not yet rendered or has no items
        return;
      }

      event.preventDefault();

      const state = getState();
      const itemId = selectLastSelectedItem(state)?.id;

      const currentIndex = getItemIndex(itemId, itemIds);

      let newIndex = currentIndex;

      switch (event.key) {
        case 'ArrowLeft':
          if (currentIndex > 0) {
            newIndex = currentIndex - 1;
          }
          break;
        case 'ArrowRight':
          if (currentIndex < itemIds.length - 1) {
            newIndex = currentIndex + 1;
          }
          break;
        case 'ArrowUp':
          // If on first row, stay on current item
          if (currentIndex < itemsPerRow) {
            newIndex = currentIndex;
          } else {
            newIndex = Math.max(0, currentIndex - itemsPerRow);
          }
          break;
        case 'ArrowDown':
          // If no items below, stay on current item
          if (currentIndex >= itemIds.length - itemsPerRow) {
            newIndex = currentIndex;
          } else {
            newIndex = Math.min(itemIds.length - 1, currentIndex + itemsPerRow);
          }
          break;
      }

      if (newIndex !== currentIndex && newIndex >= 0 && newIndex < itemIds.length) {
        const nextItemId = itemIds[newIndex];
        if (nextItemId) {
          dispatch(selectionChanged([{ type: 'video', id: nextItemId }]));
        }
      }
    },
    [rootRef, virtuosoRef, itemIds, getState, dispatch]
  );

  useRegisteredHotkeys({
    id: 'galleryNavLeft',
    category: 'gallery',
    callback: handleKeyDown,
    options: { preventDefault: true },
    dependencies: [handleKeyDown],
  });

  useRegisteredHotkeys({
    id: 'galleryNavRight',
    category: 'gallery',
    callback: handleKeyDown,
    options: { preventDefault: true },
    dependencies: [handleKeyDown],
  });

  useRegisteredHotkeys({
    id: 'galleryNavUp',
    category: 'gallery',
    callback: handleKeyDown,
    options: { preventDefault: true },
    dependencies: [handleKeyDown],
  });

  useRegisteredHotkeys({
    id: 'galleryNavDown',
    category: 'gallery',
    callback: handleKeyDown,
    options: { preventDefault: true },
    dependencies: [handleKeyDown],
  });

  useRegisteredHotkeys({
    id: 'galleryNavLeftAlt',
    category: 'gallery',
    callback: handleKeyDown,
    options: { preventDefault: true },
    dependencies: [handleKeyDown],
  });

  useRegisteredHotkeys({
    id: 'galleryNavRightAlt',
    category: 'gallery',
    callback: handleKeyDown,
    options: { preventDefault: true },
    dependencies: [handleKeyDown],
  });

  useRegisteredHotkeys({
    id: 'galleryNavUpAlt',
    category: 'gallery',
    callback: handleKeyDown,
    options: { preventDefault: true },
    dependencies: [handleKeyDown],
  });

  useRegisteredHotkeys({
    id: 'galleryNavDownAlt',
    category: 'gallery',
    callback: handleKeyDown,
    options: { preventDefault: true },
    dependencies: [handleKeyDown],
  });
};

/**
 * Keeps the last selected image in view when the gallery is scrolled.
 * This is useful for keyboard navigation and ensuring the user can see their selection.
 * It only tracks the last selected image, not the image to compare.
 */
const useKeepSelectedVideoInView = (
  videoIds: string[],
  virtuosoRef: React.RefObject<VirtuosoGridHandle>,
  rootRef: React.RefObject<HTMLDivElement>,
  rangeRef: MutableRefObject<ListRange>
) => {
  const targetVideoId = useAppSelector(selectLastSelectedItem)?.id;

  useEffect(() => {
    const virtuosoGridHandle = virtuosoRef.current;
    const rootEl = rootRef.current;
    const range = rangeRef.current;

    if (!virtuosoGridHandle || !rootEl || !targetVideoId || !videoIds || videoIds.length === 0) {
      return;
    }
    scrollIntoView(targetVideoId, videoIds, rootEl, virtuosoGridHandle, range);
  }, [targetVideoId, videoIds, rangeRef, rootRef, virtuosoRef]);
};

export const VideoGallery = memo(() => {
  const virtuosoRef = useRef<VirtuosoGridHandle>(null);
  const rangeRef = useRef<ListRange>({ startIndex: 0, endIndex: 0 });
  const rootRef = useRef<HTMLDivElement>(null);

  // Get the ordered list of image names - this is our primary data source for virtualization
  const { queryArgs, videoIds, isLoading } = useGalleryVideoIds();

  // Use range-based fetching for bulk loading image DTOs into cache based on the visible range
  const { onRangeChanged } = useRangeBasedVideoFetching({
    videoIds,
    enabled: !isLoading,
  });

  useKeepSelectedVideoInView(videoIds, virtuosoRef, rootRef, rangeRef);
  useKeyboardNavigation(videoIds, virtuosoRef, rootRef);
  const scrollerRef = useScrollableGallery(rootRef);

  /*
   * We have to keep track of the visible range for keep-selected-image-in-view functionality and push the range to
   * the range-based image fetching hook.
   */
  const handleRangeChanged = useCallback(
    (range: ListRange) => {
      rangeRef.current = range;
      onRangeChanged(range);
    },
    [onRangeChanged]
  );

  const context = useMemo<GridContext>(() => ({ videoIds, queryArgs }), [videoIds, queryArgs]);

  if (isLoading) {
    return (
      <Flex w="full" h="full" alignItems="center" justifyContent="center" gap={4}>
        <Spinner size="lg" opacity={0.3} />
        <Text color="base.300">Loading gallery...</Text>
      </Flex>
    );
  }

  if (videoIds.length === 0) {
    return (
      <Flex w="full" h="full" alignItems="center" justifyContent="center">
        <Text color="base.300">No videos found</Text>
      </Flex>
    );
  }

  return (
    // This wrapper component is necessary to initialize the overlay scrollbars!
    <Box data-overlayscrollbars-initialize="" ref={rootRef} position="relative" w="full" h="full">
      <VirtuosoGrid<string, GridContext>
        ref={virtuosoRef}
        context={context}
        data={videoIds}
        increaseViewportBy={4096}
        itemContent={itemContent}
        computeItemKey={computeItemKey}
        components={components}
        style={virtuosoGridStyle}
        scrollerRef={scrollerRef}
        scrollSeekConfiguration={scrollSeekConfiguration}
        rangeChanged={handleRangeChanged}
      />
      <GallerySelectionCountTag />
    </Box>
  );
});

VideoGallery.displayName = 'VideoGallery';

const scrollSeekConfiguration: ScrollSeekConfiguration = {
  enter: (velocity) => {
    return Math.abs(velocity) > 2048;
  },
  exit: (velocity) => {
    return velocity === 0;
  },
};

// Styles
const virtuosoGridStyle = { height: '100%', width: '100%' };

const selectGridTemplateColumns = createSelector(
  selectGalleryImageMinimumWidth,
  (galleryImageMinimumWidth) => `repeat(auto-fill, minmax(${galleryImageMinimumWidth}px, 1fr))`
);

// Grid components
const ListComponent: GridComponents<GridContext>['List'] = forwardRef(({ context: _, ...rest }, ref) => {
  const _gridTemplateColumns = useAppSelector(selectGridTemplateColumns);
  const [gridTemplateColumns] = useDebounce(_gridTemplateColumns, 300);

  return <Grid ref={ref} gridTemplateColumns={gridTemplateColumns} gap={1} {...rest} />;
});
ListComponent.displayName = 'ListComponent';

const itemContent: GridItemContent<string, GridContext> = (index, videoId) => {
  return <VideoAtPosition index={index} videoId={videoId} />;
};

const ItemComponent: GridComponents<GridContext>['Item'] = forwardRef(({ context: _, ...rest }, ref) => (
  <GridItem ref={ref} aspectRatio="1/1" {...rest} />
));
ItemComponent.displayName = 'ItemComponent';

const ScrollSeekPlaceholderComponent: GridComponents<GridContext>['ScrollSeekPlaceholder'] = (props) => (
  <GridItem aspectRatio="1/1" {...props}>
    <GalleryVideoPlaceholder />
  </GridItem>
);

ScrollSeekPlaceholderComponent.displayName = 'ScrollSeekPlaceholderComponent';

const components: GridComponents<GridContext> = {
  Item: ItemComponent,
  List: ListComponent,
  ScrollSeekPlaceholder: ScrollSeekPlaceholderComponent,
};

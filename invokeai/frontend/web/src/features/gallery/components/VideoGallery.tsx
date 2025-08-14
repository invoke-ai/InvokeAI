import { Box, Flex, forwardRef, Grid, GridItem, Spinner, Text } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { getFocusedRegion, useIsRegionFocused } from 'common/hooks/focus';
import { useRangeBasedImageFetching } from 'features/gallery/hooks/useRangeBasedImageFetching';
import type { selectGetImageNamesQueryArgs, selectGetVideoIdsQueryArgs } from 'features/gallery/store/gallerySelectors';
import {
  selectGalleryImageMinimumWidth,
  selectGalleryView,
  selectImageToCompare,
  selectLastSelectedImage,
  selectSelectionCount,
} from 'features/gallery/store/gallerySelectors';
import { imageToCompareChanged, selectionChanged } from 'features/gallery/store/gallerySlice';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useOverlayScrollbars } from 'overlayscrollbars-react';
import type { MutableRefObject, RefObject } from 'react';
import React, { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type {
  GridComponents,
  GridComputeItemKey,
  GridItemContent,
  ListRange,
  ScrollSeekConfiguration,
  VirtuosoGridHandle,
} from 'react-virtuoso';
import { VirtuosoGrid } from 'react-virtuoso';
import { useDebounce } from 'use-debounce';

import { GallerySelectionCountTag } from './ImageGrid/GallerySelectionCountTag';
import { useGalleryImageNames } from './use-gallery-image-names';
import { useGalleryVideoIds } from './use-gallery-video-ids';
import { videosApi } from 'services/api/endpoints/videos';
import { GalleryImagePlaceholder } from './ImageGrid/GalleryImage';
import { useRangeBasedVideoFetching } from '../hooks/useRangeBasedVideoFetching';
import { GalleryVideo } from './ImageGrid/GalleryVideo';

const log = logger('gallery');

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

const computeItemKey: GridComputeItemKey<string, GridContext> = (index, imageName, { queryArgs }) => {
  return `${JSON.stringify(queryArgs)}-${imageName ?? index}`;
};

/**
 * Calculate how many images fit in a row based on the current grid layout.
 *
 * TODO(psyche): We only need to do this when the gallery width changes, or when the galleryImageMinimumWidth value
 * changes. Cache this calculation.
 */
const getVideosPerRow = (rootEl: HTMLDivElement): number => {
  // Start from root and find virtuoso grid elements
  const gridElement = rootEl.querySelector('.virtuoso-grid-list');

  if (!gridElement) {
    return 0;
  }

  const firstGridItem = gridElement.querySelector('.virtuoso-grid-item');

  if (!firstGridItem) {
    return 0;
  }

  const itemRect = firstGridItem.getBoundingClientRect();
  const containerRect = gridElement.getBoundingClientRect();

  // Get the computed gap from CSS
  const gridStyle = window.getComputedStyle(gridElement);
  const gapValue = gridStyle.gap;
  const gap = parseFloat(gapValue);

  if (isNaN(gap) || !itemRect.width || !itemRect.height || !containerRect.width || !containerRect.height) {
    return 0;
  }

  /**
   * You might be tempted to just do some simple math like:
   * const imagesPerRow = Math.floor(containerRect.width / itemRect.width);
   *
   * But floating point precision can cause issues with this approach, causing it to be off by 1 in some cases.
   *
   * Instead, we use a more robust approach that iteratively calculates how many images fit in the row.
   */
  let videosPerRow = 0;
  let spaceUsed = 0;

  // Floating point precision can cause imagesPerRow to be 1 too small. Adding 1px to the container size fixes
  // this, without the possibility of accidentally adding an extra column.
  while (spaceUsed + itemRect.width <= containerRect.width + 1) {
    videosPerRow++; // Increment the number of images
    spaceUsed += itemRect.width; // Add image size to the used space
    if (spaceUsed + gap <= containerRect.width) {
      spaceUsed += gap; // Add gap size to the used space after each image except after the last image
    }
  }

    return Math.max(1, videosPerRow);
};

/**
 * Scroll the item at the given index into view if it is not currently visible.
 */
const scrollIntoView = (
  targetVideoId: string,
  videoIds: string[],
  rootEl: HTMLDivElement,
  virtuosoGridHandle: VirtuosoGridHandle,
  range: ListRange
) => {
  if (range.endIndex === 0) {
    // No range is rendered; no need to scroll to anything.
    return;
  }

  const targetIndex = videoIds.findIndex((id) => id === targetVideoId);

  if (targetIndex === -1) {
    // The image isn't in the currently rendered list.
    return;
  }

  const targetItem = rootEl.querySelector(
    `.virtuoso-grid-item:has([data-video-id="${targetVideoId}"])`
  ) as HTMLElement;

  if (!targetItem) {
    if (targetIndex > range.endIndex) {
      virtuosoGridHandle.scrollToIndex({
        index: targetIndex,
        behavior: 'auto',
        align: 'start',
      });
    } else if (targetIndex < range.startIndex) {
      virtuosoGridHandle.scrollToIndex({
        index: targetIndex,
        behavior: 'auto',
        align: 'end',
      });
    } else {
      log.debug(
        `Unable to find video ${targetVideoId} at index ${targetIndex} but it is in the rendered range ${range.startIndex}-${range.endIndex}`
      );
    }
    return;
  }

  // We found the image in the DOM, but it might be in the overscan range - rendered but not in the visible viewport.
  // Check if it is in the viewport and scroll if necessary.

  const itemRect = targetItem.getBoundingClientRect();
  const rootRect = rootEl.getBoundingClientRect();

  if (itemRect.top < rootRect.top) {
    virtuosoGridHandle.scrollToIndex({
      index: targetIndex,
      behavior: 'auto',
      align: 'start',
    });
  } else if (itemRect.bottom > rootRect.bottom) {
    virtuosoGridHandle.scrollToIndex({
      index: targetIndex,
      behavior: 'auto',
      align: 'end',
    });
  } else {
    // Image is already in view
  }

  return;
};

/**
 * Get the index of the image in the list of image names.
 * If the image name is not found, return 0.
 * If no image name is provided, return 0.
 */
const getVideoIndex = (videoId: string | undefined | null, videoIds: string[]) => {
  if (!videoId || videoIds.length === 0) {
    return 0;
  }
  const index = videoIds.findIndex((n) => n === videoId);
  return index >= 0 ? index : 0;
};

/**
 * Handles keyboard navigation for the gallery.
 */
const useKeyboardNavigation = (
  videoIds: string[],
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

      if (videoIds.length === 0) {
        return;
      }

      const videosPerRow = getVideosPerRow(rootEl);

      if (videosPerRow === 0) {
        // This can happen if the grid is not yet rendered or has no items
        return;
      }

      event.preventDefault();

      const state = getState();
      const videoId = event.altKey
        ? // When the user holds alt, we are changing the image to compare - if no image to compare is currently selected,
          // we start from the last selected image
          (selectImageToCompare(state) ?? selectLastSelectedImage(state))
        : selectLastSelectedImage(state);

      const currentIndex = getVideoIndex(videoId, videoIds);

      let newIndex = currentIndex;

      switch (event.key) {
        case 'ArrowLeft':
          if (currentIndex > 0) {
            newIndex = currentIndex - 1;
            // } else {
            //   // Wrap to last image
            //   newIndex = imageNames.length - 1;
          }
          break;
        case 'ArrowRight':
          if (currentIndex < videoIds.length - 1) {
            newIndex = currentIndex + 1;
            // } else {
            //   // Wrap to first image
            //   newIndex = 0;
          }
          break;
        case 'ArrowUp':
          // If on first row, stay on current image
          if (currentIndex < videosPerRow) {
            newIndex = currentIndex;
          } else {
            newIndex = Math.max(0, currentIndex - videosPerRow);
          }
          break;
        case 'ArrowDown':
          // If no images below, stay on current image
          if (currentIndex >= videoIds.length - videosPerRow) {
            newIndex = currentIndex;
          } else {
            newIndex = Math.min(videoIds.length - 1, currentIndex + videosPerRow);
          }
          break;
      }

      if (newIndex !== currentIndex && newIndex >= 0 && newIndex < videoIds.length) {
        const newVideoId = videoIds[newIndex];
        if (newVideoId) {
         
          dispatch(selectionChanged([newVideoId]));
         
        }
      }
    },
    [rootRef, virtuosoRef, videoIds, getState, dispatch]
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
  const targetVideoId = useAppSelector(selectLastSelectedImage);

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

/**
 * Handles the initialization of the overlay scrollbars for the gallery, returning the ref to the scroller element.
 */
const useScrollableGallery = (rootRef: RefObject<HTMLDivElement>) => {
  const [scroller, scrollerRef] = useState<HTMLElement | null>(null);
  const [initialize, osInstance] = useOverlayScrollbars({
    defer: true,
    events: {
      initialized(osInstance) {
        // force overflow styles
        const { viewport } = osInstance.elements();
        viewport.style.overflowX = `var(--os-viewport-overflow-x)`;
        viewport.style.overflowY = `var(--os-viewport-overflow-y)`;
      },
    },
    options: {
      scrollbars: {
        visibility: 'auto',
        autoHide: 'scroll',
        autoHideDelay: 1300,
        theme: 'os-theme-dark',
      },
    },
  });

  useEffect(() => {
    const { current: root } = rootRef;

    if (scroller && root) {
      initialize({
        target: root,
        elements: {
          viewport: scroller,
        },
      });
    }

    return () => {
      osInstance()?.destroy();
    };
  }, [scroller, initialize, osInstance, rootRef]);

  return scrollerRef;
};



export const VideoGallery = memo(() => {
  const virtuosoRef = useRef<VirtuosoGridHandle>(null);
  const rangeRef = useRef<ListRange>({ startIndex: 0, endIndex: 0 });
  const rootRef = useRef<HTMLDivElement>(null);
  const galleryView = useAppSelector(selectGalleryView);

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
        style={style}
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
const style = { height: '100%', width: '100%' };

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
    <GalleryImagePlaceholder />
  </GridItem>
);

ScrollSeekPlaceholderComponent.displayName = 'ScrollSeekPlaceholderComponent';

const components: GridComponents<GridContext> = {
  Item: ItemComponent,
  List: ListComponent,
  ScrollSeekPlaceholder: ScrollSeekPlaceholderComponent,
};

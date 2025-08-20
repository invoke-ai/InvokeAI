import { Box, Flex, forwardRef, Grid, GridItem, Spinner, Text } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { getFocusedRegion, useIsRegionFocused } from 'common/hooks/focus';
import { useRangeBasedImageFetching } from 'features/gallery/hooks/useRangeBasedImageFetching';
import type { selectGetImageNamesQueryArgs } from 'features/gallery/store/gallerySelectors';
import {
  selectGalleryImageMinimumWidth,
  selectImageToCompare,
  selectLastSelectedImage,
  selectSelection,
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
import { imagesApi, useImageDTO, useStarImagesMutation, useUnstarImagesMutation } from 'services/api/endpoints/images';
import { useDebounce } from 'use-debounce';

import { GalleryImage, GalleryImagePlaceholder } from './ImageGrid/GalleryImage';
import { GallerySelectionCountTag } from './ImageGrid/GallerySelectionCountTag';
import { useGalleryImageNames } from './use-gallery-image-names';

const log = logger('gallery');

type ListImageNamesQueryArgs = ReturnType<typeof selectGetImageNamesQueryArgs>;

type GridContext = {
  queryArgs: ListImageNamesQueryArgs;
  imageNames: string[];
};

const ImageAtPosition = memo(({ imageName }: { index: number; imageName: string }) => {
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
  const { currentData: imageDTO, isUninitialized } = imagesApi.endpoints.getImageDTO.useQueryState(imageName);
  imagesApi.endpoints.getImageDTO.useQuerySubscription(imageName, { skip: isUninitialized });

  if (!imageDTO) {
    return <GalleryImagePlaceholder data-image-name={imageName} />;
  }

  return <GalleryImage imageDTO={imageDTO} />;
});
ImageAtPosition.displayName = 'ImageAtPosition';

const computeItemKey: GridComputeItemKey<string, GridContext> = (index, imageName, { queryArgs }) => {
  return `${JSON.stringify(queryArgs)}-${imageName ?? index}`;
};

/**
 * Calculate how many images fit in a row based on the current grid layout.
 *
 * TODO(psyche): We only need to do this when the gallery width changes, or when the galleryImageMinimumWidth value
 * changes. Cache this calculation.
 */
const getImagesPerRow = (rootEl: HTMLDivElement): number => {
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
  let imagesPerRow = 0;
  let spaceUsed = 0;

  // Floating point precision can cause imagesPerRow to be 1 too small. Adding 1px to the container size fixes
  // this, without the possibility of accidentally adding an extra column.
  while (spaceUsed + itemRect.width <= containerRect.width + 1) {
    imagesPerRow++; // Increment the number of images
    spaceUsed += itemRect.width; // Add image size to the used space
    if (spaceUsed + gap <= containerRect.width) {
      spaceUsed += gap; // Add gap size to the used space after each image except after the last image
    }
  }

  return Math.max(1, imagesPerRow);
};

/**
 * Scroll the item at the given index into view if it is not currently visible.
 */
const scrollIntoView = (
  targetImageName: string,
  imageNames: string[],
  rootEl: HTMLDivElement,
  virtuosoGridHandle: VirtuosoGridHandle,
  range: ListRange
) => {
  if (range.endIndex === 0) {
    // No range is rendered; no need to scroll to anything.
    log.trace('Not scrolling into view: Range endIdex is 0');
    return;
  }

  const targetIndex = imageNames.findIndex((name) => name === targetImageName);

  if (targetIndex === -1) {
    // The image isn't in the currently rendered list.
    log.trace('Not scrolling into view: targetIndex is -1');
    return;
  }

  const targetItem = rootEl.querySelector(
    `.virtuoso-grid-item:has([data-image-name="${targetImageName}"])`
  ) as HTMLElement;

  if (!targetItem) {
    if (targetIndex > range.endIndex) {
      log.trace(
        {
          index: targetIndex,
          behavior: 'auto',
          align: 'start',
        },
        'Scrolling into view: not in DOM'
      );
      virtuosoGridHandle.scrollToIndex({
        index: targetIndex,
        behavior: 'auto',
        align: 'start',
      });
    } else if (targetIndex < range.startIndex) {
      log.trace(
        {
          index: targetIndex,
          behavior: 'auto',
          align: 'end',
        },
        'Scrolling into view: not in DOM'
      );
      virtuosoGridHandle.scrollToIndex({
        index: targetIndex,
        behavior: 'auto',
        align: 'end',
      });
    } else {
      log.debug(
        `Unable to find image ${targetImageName} at index ${targetIndex} but it is in the rendered range ${range.startIndex}-${range.endIndex}`
      );
    }
    return;
  }

  // We found the image in the DOM, but it might be in the overscan range - rendered but not in the visible viewport.
  // Check if it is in the viewport and scroll if necessary.

  const itemRect = targetItem.getBoundingClientRect();
  const rootRect = rootEl.getBoundingClientRect();

  if (itemRect.top < rootRect.top) {
    log.trace(
      {
        index: targetIndex,
        behavior: 'auto',
        align: 'start',
      },
      'Scrolling into view: in overscan'
    );
    virtuosoGridHandle.scrollToIndex({
      index: targetIndex,
      behavior: 'auto',
      align: 'start',
    });
  } else if (itemRect.bottom > rootRect.bottom) {
    log.trace(
      {
        index: targetIndex,
        behavior: 'auto',
        align: 'end',
      },
      'Scrolling into view: in overscan'
    );
    virtuosoGridHandle.scrollToIndex({
      index: targetIndex,
      behavior: 'auto',
      align: 'end',
    });
  } else {
    // Image is already in view
    log.debug('Not scrolling into view: Image is already in view');
  }

  return;
};

/**
 * Get the index of the image in the list of image names.
 * If the image name is not found, return 0.
 * If no image name is provided, return 0.
 */
const getImageIndex = (imageName: string | undefined | null, imageNames: string[]) => {
  if (!imageName || imageNames.length === 0) {
    return 0;
  }
  const index = imageNames.findIndex((n) => n === imageName);
  return index >= 0 ? index : 0;
};

/**
 * Handles keyboard navigation for the gallery.
 */
const useKeyboardNavigation = (
  imageNames: string[],
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

      if (imageNames.length === 0) {
        return;
      }

      const imagesPerRow = getImagesPerRow(rootEl);

      if (imagesPerRow === 0) {
        // This can happen if the grid is not yet rendered or has no items
        return;
      }

      event.preventDefault();

      const state = getState();
      const imageName = event.altKey
        ? // When the user holds alt, we are changing the image to compare - if no image to compare is currently selected,
          // we start from the last selected image
          (selectImageToCompare(state) ?? selectLastSelectedImage(state))
        : selectLastSelectedImage(state);

      const currentIndex = getImageIndex(imageName, imageNames);

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
          if (currentIndex < imageNames.length - 1) {
            newIndex = currentIndex + 1;
            // } else {
            //   // Wrap to first image
            //   newIndex = 0;
          }
          break;
        case 'ArrowUp':
          // If on first row, stay on current image
          if (currentIndex < imagesPerRow) {
            newIndex = currentIndex;
          } else {
            newIndex = Math.max(0, currentIndex - imagesPerRow);
          }
          break;
        case 'ArrowDown':
          // If no images below, stay on current image
          if (currentIndex >= imageNames.length - imagesPerRow) {
            newIndex = currentIndex;
          } else {
            newIndex = Math.min(imageNames.length - 1, currentIndex + imagesPerRow);
          }
          break;
      }

      if (newIndex !== currentIndex && newIndex >= 0 && newIndex < imageNames.length) {
        const newImageName = imageNames[newIndex];
        if (newImageName) {
          if (event.altKey) {
            dispatch(imageToCompareChanged(newImageName));
          } else {
            dispatch(selectionChanged([newImageName]));
          }
        }
      }
    },
    [rootRef, virtuosoRef, imageNames, getState, dispatch]
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
const useKeepSelectedImageInView = (
  imageNames: string[],
  virtuosoRef: React.RefObject<VirtuosoGridHandle>,
  rootRef: React.RefObject<HTMLDivElement>,
  rangeRef: MutableRefObject<ListRange>
) => {
  const selection = useAppSelector(selectSelection);

  useEffect(() => {
    const targetImageName = selection.at(-1);
    const virtuosoGridHandle = virtuosoRef.current;
    const rootEl = rootRef.current;
    const range = rangeRef.current;

    if (!virtuosoGridHandle || !rootEl || !targetImageName || !imageNames || imageNames.length === 0) {
      return;
    }

    setTimeout(() => {
      scrollIntoView(targetImageName, imageNames, rootEl, virtuosoGridHandle, range);
    }, 0);
  }, [imageNames, rangeRef, rootRef, virtuosoRef, selection]);
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

const useStarImageHotkey = () => {
  const lastSelectedImage = useAppSelector(selectLastSelectedImage);
  const selectionCount = useAppSelector(selectSelectionCount);
  const isGalleryFocused = useIsRegionFocused('gallery');
  const imageDTO = useImageDTO(lastSelectedImage);
  const [starImages] = useStarImagesMutation();
  const [unstarImages] = useUnstarImagesMutation();

  const handleStarHotkey = useCallback(() => {
    if (!imageDTO) {
      return;
    }
    if (!isGalleryFocused) {
      return;
    }
    if (imageDTO.starred) {
      unstarImages({ image_names: [imageDTO.image_name] });
    } else {
      starImages({ image_names: [imageDTO.image_name] });
    }
  }, [imageDTO, isGalleryFocused, starImages, unstarImages]);

  useRegisteredHotkeys({
    id: 'starImage',
    category: 'gallery',
    callback: handleStarHotkey,
    options: { enabled: !!imageDTO && selectionCount === 1 && isGalleryFocused },
    dependencies: [imageDTO, selectionCount, isGalleryFocused, handleStarHotkey],
  });
};

export const NewGallery = memo(() => {
  const virtuosoRef = useRef<VirtuosoGridHandle>(null);
  const rangeRef = useRef<ListRange>({ startIndex: 0, endIndex: 0 });
  const rootRef = useRef<HTMLDivElement>(null);

  // Get the ordered list of image names - this is our primary data source for virtualization
  const { queryArgs, imageNames, isLoading } = useGalleryImageNames();

  // Use range-based fetching for bulk loading image DTOs into cache based on the visible range
  const { onRangeChanged } = useRangeBasedImageFetching({
    imageNames,
    enabled: !isLoading,
  });

  useStarImageHotkey();
  useKeepSelectedImageInView(imageNames, virtuosoRef, rootRef, rangeRef);
  useKeyboardNavigation(imageNames, virtuosoRef, rootRef);
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

  const context = useMemo<GridContext>(() => ({ imageNames, queryArgs }), [imageNames, queryArgs]);

  if (isLoading) {
    return (
      <Flex w="full" h="full" alignItems="center" justifyContent="center" gap={4}>
        <Spinner size="lg" opacity={0.3} />
        <Text color="base.300">Loading gallery...</Text>
      </Flex>
    );
  }

  if (imageNames.length === 0) {
    return (
      <Flex w="full" h="full" alignItems="center" justifyContent="center">
        <Text color="base.300">No images found</Text>
      </Flex>
    );
  }

  return (
    // This wrapper component is necessary to initialize the overlay scrollbars!
    <Box data-overlayscrollbars-initialize="" ref={rootRef} position="relative" w="full" h="full">
      <VirtuosoGrid<string, GridContext>
        ref={virtuosoRef}
        context={context}
        data={imageNames}
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

NewGallery.displayName = 'NewGallery';

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

const itemContent: GridItemContent<string, GridContext> = (index, imageName) => {
  return <ImageAtPosition index={index} imageName={imageName} />;
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

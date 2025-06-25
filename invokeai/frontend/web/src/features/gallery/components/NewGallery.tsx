import { Box, Flex, forwardRef, Grid, GridItem, Skeleton, Spinner, Text } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectGalleryImageMinimumWidth,
  selectImageCollectionQueryArgs,
  selectLastSelectedImage,
} from 'features/gallery/store/gallerySelectors';
import { selectionChanged } from 'features/gallery/store/gallerySlice';
import { useOverlayScrollbars } from 'overlayscrollbars-react';
import type { MutableRefObject } from 'react';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type {
  GridComponents,
  GridComputeItemKey,
  GridItemContent,
  ListRange,
  ScrollSeekConfiguration,
  VirtuosoGridHandle,
} from 'react-virtuoso';
import { VirtuosoGrid } from 'react-virtuoso';
import { useGetImageNamesQuery, useListImagesQuery } from 'services/api/endpoints/images';
import type { ImageDTO, ListImagesArgs } from 'services/api/types';
import { useDebounce } from 'use-debounce';

import { GalleryImage } from './ImageGrid/GalleryImage';

const log = logger('gallery');

// Constants
const PAGE_SIZE = 100;
const VIEWPORT_BUFFER = 2048;
const SCROLL_SEEK_VELOCITY_THRESHOLD = 4096;
const DEBOUNCE_DELAY = 500;
const SPINNER_OPACITY = 0.3;

type GridContext = {
  queryArgs: ListImagesArgs;
  imageNames: string[];
};

export const useDebouncedImageCollectionQueryArgs = () => {
  const _galleryQueryArgs = useAppSelector(selectImageCollectionQueryArgs);
  const [queryArgs] = useDebounce(_galleryQueryArgs, DEBOUNCE_DELAY);
  return queryArgs;
};

// Hook to get an image DTO from cache or trigger loading
const useImageDTOFromListQuery = (index: number, imageName: string, queryArgs: ListImagesArgs): ImageDTO | null => {
  const { arg, options } = useMemo(() => {
    const pageOffset = Math.floor(index / PAGE_SIZE) * PAGE_SIZE;
    return {
      arg: {
        ...queryArgs,
        offset: pageOffset,
        limit: PAGE_SIZE,
      } satisfies Parameters<typeof useListImagesQuery>[0],
      options: {
        selectFromResult: ({ data }) => {
          const imageDTO = data?.items?.[index - pageOffset] || null;
          if (imageDTO && imageDTO.image_name !== imageName) {
            log.warn(`Image at index ${index} does not match expected image name ${imageName}`);
          }
          return { imageDTO };
        },
      } satisfies Parameters<typeof useListImagesQuery>[1],
    };
  }, [index, queryArgs, imageName]);

  const { imageDTO } = useListImagesQuery(arg, options);

  return imageDTO;
};

// Individual image component that gets its data from RTK Query cache
const ImageAtPosition = memo(
  ({ index, queryArgs, imageName }: { index: number; imageName: string; queryArgs: ListImagesArgs }) => {
    const imageDTO = useImageDTOFromListQuery(index, imageName, queryArgs);

    if (!imageDTO) {
      return <Skeleton w="full" h="full" />;
    }

    return <GalleryImage imageDTO={imageDTO} />;
  }
);
ImageAtPosition.displayName = 'ImageAtPosition';

// Memoized compute key function using image names
const computeItemKey: GridComputeItemKey<string, GridContext> = (index, imageName, { queryArgs }) => {
  return `${JSON.stringify(queryArgs)}-${imageName}`;
};

// Physical DOM-based grid calculation using refs (based on working old implementation)
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

  // Use the exact calculation from the working old implementation
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

// Check if an item at a given index is visible in the viewport
const scrollIntoView = (
  index: number,
  rootEl: HTMLDivElement,
  virtuosoGridHandle: VirtuosoGridHandle,
  range: ListRange
) => {
  if (range.endIndex === 0) {
    return;
  }

  // First get the virtuoso grid list root element
  const gridList = rootEl.querySelector('.virtuoso-grid-list') as HTMLElement;

  if (!gridList) {
    // No grid - cannot scroll!
    return;
  }

  // Then find the specific item within the grid list
  const targetItem = gridList.querySelector(`.virtuoso-grid-item[data-index="${index}"]`) as HTMLElement;

  if (!targetItem) {
    if (index > range.endIndex) {
      virtuosoGridHandle.scrollToIndex({
        index,
        behavior: 'auto',
        align: 'start',
      });
    } else if (index < range.startIndex) {
      virtuosoGridHandle.scrollToIndex({
        index,
        behavior: 'auto',
        align: 'end',
      });
    } else {
      log.warn(`Unable to find item index ${index} but it is in range ${range.startIndex}-${range.endIndex}`);
    }
    return;
  }

  const itemRect = targetItem.getBoundingClientRect();
  const rootRect = rootEl.getBoundingClientRect();

  if (itemRect.top < rootRect.top) {
    virtuosoGridHandle.scrollToIndex({
      index,
      behavior: 'auto',
      align: 'start',
    });
  } else if (itemRect.bottom > rootRect.bottom) {
    virtuosoGridHandle.scrollToIndex({
      index,
      behavior: 'auto',
      align: 'end',
    });
  }

  return;
};

// Hook for keyboard navigation using physical DOM measurements
const useKeyboardNavigation = (
  imageNames: string[],
  virtuosoRef: React.RefObject<VirtuosoGridHandle>,
  rootRef: React.RefObject<HTMLDivElement>,
  rangeRef: MutableRefObject<ListRange>
) => {
  const dispatch = useAppDispatch();
  const lastSelectedImage = useAppSelector(selectLastSelectedImage);

  // Get current index of selected image
  const currentIndex = useMemo(() => {
    if (!lastSelectedImage || imageNames.length === 0) {
      return 0;
    }
    const index = imageNames.findIndex((name) => name === lastSelectedImage);
    return index >= 0 ? index : 0;
  }, [lastSelectedImage, imageNames]);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      const rootEl = rootRef.current;
      const virtuosoGridHandle = virtuosoRef.current;
      const range = rangeRef.current;
      if (!rootEl || !virtuosoGridHandle) {
        return;
      }
      if (imageNames.length === 0) {
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

      const imagesPerRow = getImagesPerRow(rootEl);

      if (imagesPerRow === 0) {
        // This can happen if the grid is not yet rendered or has no items
        return;
      }

      event.preventDefault();

      let newIndex = currentIndex;

      switch (event.key) {
        case 'ArrowLeft':
          if (currentIndex > 0) {
            newIndex = currentIndex - 1;
          } else {
            // Wrap to last image
            newIndex = imageNames.length - 1;
          }
          break;
        case 'ArrowRight':
          if (currentIndex < imageNames.length - 1) {
            newIndex = currentIndex + 1;
          } else {
            // Wrap to first image
            newIndex = 0;
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
          dispatch(selectionChanged([newImageName]));
          scrollIntoView(newIndex, rootEl, virtuosoGridHandle, range);
        }
      }
    },
    [rootRef, virtuosoRef, rangeRef, imageNames, currentIndex, dispatch]
  );

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [handleKeyDown]);
};

const getImageNamesQueryOptions = {
  selectFromResult: ({ data, isLoading }) => ({
    imageNames: data ?? EMPTY_ARRAY,
    isLoading,
  }),
} satisfies Parameters<typeof useGetImageNamesQuery>[1];

// Main gallery component
export const NewGallery = memo(() => {
  const queryArgs = useDebouncedImageCollectionQueryArgs();
  const virtuosoRef = useRef<VirtuosoGridHandle>(null);
  const rangeRef = useRef<ListRange>({ startIndex: 0, endIndex: 0 });

  // Get the ordered list of image names - this is our primary data source for virtualization
  const { imageNames, isLoading } = useGetImageNamesQuery(queryArgs, getImageNamesQueryOptions);

  // Reset scroll position when query parameters change
  useEffect(() => {
    if (virtuosoRef.current && imageNames.length > 0) {
      virtuosoRef.current.scrollToIndex({ index: 0, behavior: 'auto' });
    }
  }, [queryArgs, imageNames.length]);

  const rootRef = useRef<HTMLDivElement>(null);

  // Enable keyboard navigation
  useKeyboardNavigation(imageNames, virtuosoRef, rootRef, rangeRef);

  const [scroller, setScroller] = useState<HTMLElement | null>(null);
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
  }, [scroller, initialize, osInstance]);

  // Handle range changes - RTK Query will automatically cache and manage loading
  const handleRangeChanged = useCallback((range: ListRange) => {
    rangeRef.current = range;
  }, []);

  const context = useMemo(
    () =>
      ({
        imageNames,
        queryArgs,
      }) satisfies GridContext,
    [imageNames, queryArgs]
  );

  // Item content function
  const itemContent: GridItemContent<string, GridContext> = useCallback((index, imageName, ctx) => {
    return <ImageAtPosition index={index} imageName={imageName} queryArgs={ctx.queryArgs} />;
  }, []);

  if (isLoading) {
    return (
      <Flex height="100%" alignItems="center" justifyContent="center">
        <Spinner size="lg" opacity={SPINNER_OPACITY} />
        <Text ml={4}>Loading gallery...</Text>
      </Flex>
    );
  }

  if (imageNames.length === 0) {
    return (
      <Flex height="100%" alignItems="center" justifyContent="center">
        <Text color="base.300">No images found</Text>
      </Flex>
    );
  }

  return (
    <Box data-overlayscrollbars-initialize="" ref={rootRef} w="full" h="full">
      <VirtuosoGrid<string, GridContext>
        ref={virtuosoRef}
        context={context}
        totalCount={imageNames.length}
        data={imageNames}
        increaseViewportBy={VIEWPORT_BUFFER}
        itemContent={itemContent}
        computeItemKey={computeItemKey}
        components={components}
        style={style}
        scrollerRef={setScroller}
        scrollSeekConfiguration={scrollSeekConfiguration}
        rangeChanged={handleRangeChanged}
      />
    </Box>
  );
});

NewGallery.displayName = 'NewGallery';

const scrollSeekConfiguration: ScrollSeekConfiguration = {
  enter: (velocity) => velocity > SCROLL_SEEK_VELOCITY_THRESHOLD,
  exit: (velocity) => velocity === 0,
};

// Styles
const style = { height: '100%', width: '100%' };

const selectGridTemplateColumns = createSelector(
  selectGalleryImageMinimumWidth,
  (galleryImageMinimumWidth) => `repeat(auto-fill, minmax(${galleryImageMinimumWidth}px, 1fr))`
);

// Grid components
const ListComponent: GridComponents<GridContext>['List'] = forwardRef(({ context: _, ...rest }, ref) => {
  const gridTemplateColumns = useAppSelector(selectGridTemplateColumns);

  return <Grid ref={ref} gridTemplateColumns={gridTemplateColumns} gap={1} {...rest} />;
});
ListComponent.displayName = 'ListComponent';

const ItemComponent: GridComponents<GridContext>['Item'] = forwardRef(({ context: _, ...rest }, ref) => (
  <GridItem ref={ref} aspectRatio="1/1" {...rest} />
));
ItemComponent.displayName = 'ItemComponent';

const ScrollSeekPlaceholderComponent: GridComponents<GridContext>['ScrollSeekPlaceholder'] = forwardRef(
  (props, ref) => (
    <GridItem ref={ref} aspectRatio="1/1" {...props}>
      <Skeleton w="full" h="full" />
    </GridItem>
  )
);
ScrollSeekPlaceholderComponent.displayName = 'ScrollSeekPlaceholderComponent';

const components: GridComponents<GridContext> = {
  Item: ItemComponent,
  List: ListComponent,
  ScrollSeekPlaceholder: ScrollSeekPlaceholderComponent,
};

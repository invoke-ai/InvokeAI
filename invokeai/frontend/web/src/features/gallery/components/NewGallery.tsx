import { Box, Flex, forwardRef, Grid, GridItem, Skeleton, Spinner, Text } from '@invoke-ai/ui-library';
import { logger } from 'app/logging/logger';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectGalleryImageMinimumWidth,
  selectImageCollectionQueryArgs,
  selectLastSelectedImage,
} from 'features/gallery/store/gallerySelectors';
import { selectionChanged } from 'features/gallery/store/gallerySlice';
import { useOverlayScrollbars } from 'overlayscrollbars-react';
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
import {
  useGetImageCollectionCountsQuery,
  useGetImageCollectionQuery,
  useGetImageNamesQuery,
  useLazyGetImageCollectionQuery,
} from 'services/api/endpoints/images';
import type { ImageCategory, ImageDTO, SQLiteDirection } from 'services/api/types';
import { objectEntries } from 'tsafe';
import { useDebounce } from 'use-debounce';

import { GalleryImage } from './ImageGrid/GalleryImage';

const log = logger('gallery');

// Type for image collection query arguments
type ImageCollectionQueryArgs = {
  board_id?: string;
  categories?: ImageCategory[];
  search_term?: string;
  order_dir?: SQLiteDirection;
  is_intermediate: boolean;
};

// Constants
const RANGE_SIZE = 50;
const VIEWPORT_BUFFER = 2048;
const SCROLL_SEEK_VELOCITY_THRESHOLD = 2048;
const DEBOUNCE_DELAY = 500;
const GRID_GAP = 2;
const SPINNER_OPACITY = 0.3;

type GridContext = {
  queryArgs: ImageCollectionQueryArgs;
  imageNames: string[];
  starredCount: number;
};

type PositionInfo = {
  collection: 'starred' | 'unstarred';
  offset: number;
  itemIndex: number;
};

// Helper to calculate which collection and range an index belongs to
const getPositionInfo = (index: number, starredCount: number): PositionInfo => {
  if (index < starredCount) {
    // Starred collection
    const offset = Math.floor(index / RANGE_SIZE) * RANGE_SIZE;
    return {
      collection: 'starred',
      offset,
      itemIndex: index - offset,
    };
  } else {
    // Unstarred collection
    const unstarredIndex = index - starredCount;
    const offset = Math.floor(unstarredIndex / RANGE_SIZE) * RANGE_SIZE;
    return {
      collection: 'unstarred',
      offset,
      itemIndex: unstarredIndex - offset,
    };
  }
};

// Hook to get image DTO from batched collection data
const useImageFromBatch = (
  imageName: string,
  index: number,
  starredCount: number,
  queryArgs: ImageCollectionQueryArgs
): ImageDTO | null => {
  const { arg, options } = useMemo(() => {
    const positionInfo = getPositionInfo(index, starredCount);

    const arg = {
      collection: positionInfo.collection,
      offset: positionInfo.offset,
      limit: RANGE_SIZE,
      ...queryArgs,
    } satisfies Parameters<typeof useGetImageCollectionQuery>[0];

    const options = {
      selectFromResult: ({ data }) => {
        const imageDTO = data?.items?.[positionInfo.itemIndex] || null;
        if (imageDTO && imageDTO.image_name !== imageName) {
          log.warn(`Image name mismatch at index ${index}: expected ${imageName}, got ${imageDTO.image_name}`);
        }
        return { imageDTO };
      },
    } satisfies Parameters<typeof useGetImageCollectionQuery>[1];

    return { arg, options };
  }, [imageName, index, queryArgs, starredCount]);

  const { imageDTO } = useGetImageCollectionQuery(arg, options);

  return imageDTO;
};

// Individual image component that gets its data from batched requests
const ImageAtPosition = memo(
  ({
    imageName,
    index,
    starredCount,
    queryArgs,
  }: {
    imageName: string;
    index: number;
    starredCount: number;
    queryArgs: ImageCollectionQueryArgs;
  }) => {
    const imageDTO = useImageFromBatch(imageName, index, starredCount, queryArgs);

    if (!imageDTO) {
      return <Skeleton w="full" h="full" />;
    }

    return <GalleryImage imageDTO={imageDTO} />;
  }
);
ImageAtPosition.displayName = 'ImageAtPosition';

export const useDebouncedImageCollectionQueryArgs = () => {
  const _queryArgs = useAppSelector(selectImageCollectionQueryArgs);
  const [queryArgs] = useDebounce(_queryArgs, DEBOUNCE_DELAY);
  return queryArgs;
};

// Memoized item content function that uses image names as data but batches requests
const itemContent: GridItemContent<string, GridContext> = (index, imageName, { queryArgs, starredCount }) => {
  if (!imageName) {
    return <Skeleton w="full" h="full" />;
  }
  return <ImageAtPosition imageName={imageName} index={index} starredCount={starredCount} queryArgs={queryArgs} />;
};

// Memoized compute key function using image names
const computeItemKey: GridComputeItemKey<string, GridContext> = (index, imageName, { queryArgs }) => {
  return `${JSON.stringify(queryArgs)}-${imageName || index}`;
};

// Hook to prefetch ranges based on visible area
const usePrefetchRanges = (starredCount: number, queryArgs: ImageCollectionQueryArgs) => {
  const [triggerGetImageCollection] = useLazyGetImageCollectionQuery();

  const prefetchRange = useCallback(
    (startIndex: number, endIndex: number) => {
      const ranges = {
        starred: new Set<number>(),
        unstarred: new Set<number>(),
      };

      // Collect all unique ranges needed for the visible area
      for (let i = startIndex; i <= endIndex; i++) {
        const positionInfo = getPositionInfo(i, starredCount);
        ranges[positionInfo.collection].add(positionInfo.offset);
      }

      // Trigger queries for each unique range
      for (const [collection, offsets] of objectEntries(ranges)) {
        for (const offset of offsets) {
          triggerGetImageCollection({
            collection,
            offset,
            limit: RANGE_SIZE,
            ...queryArgs,
          });
        }
      }
    },
    [starredCount, queryArgs, triggerGetImageCollection]
  );

  return prefetchRange;
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
const isItemVisible = (index: number, rootEl: HTMLDivElement): null | 'start' | 'center' | 'end' => {
  // First get the virtuoso grid list root element
  const gridList = rootEl.querySelector('.virtuoso-grid-list') as HTMLElement;

  if (!gridList) {
    return null;
  }

  // Then find the specific item within the grid list
  const targetItem = gridList.querySelector(`.virtuoso-grid-item[data-index="${index}"]`) as HTMLElement;

  if (!targetItem) {
    return null;
  }

  const itemRect = targetItem.getBoundingClientRect();
  const rootRect = rootEl.getBoundingClientRect();

  if (itemRect.top < rootRect.top) {
    return 'start';
  }

  if (itemRect.bottom > rootRect.bottom) {
    return 'end';
  }

  return 'center';
};

// Hook for keyboard navigation using physical DOM measurements
const useKeyboardNavigation = (
  imageNames: string[],
  virtuosoRef: React.RefObject<VirtuosoGridHandle>,
  rootRef: React.RefObject<HTMLDivElement>
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
      if (!rootEl) {
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

          // Only scroll if the selected item is not visible
          const vis = isItemVisible(newIndex, rootEl);
          if (!vis || vis === 'center') {
            return;
          }
          virtuosoRef.current?.scrollToIndex({
            index: newIndex,
            behavior: 'smooth',
            align: vis,
          });
        }
      }
    },
    [rootRef, imageNames, currentIndex, dispatch, virtuosoRef]
  );

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [handleKeyDown]);
};

// Main gallery component
export const NewGallery = memo(() => {
  const queryArgs = useDebouncedImageCollectionQueryArgs();
  const virtuosoRef = useRef<VirtuosoGridHandle>(null);

  // Get the ordered list of image names - this is our primary data source
  const { data: imageNames = [], isLoading } = useGetImageNamesQuery(queryArgs);

  // Get starred count for position calculations
  const { data: counts } = useGetImageCollectionCountsQuery(queryArgs);
  const starredCount = counts?.starred_count ?? 0;

  const prefetchRange = usePrefetchRanges(starredCount, queryArgs);

  // Reset scroll position when query parameters change
  useEffect(() => {
    if (virtuosoRef.current && imageNames.length > 0) {
      virtuosoRef.current.scrollToIndex({ index: 0, behavior: 'auto' });
    }
  }, [queryArgs, imageNames.length]);

  const rootRef = useRef<HTMLDivElement>(null);

  // Enable keyboard navigation
  useKeyboardNavigation(imageNames, virtuosoRef, rootRef);

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

  // Handle range changes to prefetch data for visible + buffer areas
  const handleRangeChanged = useCallback(
    (range: ListRange) => {
      prefetchRange(range.startIndex, range.endIndex);
    },
    [prefetchRange]
  );

  const context = useMemo(
    () =>
      ({
        imageNames,
        queryArgs,
        starredCount,
      }) satisfies GridContext,
    [imageNames, queryArgs, starredCount]
  );

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
        overscan={VIEWPORT_BUFFER}
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

// Grid components
const ListComponent: GridComponents<GridContext>['List'] = forwardRef((props, ref) => {
  const galleryImageMinimumWidth = useAppSelector(selectGalleryImageMinimumWidth);

  return (
    <Grid
      ref={ref}
      gridTemplateColumns={`repeat(auto-fill, minmax(${galleryImageMinimumWidth}px, 1fr))`}
      gap={GRID_GAP}
      {...props}
    />
  );
});
ListComponent.displayName = 'ListComponent';

const ItemComponent: GridComponents<GridContext>['Item'] = forwardRef((props, ref) => (
  <GridItem ref={ref} aspectRatio="1/1" {...props} />
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

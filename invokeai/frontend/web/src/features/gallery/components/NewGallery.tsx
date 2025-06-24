import { Box, Flex, forwardRef, Grid, GridItem, Skeleton, Spinner, Text } from '@invoke-ai/ui-library';
import { logger } from 'app/logging/logger';
import { useAppSelector } from 'app/store/storeHooks';
import {
  selectGalleryImageMinimumWidth,
  selectImageCollectionQueryArgs,
} from 'features/gallery/store/gallerySelectors';
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
          log.warnOnce(`Image name mismatch at index ${index}: expected ${imageName}, got ${imageDTO.image_name}`);
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
  const [queryArgs] = useDebounce(_queryArgs, 500);
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
        <Spinner size="lg" opacity={0.3} />
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
        increaseViewportBy={2048}
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
  enter: (velocity) => velocity > 2048,
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
      gap={2}
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

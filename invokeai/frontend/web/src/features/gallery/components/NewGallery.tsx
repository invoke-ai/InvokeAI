import { Box, Flex, forwardRef, Grid, GridItem, Skeleton, Spinner, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import {
  selectGalleryImageMinimumWidth,
  selectImageCollectionQueryArgs,
} from 'features/gallery/store/gallerySelectors';
import { useOverlayScrollbars } from 'overlayscrollbars-react';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { GridComponents, ListRange, ScrollSeekConfiguration, VirtuosoGridHandle } from 'react-virtuoso';
import { VirtuosoGrid } from 'react-virtuoso';
import { useGetImageCollectionCountsQuery, useGetImageCollectionQuery } from 'services/api/endpoints/images';
import type { ImageCategory, SQLiteDirection } from 'services/api/types';
import { useDebounce } from 'use-debounce';

import { GalleryImage } from './ImageGrid/GalleryImage';

// Type for image collection query arguments
type ImageCollectionQueryArgs = {
  board_id?: string;
  categories?: ImageCategory[];
  search_term?: string;
  order_dir?: SQLiteDirection;
  is_intermediate: boolean;
};

// Types
type Collection = 'starred' | 'unstarred';

interface PositionInfo {
  collection: Collection;
  offset: number;
  itemIndex: number;
}

// Constants
const RANGE_SIZE = 50;

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

// Hook to get image at a specific position
const useImageAtPosition = (index: number, starredCount: number, queryArgs: ImageCollectionQueryArgs) => {
  const positionInfo = useMemo(() => getPositionInfo(index, starredCount), [index, starredCount]);

  const arg = useMemo(
    () =>
      ({
        collection: positionInfo.collection,
        offset: positionInfo.offset,
        limit: RANGE_SIZE,
        ...queryArgs,
      }) satisfies Parameters<typeof useGetImageCollectionQuery>[0],
    [positionInfo.collection, positionInfo.offset, queryArgs]
  );

  const options = useMemo(
    () =>
      ({
        selectFromResult: ({ data }) => {
          if (!data) {
            return { imageDTO: null };
          } else {
            return {
              imageDTO: data.items[positionInfo.itemIndex] || null,
            };
          }
        },
      }) satisfies Parameters<typeof useGetImageCollectionQuery>[1],
    [positionInfo.itemIndex]
  );

  const { imageDTO } = useGetImageCollectionQuery(arg, options);

  return imageDTO;
};

type ImageAtPositionProps = {
  index: number;
  starredCount: number;
  queryArgs: ImageCollectionQueryArgs;
};

// Individual image component
const ImageAtPosition = memo(({ index, starredCount, queryArgs }: ImageAtPositionProps) => {
  const imageDTO = useImageAtPosition(index, starredCount, queryArgs);

  if (!imageDTO) {
    return <Skeleton w="full" h="full" />;
  }

  return <GalleryImage imageDTO={imageDTO} />;
});

ImageAtPosition.displayName = 'ImageAtPosition';

export const useDebouncedImageCollectionQueryArgs = () => {
  const _queryArgs = useAppSelector(selectImageCollectionQueryArgs);
  const [queryArgs] = useDebounce(_queryArgs, 500);
  return queryArgs;
};

// Main gallery component
export const NewGallery = memo(() => {
  const queryArgs = useDebouncedImageCollectionQueryArgs();
  const virtuosoRef = useRef<VirtuosoGridHandle>(null);

  const { data: counts, isLoading } = useGetImageCollectionCountsQuery(queryArgs);

  const starredCount = counts?.starred_count ?? 0;
  const totalCount = counts?.total_count ?? 0;

  // Reset scroll position when query parameters change
  useEffect(() => {
    if (virtuosoRef.current && totalCount > 0) {
      virtuosoRef.current.scrollToIndex({ index: 0, behavior: 'auto' });
    }
  }, [queryArgs, totalCount]);

  // Memoized item content function
  const itemContent = useCallback(
    (index: number) => {
      return <ImageAtPosition index={index} starredCount={starredCount} queryArgs={queryArgs} />;
    },
    [starredCount, queryArgs]
  );

  // Memoized compute key function
  const computeItemKey = useCallback(
    (index: number) => {
      return `${JSON.stringify(queryArgs)}-${index}`;
    },
    [queryArgs]
  );

  // Handle range changes (for prefetching)
  const handleRangeChanged = useCallback((_range: ListRange) => {
    // RTK Query will automatically handle caching and deduplication
    // No need to manually trigger queries here
  }, []);

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

    return () => osInstance()?.destroy();
  }, [scroller, initialize, osInstance]);

  if (isLoading) {
    return (
      <Flex height="100%" alignItems="center" justifyContent="center">
        <Spinner size="lg" />
        <Text ml={4}>Loading gallery...</Text>
      </Flex>
    );
  }

  if (totalCount === 0) {
    return (
      <Flex height="100%" alignItems="center" justifyContent="center">
        <Text color="gray.500">No images found</Text>
      </Flex>
    );
  }

  return (
    <Box data-overlayscrollbars-initialize="" ref={rootRef} w="full" h="full">
      <VirtuosoGrid
        ref={virtuosoRef}
        totalCount={totalCount}
        increaseViewportBy={1024}
        rangeChanged={handleRangeChanged}
        itemContent={itemContent}
        computeItemKey={computeItemKey}
        components={components}
        style={style}
        scrollerRef={setScroller}
        scrollSeekConfiguration={scrollSeekConfiguration}
      />
    </Box>
  );
});

NewGallery.displayName = 'NewGallery';

const scrollSeekConfiguration: ScrollSeekConfiguration = {
  enter: (velocity) => {
    return velocity > 500;
  },
  exit: (velocity) => velocity < 500,
};

// Styles
const style = { height: '100%', width: '100%' };

// Grid components
const ListComponent: GridComponents['List'] = forwardRef((props, ref) => {
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

const ItemComponent: GridComponents['Item'] = forwardRef((props, ref) => (
  <GridItem ref={ref} aspectRatio="1/1" {...props} />
));
ItemComponent.displayName = 'ItemComponent';

const FillSkeleton: GridComponents['ScrollSeekPlaceholder'] = forwardRef((props, ref) => (
  <GridItem ref={ref} {...props}>
    <Skeleton w="full" h="full" />
  </GridItem>
));
FillSkeleton.displayName = 'FillSkeleton';

const components: GridComponents = {
  Item: ItemComponent,
  List: ListComponent,
  ScrollSeekPlaceholder: FillSkeleton,
};

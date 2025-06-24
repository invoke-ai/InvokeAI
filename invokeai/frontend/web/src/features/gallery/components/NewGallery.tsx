import { Box, Flex, forwardRef, Grid, GridItem, Skeleton, Spinner, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import {
  selectGalleryImageMinimumWidth,
  selectImageCollectionQueryArgs,
} from 'features/gallery/store/gallerySelectors';
import { useOverlayScrollbars } from 'overlayscrollbars-react';
import { memo, useEffect, useMemo, useRef, useState } from 'react';
import type {
  GridComponents,
  GridComputeItemKey,
  GridItemContent,
  ScrollSeekConfiguration,
  VirtuosoGridHandle,
} from 'react-virtuoso';
import { VirtuosoGrid } from 'react-virtuoso';
import {
  useGetImageCollectionCountsQuery,
  useGetImageCollectionQuery,
  useGetImageNamesQuery,
} from 'services/api/endpoints/images';
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

type GridContext = {
  queryArgs: ImageCollectionQueryArgs;
  counts: {
    starred_count: number;
    unstarred_count: number;
    total_count: number;
  };
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

const getImageCollectionCountsOptions = {
  selectFromResult: ({ data, isLoading }) => ({
    counts: data
      ? {
          starred_count: data.starred_count,
          unstarred_count: data.unstarred_count,
          total_count: data.starred_count + data.unstarred_count,
        }
      : {
          starred_count: 0,
          unstarred_count: 0,
          total_count: 0,
        },
    isLoading,
  }),
} satisfies Parameters<typeof useGetImageCollectionCountsQuery>[1];

// Memoized item content function
const itemContent: GridItemContent<null, GridContext> = (index, _item, { queryArgs, counts }) => {
  return <ImageAtPosition index={index} starredCount={counts.starred_count} queryArgs={queryArgs} />;
};

// Memoized compute key function
const computeItemKey: GridComputeItemKey<null, GridContext> = (index, _item, { queryArgs }) => {
  return `${JSON.stringify(queryArgs)}-${index}`;
};

// Main gallery component
export const NewGallery = memo(() => {
  const queryArgs = useDebouncedImageCollectionQueryArgs();
  const virtuosoRef = useRef<VirtuosoGridHandle>(null);

  const { counts, isLoading } = useGetImageCollectionCountsQuery(queryArgs, getImageCollectionCountsOptions);

  // Load image names for selection operations - this is lightweight and ensures
  // selection operations work even before image data is fully loaded
  useGetImageNamesQuery(queryArgs);

  // Reset scroll position when query parameters change
  useEffect(() => {
    if (virtuosoRef.current && counts.total_count > 0) {
      virtuosoRef.current.scrollToIndex({ index: 0, behavior: 'auto' });
    }
  }, [counts.total_count, queryArgs]);

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

  const context = useMemo(
    () =>
      ({
        counts,
        queryArgs,
      }) satisfies GridContext,
    [counts, queryArgs]
  );

  if (isLoading) {
    return (
      <Flex height="100%" alignItems="center" justifyContent="center">
        <Spinner size="lg" opacity={0.3} />
        <Text ml={4}>Loading gallery...</Text>
      </Flex>
    );
  }

  if (counts.total_count === 0) {
    return (
      <Flex height="100%" alignItems="center" justifyContent="center">
        <Text color="base.300">No images found</Text>
      </Flex>
    );
  }

  return (
    <Box data-overlayscrollbars-initialize="" ref={rootRef} w="full" h="full">
      <VirtuosoGrid<null, GridContext>
        ref={virtuosoRef}
        context={context}
        totalCount={counts.total_count}
        increaseViewportBy={1024}
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
  enter: (velocity) => velocity > 1000,
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

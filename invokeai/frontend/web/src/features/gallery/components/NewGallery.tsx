import { Box, Flex, forwardRef, Grid, GridItem, Image, Skeleton, Spinner, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectImageCollectionQueryArgs } from 'features/gallery/store/gallerySelectors';
import React, { memo, useCallback, useMemo, useState } from 'react';
import { VirtuosoGrid } from 'react-virtuoso';
import { useGetImageCollectionCountsQuery, useGetImageCollectionQuery } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';

// Placeholder image component for now
const ImagePlaceholder = memo(({ image }: { image: ImageDTO }) => (
  <Image src={image.thumbnail_url} w="full" h="full" objectFit="contain" />
));

ImagePlaceholder.displayName = 'ImagePlaceholder';

// Loading skeleton component
const ImageSkeleton = memo(() => <Skeleton w="full" h="full" />);

ImageSkeleton.displayName = 'ImageSkeleton';

// Hook to manage image data for virtual scrolling
const useVirtualImageData = () => {
  const queryArgs = useAppSelector(selectImageCollectionQueryArgs);

  // Get total counts for position mapping
  const { data: counts, isLoading: countsLoading } = useGetImageCollectionCountsQuery(queryArgs);

  // Cache for loaded image ranges
  const [loadedRanges, setLoadedRanges] = useState<Map<string, ImageDTO[]>>(new Map());

  // Calculate position mappings
  const positionInfo = useMemo(() => {
    if (!counts) {
      return null;
    }

    const result = {
      totalCount: counts.total_count,
      starredCount: counts.starred_count ?? 0,
      unstarredCount: counts.unstarred_count ?? 0,
      starredEnd: (counts.starred_count ?? 0) - 1,
    };

    return result;
  }, [counts]);

  // Clear cache when search parameters change
  React.useEffect(() => {
    setLoadedRanges(new Map());
  }, [queryArgs.board_id, queryArgs.search_term, queryArgs.categories]);

  // Return flag to indicate when search parameters have changed
  const searchParamsChanged = useMemo(() => queryArgs, [queryArgs]);

  // Function to generate cache key for a range
  const getRangeKey = useCallback((collection: 'starred' | 'unstarred', offset: number, limit: number) => {
    return `${collection}-${offset}-${limit}`;
  }, []);

  // Function to get images for a specific position range
  const getImagesForRange = useCallback(
    (startIndex: number, endIndex: number) => {
      if (!positionInfo) {
        return [];
      }

      const requestedImages: (ImageDTO | null)[] = new Array(endIndex - startIndex + 1).fill(null);
      const rangesToLoad: Array<{
        collection: 'starred' | 'unstarred';
        offset: number;
        limit: number;
        targetStartIndex: number;
      }> = [];

      for (let i = startIndex; i <= endIndex; i++) {
        const relativeIndex = i - startIndex;

        // Handle case where there are no starred images
        if (positionInfo.starredCount === 0 || i >= positionInfo.starredCount) {
          // This position is in the unstarred collection
          const unstarredOffset = i - positionInfo.starredCount;
          const rangeKey = getRangeKey('unstarred', Math.floor(unstarredOffset / 50) * 50, 50);
          const cachedRange = loadedRanges.get(rangeKey);

          if (cachedRange) {
            const imageIndex = unstarredOffset % 50;
            if (imageIndex < cachedRange.length) {
              requestedImages[relativeIndex] = cachedRange[imageIndex] ?? null;
            }
          } else {
            // Need to load this range
            const rangeOffset = Math.floor(unstarredOffset / 50) * 50;
            rangesToLoad.push({
              collection: 'unstarred',
              offset: rangeOffset,
              limit: 50,
              targetStartIndex: i,
            });
          }
        } else {
          // This position is in the starred collection
          const starredOffset = i;
          const rangeKey = getRangeKey('starred', Math.floor(starredOffset / 50) * 50, 50);
          const cachedRange = loadedRanges.get(rangeKey);

          if (cachedRange) {
            const imageIndex = starredOffset % 50;
            if (imageIndex < cachedRange.length) {
              requestedImages[relativeIndex] = cachedRange[imageIndex] ?? null;
            }
          } else {
            // Need to load this range
            const rangeOffset = Math.floor(starredOffset / 50) * 50;
            rangesToLoad.push({
              collection: 'starred',
              offset: rangeOffset,
              limit: 50,
              targetStartIndex: i,
            });
          }
        }
      }

      return { images: requestedImages, rangesToLoad };
    },
    [positionInfo, loadedRanges, getRangeKey]
  );

  return {
    positionInfo,
    countsLoading,
    getImagesForRange,
    setLoadedRanges,
    loadedRanges,
    searchParamsChanged,
  };
};

// Component to handle loading image ranges
const ImageRangeLoader = memo(
  ({
    collection,
    offset,
    limit,
    onDataLoaded,
  }: {
    collection: 'starred' | 'unstarred';
    offset: number;
    limit: number;
    onDataLoaded: (key: string, images: ImageDTO[]) => void;
  }) => {
    const queryArgs = useAppSelector(selectImageCollectionQueryArgs);

    const { data } = useGetImageCollectionQuery({
      collection,
      offset,
      limit,
      ...queryArgs,
    });

    // Update cache when data is loaded - use useEffect to avoid state update during render
    React.useEffect(() => {
      if (data?.items) {
        const key = `${collection}-${offset}-${limit}`;
        onDataLoaded(key, data.items);
      }
    }, [data, collection, offset, limit, onDataLoaded]);

    return null;
  }
);

ImageRangeLoader.displayName = 'ImageRangeLoader';

export const NewGallery = memo(() => {
  const { positionInfo, countsLoading, getImagesForRange, setLoadedRanges, searchParamsChanged } =
    useVirtualImageData();
  const [activeRangeLoaders, setActiveRangeLoaders] = useState<Set<string>>(new Set());

  // Force initial range loading when position info becomes available
  const [hasInitiallyLoaded, setHasInitiallyLoaded] = useState(false);

  // Reset hasInitiallyLoaded when search parameters change
  React.useEffect(() => {
    setHasInitiallyLoaded(false);
    setActiveRangeLoaders(new Set());
  }, [searchParamsChanged]);

  // Use useEffect for initial load to avoid state updates during render
  React.useEffect(() => {
    if (positionInfo && !hasInitiallyLoaded) {
      // Force initial load of first 100 positions to ensure we see both starred and unstarred
      const initialResult = getImagesForRange(0, Math.min(99, positionInfo.totalCount - 1));
      if (!Array.isArray(initialResult)) {
        const { rangesToLoad } = initialResult;
        rangesToLoad.forEach((rangeInfo) => {
          const key = `${rangeInfo.collection}-${rangeInfo.offset}-${rangeInfo.limit}`;
          if (!activeRangeLoaders.has(key)) {
            setActiveRangeLoaders((prev) => new Set(prev).add(key));
          }
        });
      }
      setHasInitiallyLoaded(true);
    }
  }, [positionInfo, hasInitiallyLoaded, getImagesForRange, activeRangeLoaders]);

  // Handle range changes from virtuoso
  const handleRangeChanged = useCallback(
    (range: { startIndex: number; endIndex: number }) => {
      if (!positionInfo) {
        return;
      }

      const result = getImagesForRange(range.startIndex, range.endIndex);
      if (!Array.isArray(result)) {
        const { rangesToLoad } = result;

        // Start loading any missing ranges
        rangesToLoad.forEach((rangeInfo) => {
          const key = `${rangeInfo.collection}-${rangeInfo.offset}-${rangeInfo.limit}`;
          if (!activeRangeLoaders.has(key)) {
            setActiveRangeLoaders((prev) => new Set(prev).add(key));
          }
        });
      }
    },
    [positionInfo, getImagesForRange, activeRangeLoaders]
  );

  // Handle when range data is loaded
  const handleDataLoaded = useCallback(
    (key: string, images: ImageDTO[]) => {
      setLoadedRanges((prev) => new Map(prev).set(key, images));
      setActiveRangeLoaders((prev) => {
        const next = new Set(prev);
        next.delete(key);
        return next;
      });
    },
    [setLoadedRanges]
  );

  const computeItemKey = useCallback(
    (index: number) => {
      const result = getImagesForRange(index, index);
      if (Array.isArray(result)) {
        return `loading-${index}`;
      }
      const { images } = result;
      const image = images[0];
      return image ? `image-${index}-${image.image_name}` : `skeleton-${index}`;
    },
    [getImagesForRange]
  );

  // Render item at specific index
  const itemContent = useCallback(
    (index: number) => {
      if (!positionInfo) {
        return <ImageSkeleton />;
      }

      const result = getImagesForRange(index, index);
      if (Array.isArray(result)) {
        return <ImageSkeleton />;
      }

      const { images } = result;
      const image = images[0];

      if (image) {
        return <ImagePlaceholder image={image} />;
      }

      return <ImageSkeleton />;
    },
    [positionInfo, getImagesForRange]
  );

  if (countsLoading) {
    return (
      <Flex height="100%" alignItems="center" justifyContent="center">
        <Spinner size="lg" />
        <Text ml={4}>Loading gallery...</Text>
      </Flex>
    );
  }

  if (!positionInfo || positionInfo.totalCount === 0) {
    return (
      <Flex height="100%" alignItems="center" justifyContent="center">
        <Text color="gray.500">No images found</Text>
      </Flex>
    );
  }

  return (
    <Box height="100%" width="100%">
      {/* Render active range loaders */}
      {Array.from(activeRangeLoaders).map((key) => {
        const [collection, offset, limit] = key.split('-');
        return (
          <ImageRangeLoader
            key={key}
            collection={collection as 'starred' | 'unstarred'}
            offset={parseInt(offset ?? '0', 10)}
            limit={parseInt(limit ?? '50', 10)}
            onDataLoaded={handleDataLoaded}
          />
        );
      })}

      {/* Virtualized grid */}
      <VirtuosoGrid
        totalCount={positionInfo.totalCount}
        overscan={200}
        rangeChanged={handleRangeChanged}
        itemContent={itemContent}
        style={style}
        computeItemKey={computeItemKey}
        components={components}
      />
    </Box>
  );
});

NewGallery.displayName = 'NewGallery';

const style = { height: '100%', width: '100%' };

const ListComponent = forwardRef((props, ref) => (
  <Grid ref={ref} gridTemplateColumns="repeat(auto-fill, minmax(64px, 1fr))" gap={2} padding={2} {...props} />
));

const ItemComponent = forwardRef((props, ref) => <GridItem ref={ref} aspectRatio="1/1" {...props} />);

const components = {
  Item: ItemComponent,
  List: ListComponent,
};

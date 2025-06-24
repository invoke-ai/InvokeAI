import { Box, Flex, forwardRef, Grid, GridItem, Image, Skeleton, Spinner, Text } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { selectImageCollectionQueryArgs } from 'features/gallery/store/gallerySelectors';
import { memo, useCallback, useMemo } from 'react';
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

// Hook to manage position calculations and image access
const useVirtualImageData = () => {
  const queryArgs = useAppSelector(selectImageCollectionQueryArgs);

  // Get total counts for position mapping
  const { data: counts, isLoading: countsLoading } = useGetImageCollectionCountsQuery(queryArgs);

  // Calculate position mappings
  const positionInfo = useMemo(() => {
    if (!counts) {
      return null;
    }

    return {
      totalCount: counts.total_count,
      starredCount: counts.starred_count ?? 0,
      unstarredCount: counts.unstarred_count ?? 0,
      starredEnd: (counts.starred_count ?? 0) - 1,
    };
  }, [counts]);

  // Function to get query params for a specific position
  const getQueryParamsForPosition = useCallback(
    (index: number) => {
      if (!positionInfo) {
        return null;
      }

      if (positionInfo.starredCount === 0 || index >= positionInfo.starredCount) {
        // This position is in the unstarred collection
        const unstarredOffset = index - positionInfo.starredCount;
        const rangeOffset = Math.floor(unstarredOffset / 50) * 50;
        return {
          collection: 'unstarred' as const,
          offset: rangeOffset,
          limit: 50,
          imageIndex: unstarredOffset % 50,
        };
      } else {
        // This position is in the starred collection
        const rangeOffset = Math.floor(index / 50) * 50;
        return {
          collection: 'starred' as const,
          offset: rangeOffset,
          limit: 50,
          imageIndex: index % 50,
        };
      }
    },
    [positionInfo]
  );

  return {
    positionInfo,
    countsLoading,
    getQueryParamsForPosition,
    queryArgs,
  };
};

// Hook to get image data for a specific position using RTK Query cache
const useImageAtPosition = (index: number) => {
  const { getQueryParamsForPosition, queryArgs } = useVirtualImageData();

  const queryParams = getQueryParamsForPosition(index);

  const { data } = useGetImageCollectionQuery(
    queryParams
      ? {
          collection: queryParams.collection,
          offset: queryParams.offset,
          limit: queryParams.limit,
          ...queryArgs,
        }
      : skipToken
  );

  if (!queryParams || !data?.items) {
    return null;
  }

  return data.items[queryParams.imageIndex] || null;
};

// Component to render a single image at a position
const ImageAtPosition = memo(({ index }: { index: number }) => {
  const image = useImageAtPosition(index);

  if (image) {
    return <ImagePlaceholder image={image} />;
  }

  return <ImageSkeleton />;
});

ImageAtPosition.displayName = 'ImageAtPosition';

export const NewGallery = memo(() => {
  const { positionInfo, countsLoading } = useVirtualImageData();

  // Render item at specific index
  const itemContent = useCallback((index: number) => {
    return <ImageAtPosition index={index} />;
  }, []);

  // Compute item key using position index - let RTK Query handle the caching
  const computeItemKey = useCallback((index: number) => `position-${index}`, []);

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
      {/* Virtualized grid */}
      <VirtuosoGrid
        totalCount={positionInfo.totalCount}
        overscan={200}
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

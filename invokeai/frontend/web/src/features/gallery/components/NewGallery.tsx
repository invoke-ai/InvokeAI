import { Box, Flex, forwardRef, Grid, GridItem, Image, Skeleton, Spinner, Text } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import {
  selectGalleryImageMinimumWidth,
  selectImageCollectionQueryArgs,
} from 'features/gallery/store/gallerySelectors';
import { memo, useCallback } from 'react';
import { VirtuosoGrid } from 'react-virtuoso';
import {
  useGetImageCollectionCountsQuery,
  useGetImageCollectionQuery,
  useLazyGetImageCollectionQuery,
} from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';

// Types for range management
type Collection = 'starred' | 'unstarred';

interface RangeKey {
  collection: Collection;
  offset: number;
  limit: number;
}

interface PositionQuery extends RangeKey {
  imageIndex: number;
}

type PositionInfo = {
  totalCount: number;
  starredCount: number;
  unstarredCount: number;
  starredEnd: number;
};

// Query options factory functions to prevent recreation on every render
const countsQueryOptions = {
  selectFromResult: ({ data, isLoading }) => {
    const positionInfo: PositionInfo | null = data
      ? {
          totalCount: data.total_count ?? 0,
          starredCount: data.starred_count ?? 0,
          unstarredCount: data.unstarred_count ?? 0,
          starredEnd: (data.starred_count ?? 0) - 1,
        }
      : null;

    return {
      positionInfo,
      isLoading,
    };
  },
} satisfies Parameters<typeof useGetImageCollectionCountsQuery>[1];

const createImageCollectionQueryOptions = (queryParams: PositionQuery | null) =>
  ({
    skip: !queryParams,
    selectFromResult: (result) => {
      return {
        imageDTO: (queryParams && result.data?.items?.[queryParams.imageIndex]) || null,
      };
    },
  }) satisfies Parameters<typeof useGetImageCollectionQuery>[1];

// Placeholder image component for now
const ImagePlaceholder = memo(({ imageDTO }: { imageDTO: ImageDTO }) => (
  <Image src={imageDTO.thumbnail_url} w="full" h="full" objectFit="contain" />
));

ImagePlaceholder.displayName = 'ImagePlaceholder';

// Loading skeleton component
const ImageSkeleton = memo(() => <Skeleton w="full" h="full" />);

ImageSkeleton.displayName = 'ImageSkeleton';

// Hook to manage position calculations and range loading
const useVirtualImageData = () => {
  const queryArgs = useAppSelector(selectImageCollectionQueryArgs);

  // Get position info derived from counts using selectFromResult
  const { positionInfo, isLoading } = useGetImageCollectionCountsQuery(queryArgs, countsQueryOptions);

  const [triggerGetImageCollection] = useLazyGetImageCollectionQuery();

  // Function to get query params for a specific position
  const getQueryParamsForPosition = useCallback(
    (index: number): PositionQuery | null => {
      if (!positionInfo) {
        return null;
      }

      if (positionInfo.starredCount === 0 || index >= positionInfo.starredCount) {
        // This position is in the unstarred collection
        const unstarredOffset = index - positionInfo.starredCount;
        const rangeOffset = Math.floor(unstarredOffset / 50) * 50;
        return {
          collection: 'unstarred',
          offset: rangeOffset,
          limit: 50,
          imageIndex: unstarredOffset % 50,
        };
      } else {
        // This position is in the starred collection
        const rangeOffset = Math.floor(index / 50) * 50;
        return {
          collection: 'starred',
          offset: rangeOffset,
          limit: 50,
          imageIndex: index % 50,
        };
      }
    },
    [positionInfo]
  );

  // Function to calculate required ranges for a viewport and trigger lazy queries
  const updateRequiredRanges = useCallback(
    (startIndex: number, endIndex: number) => {
      if (!positionInfo) {
        return;
      }

      for (let i = startIndex; i <= endIndex; i++) {
        const queryParams = getQueryParamsForPosition(i);
        if (queryParams) {
          const { collection, offset, limit } = queryParams;
          triggerGetImageCollection(
            {
              collection,
              offset,
              limit,
              ...queryArgs,
            },
            true
          );
        }
      }
    },
    [positionInfo, getQueryParamsForPosition, triggerGetImageCollection, queryArgs]
  );

  return {
    positionInfo,
    isLoading,
    getQueryParamsForPosition,
    queryArgs,
    updateRequiredRanges,
  };
};

// Hook to get image data for a specific position using selectFromResult
const useImageAtPosition = (index: number) => {
  const { getQueryParamsForPosition, queryArgs } = useVirtualImageData();

  const queryParams = getQueryParamsForPosition(index);

  const { imageDTO } = useGetImageCollectionQuery(
    queryParams
      ? {
          collection: queryParams.collection,
          offset: queryParams.offset,
          limit: queryParams.limit,
          ...queryArgs,
        }
      : skipToken,
    createImageCollectionQueryOptions(queryParams)
  );

  return imageDTO;
};

// Component to render a single image at a position
const ImageAtPosition = memo(({ index }: { index: number }) => {
  const imageDTO = useImageAtPosition(index);

  if (imageDTO) {
    return <ImagePlaceholder imageDTO={imageDTO} />;
  }

  return <ImageSkeleton />;
});

ImageAtPosition.displayName = 'ImageAtPosition';

export const NewGallery = memo(() => {
  const { positionInfo, isLoading, updateRequiredRanges } = useVirtualImageData();

  // Handle range changes from VirtuosoGrid
  const handleRangeChanged = useCallback(
    (range: { startIndex: number; endIndex: number }) => {
      updateRequiredRanges(range.startIndex, range.endIndex);
    },
    [updateRequiredRanges]
  );

  // Render item at specific index
  const itemContent = useCallback((index: number) => {
    return <ImageAtPosition index={index} />;
  }, []);

  // Compute item key using position index - let RTK Query handle the caching
  const computeItemKey = useCallback((index: number) => `position-${index}`, []);

  if (isLoading) {
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
      <VirtuosoGrid
        totalCount={positionInfo.totalCount}
        increaseViewportBy={1024}
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

const ListComponent = forwardRef((props, ref) => {
  const galleryImageMinimumWidth = useAppSelector(selectGalleryImageMinimumWidth);

  return (
    <Grid
      ref={ref}
      gridTemplateColumns={`repeat(auto-fill, minmax(${galleryImageMinimumWidth}px, 1fr))`}
      gap={2}
      padding={2}
      {...props}
    />
  );
});

const ItemComponent = forwardRef((props, ref) => <GridItem ref={ref} aspectRatio="1/1" {...props} />);

const components = {
  Item: ItemComponent,
  List: ListComponent,
};

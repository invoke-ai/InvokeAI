import {
  Box,
  Flex,
  FlexProps,
  Grid,
  Skeleton,
  Spinner,
  forwardRef,
} from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import { IMAGE_LIMIT } from 'features/gallery/store/gallerySlice';
import { useOverlayScrollbars } from 'overlayscrollbars-react';

import {
  PropsWithChildren,
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { useTranslation } from 'react-i18next';
import { FaImage } from 'react-icons/fa';
import GalleryImage from './GalleryImage';

import { createSelector } from '@reduxjs/toolkit';
import { RootState, stateSelector } from 'app/store/store';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { selectFilteredImages } from 'features/gallery/store/gallerySelectors';
import { VirtuosoGrid } from 'react-virtuoso';
import { useListAllBoardsQuery } from 'services/api/endpoints/boards';
import { receivedPageOfImages } from 'services/api/thunks/image';
import { ImageDTO } from 'services/api/types';

const selector = createSelector(
  [stateSelector, selectFilteredImages],
  (state, filteredImages) => {
    const {
      categories,
      total: allImagesTotal,
      isLoading,
      isFetching,
      selectedBoardId,
    } = state.gallery;

    let images = filteredImages as (ImageDTO | 'loading')[];

    if (!isLoading && isFetching) {
      // loading, not not the initial load
      images = images.concat(Array(IMAGE_LIMIT).fill('loading'));
    }

    return {
      images,
      allImagesTotal,
      isLoading,
      isFetching,
      categories,
      selectedBoardId,
    };
  },
  defaultSelectorOptions
);

const ImageGalleryGrid = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const rootRef = useRef(null);
  const [scroller, setScroller] = useState<HTMLElement | null>(null);
  const [initialize, osInstance] = useOverlayScrollbars({
    defer: true,
    options: {
      scrollbars: {
        visibility: 'auto',
        autoHide: 'leave',
        autoHideDelay: 1300,
        theme: 'os-theme-dark',
      },
      overflow: { x: 'hidden' },
    },
  });

  const {
    images,
    isLoading,
    isFetching,
    allImagesTotal,
    categories,
    selectedBoardId,
  } = useAppSelector(selector);

  const { selectedBoard } = useListAllBoardsQuery(undefined, {
    selectFromResult: ({ data }) => ({
      selectedBoard: data?.find((b) => b.board_id === selectedBoardId),
    }),
  });

  const filteredImagesTotal = useMemo(
    () => selectedBoard?.image_count ?? allImagesTotal,
    [allImagesTotal, selectedBoard?.image_count]
  );

  const areMoreAvailable = useMemo(() => {
    return images.length < filteredImagesTotal;
  }, [images.length, filteredImagesTotal]);

  const handleLoadMoreImages = useCallback(() => {
    dispatch(
      receivedPageOfImages({
        categories,
        board_id: selectedBoardId,
        is_intermediate: false,
      })
    );
  }, [categories, dispatch, selectedBoardId]);

  const handleEndReached = useMemo(() => {
    if (areMoreAvailable && !isLoading) {
      return handleLoadMoreImages;
    }
    return undefined;
  }, [areMoreAvailable, handleLoadMoreImages, isLoading]);

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
      <Flex
        sx={{
          w: 'full',
          h: 'full',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <Spinner
          size="xl"
          sx={{ color: 'base.300', _dark: { color: 'base.700' } }}
        />
      </Flex>
    );
  }

  if (images.length) {
    return (
      <>
        <Box ref={rootRef} data-overlayscrollbars="" h="100%">
          <VirtuosoGrid
            style={{ height: '100%' }}
            data={images}
            endReached={handleEndReached}
            components={{
              Item: ItemContainer,
              List: ListContainer,
            }}
            scrollerRef={setScroller}
            itemContent={(index, item) =>
              typeof item === 'string' ? (
                <Skeleton sx={{ w: 'full', h: 'full', aspectRatio: '1/1' }} />
              ) : (
                <GalleryImage
                  key={`${item.image_name}-${item.thumbnail_url}`}
                  imageDTO={item}
                />
              )
            }
          />
        </Box>
        <IAIButton
          onClick={handleLoadMoreImages}
          isDisabled={!areMoreAvailable}
          isLoading={isFetching}
          loadingText="Loading"
          flexShrink={0}
        >
          {areMoreAvailable
            ? t('gallery.loadMore')
            : t('gallery.allImagesLoaded')}
        </IAIButton>
      </>
    );
  }

  return (
    <IAINoContentFallback
      label={t('gallery.noImagesInGallery')}
      icon={FaImage}
    />
  );
};

type ItemContainerProps = PropsWithChildren & FlexProps;
const ItemContainer = forwardRef((props: ItemContainerProps, ref) => (
  <Box className="item-container" ref={ref} p={1.5}>
    {props.children}
  </Box>
));

type ListContainerProps = PropsWithChildren & FlexProps;
const ListContainer = forwardRef((props: ListContainerProps, ref) => {
  const galleryImageMinimumWidth = useAppSelector(
    (state: RootState) => state.gallery.galleryImageMinimumWidth
  );

  return (
    <Grid
      {...props}
      className="list-container"
      ref={ref}
      sx={{
        gridTemplateColumns: `repeat(auto-fill, minmax(${galleryImageMinimumWidth}px, 1fr));`,
      }}
    >
      {props.children}
    </Grid>
  );
});

export default memo(ImageGalleryGrid);

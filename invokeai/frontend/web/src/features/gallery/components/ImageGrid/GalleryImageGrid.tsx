import { Box, Spinner } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { skipToken } from '@reduxjs/toolkit/dist/query';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIButton from 'common/components/IAIButton';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import {
  ASSETS_CATEGORIES,
  IMAGE_CATEGORIES,
  IMAGE_LIMIT,
} from 'features/gallery//store/gallerySlice';
import { selectFilteredImages } from 'features/gallery/store/gallerySelectors';
import {
  UseOverlayScrollbarsParams,
  useOverlayScrollbars,
} from 'overlayscrollbars-react';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { FaExclamationCircle, FaImage } from 'react-icons/fa';
import { VirtuosoGrid } from 'react-virtuoso';
import {
  useLazyListImagesQuery,
  useListImagesQuery,
} from 'services/api/endpoints/images';
import GalleryImage from './GalleryImage';
import ImageGridItemContainer from './ImageGridItemContainer';
import ImageGridListContainer from './ImageGridListContainer';

const selector = createSelector(
  [stateSelector, selectFilteredImages],
  (state, filteredImages) => {
    const {
      galleryImageMinimumWidth,
      selectedBoardId,
      galleryView,
      total,
      isLoading,
    } = state.gallery;

    return {
      imageNames: filteredImages.map((i) => i.image_name),
      total,
      selectedBoardId,
      galleryView,
      galleryImageMinimumWidth,
      isLoading,
    };
  },
  defaultSelectorOptions
);

const overlayScrollbarsConfig: UseOverlayScrollbarsParams = {
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
};

const GalleryImageGrid = () => {
  const { t } = useTranslation();
  const rootRef = useRef<HTMLDivElement>(null);
  const emptyGalleryRef = useRef<HTMLDivElement>(null);
  const [scroller, setScroller] = useState<HTMLElement | null>(null);
  const [initialize, osInstance] = useOverlayScrollbars(
    overlayScrollbarsConfig
  );

  const { galleryImageMinimumWidth, selectedBoardId, galleryView } =
    useAppSelector(selector);

  const [initialImageCount, setInitialImageCount] = useState(0);

  const [listImages] = useLazyListImagesQuery();

  const listImagesBaseArgs = useMemo(
    () => ({
      categories:
        galleryView === 'images' ? IMAGE_CATEGORIES : ASSETS_CATEGORIES,
      board_id: selectedBoardId === 'all' ? undefined : selectedBoardId,
      offset: 0,
      limit: initialImageCount,
      is_intermediate: false,
    }),
    [galleryView, initialImageCount, selectedBoardId]
  );

  const { currentData, isFetching, isSuccess, isError } = useListImagesQuery(
    initialImageCount ? listImagesBaseArgs : skipToken
  );

  const areMoreAvailable = useMemo(() => {
    if (!currentData) {
      return false;
    }
    return currentData.total > currentData.ids.length;
  }, [currentData]);

  const handleLoadMoreImages = useCallback(() => {
    listImages({
      ...listImagesBaseArgs,
      offset: currentData?.ids.length ?? 0,
      limit: IMAGE_LIMIT,
    });
  }, [listImages, listImagesBaseArgs, currentData?.ids.length]);

  useEffect(() => {
    // Initialize the gallery's custom scrollbar
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

  useEffect(() => {
    // rough calculation of how many images will fill up the gallery
    const galleryHeight = emptyGalleryRef.current?.clientHeight ?? 0;
    const galleryWidth = emptyGalleryRef.current?.clientHeight ?? 0;

    const rows = galleryHeight / galleryImageMinimumWidth;
    const columns = galleryWidth / galleryImageMinimumWidth;

    const count = Math.ceil(rows * columns);

    if (count === 0) {
      // in case there is a rendering issue, default to 100 images
      setInitialImageCount(100);
    } else {
      setInitialImageCount(count);
    }
  }, [galleryImageMinimumWidth]);

  if (!currentData) {
    return (
      <Box
        id="emptyGalleryRef"
        ref={emptyGalleryRef}
        sx={{ w: 'full', h: 'full' }}
      >
        <Spinner size="2xl" opacity={0.5} />
      </Box>
    );
  }

  if (isSuccess && currentData?.ids.length === 0) {
    return (
      <Box
        id="emptyGalleryRef"
        ref={emptyGalleryRef}
        sx={{ w: 'full', h: 'full' }}
      >
        <IAINoContentFallback
          label={t('gallery.noImagesInGallery')}
          icon={FaImage}
        />
      </Box>
    );
  }

  if (isSuccess && currentData) {
    return (
      <>
        <Box ref={rootRef} data-overlayscrollbars="" h="100%">
          <VirtuosoGrid
            style={{ height: '100%' }}
            data={currentData.ids}
            endReached={handleLoadMoreImages}
            components={{
              Item: ImageGridItemContainer,
              List: ImageGridListContainer,
            }}
            scrollerRef={setScroller}
            itemContent={(index, imageName) => (
              <GalleryImage key={imageName} imageName={imageName as string} />
            )}
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

  if (isError) {
    return (
      <Box ref={emptyGalleryRef} sx={{ w: 'full', h: 'full' }}>
        <IAINoContentFallback
          label="Unable to load Gallery"
          icon={FaExclamationCircle}
        />
      </Box>
    );
  }
};

export default memo(GalleryImageGrid);

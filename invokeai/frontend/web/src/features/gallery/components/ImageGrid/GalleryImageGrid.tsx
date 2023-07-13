import { Box } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import { useOverlayScrollbars } from 'overlayscrollbars-react';

import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { FaImage } from 'react-icons/fa';
import GalleryImage from './GalleryImage';

import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { IMAGE_LIMIT } from 'features/gallery//store/gallerySlice';
import { selectFilteredImages } from 'features/gallery/store/gallerySelectors';
import { VirtuosoGrid } from 'react-virtuoso';
import { receivedPageOfImages } from 'services/api/thunks/image';
import ImageGridItemContainer from './ImageGridItemContainer';
import ImageGridListContainer from './ImageGridListContainer';

const selector = createSelector(
  [stateSelector, selectFilteredImages],
  (state, filteredImages) => {
    const { galleryImageMinimumWidth, selectedBoardId, galleryView, total } =
      state.gallery;

    return {
      imageNames: filteredImages.map((i) => i.image_name),
      areMoreAvailable: total > filteredImages.length,
      selectedBoardId,
      galleryView,
      galleryImageMinimumWidth,
    };
  },
  defaultSelectorOptions
);

const GalleryImageGrid = () => {
  const { t } = useTranslation();
  const rootRef = useRef<HTMLDivElement>(null);
  const emptyGalleryRef = useRef<HTMLDivElement>(null);
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

  const [didInitialFetch, setDidInitialFetch] = useState(false);

  const dispatch = useAppDispatch();

  const {
    galleryImageMinimumWidth,
    imageNames,
    selectedBoardId,
    galleryView,
    areMoreAvailable,
  } = useAppSelector(selector);

  const handleLoadMoreImages = useCallback(() => {
    dispatch(
      receivedPageOfImages({
        offset: imageNames.length,
        limit: IMAGE_LIMIT,
      })
    );
  }, [dispatch, imageNames.length]);

  const handleEndReached = useMemo(() => {
    if (areMoreAvailable) {
      return handleLoadMoreImages;
    }
    return undefined;
  }, [areMoreAvailable, handleLoadMoreImages]);

  useEffect(() => {
    // Set up gallery scroler
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
    if (!didInitialFetch) {
      return;
    }
    // rough, conservative calculation of how many images fit in the gallery
    // TODO: this gets an incorrect value on first load...
    const galleryHeight = rootRef.current?.clientHeight ?? 0;
    const galleryWidth = rootRef.current?.clientHeight ?? 0;

    const rows = galleryHeight / galleryImageMinimumWidth;
    const columns = galleryWidth / galleryImageMinimumWidth;

    const imagesToLoad = Math.ceil(rows * columns);

    setDidInitialFetch(true);

    // load up that many images
    dispatch(
      receivedPageOfImages({
        offset: 0,
        limit: imagesToLoad,
      })
    );
  }, [
    didInitialFetch,
    dispatch,
    galleryImageMinimumWidth,
    galleryView,
    selectedBoardId,
  ]);

  if (status === 'fulfilled' && imageNames.length === 0) {
    return (
      <Box ref={emptyGalleryRef} sx={{ w: 'full', h: 'full' }}>
        <IAINoContentFallback
          label={t('gallery.noImagesInGallery')}
          icon={FaImage}
        />
      </Box>
    );
  }

  if (status !== 'rejected') {
    return (
      <>
        <Box ref={rootRef} data-overlayscrollbars="" h="100%">
          <VirtuosoGrid
            style={{ height: '100%' }}
            data={imageNames}
            endReached={handleEndReached}
            components={{
              Item: ImageGridItemContainer,
              List: ImageGridListContainer,
            }}
            scrollerRef={setScroller}
            itemContent={(index, imageName) => (
              <GalleryImage key={imageName} imageName={imageName} />
            )}
          />
        </Box>
        <IAIButton
          onClick={handleLoadMoreImages}
          isDisabled={!areMoreAvailable}
          isLoading={status === 'pending'}
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
};

export default memo(GalleryImageGrid);

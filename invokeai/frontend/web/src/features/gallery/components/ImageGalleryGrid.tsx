import { Box } from '@chakra-ui/react';
import { useAppSelector } from 'app/store/storeHooks';
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
import { VirtuosoGrid } from 'react-virtuoso';
import { useLoadMoreImages } from '../hooks/useLoadMoreImages';
import ItemContainer from './ItemContainer';
import ListContainer from './ListContainer';

const selector = createSelector(
  [stateSelector],
  (state) => {
    const { galleryImageMinimumWidth } = state.gallery;

    return {
      galleryImageMinimumWidth,
    };
  },
  defaultSelectorOptions
);

const ImageGalleryGrid = () => {
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

  const { galleryImageMinimumWidth } = useAppSelector(selector);

  const {
    imageNames,
    galleryView,
    loadMoreImages,
    selectedBoardId,
    status,
    areMoreAvailable,
  } = useLoadMoreImages();

  const handleLoadMoreImages = useCallback(() => {
    loadMoreImages({});
  }, [loadMoreImages]);

  const handleEndReached = useMemo(() => {
    if (areMoreAvailable && status !== 'pending') {
      return handleLoadMoreImages;
    }
    return undefined;
  }, [areMoreAvailable, handleLoadMoreImages, status]);

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

  useEffect(() => {
    // TODO: this doesn't actually prevent 2 intial image loads...
    if (status !== undefined) {
      return;
    }

    // rough, conservative calculation of how many images fit in the gallery
    // TODO: this gets an incorrect value on first load...
    const galleryHeight = rootRef.current?.clientHeight ?? 0;
    const galleryWidth = rootRef.current?.clientHeight ?? 0;

    const rows = galleryHeight / galleryImageMinimumWidth;
    const columns = galleryWidth / galleryImageMinimumWidth;

    const imagesToLoad = Math.ceil(rows * columns);

    // load up that many images
    loadMoreImages({
      offset: 0,
      limit: imagesToLoad,
    });
  }, [
    galleryImageMinimumWidth,
    galleryView,
    loadMoreImages,
    selectedBoardId,
    status,
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
              Item: ItemContainer,
              List: ListContainer,
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

export default memo(ImageGalleryGrid);

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
import {
  IMAGE_LIMIT,
  selectImagesAll,
} from 'features/gallery//store/gallerySlice';
import { selectFilteredImages } from 'features/gallery/store/gallerySelectors';
import { VirtuosoGrid } from 'react-virtuoso';
import { receivedPageOfImages } from 'services/api/thunks/image';
import ImageGridItemContainer from './ImageGridItemContainer';
import ImageGridListContainer from './ImageGridListContainer';
import { useListBoardImagesQuery } from '../../../../services/api/endpoints/boardImages';

const selector = createSelector(
  [stateSelector, selectImagesAll],
  (state, images) => {
    const {
      galleryImageMinimumWidth,
      selectedBoardId,
      galleryView,
      total,
      isLoading,
    } = state.gallery;

    return {
      imageNames: images.map((i) => i.image_name),
      total,
      selectedBoardId,
      galleryView,
      galleryImageMinimumWidth,
      isLoading,
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
    imageNames: imageNamesAll, //all images names loaded on main tab,
    total: totalAll,
    selectedBoardId,
    galleryView,
    isLoading: isLoadingAll,
  } = useAppSelector(selector);

  const { data: imagesForBoard, isLoading: isLoadingImagesForBoard } =
    useListBoardImagesQuery(
      { board_id: selectedBoardId },
      { skip: selectedBoardId === 'all' }
    );

  const imageNames =
    selectedBoardId === 'all'
      ? imageNamesAll
      : (imagesForBoard?.items || []).map((img) => img.image_name);
  const areMoreAvailable =
    selectedBoardId === 'all' ? totalAll > imageNamesAll.length : false;
  const isLoading =
    selectedBoardId === 'all' ? isLoadingAll : isLoadingImagesForBoard;

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
        limit: 10,
      })
    );
  }, [
    didInitialFetch,
    dispatch,
    galleryImageMinimumWidth,
    galleryView,
    selectedBoardId,
  ]);

  if (!isLoading && imageNames.length === 0) {
    return (
      <Box ref={emptyGalleryRef} sx={{ w: 'full', h: 'full' }}>
        <IAINoContentFallback
          label={t('gallery.noImagesInGallery')}
          icon={FaImage}
        />
      </Box>
    );
  }
  console.log({ selectedBoardId });

  if (status !== 'rejected') {
    return (
      <>
        <Box ref={rootRef} data-overlayscrollbars="" h="100%">
          <VirtuosoGrid
            style={{ height: '100%' }}
            data={imageNames}
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

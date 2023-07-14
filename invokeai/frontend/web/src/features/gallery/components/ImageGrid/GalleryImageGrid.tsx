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
  ASSETS_CATEGORIES,
  IMAGE_CATEGORIES,
  IMAGE_LIMIT,
  INITIAL_IMAGE_LIMIT,
} from 'features/gallery//store/gallerySlice';
import { VirtuosoGrid } from 'react-virtuoso';
import { receivedPageOfImages } from 'services/api/thunks/image';
import ImageGridItemContainer from './ImageGridItemContainer';
import ImageGridListContainer from './ImageGridListContainer';
import {
  imagesApi,
  useListImagesQuery,
} from '../../../../services/api/endpoints/images';
import { ImageDTO } from '../../../../services/api/types';

const selector = createSelector(
  [stateSelector],
  (state) => {
    const { selectedBoardId, galleryView } = state.gallery;

    return {
      selectedBoardId,
      galleryView,
    };
  },
  defaultSelectorOptions
);

const GalleryImageGrid = () => {
  const { t } = useTranslation();
  const rootRef = useRef<HTMLDivElement>(null);
  const emptyGalleryRef = useRef<HTMLDivElement>(null);
  const [scroller, setScroller] = useState<HTMLElement | null>(null);
  const [offset, setOffset] = useState(0);
  const [limit, setLimit] = useState(INITIAL_IMAGE_LIMIT);
  const [imageList, setImageList] = useState<ImageDTO[]>([]);
  const [isLoadingMore, setIsLoadingMore] = useState(true);
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

  const dispatch = useAppDispatch();

  const { selectedBoardId, galleryView } = useAppSelector(selector);

  useEffect(() => {
    setImageList([]);
    setOffset(0);
    setLimit(INITIAL_IMAGE_LIMIT);
  }, [selectedBoardId]);

  const { data: imageListResponse, isLoading: isLoading } = useListImagesQuery({
    categories: galleryView === 'images' ? IMAGE_CATEGORIES : ASSETS_CATEGORIES,
    is_intermediate: false,
    offset: 0,
    limit: INITIAL_IMAGE_LIMIT,
    ...(selectedBoardId === 'all' ? {} : { board_id: selectedBoardId }),
  });

  const { data: paginatedData } = useListImagesQuery(
    {
      categories:
        galleryView === 'images' ? IMAGE_CATEGORIES : ASSETS_CATEGORIES,
      is_intermediate: false,
      offset,
      limit,
      ...(selectedBoardId === 'all' ? {} : { board_id: selectedBoardId }),
    },
    { skip: offset === 0 }
  );

  useEffect(() => {
    if (imageListResponse) setImageList(imageListResponse.items);
  }, [imageListResponse]);

  useEffect(() => {
    if (paginatedData) {
      dispatch(
        imagesApi.util.updateQueryData(
          'listImages',
          {
            categories:
              galleryView === 'images' ? IMAGE_CATEGORIES : ASSETS_CATEGORIES,
            is_intermediate: false,
            offset: 0,
            limit: INITIAL_IMAGE_LIMIT,
            ...(selectedBoardId === 'all' ? {} : { board_id: selectedBoardId }),
          },
          (draftPosts) => {
            paginatedData.items.forEach((item) => {
              draftPosts.items.push(item);
            });
          }
        )
      );
    }
    //eslint-disable-next-line
  }, [paginatedData, dispatch]);

  const areMoreAvailable = useMemo(() => {
    if (imageListResponse?.total) {
      return imageListResponse?.total > imageList.length;
    }
  }, [imageListResponse?.total, imageList.length]);

  const handleLoadMoreImages = useCallback(() => {
    setOffset(imageList.length);
    setLimit(IMAGE_LIMIT);
  }, [imageList.length]);

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

  if (!isLoading && imageList.length === 0) {
    return (
      <Box ref={emptyGalleryRef} sx={{ w: 'full', h: 'full' }}>
        <IAINoContentFallback
          label={t('gallery.noImagesInGallery')}
          icon={FaImage}
        />
      </Box>
    );
  }

  if (!isLoading) {
    return (
      <>
        <Box ref={rootRef} data-overlayscrollbars="" h="100%">
          <VirtuosoGrid
            style={{ height: '100%' }}
            data={imageList}
            components={{
              Item: ImageGridItemContainer,
              List: ImageGridListContainer,
            }}
            scrollerRef={setScroller}
            itemContent={(index, image) => (
              <GalleryImage
                key={image.image_name}
                imageName={image.image_name}
              />
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

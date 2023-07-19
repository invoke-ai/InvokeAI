import { Box, Spinner } from '@chakra-ui/react';
import { useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { IMAGE_LIMIT } from 'features/gallery//store/gallerySlice';
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
import { selectListImagesBaseQueryArgs } from 'features/gallery/store/gallerySelectors';

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
  const [scroller, setScroller] = useState<HTMLElement | null>(null);
  const [initialize, osInstance] = useOverlayScrollbars(
    overlayScrollbarsConfig
  );

  const queryArgs = useAppSelector(selectListImagesBaseQueryArgs);

  const { currentData, isFetching, isSuccess, isError } =
    useListImagesQuery(queryArgs);

  const [listImages] = useLazyListImagesQuery();

  const areMoreAvailable = useMemo(() => {
    if (!currentData) {
      return false;
    }
    return currentData.ids.length < currentData.total;
  }, [currentData]);

  const handleLoadMoreImages = useCallback(() => {
    listImages({
      ...queryArgs,
      offset: currentData?.ids.length ?? 0,
      limit: IMAGE_LIMIT,
    });
  }, [listImages, queryArgs, currentData?.ids.length]);

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

  if (!currentData) {
    return (
      <Box sx={{ w: 'full', h: 'full' }}>
        <Spinner size="2xl" opacity={0.5} />
      </Box>
    );
  }

  if (isSuccess && currentData?.ids.length === 0) {
    return (
      <Box sx={{ w: 'full', h: 'full' }}>
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
      <Box sx={{ w: 'full', h: 'full' }}>
        <IAINoContentFallback
          label="Unable to load Gallery"
          icon={FaExclamationCircle}
        />
      </Box>
    );
  }
};

export default memo(GalleryImageGrid);

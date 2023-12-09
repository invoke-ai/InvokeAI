import { Box, Flex } from '@chakra-ui/react';
import { EntityId } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { VirtuosoGalleryContext } from 'features/gallery/components/ImageGrid/types';
import { $useNextPrevImageState } from 'features/gallery/hooks/useNextPrevImage';
import { selectListImagesBaseQueryArgs } from 'features/gallery/store/gallerySelectors';
import { IMAGE_LIMIT } from 'features/gallery/store/types';
import {
  UseOverlayScrollbarsParams,
  useOverlayScrollbars,
} from 'overlayscrollbars-react';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { FaExclamationCircle, FaImage } from 'react-icons/fa';
import {
  ItemContent,
  ListRange,
  VirtuosoGrid,
  VirtuosoGridHandle,
} from 'react-virtuoso';
import {
  useLazyListImagesQuery,
  useListImagesQuery,
} from 'services/api/endpoints/images';
import { useBoardTotal } from 'services/api/hooks/useBoardTotal';
import GalleryImage from './GalleryImage';
import ImageGridItemContainer from './ImageGridItemContainer';
import ImageGridListContainer from './ImageGridListContainer';

const overlayScrollbarsConfig: UseOverlayScrollbarsParams = {
  defer: true,
  options: {
    scrollbars: {
      visibility: 'auto',
      autoHide: 'scroll',
      autoHideDelay: 1300,
      theme: 'os-theme-dark',
    },
    overflow: { x: 'hidden' },
  },
};

// const selector = createSelector(stateSelector, (state) => {
//   const selection = state.gallery.selection;

//   if (selection.length !== 1) {
//     return undefined;
//   }
//   return selection[0]?.image_name;
// });

const GalleryImageGrid = () => {
  const { t } = useTranslation();
  const rootRef = useRef<HTMLDivElement>(null);
  const [scroller, setScroller] = useState<HTMLElement | null>(null);
  const [initialize, osInstance] = useOverlayScrollbars(
    overlayScrollbarsConfig
  );
  const selectedBoardId = useAppSelector(
    (state) => state.gallery.selectedBoardId
  );
  const { currentViewTotal } = useBoardTotal(selectedBoardId);
  const queryArgs = useAppSelector(selectListImagesBaseQueryArgs);
  const virtuosoRangeRef = useRef<ListRange | null>(null);

  const virtuosoRef = useRef<VirtuosoGridHandle>(null);

  const { currentData, isFetching, isSuccess, isError } =
    useListImagesQuery(queryArgs);

  const [listImages] = useLazyListImagesQuery();

  const areMoreAvailable = useMemo(() => {
    if (!currentData || !currentViewTotal) {
      return false;
    }
    return currentData.ids.length < currentViewTotal;
  }, [currentData, currentViewTotal]);

  const handleLoadMoreImages = useCallback(() => {
    if (!areMoreAvailable) {
      return;
    }

    listImages({
      ...queryArgs,
      offset: currentData?.ids.length ?? 0,
      limit: IMAGE_LIMIT,
    });
  }, [areMoreAvailable, listImages, queryArgs, currentData?.ids.length]);

  const virtuosoContext = useMemo<VirtuosoGalleryContext>(() => {
    return {
      virtuosoRef,
      rootRef,
      virtuosoRangeRef,
    };
  }, []);

  const itemContentFunc: ItemContent<EntityId, VirtuosoGalleryContext> =
    useCallback(
      (index, imageName, virtuosoContext) => (
        <GalleryImage
          key={imageName}
          index={index}
          imageName={imageName as string}
          virtuosoContext={virtuosoContext}
        />
      ),
      []
    );

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

  const onRangeChanged = useCallback((range: ListRange) => {
    virtuosoRangeRef.current = range;
  }, []);

  useEffect(() => {
    $useNextPrevImageState.setKey('virtuosoRef', virtuosoRef);
    $useNextPrevImageState.setKey('virtuosoRangeRef', virtuosoRangeRef);
  }, []);

  if (!currentData) {
    return (
      <Flex
        sx={{
          w: 'full',
          h: 'full',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <IAINoContentFallback label={t('gallery.loading')} icon={FaImage} />
      </Flex>
    );
  }

  if (isSuccess && currentData?.ids.length === 0) {
    return (
      <Flex
        sx={{
          w: 'full',
          h: 'full',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <IAINoContentFallback
          label={t('gallery.noImagesInGallery')}
          icon={FaImage}
        />
      </Flex>
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
            itemContent={itemContentFunc}
            ref={virtuosoRef}
            rangeChanged={onRangeChanged}
            context={virtuosoContext}
            overscan={10}
          />
        </Box>
        <IAIButton
          onClick={handleLoadMoreImages}
          isDisabled={!areMoreAvailable}
          isLoading={isFetching}
          loadingText={t('gallery.loading')}
          flexShrink={0}
        >
          {`Load More (${currentData.ids.length} of ${currentViewTotal})`}
        </IAIButton>
      </>
    );
  }

  if (isError) {
    return (
      <Box sx={{ w: 'full', h: 'full' }}>
        <IAINoContentFallback
          label={t('gallery.unableToLoad')}
          icon={FaExclamationCircle}
        />
      </Box>
    );
  }

  return null;
};

export default memo(GalleryImageGrid);

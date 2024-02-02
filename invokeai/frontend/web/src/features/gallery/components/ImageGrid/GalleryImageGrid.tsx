import { Box, Button, Flex } from '@invoke-ai/ui-library';
import type { EntityId } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { overlayScrollbarsParams } from 'common/components/OverlayScrollbars/constants';
import { virtuosoGridRefs } from 'features/gallery/components/ImageGrid/types';
import { useGalleryHotkeys } from 'features/gallery/hooks/useGalleryHotkeys';
import { useGalleryImages } from 'features/gallery/hooks/useGalleryImages';
import { useOverlayScrollbars } from 'overlayscrollbars-react';
import type { CSSProperties } from 'react';
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiImageBold, PiWarningCircleBold } from 'react-icons/pi';
import type { GridComponents, ItemContent, ListRange, VirtuosoGridHandle } from 'react-virtuoso';
import { VirtuosoGrid } from 'react-virtuoso';
import { useBoardTotal } from 'services/api/hooks/useBoardTotal';

import GalleryImage from './GalleryImage';
import ImageGridItemContainer from './ImageGridItemContainer';
import ImageGridListContainer from './ImageGridListContainer';

const components: GridComponents = {
  Item: ImageGridItemContainer,
  List: ImageGridListContainer,
};

const virtuosoStyles: CSSProperties = { height: '100%' };

const GalleryImageGrid = () => {
  const { t } = useTranslation();
  const rootRef = useRef<HTMLDivElement>(null);
  const [scroller, setScroller] = useState<HTMLElement | null>(null);
  const [initialize, osInstance] = useOverlayScrollbars(overlayScrollbarsParams);
  const selectedBoardId = useAppSelector((s) => s.gallery.selectedBoardId);
  const { currentViewTotal } = useBoardTotal(selectedBoardId);
  const virtuosoRangeRef = useRef<ListRange | null>(null);
  const virtuosoRef = useRef<VirtuosoGridHandle>(null);
  const {
    areMoreImagesAvailable,
    handleLoadMoreImages,
    queryResult: { currentData, isFetching, isSuccess, isError },
  } = useGalleryImages();
  useGalleryHotkeys();
  const itemContentFunc: ItemContent<EntityId, void> = useCallback(
    (index, imageName) => <GalleryImage key={imageName} index={index} imageName={imageName as string} />,
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
    virtuosoGridRefs.set({ rootRef, virtuosoRangeRef, virtuosoRef });
    return () => {
      virtuosoGridRefs.set({});
    };
  }, []);

  if (!currentData) {
    return (
      <Flex w="full" h="full" alignItems="center" justifyContent="center">
        <IAINoContentFallback label={t('gallery.loading')} icon={PiImageBold} />
      </Flex>
    );
  }

  if (isSuccess && currentData?.ids.length === 0) {
    return (
      <Flex w="full" h="full" alignItems="center" justifyContent="center">
        <IAINoContentFallback label={t('gallery.noImagesInGallery')} icon={PiImageBold} />
      </Flex>
    );
  }

  if (isSuccess && currentData) {
    return (
      <>
        <Box ref={rootRef} data-overlayscrollbars="" h="100%" id="gallery-grid">
          <VirtuosoGrid
            style={virtuosoStyles}
            data={currentData.ids}
            endReached={handleLoadMoreImages}
            components={components}
            scrollerRef={setScroller}
            itemContent={itemContentFunc}
            ref={virtuosoRef}
            rangeChanged={onRangeChanged}
            overscan={10}
          />
        </Box>
        <Button
          onClick={handleLoadMoreImages}
          isDisabled={!areMoreImagesAvailable}
          isLoading={isFetching}
          loadingText={t('gallery.loading')}
          flexShrink={0}
        >
          {`${t('accessibility.loadMore')} (${currentData.ids.length} / ${currentViewTotal})`}
        </Button>
      </>
    );
  }

  if (isError) {
    return (
      <Box w="full" h="full">
        <IAINoContentFallback label={t('gallery.unableToLoad')} icon={PiWarningCircleBold} />
      </Box>
    );
  }

  return null;
};

export default memo(GalleryImageGrid);

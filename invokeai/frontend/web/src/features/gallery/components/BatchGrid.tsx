import { Box } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useOverlayScrollbars } from 'overlayscrollbars-react';

import { memo, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { FaImage } from 'react-icons/fa';

import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import BatchImage from 'features/batch/components/BatchImage';
import { batchImagesAdapter } from 'features/batch/store/batchSlice';
import { VirtuosoGrid } from 'react-virtuoso';
import ItemContainer from './ItemContainer';
import ListContainer from './ListContainer';

const selector = createSelector(
  [stateSelector],
  (state) => {
    const images = batchImagesAdapter.getSelectors().selectAll(state.batch);
    return {
      images,
    };
  },
  defaultSelectorOptions
);

const BatchGrid = () => {
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

  const { images } = useAppSelector(selector);

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

  if (images.length) {
    return (
      <Box ref={rootRef} data-overlayscrollbars="" h="100%">
        <VirtuosoGrid
          style={{ height: '100%' }}
          data={images}
          components={{
            Item: ItemContainer,
            List: ListContainer,
          }}
          scrollerRef={setScroller}
          itemContent={(index, item) => (
            <BatchImage
              key={`${item.image_name}-${item.thumbnail_url}`}
              imageDTO={item}
            />
          )}
        />
      </Box>
    );
  }

  return (
    <IAINoContentFallback
      label={t('gallery.noImagesInGallery')}
      icon={FaImage}
    />
  );
};

export default memo(BatchGrid);

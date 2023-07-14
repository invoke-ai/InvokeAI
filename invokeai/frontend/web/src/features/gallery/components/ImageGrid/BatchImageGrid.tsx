import { Box } from '@chakra-ui/react';
import { useAppSelector } from 'app/store/storeHooks';
import { useOverlayScrollbars } from 'overlayscrollbars-react';

import { memo, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { FaImage } from 'react-icons/fa';

import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { VirtuosoGrid } from 'react-virtuoso';
import BatchImage from './BatchImage';
import ItemContainer from './ImageGridItemContainer';
import ListContainer from './ImageGridListContainer';

const selector = createSelector(
  [stateSelector],
  (state) => {
    return {
      imageNames: state.gallery.batchImageNames,
    };
  },
  defaultSelectorOptions
);

const BatchImageGrid = () => {
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

  const { imageNames } = useAppSelector(selector);

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

  if (imageNames.length) {
    return (
      <Box ref={rootRef} data-overlayscrollbars="" h="100%">
        <VirtuosoGrid
          style={{ height: '100%' }}
          data={imageNames}
          components={{
            Item: ItemContainer,
            List: ListContainer,
          }}
          scrollerRef={setScroller}
          itemContent={(index, imageName) => (
            <BatchImage key={imageName} imageName={imageName} />
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

export default memo(BatchImageGrid);

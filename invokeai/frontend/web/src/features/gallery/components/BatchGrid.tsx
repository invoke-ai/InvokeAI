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
import { VirtuosoGrid } from 'react-virtuoso';
import ItemContainer from './ItemContainer';
import ListContainer from './ListContainer';

const selector = createSelector(
  [stateSelector],
  (state) => {
    return {
      imageNames: state.batch.imageNames,
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

export default memo(BatchGrid);

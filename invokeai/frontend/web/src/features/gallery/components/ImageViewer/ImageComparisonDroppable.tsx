import { Flex } from '@invoke-ai/ui-library';
import { useAppStore } from 'app/store/storeHooks';
import { DndDropTarget } from 'features/dnd2/DndDropTarget';
import type { SelectForCompareDndTargetData } from 'features/dnd2/types';
import { selectForCompareDndTarget } from 'features/dnd2/types';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { selectComparisonImages } from './common';

export const ImageComparisonDroppable = memo(() => {
  const { t } = useTranslation();
  const store = useAppStore();
  const targetData = useMemo<SelectForCompareDndTargetData>(() => {
    const { firstImage, secondImage } = selectComparisonImages(store.getState());
    return selectForCompareDndTarget.getData({
      firstImageName: firstImage?.image_name,
      secondImageName: secondImage?.image_name,
    });
  }, [store]);

  return (
    <Flex position="absolute" top={0} right={0} bottom={0} left={0} gap={2} pointerEvents="none">
      <DndDropTarget targetData={targetData} label={t('gallery.selectForCompare')} />
    </Flex>
  );
});

ImageComparisonDroppable.displayName = 'ImageComparisonDroppable';
